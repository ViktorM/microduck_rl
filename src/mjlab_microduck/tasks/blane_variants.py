"""Speed lane B ("improved setup" screening) variants.

Each id is ONE change on top of the best speed recipe
``Mjlab-Velocity-Flat-MicroDuck-ADRTrack16k`` (tracking-gated ADR, sqrt
forward-bucket sampling, 16384 envs, leftover curricula scaled x0.25); the
ADR schedule itself is 16k-native and never scaled.

- ``Mjlab-Velocity-Flat-MicroDuck-Sym16k`` -- bilateral symmetry. The env is
  identical to ADRTrack16k; the arm is the LEARNER-side mirror loss this plugin
  already ships (``symmetry.SYMMETRY_CFG``: ``use_mirror_loss`` at coeff 0.5,
  no data augmentation) but every task registers with ``ENABLE_SYMMETRY =
  False``. rsl-rl: the task's rl_cfg carries ``symmetry_cfg=SYMMETRY_CFG``.
  rl_games: ``config.symmetry_loss: {coef: 0.5, maps:
  mjlab_microduck.tasks.symmetry:mirror_maps}`` (same index/sign tables).
- ``Mjlab-Velocity-Flat-MicroDuck-Gait16k`` -- gait shaping: + ``stride_air_time``,
  the legged_gym / Isaac Lab ``feet_air_time`` form (at each touchdown, reward
  ``last_air_time - threshold``; command-gated), weight 1.0, threshold 0.2 s.
  The recipe already carries mjlab's in-window ``air_time`` term at 3.0
  ([0.125, 0.300] s); the baseline gait sits inside that window (mean
  current-air-time 0.115 s => swing ~0.23 s), so the window is not binding.
  The touchdown form adds a LINEAR incentive for longer swings/steps.
- ``Mjlab-Velocity-Flat-MicroDuck-ADRTrack12-16k`` -- higher targets: tracking
  gate eps 0.15 -> 0.20 and cap ceiling 1.0 -> 1.2 (10 advances of +0.1); the
  action-rate penalty ladder is unchanged (+0.05/advance, capped at 0.5, so it
  saturates at advance 8 and the last two advances raise the cap only).
- ``Mjlab-Velocity-Flat-MicroDuck-Gait16k-Envelope`` -- SIM-ONLY headroom
  diagnostic (B4, not deployable): Gait16k with the XL330's PWM ceiling raised
  x1.5 (``max_pwm`` 1.0 -> 1.5), i.e. the same firmware P-gain and the same
  1.75 A current limit, but the back-EMF (velocity) side of the torque-speed
  envelope pushed out by 1.5x: corner speed (vin - R*Imax)/kt 4.3-9.0 ->
  13.2-20.2 rad/s, no-load speed vin/kt 17.8-22.4 -> 26.6-33.6 rad/s. Peak
  driving torque (kt*Imax = 0.64 Nm) and MuJoCo's forcerange (1.07 Nm, which
  already clips braking) are unchanged, so the ONE thing that moves is how
  much torque is left at high joint speed.
"""

from dataclasses import dataclass, replace

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.actuator import FrictionDRBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import (
    _BAM_ACTUATOR_KWARGS,
    MICRODUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.adr_variants import _make_adr_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

_ADR_TRACK_SCALE = 4096 / 16384


# ---------------------------------------------------------------------------
# B1: symmetry (mirror loss) -- env unchanged, rl_cfg enables SYMMETRY_CFG.
# ---------------------------------------------------------------------------
MicroduckSymRlCfg = replace(
    MicroduckRlCfg,
    algorithm=replace(MicroduckRlCfg.algorithm, symmetry_cfg=SYMMETRY_CFG),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Sym16k",
    env_cfg=_make_adr_cfg("tracking", env_scale=_ADR_TRACK_SCALE),
    play_env_cfg=_make_adr_cfg("tracking", play=True),
    rl_cfg=MicroduckSymRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# B2: gait shaping -- touchdown-based stride reward (legged_gym feet_air_time).
# ---------------------------------------------------------------------------
def stride_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold: float = 0.2,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """legged_gym / Isaac Lab ``feet_air_time``: at each touchdown, reward the
    duration of the swing that just ended minus ``threshold`` (negative for
    shorter swings), summed over feet and gated on a non-zero command.

    Uses the contact sensor's ``last_air_time`` (finalized on the touchdown
    step) and ``compute_first_contact(step_dt)`` to fire exactly once per
    landing, like mjlab's ``feet_swing_height``.
    """
    sensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)  # [B, F] bool
    last_air = sensor.data.last_air_time
    assert last_air is not None
    reward = torch.sum((last_air - threshold) * first_contact.float(), dim=1)
    command = env.command_manager.get_command(command_name)
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return reward * (total_command > command_threshold).float()


def _make_gait_cfg(play: bool = False):
    cfg = _make_adr_cfg("tracking", env_scale=_ADR_TRACK_SCALE, play=play)
    cfg.rewards["stride_air_time"] = RewardTermCfg(
        func=stride_air_time,
        weight=1.0,
        params=dict(
            sensor_name="feet_ground_contact",
            command_name="twist",
            threshold=0.2,
            command_threshold=0.01,
        ),
    )
    return cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Gait16k",
    env_cfg=_make_gait_cfg(),
    play_env_cfg=_make_gait_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# B3: higher targets -- looser tracking gate, cap ceiling 1.2 m/s.
# ---------------------------------------------------------------------------
_ADR12_PARAMS = dict(track_threshold=0.20, vel_max=1.2)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-ADRTrack12-16k",
    env_cfg=_make_adr_cfg("tracking", env_scale=_ADR_TRACK_SCALE, params=_ADR12_PARAMS),
    play_env_cfg=_make_adr_cfg("tracking", play=True, params=_ADR12_PARAMS),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# B4: actuator-envelope headroom diagnostic -- Gait16k with the XL330 voltage
# rail raised x1.5. SIM-ONLY, NOT DEPLOYABLE.
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class EnvelopeBamActuatorCfg(FrictionDRBamActuatorCfg):
    """FrictionDRBamActuatorCfg whose BAM control law may drive the PWM duty
    cycle past 1.0 (``max_pwm`` scaled by ``max_pwm_scale``).

    SIM-ONLY DIAGNOSTIC -- NOT DEPLOYABLE. A real XL330 cannot exceed duty
    1.0; this measures how much running speed the policy recovers when the
    back-EMF side of the actuator envelope is pushed out, nothing else.

    In BAM's voltage model ``V = vin * clip(kp * error_gain * dq_err,
    +-max_pwm)`` and ``tau = kt * V / R - kt^2 * dq / R``, so scaling
    ``max_pwm`` by s is exactly "supply rail x s with the firmware gain / s":
    the linear-region PD gains are unchanged (kp_eff = kt*vin*kp*error_gain/R,
    kd_eff = kt^2/R), the firmware current limiter (1.75 A -> 0.64 Nm peak
    driving torque) is unchanged, the battery-sag model is unchanged, and
    MuJoCo's ``forcerange`` (set from vin_range, 1.07 Nm) still clips braking
    torque exactly as before. What moves is only the speed at which the
    voltage rail starts eating the available torque (corner speed
    ``(s*vin - R*Imax)/kt``) and the no-load speed ``s*vin/kt``.

    The XL330 actuator's ``max_pwm`` is a plain attribute of the per-actuator
    BAM model (bam.dynamixel.actuator.XL330Actuator, default 1.0; bam's
    ``BamActuatorCfg`` exposes no field for it), and ``BamActuator.__init__``
    loads a fresh model instance per actuator, so the override here touches
    nothing shared.
    """

    max_pwm_scale: float = 1.5

    def build(self, entity, target_ids, target_names):
        act = super().build(entity, target_ids, target_names)
        act._bam_model.actuator.max_pwm *= self.max_pwm_scale
        return act


_ENVELOPE_ROBOT_CFG = replace(
    MICRODUCK_WALK_ROBOT_CFG,
    articulation=replace(
        MICRODUCK_WALK_ROBOT_CFG.articulation,
        actuators=(EnvelopeBamActuatorCfg(**_BAM_ACTUATOR_KWARGS, max_pwm_scale=1.5),),
    ),
)


def _make_envelope_cfg(play: bool = False):
    """Gait16k recipe (B2) with the relaxed-envelope actuator swapped in on a
    COPY of the walk robot cfg (the shared MICRODUCK_WALK_ROBOT_CFG is not
    mutated). The play cfg carries the same actuator so probes measure the
    policy inside the envelope it was trained in."""
    cfg = _make_gait_cfg(play=play)
    cfg.scene.entities = {"robot": _ENVELOPE_ROBOT_CFG}
    return cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Gait16k-Envelope",
    env_cfg=_make_envelope_cfg(),
    play_env_cfg=_make_envelope_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
