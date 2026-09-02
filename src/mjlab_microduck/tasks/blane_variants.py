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
"""

from dataclasses import replace

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

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
