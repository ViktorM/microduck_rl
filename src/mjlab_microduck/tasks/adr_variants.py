"""ADR (automatic domain randomization) variants of the flat velocity task.

Instead of step-keyed forward-speed stages (``speed_variants``), the forward
command cap advances on PERFORMANCE gates:

- ``Mjlab-Velocity-Flat-MicroDuck-ADR16k``      — reward gate: mean episode
  reward over a rolling window of finished episodes >= 130.
- ``Mjlab-Velocity-Flat-MicroDuck-ADRTrack16k`` — tracking gate: median
  |v_x - cmd_x| over walking envs (rolling window of medians) < 0.15.

Shared ADR schedule (both arms):
- forward cap: 0.2 -> 1.0 in +0.1 steps (8 advances);
- action_rate_l2 weight: -0.1 -> -0.5 in -0.05 steps, locked to the same
  advances (the base step-keyed action_rate_weight curriculum is removed);
- on each advance the gate window is cleared and episodes that STARTED
  before the advance are excluded (ignore window = one max episode length),
  plus a min-dwell cooldown of 1.5 episode lengths — so the gate always
  re-earns the threshold on post-advance data.

Both arms use sqrt command sampling in the forward-only bucket: v = cap*sqrt(U)
(density rises linearly with v), so faster commands are seen more often.
General envs stay uniform; play cfgs use the standard uniform command.

The remaining step-keyed curricula (standing envs, CoM range, ...) are scaled
x(4096/16384) so they fire at the same TOTAL env-step count as the tested
4096-env baseline when training with 16384 envs.
"""

import dataclasses
import statistics
from collections import deque

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.mdp import (
    VelocityCommandCommandOnly,
    VelocityCommandCommandOnlyCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.speed_variants import (
    _scale_curriculum_steps,
    _set_forward_range,
)


# ---------------------------------------------------------------------------
# Sqrt-sampled forward bucket.
# ---------------------------------------------------------------------------
class SqrtForwardVelocityCommand(VelocityCommandCommandOnly):
    """Forward-only bucket samples v = cap * sqrt(U) instead of |uniform|.

    Density of v rises linearly with v, so high commands are weighted more,
    smoothly (Viktor's sqrt() spec). Applies ONLY to the forward bucket; all
    other envs keep the base uniform sampling. The 0.3 clamp floor of the
    forward bucket is preserved (the ADR term gates the bucket off entirely
    while the cap is below 0.3, same as the speed curriculum).
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        fwd_ids = env_ids[self.is_forward_env[env_ids]]
        if len(fwd_ids) == 0:
            return
        cap = self.cfg.ranges.lin_vel_x[1]
        u = torch.rand(len(fwd_ids), device=self.device)
        self.vel_command_b[fwd_ids, 0] = (cap * torch.sqrt(u)).clamp(min=0.3)
        # Refresh the world-frame reference copy (same as the turn-in-place
        # override does after mutating commands).
        self.vel_command_w[fwd_ids] = self.vel_command_b[fwd_ids]


class SqrtForwardVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    def build(self, env: ManagerBasedRlEnv) -> "SqrtForwardVelocityCommand":
        return SqrtForwardVelocityCommand(self, env)


def _swap_to_sqrt_command(cfg) -> None:
    """Replace the twist command cfg with the sqrt-bucket subclass, preserving
    every field AND plain instance attributes (rel_turn_in_place_envs is set
    as an instance attribute post-construction in the base cfg)."""
    old = cfg.commands["twist"]
    field_names = {f.name for f in dataclasses.fields(VelocityCommandCommandOnlyCfg)}
    kwargs = {k: v for k, v in vars(old).items() if k in field_names}
    new = SqrtForwardVelocityCommandCfg(**kwargs)
    for k, v in vars(old).items():
        if k not in field_names:
            setattr(new, k, v)
    cfg.commands["twist"] = new


# ---------------------------------------------------------------------------
# ADR curriculum term (class-based: keeps rolling-window state).
# ---------------------------------------------------------------------------
class AdrForwardSpeed(ManagerTermBase):
    """Performance-gated forward-speed + action-penalty ADR.

    Called from ``CurriculumManager.compute`` inside ``_reset_idx`` — i.e. on
    every step where at least one env resets, with ``env_ids`` = the resetting
    envs, and BEFORE ``reward_manager.reset`` zeroes the episode sums, so the
    finished episodes' total rewards are still readable.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.cfg = cfg
        self._buf: deque | None = None
        self._stage = 0
        self._ignore_until = 0
        self._cooldown_until = 0
        self._last_val = float("nan")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids,
        command_name: str,
        reward_name: str,
        gate: str,
        threshold: float,
        window: int,
        track_threshold: float,
        track_window: int,
        vel_start: float,
        vel_step: float,
        vel_max: float,
        pen_start: float,
        pen_step: float,
        pen_max: float,
        rel_forward_envs: float,
    ):
        ep_len = int(env.max_episode_length)
        n_stages = int(round((vel_max - vel_start) / vel_step))
        win = window if gate == "reward" else track_window
        if self._buf is None:
            self._buf = deque(maxlen=win)

        # ---- collect the gate signal ----
        if env.common_step_counter >= self._ignore_until:
            if gate == "reward":
                # Total reward of the episodes finishing right now.
                if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
                    valid = env_ids[env.episode_length_buf[env_ids] > 0]
                    if len(valid) > 0:
                        total = torch.zeros(len(valid), device=env.device)
                        for sums in env.reward_manager._episode_sums.values():
                            total += sums[valid]
                        self._buf.extend(total.tolist())
            else:  # gate == "tracking"
                term = env.command_manager.get_term(command_name)
                vx = term.robot.data.root_link_lin_vel_b[:, 0]
                cmd = term.vel_command_b[:, 0]
                mask = (~term.is_standing_env) & (cmd > 0.05)
                if int(mask.sum()) >= 16:
                    err = (vx[mask] - cmd[mask]).abs()
                    self._buf.append(float(err.median()))

        # ---- evaluate gate ----
        full = len(self._buf) == win
        if self._buf:
            if gate == "reward":
                self._last_val = sum(self._buf) / len(self._buf)
                fire = full and self._last_val >= threshold
            else:
                self._last_val = statistics.median(self._buf)
                fire = full and self._last_val < track_threshold
        else:
            fire = False

        # ---- advance ----
        if (
            fire
            and self._stage < n_stages
            and env.common_step_counter >= self._cooldown_until
        ):
            self._stage += 1
            self._buf.clear()
            # Exclude episodes that started before this advance, then require
            # a freshly refilled window; plus a min-dwell cooldown.
            self._ignore_until = env.common_step_counter + ep_len
            self._cooldown_until = env.common_step_counter + int(1.5 * ep_len)

        # ---- apply (idempotent, mutates LIVE manager cfgs) ----
        cap = min(vel_max, vel_start + vel_step * self._stage)
        pen = -min(pen_max, pen_start + pen_step * self._stage)
        cmd_cfg = env.command_manager.get_term(command_name).cfg
        cmd_cfg.ranges.lin_vel_x = (cmd_cfg.ranges.lin_vel_x[0], cap)
        # Forward-only bucket is clamped to >= 0.3 at resample; gate it off
        # while the cap is below the clamp floor (same as speed_variants).
        cmd_cfg.rel_forward_envs = 0.0 if cap < 0.3 else rel_forward_envs
        env.reward_manager.get_term_cfg(reward_name).weight = pen

        return {
            "max_forward": cap,
            "action_rate_w": pen,
            "gate_value": self._last_val if self._last_val == self._last_val else 0.0,
            "advances": float(self._stage),
        }


# ---------------------------------------------------------------------------
# Cfg builders + registrations.
# ---------------------------------------------------------------------------
_ADR_PARAMS = dict(
    command_name="twist",
    reward_name="action_rate_l2",
    threshold=130.0,
    window=4096,
    track_threshold=0.15,
    track_window=500,
    vel_start=0.2,
    vel_step=0.1,
    vel_max=1.0,
    pen_start=0.1,
    pen_step=0.05,
    pen_max=0.5,
)


def _make_adr_cfg(gate: str, env_scale: float | None = None, play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    if play:
        # Play env: final range, uniform sampling, no ADR machinery.
        _set_forward_range(cfg, (-0.4, 1.0))
        return cfg
    if env_scale is not None:
        _scale_curriculum_steps(cfg, env_scale)
    # ADR drives the action-rate weight itself.
    cfg.curriculum.pop("action_rate_weight", None)
    _set_forward_range(cfg, (-0.4, 0.2))
    _swap_to_sqrt_command(cfg)
    cfg.curriculum["adr_forward_speed"] = CurriculumTermCfg(
        func=AdrForwardSpeed,
        params=dict(
            gate=gate,
            rel_forward_envs=cfg.commands["twist"].rel_forward_envs,
            **_ADR_PARAMS,
        ),
    )
    return cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-ADR16k",
    env_cfg=_make_adr_cfg("reward", env_scale=4096 / 16384),
    play_env_cfg=_make_adr_cfg("reward", play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-ADRTrack16k",
    env_cfg=_make_adr_cfg("tracking", env_scale=4096 / 16384),
    play_env_cfg=_make_adr_cfg("tracking", play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
