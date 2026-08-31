"""Speed-study variants of the flat velocity task (NEW task ids only).

All variants are built by calling the existing ``make_microduck_velocity_env_cfg``
factory (flat, no backlash) and post-processing the returned cfg — the standard
task registrations and the factory itself are untouched. Every mutated
curriculum ``params`` dict / command ``ranges`` object is deepcopied first, so
no state is shared with the standard task's registered cfg.

Variants
--------
- ``Mjlab-Velocity-Flat-MicroDuck-Speed10``: action_rate weight_stages halved at
  every curriculum stage; twist lin_vel_x fixed at (-0.4, 1.0) from step 0.
- ``Mjlab-Velocity-Flat-MicroDuck-Speed10Curr``: same halved action-rate stages;
  lin_vel_x starts at (-0.4, 0.2) and a forward-speed curriculum raises the
  forward max to 1.0 by epoch 1000 (steps keyed in per-env steps = epoch * 24,
  same convention as the other curriculum terms). The play cfg gets the final
  (-0.4, 1.0) range and NO curriculum term (the curriculum would clamp a play
  env back to 0.2 at reset, since common_step_counter starts at 0).
- ``Mjlab-Velocity-Flat-MicroDuck-Scaled8k`` / ``-Scaled16k``: the STANDARD task
  (original weights, original ±0.4 ranges) with every curriculum stage ``step``
  multiplied by 4096/8192 = 0.5 resp. 4096/16384 = 0.25, so milestones fire at
  the same TOTAL env-step count regardless of num_envs (common_step_counter
  counts per-env policy steps, i.e. iterations x num_steps_per_env).
"""

from copy import deepcopy

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg
from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


# ---------------------------------------------------------------------------
# Curriculum term: forward-speed ramp for the twist command.
# ---------------------------------------------------------------------------
def forward_speed_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    speed_stages: list[dict],
) -> torch.Tensor:
    """Raise the twist command's forward (max lin_vel_x) bound in stages.

    Same pattern as ``microduck_mdp.reward_weight`` / ``standing_envs_curriculum``:
    the latest stage whose ``step`` has elapsed wins, and the LIVE command term
    cfg is mutated (``env.command_manager.get_term(...).cfg`` — the object
    ``UniformVelocityCommand._resample_command`` actually reads at resample
    time), not ``env.cfg``. The backward bound is left untouched.

    ``speed_stages`` is a list of ``{"step": int, "max_forward": float}`` dicts,
    with ``step`` in per-env steps (epoch * num_steps_per_env).
    """
    del env_ids  # Unused

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = command_term.cfg

    current_max = speed_stages[0]["max_forward"]
    for stage in speed_stages:
        if env.common_step_counter > stage["step"]:
            current_max = stage["max_forward"]

    cfg.ranges.lin_vel_x = (cfg.ranges.lin_vel_x[0], current_max)
    return torch.tensor([current_max])


_SPEED_STAGES = [
    {"step": 0, "max_forward": 0.2},
    {"step": 150 * NUM_STEPS_PER_ENV, "max_forward": 0.3},
    {"step": 300 * NUM_STEPS_PER_ENV, "max_forward": 0.45},
    {"step": 500 * NUM_STEPS_PER_ENV, "max_forward": 0.6},
    {"step": 700 * NUM_STEPS_PER_ENV, "max_forward": 0.8},
    {"step": 1000 * NUM_STEPS_PER_ENV, "max_forward": 1.0},
]


# ---------------------------------------------------------------------------
# Cfg post-processing helpers.
# ---------------------------------------------------------------------------
def _halve_action_rate_stages(cfg) -> None:
    """Halve the action_rate weight at every curriculum stage (steps kept)."""
    term = cfg.curriculum["action_rate_weight"]
    term.params = deepcopy(term.params)
    for stage in term.params["weight_stages"]:
        stage["weight"] = stage["weight"] / 2.0


def _set_forward_range(cfg, lin_vel_x: tuple) -> None:
    twist = cfg.commands["twist"]
    twist.ranges = deepcopy(twist.ranges)
    twist.ranges.lin_vel_x = lin_vel_x


def _scale_curriculum_steps(cfg, scale: float) -> None:
    """Multiply every curriculum stage 'step' in every term by ``scale``.

    Walks all curriculum terms, finds params entries that are lists of dicts
    containing a 'step' key, and scales each step (rounded to int).
    """
    for term in cfg.curriculum.values():
        if term is None:
            continue
        term.params = deepcopy(term.params)
        for value in term.params.values():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(s, dict) and "step" in s for s in value)
            ):
                for s in value:
                    s["step"] = int(round(s["step"] * scale))


# ---------------------------------------------------------------------------
# Cfg builders.
# ---------------------------------------------------------------------------
def _make_speed10_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    _halve_action_rate_stages(cfg)
    _set_forward_range(cfg, (-0.4, 1.0))
    return cfg


def _make_speed10curr_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    _halve_action_rate_stages(cfg)
    if play:
        # Play env: final range, no ramp (the ramp would clamp it back to 0.2).
        _set_forward_range(cfg, (-0.4, 1.0))
        return cfg
    _set_forward_range(cfg, (-0.4, 0.2))
    cfg.curriculum["forward_speed_range"] = CurriculumTermCfg(
        func=forward_speed_curriculum,
        params={
            "command_name": "twist",
            "speed_stages": deepcopy(_SPEED_STAGES),
        },
    )
    return cfg


def _make_scaled_cfg(scale: float, play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    _scale_curriculum_steps(cfg, scale)
    return cfg


# ---------------------------------------------------------------------------
# Registrations (new ids only).
# ---------------------------------------------------------------------------
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Speed10",
    env_cfg=_make_speed10_cfg(),
    play_env_cfg=_make_speed10_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Speed10Curr",
    env_cfg=_make_speed10curr_cfg(),
    play_env_cfg=_make_speed10curr_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Geometry-scaled variants for the wall-clock study: curriculum milestones fire
# at the same TOTAL env-step count when training with 8192 / 16384 envs
# (baseline stages are keyed for 4096 envs).
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Scaled8k",
    env_cfg=_make_scaled_cfg(4096 / 8192),
    play_env_cfg=_make_scaled_cfg(4096 / 8192, play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Scaled16k",
    env_cfg=_make_scaled_cfg(4096 / 16384),
    play_env_cfg=_make_scaled_cfg(4096 / 16384, play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
