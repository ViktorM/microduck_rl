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
"""

from dataclasses import replace

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
