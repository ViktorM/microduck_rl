"""Export an rl_games checkpoint to the same deployable ONNX contract as `export.py`.

    python scripts/export_rl_games.py <TASK_ID> --config ppo_microduck_velocity.yaml \
        --checkpoint runs/<run>/nn/MJLab_MicroDuck_Velocity.pth --onnx-file output.onnx

The file is `actor(normalizer(obs))`: input ``obs`` float32 ``[1, obs_dim]``, output
``actions`` ``[1, act_dim]``, opset 18, with the metadata mjlab attaches to rsl-rl
exports (joint names and order, gains, default pose, command and observation
names, action scale) plus trainer provenance. The action is the policy mean,
un-clipped, exactly what rl_games' player emits with ``clip_actions: false``
(mjlab scales and clamps the joint targets in the env). ``scripts/infer_policy.py``
and the robot runtime consume it like any other policy.

``--validate-steps N`` rolls the exported file back into the task (16 envs,
un-normalised observations straight from the env) and reports falls and the
body-frame forward speed, the same check the port ran on Pollen's deployed
policies.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml


class ActorOnnx(torch.nn.Module):
    """Deterministic actor: observation normaliser (from its buffers) + network mean.

    The normaliser is applied explicitly from ``running_mean`` / ``running_var``
    instead of calling rl_games' RunningMeanStd module, which the ONNX tracer
    refuses as a nested call; the arithmetic is the module's eval-mode forward:
    clamp((obs - mean) / sqrt(var + eps), -5, 5).
    """

    def __init__(self, model):
        super().__init__()
        self.net = model.a2c_network
        self.normalize = bool(getattr(model, "normalize_input", False))
        if self.normalize:
            rms = model.running_mean_std
            self.register_buffer("mean", rms.running_mean.detach().clone().float())
            self.register_buffer("var", rms.running_var.detach().clone().float())
            self.eps = float(rms.epsilon)

    def forward(self, obs):
        if self.normalize:
            obs = torch.clamp((obs - self.mean) / torch.sqrt(self.var + self.eps), min=-5.0, max=5.0)
        mu, _logstd, _value, _states = self.net({"obs": obs, "is_train": False})
        return mu


def build_env_and_player(task_id: str, config_path: str, checkpoint: str, device: str, num_envs: int = 1, play: bool = True):
    import warp as wp

    wp.init()
    import mjlab_microduck.tasks  # noqa: F401  registers the tasks
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from rl_games.envs.mjlab_play import _build_player, pick_policy_group

    with open(config_path) as f:
        params = yaml.safe_load(f)["params"]
    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device=device)
    obs_dict, _ = env.reset()
    group = pick_policy_group(obs_dict)
    obs_dim = obs_dict[group].shape[-1]
    act_dim = env.action_space.shape[-1]
    player = _build_player(params, obs_dim, act_dim, num_envs, device)
    player.restore(checkpoint)
    player.model.eval()
    return env, player, group, obs_dim, act_dim


def export_onnx(env, player, obs_dim: int, onnx_path: str, run_path: str, extra: dict) -> None:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    model = player.model
    device = next(model.parameters()).device
    wrapper = ActorOnnx(model).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, onnx_path, export_params=True, opset_version=18,
        input_names=["obs"], output_names=["actions"], dynamo=False,
    )
    model.to(device)
    metadata = get_base_metadata(env, run_path=run_path)
    metadata.update(extra)
    attach_metadata_to_onnx(onnx_path, metadata)


def check_against_torch(player, onnx_path: str, obs_dim: int, n: int = 64) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    model = player.model
    device = next(model.parameters()).device
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        obs = rng.standard_normal((1, obs_dim)).astype(np.float32)
        ref = ActorOnnx(model)(torch.as_tensor(obs, device=device)).detach().cpu().numpy()
        out = sess.run(None, {"obs": obs})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def validate_in_env(task_id: str, onnx_path: str, device: str, steps: int, num_envs: int = 16, vx_cmd: float | None = 0.3) -> dict:
    """Drive the exported policy in the training env; falls and forward speed."""
    import onnxruntime as ort
    import warp as wp

    wp.init()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=False)
    cfg.scene.num_envs = num_envs
    tw = cfg.commands.get("twist") if hasattr(cfg.commands, "get") else getattr(cfg.commands, "twist", None)
    if tw is not None and vx_cmd is not None:
        tw.ranges.lin_vel_x = (vx_cmd, vx_cmd)
        tw.ranges.lin_vel_y = (0.0, 0.0)
        tw.ranges.ang_vel_z = (0.0, 0.0)
        for attr in ("rel_standing_envs", "rel_heading_envs"):
            if hasattr(tw, attr):
                setattr(tw, attr, 0.0)
    env = ManagerBasedRlEnv(cfg, device=device)
    sess = ort.InferenceSession(onnx_path)
    obs_dict, _ = env.reset()
    group = "actor" if "actor" in obs_dict else next(iter(obs_dict))
    falls = 0
    vx = []
    for t in range(steps):
        obs = obs_dict[group].detach().cpu().numpy().astype(np.float32)
        act = np.concatenate([sess.run(None, {"obs": obs[i : i + 1]})[0] for i in range(num_envs)], axis=0)
        obs_dict, _rew, terminated, truncated, _info = env.step(torch.as_tensor(act, device=device))
        falls += int(terminated.sum().item())
        if t >= steps // 4:
            vx.append(env.scene["robot"].data.root_link_lin_vel_b[:, 0].mean().item())
    env.close()
    return {"steps": steps, "envs": num_envs, "falls": falls, "vx_mean": float(np.mean(vx)), "vx_cmd": vx_cmd}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_id")
    p.add_argument("--config", required=True, help="rl_games yaml the checkpoint was trained with")
    p.add_argument("--checkpoint", required=True, help="rl_games .pth (runs/<run>/nn/<name>.pth is the best-reward one)")
    p.add_argument("--onnx-file", default="output.onnx")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-path", default="local")
    p.add_argument("--validate-steps", type=int, default=0, help="roll the ONNX back into the task for N steps (16 envs)")
    p.add_argument("--validate-vx", type=float, default=0.3, help="pinned forward command for the validation roll (velocity tasks)")
    args = p.parse_args()

    env, player, group, obs_dim, act_dim = build_env_and_player(args.task_id, args.config, args.checkpoint, args.device)
    extra = {
        "trainer": "rl_games",
        "rl_games_config": os.path.abspath(args.config),
        "rl_games_checkpoint": os.path.abspath(args.checkpoint),
        "obs_group": group,
    }
    export_onnx(env, player, obs_dim, args.onnx_file, args.run_path, extra)
    worst = check_against_torch(player, args.onnx_file, obs_dim)
    print(f"Written {os.path.abspath(args.onnx_file)}: obs [1, {obs_dim}] -> actions [1, {act_dim}], opset 18; "
          f"max |onnx - torch| over 64 random obs = {worst:.2e}")
    env.close()
    if args.validate_steps > 0:
        r = validate_in_env(args.task_id, args.onnx_file, args.device, args.validate_steps, vx_cmd=args.validate_vx)
        print(f"Validation roll: {r['envs']} envs x {r['steps']} steps, falls {r['falls']}, "
              f"body vx mean {r['vx_mean']:.3f} m/s at command {r['vx_cmd']}")


if __name__ == "__main__":
    main()
