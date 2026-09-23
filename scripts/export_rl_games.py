"""Export an rl_games checkpoint to ONNX — thin wrapper over `mjlab_microduck.rl_games_export`.

    python scripts/export_rl_games.py <TASK_ID> --config <rl_games yaml> --checkpoint <.pth> --onnx-file output.onnx

Same contract and metadata as `scripts/export.py`; see the module docstring.
"""

from mjlab_microduck.rl_games_export import main

if __name__ == "__main__":
    main()
