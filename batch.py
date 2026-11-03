"""Launch the paper's seed x catalog grid.

The TensorFlow ``batch.py`` immediately runs ETAS
(``empirical empirical empirical``) for seeds 1, 2 and 3 on every catalog.
The ST-NKF loop in that file is commented out.

This port prints the commands unless ``--run`` is passed, so a bare
``python batch.py`` does not start multi-day jobs. Component order on the
command line is ``kappa temporal spatial``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

DATASETS = [
    "SCEDC_20",
    "SCEDC_25",
    "SCEDC_30",
    "ComCat",
    "SaltonSea",
    "SanJac",
    "WHITE",
]

MODES = {
    "etas": ("empirical", "empirical", "empirical"),
    "nkf": ("neural", "neural", "neural"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue ST-NKF / ETAS runs.")
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["etas"],
        choices=tuple(MODES),
        help="Default matches upstream batch.py: ETAS only",
    )
    parser.add_argument("--gpu", default="0", help="Forwarded as gpu_id; use -1 for CPU")
    parser.add_argument("--epochs", type=int, default=None, help="Override run_models.py epochs")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the commands. Without this flag they are only printed.",
    )
    return parser.parse_args(argv)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    for model_name in args.models:
        kappa, temporal, spatial = MODES[model_name]
        for seed in args.seeds:
            for dataset in args.datasets:
                cmd = [
                    sys.executable,
                    "run_models.py",
                    str(seed),
                    str(args.gpu),
                    dataset,
                    kappa,
                    temporal,
                    spatial,
                ]
                if args.epochs is not None:
                    cmd.extend(["--epochs", str(args.epochs)])
                if args.patience is not None:
                    cmd.extend(["--patience", str(args.patience)])
                if args.batch_size is not None:
                    cmd.extend(["--batch-size", str(args.batch_size)])
                commands.append(cmd)
    return commands


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if args.run:
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                return completed.returncode
    if not args.run:
        print(f"{len(commands)} commands (dry run; pass --run to execute).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
