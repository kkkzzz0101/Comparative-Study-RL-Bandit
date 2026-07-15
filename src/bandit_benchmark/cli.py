from __future__ import annotations

import argparse
from pathlib import Path

from bandit_benchmark.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible bandit benchmark experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run an experiment from a TOML config")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        output_dir = run_experiment(args.config, args.output)
        print(f"Results written to {output_dir}")
