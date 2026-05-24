"""Thin orchestrator for the supervised sentiment pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys


logger = logging.getLogger(__name__)


STAGE_COMMANDS = [
    ["python", "-m", "src.sentiment.inspect_vific"],
    ["python", "-m", "src.sentiment.prepare_inputs"],
    ["python", "-m", "src.sentiment.sample_vific"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the early supervised sentiment pipeline stages in order.")
    parser.add_argument("--include-annotation", action="store_true")
    parser.add_argument("--include-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    commands = list(STAGE_COMMANDS)
    if args.include_annotation:
        commands.extend(
            [
                ["python", "-m", "src.sentiment.annotate_vific", "--require-pilot-pass"],
                ["python", "-m", "src.sentiment.build_silver_labels"],
            ]
        )
    if args.include_training:
        commands.extend(
            [
                ["python", "-m", "src.sentiment.pretrain_mlm"],
                ["python", "-m", "src.sentiment.train_classifier"],
                ["python", "-m", "src.sentiment.infer_cafef"],
                ["python", "-m", "src.sentiment.validate_inference", "--fail-on-validation"],
                ["python", "-m", "src.sentiment.validate_daily_aggregation"],
            ]
        )
    for command in commands:
        logger.info("Running: %s", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
