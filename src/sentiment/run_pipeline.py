"""Thin orchestrator for the CafeF sentiment pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys


logger = logging.getLogger(__name__)


STAGE_COMMANDS = [
    ["python", "-m", "src.sentiment.prepare_inputs"],
    ["python", "-m", "src.sentiment.infer_cafef"],
    ["python", "-m", "src.sentiment.validate_inference", "--fail-on-validation"],
    ["python", "-m", "src.sentiment.validate_daily_aggregation"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CafeF sentiment pipeline stages in order.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    commands = list(STAGE_COMMANDS)
    for command in commands:
        logger.info("Running: %s", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
