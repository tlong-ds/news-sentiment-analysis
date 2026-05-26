#!/usr/bin/env python3
"""Scaffolding tool to bootstrap new pipeline modules and tests.

Usage:
    python scripts/scaffold.py --name <module_name> --type <ingestion|preprocessing|sentiment|modeling>
"""

import argparse
import sys
from pathlib import Path

VALID_TYPES = ["ingestion", "preprocessing", "sentiment", "modeling"]


def scaffold_module(name: str, module_type: str) -> None:
    """Generates the src/ and tests/ boilerplate files for the given module name and type."""
    root_dir = Path(__file__).resolve().parents[1]
    
    # 1. Target Paths
    src_file = root_dir / "src" / module_type / f"{name}.py"
    test_file = root_dir / "tests" / f"test_{module_type}_{name}.py"

    # Validate target directories
    if not src_file.parent.exists():
        print(f"Error: Target directory '{src_file.parent}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # 2. Check if files already exist
    if src_file.exists():
        print(f"Error: Source file already exists: {src_file}", file=sys.stderr)
        sys.exit(1)
    if test_file.exists():
        print(f"Error: Test file already exists: {test_file}", file=sys.stderr)
        sys.exit(1)

    # 3. Source file content
    src_content = f'''"""Module for {name} within the {module_type} pipeline stage.

Part of the news-sentiments-analysis workflow.
"""

import argparse
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def process_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the core logic of the {name} module.

    Args:
        inputs: Input dictionary containing parameters or dataframes.

    Returns:
        Dict[str, Any]: Extracted or processed results.
    """
    logger.info("Executing {name} logic...")
    # TODO: Implement core behavior
    return {{"status": "success", "data": inputs}}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run {name} module.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input data file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output results.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution block."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    logger.info("Starting {name} module with input: %s", args.input)
    
    # Example execution
    results = process_data({{"input_path": args.input}})
    logger.info("Completed module execution with status: %s", results["status"])


if __name__ == "__main__":
    main()
'''

    # 4. Test file content (No unused imports)
    test_content = f'''"""Unit tests for the {module_type} module {name}."""

from src.{module_type}.{name} import process_data


def test_{name}_basic_behavior():
    """Verify that process_data returns a success status with the input dict."""
    sample_input = {{"test_key": "test_value"}}
    result = process_data(sample_input)
    
    assert result["status"] == "success"
    assert result["data"] == sample_input
'''

    # 5. Write files
    src_file.write_text(src_content, encoding="utf-8")
    test_file.write_text(test_content, encoding="utf-8")

    print("Successfully scaffolded components:")
    print(f"  [NEW] Source: {src_file.relative_to(root_dir)}")
    print(f"  [NEW] Test:   {test_file.relative_to(root_dir)}")


def main() -> None:
    """Entry point parsing CLI flags."""
    parser = argparse.ArgumentParser(
        description="Scaffold a new pipeline module and its test suite."
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name of the module to scaffold (e.g. data_loader, volatility_model)",
    )
    parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=VALID_TYPES,
        help=f"Pipeline folder context. Must be one of: {', '.join(VALID_TYPES)}",
    )

    args = parser.parse_args()
    scaffold_module(args.name.lower(), args.type)


if __name__ == "__main__":
    main()
