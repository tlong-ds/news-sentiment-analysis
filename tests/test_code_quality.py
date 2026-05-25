"""Unit tests verifying repository-wide code quality and structures.

Ensures coding conventions are programmatically verified as part of the test suite.
"""

import ast
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
TESTS_DIR = ROOT_DIR / "tests"


def test_naming_conventions():
    """Verify that all source python files use snake_case naming."""
    # Pattern for snake_case python file names
    snake_case_pattern = re.compile(r"^[a-z0-9_]+\.py$")

    for path in SRC_DIR.glob("**/*.py"):
        if path.name == "__init__.py":
            continue

        # Verify file name is snake_case
        assert snake_case_pattern.match(path.name), (
            f"File '{path.relative_to(ROOT_DIR)}' does not follow snake_case naming convention."
        )


def test_test_file_naming():
    """Verify that all files in tests/ start with test_ and end with .py."""
    test_file_pattern = re.compile(r"^test_[a-z0-9_]+\.py$")

    for path in TESTS_DIR.glob("**/*.py"):
        if path.name == "conftest.py" or path.name == "__init__.py":
            continue

        assert test_file_pattern.match(path.name), (
            f"Test file '{path.relative_to(ROOT_DIR)}' must start with 'test_' and be snake_case."
        )


def test_no_test_imports_in_src():
    """Verify that no file in src/ imports from tests/."""
    for path in SRC_DIR.glob("**/*.py"):
        if path.name == "__init__.py":
            continue

        content = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(path))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    assert not name.name.startswith("tests"), (
                        f"Forbidden import from 'tests' in module '{path.relative_to(ROOT_DIR)}'."
                    )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module and node.module.startswith("tests")), (
                    f"Forbidden import from 'tests' in module '{path.relative_to(ROOT_DIR)}'."
                )
