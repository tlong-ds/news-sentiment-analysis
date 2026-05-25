#!/usr/bin/env python3
"""Validation runner for news-sentiments-analysis.

Consolidates linting and tests to gate commits and pull requests.
Exits with 0 if all checks pass, and 1 otherwise.
"""

import subprocess
import sys
from pathlib import Path


def run_check(cmd, description):
    """Runs a validation command, displaying results and return code."""
    print("=" * 60)
    print(f"RUNNING VALIDATION: {description}")
    print("=" * 60)
    
    # Run synchronously and stream stdout/stderr directly
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"SUCCESS: {description} passed.\n")
        return True
    else:
        print(f"FAILED: {description} exited with code {result.returncode}.\n", file=sys.stderr)
        return False


def main():
    """Main validation runner."""
    root_dir = Path(__file__).resolve().parents[1]
    
    checks = [
        # 1. Run Linter
        ([sys.executable, str(root_dir / "scripts" / "lint.py")], "Repository Linter & AST checks"),
        # 2. Run Test Suite
        (["pytest"], "Pytest Test Suite"),
    ]
    
    all_passed = True
    for cmd, desc in checks:
        if not run_check(cmd, desc):
            all_passed = False
            # Exit early if linter fails to save test execution time
            if "Linter" in desc:
                print("Aborting remaining checks due to linter failure.", file=sys.stderr)
                break

    if not all_passed:
        print("=" * 60, file=sys.stderr)
        print("VALIDATION FAILED. Please fix the errors above.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("VALIDATION PASSED. All checks are satisfied!")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
