#!/usr/bin/env python3
"""Custom linter script for news-sentiments-analysis.

Runs Ruff for style and standard checks, then performs custom AST analysis to:
1. Disallow raw print() statements in src/ library code (outside of main functions/blocks).
2. Disallow imports from tests/ in src/.
3. Warn if public functions and classes in src/ lack docstrings.
"""

import ast
import subprocess
import sys
from pathlib import Path


class CustomASTVisitor(ast.NodeVisitor):
    """AST visitor to enforce quality rules in Python source code."""

    def __init__(self, rel_path, lines):
        self.rel_path = rel_path
        self.lines = lines
        self.in_main_context = False
        self.errors = []
        self.warnings = []

    def visit_FunctionDef(self, node):
        # 1. Warn on missing docstring for public functions
        if not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            if not doc:
                self.warnings.append(
                    f"{self.rel_path}:{node.lineno}: WARNING: Public function '{node.name}' "
                    "is missing a docstring."
                )

        # Track if we are inside a function named "main"
        old_in_main = self.in_main_context
        if node.name == "main":
            self.in_main_context = True
        
        self.generic_visit(node)
        self.in_main_context = old_in_main

    def visit_AsyncFunctionDef(self, node):
        # 1. Warn on missing docstring for public async functions
        if not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            if not doc:
                self.warnings.append(
                    f"{self.rel_path}:{node.lineno}: WARNING: Public async function '{node.name}' "
                    "is missing a docstring."
                )

        old_in_main = self.in_main_context
        if node.name == "main":
            self.in_main_context = True
            
        self.generic_visit(node)
        self.in_main_context = old_in_main

    def visit_ClassDef(self, node):
        # 1. Warn on missing docstring for public classes
        if not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            if not doc:
                self.warnings.append(
                    f"{self.rel_path}:{node.lineno}: WARNING: Public class '{node.name}' "
                    "is missing a docstring."
                )
        self.generic_visit(node)

    def visit_If(self, node):
        # Track if we enter `if __name__ == "__main__":`
        is_name_main = False
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                for op, comparator in zip(node.test.ops, node.test.comparators):
                    if isinstance(op, ast.Eq):
                        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
                            is_name_main = True
                        elif isinstance(comparator, ast.Str) and comparator.s == "__main__":
                            is_name_main = True

        old_in_main = self.in_main_context
        if is_name_main:
            self.in_main_context = True
            
        self.generic_visit(node)
        self.in_main_context = old_in_main

    def visit_Call(self, node):
        # 2. Check for print() calls outside of main context
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if not self.in_main_context:
                line_idx = node.lineno - 1
                line_text = self.lines[line_idx] if line_idx < len(self.lines) else ""
                if "noqa" not in line_text and "allow-print" not in line_text:
                    self.errors.append(
                        f"{self.rel_path}:{node.lineno}: Do not use raw print() in library code. "
                        "Use the logging module or return values instead (or add '# noqa: print' to bypass)."
                    )
        self.generic_visit(node)

    def visit_Import(self, node):
        # 3. Prevent imports from tests
        for name in node.names:
            if name.name.startswith("tests"):
                self.errors.append(
                    f"{self.rel_path}:{node.lineno}: Do not import from 'tests' inside 'src'."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # 3. Prevent imports from tests
        if node.module and node.module.startswith("tests"):
            self.errors.append(
                f"{self.rel_path}:{node.lineno}: Do not import from 'tests' inside 'src'."
            )
        self.generic_visit(node)


def run_command(cmd, description):
    """Runs a system command and returns whether it succeeded."""
    print(f"=== Running: {description} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False
    print("OK\n")
    return True


def check_ast_rules():
    """Parses src/ files using AST to enforce custom quality rules."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    errors = []
    warnings = []

    for path in src_dir.glob("**/*.py"):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()
            tree = ast.parse(content, filename=str(path))
        except Exception as e:
            errors.append(f"Failed to parse {path.relative_to(src_dir.parent)}: {e}")
            continue

        rel_path = path.relative_to(src_dir.parent)
        visitor = CustomASTVisitor(rel_path, lines)
        visitor.visit(tree)
        errors.extend(visitor.errors)
        warnings.extend(visitor.warnings)

    print("=== Running: Custom AST Checks ===")
    
    if warnings:
        print(f"Found {len(warnings)} stylistic warnings:")
        # Limit warning output to avoid cluttering CLI outputs
        for warn in warnings[:10]:
            print(warn)
        if len(warnings) > 10:
            print(f"... and {len(warnings) - 10} more warnings.")
        print()

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"FAIL ({len(errors)} custom errors found)\n", file=sys.stderr)
        return False
    print("OK\n")
    return True


def main():
    """Main linter entrypoint."""
    success = True

    # 1. Run Ruff Linter
    if not run_command(["ruff", "check", "src", "tests"], "Ruff Linter"):
        success = False

    # 2. Run Ruff Formatter check
    if not run_command(["ruff", "format", "--check", "src", "tests"], "Ruff Formatter Check"):
        success = False

    # 3. Run Custom AST Checks
    if not check_ast_rules():
        success = False

    if not success:
        print("LINTER FAILED. Please resolve the errors above.", file=sys.stderr)
        sys.exit(1)
    
    print("LINTER PASSED. All style and custom rules are satisfied.")
    sys.exit(0)


if __name__ == "__main__":
    main()
