from __future__ import annotations

import ast
import re
from pathlib import Path


def test_runtime_strings_do_not_reference_milestone_numbers() -> None:
    pattern = re.compile(r"\bM\d+\b")
    offenders: list[tuple[str, int, str]] = []
    for path in sorted(Path("acsi").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and pattern.search(node.value)
            ):
                offenders.append((str(path), node.lineno, node.value))

    assert offenders == []
