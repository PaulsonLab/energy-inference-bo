#!/usr/bin/env python3
"""Check repository-relative inline Markdown links without network access."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "artifacts", "build", "__pycache__", ".pytest_cache"}
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    ]


def local_target(raw: str) -> str | None:
    target = raw.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    return target.split("#", maxsplit=1)[0]


def main() -> int:
    missing: list[str] = []
    for source in markdown_files():
        text = source.read_text()
        for raw in LINK.findall(text):
            target = local_target(raw)
            if target is None:
                continue
            resolved = (source.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                missing.append(f"{source.relative_to(ROOT)}: escapes repository: {raw}")
            else:
                if not resolved.exists():
                    missing.append(f"{source.relative_to(ROOT)}: missing {raw}")
    if missing:
        print("Broken repository-relative Markdown links:")
        print("\n".join(missing))
        return 1
    print(f"Checked {len(markdown_files())} Markdown files: all local links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
