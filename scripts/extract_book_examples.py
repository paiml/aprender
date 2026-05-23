#!/usr/bin/env python3
"""Extract fenced code blocks from book/src/{cli,lib}/*.md as JSONL.

For each fenced ``bash or ``rust block, emit one JSON record:

    {
      "path": "book/src/cli/run.md",
      "line_start": 18,
      "line_end": 20,
      "lang": "bash",
      "cost": "model-required",
      "model": "qwen2.5-coder-1.5b",
      "code": "apr run qwen2.5-coder-1.5b ..."
    }

Cost class is read from the HTML comment one line above the opening
fence::

    <!-- example-cost: model-required model: qwen2.5-coder-1.5b -->
    ```bash
    apr run qwen2.5-coder-1.5b "What is 2+2?"
    ```

Defaults to "trivial" if no annotation is present.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^```(bash|rust)\s*$")
CLOSE_RE = re.compile(r"^```\s*$")
COST_RE = re.compile(r"^<!--\s*example-cost:\s*([^>]+?)\s*-->\s*$")

VALID_COSTS = {"trivial", "model-required", "gpu", "destructive", "interactive", "skip"}


def parse_cost_annotation(line: str) -> tuple[str, str | None]:
    """Parse the cost annotation HTML comment.

    Returns (cost_class, optional_model_name).
    Examples::

        <!-- example-cost: trivial -->
            -> ("trivial", None)
        <!-- example-cost: model-required model: qwen2.5-coder-1.5b -->
            -> ("model-required", "qwen2.5-coder-1.5b")
        <!-- example-cost: gpu -->
            -> ("gpu", None)
    """
    m = COST_RE.match(line)
    if not m:
        return ("trivial", None)
    payload = m.group(1).strip()
    # Allow both "cost" and "cost model: name" forms.
    parts = payload.split()
    cost = parts[0]
    model: str | None = None
    # Look for "model: <name>" pair.
    for i, tok in enumerate(parts):
        if tok == "model:" and i + 1 < len(parts):
            model = parts[i + 1]
            break
    if cost not in VALID_COSTS:
        cost = "trivial"  # unknown -> safe default
    return (cost, model)


def extract_from_file(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    blocks: list[dict] = []
    rel_path = str(path.relative_to(ROOT))

    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        lang = m.group(1)
        # Look one line UP for a cost annotation (HTML comment). The blank line
        # between annotation and fence is permitted in markdown; we also
        # accept the immediate predecessor.
        cost = "trivial"
        model: str | None = None
        # Search up to 2 lines back for the annotation.
        for back in (1, 2):
            j = i - back
            if j < 0:
                break
            cand = lines[j].strip()
            if not cand:
                continue
            if COST_RE.match(cand):
                cost, model = parse_cost_annotation(cand)
            break

        # Read code body until the closing fence.
        start_line = i + 1  # 1-indexed, points to opening fence
        code_lines: list[str] = []
        k = i + 1
        while k < len(lines) and not CLOSE_RE.match(lines[k]):
            code_lines.append(lines[k])
            k += 1
        end_line = k + 1  # 1-indexed, points to closing fence
        if k >= len(lines):
            # Unclosed fence -- skip but advance to avoid infinite loop.
            i = k
            continue

        record = {
            "path": rel_path,
            "line_start": start_line,
            "line_end": end_line,
            "lang": lang,
            "cost": cost,
            "code": "\n".join(code_lines),
        }
        if model is not None:
            record["model"] = model
        blocks.append(record)
        i = k + 1
    return blocks


def main() -> int:
    # Walk book/src/{cli,lib}/*.md deterministically (sorted).
    targets: list[Path] = []
    for sub in ("cli", "lib"):
        d = ROOT / "book" / "src" / sub
        if not d.is_dir():
            continue
        targets.extend(sorted(d.glob("*.md")))

    for f in targets:
        for record in extract_from_file(f):
            print(json.dumps(record, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
