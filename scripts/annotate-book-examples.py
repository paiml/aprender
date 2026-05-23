#!/usr/bin/env python3
"""Insert ``<!-- example-cost: ... -->`` annotations into existing book chapters.

Classifies the first fenced bash/rust block in each chapter under
``book/src/{cli,lib}/*.md`` and inserts the appropriate cost annotation
one line above the opening fence (preserving the surrounding markdown).

Classification rules — keep these in sync with
``scripts/check_book_examples_executable.sh``.

This script is **idempotent** — if an annotation is already present
above the fence (within 2 lines), the file is left unchanged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^```(bash|rust)\s*$")
COST_RE = re.compile(r"^<!--\s*example-cost:\s*([^>]+?)\s*-->\s*$")

# Default model name used when no override is needed.
DEFAULT_MODEL = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# Inference-style commands needing a model in ~/models/.
MODEL_REQUIRED_CMDS = {
    "run",
    "chat",
    "serve",
    "qa",
    "bench",
    "eval",
    "inspect",
    "validate",
    "lint",
    "tensors",
    "trace",
    "debug",
    "diff",
    "explain",
    "flow",
    "hex",
    "profile",
    "tokenize",
    "tree",
    "check",
    "canary",
    "qualify",
    "tune",
    "finetune",
    "distill",
    "quantize",
    "prune",
    "compile",
    "merge",
    "compare-hf",
    "parity",
    "import",
    "convert",
    "export",
    "ptx-map",
    "pull",
}

# Mutating / destructive commands (CI substitutes --dry-run).
DESTRUCTIVE_CMDS = {
    "publish",
    "encrypt",
    "decrypt",
    "rm",
    "upload",
    "stamp",
}

# Interactive REPLs / TUIs that can't be run non-interactively.
INTERACTIVE_CMDS = {
    "tui",
    "cbtop",
    "monitor",
    "rosetta",
    "showcase",
    "code",  # `apr code` is an interactive coding assistant
}

# GPU-required commands (CUDA-specific tooling).
GPU_CMDS = {
    "gpu",
    "ptx",
}

# Trivial: anything with --help or --version, plus a few special pages.
TRIVIAL_OVERRIDE_FILES = {
    # These stubs use only `--help` invocation patterns and have no real model dep.
    # The script also looks at the body of the code to confirm.
}


def classify_apr_command(code: str, filename: str) -> tuple[str, str | None]:
    """Decide a cost class for a fenced bash code block.

    Returns ``(cost, optional_model_name)``.
    """
    stripped = code.strip()
    # Strip leading "$ " prompt if present (some examples use it).
    if stripped.startswith("$ "):
        stripped = stripped[2:]

    # Tokenize the first command line.
    first_line = stripped.split("\n", 1)[0].strip()

    # `--help` and `--version` are always trivial regardless of subcommand.
    if "--help" in first_line or "--version" in first_line:
        return ("trivial", None)

    # `apr help <command>` is trivial.
    if re.match(r"^apr\s+help\b", first_line):
        return ("trivial", None)

    # Detect the apr subcommand.
    m = re.match(r"^apr\s+([a-z][a-z0-9-]*)", first_line)
    if not m:
        # Not an apr invocation (could be Python, shell setup, etc.) — trivial.
        return ("trivial", None)
    cmd = m.group(1)

    if cmd in GPU_CMDS:
        return ("gpu", None)
    if cmd in INTERACTIVE_CMDS:
        return ("interactive", None)
    if cmd in DESTRUCTIVE_CMDS:
        return ("destructive", None)
    if cmd in MODEL_REQUIRED_CMDS:
        # Try to lift an explicit model name from the command line; fall back to default.
        # Heuristics: a positional arg that looks like a model path or HF id.
        tokens = first_line.split()
        model: str | None = None
        for tok in tokens[2:]:  # skip "apr <cmd>"
            if tok.startswith("-"):
                continue
            if tok.endswith(".gguf") or tok.endswith(".apr") or tok.endswith(".safetensors"):
                model = tok
                break
            if tok.startswith("qwen") or tok.startswith("Qwen") or tok.startswith("hf://"):
                model = tok
                break
        if model is None:
            model = DEFAULT_MODEL
        return ("model-required", model)

    # Lint commands are mostly invoked with --help (already handled above);
    # any remaining apr command is treated as trivial.
    return ("trivial", None)


def classify_rust_block(code: str) -> tuple[str, str | None]:
    """Rust blocks are always "trivial" — they're compile-only in CI."""
    return ("trivial", None)


def needs_annotation(lines: list[str], fence_index: int) -> bool:
    """Return True if no `example-cost` annotation exists within 2 lines above."""
    for back in (1, 2):
        j = fence_index - back
        if j < 0:
            return True
        s = lines[j].strip()
        if not s:
            continue
        if COST_RE.match(s):
            return False
        # Non-blank, non-annotation line: stop searching.
        return True
    return True


def annotate_file(path: Path) -> tuple[int, int]:
    """Annotate every bash/rust fence in ``path``.

    Returns ``(annotated_count, skipped_count)``.
    """
    lines = path.read_text().splitlines()
    out: list[str] = []
    annotated = 0
    skipped = 0
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        lang = m.group(1)
        # Check current `out` rather than `lines` because earlier rewrites may
        # have inserted an annotation already.
        # Walk back through `out` to find first non-blank line.
        already = False
        for back in (1, 2):
            j = len(out) - back
            if j < 0:
                break
            s = out[j].strip()
            if not s:
                continue
            if COST_RE.match(s):
                already = True
            break
        if already:
            skipped += 1
            out.append(lines[i])
            i += 1
            continue

        # Read the code body for classification.
        body: list[str] = []
        k = i + 1
        while k < len(lines) and not lines[k].startswith("```"):
            body.append(lines[k])
            k += 1
        code = "\n".join(body)

        if lang == "bash":
            cost, model = classify_apr_command(code, path.name)
        else:
            cost, model = classify_rust_block(code)

        if model:
            annotation = f"<!-- example-cost: {cost} model: {model} -->"
        else:
            annotation = f"<!-- example-cost: {cost} -->"

        # Insert annotation. If the immediately preceding line in `out` is
        # non-blank, we add a blank line before the annotation so the
        # surrounding paragraph isn't merged into the comment.
        if out and out[-1].strip() != "":
            out.append("")
        out.append(annotation)
        out.append(lines[i])
        annotated += 1
        i += 1

    new_text = "\n".join(out)
    # Preserve trailing newline if original had one.
    if path.read_text().endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    path.write_text(new_text)
    return (annotated, skipped)


def main() -> int:
    total_annotated = 0
    total_skipped = 0
    files: list[Path] = []
    for sub in ("cli", "lib"):
        d = ROOT / "book" / "src" / sub
        if not d.is_dir():
            continue
        files.extend(sorted(d.glob("*.md")))

    for f in files:
        a, s = annotate_file(f)
        total_annotated += a
        total_skipped += s
        if a > 0:
            print(f"  {f.relative_to(ROOT)}: +{a} annotation(s)")

    print(
        f"\nAnnotated {total_annotated} fence(s) across {len(files)} chapter(s); "
        f"skipped {total_skipped} already-annotated fence(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
