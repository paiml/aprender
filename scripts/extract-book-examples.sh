#!/usr/bin/env bash
# FALSIFY-BOOK-EXAMPLE-EXECUTES-001 / -COMPILES-001 (Phase 6)
#
# Walks book/src/{cli,lib}/*.md, emits JSONL on stdout — one record per
# fenced code block detected (bash + rust). Cost annotation comes from
# the sibling HTML comment one line above the opening fence:
#
#   <!-- example-cost: trivial -->
#   ```bash
#   apr --version
#   ```
#
# Cost classes (exhaustive):
#   trivial          runs in <2s, no model, no network, no GPU
#   model-required   needs ~/models/<model> (specify model: <name>)
#   gpu              needs CUDA
#   destructive      mutates external state (publish/upload/etc)
#   interactive      REPL/TUI that needs stdin
#
# Defaults to "trivial" if no annotation; the executable checker
# will fail loudly if a trivial example actually needs a model.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/extract_book_examples.py "$@"
