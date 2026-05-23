#!/usr/bin/env bash
# FALSIFY-BOOK-EXAMPLE-COMPILES-001 (Phase 6 of BOOK-CLOSEOUT-001).
#
# For every fenced rust code block in book/src/lib/*.md, emit it into a
# single generated integration test (one `#[test]` per block) and ask
# `cargo check` to build it. Exit 0 iff every block compiles.
#
# Implementation note: we use a SINGLE generated file
# crates/aprender-core/tests/book_examples_compile.rs because
#   (a) it lets `cargo check -p aprender-core` build them in one shot
#       (much faster than per-block cargo invocations),
#   (b) each block becomes its own `mod block_<id>` so unrelated
#       imports don't collide,
#   (c) the generated file is deleted on exit.
#
# We deliberately do NOT compile bash blocks here — the executable gate
# (check_book_examples_executable.sh) handles those.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GEN="crates/aprender-core/tests/book_examples_compile.rs"
TMP="$ROOT/target/book-examples-tmp"
mkdir -p "$TMP"

cleanup() {
    rm -f "$GEN"
}
trap cleanup EXIT

# Write the test scaffold.
python3 scripts/_build_rust_compile_test.py > "$GEN"

# Count generated mods.
n=$(grep -c '^mod block_' "$GEN" || true)
echo "Generated $GEN with $n rust block(s)"

# Use cargo check (no need for rustc binary output -- only typeck).
# We restrict to the single integration test to avoid recompiling
# unrelated tests. Pass the feature flags that gate the optional modules
# (`audio`, `hf_hub`) so their lib-chapter use statements resolve.
echo "Running cargo check -p aprender-core --test book_examples_compile ..."
if cargo check \
    -p aprender-core \
    --test book_examples_compile \
    --features audio,hf-hub-integration \
    --message-format short 2>&1 | tee "$TMP/check.log"; then
    echo ""
    echo "FALSIFY-BOOK-EXAMPLE-COMPILES-001: PASS ($n rust block(s) compile)"
    exit 0
fi

# Extract failed mods from the error output.
echo ""
echo "FALSIFY-BOOK-EXAMPLE-COMPILES-001: FAIL"
grep -E "^error\[" "$TMP/check.log" | head -20 || true
exit 1
