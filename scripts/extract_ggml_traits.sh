#!/usr/bin/env bash
# extract_ggml_traits.sh — EXTRACT the ggml tensor-type table from upstream at
# the pinned comparator sha, and emit it as a checked-in fixture.
#
# WHY THIS EXISTS (PMAT-3430, operator ruling 2026-09-20 clauses 2'/2a/3).
# aprender carried three enums naming ggml tensor types, plus at least three
# separate byte-size tables for the same fact, and they disagreed — with each
# other and with upstream. The ruling settles the argument by naming an ORACLE:
# upstream ggml at `scripts/llama_pin.toml` `build_commit`, BY REFERENCE, never
# a literal sha. In-tree values are the defendant, never the reference.
#
# THE OUTPUT IS A FIXTURE, NOT A BUILD STEP. CI never runs this: it reads the
# committed JSON. The fixture RECORDS THE SHA IT WAS READ FROM, and
# `trueno_quant`'s tests assert that sha still equals the pin. A pin bump
# therefore turns the fixture RED until it is regenerated here, so the bump's
# cost is visible in the bump PR rather than discovered later (clause 2a; the
# general case is #3563).
#
# WHY IT COMPILES A PROBE INSTEAD OF PARSING THE TABLE. ggml's `type_traits[]`
# gives `.type_size = sizeof(block_q4_1)` — a C expression, not a number. Every
# in-tree table that got these numbers wrong got them wrong BY HAND. The only
# non-hand answer is to ask ggml itself, so the probe links against the pinned
# checkout's own libggml-base and calls `ggml_blck_size` / `ggml_type_size` /
# `ggml_type_name`.
#
# THE HEADER IS STILL PARSED, as a cross-check that can fail. `enum ggml_type`
# has EIGHT commented-out rows (4, 5, 31-33, 36-38) and only some of them say
# "removed" — `// GGML_TYPE_Q4_0_4_8 = 32,` says nothing at all. A first pass
# during the ruling classified rows by comment WORDING and returned 44/41/3
# instead of 43/35/8. Here a row is Removed because its line is COMMENTED, and
# the classification is then required to agree with the compiled table's
# blck_size == 0. If they disagree the script refuses rather than emitting.
#
# Exit codes (each names one failure, so a caller can tell them apart):
#   0 ok · 2 pin/declaration unreadable · 3 checkout missing or wrong sha
#   4 checkout dirty · 5 libggml-base not built · 6 probe build/run failed
#   7 header parse vs compiled table disagree
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/extract_ggml_traits.sh [-o OUT] [-s LLAMA_SRC]

  -o OUT         write the fixture here (default: crates/aprender-quant/fixtures/ggml_traits.json)
  -s LLAMA_SRC   the llama.cpp checkout to read (default: $LLAMA_SRC, else
                 $HOME/src/llama.cpp-<build_commit>)

The checkout must be AT the pinned build_commit, clean, and already built
(cmake -B build), because the probe links its libggml-base.
USAGE
}

# The repo root is this script's parent, not `git rev-parse --show-toplevel`:
# git refuses a tree whose owner differs from the caller ("dubious ownership"),
# which is exactly the CI container's situation, and locating our own sibling
# files never needed to ask git anything.
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
[ -f "$repo_root/scripts/llama_pin.toml" ] || {
    echo "extract_ggml_traits: no scripts/llama_pin.toml under '$repo_root'" >&2
    exit 2
}

out="$repo_root/crates/aprender-quant/fixtures/ggml_traits.json"
src="${LLAMA_SRC:-}"
opt=""
while getopts ":o:s:h" opt; do
    case "$opt" in
        o) out="$OPTARG" ;;
        s) src="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- the oracle
# BY REFERENCE. The sha is never typed into this script (clause 2').
# ONE producer for the declaration: `llama_pin_get` is the repo's only reader
# of scripts/llama_pin.toml, and a second hand-rolled sed here would be a
# second producer that can drift from it.
#
# ITS STATUS IS DELIBERATELY NOT PROPAGATED. Sourcing llama_bin.sh also runs
# `llama_bin_resolve` and returns THAT — whether a llama-bench BINARY is built
# on this host. Reading a C header needs no binary, so that status is not this
# script's business. What would be fatal is the reader not existing, which the
# next line checks.
# shellcheck source=scripts/llama_bin.sh
. "$repo_root/scripts/llama_bin.sh" || true
if ! command -v llama_pin_get >/dev/null 2>&1; then
    echo "extract_ggml_traits: scripts/llama_bin.sh did not define llama_pin_get" >&2
    exit 2
fi
pin=$(llama_pin_get build_commit "$repo_root/scripts/llama_pin.toml") || pin=""
if [ -z "$pin" ] || [ "$pin" = "UNPINNED" ]; then
    echo "extract_ggml_traits: scripts/llama_pin.toml declares no usable build_commit (got '${pin}')" >&2
    exit 2
fi

[ -n "$src" ] || src="$HOME/src/llama.cpp-$pin"
# `[ -d "$src/.git" ]` would have been wrong: the pinned checkout here is a
# git WORKTREE, whose .git is a file. Ask git, not the filesystem.
if ! git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
    echo "extract_ggml_traits: no llama.cpp checkout at '$src' (pin $pin). Pass -s, or set LLAMA_SRC." >&2
    exit 3
fi

resolved=$(git -C "$src" rev-parse HEAD)
case "$resolved" in
    "$pin"*) : ;;
    *) echo "extract_ggml_traits: '$src' is at $resolved, the pin is $pin — refusing to extract from a different oracle" >&2
       exit 3 ;;
esac

dirty=$(git -C "$src" status --porcelain)
if [ -n "$dirty" ]; then
    echo "extract_ggml_traits: '$src' has uncommitted changes — the oracle must be exactly the pinned commit" >&2
    exit 4
fi

lib_dir="$src/build/bin"
if [ ! -f "$lib_dir/libggml-base.so" ]; then
    echo "extract_ggml_traits: $lib_dir/libggml-base.so is absent — build the pinned checkout first:" >&2
    echo "    cmake -B build -S '$src' && cmake --build '$src/build' -j" >&2
    exit 5
fi

# ------------------------------------------------------------------ the probe
work=$(mktemp -d)
# "${work:?}" and not "$work": an empty variable here would make this `rm -rf /`
# (bashrs SEC011). mktemp cannot return empty under `set -e`, but the guard costs
# nothing and the failure mode is unrecoverable.
trap 'rm -rf "${work:?}"' EXIT

cat > "$work/probe.c" <<'PROBE'
#include <stdio.h>
#include "ggml.h"
int main(void) {
    printf("%d\n", (int) GGML_TYPE_COUNT);
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        printf("%d\t%s\t%lld\t%zu\t%d\n", t,
               ggml_type_name((enum ggml_type) t),
               (long long) ggml_blck_size((enum ggml_type) t),
               ggml_type_size((enum ggml_type) t),
               ggml_is_quantized((enum ggml_type) t) ? 1 : 0);
    }
    return 0;
}
PROBE

cc_bin="${CC:-cc}"
command -v "$cc_bin" >/dev/null 2>&1 || cc_bin=gcc
if ! "$cc_bin" -O0 -I "$src/ggml/include" -o "$work/probe" "$work/probe.c" \
        -L "$lib_dir" -lggml-base -Wl,-rpath,"$lib_dir" > "$work/cc.log" 2>&1; then
    echo "extract_ggml_traits: probe did not build:" >&2
    cat "$work/cc.log" >&2
    exit 6
fi
if ! "$work/probe" > "$work/probe.tsv" 2> "$work/probe.err"; then
    echo "extract_ggml_traits: probe did not run:" >&2
    cat "$work/probe.err" >&2
    exit 6
fi

# ------------------------------------------- header parse + cross-check + emit
# awk, NOT python3. This was python and `workspace-test-shard (3/3)` failed with
# `python3: command not found`: the containerized CI job has no python. The
# self-test that proves this classifier can FAIL has to run where CI runs, so the
# classifier may not depend on an interpreter that is absent there.
#
# The emitter writes to a temp file and is moved into place only on success, so a
# refusal (exit 7) never leaves a half-written fixture that the next reader
# mistakes for a real extraction.
if awk -v pin="$pin" -v resolved="$resolved" \
       -v probe="$work/probe.tsv" -v header="$src/ggml/include/ggml.h" \
       -f "$repo_root/scripts/extract_ggml_traits.awk" > "$work/fixture.json"; then
    mv "$work/fixture.json" "$out"
else
    exit 7
fi

echo "extract_ggml_traits: wrote $out from $src @ $resolved (pin $pin)"
