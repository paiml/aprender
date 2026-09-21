#!/usr/bin/env bash
# extract_llama_ftypes.sh — EXTRACT llama.cpp's `enum llama_ftype` (the scheme a
# GGUF declares in `general.file_type`) from upstream at the pinned comparator
# sha, and emit it as a checked-in fixture (#3762).
#
# WHY. `apr inspect` reported a Q4_K_M file as Q6_K: it named the dtype holding
# the most parameters (the 248k-vocab embedding) as the file's scheme. The file
# SAYS its scheme — `general.file_type` 15 — and the names of those ids belong to
# upstream. The M1 ruling (PMAT-3430, operator 2026-09-20 clauses 2'/3) settles
# where such a table comes from: upstream at `scripts/llama_pin.toml`
# `build_commit`, by reference; in-tree values are the defendant. This is the
# llama-side sibling of scripts/extract_ggml_traits.sh.
#
# THE OUTPUT IS A FIXTURE, NOT A BUILD STEP. CI never runs this; it reads the
# committed JSON. `crates/aprender-quant/tests/llama_ftypes_fixture.rs` fails if
# a row of `LLAMA_FTYPES` stops matching it, and `monorepo_invariants.rs` fails
# if the sha it records stops being the pin, so a pin bump turns RED in the bump
# PR until this is re-run.
#
# A HEADER PARSE IS ENOUGH HERE. Unlike ggml's `.type_size = sizeof(...)`, every
# value in `enum llama_ftype` is an integer literal. A row is REMOVED because its
# line is commented out, never because of its comment's wording (the ggml
# extraction's lesson). `LLAMA_FTYPE_GUESSED = 1024` is "not specified in the
# model file": a sentinel, never a file's scheme.
#
# awk, not python: the containerized CI jobs have no python3.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/extract_llama_ftypes.sh [-o OUT] [-s LLAMA_SRC]

  -o OUT         write the fixture here (default: crates/aprender-quant/fixtures/llama_ftypes.json)
  -s LLAMA_SRC   the llama.cpp checkout to read (default: $LLAMA_SRC, else
                 $HOME/src/llama.cpp-<build_commit>)

The checkout must be AT the pinned build_commit and clean. It need not be built.
USAGE
}

# The repo root is this script's parent, not `git rev-parse --show-toplevel`
# (git refuses the CI container's tree as "dubious ownership").
repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
[ -f "$repo_root/scripts/llama_pin.toml" ] || {
    echo "extract_llama_ftypes: no scripts/llama_pin.toml under '$repo_root'" >&2
    exit 2
}

out="$repo_root/crates/aprender-quant/fixtures/llama_ftypes.json"
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

# The oracle, by reference: llama_pin_get is the repo's one reader of the pin.
# Sourcing llama_bin.sh returns whether a llama-bench BINARY is built, which a
# header read does not need, so that status is not propagated.
# shellcheck source=scripts/llama_bin.sh
. "$repo_root/scripts/llama_bin.sh" || true
if ! command -v llama_pin_get >/dev/null 2>&1; then
    echo "extract_llama_ftypes: scripts/llama_bin.sh did not define llama_pin_get" >&2
    exit 2
fi
pin=$(llama_pin_get build_commit "$repo_root/scripts/llama_pin.toml") || pin=""
if [ -z "$pin" ] || [ "$pin" = "UNPINNED" ]; then
    echo "extract_llama_ftypes: scripts/llama_pin.toml declares no usable build_commit (got '${pin}')" >&2
    exit 2
fi

[ -n "$src" ] || src="$HOME/src/llama.cpp-$pin"
if ! git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
    echo "extract_llama_ftypes: no llama.cpp checkout at '$src' (pin $pin). Pass -s, or set LLAMA_SRC." >&2
    exit 3
fi
resolved=$(git -C "$src" rev-parse HEAD)
case "$resolved" in
    "$pin"*) : ;;
    *) echo "extract_llama_ftypes: '$src' is at $resolved, the pin is $pin — refusing to extract from a different oracle" >&2
       exit 3 ;;
esac
if [ -n "$(git -C "$src" status --porcelain)" ]; then
    echo "extract_llama_ftypes: '$src' has uncommitted changes — the oracle must be exactly the pinned commit" >&2
    exit 4
fi
header="$src/include/llama.h"
[ -f "$header" ] || { echo "extract_llama_ftypes: no $header" >&2; exit 3; }

work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT

# Emit to a temp file and move it into place only on success, so a refusal
# (exit 7) never leaves a half-written fixture behind.
if awk -v pin="$pin" -v resolved="$resolved" '
    function strip(e) {
        sub(/^LLAMA_FTYPE_MOSTLY_/, "", e); sub(/^LLAMA_FTYPE_ALL_/, "", e); sub(/^LLAMA_FTYPE_/, "", e)
        return e
    }
    /enum[ \t]+llama_ftype[ \t]*\{/ { inside = 1; next }
    inside && /^[ \t]*\};/ { inside = 0; next }
    inside {
        line = $0
        sub(/^[ \t]+/, "", line)
        removed = (line ~ /^\/\//)
        sub(/^\/\/[ \t]*/, "", line)
        if (line !~ /^LLAMA_FTYPE_[A-Z0-9_]+[ \t]*=[ \t]*[0-9]+/) next
        split(line, kv, /[ \t]*=[ \t]*/)
        name = kv[1]; id = kv[2] + 0
        status = removed ? "removed" : (name == "LLAMA_FTYPE_GUESSED" ? "sentinel" : "live")
        n++; ids[n] = id; names[n] = name; st[n] = status
        count[status]++
    }
    END {
        if (n == 0) { print "extract_llama_ftypes: no enum llama_ftype rows found" > "/dev/stderr"; exit 7 }
        if (count["sentinel"] != 1) { print "extract_llama_ftypes: expected exactly one sentinel (LLAMA_FTYPE_GUESSED), found " count["sentinel"] + 0 > "/dev/stderr"; exit 7 }
        for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (ids[i] == ids[j]) {
            print "extract_llama_ftypes: id " ids[i] " appears twice" > "/dev/stderr"; exit 7
        }
        printf "{\n"
        printf "  \"_comment\": \"GENERATED by scripts/extract_llama_ftypes.sh from upstream llama.cpp at the pinned comparator commit. Do not hand-edit: regenerate. aprender-quant asserts LLAMA_FTYPES matches these rows; monorepo_invariants asserts pin_build_commit equals scripts/llama_pin.toml build_commit.\",\n"
        printf "  \"source\": \"llama.cpp include/llama.h enum llama_ftype\",\n"
        printf "  \"pin_build_commit\": \"%s\",\n  \"resolved_sha\": \"%s\",\n", pin, resolved
        printf "  \"live_count\": %d,\n  \"removed_count\": %d,\n", count["live"], count["removed"]
        printf "  \"ftypes\": [\n"
        for (i = 1; i <= n; i++)
            printf "    {\"id\": %d, \"enum_name\": \"%s\", \"name\": \"%s\", \"status\": \"%s\"}%s\n", ids[i], names[i], strip(names[i]), st[i], (i < n ? "," : "")
        printf "  ]\n}\n"
    }' "$header" > "$work/fixture.json"; then
    mv "$work/fixture.json" "$out"
else
    exit 7
fi

echo "extract_llama_ftypes: wrote $out from $src @ $resolved (pin $pin)"
