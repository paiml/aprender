#!/usr/bin/env bash
# check_format_command_matrix.sh — {sharded, single-file} x {run, chat}: the model the
# user named is the model that answers, on every cell (#3024; the defect is #3022).
#
# WHY THIS EXISTS
# ---------------
# qwen-story-daily is the repo's nightly gate for the CLI's real-model story and it was
# green every night while #3022 shipped, because two whole axes of the surface it claims
# to cover are absent from it, and the defect lives at their intersection:
#
#   $ grep -c "apr chat" scripts/qwen-story.sh          -> 0   (chat is never invoked)
#   $ grep -c "index.json" scripts/qwen-story.sh        -> 0   (nothing is sharded)
#
# So `apr chat` on a `model.safetensors.index.json` silently loaded the built-in toy demo
# model — `Loaded Demo format in 0.00s (0.0 MB)`, zero tokens, EXIT 0 — under a banner
# printing the real model's path. A gate and a defect that never intersect prove nothing
# about each other, however green the gate is.
#
# THE ASSERTION IS THE GENERAL FORM, NOT THE INSTANCE
# ----------------------------------------------------
# Per cell: (1) the format the command NAMES is a real format, never Demo; (2) exit 0;
# (3) tokens were actually produced; (4) the size the command reports is not 0.0 MB. That
# catches "silently substituted something else" for any future format, not just sharded
# SafeTensors — which is what #3024 asks for, because the same `Path::extension()`
# reasoning will misfile the next multi-dot name too.
#
# NO NETWORK, NO NEW DOWNLOAD. The sharded fixture is DERIVED from a single-file model the
# story already holds, by scripts/make_sharded_safetensors.py. A shard boundary is a
# property of the index and the file split, not of the parameter count: two shards of a
# 0.5B exercise weight_map, the per-shard header rewrite and the cross-shard lookup
# exactly as four shards of a 7B do.
#
#   bash scripts/check_format_command_matrix.sh --self-test          # case table, no model
#   bash scripts/check_format_command_matrix.sh --matrix --apr <bin> [--models-dir DIR]
#                                                [--fixture DIR] [--out DIR]
#
# Exit: 0 every cell honest · 1 a cell substituted/failed · 2 environment (no binary, no model)
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_format_command_matrix

# ---------------------------------------------------------------------------
# judge_cell <command-label> <exit-code> <captured-output-file>
#   -> prints one verdict line; returns 0 honest / 1 substituted-or-broken
# The four rules above, applied to text the command actually printed. Reading a file
# rather than a pipe is deliberate: `producer | grep -q` returns 141 under pipefail when
# grep closes the pipe early, and this repo has shipped four false verdicts that way.
# ---------------------------------------------------------------------------
judge_cell() {
    local label=$1 rc=$2 out=$3 why=""
    if [ "$rc" != 0 ]; then
        printf 'FAIL  %-34s exit %s (a cell that cannot run is not a cell that passed)\n' "$label" "$rc"
        return 1
    fi
    if grep -qE 'Chat Demo \(Tiny Model\)|Loaded Demo format' "$out"; then
        why="the DEMO model answered for a file the user named (#3022)"
    elif ! grep -qE '=== (APR Run|Model Chat \()' "$out"; then
        why="no command banner in the output — nothing proves which model ran"
    fi
    if [ -z "$why" ] && grep -q 'Loaded .* format in ' "$out"; then
        local mb
        mb=$(sed -n 's/.*format in [0-9.]*s (\([0-9.]*\) MB).*/\1/p' "$out" | head -1)
        case "$mb" in ""|0|0.0|0.00) why="the load line reports ${mb:-no} MB — the real model was not read" ;; esac
    fi
    if [ -z "$why" ]; then
        local toks
        toks=$(sed -n 's/.*\[\([0-9][0-9]*\) tokens in .*/\1/p' "$out" | head -1)
        if [ -n "$toks" ] && [ "$toks" -eq 0 ] 2>/dev/null; then
            why="zero tokens generated"
        elif [ -z "$toks" ] && ! grep -q '^Output:' "$out"; then
            why="neither a token count nor an Output: section — nothing was generated"
        fi
    fi
    if [ -n "$why" ]; then
        printf 'FAIL  %-34s %s\n' "$label" "$why"
        return 1
    fi
    printf 'ok    %-34s honest: %s\n' "$label" "$(grep -m1 -E 'Loaded .* format in |^Output:' "$out" | cut -c1-70)"
    return 0
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/fcm.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    n=0; red=0
    row() { # row <want-rc> <label> <exit-code> <heredoc-file>
        local want=$1 label=$2 rc_in=$3 f=$4 rc=0
        n=$((n + 1)); judge_cell "$label" "$rc_in" "$f" > "$TD/verdict.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s %s (judge returned %s, wanted %s)\n        %s\n' "$n" "$label" "$rc" "$want" "$(cat "$TD/verdict.$n")"; red=1; fi
    }
    cat > "$TD/demo.txt" <<'EOF'
=== Chat Demo (Tiny Model) ===
Loading model...
Loaded Demo format in 0.00s (0.0 MB)
You: [0 tokens in 0.0s = 0.0 tok/s]
EOF
    cat > "$TD/good-chat.txt" <<'EOF'
=== Model Chat (Sharded SafeTensors) ===
Loaded Sharded SafeTensors format in 0.00s (988.1 MB)
Detected ChatML chat template
You: [12 tokens in 3.7s = 3.2 tok/s]
Assistant: 2 + 2 equals 4.
EOF
    cat > "$TD/good-run.txt" <<'EOF'
=== APR Run ===
Source: /models/model.safetensors.index.json
Output:
2 + 2 equals 4.
EOF
    cat > "$TD/zero-tokens.txt" <<'EOF'
=== Model Chat (SafeTensors Format) ===
Loaded SafeTensors format in 0.81s (988.1 MB)
You: [0 tokens in 0.1s = 0.0 tok/s]
EOF
    cat > "$TD/zero-mb.txt" <<'EOF'
=== Model Chat (Sharded SafeTensors) ===
Loaded Sharded SafeTensors format in 0.00s (0.0 MB)
You: [4 tokens in 0.1s = 40.0 tok/s]
EOF
    cat > "$TD/nothing.txt" <<'EOF'
some unrelated chatter with no banner at all
EOF
    row 1 "a Demo banner is RED"                     0 "$TD/demo.txt"
    row 0 "an honest sharded chat cell is GREEN"     0 "$TD/good-chat.txt"
    row 0 "an honest run cell is GREEN"              0 "$TD/good-run.txt"
    row 1 "zero tokens is RED even with a real format" 0 "$TD/zero-tokens.txt"
    row 1 "a 0.0 MB load line is RED"                0 "$TD/zero-mb.txt"
    row 1 "no banner at all is RED"                  0 "$TD/nothing.txt"
    row 1 "a non-zero exit is RED"                   6 "$TD/good-chat.txt"
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

# ---------------------------------------------------------------------------
MODE=""; APR_BIN=""; MODELS_DIR="${APR_MODELS_DIR:-$HOME/models}"; FIXTURE=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in
    --matrix) MODE=matrix; shift ;;
    --apr) APR_BIN=$2; shift 2 ;;
    --models-dir) MODELS_DIR=$2; shift 2 ;;
    --fixture) FIXTURE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) printf 'usage: %s --self-test | --matrix --apr <bin> [--models-dir D] [--fixture D] [--out D]\n' "$PROG" >&2; exit 2 ;;
esac; done
[ "$MODE" = matrix ] || { printf 'usage: %s --self-test | --matrix --apr <bin>\n' "$PROG" >&2; exit 2; }
[ -n "$APR_BIN" ] && [ -x "$APR_BIN" ] || { printf '%s: ENV - --apr <bin> must name an executable (never a bare `apr`)\n' "$PROG" >&2; exit 2; }

SINGLE="$MODELS_DIR/qwen2.5-coder-0.5b-instruct-safetensors/model.safetensors"
[ -f "$SINGLE" ] || { printf '%s: ENV - %s not on this host\n' "$PROG" "$SINGLE" >&2; exit 2; }
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/fcm-live.XXXXXX")}
case "$OUT" in *..*) printf '%s: refusing an --out with "..": %s\n' "$PROG" "$OUT" >&2; exit 2 ;; esac   # bashrs SEC010 (the 0.66.0 pre-publish dogfood, PMAT-1096)
mkdir -p "$OUT"

if [ -z "$FIXTURE" ]; then
    FIXTURE="$OUT/sharded"
    python3 "$ROOT/scripts/make_sharded_safetensors.py" "$SINGLE" "$FIXTURE" --shards 2 > /dev/null || {
        printf '%s: ENV - could not derive the sharded fixture from %s\n' "$PROG" "$SINGLE" >&2; exit 2; }
    for sib in config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json; do
        [ -f "$(dirname "$SINGLE")/$sib" ] && cp "$(dirname "$SINGLE")/$sib" "$FIXTURE/"
    done
fi
INDEX="$FIXTURE/model.safetensors.index.json"
[ -f "$INDEX" ] || { printf '%s: ENV - no %s\n' "$PROG" "$INDEX" >&2; exit 2; }

printf '=== format x command matrix on %s (%s) ===\n' "$(hostname -s)" "$("$APR_BIN" --version 2>/dev/null | head -1)"
rc=0
for cell in "single-file:$SINGLE" "sharded:$INDEX"; do
    layout=${cell%%:*}; model=${cell#*:}
    "$APR_BIN" run "$model" --prompt "What is 2+2?" --max-tokens 12 > "$OUT/$layout-run.txt" 2>&1
    judge_cell "$layout x apr run" "$?" "$OUT/$layout-run.txt" || rc=1
    printf 'What is 2+2?\n/quit\n' | "$APR_BIN" chat "$model" --max-tokens 12 > "$OUT/$layout-chat.txt" 2>&1
    judge_cell "$layout x apr chat" "${PIPESTATUS[1]}" "$OUT/$layout-chat.txt" || rc=1
done
printf 'matrix: 4 cells, rc=%s (records in %s)\n' "$rc" "$OUT"
exit "$rc"
