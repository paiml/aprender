#!/usr/bin/env bash
#
# check_no_fabricated_baselines.sh — a comparator baseline may be MEASURED or
# ABSENT, never asserted (F12, aprender#2679 / #2672).
#
# F12 — FABRICATED MEASUREMENT. A value carrying the form, units and
# provenance-shape of a measurement, produced without the measurement having
# been taken. It is distinct from its neighbours: not F7 (nothing passes
# falsely — the number may even be correct for some past run) and not F9
# (there is no coupled oracle; there is no oracle at all). The harm is that it
# is indistinguishable from evidence at the point of consumption, and it
# survives review because it LOOKS like the thing it replaces.
#
# Found 2026-08-24 in scripts/benchmark-2x-ollama.sh:27-29 —
#   OLLAMA_BASELINE="${OLLAMA_BASELINE:-291}"
#   OLLAMA_SINGLE="${OLLAMA_SINGLE:-120}"
#   OLLAMA_CPU="${OLLAMA_CPU:-15}"
# with ollama never invoked, and the same three literals emitted into JSON by
# scripts/benchmark-matrix.sh:396 as `ollama_baselines`.
#
# THE RULE: a shell variable naming a competitor's performance may not carry a
# numeric literal default. Deleting the four scripts removes the instances;
# this bans the CONSTRUCT, so the pattern cannot return under a new name.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
printf -- '--- no fabricated comparator baselines (F12) ------------------------\n'

# A competitor name adjacent to BASELINE/BENCH/TPS/THROUGHPUT, assigned a bare
# number. The `:-N}` default form is the one actually found; plain `=N` too.
# TWO shapes, because the first pattern written here caught only one of the
# two live instances. `benchmark-2x-ollama.sh` asserts via a shell variable;
# `benchmark-matrix.sh:396` emitted the same three numbers as a JSON literal:
#   JSON_RESULTS+='],"ollama_baselines":{"gpu_batched":291,"gpu_single":120,"cpu":15}}'
# A construct ban that sees one spelling is a ban on one spelling.
# The suffix ALLOWLIST was the wrong shape and the case table caught it: the
# real line `OLLAMA_CPU=15` carries no baseline-ish suffix, so a pattern keyed
# on BASELINE|TPS|THROUGHPUT would have missed one of the three live
# instances. Inverted: ANY competitor-prefixed variable assigned a numeric
# literal is suspect, MINUS an explicit denylist of configuration suffixes
# (a timeout or a port is a setting, not a measurement).
PATTERN_VAR='^[[:space:]]*((readonly|export|local|declare|typeset)[[:space:]]+(-[a-zA-Z]+[[:space:]]+)?)?(OLLAMA|LLAMA|LLAMACPP|VLLM|TGI|PYTORCH|TORCH)_[A-Z0-9_]*=("?\$\{[A-Z0-9_]+:-[0-9.]+\}"?|"?[0-9.]+"?)'
# A DECLARATION KEYWORD IS NOT AN ESCAPE HATCH. The anchor was `^[[:space:]]*`
# followed directly by the variable name, so `readonly OLLAMA_BASELINE=291` and
# `export OLLAMA_BASELINE=291` matched NOTHING — and `export` is the likelier
# form in a shell script than a bare assignment. The guard read as strict and
# was blind to the two spellings a real fabrication would most plausibly use.
# Caught by running the case table, not by reading the pattern: the plain
# `OLLAMA_TPS=163` fixture passed while `readonly OLLAMA_BASELINE=291` sailed
# through in the same sweep.
# RC/STATUS/CODE/PID/FD were added after scripts/llama_bin.sh's `LLAMA_PIN_RC=3`
# — a RETURN CODE — was flagged as a fabricated baseline. It surfaced only when
# PARITY-009 and PARITY-005 met in the cumulative stack head, which is what a
# cumulative head is for: two branches each green alone, one false positive
# together.
CONFIG_SUFFIX='(TIMEOUT|PORT|RETRIES|RETRY|SECONDS|SECS|MS|LIMIT|MAX|MIN|SIZE|COUNT|WORKERS|THREADS|RC|STATUS|CODE|PID|FD|LEVEL|VERSION)[A-Z0-9_]*='
PATTERN_JSON='"(ollama|llama|llamacpp|vllm|tgi|pytorch|torch)_[a-z0-9_]*(baseline|bench|tps|throughput|speed|latency)[a-z0-9_]*"[[:space:]]*:[[:space:]]*(\{[^}]*[0-9][^}]*\}|[0-9.]+)'
PATTERN="($PATTERN_VAR)|($PATTERN_JSON)"

# THIS FILE IS EXCLUDED FROM ITS OWN SCAN, and the reason is not convenience.
# Its case table deliberately CONTAINS the forbidden construct — that is what a
# must-match fixture IS. Scanning itself would make the guard permanently red
# against its own proof of discrimination.
#
# It passed for a while and then began failing, which is the interesting part:
# the universe is `git ls-files`, so while this file was UNTRACKED it was not
# scanned at all. The moment it was committed it began matching its own
# fixtures. That is the third instance of the tracked-only-universe shape in
# this epic (SHIM-2644-03; check_bench_threshold.sh in PARITY-008/009), so the
# universe below also unions the working tree — a new offender must not get a
# free pass merely by being uncommitted.
SELF="check_no_fabricated_baselines.sh"

# FULL-LINE COMMENTS ARE NOT CODE, and a guard that cannot tell the difference
# reds its own neighbours. This one did: check_no_claim_literals.sh:7 documents
# the very constructs banned here — "`OLLAMA_TPS=163`, `\"ollama_baseline\": 163`"
# — inside a comment explaining why the two guards differ. Matching it made a
# sibling guard's DOCUMENTATION a fabricated baseline, and the FAIL text offers
# no lever except widening the allowlist, so the pressure was to weaken the
# guard to describe it.
#
# Only lines whose first non-space character is `#` are dropped, and the true
# line number survives (grep -n runs first, the filter runs on its output). A
# trailing comment on a real assignment — `OLLAMA_TPS=163  # measured` — is
# still scanned and still caught, because the code is on that line too. Widening
# this to strip `#` to end-of-line would blind the guard to exactly that.
scan_file() {
    grep -nE "$PATTERN" "$1" 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*#'
}

scanned=0
hits=""
while IFS= read -r f; do
    [ "$(basename "$f")" = "$SELF" ] && continue
    scanned=$((scanned + 1))
    if scan_file "$f" | grep -qvE "$CONFIG_SUFFIX"; then
        hits="$hits $f"
    fi
done < <(
    { git ls-files 'scripts/*.sh' 'crates/*/scripts/*.sh' 2>/dev/null
      find scripts -maxdepth 2 -type f -name '*.sh' 2>/dev/null
    } | LC_ALL=C sort -u
)

# VACUITY: a scan over zero files sweeps clean and means nothing.
if [ "$scanned" -lt 20 ]; then
    printf 'FAIL  scanned only %s shell file(s); the universe collapsed. A clean\n' "$scanned"
    printf '      sweep over nothing is not a pass.\n'
    exit 1
fi

if [ -n "$hits" ]; then
    printf 'FAIL  fabricated comparator baseline(s):\n'
    for f in $hits; do
        printf '      %s\n' "$f"
        scan_file "$f" | grep -vE "$CONFIG_SUFFIX" | sed 's/^/        /'
    done
    printf '      A comparator baseline is MEASURED or ABSENT, never asserted.\n'
    printf '      Invoke the comparator and record its output, or record the\n'
    printf '      absence explicitly so the consuming gate can treat it as RED.\n'
    rc=1
else
    printf 'ok    %s shell file(s), no asserted competitor baseline\n' "$scanned"
fi

# CASE TABLE, not a single sentinel. Every guard regex in this repo that was
# wrong -- and the pinning walker was wrong sixteen times -- was caught by a
# must-match/must-not-match table and none by reading the pattern. This one was
# already wrong once: the first version missed the JSON-literal spelling.
ctl=$(mktemp -d) || exit 2
cat > "$ctl/must_match" <<'CASES'
OLLAMA_BASELINE="${OLLAMA_BASELINE:-291}"
OLLAMA_CPU=15
LLAMA_TPS="120"
JSON+='"ollama_baselines":{"gpu_batched":291,"gpu_single":120,"cpu":15}'
printf '{"llamacpp_throughput": 42.5}'
CASES
cat > "$ctl/must_not_match" <<'CASES'
OLLAMA_BASELINE="$(measure_ollama)"
OLLAMA_URL="http://localhost:11434"
echo "ollama baseline is measured, not asserted"
LLAMA_BIN="$(command -v llama-bench)"
LLAMA_PIN_RC=3
OLLAMA_EXIT_CODE=1
LLAMA_LOG_LEVEL=2
JSON+='"ollama_baselines":null'
readonly OLLAMA_TIMEOUT_SECONDS=30
CASES

tbl_bad=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    printf '%s\n' "$line" > "$ctl/one"
    if ! grep -E "$PATTERN" "$ctl/one" | grep -qvE "$CONFIG_SUFFIX"; then
        printf 'FAIL  MUST-MATCH not detected: %s\n' "$line"
        tbl_bad=1
    fi
done < "$ctl/must_match"

while IFS= read -r line; do
    [ -n "$line" ] || continue
    printf '%s\n' "$line" > "$ctl/one"
    if grep -E "$PATTERN" "$ctl/one" | grep -qvE "$CONFIG_SUFFIX"; then
        printf 'FAIL  MUST-NOT-MATCH falsely flagged: %s\n' "$line"
        tbl_bad=1
    fi
done < "$ctl/must_not_match"

if [ "$tbl_bad" -eq 0 ]; then
    printf 'ok    case table: %s must-match, %s must-not-match, all correct\n' \
        "$(grep -c . "$ctl/must_match")" "$(grep -c . "$ctl/must_not_match")"
else
    rc=1
fi
rm -rf "${ctl:?}"

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no comparator baseline is asserted rather than measured.\n'
else
    printf 'FAIL  see rows above (#2679).\n'
fi
exit "$rc"
