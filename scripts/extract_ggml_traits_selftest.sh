#!/usr/bin/env bash
# extract_ggml_traits_selftest.sh — the CASE TABLE for the ggml extractor.
#
# scripts/extract_ggml_traits.py classifies each ggml id live-or-Removed and
# refuses to emit a fixture when that classification disagrees with the
# compiled table. This proves that refusal FIRES, and that it fires for the
# right reasons, on synthetic inputs — no llama.cpp checkout, no compiler, so
# it runs anywhere CI runs.
#
# IT EXISTS BECAUSE THE OBVIOUS EXTRACTOR IS WRONG. A one-line grep written
# during the ruling returned 44 types / 41 live / 3 removed instead of
# 43 / 35 / 8: it counted GGML_TYPE_COUNT as a type, and it called a commented
# row "live" whenever the comment did not contain the word "removed" —
# `// GGML_TYPE_Q4_0_4_8 = 32,` contains no such word. Case 5 below is that
# exact line, and it must be classified Removed.
#
# Guard regexes ship a case table (CLAUDE.md, verification discipline 7): the
# pattern was wrong five times in this repo and review caught none of them.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
emitter="$root/scripts/extract_ggml_traits.py"
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT

fails=0
ran=0

# A minimal but REAL-SHAPED header: two live ids, one commented row whose
# comment explains itself, one that says nothing, and the COUNT sentinel.
write_header() {
    cat > "$1" <<'HDR'
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_Q4_0    = 1,
        // GGML_TYPE_Q4_2 = 2, support has been removed
        // GGML_TYPE_Q4_0_4_8 = 3,
        GGML_TYPE_COUNT   = 4,
    };
HDR
}

# The compiled table as the probe prints it: COUNT, then one row per id.
write_probe() {
    cat > "$1" <<'PROBE'
4
0	f32	1	4	0
1	q4_0	32	18	1
2	DEPRECATED	0	0	0
3	TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking	0	0	0
PROBE
}

check() {
    # check <name> <expected-exit> <header> <probe>
    ran=$((ran + 1))
    name=$1; want=$2; header=$3; probe=$4
    set +e
    python3 "$emitter" --probe "$probe" --header "$header" \
        --pin deadbeef --resolved deadbeefcafe --out "$work/out.json" > "$work/log" 2>&1
    got=$?
    set -e
    if [ "$got" -eq "$want" ]; then
        printf '  PASS  %-58s exit %s\n' "$name" "$got"
    else
        printf '  FAIL  %-58s exit %s, wanted %s\n' "$name" "$got" "$want"
        sed 's/^/        /' "$work/log"
        fails=$((fails + 1))
    fi
}

printf 'extract_ggml_traits self-test\n'

# --- MUST PASS -------------------------------------------------------------
write_header "$work/ok.h"
write_probe  "$work/ok.tsv"
check "control: header and compiled table agree" 0 "$work/ok.h" "$work/ok.tsv"

# Case 5, the trap: `// GGML_TYPE_Q4_0_4_8 = 3,` carries no explanation at all
# and must STILL be Removed. The control above already contains it, so assert
# the emitted classification rather than just the exit code.
removed=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["removed_count"])' "$work/out.json")
live=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["live_count"])' "$work/out.json")
ran=$((ran + 1))
if [ "$removed" = "2" ] && [ "$live" = "2" ]; then
    printf '  PASS  %-58s live %s removed %s\n' "a commented row with no 'removed' wording is Removed" "$live" "$removed"
else
    printf '  FAIL  %-58s live %s removed %s (wanted 2 and 2)\n' "a commented row with no 'removed' wording is Removed" "$live" "$removed"
    fails=$((fails + 1))
fi

# GGML_TYPE_COUNT is the sentinel, never a type.
ran=$((ran + 1))
names=$(python3 -c 'import json,sys; print(",".join(t["enum_name"] for t in json.load(open(sys.argv[1]))["types"]))' "$work/out.json")
case "$names" in
    *COUNT*) printf '  FAIL  %-58s %s\n' "GGML_TYPE_COUNT is not emitted as a type" "$names"; fails=$((fails + 1)) ;;
    *)       printf '  PASS  %-58s %s\n' "GGML_TYPE_COUNT is not emitted as a type" "$names" ;;
esac

# --- MUST FAIL (exit 7) ----------------------------------------------------
sed 's|// GGML_TYPE_Q4_0_4_8 = 3,|GGML_TYPE_Q4_0_4_8 = 3,|' "$work/ok.h" > "$work/uncommented.h"
check "header revives a row the compiled table calls dead" 7 "$work/uncommented.h" "$work/ok.tsv"

sed 's|        GGML_TYPE_Q4_0    = 1,|        // GGML_TYPE_Q4_0 = 1,|' "$work/ok.h" > "$work/buried.h"
check "header buries a row the compiled table calls live" 7 "$work/buried.h" "$work/ok.tsv"

sed 's|GGML_TYPE_COUNT   = 4,|GGML_TYPE_COUNT   = 5,|' "$work/ok.h" > "$work/count.h"
check "GGML_TYPE_COUNT disagrees with the compiled table" 7 "$work/count.h" "$work/ok.tsv"

# A live row whose compiled type_size is zero: the table was read from the
# wrong place, or the type is a stub.
sed 's|^1\tq4_0\t32\t18\t1$|1\tq4_0\t32\t0\t1|' "$work/ok.tsv" > "$work/zero.tsv"
check "a live id with type_size 0" 7 "$work/ok.h" "$work/zero.tsv"

printf '\n%s case(s), %s failure(s)\n' "$ran" "$fails"
[ "$fails" -eq 0 ] || exit 1
