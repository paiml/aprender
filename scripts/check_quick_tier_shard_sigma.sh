#!/usr/bin/env bash
# check_quick_tier_shard_sigma.sh -- the quick tier's tree readers run on every
# shard as a nextest hash partition, and the workspace-test verdict job owes
# their UNION (T46, #4678; Σ-executed #4433).
#
# Two halves, both read out of the files CI runs, never re-implemented:
#   * ci/sections.yml: the tree-reader step runs on every shard with
#     --partition "hash:${SHARD}/${SHARDS}", its junit is kept on every shard,
#     and the shard's staging step (which runs AFTER the quick Σ step) ships that
#     junit, plus shard 1's listed set, to the fan-in.
#   * .github/workflows/ci.yml: the fan-in step's `run:` script is EXTRACTED and
#     executed against planted shard artifacts. On tier=quick it must go RED on a
#     skipped shard (no junit), a wrong partition index (an id run on two shards),
#     and Σ executed != registered (an id lost or an id not listed).
# A step it cannot find is ENV rc=2, never a pass.
#
#   check_quick_tier_shard_sigma.sh              the case table
#   check_quick_tier_shard_sigma.sh --self-test  planted wrong rules must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${QUICK_SIGMA_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"
SEC="${QUICK_SIGMA_SECTIONS:-$ROOT/ci/sections.yml}"
FANIN_STEP="Σ-executed: the union of the shards is exactly the owed set (#4433)"
TREE_STEP="Quick tier: every test target that reads the tree (BSE-17)"
KEEP_STEP="Σ-executed: keep the quick tree-reader junit (the next nextest run overwrites it)"
QSIGMA_STEP="Σ-executed (quick tier): the tests this job ran are exactly the tests it owes (#4433)"
STAGE_STEP="Σ-executed: stage this shard's executed ids for the fan-in (#4433)"
# the only `if:` each step may carry: tree + keep on every quick shard, staging + upload on every sharded job
QUICK_IF="steps.tier.outputs.tier == 'quick'"
SHARDED_IF="matrix.shards != 1"

# step_run <file> <step name> -> that step's `run: |` body, dedented; empty if absent
step_run() {
    awk -v want="- name: \"$2\"" '
        index($0, want) && !found { found = 1; ind = -1; next }
        found && !body && /^ *run: \|/ { body = 1; next }
        found && !body && /^ *- (name|uses):/ { exit }
        body {
            if ($0 ~ /^[[:space:]]*$/) { print ""; next }
            match($0, /^ */)
            if (ind < 0) ind = RLENGTH
            if (RLENGTH < ind) exit
            print substr($0, ind + 1)
        }' "$1"
}
# step_if <file> <step name> -> that step's `if:` expression; empty if absent
step_if() {
    awk -v want="- name: \"$2\"" '
        index($0, want) && !found { found = 1; next }
        found && /^ *- (name|uses):/ { exit }
        found && /^ *if: / { sub(/^ *if: /, ""); print; exit }' "$1"
}
# step_line <file> <step name> -> the line number of that step; empty if absent
step_line() { grep -nF -- "- name: \"$2\"" "$1" | head -1 | cut -d: -f1; }
# upload_after <file> <step name> -> the first upload-artifact step after that step, as text
upload_after() {
    awk -v want="- name: \"$2\"" '
        index($0, want) && !found { found = 1; next }
        found && !up && /^ *- uses: actions\/upload-artifact@/ { up = 1; print; next }
        up && /^ *- (name|uses):/ { exit }
        up { print }' "$1"
}
# code -> stdin without comments, so a commented-out flag or cp never satisfies a row
code() { sed -e '/^[[:space:]]*#/d' -e 's/[[:space:]]#.*$//'; }
# exact_if <if-expression> <expected> -> pass iff the step's `if:` is exactly the expected one.
# An allowlist, not a pattern: a deny-list of shard spellings let `if: ${{ 0 == 1 }}` and
# `if: endsWith(github.job, '1')` through (lane-B review). Any other condition -- one that
# names a shard, is dead, or is merely reworded -- is RED until this table is changed with it.
exact_if() { if [ "$1" = "$2" ]; then echo pass; else echo fail; fi; }

# The planted shard artifacts: a universe of six ids in two binaries.
junit() { # junit <file> <binary:test>... -- nextest's attribute order, name first
    local id
    { printf '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="nextest-run">\n<testsuite name="t">\n'
      for id in "${@:2}"; do printf '<testcase name="%s" classname="%s" timestamp="2026-10-04T00:00:00Z" time="0.1">\n</testcase>\n' "${id##*:}" "${id%:*}"; done
      printf '</testsuite>\n</testsuites>\n'; } > "$1"
}
LIST='{"rust-suites":{"crate-a":{"testcases":{"t1":{"filter-match":{"status":"matches"}},"t2":{"filter-match":{"status":"matches"}},"t3":{"filter-match":{"status":"matches"}}}},"crate-b::it":{"testcases":{"t4":{"filter-match":{"status":"matches"}},"t5":{"filter-match":{"status":"matches"}},"t6":{"filter-match":{"status":"matches"}}}}}}'
P1="crate-a:t1 crate-b::it:t5"
P2="crate-a:t2 crate-b::it:t4"
P3="crate-a:t3 crate-b::it:t6"

# fanin <script> <dir> TIERS S1 S2 S3 [LIST] -> exit status of the fan-in over that layout.
# Sn is a space-separated id list, or "-" for a shard that staged no junit.
fanin() {
    local s=$1 d=$2 tiers=$3 n=1 p list=${7-$LIST}
    rm -rf -- "${d:?}/sigma"; mkdir -p "$d/sigma"
    read -ra t <<< "$tiers"
    for p in "$4" "$5" "$6"; do
        mkdir -p "$d/sigma/sigma-shard-$n"
        [ "${t[n - 1]:-}" = none ] || printf '%s\n' "${t[n - 1]:-quick}" > "$d/sigma/sigma-shard-$n/tier"
        # shellcheck disable=SC2086
        [ "$p" = - ] || junit "$d/sigma/sigma-shard-$n/quick-tree.junit.xml" $p
        n=$((n + 1))
    done
    [ "$list" = - ] || printf '%s\n' "$list" > "$d/sigma/sigma-shard-1/quick-tree.list.json"
    (cd "$ROOT" && RUNNER_TEMP="$d" bash -c "$s") > "$d/fanin.out" 2>&1
}

table() { # table <workflow> <sections> -> 0 iff every row holds, 2 when a step is missing
    local f tree keep qs stage up bad=0 n=0 want got label w
    f=$(step_run "$1" "$FANIN_STEP"); tree=$(step_run "$2" "$TREE_STEP"); qs=$(step_run "$2" "$QSIGMA_STEP")
    stage=$(step_run "$2" "$STAGE_STEP" | code); keep=$(step_run "$2" "$KEEP_STEP"); up=$(upload_after "$2" "$STAGE_STEP")
    tree=$(code <<< "$tree")
    for want in f tree keep qs stage up; do
        [ -n "${!want}" ] || { printf 'ENV   a step is missing (%s): cannot judge, not a pass\n' "$want" >&2; return 2; }
    done
    w=$(mktemp -d "${TMPDIR:-/tmp}/quick-sigma.XXXXXX") || return 2
    row() { # row WANT LABEL got-status
        n=$((n + 1)); want=$1 label=$2 got=$3
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-4s %s\n' "$n" "$want" "$label"
        else printf 'FAIL  row %-2s wanted %s, got %s: %s\n' "$n" "$want" "$got" "$label" >&2; bad=1; fi
    }
    st() { if "$@"; then echo pass; else echo fail; fi; }
    has() { if grep -qF -- "$2" <<< "$1"; then echo pass; else echo fail; fi; }
    hasnt() { if grep -qF -- "$2" <<< "$1"; then echo fail; else echo pass; fi; }
    # sections.yml: the tree readers are partitioned, on every shard
    row pass "tree step runs nextest with this shard's hash partition" "$(has "$tree" '--partition "hash:${SHARD}/${SHARDS}"')"
    row pass "tree step is not shard-1-only" "$(exact_if "$(step_if "$2" "$TREE_STEP")" "$QUICK_IF")"
    row pass "tree junit is kept on every shard" "$(exact_if "$(step_if "$2" "$KEEP_STEP")" "$QUICK_IF")"
    row pass "staging runs on every shard" "$(exact_if "$(step_if "$2" "$STAGE_STEP")" "$SHARDED_IF")"
    row pass "the upload runs on every shard" "$(exact_if "$(sed -n 's/^ *if: //p' <<< "$up" | head -1)" "$SHARDED_IF")"
    row pass "the upload names one artifact per shard" "$(if grep -qE '^ *name: sigma-shard-\$\{\{ matrix\.shard \}\}$' <<< "$(code <<< "$up")"; then echo pass; else echo fail; fi)"
    row pass "staging ships every shard's tree junit" "$(has "$stage" 'cp "$sig/quick-tree.junit.xml" sigma-shard/')"
    row pass "staging ships shard 1's listed tree set" "$(has "$stage" 'cp "$sig/quick-tree.list.json" sigma-shard/')"
    row pass "staging runs after the tree step and the quick Σ step" \
        "$(st test "$(step_line "$2" "$STAGE_STEP")" -gt "$(step_line "$2" "$QSIGMA_STEP")" -a "$(step_line "$2" "$QSIGMA_STEP")" -gt "$(step_line "$2" "$TREE_STEP")")"
    # ci.yml: the fan-in owes the union on tier=quick
    row pass "fan-in: three disjoint partitions, union == listed" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P2" "$P3")"
    row pass "fan-in: one shard's partition is empty, union == listed" "$(st fanin "$f" "$w" 'quick quick quick' "$P1 $P2" "" "$P3")"
    row fail "fan-in: shard 2 skipped (no junit)" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" - "$P3")"
    row fail "fan-in: shard 2 skipped while its partition held nothing" "$(st fanin "$f" "$w" 'quick quick quick' "$P1 $P2" - "$P3")"
    row fail "fan-in: shard 3 skipped (no junit), shards 1+2 cover nothing more" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P2" -)"
    row fail "fan-in: wrong index -- shard 2 ran partition 1" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P1" "$P3")"
    row fail "fan-in: one id run on two shards, union still == listed" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P2 crate-a:t1" "$P3")"
    row fail "fan-in: Σ executed < registered (an id lost)" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "crate-a:t2" "$P3")"
    row fail "fan-in: Σ executed > registered (an id not listed)" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P2 crate-c:t9" "$P3")"
    row fail "fan-in: shard 1 staged no listed set" "$(st fanin "$f" "$w" 'quick quick quick' "$P1" "$P2" "$P3" -)"
    row fail "fan-in: every shard ran nothing" "$(st fanin "$f" "$w" 'quick quick quick' "" "" "")"
    row fail "fan-in: shard 2 staged no tier file" "$(st fanin "$f" "$w" 'quick none quick' "$P1" "$P2" "$P3")"
    row fail "fan-in: the shards disagree on the tier" "$(st fanin "$f" "$w" 'quick full quick' "$P1" "$P2" "$P3")"
    [ -z "${QUICK_SIGMA_DEBUG:-}" ] || cat "$w/fanin.out" >&2
    rm -rf -- "${w:?}"
    return "$bad"
}

case "${1:-}" in -h | --help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

for f in "$WF" "$SEC"; do [ -f "$f" ] || { printf 'ENV   %s is missing\n' "$f" >&2; exit 2; }; done

if [ "${1:-}" = "--self-test" ]; then
    echo "=== quick-tier shard Σ: planted wrong rules must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/quick-sigma-st.XXXXXX") || exit 2
    trap 'rm -rf -- "${d:?}"' EXIT
    # ci.yml mutants -- each one a fan-in that accepts a broken quick tier
    awk '/if \[ "\$1" = quick \]; then/ { skip = 1 } skip && /^          fi$/ { skip = 0; next } !skip' "$WF" > "$d/exempt.yml"   # origin/main: quick is exempt
    # a missing junit read as an empty partition: the skipped shard is caught only where the union is short
    sed 's/\[ -f "\$f" \] || { echo "::error::Σ: shard \$n staged no quick tree-reader junit: its partition was not run"; exit 1; }/[ -f "$f" ] || printf "<testsuites\/>" > "$f"/' "$WF" > "$d/nojunit.yml"
    sed 's/\[ -z "\$dup" \] ||/true ||/' "$WF" > "$d/nodup.yml"
    sed 's/"quick tree readers (3 shards)" "\${j\[@\]}"/"quick tree readers (3 shards)" "${j[@]:0:2}"/' "$WF" > "$d/twoj.yml"
    sed 's/python3 scripts\/ci\/sigma_executed.py nextest --kind "quick tree readers (3 shards)"/true/' "$WF" > "$d/nounion.yml"
    sed 's/          \[ "\$1" = "\$2" \] \&\& \[ "\$2" = "\$3" \] ||/          true ||/' "$WF" > "$d/tiermix.yml"
    # sections.yml mutants -- each one runs or ships the tree readers wrongly
    sed 's/ --partition "hash:\${SHARD}\/\${SHARDS}" --no-tests=warn/ --no-tests=warn/' "$SEC" > "$d/nopart.yml"
    sed 's/--partition "hash:\${SHARD}\/\${SHARDS}" --no-tests/--partition "hash:1\/${SHARDS}" --no-tests/' "$SEC" > "$d/index.yml"
    awk -v s="- name: \"$TREE_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { sub(/ *$/, " \\&\\& matrix.shard == 1"); hit = 0 } 1' "$SEC" > "$d/shard1.yml"
    awk -v s="- name: \"$KEEP_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { sub(/ *$/, " \\&\\& matrix.shard == 1"); hit = 0 } 1' "$SEC" > "$d/keep1.yml"
    grep -vF 'cp "$sig/quick-tree.junit.xml" sigma-shard/' "$SEC" > "$d/nostage.yml"
    # staging moved back before the quick tier: the old order, which stages before the junit exists
    awk -v st="- name: \"$STAGE_STEP\"" -v tr="- name: \"$TREE_STEP\"" '
        { lines[NR] = $0 } index($0, st) { s = NR } index($0, tr) { t = NR }
        END {
            # the staging block is the step line through the upload-artifact step (6 lines after its `uses:`)
            for (e = s + 1; e <= NR && lines[e] !~ /upload-artifact/; e++); e += 5
            for (i = 1; i <= NR; i++) {
                if (i == t) for (k = s; k <= e; k++) print lines[k]
                if (i >= s && i <= e) continue
                print lines[i]
            }
        }' "$SEC" > "$d/order.yml"
    # lane-B mutants: a single-shard or dead `if:`, a shared artifact name, a commented-out flag or cp
    awk -v s="- name: \"$STAGE_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { sub(/if: .*/, "if: matrix.shard == 1"); hit = 0 } 1' "$SEC" > "$d/stage1.yml"
    awk -v s="- name: \"$STAGE_STEP\"" 'index($0, s) { hit = 1 } hit && /upload-artifact/ { up = 1 } up && /^ *if: / { sub(/if: .*/, "if: false"); hit = up = 0 } 1' "$SEC" > "$d/updead.yml"
    sed 's/name: sigma-shard-\${{ matrix.shard }}/name: sigma-shard-1/' "$SEC" > "$d/upname.yml"
    awk -v s="- name: \"$TREE_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { sub(/ *$/, " \\&\\& env.SHARD == '"'"'1'"'"'"); hit = 0 } 1' "$SEC" > "$d/envshard.yml"
    sed 's/-E "\$EXPR" --partition/-E "$EXPR" # --partition/' "$SEC" > "$d/cmtpart.yml"
    sed 's/^\( *\)cp "\$sig\/quick-tree.junit.xml" sigma-shard\/quick-tree.junit.xml$/\1true # cp "$sig\/quick-tree.junit.xml" sigma-shard\/quick-tree.junit.xml/' "$SEC" > "$d/cmtcp.yml"
    # lane-B round 2 survivors: a dead expression, a job-name test, a suffixed artifact name
    awk -v s="- name: \"$STAGE_STEP\"" 'index($0, s) { hit = 1 } hit && /upload-artifact/ { up = 1 } up && /^ *if: / { sub(/if: .*/, "if: ${{ 0 == 1 }}"); hit = up = 0 } 1' "$SEC" > "$d/updead0.yml"
    awk -v s="- name: \"$TREE_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { sub(/ *$/, " \\&\\& endsWith(github.job, '"'"'1'"'"')"); hit = 0 } 1' "$SEC" > "$d/endsjob.yml"
    sed 's/name: sigma-shard-\${{ matrix.shard }}$/name: sigma-shard-${{ matrix.shard }}-x/' "$SEC" > "$d/upsuffix.yml"
    printf 'jobs: {}\n' > "$d/none.yml"
    bad=0
    for m in exempt nojunit nodup twoj nounion tiermix nopart index shard1 keep1 nostage order stage1 updead upname envshard cmtpart cmtcp updead0 endsjob upsuffix; do
        case "$m" in exempt | nojunit | nodup | twoj | nounion | tiermix) wf="$d/$m.yml" sec="$SEC" src="$WF" ;; *) wf="$WF" sec="$d/$m.yml" src="$SEC" ;; esac
        if cmp -s "$src" "$d/$m.yml"; then printf 'FAIL  the %s mutant did not apply (its anchor is gone)\n' "$m"; bad=1; continue; fi
        if table "$wf" "$sec" > "$d/out" 2>&1; then printf 'FAIL  the planted %s rule passed the table\n' "$m"; bad=1
        else printf 'ok    the planted %-8s rule is RED: %s row(s), e.g. %s\n' "$m" "$(grep -c '^FAIL' "$d/out")" "$(grep -m1 '^\(FAIL\|ENV\)' "$d/out" | cut -c7-90)"; fi
    done
    rc=0; table "$d/none.yml" "$SEC" > /dev/null 2>&1 || rc=$?
    if [ "$rc" = 2 ]; then printf 'ok    a workflow with no fan-in step is ENV rc=2, never a pass\n'
    else printf 'FAIL  no fan-in step gave rc=%s\n' "$rc"; bad=1; fi
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== quick tier: tree readers partitioned per shard, fan-in owes their union (check_quick_tier_shard_sigma.sh) ==="
table "$WF" "$SEC"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
