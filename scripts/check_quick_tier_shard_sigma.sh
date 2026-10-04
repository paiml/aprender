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
#   check_quick_tier_shard_sigma.sh --update-golden  re-pin the jobs after a reviewed change
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

# step_block <file> <step name> -> that step's lines up to the next step, as text
step_block() {
    awk -v want="- name: \"$2\"" '
        index($0, want) && !found { found = 1; print; next }
        found && /^ *- (name|uses):/ { exit }
        found { print }' "$1"
}
# stage_run <script> <dir> <tier> <shard> -> the files the staging step ships, sorted, one line.
# It EXECUTES the staging body against a planted $RUNNER_TEMP/sigma, so a flipped tier test or a
# junit `cp` moved under `[ "$SHARD" = 1 ]` changes what ships -- text checks cannot see either.
stage_run() {
    local s=$1 d=$2
    rm -rf -- "${d:?}/st"; mkdir -p "$d/st/rt/sigma" "$d/st/wd"
    : > "$d/st/rt/sigma/quick-tree.junit.xml"; : > "$d/st/rt/sigma/quick-tree.list.json"
    (cd "$d/st/wd" && RUNNER_TEMP="$d/st/rt" TIER="$3" SHARD="$4" bash -c "$s") > /dev/null 2>&1 || { echo "staging-failed"; return; }
    find "$d/st/wd/sigma-shard" -type f -printf '%f\n' 2> /dev/null | sort | paste -sd' '
}

# The golden: every line of the jobs the shard split lives in, pinned by digest. The rows above
# read the steps they know; a job-level `if:`, `needs:`, `env:`, `continue-on-error:` or matrix
# edit (the #4717 m18-m21 class), or a step flag no row names, is outside them. Any change to a
# pinned job is RED until the golden is regenerated in the same diff, where review sees both.
GOLDEN="${QUICK_SIGMA_GOLDEN:-$ROOT/ci/goldens/quick-tier-shard-sigma.sha256}"
PINNED="workflow jobs workspace-test-shard
workflow jobs workspace-test
sections matrix-pins workspace-test-shard
sections jobs workspace-test-shard"
# job_block <file> <parent> <job> -> the job's lines, from its key under the top-level <parent>
# up to the next NON-COMMENT line at indent <= 2. A comment or blank ends nothing: it is held and
# printed only if the job goes on, so `  # note` inside a job cannot cut its tail out of the
# digest (lane E, round 4). A second <job> key under <parent> is rc=3: a YAML loader keeps the
# last duplicate, which is not the block pinned here.
job_block() {
    awk -v p="$2:" -v j="  $3:" '
        $0 == p { inp = 1; next }
        inp && /^[^ #]/ { inp = 0 }
        inp && $0 == j { seen++ }
        inb && !/^[[:space:]]*#/ && (/^[^ ]/ || /^ [^ ]/ || /^  [^ ]/) { inb = 0; done = 1 }
        inp && !inb && !done && $0 == j { inb = 1 }
        inb { if ($0 ~ /^[[:space:]]*(#.*)?$/) { held = held $0 "\n"; next } printf "%s%s\n", held, $0; held = "" }
        END { if (seen > 1) exit 3 }' "$1"
}
# golden_compute <workflow> <sections> -> the manifest; rc=2 (ENV) when a pinned job is absent or
# keyed twice. It also pins each file's top-level key set: a workflow-level `env:`, `defaults:` or
# `permissions:` reaches every pinned job without touching one of their lines (lane F, round 4).
golden_compute() {
    local role parent job file block rc
    while read -r role parent job; do
        if [ "$role" = workflow ]; then file=$1; else file=$2; fi
        rc=0; block=$(job_block "$file" "$parent" "$job") || rc=$?
        [ "$rc" = 3 ] && { printf 'ENV   pinned job %s.%s is keyed twice in the %s: cannot judge, not a pass\n' "$parent" "$job" "$role" >&2; return 2; }
        [ -n "$block" ] || { printf 'ENV   pinned job %s.%s is missing from the %s: cannot judge, not a pass\n' "$parent" "$job" "$role" >&2; return 2; }
        printf '%s  %s:%s.%s\n' "$(printf '%s\n' "$block" | sha256sum | cut -c1-64)" "$role" "$parent" "$job"
    done <<< "$PINNED"
    printf '%s  workflow:top-level-keys\n' "$(grep -E '^[^ #]' "$1" | cut -d: -f1 | sha256sum | cut -c1-64)"
    printf '%s  sections:top-level-keys\n' "$(grep -E '^[^ #]' "$2" | cut -d: -f1 | sha256sum | cut -c1-64)"
}
golden_header() {
    printf '%s\n' "# The jobs the quick-tier shard split lives in, pinned by the sha256 of their lines" \
        "# (scripts/check_quick_tier_shard_sigma.sh). A change to any of them is RED until this file" \
        "# is regenerated in the same diff: bash scripts/check_quick_tier_shard_sigma.sh --update-golden"
}
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
    local f tree keep qs stage up bad=0 n=0 want got label w gold
    gold=$(golden_compute "$1" "$2") || return 2
    [ -f "$GOLDEN" ] || { printf 'ENV   no golden at %s: cannot judge, not a pass\n' "$GOLDEN" >&2; return 2; }
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
    # both files: the pinned jobs, whole -- job keys and every step, not only the steps named below
    row pass "the pinned jobs are byte-identical to the golden" "$(exact_if "$gold" "$(grep -v '^#' "$GOLDEN")")"
    grep -v '^#' "$GOLDEN" | sort | comm -13 - <(sort <<< "$gold") | sed 's/^[0-9a-f]*  /      changed since the golden: /' >&2
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
    row pass "staging on quick shard 2 ships its tree junit and tier, no listed set" \
        "$(exact_if "$(stage_run "$stage" "$w" quick 2)" "quick-tree.junit.xml tier")"
    row pass "staging on quick shard 1 ships its tree junit, the listed set and tier" \
        "$(exact_if "$(stage_run "$stage" "$w" quick 1)" "quick-tree.junit.xml quick-tree.list.json tier")"
    row pass "no tree, keep or staging step may continue on error (a red test must fail the shard)" \
        "$(if step_block "$2" "$TREE_STEP" | code | grep -q 'continue-on-error:' || step_block "$2" "$KEEP_STEP" | code | grep -q 'continue-on-error:' || step_block "$2" "$STAGE_STEP" | code | grep -q 'continue-on-error:'; then echo fail; else echo pass; fi)"
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

case "${1:-}" in -h | --help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

for f in "$WF" "$SEC"; do [ -f "$f" ] || { printf 'ENV   %s is missing\n' "$f" >&2; exit 2; }; done

if [ "${1:-}" = "--update-golden" ]; then
    m=$(golden_compute "$WF" "$SEC") || exit 2
    mkdir -p "$(dirname "$GOLDEN")" && { golden_header; printf '%s\n' "$m"; } > "$GOLDEN" || exit 2
    printf 'wrote %s\n' "${GOLDEN#"$ROOT"/}"; exit 0
fi

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
    # round 3 (lane A): the staging shell conditions and the tree step's failure semantics
    sed 's/^\( *\)if \[ "\$TIER" = quick \]; then$/\1if [ "$TIER" = full ]; then/' "$SEC" > "$d/stagetier.yml"
    awk '/^ *cp "\$sig\/quick-tree.junit.xml" sigma-shard\/quick-tree.junit.xml$/ { ind = $0; sub(/[^ ].*/, "", ind); print ind "if [ \"$SHARD\" = 1 ]; then"; print ind "  " substr($0, length(ind) + 1); print ind "fi"; next } 1' "$SEC" > "$d/stagejunit1.yml"
    awk -v s="- name: \"$TREE_STEP\"" 'index($0, s) { hit = 1 } hit && /^ *if: / { print; ind = $0; sub(/[^ ].*/, "", ind); print ind "continue-on-error: true"; hit = 0; next } 1' "$SEC" > "$d/treecoe.yml"
    printf 'jobs: {}\n' > "$d/none.yml"
    bad=0
    for m in exempt nojunit nodup twoj nounion tiermix nopart index shard1 keep1 nostage order stage1 updead upname envshard cmtpart cmtcp updead0 endsjob upsuffix stagetier stagejunit1 treecoe; do
        case "$m" in exempt | nojunit | nodup | twoj | nounion | tiermix) wf="$d/$m.yml" sec="$SEC" src="$WF" ;; *) wf="$WF" sec="$d/$m.yml" src="$SEC" ;; esac
        if cmp -s "$src" "$d/$m.yml"; then printf 'FAIL  the %s mutant did not apply (its anchor is gone)\n' "$m"; bad=1; continue; fi
        # judged with a golden re-pinned to the mutant: the semantic rows must kill it on their own
        QUICK_SIGMA_WORKFLOW="$wf" QUICK_SIGMA_SECTIONS="$sec" QUICK_SIGMA_GOLDEN="$d/$m.sha256" bash "$0" --update-golden > /dev/null 2>&1 || { printf "FAIL  the %s mutant could not be re-pinned\n" "$m"; bad=1; continue; }
        rc=0; GOLDEN="$d/$m.sha256" table "$wf" "$sec" > "$d/out" 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then printf 'FAIL  the planted %s rule passed the table\n' "$m"; bad=1
        elif [ "$rc" != 1 ]; then printf 'FAIL  the planted %s rule was rc=%s (ENV), not a row going RED\n' "$m" "$rc"; bad=1
        else printf 'ok    the planted %-8s rule is RED: %s row(s), e.g. %s\n' "$m" "$(grep -c '^FAIL' "$d/out")" "$(grep -m1 '^\(FAIL\|ENV\)' "$d/out" | cut -c7-90)"; fi
    done
    # The golden's case table. edit <file> <anchor> <match> sub|after <line>: after the first line
    # containing <anchor>, the first line containing <match> is replaced by, or followed by, <line>.
    edit() {
        awk -v a="$2" -v m="$3" -v op="$4" -v t="$5" '
            index($0, a) { in_a = 1 }
            in_a && !done && index($0, m) { done = 1; if (op == "sub") { print t; next } print; print t; next }
            1' "$1"
    }
    tn='--partition "hash:${SHARD}/${SHARDS}" --no-tests=warn'
    # round-3 survivors (lanes C and D): each one the rows above passed
    edit "$SEC" "$TREE_STEP" "$tn" sub "            cargo nextest run --profile ci \$pkgs --lib --tests -E \"\$EXPR\" $tn || true" > "$d/g-treetrue.yml"
    edit "$SEC" "$TREE_STEP" "$tn" sub '            cargo nextest run --profile ci $pkgs --lib --tests -E "$EXPR" --partition "hash:${SHARD}/${SHARDS}"' > "$d/g-nonotests.yml"
    edit "$SEC" "$TREE_STEP" "$tn" sub "            cargo nextest run --profile ci \$pkgs --lib --tests $tn" > "$d/g-noexpr.yml"
    edit "$SEC" "$TREE_STEP" "$tn" sub "            cargo nextest run --profile ci \$pkgs --lib -E \"\$EXPR\" $tn" > "$d/g-libonly.yml"
    edit "$SEC" "$TREE_STEP" "-e SHARD -e SHARDS \\" sub "            -e SHARD=1 -e SHARDS=1 \\" > "$d/g-dockerenv.yml"
    edit "$SEC" "$QSIGMA_STEP" 'if [ "$SHARDS" = 1 ]; then' sub '          if true; then' > "$d/g-qsigma1.yml"
    edit "$SEC" "$STAGE_STEP" 'path: sigma-shard/' sub '          path: sigma-shardx/' > "$d/g-uppath.yml"
    sed 's/"$RUNNER_TEMP\/sigma\/quick-tree.junit.xml"$/"$RUNNER_TEMP\/sigma\/quick-treex.junit.xml"/' "$SEC" > "$d/g-keepcp.yml"
    edit "$SEC" "$STAGE_STEP" 'TIER: ${{ steps.tier.outputs.tier }}' sub '          TIER: full' > "$d/g-tierfull.yml"
    edit "$SEC" "$STAGE_STEP" 'if-no-files-found: error' sub '          if-no-files-found: ignore' > "$d/g-nofiles.yml"
    edit "$SEC" "$QSIGMA_STEP" 'shell: bash' after '        continue-on-error: true' > "$d/g-qscoe.yml"
    edit "$SEC" "$STAGE_STEP" 'name: sigma-shard-${{ matrix.shard }}' sub '          name: sigma-shard-${{ matrix.shard }}
        continue-on-error: true' > "$d/g-upcoe.yml"
    edit "$WF" "$FANIN_STEP" "awk '!/ name=\"/" sub '              true \' > "$d/g-fanintrue.yml"
    # job-level (the #4717 m18-m21 class): outside every step the rows read
    edit "$WF" '  workspace-test:' 'if: always()' sub '    if: false' > "$d/g-jif.yml"
    edit "$WF" '  workspace-test:' 'needs: [workspace-test-shard]' sub '    needs: []' > "$d/g-jneeds.yml"
    edit "$WF" '  workspace-test:' 'timeout-minutes: 10' after '    env: {SHARDS: "1"}' > "$d/g-jenv.yml"
    edit "$WF" '  workspace-test-shard:' 'timeout-minutes: 180' after '    continue-on-error: true' > "$d/g-jcoe.yml"
    edit "$WF" '  workspace-test-shard:' 'shard: [1, 2, 3]' sub '        shard: [1, 2]' > "$d/g-jmatrix.yml"
    grep -vxF '    - {shard: 3, shards: 3}' "$SEC" > "$d/g-pinsdel.yml"
    for m in treetrue nonotests noexpr libonly dockerenv qsigma1 uppath keepcp tierfull nofiles qscoe upcoe fanintrue jif jneeds jenv jcoe jmatrix pinsdel; do
        case "$m" in fanintrue | j*) wf="$d/g-$m.yml" sec="$SEC" src="$WF" ;; *) wf="$WF" sec="$d/g-$m.yml" src="$SEC" ;; esac
        if cmp -s "$src" "$d/g-$m.yml"; then printf 'FAIL  the golden %s mutant did not apply (its anchor is gone)\n' "$m"; bad=1; continue; fi
        if table "$wf" "$sec" > "$d/out" 2>&1; then printf 'FAIL  the golden %s mutant passed the table\n' "$m"; bad=1
        elif ! grep -q '^FAIL  row 1 .*golden' "$d/out"; then printf 'FAIL  the golden %s mutant was not caught by the golden row\n' "$m"; bad=1
        else printf 'ok    the %-9s mutant is RED against the stale golden: %s\n' "$m" "$(grep -m1 'changed since the golden' "$d/out" | sed 's/^ *//')"; fi
    done
    # the controls: what must stay GREEN, and the change the golden must still see
    st_rc() { local rc=0; "$@" > "$d/out" 2>&1 || rc=$?; echo "$rc"; }
    case_rc() { # case_rc WANT LABEL rc
        if [ "$3" = "$1" ]; then printf 'ok    rc=%s %s\n' "$3" "$2"; else printf 'FAIL  wanted rc=%s, got %s: %s\n' "$1" "$3" "$2"; bad=1; fi
    }
    case_rc 0 "the unchanged tree against its golden" "$(st_rc table "$WF" "$SEC")"
    edit "$WF" '  workspace-test-shard:' 'timeout-minutes: 180' sub '    timeout-minutes: 181' > "$d/b-wf.yml"
    edit "$SEC" "$TREE_STEP" "$tn" after '            # a reviewed comment' > "$d/b-sec.yml"
    case_rc 1 "a pinned job edited, golden stale" "$(st_rc table "$d/b-wf.yml" "$d/b-sec.yml")"
    QUICK_SIGMA_WORKFLOW="$d/b-wf.yml" QUICK_SIGMA_SECTIONS="$d/b-sec.yml" QUICK_SIGMA_GOLDEN="$d/b.sha256" bash "$0" --update-golden > /dev/null 2>&1
    case_rc 0 "the same edit with the golden re-pinned in the same diff" "$(GOLDEN="$d/b.sha256" st_rc table "$d/b-wf.yml" "$d/b-sec.yml")"
    edit "$WF" '  mutants-table-scope:' 'timeout-minutes:' after '    # outside the pinned jobs' > "$d/o-wf.yml"
    edit "$SEC" '  guard-tree:' 'runs-on:' after '    # outside the pinned jobs' > "$d/o-sec.yml"
    if cmp -s "$WF" "$d/o-wf.yml" || cmp -s "$SEC" "$d/o-sec.yml"; then printf 'FAIL  the outside-the-pin control did not apply\n'; bad=1
    else case_rc 0 "an edit outside the pinned jobs, golden unchanged" "$(st_rc table "$d/o-wf.yml" "$d/o-sec.yml")"; fi
    # round 4 (lanes E, F): a re-pinned `  # note` inside a job must not blind the golden to the
    # lines after it; a second job key is ENV; a workflow-level env reaches every pinned job
    edit "$WF" '  workspace-test-shard:' '    steps:' sub '  # note
    steps:' > "$d/c-note.yml"
    edit "$d/c-note.yml" '  workspace-test-shard:' 'fetch-depth: 0' sub '          fetch-depth: 1' > "$d/c-after.yml"
    QUICK_SIGMA_WORKFLOW="$d/c-note.yml" QUICK_SIGMA_SECTIONS="$SEC" QUICK_SIGMA_GOLDEN="$d/c.sha256" bash "$0" --update-golden > /dev/null 2>&1
    if cmp -s "$d/c-note.yml" "$d/c-after.yml" || cmp -s "$WF" "$d/c-note.yml"; then printf 'FAIL  the comment-blind mutant did not apply\n'; bad=1
    else case_rc 1 "an edit below a re-pinned indent-2 comment inside a job" "$(GOLDEN="$d/c.sha256" st_rc table "$d/c-after.yml" "$SEC")"; fi
    edit "$SEC" '  workspace-test:' '  workspace-test:' sub '  workspace-test-shard:
    runs-on: shadow
  workspace-test:' > "$d/c-dup.yml"
    case_rc 2 "a second workspace-test-shard key under jobs (a loader keeps the last)" "$(st_rc table "$WF" "$d/c-dup.yml")"
    edit "$WF" 'jobs:' 'jobs:' sub 'env:
  NEXTEST_PROFILE: bogus
jobs:' > "$d/c-env.yml"
    case_rc 1 "a workflow-level env added above the pinned jobs" "$(st_rc table "$d/c-env.yml" "$SEC")"
    case_rc 2 "no golden file" "$(GOLDEN="$d/absent.sha256" st_rc table "$WF" "$SEC")"
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
