#!/usr/bin/env bash
# check_coverage_has_producers.sh -- every consumer of aprender's coverage number
# still has a producer (#3676).
#
# Operator 2026-09-21: "YES, coverage on tags release only." `ci / coverage` left
# the PR / merge-group / push path: it had gated nothing and measured 0 tests on
# the facade root. A deletion like that can starve a gate that READ the number
# from CI -- rmedia's release gate did exactly that ("could not look"). So the
# chain is asserted, not remembered:
#
#   R1  `make -n coverage-check` reaches llvm-cov -- the PRODUCER the
#       pre-publish dogfood uses measures coverage itself, reads nothing from CI.
#   R2  scripts/dogfood.sh still runs `make ... coverage-check` -- the release
#       CONSUMER exists (without it the chain would be vacuously intact).
#   R3  .github/workflows/coverage-nightly.yml has the pinned `on:` block (schedule and
#       workflow_dispatch, no tag push) and the name `Coverage Nightly`, and runs
#       scripts/coverage_receipt.sh on a live line -- the nightly is the CI producer: it
#       leaves coverage-receipt-<sha>.json for the commit measured.
#   R4  the Makefile defines a numeric COV_FLOOR -- the floor `make coverage` enforces.
#   R5  ci/sections.yml (#4433) hands sovereign-ci `coverage_on: tag` (no `skip_coverage: true`)
#       AND ci.yml has the pinned `on:` block, which has NO tag trigger (T39). `tag` keeps
#       coverage off the PR / queue / branch-push path: sovereign-ci's gate prints "coverage:
#       NOT MEASURED" there. `skip_coverage: true` was the first spelling (#3688): that gate
#       counts a skipped coverage job as a mandatory failure, so it turned `ci / gate` red on
#       every PR. The tag trigger is gone because nothing read the tag run and it could not
#       pass: a tag create has no base sha, and its coverage section built only the facade.
#   R6  scripts/release/tag_coverage_gate.sh, the one file where the release's coverage
#       verdict is made, is byte for byte the reviewed file: its sha256 equals GATE_SHA. That
#       file reads coverage-nightly.yml runs only and, at run time, refuses any run whose
#       workflowName is not `Coverage Nightly`. A keyword pin was bypassable (an unpinned
#       `REPO=`, `ids=` or `jq()` line, a decoy self_test), so every byte is pinned, comments
#       too. Callers of ci.yml runs elsewhere do not matter: the gate never consumes them.
#
# WHY PINS (#4757). R3/R5 used to read `on:` with an awk parser and R6 used a regex over how
# a gh call is spelled. Three review rounds of #4745 each found a new YAML or gh spelling the
# patterns missed. A pin does not enumerate spellings: any change to these blocks or lines is
# RED until a reviewer re-pins it here. The YAML shape is an allow-list too: every top-level
# line must be one of name, run-name, on, permissions, env, defaults, concurrency or jobs,
# unquoted, with exactly one `on:`; a CR, a tab-led line, `---`, a quoted key, `true:`, `<<:`,
# `? key`, a column-0 `- item` or an indented line continuing a top-level value cannot be
# judged and is ENV rc=2, never a pass.
# To re-pin after a reviewed change: for R3/R5 update the PIN block below from the diff the
# FAIL line prints; for R6 set GATE_SHA to the sha256 the FAIL line prints.
# Any link missing is FAIL; a file that cannot be read is ENV rc=2.
#
#   check_coverage_has_producers.sh              judge the tree
#   check_coverage_has_producers.sh --self-test  each rule against fixtures that break it
#
# Test seams: COVCHAIN_ROOT (the tree judged); COVCHAIN_BASELINE=<a guard script> makes
# --self-test run every row against that guard as well and report, not judge, its result.
set -uo pipefail
ROOT="${COVCHAIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# The pins. Full-line comments and blank lines are dropped before the compare; every other
# byte counts.
CI_ON_PIN=$(cat <<'PIN'
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
  merge_group:
  workflow_dispatch:
PIN
)
NIGHTLY_ON_PIN=$(cat <<'PIN'
on:
  schedule:
    - cron: '0 22 * * *'   # 22:00 UTC -> lands ~03:00 UTC
  workflow_dispatch: {}
PIN
)
NIGHTLY_NAME='name: Coverage Nightly'
# GATE_SHA: the sha256 of the whole release gate, #4757. Any byte changed is RED until re-pinned.
GATE_SHA=98a36ea2ff55d3cfe074cca554c7c15c2d38d9777e8112867e6c51594cdd2af5

# TOP_AWK: a workflow -> "N <its name: line>" and "B <a line of its on: block>" records, or one
# "E <why>" record when the file is not in the shape this guard reads (then nothing is judged).
TOP_AWK=$(cat <<'AWK'
/\r/ { why = "a CR at line " NR; exit }
/^[[:space:]]*(#.*)?$/ { next }
/^ / && sc { why = "an indented line " NR " continues a top-level value"; exit }
/^[^ ]/ {
    if ($0 !~ /^(name|run-name|on|permissions|env|defaults|concurrency|jobs):( .*)?$/) { why = "the top-level line " NR " (" $0 ")"; exit }
    inon = ($0 ~ /^on:/); if (inon) n++
    sc = (!inon && $0 ~ /^[^:]*:[[:space:]]*[^[:space:]#]/)   # on: is judged by its pin
    if ($0 ~ /^name:/) { print "N " $0; nm++ }
}
inon { print "B " $0 }
END {
    if (why == "" && n != 1) why = n + 0 " top-level on: keys"
    if (why == "" && nm > 1) why = nm " top-level name: keys"
    if (why != "") print "E " why
}
AWK
)
# on_pinned <workflow> <label> <pin> [<name line>] -> 0 the on: block (and name) are the pinned
# ones, 1 they are not (printed), 2 an ENV line: the file is not in a shape this guard reads.
on_pinned() {
    local out got
    out=$(awk "$TOP_AWK" "$1") || { printf 'ENV   %s could not be read -- cannot judge, not a pass\n' "$2"; return 2; }
    if grep -q "^E " <<< "$out"; then
        printf 'ENV   %s is not in the YAML shape this guard reads (%s) -- cannot judge, not a pass\n' "$2" "$(sed -n 's/^E //p' <<< "$out")"
        return 2
    fi
    got=$(sed -n 's/^B //p' <<< "$out")
    if [ "$got" != "$3" ]; then
        printf 'FAIL  %s: its on: block is not the pinned one (re-pin in this guard only after review):\n' "$2"
        diff <(printf '%s\n' "$3") <(printf '%s\n' "$got") | sed 's/^/        /' | head -n 12
        return 1
    fi
    if [ -n "${4:-}" ] && ! grep -qxF -- "$4" <<< "$(sed -n 's/^N //p' <<< "$out")"; then
        printf 'FAIL  %s: its top-level name is not "%s" -- the release gate reads only runs carrying that name\n' "$2" "${4#name: }"
        return 1
    fi
    return 0
}
# live <file> -> the file with comments stripped (a setting in a comment is not a setting)
live() { sed 's/#.*$//' "$1"; }

# judge <root> -> 0 all links hold, 1 a link broke, 2 ENV
judge() {
    local r=$1 bad=0 env=0 dry ci tg got v3 v5
    local mk="$r/Makefile" df="$r/scripts/dogfood.sh" wf="$r/.github/workflows/coverage-nightly.yml" cy="$r/.github/workflows/ci.yml" sy="$r/ci/sections.yml"
    tg="$r/scripts/release/tag_coverage_gate.sh"
    for f in "$mk" "$df" "$wf" "$cy" "$sy" "$tg"; do [ -r "$f" ] || { printf 'ENV   %s is not readable -- cannot judge, not a pass\n' "$f"; return 2; }; done
    dry=$(make -n -C "$r" coverage-check 2>/dev/null) || dry=""
    if grep -q "llvm-cov" <<< "$dry"; then printf 'ok    R1 make -n coverage-check reaches llvm-cov (the release producer measures, it reads nothing from CI)\n'
    else printf 'FAIL  R1 make -n coverage-check does not reach llvm-cov -- the dogfood coverage gate would measure nothing\n'; bad=1; fi

    # WIDENED (#3844): R2 asks "does the release still CHECK coverage", and it used to
    # test for one INVOCATION SHAPE -- `gate coverage make ... coverage-check`. When the
    # coverage row moved off the `gate` helper to a `mark` row (so a miss could be an
    # owed DEFER carrying its measured percentage rather than a NO-GO, #3839), coverage
    # was still run by the very next line and this rule reported "nothing checks
    # coverage at the release". A guard that names a helper cannot answer a question
    # about a behaviour.
    #
    # The new pattern requires an UNCOMMENTED line that invokes `make ... coverage-check`,
    # whatever wraps it. The `[^#[:space:]]` is load-bearing: it keeps the m_r2_comment
    # mutant RED, which a bare `.*` would have admitted. Case table, measured:
    #     gate coverage make -C … coverage-check          MATCH
    #     cov_out=$(make -C … coverage-check 2>&1); …      MATCH
    #   # gate coverage make -C x coverage-check           no
    #      # cov_out=$(make -C x coverage-check)           no
    if grep -qE '^[[:space:]]*[^#[:space:]].*make[[:space:]].*coverage-check' "$df"; then
        printf 'ok    R2 scripts/dogfood.sh runs make ... coverage-check (the release consumer exists)\n'
    else printf 'FAIL  R2 scripts/dogfood.sh no longer runs coverage-check -- nothing checks coverage at the release\n'; bad=1; fi

    on_pinned "$wf" "R3 coverage-nightly.yml" "$NIGHTLY_ON_PIN" "$NIGHTLY_NAME"; v3=$?
    if [ "$v3" = 2 ]; then
        env=1
    elif [ "$v3" = 1 ]; then
        bad=1
    elif ! live "$wf" | grep -qE '(^|[[:space:]])scripts/coverage_receipt\.sh([[:space:]]|$)'; then
        printf 'FAIL  R3 coverage-nightly.yml no longer runs scripts/coverage_receipt.sh (on a live line) -- the release gate would find no receipt\n'; bad=1
    else printf 'ok    R3 coverage-nightly.yml has the pinned on: (schedule, no tag push) and name, and writes the sha-keyed receipt\n'; fi

    if grep -qE '^COV_FLOOR[[:space:]]*:?=[[:space:]]*[0-9]+' "$mk"; then
        printf 'ok    R4 the Makefile defines a numeric COV_FLOOR\n'
    else printf 'FAIL  R4 the Makefile has no numeric COV_FLOOR -- make coverage enforces nothing\n'; bad=1; fi

    # #4433: the sovereign-ci `with:` block moved to ci/sections.yml (sovereign-ci:);
    # the `on:` trigger stays in ci.yml.
    ci=$(live "$sy")
    if grep -qE '^[[:space:]]+skip_coverage:[[:space:]]*true' <<<"$ci"; then
        printf 'FAIL  R5 ci/sections.yml sets skip_coverage: true -- sovereign-ci'"'"'s gate reads that skip as a mandatory failure (ci / gate red on every PR); use coverage_on: tag\n'; bad=1
    elif ! grep -qE "^[[:space:]]+coverage_on:[[:space:]]*['\"]?tag['\"]?[[:space:]]*$" <<<"$ci"; then
        printf 'FAIL  R5 ci/sections.yml does not hand sovereign-ci coverage_on: tag (on a live line)\n'; bad=1
    elif on_pinned "$cy" "R5 ci.yml" "$CI_ON_PIN"; v5=$?; [ "$v5" = 2 ]; then
        env=1
    elif [ "$v5" = 1 ]; then
        bad=1
    else printf 'ok    R5 ci/sections.yml: coverage_on: tag, and ci.yml has the pinned on: (no tag trigger)\n'; fi

    got=$(sha256sum < "$tg" 2>/dev/null) || got=''
    got=${got%% *}
    if [ -z "$got" ]; then
        printf 'ENV   R6 could not hash scripts/release/tag_coverage_gate.sh -- cannot judge, not a pass\n'; env=1
    elif [ "$got" != "$GATE_SHA" ]; then
        printf 'FAIL  R6 scripts/release/tag_coverage_gate.sh is not the pinned file (sha256 %s, pinned %s) -- re-pin GATE_SHA only after review\n' "$got" "$GATE_SHA"; bad=1
    else printf 'ok    R6 the release gate is the pinned file: it reads only coverage-nightly runs and refuses any other run at run time\n'; fi
    [ "$env" = 1 ] && return 2
    return "$bad"
}

# The header is every leading comment line, found rather than counted.
case "${1:-}" in -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== coverage producer chain: each rule must turn RED when its link breaks ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/covchain.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "${1:?}" ] && { rm -rf -- "${1:?}" || return 1; } ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0; base=0; BL=${COVCHAIN_BASELINE:-}
    NW=.github/workflows/coverage-nightly.yml CY=.github/workflows/ci.yml TG=scripts/release/tag_coverage_gate.sh
    # fixture <dir> -> a tree where every link holds (the pins, as the tree has them)
    fixture() {
        mkdir -p "$1/scripts/release" "$1/.github/workflows" "$1/ci"
        printf 'COV_FLOOR := 88\nLLVMCOV := llvm-cov\ncoverage:\n\t$(LLVMCOV) report --fail-under-lines $(COV_FLOOR)\ncoverage-check: coverage\n' > "$1/Makefile"
        printf '#!/usr/bin/env bash\n  gate coverage make -C "$d" coverage-check\n' > "$1/scripts/dogfood.sh"
        printf '%s\n# the nightly\n%s\n\njobs:\n  c:\n    steps:\n      - run: |\n          bash scripts/coverage_receipt.sh cov.log "$(git rev-parse HEAD)" 89 out since\n' \
            "$NIGHTLY_NAME" "$NIGHTLY_ON_PIN" > "$1/$NW"
        printf 'name: CI\n\n%s\n\njobs: {}\n' "$CI_ON_PIN" > "$1/$CY"
        printf "sovereign-ci:\n  uses: paiml/.github/.github/workflows/sovereign-ci.yml@x\n  with:\n    coverage_on: tag\n" > "$1/ci/sections.yml"
        cp -- "$ROOT/$TG" "$1/$TG"
        printf '#!/usr/bin/env bash\ngh run list --workflow ci.yml --branch main --limit 1\n' > "$1/scripts/state.sh"
    }
    # The fixture text below spells a shell `$` as @, so no line here is shell code.
    at() { printf '%s\n' "$1" | tr '@' '$'; }
    addg() { at "$1" >> "$TG"; }                     # a live line of the gate (after self_test)
    adds() { at "$1" >> scripts/state.sh; }          # a line of a script the gate never runs
    ciyml() { printf 'name: CI\n%b\n  pull_request:\n    branches: [main, master]\njobs: {}\n' "$1" > "$CY"; }
    ciraw() { printf '%b' "$1" > "$CY"; }
    nwraw() { printf '%b' "$1" > "$NW"; }
    # one mutation per row, as a function (no eval: the mutation is code, not a string)
    m_none()          { :; }
    m_r1()            { sed -i 's/\$(LLVMCOV) report.*/echo nothing/' Makefile; }
    m_r2_gone()       { printf '#!/usr/bin/env bash\n' > scripts/dogfood.sh; }
    m_r2_comment()    { printf '#!/usr/bin/env bash\n# gate coverage make -C x coverage-check\n' > scripts/dogfood.sh; }
    m_r3_nosched()    { sed -i '/schedule:/d; /cron:/d' "$NW"; }
    m_r3_tagsback()   { sed -i "s/^  workflow_dispatch: {}/  push:\n    tags: ['v*']/" "$NW"; }
    m_r3_tagsblock()  { sed -i "s/^  workflow_dispatch: {}/  workflow_dispatch: {}\n  push:\n    tags:\n      - 'v*'/" "$NW"; }
    m_r3_namecont()   { sed -i 's/^name: Coverage Nightly$/&\n  Extra/' "$NW"; }
    m_r3_name2()      { printf 'name: Other\n' >> "$NW"; }
    m_r3_noreceipt()  { sed -i '/coverage_receipt/d' "$NW"; }
    m_r3_rcptcomment(){ sed -i 's/^\( *\)bash scripts\/coverage_receipt/\1# bash scripts\/coverage_receipt/' "$NW"; }
    m_r3_indent4()    { nwraw "name: Coverage Nightly\non:\n    schedule:\n        - cron: '0 22 * * *'\n    push:\n        tags: ['v*']\njobs:\n  c:\n    steps:\n      - run: |\n          bash scripts/coverage_receipt.sh cov.log x 89 out since\n"; }
    m_r3_flowmulti()  { sed -i "s/^  workflow_dispatch: {}/  workflow_dispatch: {}\n  push: {tags: [v1,\n    v2]}/" "$NW"; }
    m_r3_unbal()      { sed -i "s/^  workflow_dispatch: {}/  workflow_dispatch: {inputs: {a: {type: string},\n    b: {type: string}}}/" "$NW"; }
    m_r3_rename()     { sed -i 's/^name: Coverage Nightly$/name: Coverage/' "$NW"; }
    m_r3_cmtok()      { sed -i 's/^  schedule:$/  # the nightly fires on its schedule only\n  schedule:/' "$NW"; }
    m_r3_crlf()       { sed -i 's/$/\r/' "$NW"; }
    m_r3_quoted()     { sed -i 's/^on:$/"on":/' "$NW"; }
    m_r3_dupon()      { printf 'on:\n  push:\n    tags: [v1]\n' >> "$NW"; }
    m_r4()            { sed -i '/^COV_FLOOR/d' Makefile; }
    m_r5_skip()       { sed -i 's/coverage_on: tag/skip_coverage: true/' ci/sections.yml; }
    m_r5_both()       { printf '    skip_coverage: true\n' >> ci/sections.yml; }
    m_r5_nocov()      { sed -i '/coverage_on:/d' ci/sections.yml; }
    m_r5_comment()    { sed -i 's/    coverage_on: tag/    # coverage_on: tag/' ci/sections.yml; }
    m_r5_always()     { sed -i 's/coverage_on: tag/coverage_on: always/' ci/sections.yml; }
    m_r5_tagsback()   { sed -i '0,/^    branches: \[main, master\]$/s//&\n    tags: ["v*"]/' "$CY"; }
    m_r5_tagsblock()  { ciyml 'on:\n  push:\n    branches: [main]\n    tags:\n      - "v[0-9]*"'; }
    m_r5_barepush()   { ciyml 'on:\n  push:'; }
    m_r5_inline()     { ciraw 'name: CI\non: [push, pull_request]\njobs: {}\n'; }
    m_r5_tagcomment() { sed -i '0,/^    branches: \[main, master\]$/s//&\n    # tags: ["v*"]/' "$CY"; }
    m_r5_blank()      { sed -i 's/^  merge_group:$/\n  merge_group:/' "$CY"; }
    m_r5_brignore()   { ciyml 'on:\n  push:\n    branches-ignore: [gh-pages]'; }
    m_r5_nopush()     { ciyml 'on:\n  merge_group:'; }
    m_r5_indent4()    { ciraw 'name: CI\non:\n    push:\n        branches: [main]\n        tags: ["v*"]\n    pull_request: {}\njobs: {}\n'; }
    m_r5_quoted()     { ciraw '"on":\n  "push":\n    tags: ["v*"]\njobs: {}\n'; }
    m_r5_squoted()    { ciraw "'on':\n  push:\n    tags: ['v*']\njobs: {}\n"; }
    m_r5_dashpush()   { ciraw 'on:\n  - push\n  - pull_request\njobs: {}\n'; }
    m_r5_flowmulti()  { ciraw 'on: {\n  push: {tags: [v1]}\n}\njobs: {}\n'; }
    m_r5_listmulti()  { ciraw 'on: [pull_request,\n  push]\njobs: {}\n'; }
    m_r5_flowbr()     { ciyml 'on:\n  push: {branches: [main]}'; }
    m_r5_crlf()       { sed -i 's/$/\r/' "$CY"; }
    m_r5_crone()      { sed -i 's/^  merge_group:$/  merge_group:\r/' "$CY"; }
    m_r5_truekey()    { ciraw 'true:\n  push:\n    tags: ["v*"]\njobs: {}\n'; }
    m_r5_dash0()      { ciraw 'on:\n- pull_request\n- push\njobs: {}\n'; }
    m_r5_create()     { sed -i 's/^  workflow_dispatch:$/&\n  create:/' "$CY"; }
    m_r5_createlist() { ciraw 'on: [pull_request, create]\njobs: {}\n'; }
    m_r5_createword() { ciraw 'on: create\njobs: {}\n'; }
    m_r5_createdash() { ciraw 'on:\n  - pull_request\n  - create\njobs: {}\n'; }
    m_r5_dashmap()    { ciraw 'on:\n  - push:\n      tags: [v1]\njobs: {}\n'; }
    m_r5_dashcreate() { ciraw 'on:\n  - create:\njobs: {}\n'; }
    m_r5_docstart()   { sed -i '1i ---' "$CY"; }
    m_r5_merge()      { printf '<<: {on: [push]}\n' >> "$CY"; }
    m_r5_qkey()       { printf '? on\n: [push]\n' >> "$CY"; }
    m_r5_tab()        { sed -i 's/^  merge_group:$/\tmerge_group:/' "$CY"; }
    m_r5_dupon()      { printf 'on: [push]\n' >> "$CY"; }
    m_r5_noon()       { sed -i '/^on:$/d; /^  /d' "$CY"; }
    m_r5_allindent()  { sed -i 's/^/  /' "$CY"; }
    m_r5_anchor()     { sed -i 's/^on:$/on: \&a/' "$CY"; }
    m_r5_trail()      { sed -i 's/^  merge_group:$/  merge_group: /' "$CY"; }
    m_r5_onkey()      { sed -i 's/^on:$/on :/' "$CY"; }
    m_r6_wf()         { sed -i "s/^WF=.*/WF='ci.yml'/" "$TG"; }
    m_r6_wfcomment()  { sed -i 's/^WF=/# WF=/' "$TG"; }
    m_r6_wf2()        { addg "WF='ci.yml'"; }
    m_r6_wfeq()       { addg "WF2='ci.yml'"; addg '    "@GH" run list --workflow="@WF2" --branch main'; }
    m_r6_reader()     { addg '    "@GH" run list --repo "@REPO" --workflow ci.yml --event push --branch "@1" --limit 20'; }
    m_r6_readerci()   { addg 'gh run list --workflow=CI --branch=@T'; }
    m_r6_swapped()    { addg 'gh run list --branch "@T" --workflow ci.yml'; }
    m_r6_short()      { addg 'gh run list -w ci.yml -b "@T"'; }
    m_r6_path()       { addg 'gh run list --workflow .github/workflows/ci.yml --branch "@T"'; }
    m_r6_api()        { addg 'gh api "repos/x/y/actions/workflows/ci.yml/runs?branch=@T"'; }
    m_r6_contd()      { addg "X=@(true \\"; addg '    --workflow ci.yml --branch "@T")'; }
    m_r6_vprefix()    { addg 'gh run list --workflow ci.yml --branch "v@{VER}"'; }
    m_r6_refstag()    { addg 'gh run list -w ci.yml -b "refs/tags/@TAG"'; }
    m_r6_eqprefix()   { addg 'gh run list --workflow=ci.yml --branch=v@VER'; }
    m_r6_apiprefix()  { addg 'gh api "repos/x/y/actions/workflows/ci.yml/runs?branch=v@VER"'; }
    m_r6_mainlit()    { addg 'gh run list --branch main --workflow ci.yml'; }
    m_r6_curl()       { addg 'curl -s "https://api.github.com/repos/x/y/actions/runs?branch=v1"'; }
    m_r6_eval()       { addg 'eval "@CMD"'; }
    m_r6_source()     { addg '. scripts/lib/runs.sh'; }
    m_r6_nightlyok()  { addg '    "@GH" run view "@id" --repo "@REPO" --json conclusion'; }
    m_r6_comment()    { addg "# gh run list --workflow ci.yml --branch \"@T\" (a comment runs nothing)"; }
    m_r6_sourceword() { addg "source scripts/lib/runs.sh"; }
    m_r6_exec()       { addg "exec bash scripts/lib/runs.sh"; }
    m_r6_wget()       { addg "wget -qO- https://api.github.com/repos/x/y/actions/runs"; }
    m_r6_ciyml()      { addg "F=ci.yml"; }
    m_r6_contd2()     { addg "R=@(\"@G\" run list \\"; addg "    --workflow \"@W\" --branch \"@T\")"; }
    m_r6_decoy()      { sed -i '0,/^floor_at() {$/s//floor_at() {\nself_test() { :; }\n    "$GH" run list --workflow ci.yml --branch main/' "$TG"; }
    m_r6_repo()       { sed -i 's/^REPO=.*/REPO=other\/fork/' "$TG"; }
    m_r6_ids()        { sed -i 's/^    ids=.*/    ids="99999 $1"/' "$TG"; }
    m_r6_jqshadow()   { addg 'jq() { return 0; }'; }
    m_r6_norun()      { sed -i '/workflowName == env.WFN/d' "$TG"; }
    m_r6_nojson()     { sed -i 's/,workflowName 2>/ 2>/' "$TG"; }
    m_r6_inselftest() { sed -i 's/^}$/    gh api "repos\/x\/y\/actions\/workflows\/ci.yml\/runs?branch=v1"\n}/' "$TG"; }
    m_r6_elsewhere()  { adds 'gh run list --workflow ci.yml --branch "@T"'; adds 'gh run list -w ci.yml -b "refs/tags/@TAG"'
                        adds 'gh api "repos/x/y/actions/workflows/ci.yml/runs?branch=v@VER"'
                        printf 'name: other\non:\n  workflow_dispatch:\njobs:\n  a:\n    steps:\n      - run: gh run list -w ci.yml -b "%sT"\n' '$' > .github/workflows/other.yml; }
    m_missing()       { rm -f "$NW"; }
    m_missing_ci()    { rm -f "$CY"; }
    m_missing_sy()    { rm -f ci/sections.yml; }
    m_missing_tg()    { rm -f "$TG"; }
    row() { # row WANT-RC LABEL MUTATION-FUNCTION
        local want=$1 label=$2 rc=0 brc=0 bcol=''; n=$((n + 1)); rmtree "$d/t"; fixture "$d/t"
        ( cd "$d/t" && "$3" ) || { printf 'FAIL  row %s fixture mutation failed: %s\n' "$n" "$label"; bad=1; return; }
        judge "$d/t" > "$d/out" 2>&1 || rc=$?
        if [ -n "$BL" ]; then
            COVCHAIN_ROOT="$d/t" bash "$BL" > /dev/null 2>&1; brc=$?
            if [ "$brc" = "$want" ]; then bcol="base=$brc  "; base=$((base + 1)); else bcol="base=$brc! "; fi
        fi
        if [ "$rc" = "$want" ]; then printf 'ok    row %-3s rc=%s  %s%s\n' "$n" "$rc" "$bcol" "$label"
        else printf 'FAIL  row %-3s rc=%s (wanted %s)  %s%s\n' "$n" "$rc" "$want" "$bcol" "$label"; sed 's/^/        /' "$d/out"; bad=1; fi
    }
    row 0 "every link holds -> PASS"                                            m_none
    row 1 "R1: coverage no longer runs llvm-cov -> RED"                         m_r1
    row 1 "R2: dogfood stops calling coverage-check -> RED"                     m_r2_gone
    row 1 "R2: the call only in a COMMENT -> RED"                               m_r2_comment
    row 1 "R3: nightly schedule removed -> RED"                                 m_r3_nosched
    row 1 "R3: the nightly fires on v* tags (flow list) -> RED"                 m_r3_tagsback
    row 1 "R3: the nightly fires on tags (block list) -> RED"                   m_r3_tagsblock
    row 1 "R3: the nightly no longer writes the receipt -> RED"                 m_r3_noreceipt
    row 1 "R3: the receipt call only in a COMMENT -> RED"                       m_r3_rcptcomment
    row 1 "R3 r1: the nightly's tag trigger at a 4-space indent -> RED"         m_r3_indent4
    row 1 "R3 r3: the nightly's on: has a multi-line flow push value -> RED"    m_r3_flowmulti
    row 1 "R3 r3: the nightly's on: has a multi-line flow non-push value -> RED" m_r3_unbal
    row 1 "R3: the nightly renamed (the gate reads runs by that name) -> RED"   m_r3_rename
    row 2 "R3 r4: the nightly's name continued on an indented line -> ENV rc=2" m_r3_namecont
    row 2 "R3 r5: a second top-level name: in the nightly -> ENV rc=2"         m_r3_name2
    row 0 "R3: a comment line added inside the nightly's on: -> PASS"           m_r3_cmtok
    row 2 "R3 r3: CRLF line ends on the nightly are ENV rc=2, never a pass"     m_r3_crlf
    row 2 "R3 r3: a quoted \"on\" key on the nightly is ENV rc=2"                m_r3_quoted
    row 2 "R3: a second on: key on the nightly is ENV rc=2"                     m_r3_dupon
    row 2 "coverage-nightly.yml missing is ENV rc=2, never a pass"              m_missing
    row 1 "R4: COV_FLOOR removed -> RED"                                        m_r4
    row 1 "R5: skip_coverage: true instead of coverage_on (the #3688 red) -> RED" m_r5_skip
    row 1 "R5: skip_coverage: true alongside coverage_on -> RED"                m_r5_both
    row 1 "R5: coverage_on removed -> RED"                                      m_r5_nocov
    row 1 "R5: coverage_on only in a COMMENT -> RED"                            m_r5_comment
    row 1 "R5: coverage_on: always (coverage back on every PR) -> RED"          m_r5_always
    row 1 "R5: ci.yml's v* tag trigger back (flow list) -> RED"                 m_r5_tagsback
    row 1 "R5: ci.yml tag trigger as a block list -> RED"                       m_r5_tagsblock
    row 1 "R5: ci.yml bare push: (fires on tags too) -> RED"                    m_r5_barepush
    row 1 "R5: ci.yml on: [push, pull_request] (inline) -> RED"                 m_r5_inline
    row 0 "R5: ci.yml tag trigger only in a COMMENT -> PASS (not a trigger)"    m_r5_tagcomment
    row 0 "R5: a blank line inside ci.yml's on: -> PASS"                        m_r5_blank
    row 1 "R5: ci.yml push: branches-ignore only (not the pinned block) -> RED" m_r5_brignore
    row 1 "R5: ci.yml with no push trigger (not the pinned block) -> RED"       m_r5_nopush
    row 1 "R5 r1: tag trigger at a 4-space indent -> RED"                       m_r5_indent4
    row 2 "R5 r1: quoted \"on\"/\"push\" keys with tags -> ENV rc=2"              m_r5_quoted
    row 2 "R5: a single-quoted 'on' key -> ENV rc=2"                            m_r5_squoted
    row 1 "R5 r1: on: as a block list with - push -> RED"                       m_r5_dashpush
    row 2 "R5 r1: a multi-line flow on: { ... } -> ENV rc=2 (its } is a top-level line)" m_r5_flowmulti
    row 1 "R5 r1: a multi-line inline on: [ ... ] -> RED"                       m_r5_listmulti
    row 1 "R5: push: {branches: [main]} one-line flow (not the pinned block) -> RED" m_r5_flowbr
    row 2 "R5 r1: CRLF line ends with a tag trigger -> ENV rc=2"                m_r5_crlf
    row 2 "R5 r3: one CR inside the on: block -> ENV rc=2 (the strip is gone, the CR is refused)" m_r5_crone
    row 2 "R5 r1: the key spelled true: (YAML 1.1 on) -> ENV rc=2"              m_r5_truekey
    row 2 "R5 r2: on: as a zero-indent block list (- push at column 0) -> ENV rc=2" m_r5_dash0
    row 1 "R5 r2: a create: event (fires when a tag is pushed) -> RED"          m_r5_create
    row 1 "R5 r2: on: [pull_request, create] -> RED"                            m_r5_createlist
    row 1 "R5 r2: on: create -> RED"                                            m_r5_createword
    row 1 "R5 r2: on: as a block list with - create -> RED"                     m_r5_createdash
    row 1 "R5 r3: - push: with a map under a list item (GitHub-invalid) -> RED" m_r5_dashmap
    row 1 "R5 r3: - create: as a list item map (GitHub-invalid) -> RED"         m_r5_dashcreate
    row 2 "R5: a --- document marker -> ENV rc=2"                               m_r5_docstart
    row 2 "R5: a <<: merge key at the top -> ENV rc=2"                          m_r5_merge
    row 2 "R5: a ? on explicit key -> ENV rc=2"                                 m_r5_qkey
    row 2 "R5: a tab-indented line -> ENV rc=2"                                 m_r5_tab
    row 2 "R5: a second on: key -> ENV rc=2"                                    m_r5_dupon
    row 2 "R5: no on: key at all -> ENV rc=2"                                   m_r5_noon
    row 2 "R5: the whole file indented (no top-level on:) -> ENV rc=2"          m_r5_allindent
    row 1 "R5: an anchor on the on: line -> RED"                                m_r5_anchor
    row 1 "R5: a trailing space inside the on: block -> RED (every byte is pinned)" m_r5_trail
    row 2 "R5: on : with a space before the colon -> ENV rc=2"                  m_r5_onkey
    row 2 "ci.yml missing is ENV rc=2, never a pass"                            m_missing_ci
    row 2 "ci/sections.yml missing is ENV rc=2, never a pass"                   m_missing_sy
    row 1 "R6: the release gate reads ci.yml, not the nightly -> RED"           m_r6_wf
    row 1 "R6: the gate's WF= only in a COMMENT -> RED"                         m_r6_wfcomment
    row 1 "R6 r1: a later WF='ci.yml' overrides the nightly -> RED"             m_r6_wf2
    row 1 "R6: --workflow= a second variable that holds ci.yml -> RED"               m_r6_wfeq
    row 1 "R6: the gate lists ci.yml runs by a variable branch (the old main lookup) -> RED" m_r6_reader
    row 1 "R6: the same lookup as --workflow=CI, branch in a variable -> RED"   m_r6_readerci
    row 1 "R6 r1: flag order swapped (--branch VAR --workflow ci.yml) -> RED"   m_r6_swapped
    row 1 "R6 r1: short flags -w ci.yml -b VAR -> RED"                          m_r6_short
    row 1 "R6 r1: --workflow .github/workflows/ci.yml by path -> RED"           m_r6_path
    row 1 "R6 r1: gh api workflows/ci.yml/runs?branch=VAR -> RED"               m_r6_api
    row 1 "R6 r2: the lookup split over a backslash continuation -> RED"        m_r6_contd
    row 1 "R6 r3: --branch v + a braced variable -> RED"                                  m_r6_vprefix
    row 1 "R6 r3: -b refs/tags/ + a variable -> RED"                                     m_r6_refstag
    row 1 "R6 r3: --branch=v + a variable -> RED"                                        m_r6_eqprefix
    row 1 "R6 r3: gh api ci.yml runs?branch=v + a variable -> RED"                       m_r6_apiprefix
    row 1 "R6: even a literal --branch main lookup of ci.yml in the gate -> RED" m_r6_mainlit
    row 1 "R6: a curl to the actions API in the gate -> RED"                    m_r6_curl
    row 1 "R6: an eval in the gate -> RED"                                      m_r6_eval
    row 1 "R6: the gate sources another file -> RED"                            m_r6_source
    row 1 "R6: a new gh call, even a harmless one, is not pinned -> RED"        m_r6_nightlyok
    row 1 "R6: even a COMMENT added to the gate -> RED (every byte is pinned)"   m_r6_comment
    row 1 "R6: the gate runs source on another file -> RED"                  m_r6_sourceword
    row 1 "R6: the gate execs another script -> RED"                         m_r6_exec
    row 1 "R6: a wget to the actions API in the gate -> RED"                 m_r6_wget
    row 1 "R6: a bare ci.yml name in a live gate line -> RED"                 m_r6_ciyml
    row 1 "R6: a continued lookup whose only GitHub word is --workflow -> RED" m_r6_contd2
    row 1 "R6: the run-time workflowName check removed -> RED"                  m_r6_norun
    row 1 "R6: workflowName no longer fetched -> RED"                           m_r6_nojson
    row 1 "R6: a ci.yml lookup inside the gate's self_test -> RED (every byte is pinned)" m_r6_inselftest
    row 1 "R6 r4: a decoy self_test() { :; } inside a function, then a ci.yml lookup -> RED" m_r6_decoy
    row 1 "R6 r4: REPO= points at another repo (a line with no GitHub word) -> RED" m_r6_repo
    row 1 "R6 r4: the gate picks its own run id (the ids= line) -> RED"        m_r6_ids
    row 1 "R6 r4: a jq() shadow that turns every check true -> RED"           m_r6_jqshadow
    row 0 "R6: ci.yml lookups in other scripts and workflows -> PASS (the gate never reads them)" m_r6_elsewhere
    row 2 "scripts/release/tag_coverage_gate.sh missing is ENV rc=2, never a pass" m_missing_tg
    [ -n "$BL" ] && printf 'BASELINE: %s of %s rows give the wanted rc under %s\n' "$base" "$n" "$BL"
    [ "$bad" = 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== every consumer of aprender's coverage number has a producer (check_coverage_has_producers.sh) ==="
judge "$ROOT"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
