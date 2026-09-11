#!/usr/bin/env bash
# ci_test_tier.sh — decide which test tier a CI run owes (BSE-17, PMAT-1077).
#
#   quick  pull_request: the touched crates + direct reverse dependents
#          (scripts/gate_touched_crates.sh, cap -> full) PLUS every test target
#          that reads the tree (scripts/tree_reader_tests.txt, derived and
#          checked by scripts/check_tree_reader_tests.sh — drift is ENV, exit 2,
#          never a quick tier over a stale registry).
#   full   push, schedule, workflow_dispatch; pull_request when the selection
#          falls closed (root manifests touched, or over the cap); merge_group
#          when the queue ref's tree is not the tree a green PR head already ran.
#   reuse  merge_group only: HEAD^{tree} equals the PR head's tree AND that
#          head's workspace-test check-run concluded success — the same tree
#          measured twice is the definition of waste. Any doubt -> full.
#
# Output: KEY=VALUE lines — tier, crates (space list), targets (crate:--lib or
# crate:--test:name, space list), reason, cite (the PR head sha on reuse).
# Feature-gated suites (model-tests, setfit, ...) belong to the full tier only;
# the quick tier runs default features. Exit 2 on ENV (unknown event, registry
# drift); 0 otherwise. `--self-test` runs the case table.
#
#   --tier-of-record [--tsv F]   print the 80/20 tier of record (default
#          evidence/fleet/test-tier.tsv, spec §6.2/§6.3) as ONE nextest filterset
#          plus tier_of_record_{tests,modules,packages,seconds}. A `tier=pr` lib
#          module maps to `package(=C) & kind(lib) & test(/^M::/)`, an integration
#          binary (column `kind`=test) to `package(=C) & binary(=M)`; rows are
#          grouped per package. Exit 1 — never a silent full run and never a
#          silent empty set — when the table is missing, headerless, malformed,
#          has zero pr rows, or has a pr row with an empty crate/module.
#   --union-touched              with --tier-of-record and --event pull_request
#          --comparand REF (or --diff-from FILE): the tier-of-record filterset OR
#          the quick tier's touched crates, so a PR always runs its own crates'
#          full lib tests plus the cross-tree 20 %. When the quick tier falls
#          closed the FULL suite runs and no filterset is emitted.
set -euo pipefail

EVENT=""; COMPARAND=""; DIFF_FROM=""; PR_HEAD=""; PR_CONCLUSION=""; REGISTRY="scripts/tree_reader_tests.txt"; ROOT="."
TSV="evidence/fleet/test-tier.tsv"; TIER_OF_RECORD=0; UNION=0
while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT=$2; shift 2 ;;
        --comparand) COMPARAND=$2; shift 2 ;;
        --diff-from) DIFF_FROM=$2; shift 2 ;;
        --pr-head) PR_HEAD=$2; shift 2 ;;
        --pr-head-conclusion) PR_CONCLUSION=$2; shift 2 ;;
        --registry) REGISTRY=$2; shift 2 ;;
        --repo-root) ROOT=$2; shift 2 ;;
        --tsv) TSV=$2; shift 2 ;;
        --tier-of-record) TIER_OF_RECORD=1; shift ;;
        --union-touched) UNION=1; shift ;;
        --self-test) SELF_TEST=1; shift ;;
        *) printf 'usage: %s --event EVENT [--comparand REF] [--diff-from FILE] [--pr-head SHA --pr-head-conclusion C] [--registry FILE] [--repo-root DIR] | --tier-of-record [--tsv FILE] [--union-touched --event pull_request --comparand REF] | --self-test\n' "$0" >&2; exit 2 ;;
    esac
done

targets_from_registry() { # -> space list crate:--lib | crate:--test:name
    grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | awk -F"\t" '{ if ($2=="--test") printf "%s:--test:%s ", $1, $3; else printf "%s:%s ", $1, $2 }' | sed 's/ $//'
}

decide() {
    case "$EVENT" in
        push|schedule|workflow_dispatch)
            printf 'tier=full\nreason=%s event: the whole workspace, every feature-gated suite\n' "$EVENT" ;;
        merge_group)
            if [ -z "$PR_HEAD" ]; then printf 'tier=full\nreason=merge_group without a PR head to compare against\n'; return 0; fi
            local ht pt
            ht=$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || true)
            pt=$(git -C "$ROOT" rev-parse "${PR_HEAD}^{tree}" 2>/dev/null || true)
            if [ -z "$ht" ] || [ -z "$pt" ]; then printf 'tier=full\nreason=merge_group: a tree could not be resolved (HEAD=%s pr-head=%s)\n' "${ht:-?}" "${pt:-?}"; return 0; fi
            if [ "$ht" != "$pt" ]; then printf 'tier=full\nreason=merge_group: queue tree %s differs from PR head tree %s (main moved under the PR)\n' "${ht:0:9}" "${pt:0:9}"; return 0; fi
            if [ "$PR_CONCLUSION" != "success" ]; then printf 'tier=full\nreason=merge_group: same tree but the PR head'"'"'s workspace-test concluded %s, not success\n' "${PR_CONCLUSION:-unknown}"; return 0; fi
            printf 'tier=reuse\ncite=%s\nreason=merge_group: HEAD^{tree} %s equals PR head %s^{tree}, whose workspace-test succeeded — the same tree measured twice\n' "$PR_HEAD" "${ht:0:9}" "${PR_HEAD:0:9}" ;;
        pull_request)
            local chk sel crates rule
            if ! chk=$(bash "$ROOT/scripts/check_tree_reader_tests.sh" 2>&1); then printf 'ENV: %s\n' "$chk" >&2; return 2; fi
            sel=$(bash "$ROOT/scripts/gate_touched_crates.sh" --print-selection ${COMPARAND:+--comparand "$COMPARAND"} ${DIFF_FROM:+--diff-from "$DIFF_FROM"} 2>/dev/null | tail -1)
            crates=$(printf '%s' "$sel" | sed -n 's/^selection=[a-z]* crates=\(.*\) rule=.*$/\1/p'); rule=${sel#*rule=}
            case "$sel" in
                selection=full*) printf 'tier=full\nreason=pull_request: %s\n' "$rule" ;;
                selection=quick*|selection=none*) printf 'tier=quick\ncrates=%s\ntargets=%s\nreason=pull_request: %s; plus %s tree-reader target(s) from %s\n' "$crates" "$(targets_from_registry "$ROOT/$REGISTRY")" "$rule" "$(grep -vc '^#' "$ROOT/$REGISTRY")" "$REGISTRY" ;;
                *) printf 'ENV: gate_touched_crates --print-selection gave "%s"\n' "$sel" >&2; return 2 ;;
            esac ;;
        *) printf 'ENV: unknown event "%s" — refusing to guess a tier\n' "$EVENT" >&2; return 2 ;;
    esac
}

tier_of_record() { # -> filterset= + tier_of_record_* KEY=VALUE lines; 1 on an unusable table
    local out rc=0
    out=$(python3 "$ROOT/scripts/lib/test_tier.py" filterset --tsv "$TSV" 2>&1) || rc=$?
    if [ "$rc" != 0 ]; then printf 'DATA: %s\n' "$out" >&2; return 1; fi
    printf '%s\n' "$out"
}

union_touched() { # tier of record OR the touched crates; 1 on an unusable table, 2 on ENV
    local tor dec crates expr pkgs
    tor=$(tier_of_record) || return 1
    dec=$(decide) || return $?
    printf '%s\n' "$dec"
    if printf '%s\n' "$dec" | grep -qx 'tier=full'; then
        printf 'union_touched_crates=\n'
        printf '%s\n' "$tor" | grep -v '^filterset='
        return 0
    fi
    crates=$(printf '%s\n' "$dec" | sed -n 's/^crates=//p')
    expr=$(printf '%s\n' "$tor" | sed -n 's/^filterset=//p')
    pkgs=$(printf '%s' "$crates" | awk '{ for (i = 1; i <= NF; i++) printf "%spackage(=%s)", (i > 1 ? " | " : ""), $i }')
    case "$crates" in
        "") printf 'filterset=%s\n' "$expr" ;;
        *)  printf 'filterset=%s | (%s)\n' "$expr" "$pkgs" ;;
    esac
    printf 'union_touched_crates=%s\n' "$crates"
    printf '%s\n' "$tor" | grep -v '^filterset='
}

self_test() {
    local td n=0 red=0 out rc leaf
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-tier.XXXXXX"); trap 'rm -rf "${td:?}"' RETURN
    row() { local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1)); rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; printf '%s\n' "$out" | sed 's/^/        /'; red=1; fi; }
    T=$0
    row 0 "push -> full" '^tier=full' bash "$T" --event push
    row 0 "schedule -> full" '^tier=full' bash "$T" --event schedule
    row 2 "unknown event -> ENV (exit 2), never a guess" 'refusing to guess' bash "$T" --event release
    printf 'scripts/foo.sh\n' > "$td/d-scripts.txt"
    row 0 "pull_request, scripts-only diff -> quick with NO crates" '^crates=$' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...and the tree-reader targets are in it (readme_contract, the reader that bit #3039)" 'aprender-core:--test:readme_contract' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...including the lib target whose unit test reads a baseline (aprender-contracts)" 'aprender-contracts:--lib' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    printf 'Cargo.toml\n' > "$td/d-root.txt"
    row 0 "pull_request, root Cargo.toml touched -> full (fail closed)" '^tier=full' bash "$T" --event pull_request --diff-from "$td/d-root.txt"
    printf 'crates/aprender-core/src/lib.rs\n' > "$td/d-core.txt"
    row 0 "pull_request, aprender-core touched -> full (reverse dependents exceed the cap)" 'tier=full' bash "$T" --event pull_request --diff-from "$td/d-core.txt"
    leaf=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '[.packages[]|select(.manifest_path|test("/crates/"))] | (map(.name) - (map(.dependencies[]?.name)|unique)) | sort | .[0]')
    printf 'crates/%s/src/lib.rs\n' "$leaf" > "$td/d-leaf.txt"
    row 0 "pull_request, a leaf crate ($leaf) touched -> quick with that crate" "^crates=.*$leaf" bash "$T" --event pull_request --diff-from "$td/d-leaf.txt"
    # registry drift is ENV for the quick tier
    cp scripts/tree_reader_tests.txt "$td/reg.bak"; printf 'zeta\t--lib\n' >> scripts/tree_reader_tests.txt
    row 2 "pull_request with a drifted registry -> ENV (exit 2): no quick tier over a stale list" 'drifted' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    cp "$td/reg.bak" scripts/tree_reader_tests.txt
    # merge_group rows on a throwaway repo: same tree (empty commit) vs different tree
    git init -q "$td/repo"; ( cd "$td/repo" && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m base && printf 'a\n' > f && git add f && git -c user.name=t -c user.email=t@t commit -q -m one && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m "queue merge, same tree" )
    same=$(git -C "$td/repo" rev-parse HEAD~1)
    row 0 "merge_group, same tree + PR head workspace-test success -> reuse, citing the head" "^cite=$same" bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion success
    row 0 "merge_group, same tree but PR head conclusion failure -> full" 'concluded failure' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion failure
    diff1=$(git -C "$td/repo" rev-parse HEAD~2)
    row 0 "merge_group, different tree -> full (main moved under the PR)" 'differs from PR head tree' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$diff1" --pr-head-conclusion success
    row 0 "merge_group without a PR head -> full" 'without a PR head' bash "$T" --event merge_group --repo-root "$td/repo"
    # MUTANT: a copy that drops the tree-reader targets from the quick tier must lose readme_contract — the falsifier discriminates
    sed 's/targets=%s\\n/targets=\\n/; s/"\$(targets_from_registry "\$ROOT\/\$REGISTRY")" //' "$T" > "$td/mutant.sh"
    row 0 "mutant without tree-reader targets loses readme_contract (proves the inclusion is load-bearing)" 'MUTANT-LOST' bash -c "if bash '$td/mutant.sh' --event pull_request --diff-from '$td/d-scripts.txt' | grep -q readme_contract; then echo MUTANT-KEPT; else echo MUTANT-LOST; fi"
    # --- PMAT-3119: the tier of record as a filterset. Hermetic: committed fixture + temp copies, no cargo.
    local FX GOLD tor_expr realn
    FX="tests/fixtures/test_tier/tier-small.tsv"; GOLD="tests/fixtures/test_tier/tier-small.filterset.txt"
    contains_all() { # FILE PAT... -> ALL-PRESENT, or the first missing pattern and rc 1
        local f=$1 pat
        shift
        for pat in "$@"; do
            if ! grep -qF -- "$pat" "$f"; then printf 'MISSING %s\n' "$pat"; return 1; fi
        done
        printf 'ALL-PRESENT\n'
    }
    lacks() { if grep -qF -- "$2" "$1"; then printf 'STILL-PRESENT %s\n' "$2"; return 1; fi; printf 'ABSENT\n'; }
    full_ok() { if ! grep -qx 'tier=full' "$1"; then printf 'NOT-FULL\n'; return 1; fi
        if grep -q '^filterset=' "$1"; then printf 'FULL-WITH-A-FILTERSET\n'; return 1; fi; printf 'FULL-NO-FILTERSET\n'; }
    tor_expr=$(sed 's/^filterset=//' "$GOLD")
    bash "$T" --tier-of-record --tsv "$FX" > "$td/fx.out" 2>&1 || true
    grep '^filterset=' "$td/fx.out" > "$td/fx-fs.txt" || true
    row 0 "fixture TSV -> the committed GOLDEN filterset, textually (2 lib modules grouped, 1 in another crate, 1 binary)" '^$' diff "$GOLD" "$td/fx-fs.txt"
    row 0 "  ...tier_of_record_tests=8 (the nightly/full rows' 11 tests excluded)" '^tier_of_record_tests=8$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_modules=4" '^tier_of_record_modules=4$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_packages=3" '^tier_of_record_packages=3$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_seconds=10.75 at 2dp (not the table's 152.75)" '^tier_of_record_seconds=10\.75$' cat "$td/fx.out"
    # MUTATION: drop one pr row -> a DIFFERENT filterset, missing exactly that module
    grep -v '^crateA::guard' "$FX" > "$td/mut.tsv"
    bash "$T" --tier-of-record --tsv "$td/mut.tsv" > "$td/mut.out" 2>&1 || true
    grep '^filterset=' "$td/mut.out" > "$td/mut-fs.txt" || true
    row 1 "MUTATION: one pr row deleted -> the filterset DIFFERS from the golden (every row is load-bearing)" '^<' diff "$GOLD" "$td/mut-fs.txt"
    row 0 "  ...the deleted module is absent from it" '^ABSENT$' lacks "$td/mut-fs.txt" "guard"
    row 0 "  ...and the other three atoms survive" '^ALL-PRESENT$' contains_all "$td/mut-fs.txt" "dense" "test(/^(light)::/)" "binary(=readme_contract)"
    row 0 "  ...with modules=3" '^tier_of_record_modules=3$' cat "$td/mut.out"
    row 0 "  ...and tests=6 (the deleted row's 2 tests are gone from the budget too)" '^tier_of_record_tests=6$' cat "$td/mut.out"
    # exit-1 contract: never a silent full run, never a silent empty filterset
    row 1 "missing TSV -> exit 1" 'not readable' bash "$T" --tier-of-record --tsv "$td/absent.tsv"
    head -1 "$FX" > "$td/hdr.tsv"
    row 1 "header-only TSV -> exit 1 (zero pr rows)" 'zero tier=pr rows' bash "$T" --tier-of-record --tsv "$td/hdr.tsv"
    awk -F"\t" 'NR == 1 || $8 != "pr"' "$FX" > "$td/nopr.tsv"
    row 1 "rows but no pr row -> exit 1 (an empty filterset would run nothing)" 'zero tier=pr rows' bash "$T" --tier-of-record --tsv "$td/nopr.tsv"
    awk -F"\t" 'NR > 1' "$FX" > "$td/nohdr.tsv"
    row 1 "no header row -> exit 1" 'header is not' bash "$T" --tier-of-record --tsv "$td/nohdr.tsv"
    { head -1 "$FX"; printf 'crateZ::m\tm\t\t1\t1.00\t1\t0\tpr\tlib\n'; } > "$td/nocrate.tsv"
    row 1 "a pr row with an empty crate -> exit 1 (no half-formed atom)" 'empty crate/module' bash "$T" --tier-of-record --tsv "$td/nocrate.tsv"
    { head -1 "$FX"; printf 'crateZ::\t\tcrateZ\t1\t1.00\t1\t0\tpr\tlib\n'; } > "$td/nomodule.tsv"
    row 1 "a pr row with an empty module -> exit 1" 'empty crate/module' bash "$T" --tier-of-record --tsv "$td/nomodule.tsv"
    # the real table of record: rc 0 and a module count DERIVED from the file
    realn=$(awk -F"\t" 'NR > 1 && $8 == "pr"' evidence/fleet/test-tier.tsv | wc -l | tr -d ' ')
    row 0 "the real table of record -> rc 0 and modules=$realn, derived from the TSV" "^tier_of_record_modules=$realn\$" bash "$T" --tier-of-record
    # --union-touched: misuse is exit 2, never a quiet tier-of-record-only run
    row 2 "--union-touched without --tier-of-record -> exit 2" 'requires --tier-of-record' bash "$T" --union-touched --event pull_request --diff-from "$td/d-scripts.txt"
    row 2 "--union-touched on a push event -> exit 2" 'requires --event pull_request' bash "$T" --tier-of-record --union-touched --event push
    row 2 "--union-touched without a comparand/diff -> exit 2" 'requires --comparand' bash "$T" --tier-of-record --union-touched --event pull_request
    bash "$T" --tier-of-record --tsv "$FX" --union-touched --event pull_request --diff-from "$td/d-leaf.txt" > "$td/u-leaf.out" 2>&1 || true
    row 0 "--union-touched, leaf crate touched -> the tier of record OR that crate's package() atom" '^ALL-PRESENT$' contains_all "$td/u-leaf.out" "filterset=$tor_expr | (package(=$leaf)" "union_touched_crates=$leaf"
    bash "$T" --tier-of-record --tsv "$FX" --union-touched --event pull_request --diff-from "$td/d-root.txt" > "$td/u-root.out" 2>&1 || true
    row 0 "--union-touched when the quick tier falls closed -> tier=full and NO filterset (fail open to more tests)" '^FULL-NO-FILTERSET$' full_ok "$td/u-root.out"
    printf '\n%s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${SELF_TEST:-0}" = 1 ]; then self_test; exit $?; fi
if [ "$UNION" = 1 ]; then
    if [ "$TIER_OF_RECORD" != 1 ]; then
        printf 'usage: --union-touched requires --tier-of-record\n' >&2
        exit 2
    fi
    if [ "$EVENT" != "pull_request" ]; then
        printf 'usage: --union-touched requires --event pull_request (got "%s")\n' "$EVENT" >&2
        exit 2
    fi
    if [ -z "$COMPARAND$DIFF_FROM" ]; then
        printf 'usage: --union-touched requires --comparand REF (or --diff-from FILE)\n' >&2
        exit 2
    fi
    URC=0; union_touched || URC=$?; exit "$URC"
fi
if [ "$TIER_OF_RECORD" = 1 ]; then tier_of_record || exit 1; exit 0; fi
[ -n "$EVENT" ] || { printf 'usage: %s --event EVENT ...\n' "$0" >&2; exit 2; }
decide
