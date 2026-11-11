#!/usr/bin/env bash
# roadmap_fold_crates.sh — the crates a roadmap-fold input diff owes on the PR path (#6902, PMAT-6902)
#
# THE DEFECT. docs/roadmaps/roadmap.yaml is the fold of roadmap-base.yaml and entries/*.yaml. Two tests prove it:
#   cargo test -p aprender-rollup --lib the_committed_roadmap_is_the_fold_of_its_base
#   cargo test -p aprender-ci-tools --test rollup_fold
#   The PR quick tier picks crates from the diff, and a diff under docs/roadmaps/ names neither crate (and a diff of
#   that directory alone is docs-only, tier=none). A PR that added entries/PMAT-<n>.yaml without regenerating
#   roadmap.yaml passed every PR check and turned main red (PR #6899; main run 43094456537, fixed by #6901).
#
# THE RULE. A diff that touches docs/roadmaps/entries/** , docs/roadmaps/roadmap-base.yaml or
#   docs/roadmaps/roadmap.yaml owes the crates aprender-rollup and aprender-ci-tools. The match is anchored at the
#   repo root, so crates/x/docs/roadmaps/entries/a.yaml and docs/roadmaps/entries.bak/x are not fold inputs.
#   scripts/ci_test_tier.sh calls --touches (such a diff is never docs-only) and --union (add the two crates to the
#   selection the diff already has, each name once). The quick tier then runs both crates' lib and integration
#   targets, so rollup_fold runs.
#
# USAGE  roadmap_fold_crates.sh --touches DIFF            rc 0 a fold input is touched, 1 none is
#        roadmap_fold_crates.sh --union DIFF [CRATE...]    the crates, space-joined: CRATE... then, when DIFF
#                                                          touches a fold input, the two more, each name once
#        roadmap_fold_crates.sh --self-test               the case table
#        roadmap_fold_crates.sh --mutants                 each planted mutant of the rule must turn the table red
#   DIFF is a file of repo-relative paths, one per line (the file ci_test_tier.sh already holds).
# EXIT   0 ok · 1 the table broke (or a mutant survived) / --touches found nothing · 3 caller error
set -euo pipefail

RE='^docs/roadmaps/(entries/.+|roadmap-base\.yaml|roadmap\.yaml)$'
CRATES='aprender-rollup aprender-ci-tools'

touches() { # touches <diff file> -> rc 0 iff a path matches RE
    local rc=0
    [ -r "$1" ] || { echo "roadmap_fold_crates: cannot read $1" >&2; return 3; }
    grep -Eq -- "${RFC_MUTANT_RE:-$RE}" "$1" || rc=$?
    [ "$rc" -le 1 ] || { echo "roadmap_fold_crates: grep failed ($rc)" >&2; return 3; }
    return "$rc"
}

union() { # union <diff file> [crate...]
    local diff="$1" c seen=" " rc=0; shift
    local -a out=() add=()
    for c in "$@"; do
        case "$seen" in *" $c "*) ;; *) out+=("$c"); seen="$seen$c " ;; esac
    done
    touches "$diff" || rc=$?
    case "$rc" in
        0) read -ra add <<< "${RFC_MUTANT_CRATES:-$CRATES}"
           for c in "${add[@]}"; do
               case "$seen" in *" $c "*) ;; *) out+=("$c"); seen="$seen$c " ;; esac
           done ;;
        1) ;;
        *) return "$rc" ;;
    esac
    printf '%s\n' "${out[*]-}"
}

mutants() { # each mutant of the rule must turn --self-test red
    local killed=0 survived=0 from to src mut m
    src="$(cat "$0")"
    local -a M=(
        'entries/.+|@@entries/|'
        'entries/.+|@@entries/[^/]+$|'
        '|roadmap-base\.yaml@@'
        '|roadmap\.yaml)@@)'
        "RE='^docs@@RE='docs"
        '(entries/.+|roadmap-base\.yaml|roadmap\.yaml)$@@(entries/.+|roadmap-base\.yaml|roadmap\.yaml)'
        'entries/.+@@entries.+'
        'roadmap-base\.yaml@@roadmap-base.yaml'
        '|roadmap\.yaml)@@|roadmap.yaml)'
        "CRATES='aprender-rollup aprender-ci-tools'@@CRATES='aprender-ci-tools'"
        "CRATES='aprender-rollup aprender-ci-tools'@@CRATES='aprender-rollup'"
        "CRATES='aprender-rollup aprender-ci-tools'@@CRATES='aprender-rollup aprender-ci-tools aprender-core'"
        '0) read -ra add@@1) read -ra add'
        '*" $c "*) ;; *) out+=("$c"); seen="$seen$c " ;; esac
           done ;;@@*" $c "*) out+=("$c") ;; *) out+=("$c"); seen="$seen$c " ;; esac
           done ;;'
        '    touches "$diff" || rc=$?@@    rc=1'
        '[ "$rc" -le 1 ] ||@@[ "$rc" -le 2 ] ||'
    )
    for m in "${M[@]}"; do
        from="${m%%@@*}"; to="${m#*@@}"
        if [[ "$src" != *"$from"* ]]; then echo "STALE  $from (not in the script)"; survived=$((survived + 1)); continue; fi
        mut="${src/"$from"/"$to"}"
        printf '%s\n' "$mut" > "$MUT_DIR/m.sh"
        if bash "$MUT_DIR/m.sh" --self-test > /dev/null 2>&1; then
            echo "SURVIVED ${from%%$'\n'*} -> ${to%%$'\n'*}"; survived=$((survived + 1))
        else echo "killed   ${from%%$'\n'*} -> ${to%%$'\n'*}"; killed=$((killed + 1)); fi
    done
    echo "roadmap_fold_crates mutants: $killed killed, $survived survived of ${#M[@]}"
    [ "$survived" = 0 ]
}

self_test() {
    local fails=0 rows=0 T
    T="$(mktemp -d "${TMPDIR:-/tmp}/rfc.XXXXXX")"
    # shellcheck disable=SC2064  # expand T now: it is local to this function
    trap "rm -rf '${T:?}'" EXIT
    d() { local f="$T/$1.txt"; shift; printf '%s\n' "$@" > "$f"; echo "$f"; }

    touch_row() { # touch_row <name> <want rc> <path>...
        local name="$1" want="$2" f got=0; shift 2
        f="$(d "r$rows" "$@")"; rows=$((rows + 1))
        bash "$0" --touches "$f" > /dev/null 2>&1 || got=$?
        if [ "$got" = "$want" ]; then echo "ok    $name"
        else echo "BROKE $name (rc $got want $want)"; fails=$((fails + 1)); fi
    }
    union_row() { # union_row <name> <want stdout> <given crates, space list> <path>...
        local name="$1" want="$2" given="$3" f out; shift 3
        f="$(d "r$rows" "$@")"; rows=$((rows + 1))
        # shellcheck disable=SC2086  # the crates are a space list
        out="$(bash "$0" --union "$f" $given 2>/dev/null)" || out="rc$?"
        if [ "$out" = "$want" ]; then echo "ok    $name"
        else echo "BROKE $name (got '$out' want '$want')"; fails=$((fails + 1)); fi
    }
    BOTH='aprender-rollup aprender-ci-tools'
    # must-select
    touch_row fragment_is_a_fold_input                   0 docs/roadmaps/entries/PMAT-1.yaml
    touch_row fragment_among_other_paths                 0 scripts/foo.sh docs/roadmaps/entries/PMAT-6902.yaml crates/a/src/lib.rs
    touch_row nested_fragment_path_is_a_fold_input       0 docs/roadmaps/entries/sub/x.yaml
    touch_row roadmap_base_is_a_fold_input               0 docs/roadmaps/roadmap-base.yaml
    touch_row aggregate_is_a_fold_input                  0 docs/roadmaps/roadmap.yaml
    # must-not-select
    touch_row roadmaps_readme_is_not                     1 docs/roadmaps/README.md
    touch_row other_docs_path_is_not                     1 docs/audits/pp-066-plan-quorum.md
    touch_row nested_crate_docs_path_is_not              1 crates/x/docs/roadmaps/entries/a.yaml
    touch_row sibling_dir_prefix_is_not                  1 docs/roadmaps/entries.bak/x
    touch_row entries_dir_itself_is_not                  1 docs/roadmaps/entries/
    touch_row a_prefix_before_docs_is_not                1 xdocs/roadmaps/roadmap.yaml
    touch_row aggregate_dot_is_literal                   1 docs/roadmaps/roadmapXyaml
    touch_row base_dot_is_literal                        1 docs/roadmaps/roadmap-baseXyaml
    touch_row aggregate_suffix_is_not                    1 docs/roadmaps/roadmap.yaml.bak
    touch_row base_suffix_is_not                         1 docs/roadmaps/roadmap-base.yaml.orig
    touch_row aggregate_prefix_is_not                    1 docs/roadmaps/old-roadmap.yaml
    touch_row empty_diff_is_not                          1 ''
    touch_row ordinary_rust_path_is_not                  1 crates/aprender-core/src/lib.rs
    # --union
    union_row fragment_adds_both_to_empty                "$BOTH"                      ''                       docs/roadmaps/entries/PMAT-1.yaml
    union_row base_adds_both                             "$BOTH"                      ''                       docs/roadmaps/roadmap-base.yaml
    union_row aggregate_adds_both                        "$BOTH"                      ''                       docs/roadmaps/roadmap.yaml
    union_row existing_crates_come_first                 "apr-cli $BOTH"              'apr-cli'                docs/roadmaps/entries/PMAT-1.yaml
    union_row one_already_selected_no_duplicate          "aprender-rollup aprender-ci-tools"  'aprender-rollup' docs/roadmaps/entries/PMAT-1.yaml
    union_row other_already_selected_no_duplicate        "aprender-ci-tools aprender-rollup"  'aprender-ci-tools' docs/roadmaps/entries/PMAT-1.yaml
    union_row both_already_selected_no_duplicate         "$BOTH"                      "$BOTH"                  docs/roadmaps/roadmap.yaml
    union_row no_fold_input_leaves_selection_alone       'apr-cli'                    'apr-cli'                docs/roadmaps/README.md
    union_row no_fold_input_empty_selection_stays_empty  ''                          ''                       docs/audits/a.md
    union_row crate_path_with_the_string_adds_nothing    'apr-cli'                    'apr-cli'                crates/x/docs/roadmaps/entries/a.yaml
    union_row repeated_input_crate_listed_once           'apr-cli'                    'apr-cli apr-cli'        scripts/foo.sh
    # caller errors
    rows=$((rows + 1)); bash "$0" --touches "$T/nope.txt" > /dev/null 2>&1 && got=0 || got=$?
    if [ "$got" = 3 ]; then echo "ok    unreadable_diff_is_caller_error"; else echo "BROKE unreadable_diff_is_caller_error (rc $got)"; fails=$((fails + 1)); fi
    rows=$((rows + 1)); bash "$0" --union "$T/nope.txt" apr-cli > /dev/null 2>&1 && got=0 || got=$?
    if [ "$got" = 3 ]; then echo "ok    union_of_unreadable_diff_is_caller_error"; else echo "BROKE union_of_unreadable_diff_is_caller_error (rc $got)"; fails=$((fails + 1)); fi
    rows=$((rows + 1)); printf "docs/roadmaps/roadmap.yaml\n" > "$T/g.txt"; RFC_MUTANT_RE="(" bash "$0" --touches "$T/g.txt" > /dev/null 2>&1 && got=0 || got=$?
    if [ "$got" = 3 ]; then echo "ok    a_failing_grep_is_caller_error_not_no"; else echo "BROKE a_failing_grep_is_caller_error_not_no (rc $got)"; fails=$((fails + 1)); fi
    echo "roadmap_fold_crates self-test: $((rows - fails)) ok, $fails broke"
    [ "$fails" = 0 ]
}

main() {
    case "${1:-}" in
        --self-test) self_test ;;
        --mutants) MUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rfcm.XXXXXX")"
                   # shellcheck disable=SC2064
                   trap "rm -rf '${MUT_DIR:?}'" EXIT
                   mutants ;;
        --touches) [ -n "${2:-}" ] || { echo "usage: $0 --touches DIFF" >&2; exit 3; }; touches "$2" ;;
        --union) [ -n "${2:-}" ] || { echo "usage: $0 --union DIFF [CRATE...]" >&2; exit 3; }
                 local diff="$2"; shift 2; union "$diff" "$@" ;;
        *) echo "usage: $0 --touches DIFF | --union DIFF [CRATE...] | --self-test | --mutants" >&2; exit 3 ;;
    esac
}

main "$@"
