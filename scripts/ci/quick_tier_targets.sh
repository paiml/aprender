#!/usr/bin/env bash
# quick_tier_targets.sh — the target flags of ONE quick-tier nextest call, from the packages it holds (#6898)
#
# THE DEFECT. The quick tier ran every call as `cargo nextest run --lib --tests -p ...`. Cargo refuses `--lib` when
#   no package of the call has a library target: `error: no library targets found in package aprender-present-cli`,
#   rc 101, no junit. A diff that selected a bin-only crate into a call of its own turned main red (85c3a552e,
#   run 43045218074, 2026-11-11) with no test having failed.
#
# THE RULE. A call gets `--lib --tests` when at least one of its packages has a library target, and `--tests` when
#   none has. A library target is one whose `kind` holds lib, rlib, dylib, cdylib, staticlib or proc-macro.
#   So every call that ran before runs with the flags it had, and only the call cargo refused changes.
#   `--tests` alone still runs a bin-only package's unit tests and its integration targets.
#
# NOT SEEN. A package with no test-mode target at all (examples or benches only). `--tests` then selects nothing
#   and nextest decides; no workspace member has that shape today.
#
# OUTPUT. One line: the flags.
#
# USAGE  quick_tier_targets.sh --metadata FILE CRATE...   FILE = `cargo metadata --no-deps --format-version 1`
#        quick_tier_targets.sh --self-test                the case table, on synthetic metadata
#        quick_tier_targets.sh --mutants                  each planted mutant of the rule must turn the table red
# EXIT   0 printed · 1 the table broke (or a mutant survived) · 3 caller error (no metadata, no crate, a crate the
#        metadata does not name)
set -euo pipefail

# jq program: $sel = the call's crate names. Prints the flags.
# shellcheck disable=SC2016  # jq source, not shell
JQ='
  ["lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"] as $libkinds
  | [ .packages[] | select(.name as $n | $sel | index([$n]))
      | .targets[] | .kind[] | select(. as $k | $libkinds | index([$k])) ] as $libs
  | if ($libs | length) > 0 then "--lib --tests" else "--tests" end
'

flags() { # flags <metadata> <crate>...
    local meta="$1" missing; shift
    [ "$#" -gt 0 ] || { echo "quick_tier_targets: a call holds at least one crate" >&2; return 3; }
    missing="$(jq -r '[.packages[].name] as $all | $ARGS.positional[] | select(. as $c | $all | index([$c]) | not)' \
        "$meta" --args "$@")" || { echo "quick_tier_targets: cannot read $meta" >&2; return 3; }
    [ -z "$missing" ] || { echo "quick_tier_targets: not in the metadata: $missing" >&2; return 3; }
    jq -r "\$ARGS.positional as \$sel | ${QTT_MUTANT_JQ:-$JQ}" "$meta" --args "$@" || return 3
}

mutants() { # each mutant of the rule must turn --self-test red
    local killed=0 survived=0 from to mut m
    local -a M=(
        '($libs | length) > 0@@($libs | length) >= 0'
        '($libs | length) > 0@@($libs | length) > 1'
        'then "--lib --tests" else "--tests"@@then "--tests" else "--lib --tests"'
        'else "--tests" end@@else "--lib --tests" end'
        'then "--lib --tests" else@@then "--tests" else'
        'select(.name as $n | $sel | index([$n]))@@select(true)'
        '"rlib", @@'
        '"dylib", @@'
        '"cdylib", @@'
        '"staticlib", @@'
        ', "proc-macro"]@@]'
        '["lib", @@['
        '.targets[] | .kind[]@@.targets[0] | .kind[]'
        '.kind[] | select@@.kind[0] | select'
    )
    for m in "${M[@]}"; do
        from="${m%%@@*}"; to="${m#*@@}"
        if [[ "$JQ" != *"$from"* ]]; then echo "STALE  $from (not in the rule)"; survived=$((survived + 1)); continue; fi
        mut="${JQ/"$from"/"$to"}"
        if QTT_MUTANT_JQ="$mut" bash "$0" --self-test > /dev/null 2>&1; then
            echo "SURVIVED $from -> $to"; survived=$((survived + 1))
        else echo "killed   $from -> $to"; killed=$((killed + 1)); fi
    done
    echo "quick_tier_targets mutants: $killed killed, $survived survived of ${#M[@]}"
    [ "$survived" = 0 ]
}

self_test() {
    local fails=0 rows=0 T
    T="$(mktemp -d)"
    # shellcheck disable=SC2064  # expand T now: it is local to this function
    trap "rm -rf '${T:?}'" EXIT
    # pkg <name> <kind>...   one target per argument; a + joins several kinds on one target
    pkg() {
        local name="$1" k out=""; shift
        for k in "$@"; do out="$out{\"kind\":[\"${k//+/\",\"}\"]},"; done
        printf '{"name":"%s","targets":[%s]}' "$name" "${out%,}"
    }
    printf '{"packages":[%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s]}' \
        "$(pkg libonly lib)" \
        "$(pkg binonly bin)" \
        "$(pkg bintests bin test test)" \
        "$(pkg binfirst bin lib test)" \
        "$(pkg macros proc-macro)" \
        "$(pkg wasm cdylib+rlib)" \
        "$(pkg ffi staticlib)" \
        "$(pkg plug dylib)" \
        "$(pkg rl rlib)" \
        "$(pkg tools bin bin example bench custom-build)" \
        "$(pkg kindsecond bin+cdylib)" > "$T/m.json"

    check() { # check <name> <want rc> <meta> <want stdout> <crate>...
        local name="$1" want="$2" meta="$3" wantout="$4" got out; shift 4
        rows=$((rows + 1))
        out="$(bash "$0" --metadata "$meta" "$@" 2>/dev/null)" && got=0 || got=$?
        if [ "$got" = "$want" ] && [ "$out" = "$wantout" ]; then echo "ok    $name"
        else echo "BROKE $name (rc $got want $want; got '$out' want '$wantout')"; fails=$((fails + 1)); fi
    }
    check lib_only_keeps_both_flags          0 "$T/m.json" "--lib --tests" libonly
    check bin_only_drops_lib                 0 "$T/m.json" "--tests"       binonly
    check bin_with_integration_tests_drops   0 "$T/m.json" "--tests"       bintests
    check lib_after_a_bin_target_is_seen     0 "$T/m.json" "--lib --tests" binfirst
    check proc_macro_is_a_library            0 "$T/m.json" "--lib --tests" macros
    check cdylib_rlib_is_a_library           0 "$T/m.json" "--lib --tests" wasm
    check staticlib_is_a_library             0 "$T/m.json" "--lib --tests" ffi
    check dylib_is_a_library                 0 "$T/m.json" "--lib --tests" plug
    check rlib_is_a_library                  0 "$T/m.json" "--lib --tests" rl
    check second_kind_of_a_target_is_seen    0 "$T/m.json" "--lib --tests" kindsecond
    check example_bench_build_are_not_libs   0 "$T/m.json" "--tests"       tools
    check mixed_call_keeps_both_flags        0 "$T/m.json" "--lib --tests" binonly libonly
    check mixed_call_any_order               0 "$T/m.json" "--lib --tests" libonly binonly
    check two_bin_only_crates_drop_lib       0 "$T/m.json" "--tests"       binonly tools
    check unselected_lib_does_not_count      0 "$T/m.json" "--tests"       bintests
    check unknown_crate_is_caller_error      3 "$T/m.json" ""              binonly nope
    check no_crate_is_caller_error           3 "$T/m.json" ""
    check unreadable_metadata_is_error       3 "$T/nope.json" ""           binonly
    echo "quick_tier_targets self-test: $((rows - fails)) ok, $fails broke"
    [ "$fails" = 0 ]
}

main() {
    [ "${1:-}" = --self-test ] && { self_test; return; }
    [ "${1:-}" = --mutants ] && { mutants; return; }
    [ "${1:-}" = --metadata ] && [ -n "${2:-}" ] || { echo "usage: $0 --metadata FILE CRATE... | --self-test | --mutants" >&2; exit 3; }
    local meta="$2"; shift 2
    [ -r "$meta" ] || { echo "quick_tier_targets: cannot read $meta" >&2; exit 3; }
    flags "$meta" "$@"
}

main "$@"
