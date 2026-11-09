#!/usr/bin/env bash
# quick_tier_groups.sh — split the quick tier's selected crates into nextest calls that do not unify features (#5210)
#
# THE DEFECT. Cargo unifies features across every package of one call. The quick tier ran its selection as ONE
#   `cargo nextest run -p A -p B ...`, so a selected crate whose features switch on a NON-default feature of another
#   selected crate changed what that crate's tests are. #5188: aprender-test-cli has default = ["browser"] and
#   browser = ["jugar-probar/browser"], so aprender-test-lib was built with `browser`, its two
#   #![cfg(feature = "browser")] test files compiled, and 11 tests failed "Browser not found" on a host with no
#   Chromium. Alone, with its default features, aprender-test-lib compiles neither file.
#
# THE RULE. Crate X is RAISED when another selected crate Y enables a feature f on X that X's own default closure
#   does not hold, through
#     - Y's dependency entry on X (`features = [...]`), or
#     - Y's default-feature closure naming `<key>/f` or `<key>?/f`, where <key> is Y's dependency key for X.
#   Build-dependencies are skipped: resolver 2 builds them in their own feature set. The crates are layered by the
#   longest raise path into each (Y raises X => X sits in a later call than Y), so no call holds a crate together
#   with one that raises it, and the number of calls is the longest raise chain plus one, not one per crate
#   (PMAT-1098: as few build graphs as the rule allows).
#   A raise CYCLE (two crates raising each other, e.g. through dev-dependencies) has no layering; then every
#   selected crate gets its own call. The helper refuses (rc 3) any output that does not hold every selected crate
#   exactly once.
#   Nothing is filtered: every selected crate is still run, each with its own default features.
#
# NOT SEEN. Only a direct dependency between two selected crates. A raise through an unselected crate in between
#   is not detected; that call keeps today's unified behaviour.
#
# OUTPUT. One line per nextest call, crate names space-separated in selection order, layer 0 first. Why a crate
#   was raised goes to stderr.
#
# USAGE  quick_tier_groups.sh --metadata FILE CRATE...   FILE = `cargo metadata --no-deps --format-version 1`
#        quick_tier_groups.sh --self-test                the case table, on synthetic metadata
#        quick_tier_groups.sh --mutants                  each planted mutant of the rule must turn the table red
# EXIT   0 printed · 1 the table broke (or a mutant survived) · 3 caller error (no metadata, a crate the metadata does
#        not name) or a split that does not hold every selected crate exactly once
set -euo pipefail

# jq program: $sel = selected crate names. Prints "RAISED<TAB>X<TAB>by Y: f" lines, then "GROUP<TAB>a b c" lines.
# shellcheck disable=SC2016  # jq source, not shell
JQ='
  def closure($feats; $start):
    # local feature names reachable from $start through $feats
    { seen: [], todo: $start } | until(.todo == [];
      .todo[0] as $f | .todo |= .[1:]
      | if ($f | test("/|^dep:")) or (.seen | index([$f])) then .
        else .seen += [$f] | .todo += ($feats[$f] // []) end) | .seen;
  def entries($feats; $start):
    # every string reachable from $start, local names and key/feat refs alike
    closure($feats; $start) as $c | [$c[] | ($feats[.] // [])[]] + $c | unique;
  [.packages[] | select(.name as $n | $sel | index([$n]))] as $pk
  | ($pk | map({key: .name, value: .}) | from_entries) as $by
  | [ $pk[] as $y | $y.dependencies[]
      | select(.kind != "build") | select(.name as $n | $sel | index([$n])) | select(.name != $y.name)
      | . as $d | ($d.rename // $d.name) as $key
      | ( ($d.features[]),
          (entries($y.features; ["default"])[] | select(startswith($key + "/") or startswith($key + "?/"))
            | sub("^[^/]+/"; "")) )
      | . as $f
      | select(closure($by[$d.name].features; ["default"]) | index([$f]) | not)
      | {x: $d.name, y: $y.name, f: $f} ] | unique as $raises
  | ($raises | map([.y, .x]) | unique) as $edges
  # level = longest raise path into the crate: Y raises X => level(X) > level(Y), so no call holds a raise.
  | (reduce range(0; $sel | length) as $_ ($sel | map({key: ., value: 0}) | from_entries;
       . as $lv | reduce $edges[] as $e (.; .[$e[1]] = ([.[$e[1]], $lv[$e[0]] + 1] | max)))) as $level
  # A raise cycle has no layering: in a DAG the longest path is < n, so a level >= n means a cycle reaches the crate.
  # Then every crate gets its own call: correct for any raise graph, only slower.
  | [$sel[] | select($level[.] >= ($sel | length))] as $cyc
  | ($raises[] | "RAISED\t\(.x)\tby \(.y): \(.f)"),
    ($cyc | select(length > 0) | "CYCLE\t" + join(" ")),
    ($sel[] | select($cyc != []) | "GROUP\t" + .),
    (select($cyc == []) | [$level[]] | unique[] as $l | [$sel[] | select($level[.] == $l)] | "GROUP\t" + join(" "))
'

groups() { # groups <metadata> <crate>...
    local meta="$1" out missing; shift
    [ "$#" -gt 0 ] || return 0
    missing="$(jq -r '[.packages[].name] as $all | $ARGS.positional[] | select(. as $c | $all | index([$c]) | not)' \
        "$meta" --args "$@")" || { echo "quick_tier_groups: cannot read $meta" >&2; return 3; }
    [ -z "$missing" ] || { echo "quick_tier_groups: not in the metadata: $missing" >&2; return 3; }
    out="$(jq -r "\$ARGS.positional as \$sel | ${QTG_MUTANT_JQ:-$JQ}" "$meta" --args "$@")" || return 3
    grep '^RAISED' <<< "$out" | cut -f2- | sed 's/^/quick_tier_groups: raised: /' >&2 || true
    grep '^CYCLE' <<< "$out" | cut -f2- | sed 's/^/quick_tier_groups: raise cycle reaches: /; s/$/ (one call per crate)/' >&2 || true
    # every selected crate exactly once, or nothing: a split that drops a crate is a filter
    if [ "$(grep '^GROUP' <<< "$out" | cut -f2- | tr ' ' '\n' | sort)" != "$(printf '%s\n' "$@" | sort)" ]; then
        echo "quick_tier_groups: the calls do not hold every selected crate exactly once" >&2; return 3
    fi
    grep '^GROUP' <<< "$out" | cut -f2- || true
}

mutants() { # each mutant of the rule must turn --self-test red
    local killed=0 survived=0 from to mut m
    local -a M=(
        'select(.kind != "build")@@select(true)'
        'select(.name != $y.name)@@select(true)'
        '( ($d.features[]),@@( empty,'
        'startswith($key + "?/")@@false'
        '["default"]) | index([$f]) | not)@@["default"]) | index([$f]))'
        '$lv[$e[0]] + 1@@$lv[$e[0]]'
        '.todo += ($feats[$f] // [])@@.todo += []'
        '($d.rename // $d.name) as $key@@$d.name as $key'
        'entries($y.features; ["default"])[]@@closure($y.features; ["default"])[]'
        'select($level[.] >= ($sel | length))@@select($level[.] > ($sel | length))'
    )
    for m in "${M[@]}"; do
        from="${m%%@@*}"; to="${m#*@@}"
        if [[ "$JQ" != *"$from"* ]]; then echo "STALE  $from (not in the rule)"; survived=$((survived + 1)); continue; fi
        mut="${JQ/"$from"/"$to"}"
        if QTG_MUTANT_JQ="$mut" bash "$0" --self-test > /dev/null 2>&1; then
            echo "SURVIVED $from -> $to"; survived=$((survived + 1))
        else echo "killed   $from -> $to"; killed=$((killed + 1)); fi
    done
    echo "quick_tier_groups mutants: $killed killed, $survived survived of ${#M[@]}"
    [ "$survived" = 0 ]
}

self_test() {
    local fails=0 rows=0
    T="$(mktemp -d)"
    trap 'rm -rf "${T:?}"' EXIT
    # pkg <name> <features json> <deps json>
    pkg() { printf '{"name":"%s","features":%s,"dependencies":%s}' "$1" "$2" "$3"; }
    dep() { printf '{"name":"%s","rename":%s,"features":%s,"kind":%s}' "$1" "$2" "$3" "${4:-null}"; }
    # The #5188 shape, verbatim from the workspace: cli -> lib under the key jugar-probar.
    local lib cli serve app other
    lib="$(pkg tlib '{"default":["tui"],"tui":[],"browser":["dep:chromiumoxide"]}' '[]')"
    cli="$(pkg tcli '{"default":["browser"],"browser":["jugar-probar/browser"],"llm":["jugar-probar/llm"]}' \
        "[$(dep tlib '"jugar-probar"' '[]')]")"
    serve="$(pkg serve '{"default":["inference"],"inference":[],"cuda":[]}' '[]')"
    app="$(pkg tapp '{"default":["inf"],"inf":["serve/inference"],"gpu":["serve/cuda"]}' "[$(dep serve null '[]')]")"
    other="$(pkg other '{"default":[]}' "[$(dep tlib null '["browser"]' '"build"')]")"
    printf '{"packages":[%s,%s,%s,%s,%s]}' "$lib" "$cli" "$serve" "$app" "$other" > "$T/m.json"
    # deep closure: default -> a -> b -> "tlib?/browser"
    printf '{"packages":[%s,%s]}' "$lib" \
        "$(pkg deep '{"default":["a"],"a":["b"],"b":["tlib?/browser"]}' "[$(dep tlib null '[]')]")" > "$T/deep.json"
    # dependency entry features = ["browser"]
    printf '{"packages":[%s,%s]}' "$lib" "$(pkg spec '{"default":[]}' "[$(dep tlib null '["browser"]')]")" > "$T/spec.json"
    # a dev-dependency on itself raises nothing: it is one crate, not two
    printf '{"packages":[%s,%s]}' "$lib" "$(pkg selfy '{"default":[],"x":[]}' "[$(dep selfy null '["x"]' '"dev"')]")" > "$T/self.json"
    # g raises serve, tcli raises tlib: two raised crates, one later call
    printf '{"packages":[%s,%s,%s,%s]}' "$lib" "$cli" "$serve" \
        "$(pkg g '{"default":["serve/cuda"]}' "[$(dep serve null '[]')]")" > "$T/two.json"
    # a raises b raises c: three layers
    printf '{"packages":[%s,%s,%s]}' "$(pkg c '{"default":[],"x":[]}' '[]')" \
        "$(pkg b '{"default":["c/x"],"y":[]}' "[$(dep c null '[]')]")" \
        "$(pkg a '{"default":["b/y"]}' "[$(dep b null '[]')]")" > "$T/chain.json"
    # the raised feature is already in the dependency's own default closure
    printf '{"packages":[%s,%s]}' "$(pkg tlib '{"default":["tui"],"tui":["browser"],"browser":[]}' '[]')" "$cli" > "$T/held.json"
    # p and q raise each other through dev-dependencies: no layering exists
    printf '{"packages":[%s,%s,%s]}' "$lib" \
        "$(pkg p '{"default":["q/x"],"y":[]}' "[$(dep q null '[]' '"dev"')]")" \
        "$(pkg q '{"default":["p/y"],"x":[]}' "[$(dep p null '[]' '"dev"')]")" > "$T/cycle.json"

    check() { # check <name> <want rc> <meta> <want stdout, | for newline> <crate>...
        local name="$1" want="$2" meta="$3" wantout="$4" got out; shift 4
        rows=$((rows + 1))
        out="$(bash "$0" --metadata "$meta" "$@" 2>/dev/null)" && got=0 || got=$?
        if [ "$got" = "$want" ] && [ "$(tr '\n' '|' <<< "$out" | sed 's/|$//')" = "$wantout" ]; then echo "ok    $name"
        else echo "BROKE $name (rc $got want $want; got '$(tr '\n' '|' <<< "$out")' want '$wantout')"; fails=$((fails + 1)); fi
    }
    check the_5188_selection_splits_tlib  0 "$T/m.json"    "tapp serve tcli|tlib" tapp serve tcli tlib
    check tlib_alone_is_one_call          0 "$T/m.json"    "tlib"                tlib
    check tcli_alone_is_one_call          0 "$T/m.json"    "tcli"                tcli
    check default_feature_is_not_a_raise  0 "$T/m.json"    "tapp serve"          tapp serve
    check build_dep_is_not_a_raise        0 "$T/m.json"    "other tlib"          other tlib
    check deep_default_closure_raises     0 "$T/deep.json" "deep|tlib"           deep tlib
    check dep_entry_features_raise        0 "$T/spec.json" "spec|tlib"           spec tlib
    check held_by_own_default_no_split    0 "$T/held.json" "tlib tcli"           tlib tcli
    check chain_is_three_calls            0 "$T/chain.json" "a|b|c"             c b a
    check unrelated_raised_share_a_call   0 "$T/two.json"  "g tcli|serve tlib"    g serve tcli tlib
    check self_dependency_is_not_a_raise  0 "$T/self.json" "selfy tlib"           selfy tlib
    check empty_selection_prints_nothing  0 "$T/m.json"    ""
    check unknown_crate_is_caller_error   3 "$T/m.json"    ""                    tlib nope
    check unreadable_metadata_is_error    3 "$T/nope.json" ""                    tlib
    check cycle_is_one_call_per_crate     0 "$T/cycle.json" "p|q|tlib"           p q tlib
    check cycle_member_alone_is_one_call  0 "$T/cycle.json" "p"                  p
    # the reason is printed, so a split is never silent
    rows=$((rows + 1))
    err="$(bash "$0" --metadata "$T/m.json" tcli tlib 2>&1 >/dev/null)" || true
    if grep -qx 'quick_tier_groups: raised: tlib	by tcli: browser' <<< "$err"; then
        echo "ok    raise_reason_names_crate_and_feature"
    else echo "BROKE raise_reason_names_crate_and_feature"; fails=$((fails + 1)); fi
    echo "quick_tier_groups self-test: $((rows - fails)) ok, $fails broke"
    [ "$fails" = 0 ]
}

main() {
    [ "${1:-}" = --self-test ] && { self_test; return; }
    [ "${1:-}" = --mutants ] && { mutants; return; }
    [ "${1:-}" = --metadata ] && [ -n "${2:-}" ] || { echo "usage: $0 --metadata FILE CRATE... | --self-test" >&2; exit 3; }
    local meta="$2"; shift 2
    [ -r "$meta" ] || { echo "quick_tier_groups: cannot read $meta" >&2; exit 3; }
    groups "$meta" "$@"
}

main "$@"
