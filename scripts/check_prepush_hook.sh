#!/usr/bin/env bash
# check_prepush_hook.sh -- the pre-push hook runs, and refuses a push without a
# fresh PASS merge prediction (BLD-001 K1, #4791).
#
# Two checks, over the real tree:
#   1. Every hook in .githooks/ survives its own `set` line under the
#      interpreter its shebang names. `#!/bin/sh` + `set -o pipefail` exits 2 on
#      dash, which refused every push (pre-push until #4791; pre-commit, #3047).
#   2. A case table EXECUTES .githooks/pre-push through its shebang, in a
#      throwaway git repo whose scripts/predict_merge.sh and
#      scripts/ci_guards.sh are stubs and whose compiler/pmat are stubs on
#      PATH. Rows: predict rc 0 -> push allowed; rc 3 (stale), 4 (fetch failed,
#      not_measured), 1 (recorded FAIL) -> refused; a ref that is not HEAD ->
#      refused; delete-only and tag-only pushes -> allowed without a prediction.
#
#   scripts/check_prepush_hook.sh              check the real hooks
#   scripts/check_prepush_hook.sh --self-test  prove both checks turn RED on a
#                                              #!/bin/sh hook and on a hook
#                                              with no prediction step
#
# Exit: 0 pass, 1 a check failed, 2 usage or the harness could not run.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZERO=0000000000000000000000000000000000000000

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# hook_set_line_ok HOOK -- run HOOK's first `set ` line under its shebang.
hook_set_line_ok() {
    local hook="$1" shebang set_line
    local -a interp=()
    shebang="$(head -1 "$hook")"
    case "$shebang" in '#!'*) ;; *) echo "FAIL $hook: no shebang"; return 1 ;; esac
    read -r -a interp <<< "${shebang#\#!}"
    set_line="$(grep -m1 -E '^set ' "$hook")"
    [ -n "$set_line" ] || return 0
    if ! "${interp[@]}" -c "$set_line" > /dev/null 2>&1; then
        echo "FAIL $hook: '$set_line' is rejected by its own shebang (${interp[*]})"
        return 1
    fi
    echo "ok   $hook: '$set_line' runs under ${interp[*]}"
}

# make_fixture HOOK DIR -- a git repo at DIR whose .githooks/pre-push is HOOK.
make_fixture() {
    local hook="$1" fx="$2" b
    mkdir -p "$fx/.githooks" "$fx/scripts" "$fx/stubbin" || return 2
    cp "$hook" "$fx/.githooks/pre-push" && chmod +x "$fx/.githooks/pre-push" || return 2
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fx/scripts/ci_guards.sh"
    # The predict stub records that it ran and exits with the rc in .rc.
    printf '#!/usr/bin/env bash\necho called >> "%s/.called"\nexit "$(cat "%s/.rc")"\n' "$fx" "$fx" \
        > "$fx/scripts/predict_merge.sh"
    for b in pmat cargo; do printf '#!/bin/sh\nexit 0\n' > "$fx/stubbin/$b"; chmod +x "$fx/stubbin/$b"; done
    git -C "$fx" init -q && git -C "$fx" -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null \
        -c commit.gpgsign=false commit -q --allow-empty -m fixture || return 2
}

# run_row FX RC STDIN -- run the fixture hook; print "<hook rc> <predict called 0|1>".
run_row() {
    local fx="$1" rc="$2" input="$3" hrc
    printf '%s' "$rc" > "$fx/.rc"; rm -f "${fx:?}/.called"
    printf '%s' "$input" > "$fx/.in"
    (cd "$fx" && PATH="$fx/stubbin:$PATH" ./.githooks/pre-push < .in > /dev/null 2>&1)
    hrc=$?
    printf '%s %s\n' "$hrc" "$([ -f "$fx/.called" ] && echo 1 || echo 0)"
}

# case_table HOOK -- 0 if every row behaves, 1 otherwise.
case_table() {
    local hook="$1" fx head other fail=0 name rc input want got
    fx="$(mktemp -d)" || return 2
    make_fixture "$hook" "$fx" || { rm -rf "${fx:?}"; return 2; }
    head="$(git -C "$fx" rev-parse HEAD)"
    other=1111111111111111111111111111111111111111
    # name | predict rc | stdin | want: "allow|refuse called|-"
    while IFS='|' read -r name rc input want; do
        input="${input//HEAD/$head}"; input="${input//OTHER/$other}"; input="${input//ZERO/$ZERO}"
        input="${input//;/$'\n'}"
        got="$(run_row "$fx" "$rc" "$input")"
        local hrc="${got% *}" called="${got#* }" verdict
        [ "$hrc" -eq 0 ] && verdict=allow || verdict=refuse
        case "$want" in
            "allow called") [ "$verdict" = allow ] && [ "$called" = 1 ] ;;
            "allow -") [ "$verdict" = allow ] && [ "$called" = 0 ] ;;
            "refuse called") [ "$verdict" = refuse ] && [ "$called" = 1 ] ;;
            "refuse -") [ "$verdict" = refuse ] ;;
        esac
        if [ $? -eq 0 ]; then echo "ok   row $name: $verdict (predict called=$called)"
        else echo "FAIL row $name: got $verdict (hook rc $hrc, predict called=$called), want $want"; fail=1; fi
    done <<'ROWS'
fresh-pass|0|refs/heads/b HEAD refs/heads/b ZERO;|allow called
stale-record|3|refs/heads/b HEAD refs/heads/b ZERO;|refuse called
fetch-failed|4|refs/heads/b HEAD refs/heads/b ZERO;|refuse called
recorded-fail|1|refs/heads/b HEAD refs/heads/b ZERO;|refuse called
ref-not-head|0|refs/heads/x OTHER refs/heads/x ZERO;|refuse -
delete-only|3|(delete) ZERO refs/heads/b HEAD;|allow -
tag-only|3|refs/tags/v9 OTHER refs/tags/v9 ZERO;|allow -
ROWS
    rm -rf "${fx:?}"
    return "$fail"
}

check_real() {
    local fail=0 h
    for h in "$REPO_ROOT"/.githooks/*; do
        [ -f "$h" ] || continue
        hook_set_line_ok "$h" || fail=1
    done
    case_table "$REPO_ROOT/.githooks/pre-push"; case $? in 0) ;; 1) fail=1 ;; *) echo "harness error"; return 2 ;; esac
    return "$fail"
}

self_test() {
    local t fail=0
    t="$(mktemp -d)" || return 2
    # Mutant A: the pre-#4791 shebang. Both checks must turn RED.
    sed '1s|.*|#!/bin/sh|' "$REPO_ROOT/.githooks/pre-push" > "$t/sh-hook"
    if hook_set_line_ok "$t/sh-hook" > /dev/null; then echo "FAIL self-test: #!/bin/sh hook passed the set-line check"; fail=1
    else echo "ok   self-test: #!/bin/sh hook is RED on the set-line check"; fi
    if case_table "$t/sh-hook" > /dev/null; then echo "FAIL self-test: #!/bin/sh hook passed the case table"; fail=1
    else echo "ok   self-test: #!/bin/sh hook is RED on the case table"; fi
    # Mutant B: the prediction step deleted (everything from the K1 banner on).
    sed '/BLD-001 K1 (#4791): merge prediction/,$d' "$REPO_ROOT/.githooks/pre-push" > "$t/no-predict"
    if case_table "$t/no-predict" > /dev/null; then echo "FAIL self-test: hook with no prediction step passed"; fail=1
    else echo "ok   self-test: hook with no prediction step is RED"; fi
    # Mutant C: the prediction made advisory.
    sed '/REFUSED: \(no fresh\|git fetch\|the recorded\)/s/; exit 1 ;;$/ ;;/' "$REPO_ROOT/.githooks/pre-push" > "$t/advisory"
    if cmp -s "$t/advisory" "$REPO_ROOT/.githooks/pre-push"; then echo "FAIL self-test: mutant C did not apply"; fail=1
    elif case_table "$t/advisory" > /dev/null; then echo "FAIL self-test: advisory prediction passed"; fail=1
    else echo "ok   self-test: advisory prediction is RED"; fi
    # The real hook must be GREEN, or the RED rows above prove nothing.
    if case_table "$REPO_ROOT/.githooks/pre-push" > /dev/null; then echo "ok   self-test: the real hook is GREEN"
    else echo "FAIL self-test: the real hook is RED"; fail=1; fi
    rm -rf "${t:?}"
    return "$fail"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test) self_test; rc=$? ;;
    "") check_real; rc=$? ;;
    *) usage >&2; exit 2 ;;
esac
[ "$rc" -eq 0 ] && echo "check_prepush_hook: PASS" || echo "check_prepush_hook: FAIL (rc=$rc)"
exit "$rc"
