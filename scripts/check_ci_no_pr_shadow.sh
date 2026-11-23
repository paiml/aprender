#!/usr/bin/env bash
# check_ci_no_pr_shadow.sh -- pr-review-shadow is not run on the PR or merge-queue path. #5996, C324.
#
# THE DEFECT. The pr-review-shadow section records a REFUSE verdict and exits 0 on a pull request: it can neither
# block nor pass, which is the third state C324 bans on a PR or release path (a check is a GATE that can block, or a
# LAB that never runs there). It was found by the quorum on #5994 and had been inside x86-main's section lists.
#
# RULE. STRICT ALLOWLIST, not pattern-hunting (five rounds of regex patching were bypassed). Every line of a workflow that
# names the driver (fat_d..., ci/fat_...; comments included, nothing is dropped) is one of: a known benign form (a step
# name, `fat_driver.py self-test`, `fat_driver.py wait --results "$RUNNER_TEMP/fat/results.json"`), a prose comment with no
# `sections`/`--s`/`<<`/backslash, or EXACTLY
#     [bash scripts/ci/idle_watchdog.sh "$RUNNER_TEMP/fat" -- ]python3 scripts/ci/fat_driver.py run --sections <'LIST'|LIST>
#         [ --results "$RUNNER_TEMP/<path>"][ --background-until '<LIST>']
# after the backslash-space / folded-line join. Anything else on such a line is RED. Rules (the tag is in each FAIL line):
#   R1 form      the statement is not the accepted form (option spelled with a quote, brace, ${IFS}, =, abbreviation, a
#                variable or a here-string instead of the value, extra tokens ...)
#   R2 bracket   `[` followed by . = : [ ^ ! ] in a list (bash and Python fnmatch read these differently)
#   R3 escape    any backslash left after the join (\x73, \u0073, \t, \=, ...)
#   R4 charset   the list holds a character outside [A-Za-z0-9._?*@,[]-] (quote, $, space, brace, backtick ...)
#   R5 resolves  a list part matches pr-review-shadow (globs included)
#   R6 glued     a line ends in a backslash glued to a non-space character (bash joins it mid-word)
#   R7 heredoc   `<<` on a driver line
#   R8 vacuous   ci.yml holds no accepted list
#   R9 expr      a ${{ }} expression other than the workspace-test shard index
#   R10 comment  a comment that names the driver and `sections`, `--s` or carries code-looking syntax
#   R11 orphan   a `--se...` option on a line that is not part of a driver statement
# SCOPE. ci.yml (must hold >=1 list) and every other .github/workflows/*.yml that names the driver AND can run on the PR or
# merge-queue path (its `on:` block names pull_request, pull_request_target, merge_group or workflow_call, or has no `on:` block).
# A schedule/workflow_dispatch-only LAB workflow may carry the section.
#
# Not covered, stated plainly: a driver reference that hides every `fat_d`/`ci/fat_` substring (`fa""t_driver`).
#
# Usage:  bash scripts/check_ci_no_pr_shadow.sh [--self-test] [WORKFLOW_FILE]
#         WORKFLOW_FILE given: only that file (>=1 list required). Default: ci.yml plus every PR-path workflow beside it.
# Wired:  ci/explicit-test-commands.d/893-ci-no-pr-shadow.cmd (--self-test).
set -euo pipefail

TRIG='fat_d|ci/fat|fat_["'"'"'\\]'
PRE='^[[:space:]]*(bash scripts/ci/idle_watchdog\.sh "\$RUNNER_TEMP/fat" -- )?python3 scripts/ci/fat_driver\.py run --sections (.*)$'
TAIL="^( --results \"\\\$RUNNER_TEMP/[A-Za-z0-9_./-]+\")?( --background-until '[]A-Za-z0-9._?*@,[-]+')?\$"
NAMEF='^[[:space:]]*-?[[:space:]]*name: fat_driver case table must still turn RED$'
SELFF='^[[:space:]]*run: setsid --wait bash scripts/ci/fixture_quiet\.sh "fat_driver self-test" -- python3 scripts/ci/fat_driver\.py self-test$'
TEARF='^[[:space:]]*run: if \[ -f scripts/ci/fat_teardown\.sh \]; then bash scripts/ci/fat_teardown\.sh "\$GITHUB_RUN_ID" "\$RUNNER_TEMP"; fi$'
WAITF='^[[:space:]]*run: python3 scripts/ci/fat_driver\.py wait --results "\$RUNNER_TEMP/fat/results\.json"$'
N_LISTS=0

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }

# stmt_ok LINENO STATEMENT -> prints FAIL lines, returns 1 on any
stmt_ok() {
    local at="$1" stmt="$2" rest value after part bad=0
    case "$stmt" in *'<<'*) echo "FAIL  [R7-heredoc] line $at: here-doc on a driver line: $stmt"; return 1 ;; esac
    case "$stmt" in *\\*) echo "FAIL  [R3-escape] line $at: backslash escape left in a driver statement: $stmt"; return 1 ;; esac
    if [[ $stmt =~ $NAMEF || $stmt =~ $SELFF || $stmt =~ $WAITF || $stmt =~ $TEARF ]]; then return 0; fi
    if [[ $stmt =~ ^[[:space:]]*# ]]; then
        case "$stmt" in *--s*|*"sections '"*|*'sections "'*) echo "FAIL  [R10-comment] line $at: comment names the driver and a sections option: $stmt"; return 1 ;; esac
        return 0
    fi
    stmt="$(printf '%s' "$stmt" | sed -E 's/\$\{\{[^}]*\}\}/@EXPR@/g')"
    if ! [[ $stmt =~ $PRE ]]; then echo "FAIL  [R1-form] line $at: driver line is not the accepted 'fat_driver.py run --sections <list>' form: $stmt"; return 1; fi
    rest="${BASH_REMATCH[2]}"
    if [[ $rest == \'* ]]; then
        rest="${rest#\'}"
        if [[ $rest != *\'* ]]; then echo "FAIL  [R1-form] line $at: unterminated quote in the list: $stmt"; return 1; fi
        value="${rest%%\'*}"; after="${rest#*\'}"
    else
        value="${rest%% *}"; after="${rest#"$value"}"
    fi
    case "$value" in *'[.'*|*'[='*|*'[:'*|*'[['*|*'[^'*|*'[!'*|*'[]'*) echo "FAIL  [R2-bracket] line $at: bracket form that bash and fnmatch read differently: '$value'"; return 1 ;; esac
    case "$value" in ''|*[!A-Za-z0-9._?*@,\[\]-]*) echo "FAIL  [R4-charset] line $at: list is empty or holds a character outside the allowlist (quote, \$, space, brace ...): '$value'"; return 1 ;; esac
    if ! [[ $after =~ $TAIL ]]; then echo "FAIL  [R1-form] line $at: tokens after the list are not exactly --results/--background-until: '$after'"; return 1; fi
    local IFS=,
    for part in $value; do
        case "$part" in *@EXPR@*) case "$part" in workspace-test-shard?@EXPR@?@EXPR@?) part="${part//@EXPR@/?}" ;; *) echo "FAIL  [R9-expr] line $at: list part holds an expression and cannot be resolved: '$part'"; bad=1; continue ;; esac ;; esac
        # shellcheck disable=SC2053  # the unquoted right side is the point: fat_driver matches a section name with fnmatch
        if [[ pr-review-shadow == $part ]]; then echo "FAIL  [R5-resolves] line $at: --sections '$value' resolves to pr-review-shadow (via '$part')"; bad=1; fi
    done
    N_LISTS=$((N_LISTS + 1))
    return "$bad"
}

check_file() { # FILE REQUIRE(1|0) -> FAIL lines; rc 1 when any
    local f="$1" need="$2" bad=0 i j total start cur nxt line
    local -a L
    mapfile -t L < "$f"
    total="${#L[@]}"; N_LISTS=0
    for ((i = 0; i < total; i++)); do
        line="${L[i]}"
        if ! [[ $line =~ $TRIG ]]; then
            if [[ $line =~ (^|[[:space:]\'\"])--s(e|ec|ect|ecti|ectio|ection|ections)?([^a-z-]|$) && ! $line =~ ^[[:space:]]*# ]]; then echo "FAIL  [R11-orphan] line $((i + 1)): a --se... option outside a driver statement: $line"; bad=1; fi
            continue
        fi
        start=$((i + 1)); cur="$line"
        if ! [[ $cur =~ ^[[:space:]]*# ]]; then
            while ((i + 1 < total)); do
                nxt="${L[i + 1]}"
                if [[ $cur =~ [^[:space:]]\\$ ]]; then echo "FAIL  [R6-glued] line $((i + 1)): a line ends in a backslash glued to a non-space character (bash joins it mid-word): $cur"; bad=1; break; fi
                if [[ $cur =~ [[:space:]]\\$ ]]; then cur="$(trim "${cur%\\}") $(trim "$nxt")"; i=$((i + 1)); continue; fi
                if [[ $nxt =~ ^[[:space:]]*-- || ( $(trim "$cur") == *' --sections' && $nxt =~ ^[[:space:]]*[\'\"] ) ]]; then cur="$(trim "$cur") $(trim "$nxt")"; i=$((i + 1)); continue; fi
                break
            done
            if [[ $cur =~ [^[:space:]]\\$ ]]; then echo "FAIL  [R6-glued] line $((i + 1)): a line ends in a backslash glued to a non-space character (bash joins it mid-word): $cur"; bad=1; continue; fi
        fi
        j=$(stmt_ok "$start" "$cur" && echo OK) || true
        # stmt_ok runs in a subshell above, so counting is redone here for the accepted run forms
        if [[ $j == *FAIL* ]]; then printf '%s\n' "$j"; bad=1; elif [[ $cur =~ $PRE ]]; then N_LISTS=$((N_LISTS + 1)); fi
    done
    if [ "$need" = 1 ] && [ "$N_LISTS" -eq 0 ]; then echo "FAIL  [R8-vacuous] no accepted --sections list found in $f (the check would be vacuous)"; bad=1; fi
    return "$bad"
}

pr_path() { # FILE -> 0 when the workflow can run on the PR / merge-queue path
    local on
    on="$(awk '/^("on"|on|true):/{p=1;print;next} p&&/^[^[:space:]#]/{p=0} p' "$1" | grep -v '^[[:space:]]*#' || true)"
    [ -z "$on" ] && return 0
    printf '%s\n' "$on" | grep -qE 'pull_request|merge_group|workflow_call|workflow_run'
}

check() { # FILE-or-DIR-anchor -> ci.yml (required) plus PR-path siblings
    local ci="$1" bad=0 d w out
    check_file "$ci" 1 || bad=1
    d="$(dirname "$ci")"
    for w in "$d"/*.yml "$d"/*.yaml; do
        [ -f "$w" ] || continue
        [ "$w" -ef "$ci" ] && continue
        grep -qE "$TRIG" "$w" || continue
        pr_path "$w" || continue
        if out="$(check_file "$w" 0)"; then :; else printf "%s\n" "$out" | sed "s|^FAIL  |FAIL  ($(basename "$w")) |"; bad=1; fi
    done
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: committed workflow is RED: $got"; fails=1; fi
    local anchor="--sections 'guard-tree,pr-review-sign"
    if ! grep -qF -- "$anchor" "$src"; then echo "FAIL  anchor missing from $src; self-test is vacuous"; return 1; fi
    mutant() { # LABEL RULES(ERE alternation of Rn) SED-EXPR -> red, and with one of those rule tags
        sed -e "$3" "$src" > "$td/ci.yml"
        if cmp -s "$src" "$td/ci.yml"; then echo "FAIL  mutant did not change the file: $1"; fails=1; return; fi
        if got="$(check "$td/ci.yml")"; then echo "FAIL  mutant stayed green: $1"; fails=1
        elif printf '%s\n' "$got" | grep -qE "\[($2)-"; then echo "ok    red   MUTANT [$2]: $1"
        else echo "FAIL  mutant red for the WRONG reason (wanted $2): $1 :: $(printf '%s' "$got" | head -n 2)"; fails=1; fi
    }
    local A="s/--sections 'guard-tree,pr-review-sign/"
    mutant "shadow put back in a list" R5 "${A}--sections 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "glob pr-review-* resolves to shadow" R5 "${A}--sections 'guard-tree,pr-review-*/"
    mutant "glob pr-review-s?a*" R5 "${A}--sections 'guard-tree,pr-review-s?a*/"
    mutant "value on the next line (folded block)" R5 "${A}--sections\n          'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "--sections=value form" R1 "${A}--sections='guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "indirect \$VAR value" R4 "${A}--sections \"\$PR_SECS\" --x 'guard-tree,pr-review-sign/"
    mutant "no --sections list is vacuous, not green" "R1|R8" "s/--sections/--other/"
    mutant "space after a comma" R4 "${A}--sections 'guard-tree, pr-review-shadow,pr-review-sign/"
    mutant "abbreviated option --section" R1 "${A}--section 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "abbreviated option --se" R1 "${A}--se 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "quote splicing" R1 "${A}--sections 'guard-tree,'pr-review-shadow,pr-review-sign/"
    mutant "whole-value expression" "R9|R1" "${A}--sections '\${{ vars.PR_SECS }}' --x 'guard-tree,pr-review-sign/"
    mutant "partial expression" R9 "${A}--sections 'guard-tree,pr-review-\${{ env.S }},pr-review-sign/"
    mutant "adjacent quoting" R1 "${A}--sections 'guard-tree,pr-review-''shadow,pr-review-sign/"
    mutant "quote splicing after a space" R4 "${A}--sections 'guard-tree, ''pr-review-shadow'\\'',pr-review-sign/"
    mutant "unquoted prefix splicing a quoted suffix with a space" R4 "${A}--sections guard-tree,' pr-review-shadow'\\'',x/"
    mutant "brace expansion (last value wins)" R1 "${A}--sections={guard-tree,pr-review-shadow}/"
    mutant "glued backslash-newline inside the option name" R6 "${A}--sec\\\\\ntions pr-review-shadow --x 'guard-tree,pr-review-sign/"
    mutant "glued backslash-newline inside the value" R6 "${A}--sections guard-tree,pr-review-sha\\\\\ndow --x 'guard-tree,pr-review-sign/"
    mutant "negated bracket fnmatch vs bash" R2 "${A}--sections 'guard-tree,pr-review-[^s]hadow,pr-review-sign/"
    # bypass 1: the option name hidden by quoting, escaping or expansion
    mutant "option name + empty quotes (--sections\"\")" R1 "${A}--sections\"\" 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "option name quoted in the middle (--'sections')" R1 "${A}--'sections' 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "whole option quoted ('--sections')" R1 "${A}'--sections' 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "empty quotes inside the option (--sec''tions)" R1 "${A}--sec''tions 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "backslash inside the option (--sec\\tions)" R3 "${A}--sec\\\\tions 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "escaped equals (--sections\\=x)" R3 "${A}--sections\\\\=x 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "\${IFS} as the separator" R1 "${A}--sections\${IFS}pr-review-shadow --x 'guard-tree,pr-review-sign/"
    # bypass 2: POSIX bracket forms
    mutant "bracket collating [[.s]hadow" R2 "${A}--sections 'guard-tree,pr-review-[[.s]hadow,pr-review-sign/"
    mutant "bracket equivalence [[=s=]]hadow" R2 "${A}--sections 'guard-tree,pr-review-[[=s=]]hadow,pr-review-sign/"
    mutant "bracket class [[:alpha:]]hadow" R2 "${A}--sections 'guard-tree,pr-review-[[:alpha:]]hadow,pr-review-sign/"
    mutant "empty bracket []s]hadow" R2 "${A}--sections 'guard-tree,pr-review-[]s]hadow,pr-review-sign/"
    # bypass 3: YAML double-quoted escape
    mutant "YAML escape --\\x73ections" R3 "${A}--\\\\x73ections 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "unicode escape \\u0073" R3 "${A}--\\\\u0073ections 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "here-doc on the driver line" R7 "${A}--sections 'guard-tree,pr-review-sign' <<EOF #/"
    # code in a comment-looking line, with and without a here-doc
    printf '  - run: |\n      cat <<EOF\n      # python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n      EOF\n' > "$td/h.yml"
    if got="$(check_file "$td/h.yml" 0)"; then echo "FAIL  mutant: #-prefixed here-doc data line hid code"; fails=1
    elif printf '%s\n' "$got" | grep -qE '\[R10-'; then echo "ok    red   MUTANT [R10]: #-prefixed here-doc data line"; else echo "FAIL  wrong reason for the here-doc comment: $got"; fails=1; fi
    printf '  - run: |\n      cat <<EOF\n      python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n      EOF\n' > "$td/h2.yml"
    if got="$(check_file "$td/h2.yml" 0)"; then echo "FAIL  mutant: here-doc data line hid code"; fails=1
    elif printf '%s\n' "$got" | grep -qE '\[R5-'; then echo "ok    red   MUTANT [R5]: here-doc data line"; else echo "FAIL  wrong reason for the here-doc line: $got"; fails=1; fi
    printf '  - run: |\n      SECS="--sections pr-review-shadow"\n      python3 scripts/ci/fat_driver.py run $SECS\n' > "$td/o.yml"
    if got="$(check_file "$td/o.yml" 0)"; then echo "FAIL  mutant: orphan --sections hid code"; fails=1
    elif printf '%s\n' "$got" | grep -qE '\[R(11|1)-'; then echo "ok    red   MUTANT [R11|R1]: --sections kept in a variable"; else echo "FAIL  wrong reason for the orphan: $got"; fails=1; fi
    # a comment ending in a backslash must not hide the code on the next line
    printf '  - run: |\n      # merge queue: sign only \\\n      python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/c.yml"
    if got="$(check_file "$td/c.yml" 0)"; then echo "FAIL  mutant: comment ending in a backslash hid the next line"; fails=1
    elif printf '%s\n' "$got" | grep -qE '\[R5-'; then echo "ok    red   MUTANT [R5]: comment ending in a backslash does not swallow the next line"; else echo "FAIL  wrong reason: $got"; fails=1; fi
    # bypass 4: a workflow other than ci.yml
    mkdir "$td/w"; cp "$src" "$td/w/ci.yml"
    pw() { printf 'name: x\non:\n  %s:\njobs:\n  j:\n    steps:\n      - run: |\n          %s\n' "$1" "$2" > "$td/w/new.yml"; }
    pw pull_request "python3 scripts/ci/fat_driver.py run --sections 'guard-tree,pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a new on:pull_request workflow with the shadow stayed green"; fails=1
    elif printf '%s\n' "$got" | grep -q '(new.yml).*\[R5-'; then echo "ok    red   MUTANT [R5]: new pull_request workflow names the shadow"; else echo "FAIL  wrong reason for new.yml: $got"; fails=1; fi
    pw merge_group "python3 scripts/ci/fat_driver.py run --sec''tions 'pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a new merge_group workflow with a hidden option stayed green"; fails=1
    elif printf '%s\n' "$got" | grep -q '(new.yml).*\[R1-'; then echo "ok    red   MUTANT [R1]: new merge_group workflow hides the option"; else echo "FAIL  wrong reason for new.yml: $got"; fails=1; fi
    pw workflow_dispatch "python3 scripts/ci/fat_driver.py run --sections 'pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "ok    green control: a workflow_dispatch-only LAB workflow may carry the shadow"; else echo "FAIL  LAB workflow flagged: $got"; fails=1; fi
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test "${2:-.github/workflows/ci.yml}"; exit $? ;;
    *)
        if [ -n "${1:-}" ]; then check_file "$1" 1 && echo "ok    no --sections list resolves to pr-review-shadow"
        else check ".github/workflows/ci.yml" && echo "ok    no --sections list resolves to pr-review-shadow"; fi ;;
esac
