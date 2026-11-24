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
#   R12 unread   a workflow file that cannot be read, or a grep that exits >=2 (fail closed, never "no match")
# SCOPE. ci.yml (must hold >=1 list) and every other .github/workflows/*.yml that names the driver AND can run on the PR or
# merge-queue path (its `on:` block names pull_request, pull_request_target, merge_group or workflow_call, or has no `on:` block).
# A schedule/workflow_dispatch-only LAB workflow may carry the section.
#
# A backslash-newline inside the driver name (fa\<newline>t_driver) is joined before the trigger test (R6).
# Not covered, stated plainly: a driver reference that hides every `fat_d`/`ci/fat_` substring by quoting (`fa""t_driver`).
#
# Usage:  bash scripts/check_ci_no_pr_shadow.sh [--self-test] [WORKFLOW_FILE]
#         WORKFLOW_FILE given: only that file (>=1 list required). Default: ci.yml plus every PR-path workflow beside it.
# Wired:  PR and merge-queue path: guard-tree (scripts/guard_tree.sh, pr-light) runs [self-test] then [run]; it finds the
#         self-test through --help. Full tier: ci/explicit-test-commands.d/893-ci-no-pr-shadow.cmd (--self-test).
set -euo pipefail

TRIG='fat_d|ci/fat|fat_["'"'"'\\]'
PRE='^[[:space:]]*(bash scripts/ci/idle_watchdog\.sh "\$RUNNER_TEMP/fat" -- )?python3 scripts/ci/fat_driver\.py run --sections (.*)$'
TAIL="^( --results \"\\\$RUNNER_TEMP/[A-Za-z0-9_./-]+\")?( --background-until '[]A-Za-z0-9._?*@,[-]+')?\$"
NAMEF='^[[:space:]]*-?[[:space:]]*name: fat_driver case table must still turn RED$'
SELFF='^[[:space:]]*run: setsid --wait bash scripts/ci/fixture_quiet\.sh "fat_driver self-test" -- python3 scripts/ci/fat_driver\.py self-test$'
TEARF='^[[:space:]]*run: if \[ -f scripts/ci/fat_teardown\.sh \]; then bash scripts/ci/fat_teardown\.sh "\$GITHUB_RUN_ID" "\$RUNNER_TEMP"; fi$'
WAITF='^[[:space:]]*run: python3 scripts/ci/fat_driver\.py wait --results "\$RUNNER_TEMP/fat/results\.json"$'
N_LISTS=0

has() { # ERE FILE -> 0 match, 1 no match, 2 grep failed (callers treat 2 as RED, never as no match)
    local rc=0; LC_ALL=C grep -qE -- "$1" "$2" 2>/dev/null || rc=$?
    [ "$rc" -lt 2 ] || return 2
    return "$rc"
}
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
    stmt="$(printf '%s' "$stmt" | sed -E 's/\$\{\{ matrix\.(shard|shards) \}\}/@EXPR@/g')"   # only these two exact matrix expressions are pinned; any other ${{ }} stays and fails R4 (it can evaluate to commas and new parts)
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
    local IFS=,; set -f   # no pathname expansion: a part like * must stay a pattern, not become the repo-root file names
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
    local -a L=()
    if ! [ -f "$f" ] || ! [ -r "$f" ]; then echo "FAIL  [R12-unread] cannot read workflow file: $f"; return 1; fi
    if ! mapfile -t L < "$f"; then echo "FAIL  [R12-unread] mapfile failed on: $f"; return 1; fi
    total="${#L[@]}"; N_LISTS=0
    for ((i = 0; i < total; i++)); do
        line="${L[i]}"
        if ! [[ $line =~ $TRIG ]]; then
            if [[ $line =~ [^[:space:]]\\$ && ! $line =~ ^[[:space:]]*# ]]; then # backslash-newline splitting the driver name: join (bash drops both) and test again
                nxt="$line"; j=$i
                while [[ $nxt =~ [^[:space:]]\\$ ]] && ((j + 1 < total)); do nxt="${nxt%\\}$(trim "${L[j + 1]}")"; j=$((j + 1)); done
                if [[ $nxt =~ $TRIG ]]; then echo "FAIL  [R6-glued] line $((i + 1)): a backslash-newline splits the driver name: $line"; bad=1; continue; fi
            fi
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

pr_path() { # FILE -> 0 when the workflow can run on the PR / merge-queue / push path. STRICT ALLOWLIST: skipped (1) only if the file has exactly ONE on/true-ish key, it is the bare `on:`, and everything under it is `workflow_dispatch:` (any nested inputs), `schedule:` or plain `- cron: "<cron chars>"` items. Anything else is scanned.
    local n
    LC_ALL=C tr -d '\000' < "$1" | cmp -s - "$1" || return 0
    LC_ALL=C grep -q '[^[:print:][:space:]]' "$1" && return 0
    LC_ALL=C grep -q "$(printf '[\r\v\f]')" "$1" && return 0
    n="$(grep -cE '^[?"'"'"'[:space:]]*(on|true|"on"|'"'"'on'"'"')[[:space:]]*(:|$)' "$1" || true)"
    grep -qE '^[?"'"'"']' "$1" && return 0
    grep -qvE '^(name|run-name):[[:space:]]*[^"'"'"'&*!|>{[%@`#:]+$|^(on|jobs|permissions|concurrency|env|defaults):[[:space:]]*(#.*)?$' < <(grep -vE '^[[:space:]]|^#|^$' "$1") && return 0
    [ "$n" = 1 ] || return 0
    grep -qE '^on:[[:space:]]*(#.*)?$' "$1" || return 0
    grep -q BAD < <(awk 'BEGIN{b=-1} /^on:/{p=1;next} p&&/^[^-[:space:]#]/{p=0} p&&!/^[[:space:]]*(#.*)?$/{match($0,/^ */);i=RLENGTH;t=substr($0,i+1);if(b<0)b=i;if(i<b){print "BAD";next}if(i==b&&t!~/^-/){if(t ~ /^workflow_dispatch:[[:space:]]*(#.*)?$/)c="d";else if(t ~ /^schedule:[[:space:]]*(#.*)?$/)c="s";else print "BAD";next}if(c=="d"){if(t ~ /["\047\[{]/)print "BAD";next}if(c=="s"&&t ~ /^-[[:space:]]+cron:[[:space:]]*["\047]?[-0-9*\/, ]+["\047]?[[:space:]]*(#.*)?$/)next;print "BAD"}' "$1") && return 0
    return 1
}

check() { # FILE-or-DIR-anchor -> ci.yml (required) plus PR-path siblings
    local ci="$1" bad=0 d w out
    check_file "$ci" 1 || bad=1
    d="$(dirname "$ci")"
    for w in "$d"/*.yml "$d"/*.yaml; do
        [ -e "$w" ] || [ -L "$w" ] || continue   # an unmatched glob stays literal; a dangling link is -L and is RED below
        [ "$w" -ef "$ci" ] && continue
        if ! [ -f "$w" ] || ! [ -r "$w" ]; then echo "FAIL  ($(basename "$w")) [R12-unread] cannot read workflow file: $w"; bad=1; continue; fi
        pr_path "$w" || continue
        if out="$(check_file "$w" 0)"; then :; else printf "%s\n" "$out" | sed "s|^FAIL  |FAIL  ($(basename "$w")) |"; bad=1; fi
    done
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got rc
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: committed workflow is RED: $got"; fails=1; fi
    local anchor="--sections 'guard-tree,pr-review-sign"
    if ! grep -qF -- "$anchor" "$src"; then echo "FAIL  anchor missing from $src; self-test is vacuous"; return 1; fi
    mutant() { # LABEL RULES(ERE alternation of Rn) SED-EXPR -> red, and with one of those rule tags
        sed -e "$3" "$src" > "$td/ci.yml"
        if cmp -s "$src" "$td/ci.yml"; then echo "FAIL  mutant did not change the file: $1"; fails=1; return; fi
        if got="$(check "$td/ci.yml")"; then echo "FAIL  mutant stayed green: $1"; fails=1
        elif grep -qE "\[($2)-" <<<"$got"; then echo "ok    red   MUTANT [$2]: $1"
        else echo "FAIL  mutant red for the WRONG reason (wanted $2): $1 :: $(printf '%s' "$got" | head -n 2)"; fails=1; fi
    }
    local A="s/--sections 'guard-tree,pr-review-sign/"
    mutant "shadow put back in a list" R5 "${A}--sections 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "glob pr-review-* resolves to shadow" R5 "${A}--sections 'guard-tree,pr-review-*/"
    mutant "glob pr-review-s?a*" R5 "${A}--sections 'guard-tree,pr-review-s?a*/"
    mutant "glob * matches a repo-root file name (no pathname expansion in the guard)" R5 "${A}--sections '*/"
    mutant "glob [a-z]* with a comma list" R5 "${A}--sections 'guard-tree,[a-z]*/"
    mutant "value on the next line (folded block)" R5 "${A}--sections\n          'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "--sections=value form" R1 "${A}--sections='guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "indirect \$VAR value" R4 "${A}--sections \"\$PR_SECS\" --x 'guard-tree,pr-review-sign/"
    mutant "no --sections list is vacuous, not green" "R1|R8" "s/--sections/--other/"
    mutant "space after a comma" R4 "${A}--sections 'guard-tree, pr-review-shadow,pr-review-sign/"
    mutant "abbreviated option --section" R1 "${A}--section 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "abbreviated option --se" R1 "${A}--se 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "quote splicing" R1 "${A}--sections 'guard-tree,'pr-review-shadow,pr-review-sign/"
    mutant "whole-value expression" "R9|R4|R1" "${A}--sections '\${{ vars.PR_SECS }}' --x 'guard-tree,pr-review-sign/"
    mutant "partial expression" "R9|R4" "${A}--sections 'guard-tree,pr-review-\${{ env.S }},pr-review-sign/"
    mutant "shard expression that evaluates to a comma list" "R9|R4" "${A}--sections 'workspace-test-shard?\${{ 'x,pr-review-shadow,' }}?\${{ matrix.shards }}?/"
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
    mutant "here-doc on the driver line" R7 "${A}--sections 'guard-tree,pr-review-sign' <<EOF/"
    # code in a comment-looking line, with and without a here-doc
    printf '  - run: |\n      cat <<EOF\n      # python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n      EOF\n' > "$td/h.yml"
    if got="$(check_file "$td/h.yml" 0)"; then echo "FAIL  mutant: #-prefixed here-doc data line hid code"; fails=1
    elif grep -qE '\[R10-' <<<"$got"; then echo "ok    red   MUTANT [R10]: #-prefixed here-doc data line"; else echo "FAIL  wrong reason for the here-doc comment: $got"; fails=1; fi
    printf '  - run: |\n      cat <<EOF\n      python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n      EOF\n' > "$td/h2.yml"
    if got="$(check_file "$td/h2.yml" 0)"; then echo "FAIL  mutant: here-doc data line hid code"; fails=1
    elif grep -qE '\[R5-' <<<"$got"; then echo "ok    red   MUTANT [R5]: here-doc data line"; else echo "FAIL  wrong reason for the here-doc line: $got"; fails=1; fi
    printf '  - run: |\n      SECS="--sections pr-review-shadow"\n      python3 scripts/ci/fat_driver.py run $SECS\n' > "$td/o.yml"
    if got="$(check_file "$td/o.yml" 0)"; then echo "FAIL  mutant: orphan --sections hid code"; fails=1
    elif grep -qE '\[R(11|1)-' <<<"$got"; then echo "ok    red   MUTANT [R11|R1]: --sections kept in a variable"; else echo "FAIL  wrong reason for the orphan: $got"; fails=1; fi
    # a comment ending in a backslash must not hide the code on the next line
    printf '  - run: |\n      # merge queue: sign only \\\n      python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/c.yml"
    if got="$(check_file "$td/c.yml" 0)"; then echo "FAIL  mutant: comment ending in a backslash hid the next line"; fails=1
    elif grep -qE '\[R5-' <<<"$got"; then echo "ok    red   MUTANT [R5]: comment ending in a backslash does not swallow the next line"; else echo "FAIL  wrong reason: $got"; fails=1; fi
    # bypass 4: a workflow other than ci.yml
    mkdir "$td/w"; cp "$src" "$td/w/ci.yml"
    pw() { printf 'name: x\non:\n  %s:\njobs:\n  j:\n    steps:\n      - run: |\n          %s\n' "$1" "$2" > "$td/w/new.yml"; }
    pw pull_request "python3 scripts/ci/fat_driver.py run --sections 'guard-tree,pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a new on:pull_request workflow with the shadow stayed green"; fails=1
    elif grep -q '(new.yml).*\[R5-' <<<"$got"; then echo "ok    red   MUTANT [R5]: new pull_request workflow names the shadow"; else echo "FAIL  wrong reason for new.yml: $got"; fails=1; fi
    pw merge_group "python3 scripts/ci/fat_driver.py run --sec''tions 'pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a new merge_group workflow with a hidden option stayed green"; fails=1
    elif grep -q '(new.yml).*\[R1-' <<<"$got"; then echo "ok    red   MUTANT [R1]: new merge_group workflow hides the option"; else echo "FAIL  wrong reason for new.yml: $got"; fails=1; fi
    pw push "python3 scripts/ci/fat_driver.py run --sections 'pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a new on:push workflow with the shadow stayed green"; fails=1
    elif grep -q '(new.yml).*\[R5-' <<<"$got"; then echo "ok    red   MUTANT [R5]: new push workflow names the shadow"; else echo "FAIL  wrong reason for new.yml: $got"; fails=1; fi
    pw workflow_dispatch "python3 scripts/ci/fat_driver.py run --sections 'pr-review-shadow'"
    if got="$(check "$td/w/ci.yml")"; then echo "ok    green control: a workflow_dispatch-only LAB workflow may carry the shadow"; else echo "FAIL  LAB workflow flagged: $got"; fails=1; fi
    printf 'name: x\non:\n- pull_request\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/w/new.yml"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: an on: sequence at column 0 skipped the workflow"; fails=1; else echo "ok    red   MUTANT: on: block as a column-0 sequence is still scanned"; fi
    printf 'x: &t [pull_request]\nname: x\non: *t\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/w/new.yml"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: an aliased on: skipped the workflow"; fails=1; else echo "ok    red   MUTANT: aliased on: (*t) is still scanned"; fi
    printf 'name: x\non: {workflow_dispatch: {},\n  schedule: [{cron: "0 0 * * *"}], pull_request: {}}\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/w/new.yml"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: cron on the same line as pull_request hid the trigger"; fails=1; else echo "ok    red   MUTANT: flow-mapping cron beside pull_request is still scanned"; fi
    printf 'name: x\non:\n  workflow_dispatch:\n  "\\u0070ull_request":\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/w/new.yml"
    if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: a YAML-escaped trigger key hid pull_request"; fails=1; else echo "ok    red   MUTANT: YAML-escaped trigger key is still scanned"; fi
    for case_ in 'flow on: continued at column 0|name: x\non: {workflow_dispatch: {},\npull_request: {}}\n' 'decoy true: plus spaced on :|name: x\non : pull_request\ntrue: workflow_dispatch\n' 'complex key ? on plus decoy true:|name: x\n? on\n: pull_request\ntrue: workflow_dispatch\n' 'trigger nested under schedule list|name: x\non:\n  schedule:\n    - cron: "0 0 * * *"\n    - pull_request\n'; do
        lbl="${case_%%|*}"; body="${case_#*|}"
        printf '%b' "${body}jobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n" > "$td/w/new.yml"
        if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: $lbl skipped the workflow"; fails=1; else echo "ok    red   MUTANT: $lbl is still scanned"; fi
    done
    for case_ in 'quoted-scalar decoy plus anchored real on|name: "x\non:\n  workflow_dispatch:\nz"\n&a on:\n  pull_request:\n' 'tagged real on key|name: x\non:\n  workflow_dispatch:\n!!str on:\n  pull_request:\n'; do
        lbl="${case_%%|*}"; body="${case_#*|}"
        printf '%b' "${body}jobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n" > "$td/w/new.yml"
        if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: $lbl skipped the workflow"; fails=1; else echo "ok    red   MUTANT: $lbl is still scanned"; fi
    done
    for case_ in 'bare CR inside a comment|name: x\non:\n  workflow_dispatch:\n  # c\r  pull_request:\n' 'vertical tab line break|name: x\non:\n  workflow_dispatch:\n  # c\v  pull_request:\n'; do
        lbl="${case_%%|*}"; body="${case_#*|}"
        printf '%b' "${body}jobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n" > "$td/w/new.yml"
        if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: $lbl skipped the workflow"; fails=1; else echo "ok    red   MUTANT: $lbl is still scanned"; fi
    done
    for case_ in 'quoted multi-line scalar hides jobs: under workflow_dispatch|name: x\non:\n  workflow_dispatch:\n    inputs:\n      x:\n        default: "hello\njobs:\n        world"\n  pull_request:\n'; do
        lbl="${case_%%|*}"; body="${case_#*|}"
        printf '%b' "${body}jobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n" > "$td/w/new.yml"
        if got="$(check "$td/w/ci.yml")"; then echo "FAIL  mutant: $lbl skipped the workflow"; fails=1; else echo "ok    red   MUTANT: $lbl is still scanned"; fi
    done
    # R8 and R11 in isolation: the rule must be the ONLY thing that can turn these red
    printf 'name: x\njobs:\n  j:\n    steps:\n      - run: echo hi\n' > "$td/v.yml"
    if got="$(check_file "$td/v.yml" 1)"; then echo "FAIL  mutant: a workflow with no list stayed green (R8)"; fails=1
    elif [ "$(grep -c '^FAIL' <<<"$got")" = 1 ] && grep -qE '\[R8-' <<<"$got"; then echo "ok    red   MUTANT [R8]: no accepted list, nothing else wrong"; else echo "FAIL  R8 not isolated: $got"; fails=1; fi
    if got="$(check_file "$td/v.yml" 0)"; then echo "ok    green control: need=0 does not demand a list"; else echo "FAIL  need=0 demanded a list: $got"; fails=1; fi
    printf '  - run: |\n      SECS="--sections pr-review-shadow"\n' > "$td/r11.yml"
    if got="$(check_file "$td/r11.yml" 0)"; then echo "FAIL  mutant: a lone --sections variable stayed green (R11)"; fails=1
    elif [ "$(grep -c '^FAIL' <<<"$got")" = 1 ] && grep -qE '\[R11-' <<<"$got"; then echo "ok    red   MUTANT [R11]: orphan --sections, nothing else wrong"; else echo "FAIL  R11 not isolated: $got"; fails=1; fi
    # R12: unreadable or missing files fail closed
    if got="$(check_file "$td/none.yml" 1)"; then echo "FAIL  mutant: a missing file stayed green"; fails=1
    elif grep -qE '\[R12-' <<<"$got" && ! grep -q 'unbound' <<<"$got"; then echo "ok    red   MUTANT [R12]: missing file is named, not an unbound variable"; else echo "FAIL  missing file: $got"; fails=1; fi
    rc=0; has x "$td/none.yml" 2>/dev/null || rc=$?
    if [ "$rc" = 2 ]; then echo "ok    grep failure surfaces as exit 2 (not as no-match)"; else echo "FAIL  has() on a missing file returned $rc, want 2"; fails=1; fi
    mkdir "$td/u"; cp "$src" "$td/u/ci.yml"; ln -s "$td/u/gone.yml" "$td/u/dangling.yml"
    if got="$(check "$td/u/ci.yml")"; then echo "FAIL  mutant: a dangling workflow link stayed green"; fails=1
    elif grep -qE '\[R12-' <<<"$got"; then echo "ok    red   MUTANT [R12]: dangling workflow link"; else echo "FAIL  dangling link wrong reason: $got"; fails=1; fi
    rm -f "${td:?}/u/dangling.yml"
    printf 'name: x\non:\n  pull_request:\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/u/bad.yml"; chmod 000 "$td/u/bad.yml"
    if [ -r "$td/u/bad.yml" ]; then echo "skip  chmod 000 mutant (reader can still read it, e.g. root)"
    elif got="$(check "$td/u/ci.yml")"; then echo "FAIL  mutant: an unreadable shadow workflow stayed green"; fails=1
    elif grep -qE '\[R12-' <<<"$got"; then echo "ok    red   MUTANT [R12]: chmod 000 workflow"; else echo "FAIL  chmod 000 wrong reason: $got"; fails=1; fi
    chmod 600 "$td/u/bad.yml"; rm -f "${td:?}/u/bad.yml"
    # backslash-newline inside the driver name (fa\<newline>t_driver)
    printf '  - run: |\n      python3 scripts/ci/fa\\\nt_driver.py run --sections pr-review-shadow\n' > "$td/bn.yml"
    if got="$(check_file "$td/bn.yml" 0)"; then echo "FAIL  mutant: backslash-newline in the driver name hid the line"; fails=1
    elif grep -qE '\[R6-' <<<"$got"; then echo "ok    red   MUTANT [R6]: backslash-newline inside the driver name"; else echo "FAIL  wrong reason: $got"; fails=1; fi
    printf '  - run: |\n      echo a\\\n      echo b\n' > "$td/bn2.yml"
    if got="$(check_file "$td/bn2.yml" 0)"; then echo "ok    green control: an unrelated backslash continuation"; else echo "FAIL  unrelated continuation flagged: $got"; fails=1; fi
    # a comment ending in a backslash and a ' \' shell continuation are not a glued split (green controls)
    printf '  - run: |\n      # note ends in a backslash\\\n      echo hi\n' > "$td/cm.yml"
    if got="$(check_file "$td/cm.yml" 0)"; then echo "ok    green control: comment ending in a backslash"; else echo "FAIL  comment backslash flagged: $got"; fails=1; fi
    printf '  - run: |\n      echo a \\\n        b\n' > "$td/sp.yml"
    if got="$(check_file "$td/sp.yml" 0)"; then echo "ok    green control: space-backslash continuation"; else echo "FAIL  space-backslash flagged: $got"; fails=1; fi
    # sibling PR-path workflow with a driver name split by backslash-newline must be RED through check()
    mkdir "$td/s"; cp "$src" "$td/s/ci.yml"
    printf 'name: x\non:\n  pull_request:\njobs:\n  j:\n    steps:\n      - run: |\n          python3 scripts/ci/fa\\\nt_driver.py run --sections pr-review-shadow\n' > "$td/s/sib.yml"
    if got="$(check "$td/s/ci.yml")"; then echo "FAIL  mutant: split driver name in a sibling stayed green"; fails=1
    elif grep -qE '\(sib.yml\).*\[R6-' <<<"$got"; then echo "ok    red   MUTANT [R6]: split driver name in a sibling workflow"; else echo "FAIL  sibling wrong reason: $got"; fails=1; fi
    return "$fails"
}

case "${1:-}" in
    -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test "${2:-.github/workflows/ci.yml}"; exit $? ;;
    *)
        if [ -n "${1:-}" ]; then check_file "$1" 1 && echo "ok    no --sections list resolves to pr-review-shadow"
        else check ".github/workflows/ci.yml" && echo "ok    no --sections list resolves to pr-review-shadow"; fi ;;
esac
