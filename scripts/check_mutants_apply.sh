#!/usr/bin/env bash
# check_mutants_apply.sh -- every release/rehearsal mutant row must still CHANGE its target.
#
# WHY (aprender#6824, APR-RELEASE-TRAIN T2, shift-left)
# ----------------------------------------------------
# A mutant row is a sed expression that breaks a rule in a COPY of a script; the
# script's --selftest must then go red. When the code the expression quotes is
# edited, the pattern stops matching, the copy equals the original, and the row
# mutates nothing. That surfaced only in the release-time `--mutants` run (>580 s):
# "changed nothing: its pattern no longer matches". `cascade_stage_as_text`
# (scripts/release/rehearse.sh) went stale when #6808 added `stage_t2` to the
# dispatch line it quotes, and release-rehearsal-nightly sat red until #6819.
#
# WHAT: apply every row of every `done <<'MUT...'` table under scripts/release/ to
# a copy of its target and fail if the copy is unchanged. O(#rows) string edits:
# no build, no self-test run. The table format comes from each table's own `read`
# line; a format this guard does not know FAILS CLOSED (a guard that skips what it
# cannot read reports a pass it did not measure).
#
# It prints how many files, tables and rows it measured, and fails on zero tables:
# "0 stale over 0 rows" is the vacuous pass this fleet keeps rediscovering.
#
# Formats (the `read -r` variable list names it):
#   IFS='|' read -r m e                  name|sed-expr          target = the script itself
#   read -r name expr                    name sed-expr          target = the script itself
#   read -r name expr  + `m:${name}`     marker-scoped sed -E   target = the script itself
#   read -r name file expr               name file sed-expr     target = scripts/release/<file>
#   IFS='~' read -r label must old new   label~must~old~new     old must occur once in the code
#                                                               above the `if [ "$SELF_TEST" = 1 ]` line
#
# Usage
#   bash scripts/check_mutants_apply.sh [ROOT]     measure ROOT/scripts/release (default: this checkout)
#   bash scripts/check_mutants_apply.sh --self-test
set -uo pipefail

REL=scripts/release
SELF_TEST_ANCHOR=$'\nif [ "$SELF_TEST" = 1 ]; then'

# row_applies FMT FILE RELDIR ROW -> 0 if the row changes its target; prints why not.
row_applies() {
    local fmt=$1 file=$2 reldir=$3 row=$4 tmp rc=0 name expr tgt a b
    tmp=$(mktemp) || return 2
    case $fmt in
        pipe)
            name=${row%%|*}; expr=${row#*|}
            sed -e "$expr" "$file" > "$tmp" 2> /dev/null || rc=$? ;;
        ws2)
            read -r name expr <<< "$row"
            sed -e "$expr" "$file" > "$tmp" 2> /dev/null || rc=$? ;;
        marker)
            read -r name expr <<< "$row"
            sed -E "/# (m:[a-zA-Z-]+ )*m:${name}( |\$)/${expr}" "$file" > "$tmp" 2> /dev/null || rc=$? ;;
        ws3)
            read -r name tgt expr <<< "$row"
            [ -f "$reldir/$tgt" ] || { echo "target $tgt is missing"; rm -f "$tmp"; return 1; }
            file=$reldir/$tgt
            sed -e "$expr" "$file" > "$tmp" 2> /dev/null || rc=$? ;;
        tilde)
            IFS='~' read -r name a b expr <<< "$row"   # name must old new -> a=must b=old expr=new
            local code cnt rest
            code=$(cat "$file"; printf x); code=${code%x}
            code=${code%%"$SELF_TEST_ANCHOR"*}
            rest=${code//"$b"/}
            cnt=$(( (${#code} - ${#rest}) / (${#b} == 0 ? 1 : ${#b}) ))
            rm -f "$tmp"
            [ "$cnt" = 1 ] && return 0
            echo "its anchor occurs $cnt times in the code above the self-test (need exactly 1)"; return 1 ;;
        *) rm -f "$tmp"; echo "unknown table format '$fmt'"; return 1 ;;
    esac
    if [ "$rc" != 0 ]; then rm -f "$tmp"; echo "sed rejected its expression (rc $rc)"; return 1; fi
    if cmp -s "$file" "$tmp"; then rm -f "$tmp"; echo "changed nothing: its pattern no longer matches"; return 1; fi
    rm -f "$tmp"; return 0
}

# table_fmt WHILE_LINE BODY -> prints the format name, or nothing if unknown
table_fmt() {
    local w=$1 body=$2 vars
    vars=$(sed -n 's/.*read -r \([a-z ]*\); do.*/\1/p' <<< "$w")
    case "$w" in
        *"IFS='|'"*) [ "$vars" = "m e" ] && echo pipe ;;
        *"IFS='~'"*) [ "$vars" = "label must old new" ] && echo tilde ;;
        *)  case "$vars" in
                "name expr") if grep -qF 'm:${name}' <<< "$body"; then echo marker; else echo ws2; fi ;;
                "name file expr") echo ws3 ;;
            esac ;;
    esac
}

# measure ROOT -> prints findings; sets globals N_FILES N_TABLES N_ROWS N_BAD
measure() {
    local root=$1 reldir f lines i n tag w body fmt row r why
    reldir=$root/$REL
    N_FILES=0 N_TABLES=0 N_ROWS=0 N_BAD=0
    [ -d "$reldir" ] || { echo "FAIL  no $REL under $root"; N_BAD=1; return; }
    for f in "$reldir"/*.sh; do
        [ -f "$f" ] || continue
        grep -qE "done <<'MUT[A-Z]*'" "$f" || continue
        N_FILES=$((N_FILES + 1))
        mapfile -t lines < "$f"; n=${#lines[@]}
        for ((i = 0; i < n; i++)); do
            [[ ${lines[i]} =~ done\ \<\<\'(MUT[A-Z]*)\' ]] || continue
            tag=${BASH_REMATCH[1]}; N_TABLES=$((N_TABLES + 1))
            w=""; body=""; wr=-1
            for ((r = i; r >= 0 && r >= i - 60; r--)); do
                case ${lines[r]} in *"read -r"*) case ${lines[r]} in *wh"ile "*) w=${lines[r]}; wr=$r; break ;; esac ;; esac
            done
            [ "$wr" -ge 0 ] && body=$(printf '%s\n' "${lines[@]:$wr:$((i - wr + 1))}")
            fmt=$(table_fmt "$w" "$body")
            if [ -z "$fmt" ]; then
                echo "FAIL  $(basename "$f") line $((i + 1)): unknown mutant-table format (read line: ${w:-none found})"
                N_BAD=$((N_BAD + 1)); continue
            fi
            r=$((i + 1))
            while [ "$r" -lt "$n" ] && [ "${lines[r]}" != "$tag" ]; do
                row=${lines[r]}; r=$((r + 1))
                [ -n "${row//[[:space:]]/}" ] || continue
                [[ $row =~ ^[[:space:]]*# ]] && continue
                N_ROWS=$((N_ROWS + 1))
                if ! why=$(row_applies "$fmt" "$f" "$reldir" "$row"); then
                    N_BAD=$((N_BAD + 1))
                    echo "STALE $(basename "$f") $(awk '{sub(/[|~].*/,""); print $1}' <<< "$row"): $why"
                fi
            done
        done
    done
}

run() {
    local root=$1
    measure "$root"
    echo "check_mutants_apply: $N_FILES file(s), $N_TABLES table(s), $N_ROWS row(s) measured, $N_BAD bad"
    [ "$N_TABLES" -gt 0 ] || { echo "FAIL  measured 0 tables: a pass over nothing proves nothing"; return 1; }
    [ "$N_BAD" = 0 ]
}

self_test() {
    local d fails=0 out rc
    d=$(mktemp -d) || exit 2; [ -n "$d" ] || exit 2; trap 'rm -rf -- "${d:?}"' RETURN
    # case NAME WANT_RC WANT_TEXT : the fixture tree is already in $d/$NAME
    check() {
        local name=$1 want=$2 text=$3
        out=$(run "$d/$name" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && { [ -z "$text" ] || grep -qF -- "$text" <<< "$out"; }; then
            echo "ok    $name"
        else
            echo "FAIL  $name: rc=$rc want=$want text='$text'"; printf '%s\n' "$out" | sed 's/^/        /'; fails=$((fails + 1))
        fi
    }
    mk() { mkdir -p "$d/$1/$REL"; }
    # --- every format, applying
    mk good; cat > "$d/good/$REL/a.sh" <<'EOF'
LIMIT=1
x() { :; }
while IFS='|' read -r m e; do :; done <<'MUT'
one|s/^LIMIT=1/LIMIT=2/
MUT
while read -r name expr; do :; done <<'MUTANTS'
two /^LIMIT=1/s/1/3/
MUTANTS
EOF
    cat > "$d/good/$REL/b.sh" <<'EOF'
echo hi
while read -r name file expr; do :; done <<'MUTANTS'
three   a.sh   s/^LIMIT=1/LIMIT=4/
MUTANTS
EOF
    cat > "$d/good/$REL/c.sh" <<'EOF'
echo code ANCHOR here
if [ "$SELF_TEST" = 1 ]; then
    while IFS='~' read -r label must old new; do :; done <<'MUT'
four~row~echo code ANCHOR here~echo code
MUT
fi
EOF
    cat > "$d/good/$REL/d.sh" <<'EOF'
# m:five line
while read -r name expr; do sed -E "/# (m:[a-zA-Z-]+ )*m:${name}( |\$)/${expr}" f; done <<'MUTANTS'
five s/five/six/
MUTANTS
EOF
    check good 0 "4 file(s), 5 table(s), 5 row(s)"
    # --- one stale row per format: each must be named
    for v in pipe ws2 ws3 tilde marker; do
        mkdir -p "$d/stale-$v"; cp -r "$d/good/scripts" "$d/stale-$v/"
    done
    sed -i 's#^one|s/^LIMIT=1#one|s/^LIMIT=9#' "$d/stale-pipe/$REL/a.sh"
    sed -i 's#^two /^LIMIT=1/s/1/3/#two /^LIMIT=9/s/1/3/#' "$d/stale-ws2/$REL/a.sh"
    sed -i 's#^three   a.sh   s/^LIMIT=1#three   a.sh   s/^LIMIT=9#' "$d/stale-ws3/$REL/b.sh"
    sed -i 's#^four~row~echo code ANCHOR here#four~row~echo code GONE here#' "$d/stale-tilde/$REL/c.sh"
    sed -i 's#^five s/five/six/#five s/nomatch/six/#' "$d/stale-marker/$REL/d.sh"
    check stale-pipe 1 "STALE a.sh one:"
    check stale-ws2 1 "STALE a.sh two:"
    check stale-ws3 1 "STALE b.sh three:"
    check stale-tilde 1 "STALE c.sh four:"
    check stale-marker 1 "STALE d.sh five:"
    # --- fails closed
    mk unknown; cat > "$d/unknown/$REL/a.sh" <<'EOF'
while read -r a b c d; do :; done <<'MUT'
r1 r2 r3 r4
MUT
EOF
    check unknown 1 "unknown mutant-table format"
    mk notable; echo 'echo no tables' > "$d/notable/$REL/a.sh"
    check notable 1 "measured 0 tables"
    mkdir -p "$d/notree"; check notree 1 "no $REL"
    mk missing; cat > "$d/missing/$REL/a.sh" <<'EOF'
while read -r name file expr; do :; done <<'MUT'
r nope.sh s/a/b/
MUT
EOF
    check missing 1 "target nope.sh is missing"
    mk badsed; cat > "$d/badsed/$REL/a.sh" <<'EOF'
while read -r name expr; do :; done <<'MUT'
r s/a/b
MUT
EOF
    check badsed 1 "sed rejected"
    mk tilde-twice; cat > "$d/tilde-twice/$REL/a.sh" <<'EOF'
echo ANCHOR; echo ANCHOR
if [ "$SELF_TEST" = 1 ]; then
    while IFS='~' read -r label must old new; do :; done <<'MUT'
r~row~echo ANCHOR~echo
MUT
fi
EOF
    check tilde-twice 1 "occurs 2 times"
    # --- the real tree: green, then red when the rehearse.sh dispatch text is edited (#6808's shape)
    local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    if [ -d "$here/$REL" ]; then
        check_real() { out=$(run "$here" 2>&1); rc=$?; if [ "$rc" = 0 ]; then echo "ok    real tree is green"; else echo "FAIL  real tree: rc=$rc"; printf '%s\n' "$out" | sed 's/^/        /'; fails=$((fails + 1)); fi; }
        check_real
        mkdir -p "$d/real/scripts"; cp -r "$here/$REL" "$d/real/scripts/"
        sed -i 's/stage_summary|stage_cascade|stage_t2) "\$cmd" ;;/stage_summary|stage_cascade|stage_t3) "$cmd" ;;/' "$d/real/$REL/rehearse.sh"
        check real 1 "STALE rehearse.sh cascade_stage_as_text"
    fi
    [ "$fails" = 0 ] && { echo "self-test: all green"; return 0; }
    echo "self-test: $fails FAILED"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,36p' "$0" ;;
    *) root=${1:-$(cd "$(dirname "$0")/.." && pwd)}; run "$root" ;;
esac
