#!/usr/bin/env bash
# check_mutants_apply.sh -- every release/rehearsal mutant still APPLIES to the code it mutates.
#
# WHY (APR-RELEASE-TRAIN T2, shift-left)
# The release scripts under scripts/release/ carry mutant tables (`done <<'MUTANTS'`). Each row is a
# planted defect that the script's own case table must turn RED. A row whose pattern no longer
# matches the code mutates nothing, and the release-time `--mutants` run reports
# "changed nothing: its pattern no longer matches". That was only ever seen at release time:
# release-rehearsal-nightly was red four nights running (11-06..11-09) on `cascade_stage_as_text`
# after #6808 added `stage_t2` to the line the row quoted.
#
# This guard runs in pr-light, through guard_tree.sh's no-toolchain subset. It applies every row
# to a copy of its target and checks that the copy differs. That is O(#rows) string edits: no build
# and no self-test run. Whether the mutant is KILLED stays the release script's own --mutants job.
#
# FORMATS (derived from each table's own `read` line, never from a name list)
#   name expr             sed -e "$expr" on the script itself
#   name expr (m: tags)   sed -E "/# (m:..)*m:NAME( |$)/EXPR" on the script itself (anchored rows)
#   name file expr        sed -e "$expr" on scripts/release/<file>
#   label~must~old~new    <old> occurs exactly once in the code above the self-test block
# A table whose format is none of these is a FAIL (fail closed), never a skip.
#
# USAGE
#   bash scripts/check_mutants_apply.sh               check every table under scripts/release/
#   bash scripts/check_mutants_apply.sh --dir DIR     check DIR/*.sh instead (fixtures)
#   bash scripts/check_mutants_apply.sh --self-test   known-good and known-bad rows
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# tables FILE -> one line per mutant table in FILE: "<start-line> <terminator>"
tables() {
    awk 'match($0, /done <<-?\047?(MUT[A-Z_]*)\047?[[:space:]]*$/) {
        t = substr($0, RSTART); sub(/done <<-?\047?/, "", t); sub(/\047?[[:space:]]*$/, "", t); print NR, t }' "$1"
}

# fmt FILE LINE -> the table format, read from the loop above the table's `done`
fmt() {
    local f="$1" n="$2" body
    body="$(awk -v n="$n" 'NR <= n' "$f" | tac | awk '/while .*read -r/ { print; exit }')"
    case "$body" in
        *"IFS='~' read -r label must old new"*) echo tilde ;;
        *"read -r name file expr"*) echo file3 ;;
        *"read -r name expr"*)
            if awk -v n="$n" 'NR <= n' "$f" | tail -n 40 | grep -qF 'm:${name}( |'; then echo anchored; else echo self2; fi ;;
        *) echo unknown ;;
    esac
}

# rows FILE LINE TERM -> the table body, exactly as the heredoc hands it to `read`
rows() {
    awk -v n="$1" -v t="$3" 'NR > n { if ($0 == t) exit; print }' "$2"
}

# check_file FILE -> prints one row per mutant; returns 1 if any did not apply
check_file() {
    local f="$1" dir n term kind bad=0 name rest file expr target label must old new tmp
    dir="$(dirname "$f")"
    tmp="$(mktemp)" || return 2
    while read -r n term; do
        kind="$(fmt "$f" "$n")"
        if [ "$kind" = unknown ]; then
            printf 'FAIL  %s:%s table %s: unknown row format (no recognised read line above it)\n' "${f#"$ROOT"/}" "$n" "$term"; bad=1; continue
        fi
        case "$kind" in
            tilde)
                while IFS='~' read -r label must old new; do
                    [ -n "$label" ] || continue
                    if [ "$(awk '/^if \[ "\$SELF_TEST" = 1 \]; then/ { exit } { print }' "$f" | grep -cF -- "$old")" = 1 ]; then
                        printf 'ok    %s %s\n' "${f##*/}" "$label"
                    else
                        printf 'FAIL  %s %s: its old text does not occur exactly once in the code\n' "${f##*/}" "$label"; bad=1
                    fi
                done < <(rows "$n" "$f" "$term") ;;
            *)
                while read -r name rest; do
                    [ -n "$name" ] || continue
                    case "$kind" in
                        # read -r splits on runs of IFS, so take the expr as the raw remainder: a sed
                        # expr may itself hold runs of spaces (step_table's istime rows do)
                        file3) file="${rest%%[[:space:]]*}"; expr="${rest#"$file"}"; expr="${expr#"${expr%%[![:space:]]*}"}"; target="$dir/$file" ;;
                        *) expr="$rest"; target="$f" ;;
                    esac
                    if [ ! -f "$target" ]; then
                        printf 'FAIL  %s %s: target %s does not exist\n' "${f##*/}" "$name" "$target"; bad=1; continue
                    fi
                    case "$kind" in
                        anchored) sed -E "/# (m:[a-zA-Z-]+ )*m:${name}( |\$)/${expr}" "$target" > "$tmp" 2>/dev/null ;;
                        *) sed -e "$expr" "$target" > "$tmp" 2>/dev/null ;;
                    esac
                    if cmp -s "$target" "$tmp"; then
                        printf 'FAIL  %s %s: changed nothing in %s (its pattern no longer matches)\n' "${f##*/}" "$name" "${target##*/}"; bad=1
                    else
                        printf 'ok    %s %s\n' "${f##*/}" "$name"
                    fi
                done < <(rows "$n" "$f" "$term") ;;
        esac
    done < <(tables "$f")
    rm -f -- "$tmp"
    return "$bad"
}

# run DIR -> checks every *.sh in DIR that holds a mutant table
run() {
    local d="$1" f bad=0 seen=0
    for f in "$d"/*.sh; do
        [ -f "$f" ] || continue
        [ -n "$(tables "$f")" ] || continue
        seen=$((seen + 1))
        check_file "$f" || bad=1
    done
    if [ "$seen" = 0 ]; then echo "FAIL  no mutant table found under $d (UNMEASURED)"; return 1; fi
    if [ "$bad" = 0 ]; then echo "PASS  every mutant in $seen table file(s) under ${d#"$ROOT"/} applies"; else echo "FAIL  a mutant no longer applies (see FAIL rows)"; fi
    return "$bad"
}

selftest() {
    local t rc out fail=0
    t="$(mktemp -d)" || return 2
    case "$t" in ?*/tmp.?*) : ;; *) echo "mktemp gave '$t', refusing"; return 2 ;; esac
    # shellcheck disable=SC2064
    trap "rm -rf -- '$t'" RETURN
    expect() { # expect NAME WANT_RC NEEDLE -- runs `run $t/NAME`
        rc=0; out="$(run "$t/$1" 2>&1)" || rc=$?
        if [ "$rc" != "$2" ]; then echo "  BROKE $1: rc $rc (want $2): $(printf '%s' "$out" | tail -n 3 | tr '\n' '|')"; fail=1; return; fi
        case "$out" in *"$3"*) echo "  ok    $1" ;; *) echo "  BROKE $1: output lacks '$3'"; fail=1 ;; esac
    }
    mkdir -p "$t/good" "$t/stale" "$t/file3" "$t/tilde" "$t/unknown" "$t/none" "$t/missing"
    # self2: a row that applies, and its stale twin
    printf '%s\n' 'LIMIT=86400' 'm() {' '    while read -r name expr; do :; done <<'"'"'MUTANTS'"'"'' \
        'limit_is_48h   /^LIMIT=86400/s/^LIMIT=86400/LIMIT=172800/' 'MUTANTS' '}' > "$t/good/a.sh"
    sed 's/^LIMIT=86400$/LIMIT=3600/' "$t/good/a.sh" > "$t/stale/a.sh"
    expect good 0 "PASS"
    expect stale 1 "a.sh limit_is_48h: changed nothing"
    # file3: rows against a sibling, one of which drifted
    printf '%s\n' 'case $cmd in stage_a|stage_b) "$cmd" ;; esac' > "$t/file3/lib.sh"
    printf '%s\n' 'm() {' '    while read -r name file expr; do :; done <<'"'"'MUTANTS'"'"'' \
        'b_as_text   lib.sh   /case \$cmd in stage_[a-z_|]*) "\$cmd" ;;/s/stage_a|//' \
        'old_text    lib.sh   s/stage_a|stage_b|stage_c) "\$cmd" ;;/stage_a) "$cmd" ;;/' 'MUTANTS' '}' > "$t/file3/rehearse.sh"
    expect file3 1 "rehearse.sh old_text: changed nothing in lib.sh"
    rc=0; out="$(run "$t/file3")" || rc=$?
    case "$out" in *"ok    rehearse.sh b_as_text"*) echo "  ok    file3 line-addressed row applies" ;; *) echo "  BROKE file3: b_as_text did not apply"; fail=1 ;; esac
    # tilde: old text must occur once in the code above the self-test block
    printf '%s\n' 'if [ "$u" != 0 ]; then echo red; fi' 'if [ "$SELF_TEST" = 1 ]; then' \
        "    while IFS='~' read -r label must old new; do :; done <<'MUT'" \
        'gone~is-red~if [ "$v" != 0 ]; then~if false; then' 'MUT' 'fi' > "$t/tilde/g.sh"
    expect tilde 1 "g.sh gone: its old text does not occur"
    # unknown format and no tables: both fail closed
    printf '%s\n' 'm() {' '    for x in a; do :; done <<'"'"'MUTANTS'"'"'' 'x s/a/b/' 'MUTANTS' '}' > "$t/unknown/u.sh"
    expect unknown 1 "unknown row format"
    printf 'echo hi\n' > "$t/none/n.sh"
    expect none 1 "no mutant table found"
    printf '%s\n' 'm() {' '    while read -r name file expr; do :; done <<'"'"'MUTANTS'"'"'' 'x gone.sh s/a/b/' 'MUTANTS' '}' > "$t/missing/r.sh"
    expect missing 1 "target"
    # the real tree, with the dispatch line the rehearsal rows quote edited: must turn RED (the brief's falsifier)
    mkdir -p "$t/real"
    cp -- "$ROOT"/scripts/release/*.sh "$t/real/"
    sed -i -e 's/case \$cmd in stage_summary|stage_cascade|stage_t2) "\$cmd" ;;/case $cmd in stage_summary|stage_t2|stage_kascade) "$cmd" ;;/' "$t/real/rehearse.sh"
    if cmp -s "$ROOT/scripts/release/rehearse.sh" "$t/real/rehearse.sh"; then
        echo "  BROKE real: the fixture edit changed nothing (rehearse.sh dispatch line moved)"; fail=1
    else
        expect real 1 "rehearse.sh cascade_stage_as_text: changed nothing"
    fi
    if [ "$fail" = 0 ]; then echo "PASS  check_mutants_apply self-test"; else echo "FAIL  check_mutants_apply self-test"; fi
    return "$fail"
}

case "${1:-}" in
    '') run "$ROOT/scripts/release" ;;
    --dir) [ -n "${2:-}" ] || { echo "usage: $0 --dir DIR" >&2; exit 2; }; run "$2" ;;
    --self-test) selftest ;;
    -h|--help) sed -n '2,26p' "$0" ;;
    *) echo "usage: $0 [--dir DIR | --self-test | --help]" >&2; exit 2 ;;
esac
