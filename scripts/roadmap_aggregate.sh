#!/usr/bin/env bash
# roadmap_aggregate.sh — the ONE writer of docs/roadmaps/roadmap.yaml (T21, the RQ-8
# one-writer pattern; contract contracts/apr-roadmap-one-writer-v1.yaml).
#
# WHY. #3297 made a ticket write docs/roadmaps/entries/<ID>.yaml so that two PRs touch
# two different files. But every PR ALSO committed the regenerated aggregate
# roadmap.yaml, so the shared file came straight back: measured 2026-10-04 on main
# 316dee2cd4, 11 of 21 open PRs carried hand-merged copies (239 new ids, 4 edited, every
# one already shipping its fragment), roadmap.yaml was a conflicted file in 9 of them, and
# main took 98 roadmap.yaml commits in 30 days (96 adding an entry). Under one writer a PR
# commits ONLY its fragment. CI regenerates roadmap.yaml from the PR's own tree before
# any reader runs; a nightly writer on main commits the aggregate through one PR; the
# release preflight refuses a tag whose committed aggregate is stale.
#
# THE FUNCTION (byte-for-byte the one scripts/lib/roadmap_fragments.py `aggregate`
# computes, so the two cannot drift; --selftest proves it on the real tree when python3
# is present, and never needs it otherwise):
#   aggregate(base, fragments) = base's preamble + base's entries, where every fragment
#   SUPERSEDES the base entry with its id (drop), then is inserted, in filename order,
#   before the first entry of its own id prefix whose number is greater (else appended;
#   a legacy id with no PREFIX-NUMBER shape is appended). Idempotent by construction.
# PARITY, NOT POLICY (fork (i), 2026-10-04): it refuses EXACTLY what the python refuses (a *.yaml it cannot open
# as UTF-8 text) and reads like python (id = filename stem, universal newlines, text pasted verbatim, symlinks
# followed). A refusal only bash made would pass a PR in shadow and then wedge the nightly writer and every tag.
#
#   bash scripts/roadmap_aggregate.sh --print     # the aggregate on stdout
#   bash scripts/roadmap_aggregate.sh --write     # rewrite roadmap.yaml (CI, nightly writer)
#   bash scripts/roadmap_aggregate.sh --check     # committed == aggregate, and idempotent (tag)
#   bash scripts/roadmap_aggregate.sh --roundtrip # D4 measurement: split every entry, rebuild
#   bash scripts/roadmap_aggregate.sh --selftest  # case table
#   bash scripts/roadmap_aggregate.sh --mutants   # each planted mutant must turn a row RED
#   options: --roadmap FILE  --entries DIR
#
# EXIT 0 ok · 1 a rule refused (named) · 2 usage / the box cannot answer. 2 is not a pass.
set -uo pipefail

PROG=${0##*/}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$PROG"
ROADMAP="$ROOT/docs/roadmaps/roadmap.yaml"
ENTRIES="$ROOT/docs/roadmaps/entries"
# Mutation seam: production never sets it. --mutants sets each value and requires a RED row.
MUT="${ROADMAP_AGG_MUTANT:-}"
# -H: a symlinked entries DIR is followed, as python's os.listdir follows it (round-3 M; mutant nodirlink)
FH=-H; [ "$MUT" = nodirlink ] && FH=-P

die2() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }

# fragments <dir> -> one path per line, filename order (LC_ALL=C = python's sorted(str)): EVERY *.yaml entry, of
# any type, exactly as python's os.listdir sees it. aggregate() then reads it the way python's open() would.
fragments() {
    local d=$1
    [ -d "$d" ] || return 0
    find "$FH" "$d" -maxdepth 1 -mindepth 1 -name '*.yaml' -print | LC_ALL=C sort
}

# norm <in> <out>: python's universal-newline read (open(encoding="utf-8"), newline=None): CRLF -> LF, lone CR -> LF.
# A missing final newline stays missing. Prints the last byte of the result: n newline, x other, e empty.
norm() {
    local b
    if [ "$MUT" = nocr ]; then cp -- "$1" "$2" || return 2
    else LC_ALL=C sed -z -e 's/\r\n/\n/g' -e 's/\r/\n/g' -- "$1" > "$2" || return 2; fi
    [ -s "$2" ] || { printf 'e'; return 0; }
    b=$(tail -c 1 -- "$2" | od -An -tx1 | tr -d " \n")
    if [ "$b" = 0a ]; then printf 'n'; else printf 'x'; fi
}

# aggregate <base> <entries-dir> -> stdout. rc 1 on a fragment the existing generator also cannot read, 2 when this
# box cannot answer.
# PARITY, NOT POLICY (fork (i), 2026-10-04): this refuses EXACTLY what scripts/lib/roadmap_fragments.py refuses, so
# moving the aggregate onto bash adds no verdict and a fragment can never pass a PR and then wedge the writer or the
# tag. python takes a fragment's id from its FILENAME, pastes its text verbatim, follows symlinks, and fails only on
# a *.yaml it cannot open as a UTF-8 file. So does this. Shape rules on a fragment (one `- id:` line, id ==
# filename, ...) belong to a NEW check with its own admission, not to the generator.
aggregate() {
    local base=$1 dir=$2 list rc nd lf f k fl bl
    [ -r "$base" ] || { printf '%s: ENV - cannot read %s\n' "$PROG" "$base" >&2; return 2; }
    command -v iconv > /dev/null 2>&1 || { printf '%s: ENV - no iconv to prove fragments are UTF-8\n' "$PROG" >&2; return 2; }
    list=$(fragments "$dir"); rc=$?
    [ "$rc" = 0 ] || return 2
    # A filename holding a newline splits this list: this box cannot answer (2, never a pass, never a refusal).
    if [ -d "$dir" ] && [ "$(find "$FH" "$dir" -maxdepth 1 -mindepth 1 -name '*.yaml' -print0 | tr -cd '\0' | wc -c)" != "$(grep -c -e . <<< "$list")" ]; then
        printf '%s: ENV - a fragment filename holds a newline; this aggregator cannot answer\n' "$PROG" >&2; return 2
    fi
    nd=$(mktemp -d) || return 2
    lf="$nd/list"; : > "$lf"
    if ! iconv -f UTF-8 -t UTF-8 -- "$base" > /dev/null 2>&1; then
        printf '%s: FAIL  %s is not valid UTF-8\n' "$PROG" "$base" >&2; rm -rf -- "${nd:?}"; return 1
    fi
    bl=$(norm "$base" "$nd/base") || { rm -rf -- "${nd:?}"; return 2; }
    k=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        k=$((k + 1))
        # python open() follows a symlink and fails on a directory, a dangling link or an unreadable file: rc 1 there too.
        if { [ "$MUT" = nofollow ] && [ -L "$f" ]; } || [ ! -f "$f" ] || [ ! -r "$f" ]; then
            printf '%s: FAIL  fragment %s cannot be opened as a file (the existing generator fails on it too)\n' "$PROG" "${f##*/}" >&2
            rm -rf -- "${nd:?}"; return 1
        fi
        if [ "$MUT" != noutf8 ] && ! iconv -f UTF-8 -t UTF-8 -- "$f" > /dev/null 2>&1; then
            printf '%s: FAIL  fragment %s is not valid UTF-8\n' "$PROG" "${f##*/}" >&2; rm -rf -- "${nd:?}"; return 1
        fi
        fl=$(norm "$f" "$nd/$k") || { rm -rf -- "${nd:?}"; return 2; }
        printf '%s%s\n' "$fl" "$f" >> "$lf"
    done <<< "$list"
    LC_ALL=C awk -v MUT="$MUT" -v LISTF="$lf" -v ND="$nd" -v BASEL="$bl" '
function trim(s) { sub(/^[ \t\r\n\v\f]+/, "", s); sub(/[ \t\r\n\v\f]+$/, "", s); return s }
function idval(line,   v) {            # roadmap_diff.parse_id_value
    sub(/^- id:[ \t]*/, "", line); v = trim(line)
    if (length(v) >= 2 && substr(v,1,1) == "\047" && substr(v,length(v),1) == "\047") { v = substr(v,2,length(v)-2); gsub(/\047\047/, "\047", v) }
    else if (length(v) >= 2 && substr(v,1,1) == "\"" && substr(v,length(v),1) == "\"") { v = substr(v,2,length(v)-2); gsub(/\\"/, "\"", v) }
    return v
}
function parse(id) {                   # roadmap_merge.parse_id -> sets P_PRE, P_NUM; 0 = legacy
    if (id !~ /^[A-Za-z][A-Za-z0-9_]*-[0-9]+([^0-9A-Za-z_].*)?$/) return 0
    match(id, /^[A-Za-z][A-Za-z0-9_]*-/); P_PRE = substr(id, 1, RLENGTH - 1)
    P_NUM = substr(id, RLENGTH + 1); match(P_NUM, /^[0-9]+/); P_NUM = substr(P_NUM, 1, RLENGTH)
    if (MUT != "strnum") sub(/^0+/, "", P_NUM)
    return 1
}
function numgt(a, b) {                 # a > b over arbitrary-length digit strings (python int)
    if (MUT == "strnum") return a > b
    if (MUT == "numge") { if (length(a) != length(b)) return length(a) > length(b); return a >= b }
    if (length(a) != length(b)) return length(a) > length(b)
    return a > b
}
function slot(id,   i, pre, num) {     # roadmap_fragments.insertion_index
    if (!parse(id) || MUT == "append") return N + 1
    pre = P_PRE; num = P_NUM
    for (i = 1; i <= N; i++) {
        if (!parse(E_ID[i]) || P_PRE != pre) continue
        if (numgt(P_NUM, num)) return i
    }
    return N + 1
}
function chop(s) { return (MUT == "addnl") ? s : substr(s, 1, length(s) - 1) }   # undo the "\n" awk added to a last line that had none
BEGIN {
    nf = 0; while ((getline lfl < LISTF) > 0) { nf++; LASTB = LASTB substr(lfl, 1, 1); FP[nf] = substr(lfl, 2) }; close(LISTF)
    # 1. fragments: id = filename stem (python: name[:-len(".yaml")]), text verbatim after newline normalisation
    for (k = 1; k <= nf; k++) {
        path = FP[k]; name = path; sub(/^.*\//, "", name); stem = substr(name, 1, length(name) - 5)
        body = ""; got = ""
        while ((r = (getline line < (ND "/" k))) > 0) {
            if (got == "" && line ~ /^- id:/) got = idval(line)
            body = body line "\n"
        }
        close(ND "/" k)
        if (r < 0) { printf "%s: ENV - cannot read %s\n", "roadmap_aggregate.sh", path > "/dev/stderr"; exit 2 }
        if (substr(LASTB, k, 1) == "x") body = chop(body)
        F_ID[k] = (MUT == "idline" && got != "") ? got : stem
        F_BODY[k] = body
        SUP[F_ID[k]] = 1
    }
}
# 2. the base: preamble, then one block per top-level `- id:` line
/^- id:/ { N0++; B_ID[N0] = idval($0); B_BODY[N0] = $0 "\n"; next }
{ if (N0 == 0) PRE = PRE $0 "\n"; else B_BODY[N0] = B_BODY[N0] $0 "\n" }
END {
    if (BASEL == "x") { if (N0 > 0) B_BODY[N0] = chop(B_BODY[N0]); else PRE = chop(PRE) }
    N = 0
    for (i = 1; i <= N0; i++) {
        if (MUT != "nosup" && (B_ID[i] in SUP)) continue
        N++; E_ID[N] = B_ID[i]; E_BODY[N] = B_BODY[i]
    }
    for (k = 1; k <= nf; k++) {
        s = slot(F_ID[k])
        for (i = N; i >= s; i--) { E_ID[i+1] = E_ID[i]; E_BODY[i+1] = E_BODY[i] }
        E_ID[s] = F_ID[k]; E_BODY[s] = F_BODY[k]; N++
    }
    printf "%s", PRE
    for (i = 1; i <= N; i++) printf "%s", E_BODY[i]
}' "$nd/base"
    rc=$?
    rm -rf -- "${nd:?}"
    return "$rc"
}

# check <roadmap> <entries> -> 0 committed == aggregate AND aggregate is a fixed point; 1 stale/refused.
check() {
    local rm=$1 dir=$2 td rc=0
    td=$(mktemp -d) || return 2
    aggregate "$rm" "$dir" > "$td/a1" || { rc=$?; rm -rf -- "${td:?}"; return "$rc"; }
    aggregate "$td/a1" "$dir" > "$td/a2" || { rc=$?; rm -rf -- "${td:?}"; return "$rc"; }
    if [ "$MUT" != nofix ] && ! cmp -s -- "$td/a1" "$td/a2"; then
        printf 'FAIL  %s: the aggregate is not a fixed point (aggregate(aggregate) differs)\n' "$PROG"; rc=1
    fi
    if ! cmp -s -- "$rm" "$td/a1"; then
        printf 'FAIL  %s: committed %s is STALE — it is not aggregate(base, %s fragment(s)); %s line(s) differ. Remedy: bash scripts/%s --write\n' \
            "$PROG" "${rm#"$ROOT"/}" "$(fragments "$dir" | grep -c -e .)" "$(diff -- "$rm" "$td/a1" | grep -c -e '^[<>]')" "$PROG"
        rc=1
    fi
    [ "$rc" = 0 ] && printf 'ok    %s: %s == aggregate(%s fragment(s)), idempotent\n' "$PROG" "${rm#"$ROOT"/}" "$(fragments "$dir" | grep -c -e .)"
    rm -rf -- "${td:?}"
    return "$rc"
}

# write <roadmap> <entries>: atomic replace, only when the bytes change.
write() {
    local rm=$1 dir=$2 tmp rc
    tmp=$(mktemp "${rm}.XXXXXX") || return 2
    aggregate "$rm" "$dir" > "$tmp"; rc=$?
    if [ "$rc" != 0 ]; then rm -f -- "${tmp:?}"; return "$rc"; fi
    if cmp -s -- "$rm" "$tmp"; then rm -f -- "${tmp:?}"; printf 'ok    %s: %s already == aggregate\n' "$PROG" "${rm#"$ROOT"/}"; return 0; fi
    # cat over, not mv: mktemp makes 0600 and mv would carry that mode onto the tracked file
    cat -- "$tmp" > "$rm" || { rm -f -- "${tmp:?}"; return 2; }
    rm -f -- "${tmp:?}"
    printf 'wrote %s: %s regenerated from %s fragment(s)\n' "$PROG" "${rm#"$ROOT"/}" "$(fragments "$dir" | grep -c -e .)"
}

# roundtrip <roadmap> -> the D4 plant: split EVERY entry whose id can be a filename into its own
# fragment, keep the rest (prose ids) as the base in file order, rebuild, and compare bytes.
roundtrip() {
    local rm=$1 td rc=0 moved
    td=$(mktemp -d) || return 2
    mkdir -p "$td/entries"
    LC_ALL=C awk -v D="$td/entries" -v B="$td/base" '
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
function flush() { if (id == "") return; if (safe) { f = D "/" id ".yaml"; printf "%s", blk > f; close(f) } else printf "%s", blk > B }
/^- id:/ { flush(); id = $0; sub(/^- id:[ \t]*/, "", id); id = trim(id)
           if (id ~ /^\047.*\047$/ || id ~ /^".*"$/) id = substr(id, 2, length(id) - 2)
           safe = (id ~ /^[A-Za-z0-9][A-Za-z0-9._-]*$/ && length(id) <= 111); blk = $0 "\n"; next }
{ if (id == "") printf "%s\n", $0 > B; else blk = blk $0 "\n" }
END { flush(); printf "" > B }' "$rm"
    aggregate "$td/base" "$td/entries" > "$td/out" || { rc=$?; rm -rf -- "${td:?}"; return "$rc"; }
    moved=$(find "$td/entries" -name '*.yaml' | grep -c -e .)
    if cmp -s -- "$rm" "$td/out"; then
        printf 'ok    %s: roundtrip — %s entries split into fragments, rebuilt BYTE-IDENTICAL\n' "$PROG" "$moved"
    else
        printf 'FAIL  %s: roundtrip — %s entries split; rebuilt file differs in %s line(s); first out-of-place ids:\n' "$PROG" "$moved" "$(diff -- "$rm" "$td/out" | grep -c -e '^[<>]')"
        diff <(grep -e '^- id:' "$rm") <(grep -e '^- id:' "$td/out") | grep -e '^[<>]' | head -n 6
        printf 'legacy (prose) ids kept in base: %s; ids out of their prefix order in the source: %s\n' \
            "$(grep -c -e '^- id:' "$td/base")" "$(out_of_order "$rm")"
        rc=1
    fi
    rm -rf -- "${td:?}"
    return "$rc"
}

# out_of_order <roadmap> -> count of PREFIX-NUMBER entries whose number is below an earlier same-prefix one
out_of_order() {
    LC_ALL=C awk '/^- id:/ { id = $0; sub(/^- id:[ \t]*/, "", id); gsub(/^[\047"]|[\047"][ \t]*$/, "", id)
        if (id !~ /^[A-Za-z][A-Za-z0-9_]*-[0-9]+([^0-9A-Za-z_].*)?$/) next
        match(id, /^[A-Za-z][A-Za-z0-9_]*-/); p = substr(id, 1, RLENGTH - 1); n = substr(id, RLENGTH + 1); match(n, /^[0-9]+/); n = substr(n, 1, RLENGTH) + 0
        if ((p in MX) && n < MX[p]) c++; if (!(p in MX) || n > MX[p]) MX[p] = n }
        END { print c + 0 }' "$1"
}


# ---------- self-test: every row is a fixture with a known answer; --mutants proves each row can go RED ----------
ST_PASS=0; ST_FAIL=0; ST_NM=0
st_row() {      # st_row <id> <what> <expected-rc> <actual-rc> <must-grep|-> <output-file>
    local ok=1
    [ "$3" = "$4" ] || ok=0
    if [ "$5" != - ] && ! grep -q -e "$5" -- "$6"; then ok=0; fi
    if [ "$ok" = 1 ]; then ST_PASS=$((ST_PASS + 1)); printf 'PASS  %s %s\n' "$1" "$2"
    else ST_FAIL=$((ST_FAIL + 1)); printf 'FAIL  %s %s (rc %s, want %s)\n' "$1" "$2" "$4" "$3"; sed -n '1,4p' -- "$6" | sed 's/^/        /'; fi
}
ent() { printf -- '- id: %s\n  title: %s\n  status: %s\n' "$1" "${2:-t}" "${3:-open}"; }   # one entry block
ids_of() { grep -e '^- id:' -- "$1" | sed 's/^- id:[ \t]*//' | tr '\n' ' '; }
st_case() {     # st_case <id> <what> <want-rc> <want-grep|-> <want-id-order|-> ; uses $T/base and $T/e
    local rc got
    aggregate "$T/base" "$T/e" > "$T/out" 2> "$T/err"; rc=$?
    cat -- "$T/err" >> "$T/out"
    if [ "$5" != - ]; then got=$(ids_of "$T/out"); [ "$got" = "$5 " ] || { printf 'ids: %s\n' "$got" >> "$T/out"; rc=99; }; fi
    st_row "$1" "$2" "$3" "$rc" "$4" "$T/out"
}
fresh() { rm -rf -- "${T:?}/e" "${T:?}/base"; mkdir -p "$T/e"; printf 'version: 1\nroadmap:\n' > "$T/base"; }

selftest() {
    local T rc
    T=$(mktemp -d) || return 2
    # A1-A6: ordering and supersede (mutants strnum numge append nosup)
    fresh; { ent PMAT-1; ent PMAT-3; } >> "$T/base"; ent PMAT-2 > "$T/e/PMAT-2.yaml"
    st_case A1 "fragment lands in its numeric slot" 0 - "PMAT-1 PMAT-2 PMAT-3"
    fresh; { ent PMAT-9; ent PMAT-100; } >> "$T/base"; ent PMAT-10 > "$T/e/PMAT-10.yaml"
    st_case A2 "slot is numeric, not string (10 between 9 and 100)" 0 - "PMAT-9 PMAT-10 PMAT-100"
    fresh; { ent PMAT-9; ent PMAT-100; } >> "$T/base"; ent PMAT-010 > "$T/e/PMAT-010.yaml"
    st_case A3 "leading zeros do not move the slot" 0 - "PMAT-9 PMAT-010 PMAT-100"
    fresh; { ent PMAT-5-old; ent PMAT-6; } >> "$T/base"; ent PMAT-5 > "$T/e/PMAT-5.yaml"
    st_case A4 "equal number goes AFTER the base entry (strictly-greater rule)" 0 - "PMAT-5-old PMAT-5 PMAT-6"
    fresh; { ent PMAT-1 t open; ent PMAT-2; } >> "$T/base"; ent PMAT-1 t done > "$T/e/PMAT-1.yaml"
    st_case A5 "fragment overrides the base entry with its id (no PR edits base)" 0 "status: done" "PMAT-1 PMAT-2"
    fresh; { ent PMAT-1; ent GH-4; } >> "$T/base"; ent nlp-prose > "$T/e/nlp-prose.yaml"; ent GH-2 > "$T/e/GH-2.yaml"
    st_case A6 "legacy id appends; other prefix slots in its own run" 0 - "PMAT-1 GH-2 GH-4 nlp-prose"
    # P1-P7: PARITY with the existing generator (fork (i)): it refuses only what python refuses, and reads like python
    # (mutants idline addnl nocr nofollow noutf8)
    fresh; { ent PMAT-1; ent PMAT-7; ent PMAT-9; } >> "$T/base"; ent PMAT-7 > "$T/e/PMAT-8.yaml"
    st_case P1 "id is the FILENAME (python), not the '- id:' line: PMAT-8.yaml holding PMAT-7 supersedes nothing" 0 - "PMAT-1 PMAT-7 PMAT-7 PMAT-9"
    fresh; { ent PMAT-1; ent PMAT-9; } >> "$T/base"; ent PMAT-5 > "$T/e/PMAT-5.yaml"; ent PMAT-05 > "$T/e/PMAT-05.yaml"
    st_case P2 "PMAT-5 and PMAT-05 are two files, so two entries, in filename order (python accepts them)" 0 - "PMAT-1 PMAT-05 PMAT-5 PMAT-9"
    fresh; { ent PMAT-1; ent PMAT-3; } >> "$T/base"; ent PMAT-2 | head -c -1 > "$T/e/PMAT-2.yaml"
    st_case P3 "no final newline is pasted verbatim, as python does" 0 "status: open- id: PMAT-3" -
    fresh; ent PMAT-1 >> "$T/base"; ent PMAT-2 | sed 's/$/\r/' > "$T/e/PMAT-2.yaml"
    aggregate "$T/base" "$T/e" > "$T/out" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ "$(tr -cd '\r' < "$T/out" | wc -c)" != 0 ] && rc=3
    st_row P4 "CRLF reads as LF (python universal newlines): no CR byte reaches the output" 0 "$rc" "id: PMAT-2" "$T/out"
    fresh; ent PMAT-1 >> "$T/base"; { ent PMAT-2; ent PMAT-3; } > "$T/e/PMAT-2.yaml"; : > "$T/e/PMAT-4.yaml"
    st_case P5 "two entries in one fragment and an empty fragment are pasted as python pastes them (readers judge them)" 0 - "PMAT-1 PMAT-2 PMAT-3"
    fresh; ent PMAT-1 >> "$T/base"; printf -- '- id: PMAT-2\n  title: \377\376\n' > "$T/e/PMAT-2.yaml"
    st_case P6 "invalid UTF-8 is refused (1): python dies on it too" 1 "not valid UTF-8" -
    fresh; ent PMAT-1 >> "$T/base"; ln -s "$T/nowhere" "$T/e/PMAT-2.yaml"
    st_case P7 "a dangling symlink is refused (1): python's open() fails on it too" 1 "cannot be opened" -
    # C1-C4: --check / --write
    fresh; { ent PMAT-1; ent PMAT-3; } >> "$T/base"; ent PMAT-2 > "$T/e/PMAT-2.yaml"; cp -- "$T/base" "$T/rm"
    check "$T/rm" "$T/e" > "$T/out" 2>&1; rc=$?; st_row C1 "--check is RED on a stale committed copy" 1 "$rc" "FAIL" "$T/out"
    write "$T/rm" "$T/e" > "$T/out" 2>&1; rc=$?; st_row C2 "--write regenerates" 0 "$rc" "wrote" "$T/out"
    check "$T/rm" "$T/e" > "$T/out" 2>&1; rc=$?; st_row C3 "--check is GREEN after --write" 0 "$rc" "^ok" "$T/out"
    write "$T/rm" "$T/e" > "$T/out" 2>&1; rc=$?; st_row C4 "--write is a no-op on a fresh copy" 0 "$rc" "already ==" "$T/out"
    # T1-T2: round trip
    fresh; { ent PMAT-1; ent PMAT-2; ent prose-x; } >> "$T/base"
    roundtrip "$T/base" > "$T/out" 2>&1; rc=$?; st_row T1 "roundtrip of a sorted file is byte-identical" 0 "$rc" "BYTE-IDENTICAL" "$T/out"
    fresh; { ent PMAT-2; ent PMAT-1; } >> "$T/base"
    roundtrip "$T/base" > "$T/out" 2>&1; rc=$?; st_row T2 "roundtrip of an out-of-order file is RED" 1 "$rc" "FAIL" "$T/out"
    # E1-E2: environment
    mkdir -p "$T/l" && ent B-9 > "$T/outside.yaml" && ln -s "$T/outside.yaml" "$T/l/B-9.yaml"
    ent A-1 > "$T/lbase"; aggregate "$T/lbase" "$T/l" > "$T/out" 2>&1; rc=$?; st_row L1 "a symlinked fragment is followed, as python's open() follows it" 0 "$rc" "id: B-9" "$T/out"
    mkdir -p "$T/real" && ent B-7 > "$T/real/B-7.yaml" && ln -s "$T/real" "$T/ldir"
    aggregate "$T/lbase" "$T/ldir" > "$T/out" 2>&1; rc=$?; st_row L2 "a symlinked entries DIR is followed, as python's os.listdir follows it" 0 "$rc" "id: B-7" "$T/out"
    # F1: --check proves the fixed point (python _check does too); a no-final-newline fragment mid-list breaks it
    fresh; { ent PMAT-1; ent PMAT-3; } >> "$T/base"; ent PMAT-2 | head -c -1 > "$T/e/PMAT-2.yaml"
    cp -- "$T/base" "$T/rm"
    check "$T/rm" "$T/e" > "$T/out" 2>&1; rc=$?; st_row F1 "--check is RED when aggregate(aggregate) differs" 1 "$rc" "not a fixed point" "$T/out"
    aggregate "$T/missing" "$T/e" > "$T/out" 2>&1; rc=$?; st_row E1 "missing base is ENV (2), never a verdict" 2 "$rc" "ENV" "$T/out"
    bash "$SELF" --bogus > "$T/out" 2>&1; rc=$?; st_row E2 "unknown argument is 2" 2 "$rc" "unknown" "$T/out"
    # D1: differential against the existing aggregator on the real tree (not_measured, never PASS, when absent)
    if command -v python3 > /dev/null 2>&1 && [ -r "$ROOT/scripts/lib/roadmap_fragments.py" ] && [ -r "$ROADMAP" ]; then
        aggregate "$ROADMAP" "$ENTRIES" > "$T/sh.out" 2> "$T/out"
        python3 "$ROOT/scripts/lib/roadmap_fragments.py" aggregate --roadmap "$ROADMAP" --entries "$ENTRIES" > "$T/py.out" 2>> "$T/out"
        cmp -s -- "$T/sh.out" "$T/py.out"; rc=$?
        [ -s "$T/sh.out" ] || rc=3
        st_row D1 "real tree: bash aggregate == roadmap_fragments.py aggregate, byte for byte" 0 "$rc" - "$T/out"
    else ST_NM=$((ST_NM + 1)); printf 'NOT_MEASURED  D1 differential (python3 or the real tree is absent) — counted as not_measured, never PASS\n'; fi
    rm -rf -- "${T:?}"
    printf '%s --selftest: %s PASS, %s FAIL, %s NOT_MEASURED\n' "$PROG" "$ST_PASS" "$ST_FAIL" "$ST_NM"
    [ "$ST_FAIL" = 0 ] && [ "$ST_PASS" -gt 0 ] || return 1
    [ "$ST_NM" = 0 ] || { printf '%s --selftest: a row was NOT_MEASURED, so this run is not green (exit 2)\n' "$PROG"; return 2; }
}

# every planted mutant must turn at least one row RED; a mutant that survives is a row that cannot fail
mutants() {
    local m killed=0 alive=0
    for m in strnum numge append idline addnl nocr nofollow nosup noutf8 nofix nodirlink; do
        if ROADMAP_AGG_MUTANT=$m bash "$SELF" --selftest > /dev/null 2>&1; then
            alive=$((alive + 1)); printf 'SURVIVED  mutant %s\n' "$m"
        else killed=$((killed + 1)); printf 'killed    mutant %s\n' "$m"; fi
    done
    printf '%s --mutants: %s killed, %s survived\n' "$PROG" "$killed" "$alive"
    [ "$alive" = 0 ]
}

main() {
    local mode="" rm="$ROADMAP" dir="$ENTRIES"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --print|--write|--check|--roundtrip|--selftest|--self-test|--mutants) mode=$1 ;;
            --roadmap) [ "$#" -ge 2 ] || die2 "--roadmap needs a file"; rm=$2; shift ;;
            --entries) [ "$#" -ge 2 ] || die2 "--entries needs a dir"; dir=$2; shift ;;
            -h|--help) sed -n '2,35p' "$SELF" | sed 's/^# \{0,1\}//'; return 0 ;;
            *) die2 "unknown argument: $1 (see --help)" ;;
        esac
        shift
    done
    case "$mode" in
        --print) aggregate "$rm" "$dir" ;;
        --write) write "$rm" "$dir" ;;
        --check) check "$rm" "$dir" ;;
        --roundtrip) roundtrip "$rm" ;;
        --selftest|--self-test) selftest ;;
        --mutants) mutants ;;
        *) die2 "a mode is required: --print | --write | --check | --roundtrip | --selftest | --mutants" ;;
    esac
}
main "$@"
