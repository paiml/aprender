#!/usr/bin/env bash
# evidence_store.sh -- the evidence store (BLD-002 R2). Receipts, rulings, known-failure marks and
# release notes are kept OUTSIDE the code they describe, keyed by its code identity H, and written by
# one writer.
# WHY       Evidence was committed into the tree it judged. Recording a receipt made a new commit, and
#           every check bound to "this commit" had to run again; a tag whose tree held no receipts
#           could not be checked at all. The R0 baseline has both: 0.70.0's readiness check never ran
#           on its tag (the tagged tree had no receipts), and a 0.70.1 pre-tag check failed on
#           uncommitted evidence.
# THE RULE  A record is written once and never changed. Whether a record is valid depends only on its
#           own bytes, the H asked for and the store's writer, so adding a record can never change
#           whether another record is valid. A record for another H is refused, by name.
# LAYOUT    STORE/WRITER                 the one writer's id, set once by init
#           STORE/<H>/<kind>/<name>      one record; kind is receipt, ruling, known-failure or
#                                        release-note
#           STORE/.tmp/                  unfinished writes, never read
#           H is 40 or 64 lowercase hex. How H is computed is not decided here (BLD-002 R1), and
#           neither is where STORE lives: it is an argument.
# RECORD    Seven header lines, then the body byte for byte:
#             evidence-store-record/v1 | h <H> | kind <kind> | name <name> | writer <id> |
#             body-sha256 <64 hex> | --
# VALID     A regular file, not a symlink and not reached through a symlinked directory, whose header
#           is byte for byte the one put writes: its h is the H asked for, its kind and name are the
#           ones it is filed under, its writer is the store's writer, and its body is not empty and
#           still hashes to body-sha256.
# EXIT      0 done; 1 REFUSED, with a FAIL line naming every record or write refused; 3 caller error
# USAGE     evidence_store.sh init STORE WRITER-ID
#           EVIDENCE_STORE_WRITER=<id> evidence_store.sh put STORE H KIND NAME FILE
#           evidence_store.sh get STORE H KIND NAME       prints the body; nothing when refused
#           evidence_store.sh verify STORE H              one line per record under H
#           evidence_store.sh --selftest | --mutants | -h
set -uo pipefail
export LC_ALL=C
shopt -s nullglob dotglob

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
MAGIC='evidence-store-record/v1'
EMPTY_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
W=''
TMPF=''
TMPB=''

caller_error() { printf 'FAIL  STORE %s: caller error: %s\n' "$PROG" "$*"; exit 3; }
refuse() { printf 'FAIL  STORE %s\n' "$*"; }
ok() { printf 'ok    STORE %s\n' "$*"; }
short() { printf '%.12s' "$1"; }
# The header put writes, line for line (H KIND NAME WRITER BODY-SHA256). check_record compares a record's
# first seven lines to it byte for byte, so the format is written down once.
header() { printf '%s\n' "$MAGIC" "h $1" "kind $2" "name $3" "writer $4" "body-sha256 $5" '--'; }
cleanup() {
    if [ -n "$TMPF" ]; then rm -f -- "${TMPF:?}"; fi
    if [ -n "$TMPB" ]; then rm -f -- "${TMPB:?}"; fi
}
trap cleanup EXIT
# A signal ends the script through exit, so the EXIT trap still removes the temporary files.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

is_h() {
    case "$1" in ''|*[!0-9a-f]*) return 1 ;; esac
    [ "${#1}" -eq 40 ] || [ "${#1}" -eq 64 ]
}
is_kind() {
    case "$1" in receipt|ruling|known-failure|release-note) return 0 ;; *) return 1 ;; esac
}
# A record name or a writer id: letters, digits, '.', '_' and '-', at most 128, never a leading '.'.
is_name() {
    case "$1" in ''|.*|*[!A-Za-z0-9._-]*) return 1 ;; esac
    [ "${#1}" -le 128 ]
}
check_key() { # H KIND NAME: a caller error unless each one is well formed
    is_h "$1" || caller_error "H '$1' is not a code identity (40 or 64 lowercase hex)"
    is_kind "$2" || caller_error "unknown kind '$2' (receipt, ruling, known-failure or release-note)"
    is_name "$3" || caller_error "'$3' is not a record name (letters, digits, '.', '_', '-'; at most 128; no leading '.')"
}
# STORE -> W, the store's one writer. A directory without a WRITER is not a store.
load_writer() {
    local w=''
    if [ ! -d "$1" ]; then caller_error "STORE '$1' is not a directory"; fi
    if [ -L "$1/WRITER" ] || [ ! -f "$1/WRITER" ]; then caller_error "'$1' is not an evidence store: it has no WRITER (run init)"; fi
    IFS= read -r w < "$1/WRITER" || true
    is_name "$w" || caller_error "'$1' has a WRITER that is not a writer id"
    W="$w"
}

# check_record FILE H KIND NAME: 0, with the first 16 hex of the body's sha256 on stdout, when FILE is a
# valid record of KIND/NAME for H written by W; 1, with the reason on stdout, when it is not.
check_record() {
    local f="$1" h="$2" kind="$3" name="$4" l1='' l2='' l3='' l4='' l5='' l6='' l7='' oh sum hs want
    if [ -L "$f" ]; then echo "is a symlink: a record is a file of its own, never a pointer to another"; return 1; fi
    if [ ! -f "$f" ]; then echo "is not a file"; return 1; fi
    if ! { IFS= read -r l1 && IFS= read -r l2 && IFS= read -r l3 && IFS= read -r l4 \
            && IFS= read -r l5 && IFS= read -r l6 && IFS= read -r l7; } < "$f"; then
        echo "is not a record: its header is incomplete"; return 1
    fi
    if [ "$l1" != "$MAGIC" ] || [ "$l7" != '--' ]; then echo "is not a record: it has no $MAGIC header"; return 1; fi
    oh="${l2#h }"
    if [ "$l2" != "h $h" ]; then
        if [ "$l2" != "h $oh" ] || ! is_h "$oh"; then echo "is not a record: its h line is not a code identity"; return 1; fi
        echo "is recorded for H=$oh, not H=$h: a record for another code identity is refused"; return 1
    fi
    if [ "$l3" != "kind $kind" ] || [ "$l4" != "name $name" ]; then
        echo "is filed as $kind/$name but recorded as ${l3#kind }/${l4#name }"; return 1
    fi
    if [ "$l5" != "writer $W" ]; then echo "was written by '${l5#writer }', not by the store's writer '$W'"; return 1; fi
    if ! sum="$(tail -n +8 -- "$f" | sha256sum)"; then echo "could not be read"; return 1; fi
    sum="${sum%% *}"
    if [ "$sum" = "$EMPTY_SHA256" ]; then echo "has an empty body"; return 1; fi
    if [ "$l6" != "body-sha256 $sum" ]; then
        echo "has a body that changed after it was written: body-sha256 ${l6#body-sha256 } recorded, $sum now"; return 1
    fi
    # read drops NUL bytes, so the fields above can match a header that put never wrote: the header's
    # bytes are compared as well.
    hs="$(head -n 7 -- "$f" | sha256sum)"
    want="$(header "$h" "$kind" "$name" "$W" "$sum" | sha256sum)"
    if [ "$hs" != "$want" ]; then
        echo "is not a record: its header is not, byte for byte, the header put writes"; return 1
    fi
    printf '%.16s' "$sum"
}

cmd_init() {
    [ $# -eq 2 ] || caller_error "init needs STORE WRITER-ID"
    local store="$1" id="$2" w=''
    local -a e
    is_name "$id" || caller_error "writer id '$id' is not a name (letters, digits, '.', '_', '-'; no leading '.')"
    if [ -e "$store" ] && [ ! -d "$store" ]; then caller_error "STORE '$store' is not a directory"; fi
    if [ -L "$store/WRITER" ]; then refuse "$store/WRITER is a symlink: the store is not adopted"; return 1; fi
    if [ -f "$store/WRITER" ]; then
        IFS= read -r w < "$store/WRITER" || true
        if [ "$w" = "$id" ]; then ok "$store is already the store of writer $id"; return 0; fi
        refuse "$store already has writer '$w': a store has one writer"; return 1
    fi
    e=("$store"/*)
    # An init that did not finish leaves only STORE/.tmp/, which nothing reads: that is still empty.
    if [ "${#e[@]}" -eq 1 ] && [ "${e[0]}" = "$store/.tmp" ] && [ ! -L "${e[0]}" ]; then e=(); fi
    if [ "${#e[@]}" -gt 0 ]; then refuse "$store is not empty and has no WRITER: it is not adopted"; return 1; fi
    if ! mkdir -p -- "$store/.tmp"; then refuse "could not create $store/.tmp"; return 1; fi
    TMPF="$(mktemp "$store/.tmp/WRITER.XXXXXX")" || { refuse "mktemp failed in $store/.tmp"; return 1; }
    # ln -T sets the WRITER whole, at exactly that path, or not at all. Of two inits at once, one sets
    # it and the other is refused; run again, that one reports the writer that was set.
    if ! printf '%s\n' "$id" > "$TMPF" || ! ln -T -- "$TMPF" "$store/WRITER" 2>/dev/null; then
        refuse "could not set the writer of $store"; return 1
    fi
    ok "$store is the store of writer $id"
}

cmd_put() {
    [ $# -eq 5 ] || caller_error "put needs STORE H KIND NAME FILE"
    local store="$1" h="$2" kind="$3" name="$4" src="$5" me f sum why
    load_writer "$store"
    check_key "$h" "$kind" "$name"
    if [ ! -f "$src" ] || [ ! -r "$src" ]; then caller_error "FILE '$src' is not a readable file"; fi
    me="${EVIDENCE_STORE_WRITER:-}"
    if [ "$me" != "$W" ]; then refuse "writer '${me:-<unset>}' is not this store's writer '$W': the store has one writer"; return 1; fi
    if [ -L "$store/.tmp" ]; then refuse "$store/.tmp is a symlink: an unfinished write stays in the store"; return 1; fi
    if [ -L "$store/$h" ] || [ -L "$store/$h/$kind" ]; then refuse "the path to $kind/$name for H=$(short "$h") holds a symlink"; return 1; fi
    f="$store/$h/$kind/$name"
    if ! mkdir -p -- "$store/.tmp"; then refuse "could not create $store/.tmp"; return 1; fi
    # The body is copied first and hashed from the copy, so the hash is of the bytes recorded.
    TMPB="$(mktemp "$store/.tmp/body.XXXXXX")" || { refuse "mktemp failed in $store/.tmp"; return 1; }
    TMPF="$(mktemp "$store/.tmp/put.XXXXXX")" || { refuse "mktemp failed in $store/.tmp"; return 1; }
    if ! cat -- "$src" > "$TMPB"; then refuse "could not copy $src"; return 1; fi
    if ! sum="$(sha256sum < "$TMPB")"; then refuse "could not hash $src"; return 1; fi
    sum="${sum%% *}"
    # Refused before anything is made under STORE/<H>/.
    if [ "$sum" = "$EMPTY_SHA256" ]; then refuse "$kind/$name for H=$(short "$h"): $src is empty, so nothing is recorded"; return 1; fi
    if ! header "$h" "$kind" "$name" "$W" "$sum" > "$TMPF" || ! cat -- "$TMPB" >> "$TMPF"; then
        refuse "could not write $kind/$name for H=$(short "$h")"; return 1
    fi
    if ! mkdir -p -- "$store/$h/$kind"; then refuse "could not create $store/$h/$kind"; return 1; fi
    # ln -T makes the record appear whole or not at all, at exactly its path: it never replaces a file,
    # never writes into a directory and never writes through a symlink. A record is written once.
    if ! ln -T -- "$TMPF" "$f" 2>/dev/null; then
        if [ -L "$f" ]; then refuse "$kind/$name for H=$(short "$h") is a symlink"; return 1; fi
        if [ ! -e "$f" ]; then refuse "could not record $kind/$name for H=$(short "$h")"; return 1; fi
        if [ ! -f "$f" ]; then refuse "$kind/$name for H=$(short "$h") is not a file"; return 1; fi
        if cmp -s -- "$TMPF" "$f"; then ok "$kind/$name for H=$(short "$h") unchanged: already recorded with this body"; return 0; fi
        refuse "$kind/$name is already recorded for H=$(short "$h") with another body: a record is written once and never changed"
        return 1
    fi
    if ! why="$(check_record "$f" "$h" "$kind" "$name")"; then refuse "$kind/$name for H=$(short "$h") $why"; return 1; fi
    ok "$kind/$name recorded for H=$(short "$h") body-sha256=$why"
}

cmd_get() {
    [ $# -eq 4 ] || caller_error "get needs STORE H KIND NAME"
    local store="$1" h="$2" kind="$3" name="$4" f d o others='' why
    load_writer "$store"
    check_key "$h" "$kind" "$name"
    if [ -L "$store/$h" ] || [ -L "$store/$h/$kind" ]; then refuse "the path to $kind/$name for H=$(short "$h") holds a symlink"; return 1; fi
    f="$store/$h/$kind/$name"
    if [ ! -e "$f" ] && [ ! -L "$f" ]; then
        for d in "$store"/*; do
            o="${d##*/}"
            if is_h "$o" && [ "$o" != "$h" ] && { [ -e "$d/$kind/$name" ] || [ -L "$d/$kind/$name" ]; }; then
                others="$others${others:+, }H=$o"
            fi
        done
        # Every H is printed in full, so two H that share a prefix are still told apart.
        if [ -n "$others" ]; then
            refuse "$kind/$name is not recorded for H=$h; it is filed only under $others: a record for another code identity is refused"
        else
            refuse "$kind/$name is not recorded for H=$h"
        fi
        return 1
    fi
    if [ -L "$f" ]; then refuse "$kind/$name for H=$(short "$h") is a symlink: a record is a file of its own, never a pointer to another"; return 1; fi
    if [ ! -f "$f" ]; then refuse "$kind/$name for H=$(short "$h") is not a file"; return 1; fi
    # Judged and printed from one copy, so the bytes printed are the bytes judged. cp -P never follows a
    # symlink: one swapped in after the checks above is copied as a symlink, and refused.
    TMPF="$(mktemp)" || { refuse "mktemp failed: $kind/$name was not read"; return 1; }
    if ! cp -P -- "$f" "$TMPF"; then refuse "$kind/$name for H=$(short "$h") could not be read"; return 1; fi
    if ! why="$(check_record "$TMPF" "$h" "$kind" "$name")"; then refuse "$kind/$name for H=$(short "$h") $why"; return 1; fi
    tail -n +8 -- "$TMPF"
}

cmd_verify() {
    [ $# -eq 2 ] || caller_error "verify needs STORE H"
    local store="$1" h="$2" dir kd k f name out n=0 bad=0
    load_writer "$store"
    is_h "$h" || caller_error "H '$h' is not a code identity (40 or 64 lowercase hex)"
    dir="$store/$h"
    if [ -L "$dir" ]; then refuse "H=$(short "$h") is a symlink"; return 1; fi
    for kd in "$dir"/*; do
        k="${kd##*/}"
        if [ -L "$kd" ] || [ ! -d "$kd" ] || ! is_kind "$k"; then
            refuse "H=$(short "$h") holds '$k', which is not a record kind directory"; n=$((n + 1)); bad=$((bad + 1)); continue
        fi
        for f in "$kd"/*; do
            name="${f##*/}"; n=$((n + 1))
            if ! is_name "$name"; then refuse "$k/$name for H=$(short "$h") is not a record name"; bad=$((bad + 1)); continue; fi
            if out="$(check_record "$f" "$h" "$k" "$name")"; then
                ok "$k/$name H=$(short "$h") body-sha256=$out"
            else
                refuse "$k/$name for H=$(short "$h") $out"; bad=$((bad + 1))
            fi
        done
    done
    if [ "$n" -eq 0 ]; then refuse "no records for H=$(short "$h"): nothing to verify is not a pass"; return 1; fi
    printf -- '--- H=%s: %s checked, %s refused ---\n' "$(short "$h")" "$n" "$bad"
    if [ "$bad" -ne 0 ]; then return 1; fi
}

selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }

selftest() {
    local tmp pass=0 fail=0 s s2 s3 s5 s6 out before after rc b n128 n129
    local H1=1111111111111111111111111111111111111111
    local H2=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    local H3=ffffffffffffffffffffffffffffffffffffffff
    local H5=1111111111111111111111111111111111111111111111111111111111111111
    local HU=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    export EVIDENCE_STORE_WRITER=train
    row() { # name expect-rc needle -- script-args ... ; a needle that starts with ! must NOT be said
        local name="$1" expect="$2" needle="$3" o rc=0
        shift 3
        if [ "${1:-}" = -- ]; then shift; fi
        o="$(timeout 20 bash "$SCRIPT_PATH" "$@" < /dev/null 2>&1)" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-66s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$needle" in
            '!'*)
                case "$o" in
                    *"${needle#!}"*) printf '  BROKE %-66s exit %s but said: %s\n%s\n' "$name" "$rc" "${needle#!}" "$o"; fail=$((fail + 1)) ;;
                    *) printf '  ok    %-66s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                esac ;;
            *)
                case "$o" in
                    *"$needle"*) printf '  ok    %-66s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                    *) printf '  BROKE %-66s exit %s but never said: %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)) ;;
                esac ;;
        esac
    }
    body_row() { # name STORE H KIND NAME expected-file: get prints exactly the expected bytes
        local name="$1" rc=0
        timeout 20 bash "$SCRIPT_PATH" get "$2" "$3" "$4" "$5" < /dev/null > "$tmp/got" 2>&1 || rc=$?
        if [ "$rc" = 0 ] && cmp -s -- "$tmp/got" "$6"; then
            printf '  ok    %-66s exit=0, the same bytes\n' "$name"; pass=$((pass + 1))
        else
            printf '  BROKE %-66s exit %s, not the bytes of %s\n' "$name" "$rc" "${6##*/}"; fail=$((fail + 1))
        fi
    }
    fingerprint() { # STORE H KIND NAME -> what a reader sees of one record: get's bytes, verify's line, the file
        local g v f
        g="$(bash "$SCRIPT_PATH" get "$1" "$2" "$3" "$4" < /dev/null 2>&1 | sha256sum)"
        v="$(bash "$SCRIPT_PATH" verify "$1" "$2" < /dev/null 2>&1 | grep -F " $3/$4 ")"
        f="$(sha256sum < "$1/$2/$3/$4")"
        printf '%s|%s|%s' "${g%% *}" "$v" "${f%% *}"
    }
    plant() { # FILE H KIND NAME WRITER BODY-FILE: a record written by hand, as a forger or a broken writer would
        local b
        b="$(sha256sum < "$6")"
        { printf '%s\n' "$MAGIC" "h $2" "kind $3" "name $4" "writer $5" "body-sha256 ${b%% *}" '--'; cat -- "$6"; } > "$1"
    }
    printf 'receipt one\n' > "$tmp/r1"
    printf 'receipt one, measured again\n' > "$tmp/r1b"
    printf '{"verdict":"GO"}\n' > "$tmp/r2"
    printf 'ruling text\n' > "$tmp/ru"
    printf 'no final newline' > "$tmp/nonl"
    printf '\000\001\377binary\n' > "$tmp/bin"
    : > "$tmp/empty"
    n128="$(printf '%0128d' 0)"; n129="${n128}0"
    s="$tmp/store"
    mkdir -p "$tmp/busy"; printf 'x\n' > "$tmp/busy/x"

    echo "--- init: a store has one writer ---"
    row 'init a new store' 0 'is the store of writer train' -- init "$s" train
    row 'init again with the same writer' 0 'is already the store of writer train' -- init "$s" train
    row 'init again with another writer' 1 'a store has one writer' -- init "$s" other
    row 'init over a directory that is not empty' 1 'is not empty and has no WRITER' -- init "$tmp/busy" train
    row 'init with a writer id that is not a name' 3 'is not a name' -- init "$tmp/s9" 'a b'

    echo "--- caller errors ---"
    row 'no command' 3 'no command' --
    row 'an unknown command' 3 'unknown command' -- bogus
    row 'put with too few arguments' 3 'put needs STORE H KIND NAME FILE' -- put "$s" "$H1"
    row 'a store that is not a directory' 3 'is not a directory' -- get "$tmp/none" "$H1" receipt r1
    row 'a directory that is not a store' 3 'it has no WRITER' -- get "$tmp/busy" "$H1" receipt r1
    row 'an H that is not hex' 3 'is not a code identity' -- get "$s" xyz receipt r1
    row 'an H of 39 hex digits' 3 'is not a code identity' -- get "$s" "${H1:1}" receipt r1
    row 'an H in upper case' 3 'is not a code identity' -- get "$s" "$HU" receipt r1
    row 'an unknown kind' 3 'unknown kind' -- get "$s" "$H1" receipts r1
    row 'a name with a slash' 3 'is not a record name' -- get "$s" "$H1" receipt a/b
    row 'a name with a leading dot' 3 'is not a record name' -- get "$s" "$H1" receipt .r1
    row 'a name of two dots' 3 'is not a record name' -- get "$s" "$H1" receipt ..
    row 'put of a file that does not exist' 3 'is not a readable file' -- put "$s" "$H1" receipt r1 "$tmp/none"
    row 'a name of 128 characters is a record name' 1 "receipt/$n128 is not recorded for H=$H1" -- get "$s" "$H1" receipt "$n128"
    row 'a name of 129 characters is not' 3 'is not a record name' -- get "$s" "$H1" receipt "$n129"

    echo "--- put: one writer, each record written once ---"
    EVIDENCE_STORE_WRITER=other row 'put by another writer' 1 "writer 'other' is not this store's writer 'train'" -- put "$s" "$H1" receipt r1 "$tmp/r1"
    EVIDENCE_STORE_WRITER='' row 'put with no writer named' 1 "writer '<unset>' is not this store's writer" -- put "$s" "$H1" receipt r1 "$tmp/r1"
    row 'put a receipt' 0 'receipt/r1 recorded for H=111111111111' -- put "$s" "$H1" receipt r1 "$tmp/r1"
    row 'put the same body again' 0 'unchanged: already recorded with this body' -- put "$s" "$H1" receipt r1 "$tmp/r1"
    row 'FALSIFIER: put another body under the same name is refused' 1 'a record is written once and never changed' -- put "$s" "$H1" receipt r1 "$tmp/r1b"
    row 'put an empty file' 1 'is empty, so nothing is recorded' -- put "$s" "$H3" receipt e1 "$tmp/empty"
    if [ -e "$s/$H3" ] || [ -L "$s/$H3" ]; then
        printf '  BROKE %-66s it made STORE/%s\n' 'a refused put of an empty body makes nothing under STORE/<H>' "$(short "$H3")"; fail=$((fail + 1))
    else
        printf '  ok    %-66s nothing made\n' 'a refused put of an empty body makes nothing under STORE/<H>'; pass=$((pass + 1))
    fi
    row 'put a ruling' 0 'ruling/c1 recorded for H=111111111111' -- put "$s" "$H1" ruling c1 "$tmp/ru"
    row 'put a body with no final newline' 0 'release-note/n1 recorded' -- put "$s" "$H1" release-note n1 "$tmp/nonl"
    row 'put a binary body' 0 'known-failure/k1 recorded' -- put "$s" "$H1" known-failure k1 "$tmp/bin"
    row 'put the same name for another H' 0 'receipt/r1 recorded for H=aaaaaaaaaaaa' -- put "$s" "$H2" receipt r1 "$tmp/r1b"

    echo "--- get: the bytes recorded, for the H asked for ---"
    body_row 'get prints the body byte for byte' "$s" "$H1" receipt r1 "$tmp/r1"
    body_row 'get keeps a body with no final newline' "$s" "$H1" release-note n1 "$tmp/nonl"
    body_row 'get keeps a binary body' "$s" "$H1" known-failure k1 "$tmp/bin"
    body_row 'the same name for another H is a record of its own' "$s" "$H2" receipt r1 "$tmp/r1b"
    row 'get a record that was never written' 1 "receipt/r7 is not recorded for H=$H1" -- get "$s" "$H1" receipt r7
    row 'FALSIFIER: a receipt that exists only for other H is refused by name' 1 "receipt/r1 is not recorded for H=$H3; it is filed only under H=$H1, H=$H2: a record for another code identity is refused" -- get "$s" "$H3" receipt r1
    row 'FALSIFIER: every H is named in full, even two that share a prefix' 1 "receipt/r1 is not recorded for H=$H5; it is filed only under H=$H1, H=$H2" -- get "$s" "$H5" receipt r1
    row 'get a receipt asked for as a ruling' 1 "ruling/r1 is not recorded for H=$H1" -- get "$s" "$H1" ruling r1

    echo "--- verify ---"
    row 'verify an H' 0 '4 checked, 0 refused' -- verify "$s" "$H1"
    row 'verify names each record' 0 'ok    STORE receipt/r1 H=111111111111 body-sha256=' -- verify "$s" "$H1"
    row 'verify an H with no records' 1 'nothing to verify is not a pass' -- verify "$s" "$H3"

    echo "--- adding records changes no record's validity ---"
    before="$(fingerprint "$s" "$H1" receipt r1; fingerprint "$s" "$H2" receipt r1)"
    rc=all-as-expected
    case "$before" in
        *"ok    STORE receipt/r1 H=111111111111 "*"ok    STORE receipt/r1 H=aaaaaaaaaaaa "*) : ;;
        *) rc='the receipts did not verify before the puts' ;;
    esac
    bash "$SCRIPT_PATH" put "$s" "$H1" ruling c2 "$tmp/ru" > /dev/null 2>&1 || rc='a put was refused'
    bash "$SCRIPT_PATH" put "$s" "$H1" receipt r2 "$tmp/r2" > /dev/null 2>&1 || rc='a put was refused'
    bash "$SCRIPT_PATH" put "$s" "$H2" receipt r3 "$tmp/r2" > /dev/null 2>&1 || rc='a put was refused'
    bash "$SCRIPT_PATH" put "$s" "$H2" known-failure k2 "$tmp/r2" > /dev/null 2>&1 || rc='a put was refused'
    if bash "$SCRIPT_PATH" put "$s" "$H1" receipt r1 "$tmp/r1b" > /dev/null 2>&1; then rc='a rewrite of receipt/r1 was taken'; fi
    after="$(fingerprint "$s" "$H1" receipt r1; fingerprint "$s" "$H2" receipt r1)"
    if [ "$rc" = all-as-expected ] && [ "$before" = "$after" ]; then
        printf '  ok    %-66s the same, before and after\n' 'records added and a rewrite refused: earlier receipts read the same'; pass=$((pass + 1))
    else
        printf '  BROKE %-66s %s\n  before %s\n  after  %s\n' 'records added and a rewrite refused: earlier receipts read the same' "$rc" "$before" "$after"; fail=$((fail + 1))
    fi
    row 'verify after adding' 0 '6 checked, 0 refused' -- verify "$s" "$H1"

    echo "--- planted falsifiers: each must go RED, by name ---"
    s2="$tmp/planted"
    cp -a -- "$s" "$s2"
    cp -- "$s2/$H2/receipt/r1" "$s2/$H1/receipt/r9"
    ln -s -- "../../$H2/receipt/r1" "$s2/$H1/receipt/r8"
    printf 'x' >> "$s2/$H1/ruling/c1"
    mv -- "$s2/$H1/release-note/n1" "$s2/$H1/release-note/n2"
    plant "$s2/$H1/receipt/r1" "$H1" receipt r1 someone "$tmp/r1"
    printf 'just bytes\n' > "$s2/$H1/receipt/r5"
    plant "$s2/$H1/receipt/r4" "$H1" receipt r4 train "$tmp/empty"
    plant "$s2/$H1/receipt/r6" "$H1" receipt r6 train "$tmp/r1"
    sed -i -e '1s|.*|evidence-store-record/v0|' "$s2/$H1/receipt/r6"
    mkdir -p "$s2/$H1/receipts"
    : > "$s2/$H1/receipt/.x"
    b="$(sha256sum < "$tmp/r1")"
    { printf '%s\n' "$MAGIC" "h $H1" 'kind receipt' 'name r7'; printf 'writer tr\000ain\n'; printf '%s\n' "body-sha256 ${b%% *}" '--'; cat -- "$tmp/r1"; } > "$s2/$H1/receipt/r7"
    row 'FALSIFIER: get a receipt for another H filed under this H' 1 "FAIL  STORE receipt/r9 for H=111111111111 is recorded for H=$H2, not H=$H1" -- get "$s2" "$H1" receipt r9
    row 'FALSIFIER: get a symlink to the receipt of another H' 1 'FAIL  STORE receipt/r8 for H=111111111111 is a symlink' -- get "$s2" "$H1" receipt r8
    row 'FALSIFIER: get a record edited in place' 1 'FAIL  STORE ruling/c1 for H=111111111111 has a body that changed after it was written' -- get "$s2" "$H1" ruling c1
    row 'FALSIFIER: get a record filed under another name' 1 'FAIL  STORE release-note/n2 for H=111111111111 is filed as release-note/n2 but recorded as release-note/n1' -- get "$s2" "$H1" release-note n2
    row 'FALSIFIER: get a record from another writer' 1 "FAIL  STORE receipt/r1 for H=111111111111 was written by 'someone'" -- get "$s2" "$H1" receipt r1
    row 'FALSIFIER: get a file that is not a record' 1 'FAIL  STORE receipt/r5 for H=111111111111 is not a record: its header is incomplete' -- get "$s2" "$H1" receipt r5
    row 'FALSIFIER: get a record in another format' 1 'FAIL  STORE receipt/r6 for H=111111111111 is not a record: it has no evidence-store-record/v1 header' -- get "$s2" "$H1" receipt r6
    row 'FALSIFIER: get a record with an empty body' 1 'FAIL  STORE receipt/r4 for H=111111111111 has an empty body' -- get "$s2" "$H1" receipt r4
    row 'FALSIFIER: get a record whose header holds a NUL byte' 1 'FAIL  STORE receipt/r7 for H=111111111111 is not a record: its header is not, byte for byte, the header put writes' -- get "$s2" "$H1" receipt r7
    row 'FALSIFIER: verify names the receipt for another H' 1 "FAIL  STORE receipt/r9 for H=111111111111 is recorded for H=$H2, not H=$H1" -- verify "$s2" "$H1"
    row 'FALSIFIER: verify names the symlink' 1 'FAIL  STORE receipt/r8 for H=111111111111 is a symlink' -- verify "$s2" "$H1"
    row 'FALSIFIER: verify names the record edited in place' 1 'FAIL  STORE ruling/c1 for H=111111111111 has a body that changed' -- verify "$s2" "$H1"
    row 'FALSIFIER: verify names a directory that is not a kind' 1 "FAIL  STORE H=111111111111 holds 'receipts', which is not a record kind directory" -- verify "$s2" "$H1"
    row 'FALSIFIER: verify names a stray file' 1 'FAIL  STORE receipt/.x for H=111111111111 is not a record name' -- verify "$s2" "$H1"
    row 'FALSIFIER: verify counts every refusal' 1 '14 checked, 11 refused' -- verify "$s2" "$H1"
    row 'verify still passes the good records beside the bad ones' 1 'ok    STORE known-failure/k1 H=111111111111' -- verify "$s2" "$H1"
    body_row 'a good record beside the planted ones still reads' "$s2" "$H1" receipt r2 "$tmp/r2"
    row 'the plants under one H leave another H valid' 0 '3 checked, 0 refused' -- verify "$s2" "$H2"

    echo "--- links, directories and FIFOs in a store: refused, and nothing is written outside it ---"
    s3="$tmp/linked"; s6="$tmp/tmplinked"; out="$tmp/outside"
    row 'init a store to plant links in' 0 'is the store of writer train' -- init "$s3" train
    bash "$SCRIPT_PATH" init "$s6" train > /dev/null 2>&1
    mkdir -p "$out/hdir/receipt" "$out/kdir" "$out/ddir" "$out/tmpdir" "$s3/$H1/receipt/dd"
    plant "$out/hdir/receipt/r1" "$H2" receipt r1 train "$tmp/r1"
    plant "$out/kdir/c1" "$H1" ruling c1 train "$tmp/ru"
    plant "$out/same" "$H1" receipt rs train "$tmp/r1"
    ln -s -- "$out/hdir" "$s3/$H2"
    ln -s -- "$out/kdir" "$s3/$H1/ruling"
    ln -s -- "$out/same" "$s3/$H1/receipt/rs"
    ln -s -- "$out/ddir" "$s3/$H1/receipt/rd"
    mkfifo -- "$s3/$H1/receipt/ff"
    rmdir -- "$s6/.tmp"; ln -s -- "$out/tmpdir" "$s6/.tmp"
    before="$(find "$out" "$s3/$H1/receipt/dd" | sort)"
    row 'FALSIFIER: put through a symlinked H directory' 1 'the path to receipt/r2 for H=aaaaaaaaaaaa holds a symlink' -- put "$s3" "$H2" receipt r2 "$tmp/r2"
    row 'FALSIFIER: put through a symlinked kind directory' 1 'the path to ruling/c2 for H=111111111111 holds a symlink' -- put "$s3" "$H1" ruling c2 "$tmp/ru"
    row 'FALSIFIER: put onto a symlink to a record of the same bytes' 1 'receipt/rs for H=111111111111 is a symlink' -- put "$s3" "$H1" receipt rs "$tmp/r1"
    row 'FALSIFIER: put onto a symlink to a directory' 1 'receipt/rd for H=111111111111 is a symlink' -- put "$s3" "$H1" receipt rd "$tmp/r1"
    row 'FALSIFIER: put onto a directory' 1 'receipt/dd for H=111111111111 is not a file' -- put "$s3" "$H1" receipt dd "$tmp/r1"
    row 'FALSIFIER: put with a .tmp that links out of the store' 1 '.tmp is a symlink: an unfinished write stays in the store' -- put "$s6" "$H1" receipt r1 "$tmp/r1"
    after="$(find "$out" "$s3/$H1/receipt/dd" | sort)"
    if [ "$before" = "$after" ]; then
        printf '  ok    %-66s nothing\n' 'FALSIFIER: the refused puts wrote nothing outside the store'; pass=$((pass + 1))
    else
        printf '  BROKE %-66s\n  before %s\n  after  %s\n' 'FALSIFIER: the refused puts wrote nothing outside the store' "$before" "$after"; fail=$((fail + 1))
    fi
    row 'FALSIFIER: get through a symlinked H directory' 1 'the path to receipt/r1 for H=aaaaaaaaaaaa holds a symlink' -- get "$s3" "$H2" receipt r1
    row 'FALSIFIER: get through a symlinked kind directory' 1 'the path to ruling/c1 for H=111111111111 holds a symlink' -- get "$s3" "$H1" ruling c1
    row 'FALSIFIER: get a symlink to a directory' 1 'receipt/rd for H=111111111111 is a symlink' -- get "$s3" "$H1" receipt rd
    row 'FALSIFIER: get a directory' 1 'receipt/dd for H=111111111111 is not a file' -- get "$s3" "$H1" receipt dd
    row 'FALSIFIER: get a FIFO: refused, never read' 1 'receipt/ff for H=111111111111 is not a file' -- get "$s3" "$H1" receipt ff
    row 'FALSIFIER: verify a symlinked H directory' 1 'FAIL  STORE H=aaaaaaaaaaaa is a symlink' -- verify "$s3" "$H2"
    row 'FALSIFIER: verify names every link, directory and FIFO' 1 '5 checked, 5 refused' -- verify "$s3" "$H1"

    echo "--- WRITER and .tmp: a link is never adopted; a killed init is ---"
    s5="$tmp/writerlinked"
    mkdir -p "$s5" "$tmp/killed/.tmp" "$tmp/tmpmore/.tmp" "$tmp/tmplink"
    printf 'train\n' > "$tmp/wfile"
    ln -s -- "$tmp/wfile" "$s5/WRITER"
    printf 'other\n' > "$tmp/killed/.tmp/WRITER.abcdef"
    printf 'x\n' > "$tmp/tmpmore/x"
    ln -s -- "$out/tmpdir" "$tmp/tmplink/.tmp"
    row 'FALSIFIER: init where WRITER is a symlink' 1 'WRITER is a symlink: the store is not adopted' -- init "$s5" train
    row 'FALSIFIER: a store whose WRITER is a symlink is not a store' 3 'it has no WRITER' -- get "$s5" "$H1" receipt r1
    row 'init over a store whose first init was killed' 0 'is the store of writer train' -- init "$tmp/killed" train
    row 'FALSIFIER: init over .tmp and anything else' 1 'is not empty and has no WRITER' -- init "$tmp/tmpmore" train
    row 'FALSIFIER: init over a .tmp that is a symlink' 1 'is not empty and has no WRITER' -- init "$tmp/tmplink" train

    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# Planted defects: each must turn a row of the case table RED, or the case table proves nothing.
mutants() {
    local tmp pass=0 fail=0 name expr copy o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    while read -r name expr; do
        [ -n "$name" ] || continue
        copy="$tmp/$name.sh"
        sed -e "$expr" "$SCRIPT_PATH" > "$copy"
        if cmp -s "$SCRIPT_PATH" "$copy"; then
            printf '  BROKE %-34s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$copy" 2>/dev/null; then
            printf '  BROKE %-34s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$copy" --selftest < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-34s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"syntax error"*|*"command not found"*|*"unbound variable"*) printf '  BROKE %-34s the mutant does not run: a RED from it proves nothing\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-34s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-34s exit %s with no broken row: not a kill\n%s\n' "$name" "$rc" "$o"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
h_not_checked               /^check_record() {$/,/^}$/s/if \[ "\$l2" != "h \$h" \]; then/if false; then/
kind_name_not_checked       /^check_record() {$/,/^}$/s/if \[ "\$l3" != "kind \$kind" \] || \[ "\$l4" != "name \$name" \]; then/if false; then/
writer_not_checked          /^check_record() {$/,/^}$/s/if \[ "\$l5" != "writer \$W" \]; then/if false; then/
body_not_hashed             /^check_record() {$/,/^}$/s/if \[ "\$l6" != "body-sha256 \$sum" \]; then/if false; then/
empty_body_kept             /^check_record() {$/,/^}$/s/if \[ "\$sum" = "\$EMPTY_SHA256" \]; then/if false; then/
symlink_followed            /^check_record() {$/,/^}$/s/if \[ -L "\$f" \]; then/if false; then/
get_follows_symlink         /^cmd_get() {$/,/^}$/s/if \[ -L "\$f" \]; then refuse/if false; then refuse/
header_not_required         /^check_record() {$/,/^}$/s/if \[ "\$l1" != "\$MAGIC" \] || \[ "\$l7" != '--' \]; then/if false; then/
put_overwrites              /^cmd_put() {$/,/^}$/s/if ! ln -T -- "\$TMPF" "\$f" 2>\/dev\/null; then/if ! ln -T -f -- "$TMPF" "$f" 2>\/dev\/null; then/
unchanged_without_compare   /^cmd_put() {$/,/^}$/s/if cmp -s -- "\$TMPF" "\$f"; then/if true; then/
put_takes_empty             /^cmd_put() {$/,/^}$/s/if \[ "\$sum" = "\$EMPTY_SHA256" \]; then refuse/if false; then refuse/
any_writer                  /^cmd_put() {$/,/^}$/s/if \[ "\$me" != "\$W" \]; then/if false; then/
others_not_named            /^cmd_get() {$/,/^}$/s/if is_h "\$o" && \[ "\$o" != "\$h" \] &&/if false \&\&/
empty_verify_passes         /^cmd_verify() {$/,/^}$/s/if \[ "\$n" -eq 0 \]; then/if false; then/
verify_exit_ignores_refusal /^cmd_verify() {$/,/^}$/s/if \[ "\$bad" -ne 0 \]; then return 1; fi/if false; then return 1; fi/
stray_names_read            /^cmd_verify() {$/,/^}$/s/if ! is_name "\$name"; then/if false; then/
non_kinds_skipped           /^cmd_verify() {$/,/^}$/s/if \[ -L "\$kd" \] || \[ ! -d "\$kd" \] || ! is_kind "\$k"; then/if false; then/
h_any_case                  /^is_h() {$/,/^}$/s/''|\*\[!0-9a-f\]\*) return 1 ;;/'') return 1 ;;/
h_any_length                /^is_h() {$/,/^}$/s/\[ "\${#1}" -eq 40 \] || \[ "\${#1}" -eq 64 \]/true/
name_takes_dots             /^is_name() {$/,/^}$/s/''|\.\*|\*\[!A-Za-z0-9\._-\]\*) return 1 ;;/''|*[!A-Za-z0-9._-]*) return 1 ;;/
init_adopts_nonempty        /^cmd_init() {$/,/^}$/s/if \[ "\${#e\[@\]}" -gt 0 \]; then/if false; then/
init_second_writer          /^cmd_init() {$/,/^}$/s/if \[ "\$w" = "\$id" \]; then/if true; then/
kind_any                    /^is_kind() {$/,/^}$/s/receipt|ruling|known-failure|release-note) return 0 ;;/*) return 0 ;;/
get_skips_check             /^cmd_get() {$/,/^}$/s/if ! why="\$(check_record "\$TMPF" "\$h" "\$kind" "\$name")"; then refuse/if false; then refuse/
verify_skips_check          /^cmd_verify() {$/,/^}$/s/if out="\$(check_record "\$f" "\$h" "\$k" "\$name")"; then/if out="$(check_record "$f" "$h" "$k" "$name")" || true; then/
get_follows_h_symlink       /^cmd_get() {$/,/^}$/s/if \[ -L "\$store\/\$h" \] || \[ -L "\$store\/\$h\/\$kind" \]; then refuse/if [ -L "$store\/$h\/$kind" ]; then refuse/
get_follows_kind_symlink    /^cmd_get() {$/,/^}$/s/if \[ -L "\$store\/\$h" \] || \[ -L "\$store\/\$h\/\$kind" \]; then refuse/if [ -L "$store\/$h" ]; then refuse/
put_follows_h_symlink       /^cmd_put() {$/,/^}$/s/if \[ -L "\$store\/\$h" \] || \[ -L "\$store\/\$h\/\$kind" \]; then refuse/if [ -L "$store\/$h\/$kind" ]; then refuse/
put_follows_kind_symlink    /^cmd_put() {$/,/^}$/s/if \[ -L "\$store\/\$h" \] || \[ -L "\$store\/\$h\/\$kind" \]; then refuse/if [ -L "$store\/$h" ]; then refuse/
put_follows_tmp_symlink     /^cmd_put() {$/,/^}$/s/if \[ -L "\$store\/.tmp" \]; then refuse/if false; then refuse/
put_symlink_dest_taken      /^cmd_put() {$/,/^}$/s/if \[ -L "\$f" \]; then refuse/if false; then refuse/
ln_without_T                /^cmd_put() {$/,/^}$/s/if ! ln -T -- "\$TMPF" "\$f" 2>\/dev\/null; then/if ! ln -- "$TMPF" "$f" 2>\/dev\/null; then/
put_dir_dest_not_refused    /^cmd_put() {$/,/^}$/s/if \[ ! -f "\$f" \]; then refuse/if false; then refuse/
get_reads_non_file          /^cmd_get() {$/,/^}$/s/if \[ ! -f "\$f" \]; then refuse/if false; then refuse/
empty_put_makes_dirs        /^cmd_put() {$/,/^}$/s/if ! mkdir -p -- "\$store\/.tmp"; then/if ! mkdir -p -- "$store\/.tmp" "$store\/$h\/$kind"; then/
foreign_h_shortened         /^cmd_get() {$/,/^}$/s/others="\$others\${others:+, }H=\$o"/others="$others${others:+, }H=$(short "$o")"/
record_h_shortened          /^check_record() {$/,/^}$/s/echo "is recorded for H=\$oh, not H=\$h:/echo "is recorded for H=$(short "$oh"), not H=$(short "$h"):/
header_bytes_not_compared   /^check_record() {$/,/^}$/s/if \[ "\$hs" != /if false \&\& [ "$hs" != /
name_any_length             /^is_name() {$/,/^}$/s/\[ "\${#1}" -le 128 \]/true/
name_limit_off_by_one       /^is_name() {$/,/^}$/s/-le 128/-lt 128/
verify_follows_h_symlink    /^cmd_verify() {$/,/^}$/s/if \[ -L "\$dir" \]; then refuse/if false; then refuse/
init_follows_writer_symlink /^cmd_init() {$/,/^}$/s/if \[ -L "\$store\/WRITER" \]; then refuse/if false; then refuse/
load_writer_follows_symlink /^load_writer() {$/,/^}$/s/if \[ -L "\$1\/WRITER" \] || \[ ! -f "\$1\/WRITER" \]; then/if [ ! -f "$1\/WRITER" ]; then/
init_counts_tmp             /^cmd_init() {$/,/^}$/s/; then e=(); fi$/; then :; fi/
init_ignores_tmp_and_more   /^cmd_init() {$/,/^}$/s/if \[ "\${#e\[@\]}" -eq 1 \] &&/if [ "${#e[@]}" -ge 1 ] \&\&/
init_takes_tmp_symlink      /^cmd_init() {$/,/^}$/s/ && \[ ! -L "\${e\[0\]}" \]; then e=()/; then e=()/
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    init) shift; cmd_init "$@" ;;
    put) shift; cmd_put "$@" ;;
    get) shift; cmd_get "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    --selftest) selftest ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    '') caller_error "no command (try -h)" ;;
    *) caller_error "unknown command '$1' (try -h)" ;;
esac
