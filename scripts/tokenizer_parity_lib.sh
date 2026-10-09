#!/usr/bin/env bash
# tokenizer_parity_lib.sh - comparison helpers shared by the #3726 tokenizer-parity gate
# (scripts/tokenizer_parity.sh, model-backed, release time) and its PR-time case table
# (scripts/check_tokenizer_parity.sh). A SOURCED library: it sets no shell options and
# reports by return status, so sourcing it cannot change the caller's shell.
#
# Pure bash on purpose: CI clean-rooms carry no jq, and the fleet is python-free for
# automation (infra#708).

# tp_llama_ids TEXT -> the ids of `llama-tokenize --ids` output ("[1, 2, 3]"), space-separated.
tp_llama_ids() {
    local s=${1-}
    s=${s//[/}
    s=${s//]/}
    s=${s//,/ }
    local -a ids
    read -r -a ids <<<"$s"
    printf '%s' "${ids[*]}"
}

# tp_first_mismatch A B -> the 0-based index of the first position where the space-separated
# id lists differ, or -1 when they are identical. A length difference is a mismatch at the
# shorter list's length.
tp_first_mismatch() {
    local -a a b
    read -r -a a <<<"${1-}"
    read -r -a b <<<"${2-}"
    local n=${#a[@]} m=${#b[@]} i
    local short=$n
    [ "$m" -lt "$short" ] && short=$m
    for ((i = 0; i < short; i++)); do
        if [ "${a[i]}" != "${b[i]}" ]; then
            printf '%s' "$i"
            return 0
        fi
    done
    if [ "$n" -ne "$m" ]; then
        printf '%s' "$short"
    else
        printf '%s' -1
    fi
}

# tp_window LIST INDEX -> up to 4 ids either side of INDEX, for a failure message.
tp_window() {
    local -a a
    read -r -a a <<<"${1-}"
    local i=${2:-0} lo hi out="" k
    lo=$((i - 4))
    [ "$lo" -lt 0 ] && lo=0
    hi=$((i + 4))
    [ "$hi" -ge "${#a[@]}" ] && hi=$((${#a[@]} - 1))
    for ((k = lo; k <= hi; k++)); do
        if [ "$k" -eq "$i" ]; then out+="[${a[k]}] "; else out+="${a[k]} "; fi
    done
    printf '%s' "${out% }"
}

# tp_apr_status STDERR -> "path|roundtrip" parsed from `apr tokenize encode`'s status line
# ("N ids; path: P; roundtrip: R"). Empty when the line is absent.
tp_apr_status() {
    local line path rt
    line=$(printf '%s\n' "${1-}" | grep -E '^[0-9]+ ids; path: .*; roundtrip: (true|false)$' | tail -1)
    [ -n "$line" ] || return 1
    path=${line#*; path: }
    path=${path%; roundtrip: *}
    rt=${line##*; roundtrip: }
    printf '%s|%s' "$path" "$rt"
}

# tp_is_canonical PATH -> status 0 iff the encoder took the canonical byte-level BPE path.
tp_is_canonical() {
    [ "${1-}" = "canonical" ]
}

# tp_commit_matches WANT VERSION_LINE -> status 0 iff the `commit <hex>` in a llama.cpp
# `--version` line names the pinned commit WANT. git abbreviates a hash by clone size, so the
# same build prints `d1d3c3396` on one host and `d1d3c339` on another (gx10, measured
# 2026-09-21): the two match when one is a prefix of the other and the shorter has at least 7
# hex digits (git's minimum abbreviation). Anything shorter is ambiguous and refused.
tp_commit_matches() {
    local want=${1-} got
    got=$(printf '%s\n' "${2-}" | sed -n 's/.*commit \([0-9a-f]\{1,\}\).*/\1/p' | head -1)
    [ -n "$want" ] && [ -n "$got" ] || return 1
    local short=$got
    [ ${#want} -lt ${#short} ] && short=$want
    [ ${#short} -ge 7 ] || return 1
    case "$want" in "$got"*) return 0 ;; esac
    case "$got" in "$want"*) return 0 ;; esac
    return 1
}


# tp_inventory_rows FILE -> the model rows of a declared inventory (#4981), one "<sha256> <name>"
# per line. The file is sha256sum format, "<64 lowercase hex>  <file name>", with '#' comment
# and blank lines allowed. A name is a bare *.gguf file name, never a path. Status 1, naming the
# reason on stderr, when the file cannot be read, a line is malformed, a name is listed twice, or
# no model is listed: an inventory the gate cannot read is a refusal, never an empty pass.
tp_inventory_rows() {
    local file=${1-} line n=0 seen=" "
    [ -f "$file" ] && [ -r "$file" ] || { printf 'inventory %s: cannot read it\n' "$file" >&2; return 1; }
    local re='^([0-9a-f]{64})  ([A-Za-z0-9._+-]+\.gguf)$'
    local out=""
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in '' | '#'*) continue ;; esac
        if ! [[ $line =~ $re ]]; then
            printf 'inventory %s: malformed line %q\n' "$file" "$line" >&2
            return 1
        fi
        case "$seen" in *" ${BASH_REMATCH[2]} "*)
            printf 'inventory %s: %s is listed twice\n' "$file" "${BASH_REMATCH[2]}" >&2
            return 1 ;;
        esac
        seen="$seen${BASH_REMATCH[2]} "
        out="$out${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"$'\n'
        n=$((n + 1))
    done <"$file"
    [ "$n" -gt 0 ] || { printf 'inventory %s: lists no model\n' "$file" >&2; return 1; }
    printf '%s' "$out"
}

# tp_resolve_inventory DIR < ROWS -> one verdict per "<sha256> <name>" row of tp_inventory_rows:
#   ok NAME                      DIR/NAME is there and its sha256 is the declared one
#   absent NAME                  no regular file DIR/NAME
#   mismatch NAME GOT_SHA256     the file is there with another sha256
#   unhashable NAME              sha256sum could not read it
# Status 0 iff every row is ok. A file in DIR that no row names is never looked at.
# A verdict never carries DIR: a name has no whitespace (tp_inventory_rows' pattern) and a
# sha256 is hex, so `read -r verdict name sha` splits every verdict whatever DIR holds. The
# caller joins DIR/NAME itself.
tp_resolve_inventory() {
    local dir=${1-} want name got verdict bad=0
    while read -r want name; do
        [ -n "$want" ] || continue
        if [ ! -f "$dir/$name" ]; then
            verdict="absent $name"
        elif ! got=$(sha256sum -- "$dir/$name" 2>/dev/null) || [ -z "$got" ]; then
            verdict="unhashable $name"
        elif [ "${got%% *}" != "$want" ]; then
            verdict="mismatch $name ${got%% *}"
        else
            verdict="ok $name"
        fi
        printf '%s\n' "$verdict"
        case "$verdict" in ok\ *) ;; *) bad=1 ;; esac
    done
    return "$bad"
}
