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

