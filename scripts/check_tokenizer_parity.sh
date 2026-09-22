#!/usr/bin/env bash
# check_tokenizer_parity.sh - the PR-time half of the #3726 tokenizer-parity gate.
#
# scripts/tokenizer_parity.sh compares the token ids `apr tokenize encode` gives each corpus
# file with the pinned llama.cpp's `llama-tokenize`, model by model. That run needs the GGUF
# models and the llama.cpp build, which CI runners do not carry, so it runs at release time:
# it is declared in Cargo.toml [package.metadata.dogfood] and executed by
# `scripts/dogfood.sh --phase pre-publish`.
#
# This is what scripts/guard_tree.sh runs on every pull request: the case table of the
# comparison helpers both halves share (scripts/tokenizer_parity_lib.sh). It needs no models
# and runs in well under a second. Without it the comparison logic could rot between
# releases and still look wired. Exit 0 iff every row holds.
set -euo pipefail

# shellcheck source=tokenizer_parity_lib.sh
. "$(dirname "$0")/tokenizer_parity_lib.sh" || exit 2

fails=0
row() { # name got want
    if [ "$2" = "$3" ]; then
        printf 'ok    %s\n' "$1"
    else
        printf 'FAIL  %s: got %q, want %q\n' "$1" "$2" "$3"
        fails=$((fails + 1))
    fi
}

# llama-tokenize --ids output, normalised
row "llama ids: bracketed list" "$(tp_llama_ids '[760, 893, 31913]')" "760 893 31913"
row "llama ids: single id" "$(tp_llama_ids '[42]')" "42"
row "llama ids: empty list" "$(tp_llama_ids '[]')" ""
row "llama ids: trailing newline" "$(tp_llama_ids $'[1, 2]\n')" "1 2"

# first mismatch
row "identical lists" "$(tp_first_mismatch '1 2 3' '1 2 3')" "-1"
row "differ in the middle" "$(tp_first_mismatch '1 2 3' '1 9 3')" "1"
row "apr shorter" "$(tp_first_mismatch '1 2' '1 2 3')" "2"
row "apr longer" "$(tp_first_mismatch '1 2 3 4' '1 2 3')" "3"
row "both empty" "$(tp_first_mismatch '' '')" "-1"
# the #3726 mutant: the old encoder's id-0 byte fallback, one byte of U+2500 as 0
row "id-0 byte fallback is a mismatch" "$(tp_first_mismatch '52453 0 0 0' '52453 54907')" "1"
# the segmentation defect: " quorum" as Ġquo|rum (apr) vs Ġqu|orum (llama.cpp)
row "greedy segmentation is a mismatch" "$(tp_first_mismatch '760 39170 10405' '760 893 31913')" "1"

# failure window
row "window marks the index" "$(tp_window '1 2 3 4 5 6 7 8 9 10' 5)" "2 3 4 5 [6] 7 8 9 10"
row "window clips at the start" "$(tp_window '1 2 3' 0)" "[1] 2 3"

# apr status line
row "apr status: canonical" "$(tp_apr_status '3 ids; path: canonical; roundtrip: true')" "canonical|true"
fallback_status="noise
5 ids; path: greedy-fallback: pre-tokenizer 'llama-bpe' is not implemented; roundtrip: false"
row "apr status: fallback with reason" "$(tp_apr_status "$fallback_status")" \
    "greedy-fallback: pre-tokenizer 'llama-bpe' is not implemented|false"
if tp_apr_status 'no status line here' >/dev/null; then
    row "apr status: absent line is refused" "parsed" "refused"
else
    row "apr status: absent line is refused" "refused" "refused"
fi
if tp_is_canonical canonical; then row "canonical path accepted" yes yes; else row "canonical path accepted" no yes; fi
if tp_is_canonical 'greedy-fallback: x'; then row "fallback path refused" no yes; else row "fallback path refused" yes yes; fi

# pinned-commit match: same build, abbreviated differently per host
cm() { if tp_commit_matches "$1" "$2"; then printf 'match'; else printf 'refuse'; fi; }
row "commit: exact 9 digits" "$(cm d1d3c3396 'version: 0.4.1-dev (build 10987, commit d1d3c3396)')" match
row "commit: gx10's 8-digit abbreviation" "$(cm d1d3c3396 'version: 0.4.1-dev (build 2423, commit d1d3c339)')" match
row "commit: a longer hash of the pin" "$(cm d1d3c3396 'version: 1 (build 1, commit d1d3c3396abcdef)')" match
row "commit: another build" "$(cm d1d3c3396 'version: 0.4.1-dev (build 7746, commit 39173bcac)')" refuse
row "commit: 6 digits is ambiguous" "$(cm d1d3c3396 'version: x (build 1, commit d1d3c3)')" refuse
row "commit: no commit in the line" "$(cm d1d3c3396 'version: 0.4.1-dev')" refuse
row "commit: empty pin" "$(cm '' 'version: x (build 1, commit d1d3c3396)')" refuse

printf -- '--- case table: %s failure(s) ---\n' "$fails"
[ "$fails" -eq 0 ]
