#!/usr/bin/env bash
# tokenizer_parity.sh - #3726: apr's token ids against the pinned llama.cpp, model by model.
#
# For every (model, corpus file) pair, compare the ids `apr tokenize encode MODEL -f FILE`
# prints with the ids the pinned `llama-tokenize` prints for the same bytes:
#   PASS       identical ids, apr took the canonical path, decode(encode(x)) == x
#   FAIL       any id differs (the index and a window of ids either side are printed), apr
#              took the greedy fallback on a byte-level vocabulary, or the round trip failed
#   UNCOVERED  the model's vocabulary is not byte-level BPE (tokenizer.ggml.model is not
#              gpt2): nothing canonical exists in apr to compare. Counted against the gate,
#              never passed: the fleet's release inventory must be covered.
# Exit 0 iff every pair PASSes. Exit 1 on any FAIL or UNCOVERED. Exit 2 when the comparison
# cannot be made honestly: no apr, no pinned comparator, no model or no corpus.
#
# WHY IDS AND NOT TEXT. apr and llama.cpp could both decode a wrong split back to the same
# text; the model sees ids. `quorum` split as Ġquo|rum reads fine and was quoted back as
# "quoorum" (#3693). Only an id comparison sees that, and `apr parity` (CPU vs GPU, one shared
# tokenizer) cannot see it at all.
#
# THE EXACT llama-tokenize INVOCATION, and why each flag:
#   --ids         the id list, not id/piece pairs
#   --no-escape   by default llama-tokenize turns a literal \n, \t, \" ... in the FILE into
#                 control characters (tools/tokenize/tokenize.cpp: params.escape, default on),
#                 which would compare apr on the file against llama.cpp on different bytes
#   --no-bos      `apr tokenize encode` prints the encoder's ids; BOS is its callers' job
#   (default)     parse_special on, as apr's encoder: <|im_start|>, <think> ... are one id
#
# A HOST-SIDE GATE, NOT A TREE GUARD. It needs the models and the pinned llama.cpp build,
# which CI runners do not carry, so it is not named check_*.sh (scripts/guard_tree.sh runs
# those bare, in a required job). Its PR-time half is scripts/check_tokenizer_parity.sh, the
# case table of the helpers both share. It measures agreement only.
#
# THE COMPARATOR IS PINNED. llama-tokenize must report `commit <build_commit>` from
# scripts/llama_pin.toml (git's abbreviation may be shorter on another host, down to 7 hex
# digits: tp_commit_matches). Any other build is refused (rc 2).
#
# Usage:
#   scripts/tokenizer_parity.sh [--apr BIN] [--llama-tokenize BIN] [--model GGUF]... [--corpus FILE]...
# Defaults: --apr from scripts/apr_bin.sh (built from HEAD); --llama-tokenize from
# $LLAMA_TOKENIZE, else ~/src/llama.cpp-<build_commit>/build/bin/llama-tokenize; --model every
# *.gguf in ${APR_MODEL_DIR:-$HOME/models} except shards 2+ of a split file (they carry no
# tokenizer); --corpus every file in evidence/tokenizer-parity/corpus/.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
# shellcheck source=tokenizer_parity_lib.sh
. "$here/tokenizer_parity_lib.sh" || { printf 'REFUSE: cannot source tokenizer_parity_lib.sh\n' >&2; exit 2; }
# llama_bin.sh resolves the llama-BENCH binary as it is sourced and returns that verdict.
# This gate needs only its pin reader, so the bench verdict is not this gate's to act on.
# shellcheck source=llama_bin.sh
. "$here/llama_bin.sh" || true
declare -F llama_pin_get >/dev/null || { printf 'REFUSE: scripts/llama_bin.sh defines no llama_pin_get\n' >&2; exit 2; }

apr_bin=""
lt_bin="${LLAMA_TOKENIZE:-}"
models=()
corpus=()
while [ $# -gt 0 ]; do
    case "$1" in
        --apr) apr_bin=${2:?--apr needs a path}; shift 2 ;;
        --llama-tokenize) lt_bin=${2:?--llama-tokenize needs a path}; shift 2 ;;
        --model) models+=("${2:?--model needs a path}"); shift 2 ;;
        --corpus) corpus+=("${2:?--corpus needs a path}"); shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) printf 'tokenizer_parity: unknown argument %q\n' "$1" >&2; exit 2 ;;
    esac
done

if [ -z "$apr_bin" ]; then
    # shellcheck source=apr_bin.sh
    if ! . "$here/apr_bin.sh"; then
        printf 'REFUSE: no apr built from HEAD (scripts/apr_bin.sh); pass --apr\n' >&2
        exit 2
    fi
    apr_bin=$APR
fi
[ -x "$apr_bin" ] || { printf 'REFUSE: apr %s is not executable\n' "$apr_bin" >&2; exit 2; }

want_commit=$(llama_pin_get build_commit "$root/scripts/llama_pin.toml")
[ -n "$want_commit" ] || { printf 'REFUSE: no build_commit in scripts/llama_pin.toml\n' >&2; exit 2; }
[ -n "$lt_bin" ] || lt_bin="$HOME/src/llama.cpp-$want_commit/build/bin/llama-tokenize"
if [ ! -x "$lt_bin" ]; then
    printf 'REFUSE: no llama-tokenize at %s (build the pinned %s tree with --target llama-tokenize, or pass --llama-tokenize)\n' "$lt_bin" "$want_commit" >&2
    exit 2
fi
lt_version=$("$lt_bin" --version 2>&1 | grep -i '^version:' | head -1)
if ! tp_commit_matches "$want_commit" "$lt_version"; then
    printf 'REFUSE: %s reports %q, not the pinned commit %s\n' "$lt_bin" "$lt_version" "$want_commit" >&2
    exit 2
fi

if [ ${#models[@]} -eq 0 ]; then
    for m in "${APR_MODEL_DIR:-$HOME/models}"/*.gguf; do
        [ -e "$m" ] || continue
        case "$m" in *-0000[2-9]-of-0000[0-9].gguf) continue ;; esac
        models+=("$m")
    done
fi
if [ ${#corpus[@]} -eq 0 ]; then
    for f in "$root"/evidence/tokenizer-parity/corpus/*; do
        [ -f "$f" ] && corpus+=("$f")
    done
fi
[ ${#models[@]} -gt 0 ] || { printf 'REFUSE: no models\n' >&2; exit 2; }
[ ${#corpus[@]} -gt 0 ] || { printf 'REFUSE: no corpus files\n' >&2; exit 2; }

printf 'apr:            %s (%s)\n' "$apr_bin" "$("$apr_bin" --version 2>/dev/null | head -1)"
printf 'llama-tokenize: %s (%s)\n' "$lt_bin" "$lt_version"
printf 'models: %s   corpus files: %s\n' "${#models[@]}" "${#corpus[@]}"

tmp=$(mktemp -d) || exit 2
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 uncovered=0
for m in "${models[@]}"; do
    mname=$(basename "$m")
    for f in "${corpus[@]}"; do
        fname=$(basename "$f")
        apr_ids=$("$apr_bin" tokenize encode "$m" -f "$f" 2>"$tmp/apr.err")
        apr_rc=$?
        if [ "$apr_rc" -ne 0 ]; then
            printf 'FAIL       %s  %s  apr rc=%s: %s\n' "$mname" "$fname" "$apr_rc" "$(tail -1 "$tmp/apr.err")"
            fail=$((fail + 1))
            continue
        fi
        if ! status=$(tp_apr_status "$(cat "$tmp/apr.err")"); then
            printf 'FAIL       %s  %s  apr printed no status line\n' "$mname" "$fname"
            fail=$((fail + 1))
            continue
        fi
        path=${status%|*}
        roundtrip=${status##*|}
        case "$path" in
            *"not a byte-level vocabulary"*)
                printf 'UNCOVERED  %s  %s  %s\n' "$mname" "$fname" "$path"
                uncovered=$((uncovered + 1))
                continue ;;
        esac
        if ! tp_is_canonical "$path"; then
            printf 'FAIL       %s  %s  apr took %s\n' "$mname" "$fname" "$path"
            fail=$((fail + 1))
            continue
        fi
        "$lt_bin" -m "$m" -f "$f" --ids --no-escape --no-bos --log-disable >"$tmp/lt.out" 2>"$tmp/lt.err"
        lt_rc=$?
        lt_out=$(tail -1 "$tmp/lt.out")
        if [ "$lt_rc" -ne 0 ]; then
            printf 'FAIL       %s  %s  llama-tokenize rc=%s: %s\n' "$mname" "$fname" "$lt_rc" "$(tail -1 "$tmp/lt.err")"
            fail=$((fail + 1))
            continue
        fi
        lt_ids=$(tp_llama_ids "$lt_out")
        idx=$(tp_first_mismatch "$apr_ids" "$lt_ids")
        n_apr=$(wc -w <<<"$apr_ids")
        n_lt=$(wc -w <<<"$lt_ids")
        if [ "$idx" != "-1" ]; then
            printf 'FAIL       %s  %s  first id mismatch at %s (apr %s ids, llama.cpp %s)\n' "$mname" "$fname" "$idx" "$n_apr" "$n_lt"
            printf '             apr:       %s\n' "$(tp_window "$apr_ids" "$idx")"
            printf '             llama.cpp: %s\n' "$(tp_window "$lt_ids" "$idx")"
            fail=$((fail + 1))
        elif [ "$roundtrip" != "true" ]; then
            printf 'FAIL       %s  %s  ids match but decode(encode(x)) != x\n' "$mname" "$fname"
            fail=$((fail + 1))
        else
            printf 'PASS       %s  %s  %s ids\n' "$mname" "$fname" "$n_apr"
            pass=$((pass + 1))
        fi
    done
done

printf -- '--- tokenizer parity: %s pass, %s fail, %s uncovered ---\n' "$pass" "$fail" "$uncovered"
[ "$fail" -eq 0 ] && [ "$uncovered" -eq 0 ]
