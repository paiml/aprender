#!/usr/bin/env bash
# #4026 positive control, offline: the comparator in c4026_live.sh, fed the
# oracle against ITSELF (the oracle's jsonl reshaped into apr's response shape),
# must pass with max |Δlogprob| = 0. If it cannot pass on identical data, a RED
# from it says nothing about apr. The comparator is not copied: it is extracted
# from c4026_live.sh at run time, so this tests the exact pre-registered bar.
# Usage: positive_control.sh [DIR]   (DIR defaults to this receipt's directory)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dir="${1:-$here}"
cmp_fn="$(sed -n '/^compare() {$/,/^}$/p' "$here/c4026_live.sh")"
[ -n "$cmp_fn" ] || { echo "RED: compare() not found in c4026_live.sh"; exit 2; }
eval "$cmp_fn"
rc=0
for t in p2 p3 p5; do
    # oracle jsonl -> {choices:[{logprobs:{content:[{token_id,logprob,top_logprobs:[{token_id,logprob}]}]}}]}
    jq -s '{choices:[{logprobs:{content:[.[] | {token_id:.id, logprob:.logprob,
            top_logprobs:[.top_logprobs[] | {token_id:.id, logprob:.logprob}]}]}}]}' \
        "$dir/$t.oracle.jsonl" >"$dir/$t.oracle_as_apr.json"
    compare "$dir/$t.oracle_as_apr.json" "$dir/$t.oracle.jsonl" 0 >"$dir/$t.posctl.json"
    pass=$(jq .pass "$dir/$t.posctl.json")
    maxd=$(jq '[.steps[].max_abs_dlogprob] | max' "$dir/$t.posctl.json")
    echo "$t positive control: pass=$pass steps=$(jq '.steps|length' "$dir/$t.posctl.json") max_abs_dlogprob=$maxd"
    [ "$pass" = true ] && [ "$maxd" = 0 ] || rc=1
done
exit "$rc"
