#!/usr/bin/env bash
# #4026 live control: apr serve chat top_logprobs vs a llama.cpp oracle on the
# official template, teacher-forced per step on the ids apr generated. CPU only.
# Usage: c4026_live.sh APR_BIN MODEL OUT_DIR
set -euo pipefail
APR="$1"; MODEL="$2"; OUT="$3"
LLAMA="$HOME/src/llama.cpp/build/bin/llama-server"
PA=18432; PL=18431; K=20; N=8
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

"$LLAMA" -m "$MODEL" --port "$PL" -ngl 0 --jinja -t 8 -c 4096 >"$OUT/llama.log" 2>&1 &
pids+=("$!")
"$APR" serve run "$MODEL" --port "$PA" --no-gpu >"$OUT/apr.log" 2>&1 &
pids+=("$!")
echo "pids ${pids[*]}" >"$OUT/pids"
for u in "http://127.0.0.1:$PL/health" "http://127.0.0.1:$PA/health"; do
    for _ in $(seq 1 180); do curl -sf "$u" >/dev/null && break; sleep 1; done
    curl -sf "$u" >/dev/null || { echo "RED: $u never healthy"; exit 2; }
done

run_prompt() { # $1 tag, $2 user text
    local tag="$1" text="$2"
    local msgs; msgs=$(jq -nc --arg t "$text" '[{role:"user",content:$t}]')
    curl -sf "http://127.0.0.1:$PA/v1/chat/completions" -H 'content-type: application/json' \
        -d "$(jq -nc --argjson m "$msgs" --argjson k "$K" --argjson n "$N" \
            '{model:"q",messages:$m,max_tokens:$n,temperature:0,logprobs:true,top_logprobs:$k}')" \
        >"$OUT/$tag.apr.json"
    # The oracle's prompt: the GGUF's own jinja template (thinking as apr's default).
    local think; think=$(jq -r '.choices[0].message.content' "$OUT/$tag.apr.json" | grep -c '<think>' || true)
    curl -sf "http://127.0.0.1:$PL/apply-template" -H 'content-type: application/json' \
        -d "$(jq -nc --argjson m "$msgs" '{messages:$m, chat_template_kwargs:{enable_thinking:false}}')" >"$OUT/$tag.tmpl.json"
    curl -sf "http://127.0.0.1:$PL/tokenize" -H 'content-type: application/json' \
        -d "$(jq -c '{content:.prompt, add_special:false, parse_special:true}' "$OUT/$tag.tmpl.json")" \
        >"$OUT/$tag.ids.json"
    local n_prompt_llama n_prompt_apr
    n_prompt_llama=$(jq '.tokens|length' "$OUT/$tag.ids.json")
    n_prompt_apr=$(jq '.usage.prompt_tokens' "$OUT/$tag.apr.json")
    echo "$tag prompt_tokens apr=$n_prompt_apr llama=$n_prompt_llama think_tag=$think"
    local gen; gen=$(jq -c '[.choices[0].logprobs.content[].token_id]' "$OUT/$tag.apr.json")
    local steps; steps=$(jq 'length' <<<"$gen")
    : >"$OUT/$tag.oracle.jsonl"
    for i in $(seq 0 $((steps - 1))); do
        local prompt; prompt=$(jq -c --argjson g "$gen" --argjson i "$i" '.tokens + $g[:$i]' "$OUT/$tag.ids.json")
        curl -sf "http://127.0.0.1:$PL/completion" -H 'content-type: application/json' \
            -d "$(jq -nc --argjson p "$prompt" --argjson k "$K" \
                '{prompt:$p,n_predict:1,n_probs:$k,temperature:0,post_sampling_probs:false,cache_prompt:false}')" \
            | jq -c '.completion_probabilities[0]' >>"$OUT/$tag.oracle.jsonl"
    done
}

# compare APR_JSON ORACLE_JSONL SHIFT -> one line per step + verdict line
compare() {
    jq -n --slurpfile a "$1" --slurpfile o "$2" --argjson s "$3" '
      [$a[0].choices[0].logprobs.content, $o] as [$ac, $oc]
      | [range(0; ($ac|length) - $s) as $i
         | ($ac[$i]) as $x | ($oc[$i + $s]) as $y
         | ($y.top_logprobs | map({key: (.id|tostring), value: .logprob}) | from_entries) as $om
         | ($x.top_logprobs | map(select($om[(.token_id|tostring)] != null)
             | (.logprob - $om[(.token_id|tostring)]) | fabs)) as $d
         | {step: $i,
            top1_apr: $x.top_logprobs[0].token_id, top1_oracle: $y.top_logprobs[0].id,
            overlap: ($d|length),
            max_abs_dlogprob: ($d|max // null),
            chosen_dlogprob: (($x.logprob - ($om[($x.token_id|tostring)] // -1e9))|fabs),
            apr_mass: ([$x.top_logprobs[].logprob | exp] | add)}]
      | {steps: ., pass: (length > 0 and all(.[];
            .top1_apr == .top1_oracle and .overlap >= 16
            and .max_abs_dlogprob <= 0.25 and .chosen_dlogprob <= 0.25 and .apr_mass <= 1.0001))}'
}

run_prompt p1 "What is the capital of France? Answer in one word."
run_prompt p2 "Write the first three prime numbers, separated by commas."
run_prompt p3 "Name a primary color."
run_prompt p4 "Translate hello into Spanish."
run_prompt p5 "What is 12 times 12?"
run_prompt p6 "Say the word blue."
for t in p3 p4 p5 p6; do compare "$OUT/$t.apr.json" "$OUT/$t.oracle.jsonl" 0 >"$OUT/$t.cmp.json"; done
for t in p1 p2; do compare "$OUT/$t.apr.json" "$OUT/$t.oracle.jsonl" 0 >"$OUT/$t.cmp.json"; done
# Planted: the oracle one step late. It must fail, or the comparator cannot.
compare "$OUT/p1.apr.json" "$OUT/p1.oracle.jsonl" 1 >"$OUT/p1.planted_shift.json"
# Planted: p1's record against p2's oracle.
compare "$OUT/p1.apr.json" "$OUT/p2.oracle.jsonl" 0 >"$OUT/p1_vs_p2.planted.json"
# Refusal controls: must be 400 and named.
code() { curl -s -o "$OUT/$1.body" -w '%{http_code}' "http://127.0.0.1:$PA/v1/chat/completions" \
    -H 'content-type: application/json' -d "$2"; }
r1=$(code r_k21 '{"model":"q","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":21}')
r2=$(code r_nolp '{"model":"q","messages":[{"role":"user","content":"hi"}],"top_logprobs":5}')
r3=$(code r_stream '{"model":"q","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":5,"stream":true}')
echo "p1 pass=$(jq .pass "$OUT/p1.cmp.json") p2 pass=$(jq .pass "$OUT/p2.cmp.json") planted_shift pass=$(jq .pass "$OUT/p1.planted_shift.json") planted_cross pass=$(jq .pass "$OUT/p1_vs_p2.planted.json") refusals=$r1/$r2/$r3"
grep -iE 'gpu|cuda|cpu' "$OUT/apr.log" | head -5 || true
