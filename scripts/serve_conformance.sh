#!/usr/bin/env bash
# serve_conformance.sh — the `apr serve` agent conformance suite runner (#5349, EPIC 0.73 row R-6).
#
# Sends a FIXED set of client-observable cases (OpenAI + Anthropic shapes) to a RUNNING `apr serve`
# and writes one receipt cell per case x model x backend. Every cell is one of
#     pass           the response met every assertion of the case
#     fail           an assertion failed; `detail` names the first one (a 404 is a fail, never a skip)
#     not_measured   the server could not be asked (connection refused / timeout); `detail` says why
# A registered case with NO cell is a missing measurement and the run exits non-zero.
#
# `backend` is the label the CALLER declares (--backend); nothing here measures which device ran, so the
# cell records it as `backend_declared`. Do not read it as proof of CUDA/wgpu (verification discipline 2).
#
# Usage:
#   bash scripts/serve_conformance.sh --base http://127.0.0.1:8080 --model NAME --backend cpu \
#        [--shape openai|anthropic|all] [--out receipt.json] [--timeout SECS]
#   bash scripts/serve_conformance.sh --self-test     # fixtures + planted mutants, no server needed
#   bash scripts/serve_conformance.sh --list          # the registered case ids
# Exit: 0 every cell pass · 1 a cell failed · 2 usage/ENV · 3 a cell not_measured or missing.
#       --self-test: 0 when every fixture row lands as expected AND every planted mutant turns RED.
#
# LAB: no PR or release step runs this (C324). Nightly: scripts/serve_conformance_nightly.sh.
# Cases: OpenAI ones come from contracts/apr-serve-openai-compat-v1.yaml; Anthropic ones are
# contracts/apr-serve-anthropic-messages-v1.yaml. Rows R-1..R-5 of the EPIC each have >= 1 case per shape.
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ANTHROPIC_CONTRACT="$ROOT/contracts/apr-serve-anthropic-messages-v1.yaml"

# id | shape | EPIC row | what it asks
CASES=(
  "oa-models|openai|R-5|GET /v1/models lists the served model"
  "oa-chat-nonstream|openai|R-5|chat completion has content, finish_reason and usage"
  "oa-chat-finish-length|openai|R-5|max_tokens 1 ends with finish_reason length"
  "oa-chat-finish-stop|openai|R-5|a stop sequence ends with finish_reason stop"
  "oa-chat-sse|openai|R-5|SSE chat: JSON chunks, a finish_reason chunk, then [DONE]"
  "oa-completions|openai|R-5|/v1/completions has text and finish_reason"
  "oa-usage-exact|openai|R-4|usage sums, and completion_tokens equals max_tokens 1"
  "oa-seed-repro|openai|R-3|same seed, same request, same text"
  "oa-tool-call|openai|R-2|a forced tool call returns tool_calls with JSON arguments"
  "an-messages-nonstream|anthropic|R-1|POST /v1/messages returns a message with text and stop_reason"
  "an-stop-reason-max|anthropic|R-5|max_tokens 1 ends with stop_reason max_tokens"
  "an-messages-sse|anthropic|R-5|SSE message: start, block events, message_delta, message_stop"
  "an-usage-exact|anthropic|R-4|usage.output_tokens equals max_tokens 1; input_tokens positive"
  "an-greedy-repro|anthropic|R-3|temperature 0 twice gives the same text"
  "an-tool-use|anthropic|R-2|a forced tool returns a tool_use block and stop_reason tool_use"
  "an-tool-result|anthropic|R-1|a tool_result turn is accepted and answered with text"
)

usage() { sed -n '/^# Usage:/,/^# Exit:/p' "$SELF" | sed 's/^# \{0,1\}//'; }

BASE="" MODEL="" BACKEND="" SHAPE=all OUT="" TMO=120
SELFTEST=0 LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=${2:?--base wants a URL}; shift 2 ;;
    --model) MODEL=${2:?--model wants a name}; shift 2 ;;
    --backend) BACKEND=${2:?--backend wants a label}; shift 2 ;;
    --shape) SHAPE=${2:?--shape wants openai|anthropic|all}; shift 2 ;;
    --out) OUT=${2:?--out wants a path}; shift 2 ;;
    --timeout) TMO=${2:?--timeout wants seconds}; shift 2 ;;
    --self-test) SELFTEST=1; shift ;;
    --list) LIST=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) printf 'serve_conformance: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$LIST" = 1 ]; then
  for c in "${CASES[@]}"; do IFS='|' read -r id shape row what <<< "$c"; printf '%s\t%s\t%s\t%s\n' "$id" "$shape" "$row" "$what"; done
  exit 0
fi

for t in jq curl; do command -v "$t" > /dev/null 2>&1 || { printf 'serve_conformance: ENV - %s missing\n' "$t" >&2; exit 2; }; done
case "$SHAPE" in openai | anthropic | all) ;; *) printf 'serve_conformance: --shape wants openai|anthropic|all\n' >&2; exit 2 ;; esac

T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT

# ---- transport: curl, or canned fixtures when SC_FIXTURE_DIR is set (the self-test) -------------------
R_CODE="" R_BODY="" R_ERR=""
# req <tag> <METHOD> <path> [json body] [accept]
req() {
  local tag="$1" method="$2" path="$3" body="${4:-}" accept="${5:-application/json}"
  R_ERR=""
  if [ -n "${SC_FIXTURE_DIR:-}" ]; then
    if [ -r "$SC_FIXTURE_DIR/$tag.code" ]; then
      R_CODE=$(< "$SC_FIXTURE_DIR/$tag.code")
      R_BODY=$(< "$SC_FIXTURE_DIR/$tag.body")
    else
      R_CODE=000 R_BODY="" R_ERR="no fixture for $tag"
    fi
    return 0
  fi
  local out rc=0
  if [ -n "$body" ]; then
    out=$(curl -sS --max-time "$TMO" -X "$method" -H 'Content-Type: application/json' -H "Accept: $accept" \
      -H 'anthropic-version: 2023-06-01' -H 'x-api-key: conformance' --data "$body" \
      -w $'\n%{http_code}' "$BASE$path" 2> "$T/curl.err" < /dev/null) || rc=$?
  else
    out=$(curl -sS --max-time "$TMO" -X "$method" -H "Accept: $accept" \
      -w $'\n%{http_code}' "$BASE$path" 2> "$T/curl.err" < /dev/null) || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    R_CODE=000 R_BODY="" R_ERR="curl rc=$rc: $(head -c 200 "$T/curl.err" | tr '\n' ' ')"
    return 0
  fi
  R_CODE=${out##*$'\n'}
  R_BODY=${out%$'\n'*}
}

# ---- assertions: the FIRST failure is the cell's detail ----------------------------------------------
CUR="" FAILMSG="" SKIPMSG=""
begin() { CUR="$1"; FAILMSG=""; SKIPMSG=""; }
no_server() { # a refused connection is not a verdict about the server's shape
  if [ "$R_CODE" = 000 ]; then SKIPMSG="${SKIPMSG:-${R_ERR:-no response}}"; return 1; fi
  return 0
}
code_is() { [ -z "$FAILMSG$SKIPMSG" ] || return 0; no_server || return 0; [ "$R_CODE" = "$1" ] || FAILMSG="HTTP $R_CODE, wanted $1"; }
need() { # <jq filter over R_BODY> <message>
  [ -z "$FAILMSG$SKIPMSG" ] || return 0
  jq -e "$1" <<< "$R_BODY" > /dev/null 2>&1 || FAILMSG="$2"
}
need_in() { # <json value> <jq filter over it> <message>
  [ -z "$FAILMSG$SKIPMSG" ] || return 0
  jq -e "$2" <<< "$1" > /dev/null 2>&1 || FAILMSG="$3"
}
fail_with() { [ -z "$FAILMSG$SKIPMSG" ] || return 0; FAILMSG="$1"; }
# sse_json: the JSON payloads of an SSE body (without [DONE]) as one array; empty string when a payload is not JSON
sse_json() {
  local lines
  lines=$(sed -n 's/^data: \{0,1\}//p' <<< "$R_BODY" | sed '/^\[DONE\]$/d' | sed '/^$/d' || true)
  [ -n "$lines" ] || { echo '[]'; return 0; }
  jq -s . <<< "$lines" 2> /dev/null || echo 'NOTJSON'
}
emit() { # write the cell for $CUR
  local id shape row what status detail
  for c in "${CASES[@]}"; do IFS='|' read -r id shape row what <<< "$c"; [ "$id" = "$CUR" ] && break; done
  if [ -n "$SKIPMSG" ]; then status=not_measured detail=$SKIPMSG
  elif [ -n "$FAILMSG" ]; then status=fail detail=$FAILMSG
  else status=pass detail=""; fi
  jq -cn --arg case "$CUR" --arg shape "$shape" --arg row "$row" --arg model "$MODEL" --arg backend "$BACKEND" \
    --arg status "$status" --arg detail "$detail" \
    '{case:$case,shape:$shape,row:$row,model:$model,backend_declared:$backend,status:$status,detail:$detail}' >> "$T/cells.jsonl"
}

chat() { # <extra json merged into the request>
  jq -cn --arg m "$MODEL" --argjson x "${1:-null}" \
    '{model:$m,messages:[{role:"user",content:"Reply with the single word OK."}],temperature:0} + ($x // {})'
}
msg() { # <extra json merged into the request>
  jq -cn --arg m "$MODEL" --argjson x "${1:-null}" \
    '{model:$m,max_tokens:32,messages:[{role:"user",content:"Reply with the single word OK."}],temperature:0} + ($x // {})'
}
OA_TOOL='{"type":"function","function":{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}'
AN_TOOL='{"name":"get_weather","description":"Weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}'

# ---- OpenAI cases ------------------------------------------------------------------------------------
case_oa_models() {
  begin oa-models; req oa-models GET /v1/models
  code_is 200
  need '.data | type == "array" and length >= 1' "data is not a non-empty array"
  need 'all(.data[]; (.id | type == "string") and (.id | length > 0))' "a model entry has no string id"
  emit
}
case_oa_chat_nonstream() {
  begin oa-chat-nonstream; req oa-chat-nonstream POST /v1/chat/completions "$(chat '{"max_tokens":16}')"
  code_is 200
  need '.choices[0].message.content | type == "string"' "choices[0].message.content is not a string"
  need '.choices[0].finish_reason | IN("stop","length")' "finish_reason missing or not stop|length"
  need '.usage | (.prompt_tokens|type=="number") and (.completion_tokens|type=="number") and (.total_tokens|type=="number")' "usage lacks numeric prompt/completion/total tokens"
  emit
}
case_oa_chat_finish_length() {
  begin oa-chat-finish-length; req oa-chat-finish-length POST /v1/chat/completions "$(chat '{"max_tokens":1}')"
  code_is 200
  need '.choices[0].finish_reason == "length"' "finish_reason is not length for max_tokens 1"
  emit
}
case_oa_chat_finish_stop() {
  begin oa-chat-finish-stop
  req oa-chat-finish-stop POST /v1/chat/completions \
    "$(jq -cn --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Count from 1 to 9, separated by spaces."}],temperature:0,max_tokens:64,stop:["3"]}')"
  code_is 200
  need '.choices[0].finish_reason == "stop"' "finish_reason is not stop after a stop sequence"
  need '(.choices[0].message.content // "") | contains("3") | not' "the stop sequence appears in the content"
  emit
}
case_oa_chat_sse() {
  begin oa-chat-sse; req oa-chat-sse POST /v1/chat/completions "$(chat '{"max_tokens":16,"stream":true}')" text/event-stream
  code_is 200
  local j; j=$(sse_json)
  [ "$j" != NOTJSON ] || fail_with "an SSE data line is not JSON"
  need_in "$j" 'length >= 1' "no data chunks"
  need_in "$j" 'any(.[]; .choices[0].finish_reason != null)' "no chunk carries a non-null finish_reason"
  if [ -z "$FAILMSG$SKIPMSG" ]; then
    [ "$(sed -n 's/^data: \{0,1\}//p' <<< "$R_BODY" | sed '/^$/d' | tail -n 1)" = "[DONE]" ] || FAILMSG="stream does not end with data: [DONE]"
  fi
  emit
}
case_oa_completions() {
  begin oa-completions
  req oa-completions POST /v1/completions "$(jq -cn --arg m "$MODEL" '{model:$m,prompt:"The capital of France is",temperature:0,max_tokens:8}')"
  code_is 200
  need '.choices[0].text | type == "string"' "choices[0].text is not a string"
  need '.choices[0].finish_reason | IN("stop","length")' "finish_reason missing or not stop|length"
  emit
}
case_oa_usage_exact() {
  begin oa-usage-exact; req oa-usage-exact POST /v1/chat/completions "$(chat '{"max_tokens":1}')"
  code_is 200
  need '.usage.completion_tokens == 1' "usage.completion_tokens is not 1 for max_tokens 1"
  need '.usage.prompt_tokens > 0' "usage.prompt_tokens is not positive"
  need '.usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)' "usage.total_tokens is not prompt + completion"
  emit
}
case_oa_seed_repro() {
  begin oa-seed-repro
  local b; b=$(chat '{"max_tokens":24,"temperature":0.8,"seed":1234}')
  req oa-seed-repro-1 POST /v1/chat/completions "$b"
  code_is 200
  need '.choices[0].message.content | type == "string"' "first reply has no content"
  local a; a=$(jq -r '.choices[0].message.content // empty' <<< "$R_BODY" 2> /dev/null || true)
  req oa-seed-repro-2 POST /v1/chat/completions "$b"
  code_is 200
  [ -n "$FAILMSG$SKIPMSG" ] || [ "$(jq -r '.choices[0].message.content // empty' <<< "$R_BODY" 2> /dev/null || true)" = "$a" ] || FAILMSG="same seed gave two different texts"
  emit
}
case_oa_tool_call() {
  begin oa-tool-call
  req oa-tool-call POST /v1/chat/completions "$(jq -cn --arg m "$MODEL" --argjson t "$OA_TOOL" \
    '{model:$m,temperature:0,max_tokens:64,messages:[{role:"user",content:"What is the weather in Paris?"}],tools:[$t],tool_choice:{type:"function",function:{name:"get_weather"}}}')"
  code_is 200
  need '.choices[0].message.tool_calls | type == "array" and length >= 1' "no tool_calls in the message"
  need '.choices[0].message.tool_calls[0].function.name == "get_weather"' "tool call does not name get_weather"
  need '.choices[0].message.tool_calls[0].function.arguments | type == "string" and (try fromjson | type == "object")' "tool call arguments are not a JSON object string"
  need '.choices[0].finish_reason == "tool_calls"' "finish_reason is not tool_calls"
  emit
}

# ---- Anthropic cases ---------------------------------------------------------------------------------
case_an_messages_nonstream() {
  begin an-messages-nonstream; req an-messages-nonstream POST /v1/messages "$(msg '{}')"
  code_is 200
  need '.type == "message" and .role == "assistant"' "not a message with role assistant"
  need '.content | type == "array" and any(.[]; .type == "text" and (.text | type == "string"))' "content has no text block"
  need '.stop_reason | IN("end_turn","max_tokens","stop_sequence","tool_use")' "stop_reason missing or unknown"
  emit
}
case_an_stop_reason_max() {
  begin an-stop-reason-max; req an-stop-reason-max POST /v1/messages "$(msg '{"max_tokens":1}')"
  code_is 200
  need '.stop_reason == "max_tokens"' "stop_reason is not max_tokens for max_tokens 1"
  emit
}
case_an_messages_sse() {
  begin an-messages-sse; req an-messages-sse POST /v1/messages "$(msg '{"stream":true}')" text/event-stream
  code_is 200
  local j; j=$(sse_json)
  [ "$j" != NOTJSON ] || fail_with "an SSE data line is not JSON"
  need_in "$j" 'length >= 1 and .[0].type == "message_start"' "first event is not message_start"
  need_in "$j" 'any(.[]; .type == "content_block_start") and any(.[]; .type == "content_block_delta") and any(.[]; .type == "content_block_stop")' "a content_block_* event is missing"
  need_in "$j" 'any(.[]; .type == "message_delta" and (.delta.stop_reason | type == "string"))' "no message_delta with a stop_reason"
  need_in "$j" '.[-1].type == "message_stop"' "last event is not message_stop"
  emit
}
case_an_usage_exact() {
  begin an-usage-exact; req an-usage-exact POST /v1/messages "$(msg '{"max_tokens":1}')"
  code_is 200
  need '.usage.output_tokens == 1' "usage.output_tokens is not 1 for max_tokens 1"
  need '.usage.input_tokens > 0' "usage.input_tokens is not positive"
  emit
}
case_an_greedy_repro() {
  begin an-greedy-repro
  local b; b=$(msg '{"max_tokens":24}')
  req an-greedy-repro-1 POST /v1/messages "$b"
  code_is 200
  need '.content[0].text | type == "string"' "first reply has no text"
  local a; a=$(jq -r '.content[0].text // empty' <<< "$R_BODY" 2> /dev/null || true)
  req an-greedy-repro-2 POST /v1/messages "$b"
  code_is 200
  [ -n "$FAILMSG$SKIPMSG" ] || [ "$(jq -r '.content[0].text // empty' <<< "$R_BODY" 2> /dev/null || true)" = "$a" ] || FAILMSG="temperature 0 gave two different texts"
  emit
}
case_an_tool_use() {
  begin an-tool-use
  req an-tool-use POST /v1/messages "$(jq -cn --arg m "$MODEL" --argjson t "$AN_TOOL" \
    '{model:$m,max_tokens:64,temperature:0,messages:[{role:"user",content:"What is the weather in Paris?"}],tools:[$t],tool_choice:{type:"tool",name:"get_weather"}}')"
  code_is 200
  need '.content | any(.[]; .type == "tool_use")' "no tool_use block"
  need '[.content[] | select(.type == "tool_use")][0] | (.id | type == "string") and .name == "get_weather" and (.input | type == "object")' "tool_use block lacks id, name get_weather or an object input"
  need '.stop_reason == "tool_use"' "stop_reason is not tool_use"
  emit
}
case_an_tool_result() {
  begin an-tool-result
  req an-tool-result POST /v1/messages "$(jq -cn --arg m "$MODEL" --argjson t "$AN_TOOL" \
    '{model:$m,max_tokens:32,temperature:0,tools:[$t],messages:[
       {role:"user",content:"What is the weather in Paris?"},
       {role:"assistant",content:[{type:"tool_use",id:"toolu_conf1",name:"get_weather",input:{city:"Paris"}}]},
       {role:"user",content:[{type:"tool_result",tool_use_id:"toolu_conf1",content:"18C and sunny"}]}]}')"
  code_is 200
  need '.content | any(.[]; .type == "text" and (.text | type == "string"))' "no text block after a tool_result turn"
  emit
}

run_suite() {
  : > "$T/cells.jsonl"
  local c id shape row what fn
  for c in "${CASES[@]}"; do
    IFS='|' read -r id shape row what <<< "$c"
    case "$SHAPE" in all) ;; "$shape") ;; *) continue ;; esac
    fn="case_${id//-/_}"
    "$fn"
  done
}

# summarize <receipt path>: write the receipt, print one line per cell, return the exit code
summarize() {
  local out="$1" want got id shape row what
  want=0 got=0
  for c in "${CASES[@]}"; do
    IFS='|' read -r id shape row what <<< "$c"
    case "$SHAPE" in all) ;; "$shape") ;; *) continue ;; esac
    want=$((want + 1))
    if [ "$(jq -sr --arg i "$id" '[.[] | select(.case == $i)] | length' "$T/cells.jsonl")" -ge 1 ]; then got=$((got + 1))
    else printf '  MISSING %s (no cell written)\n' "$id"
    fi
  done
  jq -s --arg base "$BASE" --arg model "$MODEL" --arg backend "$BACKEND" --arg shape "$SHAPE" \
    --argjson want "$want" --argjson got "$got" \
    '{schema:"apr-serve-conformance-receipt-v1",base:$base,model:$model,backend_declared:$backend,shape:$shape,
      registered:$want,cells_written:$got,cells:.,
      pass:([.[]|select(.status=="pass")]|length),fail:([.[]|select(.status=="fail")]|length),
      not_measured:([.[]|select(.status=="not_measured")]|length)}' "$T/cells.jsonl" > "$out"
  jq -r '.cells[] | "  \(.status|ascii_upcase)\t\(.row)\t\(.case)\(if .detail != "" then " - " + .detail else "" end)"' "$out"
  local nf nn
  nf=$(jq '.fail' "$out"); nn=$(jq '.not_measured' "$out")
  printf 'serve_conformance: %s pass · %s fail · %s not_measured · %s/%s cells\n' "$(jq .pass "$out")" "$nf" "$nn" "$got" "$want"
  [ "$got" -eq "$want" ] || return 3
  [ "$nf" -eq 0 ] || return 1
  [ "$nn" -eq 0 ] || return 3
  return 0
}

main_run() {
  [ -n "$BASE" ] && [ -n "$MODEL" ] && [ -n "$BACKEND" ] || { printf 'serve_conformance: --base, --model and --backend are required\n' >&2; usage >&2; exit 2; }
  local out="${OUT:-$T/receipt.json}" rc=0
  run_suite
  summarize "$out" || rc=$?
  [ -z "$OUT" ] || printf 'serve_conformance: receipt %s\n' "$OUT"
  exit "$rc"
}

# ---- self-test: canned responses + planted mutants ---------------------------------------------------
fx() { printf '%s' "$3" > "$FX/$1.body"; printf '%s' "$2" > "$FX/$1.code"; } # <tag> <code> <body>
build_fixtures() {
  FX=$1; mkdir -p "$FX"
  local oa='{"id":"c1","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}'
  fx oa-models 200 '{"object":"list","data":[{"id":"m","object":"model"}]}'
  fx oa-chat-nonstream 200 "$oa"
  fx oa-chat-finish-length 200 '{"choices":[{"message":{"content":"O"},"finish_reason":"length"}],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}'
  fx oa-chat-finish-stop 200 '{"choices":[{"message":{"content":"1 2 "},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}'
  fx oa-chat-sse 200 $'data: {"choices":[{"delta":{"content":"O"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
  fx oa-completions 200 '{"choices":[{"text":" Paris","finish_reason":"length"}]}'
  fx oa-usage-exact 200 '{"choices":[{"message":{"content":"O"},"finish_reason":"length"}],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}'
  fx oa-seed-repro-1 200 "$oa"; fx oa-seed-repro-2 200 "$oa"
  fx oa-tool-call 200 '{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}]}'
  local an='{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":2}}'
  fx an-messages-nonstream 200 "$an"
  fx an-stop-reason-max 200 '{"type":"message","role":"assistant","content":[{"type":"text","text":"O"}],"stop_reason":"max_tokens","usage":{"input_tokens":9,"output_tokens":1}}'
  fx an-messages-sse 200 $'event: message_start\ndata: {"type":"message_start","message":{"id":"m"}}\n\nevent: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"O"}}\n\nevent: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
  fx an-usage-exact 200 '{"type":"message","role":"assistant","content":[{"type":"text","text":"O"}],"stop_reason":"max_tokens","usage":{"input_tokens":9,"output_tokens":1}}'
  fx an-greedy-repro-1 200 "$an"; fx an-greedy-repro-2 200 "$an"
  fx an-tool-use 200 '{"type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}],"stop_reason":"tool_use","usage":{"input_tokens":9,"output_tokens":8}}'
  fx an-tool-result 200 "$an"
}
SELF_FAIL=0
st_ok() { printf '  ok    %s\n' "$1"; }
st_bad() { printf '  BROKE %s\n' "$1"; SELF_FAIL=$((SELF_FAIL + 1)); }
# one_case <label> <fixture dir> <case id> <wanted status>
one_case() {
  local label="$1" dir="$2" id="$3" want="$4" got
  SC_FIXTURE_DIR="$dir"; : > "$T/cells.jsonl"; "case_${id//-/_}"
  got=$(jq -r '.status' <<< "$(tail -n 1 "$T/cells.jsonl")")
  if [ "$got" = "$want" ]; then st_ok "$label ($id -> $got)"; else st_bad "$label: $id -> $got, wanted $want ($(tail -n 1 "$T/cells.jsonl" | jq -r .detail))"; fi
}
# mutate <tag> <jq filter over the fixture body>: a copy of the fixtures with one body changed
mutate() {
  local tag="$1" filt="$2" dir="$T/mut-$1"
  rm -rf -- "${dir:?}"; cp -r "$FXD" "$dir"
  jq -c "$filt" "$FXD/$tag.body" > "$dir/$tag.body"
  if cmp -s "$FXD/$tag.body" "$dir/$tag.body"; then printf 'MUTANT-VACUOUS\n'; else printf '%s\n' "$dir"; fi
}
mut_case() { # <label> <tag> <jq filter> <case id>: the mutant must turn the case to fail
  local d; d=$(mutate "$2" "$3")
  if [ "$d" = MUTANT-VACUOUS ]; then st_bad "$1: mutant changed nothing (vacuous)"; return; fi
  one_case "$1" "$d" "$4" fail
}
self_test() {
  MODEL=fixture BACKEND=cpu SHAPE=all
  FXD="$T/fx"; build_fixtures "$FXD"
  local c id shape row what
  # 1. clean fixtures: every case passes
  for c in "${CASES[@]}"; do IFS='|' read -r id shape row what <<< "$c"; one_case "clean" "$FXD" "$id" pass; done
  # 2. the planted mutant of the issue: finish_reason dropped -> RED
  mut_case "mutant: finish_reason dropped" oa-chat-nonstream 'del(.choices[0].finish_reason)' oa-chat-nonstream
  mut_case "mutant: finish_reason dropped (length)" oa-chat-finish-length 'del(.choices[0].finish_reason)' oa-chat-finish-length
  mut_case "mutant: stop_reason dropped" an-messages-nonstream 'del(.stop_reason)' an-messages-nonstream
  mut_case "mutant: usage dropped" oa-usage-exact 'del(.usage)' oa-usage-exact
  mut_case "mutant: usage sum wrong" oa-usage-exact '.usage.total_tokens = 99' oa-usage-exact
  mut_case "mutant: output_tokens wrong" an-usage-exact '.usage.output_tokens = 5' an-usage-exact
  mut_case "mutant: tool arguments not JSON" oa-tool-call '.choices[0].message.tool_calls[0].function.arguments = "not json"' oa-tool-call
  mut_case "mutant: tool_use block gone" an-tool-use '.content = [{"type":"text","text":"no"}]' an-tool-use
  mut_case "mutant: second seeded reply differs" oa-seed-repro-2 '.choices[0].message.content = "DIFFERENT"' oa-seed-repro
  mut_case "mutant: stop text leaked" oa-chat-finish-stop '.choices[0].message.content = "1 2 3"' oa-chat-finish-stop
  # 3. SSE mutants (text bodies): [DONE] gone, message_stop gone, non-JSON data line
  local d
  d="$T/mut-nodone"; rm -rf -- "${d:?}"; cp -r "$FXD" "$d"; sed '/\[DONE\]/d' "$FXD/oa-chat-sse.body" > "$d/oa-chat-sse.body"
  one_case "mutant: [DONE] missing" "$d" oa-chat-sse fail
  d="$T/mut-nostop"; rm -rf -- "${d:?}"; cp -r "$FXD" "$d"; grep -v 'message_stop' "$FXD/an-messages-sse.body" > "$d/an-messages-sse.body" || true
  one_case "mutant: message_stop missing" "$d" an-messages-sse fail
  d="$T/mut-badjson"; rm -rf -- "${d:?}"; cp -r "$FXD" "$d"; printf 'data: {truncated\n\n' >> "$d/oa-chat-sse.body"
  one_case "mutant: SSE data not JSON" "$d" oa-chat-sse fail
  # 4. a 404 (R-1 not built yet) is a fail, never a skip; a refused connection is not_measured
  d="$T/mut-404"; rm -rf -- "${d:?}"; cp -r "$FXD" "$d"; printf '404' > "$d/an-messages-nonstream.code"; printf '{"error":"not found"}' > "$d/an-messages-nonstream.body"
  one_case "404 on /v1/messages" "$d" an-messages-nonstream fail
  d="$T/mut-down"; rm -rf -- "${d:?}"; cp -r "$FXD" "$d"; rm -f "$d/oa-models.code"
  one_case "no response" "$d" oa-models not_measured
  # 5. a missing cell exits non-zero (3): run the summary over a cells file missing one case
  SC_FIXTURE_DIR="$FXD"; run_suite; grep -v '"case":"oa-models"' "$T/cells.jsonl" > "$T/cells.cut" || true; mv "$T/cells.cut" "$T/cells.jsonl"
  local rc=0; summarize "$T/r.json" > "$T/sum.txt" 2>&1 || rc=$?
  if [ "$rc" = 3 ] && grep -q 'MISSING oa-models' "$T/sum.txt"; then st_ok "missing cell -> exit 3"; else st_bad "missing cell: rc=$rc"; fi
  # 6. whole-run exit codes on clean fixtures (0) and on the 404 fixtures (1)
  rc=0; ( SC_FIXTURE_DIR="$FXD"; BASE=x; OUT="$T/clean.json"; main_run ) > /dev/null 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then st_ok "clean run exits 0"; else st_bad "clean run exited $rc"; fi
  rc=0; ( SC_FIXTURE_DIR="$T/mut-404"; BASE=x; OUT="$T/r404.json"; main_run ) > /dev/null 2>&1 || rc=$?
  if [ "$rc" = 1 ]; then st_ok "failing run exits 1"; else st_bad "failing run exited $rc"; fi
  # 7. the registry and the Anthropic contract agree on the Anthropic case ids
  if [ -r "$ANTHROPIC_CONTRACT" ]; then
    for c in "${CASES[@]}"; do
      IFS='|' read -r id shape row what <<< "$c"
      [ "$shape" = anthropic ] || continue
      if grep -Eq "name: ASAM-CASE-$id\$" "$ANTHROPIC_CONTRACT"; then st_ok "contract names $id"; else st_bad "contract lacks 'ASAM-CASE-$id'"; fi
    done
  else st_bad "contract $ANTHROPIC_CONTRACT unreadable"; fi
  # 8. every EPIC row R-1..R-5 has a case per shape
  local r s
  for s in openai anthropic; do
    for r in R-1 R-2 R-3 R-4 R-5; do
      [ "$s$r" != openaiR-1 ] || continue   # R-1 is the Anthropic adapter itself
      local n=0
      for c in "${CASES[@]}"; do IFS='|' read -r id shape row what <<< "$c"; [ "$shape" = "$s" ] && [ "$row" = "$r" ] && n=$((n + 1)); done
      if [ "$n" -ge 1 ]; then st_ok "row $r has $n $s case(s)"; else st_bad "row $r has no $s case"; fi
    done
  done
  printf 'serve_conformance --self-test: %d broke\n' "$SELF_FAIL"
  [ "$SELF_FAIL" -eq 0 ]
}

if [ "$SELFTEST" = 1 ]; then self_test; exit $?; fi
main_run
