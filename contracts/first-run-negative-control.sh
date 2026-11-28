#!/usr/bin/env bash
# Negative control for contracts/first-run-ux-v1.yaml (FR-NEG-002, F-FR-U00-014).
# Runs every first-run row's falsification test against apr 0.75.4 and requires
# each row to be RED (exit nonzero, not 75). Each row runs from nothing: the row
# command sets its own empty HOME and XDG_CACHE_HOME.
#   exit 0  every row RED on 0.75.4 (control holds)
#   exit 1  a row PASSED on 0.75.4 (the harness is broken)
#   exit 75 NOT RUN: apr is not 0.75.4, or a prerequisite is missing. Never a pass.
# The row commands below are copies of the test lines in first-run-ux-v1.yaml;
# if one changes there, change it here in the same commit.

want="0.75.4"
fail=0
notrun=0

for bin in apr jq curl cargo pv; do
  command -v "$bin" >/dev/null 2>&1 || { echo "NOT RUN: $bin not on PATH" >&2; exit 75; }
done
apr --version 2>/dev/null | grep -q "$want" || { echo "NOT RUN: apr is not $want" >&2; exit 75; }

row() {
  id=$1
  shift
  bash -c "$1" >/dev/null 2>&1
  rc=$?
  case $rc in
    0)  echo "$id PASS on $want: the harness does not catch this row" >&2; fail=1 ;;
    75) echo "$id NOT RUN (CANNOT MEASURE)" >&2; notrun=1 ;;
    *)  echo "$id RED on $want (exit $rc), as required" ;;
  esac
}

row F-FR-U00-002 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; out=$(apr run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct "What is 2+2?" --temperature 0 --json) || { echo "F-FR-U00-002 (U-01): apr run exited nonzero" >&2; exit 1; }; jq -e ".text | type == \"string\" and (test(\"Completed in\") | not) and test(\"(?<![0-9.])4(?![0-9]|[.][0-9]|[a-zA-Z])\")" <<<"$out" >/dev/null || { echo "F-FR-U00-002 (U-01): reply is not the answer 4, or it is a timing line" >&2; exit 1; }'
row F-FR-U00-003 'G=Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF; D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; apr run hf://$G "hi" >/dev/null 2>&1 || exit 1; L=$(apr list) || exit 1; grep -qF "$G" <<<"$L" || exit 1; find "$D" -type f \( -name "*.gguf" -o -name "*.safetensors" \) -size +1M | grep -q . || exit 1; apr rm $G >/dev/null 2>&1 || exit 1; L=$(apr list) || exit 1; ! grep -qF "$G" <<<"$L"'
row F-FR-U00-004 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; apr pull hf://HuggingFaceTB/SmolLM2-135M-Instruct >/dev/null 2>&1; apr list | grep -q "HuggingFaceTB/SmolLM2-135M-Instruct" && ! apr list | grep -q "(orphan)"'
row F-FR-U00-005 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; out=$(apr run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF "Explain quicksort in three sentences." --temperature 0 --json) || { echo "F-FR-U00-005 (U-04): apr run exited nonzero" >&2; exit 1; }; jq -e "(.completion_tokens > 32) and (.finish_reason == \"stop\") and ((.text | type) == \"string\") and (.text | test(\"[.!?][[:space:]]*$\"))" <<<"$out" >/dev/null || { echo "F-FR-U00-005 (U-04): reply cut off at the token cap, or .text does not end with terminal punctuation" >&2; exit 1; }'
row F-FR-U00-006 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; apr run hf://HuggingFaceTB/SmolLM2-135M-Instruct "hi" 2>&1 >/dev/null | grep -qiE "backend|cpu|cuda"'
row F-FR-U00-007 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; out=$(apr doctor --json 2>$D/err) || { echo "F-FR-U00-007 (U-06): apr doctor is not a top-level command at this head (only apr cgp doctor exists, and it has no backend key), or exited nonzero" >&2; exit 1; }; jq -e "(.backend | type == \"string\") and (.backend | length > 0) and (.backend | ascii_downcase != \"unknown\")" <<<"$out" >/dev/null || { echo "F-FR-U00-007 (U-06): apr doctor --json backend is missing or unknown" >&2; exit 1; }'
row F-FR-U00-008 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; apr run hf://HuggingFaceTB/SmolLM2-135M-Instruct "hi" >$D/out 2>$D/err || { echo "F-FR-U00-008 (U-07): apr run exited nonzero" >&2; exit 1; }; grep -q "[^[:space:]]" $D/out || { echo "F-FR-U00-008 (U-07): stdout is empty" >&2; exit 1; }; grep -qE "=== APR Run|^[[:space:]]*Source:|^[[:space:]]*Output:|Completed in" $D/out && { echo "F-FR-U00-008 (U-07): banner or log line on stdout" >&2; exit 1; }; grep -qE "GH-[0-9]+|PMAT-[0-9]+|CRUX-[A-Z]-[0-9]+" $D/err && { echo "F-FR-U00-008 (U-07): ticket tag on stderr" >&2; exit 1; }; exit 0'
row F-FR-U00-009 'out=$(apr --help) || exit 1; grep -q "^Usage:" <<<"$out" && ! grep -qE "GH-[0-9]+|PMAT-[0-9]+|CRUX-[A-Z]-[0-9]+|#[0-9]{3,}" <<<"$out"'
row F-FR-U00-010 'command -v cargo >/dev/null 2>&1 || { echo "CANNOT MEASURE: cargo not on PATH" >&2; exit 75; }; out=$(cargo test -p aprender-core --test readme_contract readme_library_usage_snippet_compiles 2>&1); rc=$?; if [ $rc -ne 0 ] && grep -q "^cargo-memcap: REFUSED" <<<"$out"; then echo "CANNOT MEASURE: cargo-memcap refused the build" >&2; exit 75; fi; exit $rc'
row F-FR-U00-011 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; M=Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF; apr pull hf://$M >/dev/null 2>&1 || { echo "F-FR-U00-011 (G-01): apr pull hf://$M failed" >&2; exit 1; }; P0=$((20000 + RANDOM % 20000)); apr serve run $M --port $P0 >/dev/null 2>&1 & P=$!; trap "kill $P 2>/dev/null" EXIT; T=0; until curl -fsS http://127.0.0.1:$P0/health >/dev/null 2>&1; do kill -0 $P 2>/dev/null || { echo "F-FR-U00-011 (G-01): apr serve exited before /health answered" >&2; exit 1; }; T=$((T+1)); [ $T -ge 60 ] && { echo "F-FR-U00-011 (G-01): /health did not answer within 60 s" >&2; exit 1; }; sleep 1; done; CODE=$(curl -s -o $D/body -w "%{http_code}" -X POST http://127.0.0.1:$P0/v1/messages -H "content-type: application/json" -d "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}]}"); [ "$CODE" = 200 ] || { echo "F-FR-U00-011 (G-01): POST /v1/messages returned $CODE, not 200" >&2; exit 1; }; jq -e "(.type == \"message\") and (.role == \"assistant\") and (.content[0].type == \"text\") and ((.content[0].text | type) == \"string\") and ((.content[0].text | length) > 0) and ((.content[0].text | ascii_downcase | startswith(\"error:\")) | not)" $D/body >/dev/null || { echo "F-FR-U00-011 (G-01): reply is empty, not a message, or an error string" >&2; exit 1; }'
row F-FR-U00-012 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; M=Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF; apr pull hf://$M >/dev/null 2>&1 || { echo "F-FR-U00-012 (G-02): apr pull hf://$M failed" >&2; exit 1; }; P0=$((20000 + RANDOM % 20000)); apr serve run $M --port $P0 >/dev/null 2>&1 & P=$!; trap "kill $P 2>/dev/null" EXIT; T=0; until curl -fsS http://127.0.0.1:$P0/health >/dev/null 2>&1; do kill -0 $P 2>/dev/null || { echo "F-FR-U00-012 (G-02): apr serve exited before /health answered" >&2; exit 1; }; T=$((T+1)); [ $T -ge 60 ] && { echo "F-FR-U00-012 (G-02): /health did not answer within 60 s" >&2; exit 1; }; sleep 1; done; CODE=$(curl -s -o $D/body -w "%{http_code}" -X POST http://127.0.0.1:$P0/v1/responses -H "content-type: application/json" -d "{\"model\":\"m\",\"input\":\"What is 2+2?\"}"); [ "$CODE" = 200 ] || { echo "F-FR-U00-012 (G-02): POST /v1/responses returned $CODE, not 200" >&2; exit 1; }; jq -e "(.object == \"response\") and (.status == \"completed\") and (.output[0].content[0].type == \"output_text\") and ((.output[0].content[0].text | type) == \"string\") and ((.output[0].content[0].text | length) > 0)" $D/body >/dev/null || { echo "F-FR-U00-012 (G-02): response is not completed output_text with non-empty text" >&2; exit 1; }'
row F-FR-U00-013 'D=$(mktemp -d); export HOME=$D XDG_CACHE_HOME=$D/.cache; M=Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF; apr pull hf://$M >/dev/null 2>&1 || { echo "F-FR-U00-013 (G-03): apr pull hf://$M failed" >&2; exit 1; }; P0=$((20000 + RANDOM % 20000)); apr serve run $M --port $P0 >/dev/null 2>&1 & P=$!; trap "kill $P 2>/dev/null" EXIT; T=0; until curl -fsS http://127.0.0.1:$P0/health >/dev/null 2>&1; do kill -0 $P 2>/dev/null || { echo "F-FR-U00-013 (G-03): apr serve exited before /health answered" >&2; exit 1; }; T=$((T+1)); [ $T -ge 60 ] && { echo "F-FR-U00-013 (G-03): /health did not answer within 60 s" >&2; exit 1; }; sleep 1; done; CODE=$(curl -s -o $D/body -w "%{http_code}" -X POST http://127.0.0.1:$P0/v1/chat/completions -H "content-type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"); case $CODE in 4??) ;; *) echo "F-FR-U00-013 (G-03): request without model returned $CODE, not 4xx" >&2; exit 1 ;; esac; jq -e "(.error.message | type == \"string\") and (.error.message | ascii_downcase | contains(\"model\"))" $D/body >/dev/null || { echo "F-FR-U00-013 (G-03): .error.message does not name model" >&2; exit 1; }'

if [ "$fail" -ne 0 ]; then
  echo "F-FR-U00-014: a row PASSED on apr $want; the negative control is broken" >&2
  exit 1
fi
if [ "$notrun" -ne 0 ]; then
  echo "F-FR-U00-014: NOT RUN (at least one row could not be measured)" >&2
  exit 75
fi
echo "F-FR-U00-014: every row RED on apr $want"
exit 0
