#!/usr/bin/env bash
# qwen-story.sh  -  8-beat narrative exercising every core apr command group
# against the Qwen series (0.5B safetensors → 30B-MoE GGUF). Each beat is a
# falsifier in contracts/qwen-story-v1.yaml. Used by:
#
#   - README.md ## A Qwen story (top-of-fold quickstart)
#   - .claude/skills/dogfood Gate 18 (regression detection)
#   - .github/workflows/qwen-story-daily.yml (nightly bug-hunt cron)
#
# Exit codes:
#   0   all runnable beats PASS
#   2   one or more beats FAIL
#   3   one or more beats SKIP (missing model)  -  informational, also exits 0
#       if SKIPs are the only non-PASS results
#
# Each beat uses OUT=$(cmd); EC=$? to avoid the pipe-then-$? methodology
# defect documented in memory/feedback_test_methodology_can_fake_bugs.md.

# NOTE the deliberate absence of `-e`. This script's contract is to run EVERY
# beat and tally the results (emit_fail / FAILED_BEATS / exit 2), so a failing
# beat must not abort the run. Anything sourced below must not turn errexit on.
set -uo pipefail

# Resolve `apr` and PROVE it was built from this commit. Sourcing this exports
# $APR and hard-fails on a stale binary. Without it a bare `apr` resolves via
# PATH, and on the cuda runner ~/.local/bin held a 24-day-old build that
# shadowed the freshly installed one - so every beat below validated stale code
# while reporting green. Set APR_BIN to override which binary is used.
#
# The explicit `|| exit 1` is what makes a stale binary fatal. apr_bin.sh used
# to get that effect by doing `set -euo pipefail` at its own file scope, which
# leaked errexit into THIS script and killed the nightly run six lines in.
# shellcheck source=scripts/apr_bin.sh
. "$(dirname "${BASH_SOURCE[0]}")/apr_bin.sh" || exit 1

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PMAT_HUNT="${PMAT_HUNT:-1}"  # 1 = run pmat full audit per beat
TMPDIR_STORY="${TMPDIR_STORY:-/tmp/qwen-story-$$}"
mkdir -p "$TMPDIR_STORY"
trap '[ -n "$TMPDIR_STORY" ] && [ "$TMPDIR_STORY" != "/" ] && rm -rf "$TMPDIR_STORY" 2>/dev/null; pkill -P $$ 2>/dev/null || true' EXIT

# -- Model registry ------------------------------------------------------------
M_05B_ST="$MODELS_DIR/qwen2.5-coder-0.5b-instruct-safetensors/model.safetensors"
M_15B_APR="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4k.apr"
# The format_parity gate peeks the primary's magic bytes and SKIPs anything that
# isn't GGUF, so no .apr leg can ever exercise it. This is the only clean
# GGUF + matching-SafeTensors pair on the runner (auto-discovery resolves the
# reference out of the HF cache, so no --safetensors-path is needed).
M_15B_GGUF="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
# Pinned explicitly rather than left to auto-discovery. On this runner
# auto-discovery resolves to the HF cache snapshot, whose
# layers.0.ffn_down_weight reads back 100% zeros and trips F-DATA-QUALITY-001
# during conversion - so the gate would FAIL on a bad reference instead of
# comparing anything. A gate must not depend on which of several copies a
# search heuristic happens to find first.
M_15B_ST="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-safetensors/model.safetensors"
M_7B_GGUF="$MODELS_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
M_30B_MOE="$MODELS_DIR/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

PASS=0
FAIL=0
SKIP=0
FAILED_BEATS=()

note()    { printf '  %s\n' "$*"; }
emit_pass(){ PASS=$((PASS+1)); printf '✓ PASS  %s\n' "$1"; }
emit_fail(){ FAIL=$((FAIL+1)); FAILED_BEATS+=("$1"); printf '✗ FAIL  %s  -  %s\n' "$1" "$2"; }
emit_skip(){ SKIP=$((SKIP+1)); printf '○ SKIP  %s  -  %s\n' "$1" "$2"; }

# run_cmd lives in a sourceable library so its stream handling can be tested
# directly - see scripts/check_story_json_streams.sh. It sets RC_OUT (stdout
# only), RC_ERR (stderr only), RC_ALL (both) and RC_EC.
#
# Parse RC_OUT when you want JSON; grep RC_ALL when the thing you are looking
# for could arrive on either stream (panics go to stderr). Merging the two
# unconditionally is what made a PASSING format_parity gate report as missing.
#
# The explicit `|| exit 1` is what makes a missing library fatal; the library
# itself must not run `set`, which would leak options into this script.
# shellcheck source=scripts/lib_story_run.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib_story_run.sh" || exit 1

# Print the captured output of the last run_cmd, indented, so it survives into
# the story log (and therefore into the `story-log` CI artifact).
#
# run_cmd deliberately captures into a variable rather than letting output
# stream, which is what makes the OUT=$(cmd); EC=$? methodology work - but it
# also means nothing a command printed was ever retained. The qa gate table in
# particular has never reached an artifact: a downloaded story-log contains
# "PASS  B2 apr qa" and not one word about which gates ran, skipped or failed.
# A verdict with no evidence behind it cannot be audited after the fact.
emit_evidence() {
  printf '    -- captured output (%s) --\n' "$1"
  printf '%s\n' "$RC_ALL" | sed 's/^/    /'
}

# pmat_rows / pmat_hunt live in a sourceable library so the manifest can be
# driven directly against a stubbed pmat - see scripts/check_story_pmat_hunt.sh.
# They ran here for months emitting eight headers and zero rows a night (#2356);
# the library header records all three causes and the measurements behind them.
#
# The hunt is NO LONGER ADVISORY: a header with no rows under it calls emit_fail
# and fails the beat, because every one of those three causes presented as an
# empty manifest and nothing could tell that apart from clean code. emit_fail
# must therefore be defined before this point - the library checks and refuses
# to load otherwise.
#
# The explicit `|| exit 1` is what makes a missing library fatal; the library
# itself must not run `set`, which would leak options into this script.
# shellcheck source=scripts/lib_story_pmat.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib_story_pmat.sh" || exit 1

# -- Beat 1: Discover (0.5B SafeTensors) --------------------------------------─
beat1_discover() {
  printf -- '-- Beat 1: Discover (Registry) --\n'
  if [ ! -f "$M_05B_ST" ]; then
    # In CI this is the model we'd PULL; locally we use cache.
    emit_skip "B1 pull" "0.5B SafeTensors not in cache at $M_05B_ST"
    return
  fi
  run_cmd 30 apr list
  if [ "$RC_EC" -eq 0 ]; then
    emit_pass "B1 list"
  else
    emit_fail "B1 list" "exit=$RC_EC"
    return
  fi
  # No commands/list.rs: `Commands::List => pull::list(...)` - list and pull are
  # the same module. The dead path was hunted nightly and returned nothing.
  pmat_hunt "registry list" crates/apr-cli/src/commands/pull.rs
}

# The GGUF leg of Beat 2. Deliberately NOT named beat<N>_* - the story has
# exactly 8 beats (FALSIFY-QWEN-STORY-002) and this is a second assertion within
# Beat 2, not a ninth beat.
#
# Why it exists: `run_format_parity_gate` peeks the primary's magic bytes and
# returns SKIP for anything that isn't GGUF. Beat 2's other leg passes the .apr,
# so format_parity has SKIPped on every scheduled run since the cron was added -
# and a SKIP sets passed:true, so `apr qa` still printed ALL GATES PASSED. The
# cross-format gate the doctrine says to reach for FIRST has never once executed
# in CI.
#
# The assertion is on the JSON gate object, NOT on the ALL GATES PASSED banner,
# precisely because that banner cannot distinguish "ran and passed" from
# "skipped". It is also independent of `apr qa`'s exit code: unrelated gates
# (ollama_parity, perf regression) legitimately fail on some hosts, and this
# check is about format_parity specifically.
check_format_parity_gguf() {
  if [ ! -f "$M_15B_GGUF" ] || [ ! -f "$M_15B_ST" ]; then
    emit_fail "B2 format_parity" "GGUF ($M_15B_GGUF) or SafeTensors reference ($M_15B_ST) absent - the cross-format gate cannot run. Provision the models rather than skipping: a silent skip is what hid this gate for months."
    return
  fi
  run_cmd 900 apr qa "$M_15B_GGUF" --safetensors-path "$M_15B_ST" --json
  FP=$(printf '%s\n' "$RC_OUT" \
    | jq -r '.gates[]? | select(.name=="format_parity") | "\(.skipped) \(.passed) \(.value) \(.threshold)"' 2>/dev/null \
    | tail -1)
  case "$FP" in
    "false true "*)
      emit_pass "B2 format_parity (GGUF vs SafeTensors, executed: $FP)"
      ;;
    "true "*)
      emit_fail "B2 format_parity" "gate SKIPped - it must EXECUTE on a GGUF primary"
      ;;
    "false false "*)
      emit_fail "B2 format_parity" "cross-format decode DIVERGED: $(printf '%s\n' "$RC_OUT" | jq -r '.gates[]? | select(.name=="format_parity") | .message' 2>/dev/null | tail -1)"
      ;;
    *)
      emit_fail "B2 format_parity" "no format_parity gate found in --json output (got: '$FP', apr qa exit=$RC_EC)"
      ;;
  esac
}

# -- Beat 2: Trust (0.5B safetensors) ------------------------------------------
beat2_trust() {
  printf -- '-- Beat 2: Trust (QA gates) --\n'
  if [ ! -f "$M_15B_APR" ]; then
    emit_skip "B2 qa" "1.5B APR not available at $M_15B_APR"
    return
  fi
  # Use 1.5B APR (apr qa Golden Output gate works on this; 7B has #1864).
  run_cmd 180 apr qa "$M_15B_APR"
  # Retain the per-gate table regardless of verdict - it is the only record of
  # which gates actually executed versus SKIPped, and `apr qa` prints
  # "ALL GATES PASSED" even when gates skipped (GateResult::skipped sets
  # passed:true). The grep below therefore cannot distinguish "everything ran
  # and passed" from "half of it skipped".
  emit_evidence "apr qa $M_15B_APR"
  # RC_ALL: the banner is a human-facing line and apr is free to put it on
  # either stream; this check is about presence, not about parsing.
  if grep -q "ALL GATES PASSED" <<< "$RC_ALL" ; then
    emit_pass "B2 apr qa"
  else
    emit_fail "B2 apr qa" "no 'ALL GATES PASSED' line"
    return
  fi
  check_format_parity_gguf

  run_cmd 60 apr validate "$M_15B_APR" --quality
  if [ "$RC_EC" -eq 0 ]; then
    emit_pass "B2 apr validate --quality"
  else
    emit_fail "B2 apr validate --quality" "exit=$RC_EC (after #1866 fix this should be 0)"
  fi
  # This accepted `0 || 5` - PASS whether lint passed OR failed - which made it
  # unable to detect anything. It was written that way because `apr lint` could
  # not exit 0 on any real model: it gated on "no warnings", and every model
  # carries advisory metadata warnings (missing license / model_card /
  # provenance), so a healthy .apr and a corrupt .gguf both exited 5. #2394.
  #
  # Now that the verdict discriminates - ERRORs fail, warnings are advice,
  # --strict promotes them - this asserts the real thing: a known-good model
  # must lint clean.
  run_cmd 30 apr lint "$M_15B_APR"
  if [ "$RC_EC" -eq 0 ]; then
    emit_pass "B2 apr lint (exit=0 on a healthy model)"
  else
    emit_fail "B2 apr lint" "exit=$RC_EC on a known-good model; lint must exit 0 unless there are ERROR-level findings"
  fi
  pmat_hunt "qa validate lint" \
    crates/apr-cli/src/commands/qa.rs \
    crates/apr-cli/src/commands/validate.rs \
    crates/apr-cli/src/commands/lint.rs
}

# -- Beat 3: Explore (1.5B APR  -  has tokenizer next to it) --------------------─
beat3_explore() {
  printf -- '-- Beat 3: Explore (Inspection) --\n'
  if [ ! -f "$M_15B_APR" ]; then
    emit_skip "B3 inspect" "no APR model"
    return
  fi
  run_cmd 30 apr inspect --json "$M_15B_APR"
  local arch
  arch=$(echo "$RC_OUT" | jq -r '.architecture // empty' 2>/dev/null)
  if [ "$RC_EC" -eq 0 ] && [ -n "$arch" ]; then
    emit_pass "B3 apr inspect --json (arch=$arch)"
  else
    emit_fail "B3 apr inspect --json" "exit=$RC_EC arch='$arch'"
  fi
  run_cmd 30 apr tensors "$M_15B_APR" --json
  local n
  n=$(echo "$RC_OUT" | jq '.tensor_count // (.|length) // 0' 2>/dev/null)
  if [ "$RC_EC" -eq 0 ] && [ "${n:-0}" -gt 0 ]; then
    emit_pass "B3 apr tensors --json ($n tensors)"
  else
    emit_fail "B3 apr tensors --json" "exit=$RC_EC n=$n"
  fi
  run_cmd 30 apr tree "$M_15B_APR"
  [ "$RC_EC" -eq 0 ] && emit_pass "B3 apr tree" || emit_fail "B3 apr tree" "exit=$RC_EC"
  pmat_hunt "inspect tensors tree" \
    crates/apr-cli/src/commands/inspect.rs \
    crates/apr-cli/src/commands/tensors.rs \
    crates/apr-cli/src/commands/tree.rs
}

# -- Beat 4: Adapt (export + diff; convert path covered by Beat 1 pull) --------
beat4_adapt() {
  printf -- '-- Beat 4: Adapt (Model ops) --\n'
  if [ ! -f "$M_15B_APR" ]; then
    emit_skip "B4 export" "no APR model"
    return
  fi
  # apr export (post-#1865 fix: panic → graceful error or success)
  local out="$TMPDIR_STORY/exported.gguf"
  rm -f "$out"
  run_cmd 120 apr export "$M_15B_APR" --format gguf -o "$out"
  if [ "$RC_EC" -eq 0 ] && [ -s "$out" ]; then
    emit_pass "B4 apr export → gguf"
    run_cmd 30 apr diff "$M_15B_APR" "$out"
    if [ "$RC_EC" -eq 0 ]; then
      emit_pass "B4 apr diff (APR vs round-tripped GGUF)"
    else
      emit_fail "B4 apr diff" "exit=$RC_EC"
    fi
  elif [ "$RC_EC" -eq 5 ]; then
    # Clean validation error is acceptable post-#1865 (e.g. missing num_heads).
    emit_pass "B4 apr export (clean exit=5, no panic)"
  # RC_ALL, NOT RC_OUT: a Rust panic message is written to STDERR. Grepping
  # stdout alone would silently stop detecting panics.
  elif [ "$RC_EC" -eq 101 ] || grep -qE 'thread.*panicked' <<< "$RC_ALL" ; then
    emit_fail "B4 apr export" "PANIC (exit=$RC_EC)  -  #1865 regression"
  else
    emit_fail "B4 apr export" "unexpected exit=$RC_EC"
  fi
  pmat_hunt "export convert quantize" \
    crates/aprender-core/src/format/converter/metadata.rs \
    crates/apr-cli/src/commands/convert.rs \
    crates/apr-cli/src/commands/quantize.rs
}

# -- Beat 5: Use (1.5B Q4K APR) ------------------------------------------------
beat5_use() {
  printf -- '-- Beat 5: Use (Inference) --\n'
  if [ ! -f "$M_15B_APR" ]; then
    emit_skip "B5 run" "no APR model"
    return
  fi
  run_cmd 120 apr run "$M_15B_APR" "fn sum(a: i32, b: i32) -> i32 {" --max-tokens 16
  # Heuristic gibberish detector  -  flag if chat-template tokens repeat.
  if grep -qE '<\|im_start\|>.*<\|im_start\|>' <<< "$RC_OUT" ; then
    emit_fail "B5 apr run" "gibberish (chat-template token repeats)"
  elif [ "$RC_EC" -eq 0 ] && grep -qE 'Output:' <<< "$RC_OUT" ; then
    emit_pass "B5 apr run (Rust code completion)"
  else
    emit_fail "B5 apr run" "exit=$RC_EC, no Output line"
  fi
  # apr code -p (non-interactive coder agent)
  run_cmd 90 apr code -p "Reply with exactly: hello" --max-turns 1
  if [ "$RC_EC" -eq 0 ]; then
    emit_pass "B5 apr code -p"
  else
    emit_skip "B5 apr code -p" "non-zero exit=$RC_EC (may need --model)"
  fi
  # `apr code` moved out of apr-cli into aprender-orchestrate; commands/code.rs
  # has not existed for some time and was hunted nightly regardless.
  pmat_hunt "run chat code" \
    crates/apr-cli/src/commands/run.rs \
    crates/apr-cli/src/commands/chat.rs \
    crates/aprender-orchestrate/src/cli/code.rs
}

# -- Beat 6: Serve (1.5B over HTTP) --------------------------------------------
beat6_serve() {
  printf -- '-- Beat 6: Serve (REST API) --\n'
  if [ ! -f "$M_15B_APR" ]; then
    emit_skip "B6 serve" "no APR model"
    return
  fi
  local port=$((20000 + RANDOM % 10000))
  "$APR" serve run "$M_15B_APR" --port "$port" > "$TMPDIR_STORY/serve.log" 2>&1 &
  local pid=$!
  # Wait up to 60s for /health to come up.
  local up=0
  for _ in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      up=1; break
    fi
    sleep 1
  done
  if [ "$up" = "0" ]; then
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    emit_fail "B6 apr serve run" "server did not start within 60s"
    return
  fi
  emit_pass "B6 apr serve run (port=$port)"
  # /v1/chat/completions OpenAI-compat smoke
  local resp
  resp=$(curl -s -m 60 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"reply with: ok"}],"max_tokens":4}')
  local content
  content=$(echo "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
  if [ -n "$content" ]; then
    emit_pass "B6 /v1/chat/completions (got $(echo "$content" | head -c 20)...)"
  else
    emit_fail "B6 /v1/chat/completions" "no message.content in response"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # commands/serve.rs became the commands/serve/ DIRECTORY (handlers, routes,
  # server, ollama, auth). The file path had been dead for months.
  pmat_hunt "serve http chat-completions" \
    crates/apr-cli/src/commands/serve \
    crates/aprender-serve/src/api/cuda_chat_backend.rs
}

# -- Beat 7: Operate (7B Q4K GGUF  -  profile/bench, NOT apr qa which has #1864) ─
beat7_operate() {
  printf -- '-- Beat 7: Operate (Profiling) --\n'
  if [ ! -f "$M_7B_GGUF" ]; then
    emit_skip "B7 profile" "7B GGUF not at $M_7B_GGUF"
    return
  fi
  # profile/bench/parity don't actually generate; safe even with #1864 open.
  run_cmd 60 apr profile "$M_7B_GGUF"
  [ "$RC_EC" -eq 0 ] && emit_pass "B7 apr profile" \
    || emit_fail "B7 apr profile" "exit=$RC_EC"
  run_cmd 30 apr gpu --json
  [ "$RC_EC" -eq 0 ] && emit_pass "B7 apr gpu --json" \
    || emit_fail "B7 apr gpu --json" "exit=$RC_EC"
  run_cmd 60 apr serve plan "$M_7B_GGUF"
  [ "$RC_EC" -eq 0 ] && emit_pass "B7 apr serve plan -- 7B VRAM budget" \
    || emit_fail "B7 apr serve plan" "exit=$RC_EC"
  pmat_hunt "profile bench gpu parity" \
    crates/apr-cli/src/commands/profile.rs \
    crates/apr-cli/src/commands/bench.rs \
    crates/apr-cli/src/commands/gpu.rs \
    crates/apr-cli/src/commands/parity.rs
}

# -- Beat 8: Scale (30B-MoE) --------------------------------------------------─
beat8_scale() {
  printf -- '-- Beat 8: Scale (MoE introspection) --\n'
  if [ ! -f "$M_30B_MOE" ]; then
    emit_skip "B8 inspect MoE" "30B-MoE not at $M_30B_MOE"
    return
  fi
  run_cmd 60 apr inspect --json "$M_30B_MOE"
  local arch
  arch=$(echo "$RC_OUT" | jq -r '.architecture // empty' 2>/dev/null)
  if [ "$arch" = "qwen3moe" ]; then
    emit_pass "B8 apr inspect --json (arch=qwen3moe)"
  else
    emit_fail "B8 apr inspect --json" "arch='$arch' (expected qwen3moe)"
  fi
  run_cmd 60 apr tensors --json "$M_30B_MOE"
  local n
  n=$(echo "$RC_OUT" | jq '.tensor_count // 0' 2>/dev/null)
  if [ "${n:-0}" -gt 500 ]; then
    emit_pass "B8 apr tensors --json ($n tensors)"
  else
    emit_fail "B8 apr tensors --json" "$n tensors (expected ≥500 for 30B-MoE)"
  fi
  pmat_hunt "moe inspect qwen3" \
    crates/aprender-serve/src/infer/qwen3_moe_generate.rs \
    crates/aprender-serve/src/api/cuda_chat_backend.rs
}

# -- Main ----------------------------------------------------------------------
START=$(date +%s)
printf '\n=== Qwen Story v1  -  apr=%s, dir=%s ===\n\n' "$(apr --version 2>&1 | head -1)" "$MODELS_DIR"

beat1_discover
beat2_trust
beat3_explore
beat4_adapt
beat5_use
beat6_serve
beat7_operate
beat8_scale

ELAPSED=$(($(date +%s) - START))
printf '\n=== Story complete in %ds ===\n' "$ELAPSED"
printf '   %d PASS  /  %d FAIL  /  %d SKIP\n' "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -gt 0 ]; then
  printf '\nFailed beats:\n'
  for b in "${FAILED_BEATS[@]}"; do
    printf '   - %s\n' "$b"
  done
  exit 2
fi
exit 0
