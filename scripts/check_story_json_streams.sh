#!/usr/bin/env bash
# check_story_json_streams.sh - prove run_cmd keeps stdout and stderr apart.
#
# The nightly story failed on 2026-08-11 with
#
#     ✗ FAIL  B2 format_parity  -  no format_parity gate found in --json
#             output (got: '', apr qa exit=0)
#
# while the gate had actually run and PASSED. run_cmd captured `cmd 2>&1`, so
# `apr qa --json`'s stderr diagnostics (the CUDA path emits unconditional
# `[trueno#243] ...` lines) were prepended to its JSON. `jq` could not parse
# that, the extraction came back empty, and the harness reported a defect that
# did not exist.
#
# These are BEHAVIOURAL assertions - they run the real run_cmd from the real
# library over commands with known stream behaviour. A grep-for-the-idiom guard
# would have passed the whole time the bug was live, because `2>&1` is correct
# in most of the places it appears.

set -uo pipefail

LIB="$(dirname "$0")/lib_story_run.sh"
[ -f "$LIB" ] || { echo "check_story_json_streams: missing $LIB"; exit 1; }
# shellcheck source=scripts/lib_story_run.sh
. "$LIB" || { echo "check_story_json_streams: could not source $LIB"; exit 1; }

fails=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want() { # name expected actual
  if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi
}

echo "check_story_json_streams: run_cmd stream separation"

# -- 1. The exact failing shape --------------------------------------------
# A command that writes JSON to stdout and diagnostics to stderr. Parsing
# RC_OUT must yield the field; this is what the format_parity check does.
run_cmd 10 bash -c 'echo "[trueno#243] Manual graph construction: pos=0" >&2; echo "{\"gates\":[{\"name\":\"format_parity\",\"skipped\":false,\"passed\":true}]}"'
got=$(printf '%s\n' "$RC_OUT" | jq -r '.gates[] | select(.name=="format_parity") | "\(.skipped) \(.passed)"' 2>/dev/null)
want "JSON on stdout survives diagnostics on stderr" "false true" "$got"

# -- 2. Each stream lands in its own variable -------------------------------
run_cmd 10 bash -c 'echo OUTLINE; echo ERRLINE >&2'
want "RC_OUT is stdout only" "OUTLINE" "$RC_OUT"
want "RC_ERR is stderr only" "ERRLINE" "$RC_ERR"

# -- 3. RC_ALL keeps the merged view the panic checks depend on -------------
# A Rust panic is written to STDERR. If RC_ALL dropped it, the story would stop
# detecting panics - a strictly worse failure than the one being fixed here.
run_cmd 10 bash -c "echo normal; echo \"thread 'main' panicked at src/x.rs:1:1\" >&2"
if printf '%s\n' "$RC_ALL" | grep -qE 'thread.*panicked'; then
  ok "RC_ALL still sees a panic written to stderr"
else
  bad "RC_ALL still sees a panic written to stderr" "match" "$RC_ALL"
fi

# -- 4. Exit status belongs to the COMMAND, not to the capture plumbing -----
run_cmd 10 bash -c 'echo x >&2; exit 7'
want "RC_EC is the command exit status" "7" "$RC_EC"
run_cmd 1 sleep 5
want "RC_EC is 124 on timeout" "124" "$RC_EC"

# -- 5. Empty stderr must not inject a blank line into RC_ALL ---------------
# RC_ALL is greped and tailed; a stray trailing newline changes RC_TAIL.
run_cmd 10 bash -c 'echo only-stdout'
want "RC_ALL has no phantom blank line when stderr is empty" "only-stdout" "$RC_ALL"
want "RC_TAIL is the last real line" "only-stdout" "$RC_TAIL"

# -- 6. No JSON-parsing call site may read the merged stream ----------------
# Behavioural checks above cover run_cmd itself; this covers the CALLERS, which
# is where the defect actually surfaced. Any `jq` fed from RC_ALL/RC_ERR is the
# same bug in a new place.
STORY="$(dirname "$0")/qwen-story.sh"
if [ -f "$STORY" ]; then
  offenders=$(grep -nE '\$RC_(ALL|ERR)"?[^|]*\|[[:space:]]*jq' "$STORY" || true)
  if [ -z "$offenders" ]; then
    ok "no jq call site parses the merged stream"
  else
    bad "no jq call site parses the merged stream" "none" "$offenders"
  fi
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "check_story_json_streams: OK - stdout and stderr stay separate"
  exit 0
fi
echo "check_story_json_streams: $fails assertion(s) FAILED"
exit 1
