#!/usr/bin/env bash
# lib_story_run.sh - command capture for the qwen story, with the two output
# streams kept APART.
#
# Why this exists
# ---------------
# run_cmd used to capture `"$@" 2>&1` into a single RC_OUT, and nine call sites
# then read RC_OUT: some pipe it to `jq`, some grep it for a panic. Those two
# groups want opposite things, and merging served neither.
#
# On 2026-08-11 the nightly failed with
#
#     ✗ FAIL  B2 format_parity  -  no format_parity gate found in --json
#             output (got: '', apr qa exit=0)
#
# The gate had in fact run and PASSED. `apr qa --json` writes its JSON to
# stdout and its diagnostics to stderr (the CUDA path emits unconditional
# `[trueno#243] Manual graph construction: ...` lines via eprintln!). Merging
# put those lines in front of the JSON, `jq` could not parse the result, the
# extraction came back empty, and the story reported a defect that did not
# exist. A false FAIL is exactly as corrosive as a gate that cannot fail: main
# goes red for a non-reason and the andon stops meaning anything.
#
# Conversely a panic message arrives on STDERR, so the checks that grep for
# `thread ... panicked` genuinely need the merged view. Hence three variables
# rather than one, and each call site states which view it needs.
#
# OPTION NEUTRALITY (load-bearing)
# --------------------------------
# This file is SOURCED. It must not run `set`: doing so mutates the CALLER's
# shell. apr_bin.sh once opened with `set -euo pipefail`, qwen-story.sh sourced
# it, and the leaked errexit killed the nightly six lines in - the story's whole
# contract is to run every beat and tally failures, which errexit forbids.
# Sourceable libraries here fail by RETURN STATUS, never by shell option.
# Enforced by scripts/check_sourced_libs_option_neutral.sh.

# Run one command under a timeout, capturing its streams separately.
#
# Args: timeout_seconds command...
# Sets: RC_EC   exit status of the command (124 on timeout)
#       RC_OUT  STDOUT only  - parse this when you want JSON
#       RC_ERR  STDERR only  - diagnostics, warnings, progress
#       RC_ALL  both, stdout first - grep this for panics and banners
#       RC_TAIL last line of RC_ALL
#
# If the first argument is the literal `apr`, it is rewritten to "$APR" so every
# call site is pinned to the freshness-asserted binary without touching them all.
run_cmd() {
  local t="$1"; shift
  if [ "${1:-}" = "apr" ]; then shift; set -- "${APR:-apr}" "$@"; fi

  local errfile
  errfile=$(mktemp "${TMPDIR:-/tmp}/story-stderr.XXXXXX") || return 1

  # Only stderr is redirected, so RC_OUT is exactly what the command wrote to
  # stdout - byte for byte, with nothing interleaved.
  RC_OUT=$(timeout "$t" "$@" 2>"$errfile"); RC_EC=$?
  RC_ERR=$(cat "$errfile")
  rm -f "$errfile"

  # printf rather than a double-quoted string containing a literal newline:
  # bashrs flags the latter as SC1078 (an unclosed double-quoted string).
  if [ -n "$RC_ERR" ]; then
    RC_ALL=$(printf '%s\n%s' "$RC_OUT" "$RC_ERR")
  else
    RC_ALL="$RC_OUT"
  fi
  RC_TAIL=$(printf '%s\n' "$RC_ALL" | tail -1)
}
