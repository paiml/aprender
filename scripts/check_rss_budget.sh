#!/usr/bin/env bash
# check_rss_budget.sh - dogfood C1 (apr-dogfood Gate 11): peak RSS of one real
# `apr run` against the F-CHAOS-001 / AQC-BND-001 bound
#
#     peak_rss(apr run M) < 3 * size(M) + 512 MiB
#
# The multiple and the overhead are READ from scripts/perf-matrix.yaml arms.R
# (rss_model_multiple, rss_overhead_bytes), so the dogfood check and perf_gate's
# Arm R cannot drift apart (aprender#4522 R2).
#
# WHY THIS REPLACES THE SKILL SNIPPET. The C1 block in the apr-dogfood skill
# could not fail: over budget it printed "C1 WARN" and exited 0; it measured
# `apr inspect` (a header read) instead of `apr run`; it called a bare `apr`;
# with no /usr/bin/time it SKIPped. A memory check that cannot go red is not
# a check.
#
# Exit: 0 under budget; 1 over budget, or the run itself failed; 2 cannot
# measure (no model, no GNU time, no RSS line, unreadable matrix). 2 is a
# FAILURE for the caller, never a skip: "could not measure" is not "passed".
#
# Usage:
#   bash scripts/check_rss_budget.sh [--model PATH]   (default: first *.gguf < 1G under ~/models)
#   bash scripts/check_rss_budget.sh --self-test
#
# Env (self-test seams): CHECK_RSS_TIME (GNU time, default /usr/bin/time),
# CHECK_RSS_APR (skip apr_bin.sh and use this binary).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="${CHECK_RSS_MATRIX:-$ROOT/scripts/perf-matrix.yaml}"

budget_bytes() { # model-bytes -> ceiling bytes on stdout, from arms.R
  python3 - "$MATRIX" "$1" <<'PY'
import sys, yaml
a = ((yaml.safe_load(open(sys.argv[1])) or {}).get("arms") or {}).get("R") or {}
k, o = a.get("rss_model_multiple"), a.get("rss_overhead_bytes")
if not isinstance(k, int) or not isinstance(o, int):
    sys.exit("arms.R.rss_model_multiple / rss_overhead_bytes absent from the matrix")
print(k * int(sys.argv[2]) + o)
PY
}

measure() { # model -> 0/1/2, prints one C1 line
  local model="$1" tbin="${CHECK_RSS_TIME:-/usr/bin/time}" apr out rc=0 rss_kb mbytes budget
  if [ ! -f "$model" ]; then echo "C1 FAIL(2): model not found: $model"; return 2; fi
  if ! "$tbin" -v true >/dev/null 2>&1; then
    echo "C1 FAIL(2): GNU time not usable at $tbin; RSS cannot be measured"; return 2
  fi
  if [ -n "${CHECK_RSS_APR:-}" ]; then apr="$CHECK_RSS_APR"
  else
    # shellcheck source=/dev/null
    . "$ROOT/scripts/apr_bin.sh" || { echo "C1 FAIL(2): apr_bin.sh could not pin apr"; return 2; }
    apr="$APR"
  fi
  mbytes="$(stat -c %s "$model")"
  budget="$(budget_bytes "$mbytes")" || { echo "C1 FAIL(2): no budget in $MATRIX"; return 2; }
  out="$("$tbin" -v timeout 120 "$apr" run "$model" --prompt "Hi" --max-tokens 8 2>&1)" || rc=$?
  rss_kb="$(printf '%s\n' "$out" | awk -F: '/Maximum resident set size/ {gsub(/ /,"",$2); print $2}')"
  case "$rss_kb" in
    ''|*[!0-9]*) echo "C1 FAIL(2): no 'Maximum resident set size' line from $tbin"; return 2 ;;
  esac
  if [ "$rc" != 0 ]; then
    echo "C1 FAIL: apr run exited $rc (rss ${rss_kb} KiB); a run that fails is not under budget"; return 1
  fi
  if [ $((rss_kb * 1024)) -ge "$budget" ]; then
    echo "C1 FAIL: peak RSS $((rss_kb * 1024)) B >= budget $budget B (3 x model $mbytes B + 512 MiB, F-CHAOS-001)"
    return 1
  fi
  echo "C1 PASS: peak RSS $((rss_kb * 1024)) B < budget $budget B (model $mbytes B)"
  return 0
}

self_test() {
  local tmp pass=0 fail=0
  tmp="$(mktemp -d)"
  case "$tmp" in /tmp/*) : ;; *) echo "mktemp gave ${tmp:-<empty>}" >&2; return 2 ;; esac
  # A 1 MiB model: budget = 3 MiB + 512 MiB = 540,016,640 B = 527,360 KiB exactly.
  head -c 1048576 /dev/zero > "$tmp/m.gguf"
  printf '#!/bin/sh\nexit "${FAKE_APR_RC:-0}"\n' > "$tmp/apr"
  # Fake GNU time: runs the command, then reports FAKE_RSS_KB the way `time -v` does.
  printf '#!/bin/sh\nshift\n"$@"; rc=$?\n[ -n "${FAKE_RSS_KB:-}" ] && echo "\tMaximum resident set size (kbytes): $FAKE_RSS_KB" >&2\nexit $rc\n' > "$tmp/time"
  chmod +x "$tmp/apr" "$tmp/time"
  _case() { # name expect-rc needle env...
    local name="$1" want="$2" needle="$3" out got=0; shift 3
    out="$(env CHECK_RSS_TIME="$tmp/time" CHECK_RSS_APR="$tmp/apr" "$@" bash "$0" --model "$tmp/m.gguf" 2>&1)" || got=$?
    if [ "$got" = "$want" ] && case "$out" in *"$needle"*) true ;; *) false ;; esac; then
      printf '  ok    %-36s rc=%s\n' "$name" "$got"; pass=$((pass + 1))
    else
      printf '  BROKE %-36s want rc=%s ~%s, got rc=%s: %s\n' "$name" "$want" "$needle" "$got" "$out"; fail=$((fail + 1))
    fi
  }
  _case within_budget_passes          0 "C1 PASS"  FAKE_RSS_KB=200000
  # THE PLANTED OVER-BUDGET RUN (#4522 R2 done-when): 600 MB resident for a 1 MiB model.
  _case planted_over_budget_is_red     1 "C1 FAIL: peak RSS" FAKE_RSS_KB=600000
  _case at_budget_is_red_strict_lt     1 "C1 FAIL: peak RSS" FAKE_RSS_KB=527360
  _case one_kib_under_passes           0 "C1 PASS"  FAKE_RSS_KB=527359
  _case crashed_run_is_red             1 "apr run exited 3" FAKE_RSS_KB=200000 FAKE_APR_RC=3
  _case no_rss_line_cannot_measure     2 "no 'Maximum resident"
  _case no_gnu_time_cannot_measure     2 "GNU time not usable" CHECK_RSS_TIME=/nonexistent/time FAKE_RSS_KB=1
  local out got=0
  out="$(env CHECK_RSS_TIME="$tmp/time" CHECK_RSS_APR="$tmp/apr" bash "$0" --model "$tmp/absent.gguf" 2>&1)" || got=$?
  if [ "$got" = 2 ]; then printf '  ok    %-36s rc=2\n' missing_model_cannot_measure; pass=$((pass + 1))
  else printf '  BROKE %-36s got rc=%s: %s\n' missing_model_cannot_measure "$got" "$out"; fail=$((fail + 1)); fi
  rm -rf "${tmp:?}"
  echo "$pass passed, $fail broken"
  [ "$fail" = 0 ]
}

main() {
  local model=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --self-test) self_test; return $? ;;
      --model) model="${2:-}"; shift 2 ;;
      -h|--help) sed -n '2,27p' "$0"; return 0 ;;
      *) echo "unknown arg: $1" >&2; return 2 ;;
    esac
  done
  if [ -z "$model" ]; then
    # `-size -1G` (the old snippet) matches EMPTY files only: find rounds the size UP
    # to whole GiB first. And ~/models is a symlink on the dev box, which find does
    # not descend without the trailing slash. Both made the old C1 SKIP forever.
    model="$(find "$HOME/models/" -maxdepth 2 -name '*.gguf' -type f -size -1024M 2>/dev/null | sort | head -1)"
    [ -n "$model" ] || { echo "C1 FAIL(2): no *.gguf under 1G in ~/models; pass --model"; return 2; }
  fi
  measure "$model"
}

main "$@"
