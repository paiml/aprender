#!/usr/bin/env bash
# PERF-009 / H4 — one benchmarking entrypoint, not N.
#
# A "competing harness" is a script that STARTS A SERVER and COMPUTES A
# THROUGHPUT NUMBER on its own. Every such script is a second definition of how
# this project measures itself, free to drift from the gate, and drift is how a
# 2.93x that no harness produced reached the book.
#
# The canonical entrypoint is `apr test llm bench` driven by
# scripts/perf_gate.sh. Anything else that measures is either allowlisted with a
# stated reason or counted against a shrink-only baseline.
#
#   bash scripts/check_no_competing_harnesses.sh            # gate
#   bash scripts/check_no_competing_harnesses.sh --selftest # case table
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Shrink-only. Lower it when you delete a harness; NEVER raise it.
#
# RE-DERIVED TO 0. It was 1, measured with the old predicate that detected none
# of the four harnesses it exists to keep deleted. A stale baseline is not a
# harmless off-by-one here: with BASELINE=1, restoring any single harness gives
# count=1, and `n > BASELINE` is false, so the guard reports OK. Fixing the
# predicate alone left all three mutations GREEN — the baseline had to move too.
#
# 0 is the honest count on a clean tree after PERF-009: every competing harness
# is deleted, and the three files that trip the predicate are allowlisted above
# with their reasons. Any harness reappearing now gives count=1 > 0 and reds.
BASELINE=0

# Allowlisted, each with the reason it is not a competing definition.
is_allowed() {
  case "$1" in
    */perf_gate.sh)                 return 0 ;;  # THE entrypoint
    */ship-discharges/ship-002-discharge.sh) return 0 ;;  # a SHIP-002 DISCHARGE
                                         # RECEIPT, not a harness: it runs apr
                                         # once and records duration_sec as
                                         # provenance of that one run. It states
                                         # no tok/s and no comparison; the
                                         # number is never quoted as a result.
                                         # (PMAT-949: invisible to the old
                                         # regex because it wrote `date -u +%s`.)
    */ship-discharges/ship-008-discharge.sh) return 0 ;;  # likewise: SHIP-008
                                         # discharge receipt, one run, its
                                         # duration recorded as provenance only.
    */check_no_competing_harnesses.sh) return 0 ;;  # this detector: its own
                                                     # selftest fixtures contain the
                                                     # trigger strings, so without
                                                     # this it matches itself and
                                                     # inflates the baseline by one,
                                                     # hiding a real harness
    */check_no_fabricated_baselines.sh) return 0 ;;  # a GUARD, not a harness: its
                                                     # must-match fixtures contain
                                                     # `command -v llama-bench` and
                                                     # OLLAMA_TIMEOUT_SECONDS on
                                                     # purpose — that is what a
                                                     # must-match fixture IS. Same
                                                     # reason this detector exempts
                                                     # itself, one line above.
    */qwen-story.sh) return 0 ;;  # RETAINED BY SPEC, not overlooked.
                                  # APR-PERF-GATE-001 v2.2 §9: "qwen-story —
                                  # resolved. Two subjects, not two
                                  # methodologies. Retained, runs at merge
                                  # phase, shares the receipt schema and the
                                  # comparator-required rule." It is a
                                  # correctness/determinism story that happens
                                  # to time itself; it derives no comparator
                                  # ratio. If it ever computes one, delete this
                                  # line rather than widening the predicate.
    */check_comparator_flags.sh) return 0 ;;  # a GUARD, not a harness — the
                                              # PP-15/PP-20 comparator-flag
                                              # checker. Its must-match table
                                              # contains `apr serve run m.gguf
                                              # --gpu` and its header quotes
                                              # "15.7 tok/s" because that is
                                              # what a must-match fixture IS.
                                              # Same reason as
                                              # check_no_fabricated_baselines.sh
                                              # below.
    */lib/bench_receipt.py) return 0 ;;   # a RECEIPT VALIDATOR, not a harness:
                                         # stdlib json/statistics only, no HTTP,
                                         # no subprocess. It trips the predicate
                                         # on PROSE (`compare-llama-bench.py`,
                                         # `15.7 tok/s`, `7.5 SECONDS`) in the
                                         # comments that record what it was
                                         # written to catch — the same reason
                                         # check_no_fabricated_baselines.sh is
                                         # exempt two entries above.
    */lib/perf_receipt.py) return 0 ;;    # likewise: reads receipts, prints an
                                         # `agg_tok_s` COLUMN, and its selftest
                                         # argv fixture names `/opt/llama-server`.
                                         # It measures nothing; it validates what
                                         # scripts/perf_gate.sh measured.
    */perf041_client.py) return 0 ;;  # PERF-041 serialisation client; retired
                                      # when PP-LLAMA-001 §12 row 7 lands.
                                      # It is a MEASUREMENT client, not a second
                                      # definition of the gate: it starts no
                                      # server, derives no comparator ratio and
                                      # reports the token-normalised index beside
                                      # the wall-clock one precisely so a length
                                      # divergence cannot be read as
                                      # "serialization". The canonical client is
                                      # `apr test llm bench`; when row 7 gives it
                                      # the token-normalised index, delete this
                                      # file and this line together.
    *) return 1 ;;
  esac
}

# A fixed, world-writable path is both a symlink-attack surface and a collision
# between two concurrent runs on the same box.
COUNT_FILE="$(mktemp)"
trap 'rm -f "$COUNT_FILE"' EXIT

# A script qualifies only if it does BOTH.
#
# THIS PREDICATE IS DERIVED FROM THE FOUR DELETED BLOBS, NOT FROM INTUITION, and
# the first version was derived from intuition and therefore detected NONE of
# them. Restoring scripts/gpu_2x_benchmark.sh verbatim
# (`git show 64cb68177^:scripts/gpu_2x_benchmark.sh`, 171 lines) left this guard
# at rc=0, `count=1 baseline=1 OK` — a guard whose whole purpose is keeping four
# specific files deleted, blind to all four.
#
# What they actually invoke, read out of the blobs:
#
#   gpu_2x_benchmark.sh     ollama run                       + tok/s
#   benchmark-2x-ollama.sh  apr run, curl POST /v1/          + tok/s, bc -l
#   benchmark-matrix.sh     curl POST /v1/                   + tok/s, date +%s, bc -l
#
# The old pattern asked for `ollama serve`. Every one of them uses `ollama run`
# or drives an already-running server over HTTP — which is the whole point of a
# benchmark harness and the obvious thing to look for once you look at the
# files instead of guessing. `llama-cli` and `llama-bench` are included because
# §4.4.8 forbids driving the comparator with llama-bench at all, so a script
# reaching for it is a competing harness by definition.
#
# PYTHON. The universe below was `scripts/*.sh` only, and scripts/perf041_client.py
# — 216 lines that drive a server over HTTP and print `agg_tok_s` — sat outside
# it, so this guard reported `count=0 baseline=0 OK` over a live Python harness.
# A harness is a harness in any language; the two predicates therefore also
# recognise the shapes a Python one takes: `urllib.request.urlopen` /
# `requests.post` for the driving half, and `agg_tok_s` / `tok_s` for the rate.
starts_server() {
  grep -qE "serve run|llama-server|llama-cli|llama-bench|ollama (serve|run)|vllm serve|apr[[:space:]]+run|curl[^|]*(/v1/|/api/generate)|urllib\.request\.urlopen|requests\.(post|get)\(" "$1" 2>/dev/null
}
# `date -u +%s` is the same clock as `date +%s`; the first regex missed the -u
# form, so two discharge scripts measured an apr run on main for months while
# this guard reported count=0 (PMAT-949). Pinned by the "-u" selftest row.
computes_rate() { grep -qE "date( -u)? \+%s|tok/s|tokens?_per_sec|agg_tok_s|tok_s|SECONDS" "$1" 2>/dev/null; }

# UNIVERSE: tracked UNION working tree. A `git ls-files`-only universe gives an
# untracked harness a free pass, which is how three earlier guards were blind.
universe() {
  { git -C "$ROOT" ls-files 'scripts/*.sh' 'scripts/**/*.sh' \
        'scripts/*.py' 'scripts/**/*.py' 2>/dev/null || true
    find "$ROOT/scripts" \( -name '*.sh' -o -name '*.py' \) -type f 2>/dev/null \
      | sed "s|^$ROOT/||" || true
  } | sort -u
}

scan() {
  local base="${1:-$ROOT}" found=0
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    local f="$base/$rel"
    [ -f "$f" ] || continue
    is_allowed "$f" && continue
    if starts_server "$f" && computes_rate "$f"; then
      echo "  COMPETING  $rel"
      found=$((found + 1))
    fi
  done < <(universe)
  echo "$found" > "$COUNT_FILE"
}

gate() {
  echo "=== competing benchmark harnesses (PERF-009 / H4) ==="
  scan "$ROOT"
  local n; n=$(cat "$COUNT_FILE")
  echo "  count=$n baseline=$BASELINE"
  if [ "$n" -gt "$BASELINE" ]; then
    echo "FAIL: a new harness appeared. Measure through scripts/perf_gate.sh, or"
    echo "      allowlist it here with the reason it is not a second definition."
    return 1
  fi
  if [ "$n" -lt "$BASELINE" ]; then
    echo "NOTE: count fell below the baseline — lower BASELINE to $n in this file."
  fi
  echo "OK"
}

# A guard is admissible only if a mutation it should catch turns it RED.
selftest() {
  local tmp pass=0 fail=0
  tmp="$(mktemp -d)"
  case "$tmp" in /tmp/*|/var/folders/*) : ;; *) echo "refusing $tmp"; exit 2 ;; esac
  mkdir -p "$tmp/scripts"
  cp "$ROOT/scripts/perf_gate.sh" "$tmp/scripts/" 2>/dev/null || true

  _row() { # name, expect(detect|ignore), file-body
    printf '%s' "$3" > "$tmp/scripts/probe.sh"
    local got=ignore
    if starts_server "$tmp/scripts/probe.sh" && computes_rate "$tmp/scripts/probe.sh"; then got=detect; fi
    if [ "$got" = "$2" ]; then
      printf '  ok    %-40s expect=%s\n' "$1" "$2"; pass=$((pass + 1))
    else
      printf '  BROKE %-40s expected %s got %s\n' "$1" "$2" "$got"; fail=$((fail + 1))
    fi
  }
  _row "a real harness is detected"        detect 'apr serve run m.gguf & \n t=$(date +%s)'
  _row "server without timing is ignored"  ignore 'apr serve run m.gguf --port 8080'
  _row "timing without a server is ignored" ignore 't=$(date +%s); echo done'
  _row "date -u +%s counts as timing"       detect 'apr run m.gguf --prompt hi \n t=$(date -u +%s)'
  _row "a pinned calendar stamp is ignored"  ignore 'apr run m.gguf --prompt hi \n d=$(date -u "${PIN[@]}" +%Y-%m-%d)'
  _row "llama-server counts as a server"   detect 'llama-server -m m.gguf & \n echo tok/s'
  _row "ollama counts as a server"         detect 'ollama serve & \n SECONDS=0'
  _row "vllm counts as a server"           detect 'vllm serve m & \n echo tokens_per_sec'
  _row "plain script is ignored"           ignore 'echo hello; ls -1'
  # THE WIDENING'S OWN ROWS. Restoring `git show 64cb68177^:scripts/gpu_2x_benchmark.sh`
  # verbatim as scripts/*.py left the PREVIOUS guard at rc=0 "count=0 baseline=0 OK"
  # — measured, not argued — because its universe was scripts/*.sh. These rows pin
  # the shapes a Python harness takes so the extension cannot silently rot back.
  _row "a python HTTP harness is detected"  detect 'import urllib.request
urllib.request.urlopen(req)
agg_tok_s = total / wall'
  _row "requests.post counts as a server"   detect 'import requests
requests.post(url, json=body)
print("tok/s", n / dt)'
  _row "a python report printer is ignored" ignore 'print(f"{agg_tok_s:>12.1f}")
json.load(fh)'
  _row "a python parser is ignored"         ignore 'import json
rec = json.load(open(path))'

  # the allowlist must actually exempt, and must not exempt everything
  if is_allowed "$ROOT/scripts/perf_gate.sh"; then
    printf '  ok    %-40s expect=exempt\n' "perf_gate.sh is allowlisted"; pass=$((pass + 1))
  else
    printf '  BROKE %-40s\n' "perf_gate.sh should be allowlisted"; fail=$((fail + 1))
  fi
  # The two entries the deletion/widening pair added must actually exempt, or
  # `ci / gate` reds on main the moment the universe grows (BASELINE is 0 and
  # shrink-only, so the widening and the allowlist have to land together).
  if is_allowed "$ROOT/scripts/perf041_client.py"; then
    printf '  ok    %-40s expect=exempt\n' "perf041_client.py is allowlisted"; pass=$((pass + 1))
  else
    printf '  BROKE %-40s\n' "perf041_client.py should be allowlisted"; fail=$((fail + 1))
  fi
  if is_allowed "$ROOT/scripts/lib/perf_receipt.py"; then
    printf '  ok    %-40s expect=exempt\n' "perf_receipt.py is allowlisted"; pass=$((pass + 1))
  else
    printf '  BROKE %-40s\n' "perf_receipt.py should be allowlisted"; fail=$((fail + 1))
  fi
  if is_allowed "$tmp/scripts/probe.sh"; then
    printf '  BROKE %-40s\n' "allowlist must not exempt everything"; fail=$((fail + 1))
  else
    printf '  ok    %-40s expect=not-exempt\n' "allowlist is not blanket"; pass=$((pass + 1))
  fi

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  rm -rf "${tmp:?refusing to rm an empty path}"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --selftest) selftest ;;
  "") gate ;;
  *) echo "usage: $0 [--selftest]" >&2; exit 2 ;;
esac
