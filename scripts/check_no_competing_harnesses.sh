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
BASELINE=1

# Allowlisted, each with the reason it is not a competing definition.
is_allowed() {
  case "$1" in
    */perf_gate.sh)                 return 0 ;;  # THE entrypoint
    */check_no_competing_harnesses.sh) return 0 ;;  # this detector: its own
                                                     # selftest fixtures contain the
                                                     # trigger strings, so without
                                                     # this it matches itself and
                                                     # inflates the baseline by one,
                                                     # hiding a real harness
    */perf000_serialization_probe.sh) return 0 ;;  # PERF-000 falsifier: measures
                                                   # wall-clock scaling, derives no
                                                   # tok/s and no comparator ratio
    *) return 1 ;;
  esac
}

# A script qualifies only if it does BOTH.
starts_server() { grep -qE "serve run|llama-server|ollama serve|vllm serve" "$1" 2>/dev/null; }
computes_rate() { grep -qE "date \+%s|tok/s|tokens?_per_sec|SECONDS" "$1" 2>/dev/null; }

# UNIVERSE: tracked UNION working tree. A `git ls-files`-only universe gives an
# untracked harness a free pass, which is how three earlier guards were blind.
universe() {
  { git -C "$ROOT" ls-files 'scripts/*.sh' 'scripts/**/*.sh' 2>/dev/null || true
    find "$ROOT/scripts" -name '*.sh' -type f 2>/dev/null | sed "s|^$ROOT/||" || true
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
  echo "$found" > /tmp/.cnch_count
}

gate() {
  echo "=== competing benchmark harnesses (PERF-009 / H4) ==="
  scan "$ROOT"
  local n; n=$(cat /tmp/.cnch_count)
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
  _row "llama-server counts as a server"   detect 'llama-server -m m.gguf & \n echo tok/s'
  _row "ollama counts as a server"         detect 'ollama serve & \n SECONDS=0'
  _row "vllm counts as a server"           detect 'vllm serve m & \n echo tokens_per_sec'
  _row "plain script is ignored"           ignore 'echo hello; ls -1'

  # the allowlist must actually exempt, and must not exempt everything
  if is_allowed "$ROOT/scripts/perf_gate.sh"; then
    printf '  ok    %-40s expect=exempt\n' "perf_gate.sh is allowlisted"; pass=$((pass + 1))
  else
    printf '  BROKE %-40s\n' "perf_gate.sh should be allowlisted"; fail=$((fail + 1))
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
