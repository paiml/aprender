#!/usr/bin/env bash
# check_harness_server_cleanup.sh: a CRUX harness that is TERMed or INTed leaves no server behind.
# infra-66 found stub_server.py and crux_fake_serve.py servers 2-3 days old (ppid 1) on the intel
# clean-room runners: the harnesses trapped EXIT only, and a signal-killed bash does not run that trap
# unless INT and TERM exit. Behavioural: start the real harness, wait for its server, signal the
# harness, and require the server gone.
# Rows: 1 serve_code + TERM · 2 serve_code + KILL (no trap sees it; the watcher must) · 3 MUTANT without the watcher leaks
# (INT is not driven: a background job of a non-interactive shell starts with SIGINT ignored, which no trap can undo.)
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail
case "${1:-}" in -h|--help) printf 'usage: bash scripts/check_harness_server_cleanup.sh (no arguments)\n'; exit 0 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
for t in python3 flock curl; do command -v "$t" > /dev/null 2>&1 || { printf 'check_harness_server_cleanup: ENV - %s missing\n' "$t" >&2; exit 2; }; done
T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
FAIL=0
# serve_pid <harness pid>: the crux_fake_serve.py child of that harness, if any
serve_pid() { pgrep -P "$1" -f crux_fake_serve.py 2> /dev/null | head -1; }
# row <label> <signal> <harness script> <want: gone|leaked>
row() {
  local label="$1" sig="$2" script="$3" want="$4" h s i state
  bash "$script" > "$T/out.log" 2>&1 < /dev/null &
  h=$!
  s=""
  for i in $(seq 1 150); do s=$(serve_pid "$h"); [ -n "$s" ] && break; kill -0 "$h" 2> /dev/null || break; sleep 0.1; done
  if [ -z "$s" ]; then printf '  BROKE %s: harness never started its server\n' "$label"; FAIL=$((FAIL + 1)); kill "$h" 2> /dev/null; return; fi
  kill -"$sig" "$h"; wait "$h" 2> /dev/null
  for i in $(seq 1 80); do kill -0 "$s" 2> /dev/null || break; sleep 0.1; done
  if kill -0 "$s" 2> /dev/null; then state=leaked; kill -KILL "$s" 2> /dev/null; else state=gone; fi
  if [ "$state" = "$want" ]; then printf '  ok   %s (server %s)\n' "$label" "$state"
  else printf '  BROKE %s: server %s, wanted %s\n' "$label" "$state" "$want"; FAIL=$((FAIL + 1)); fi
}
H="$ROOT/scripts/check_crux_serve_code.sh"
row "serve_code TERM kills its server" TERM "$H" gone
row "serve_code KILL: the watcher ends its server" KILL "$H" gone
sed -e 's/; guard_server "$SRV"//' -e "s|^ROOT=.*|ROOT=\"$ROOT\"|" "$H" > "$T/mutant.sh"
cmp -s "$H" "$T/mutant.sh" && { echo "  BROKE mutant did not change the harness"; FAIL=$((FAIL + 1)); }
row "MUTANT without guard_server leaks the server on KILL" KILL "$T/mutant.sh" leaked
printf 'check_harness_server_cleanup: %d broke\n' "$FAIL"
[ "$FAIL" -eq 0 ]
