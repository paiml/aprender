#!/usr/bin/env bash
# check_harness_server_cleanup.sh: a CRUX harness that dies leaves no fixture server behind.
# infra-66 found stub_server.py and crux_fake_serve.py servers 2-3 days old (ppid 1) on the intel
# clean-room runners. bash runs its EXIT trap on TERM but nothing runs on SIGKILL (a cancelled job), so
# each launcher starts a watcher per server (guard_server). Behavioural rows; every mutant strips the
# mechanism from a copy of the launcher and must leak (a mutant that changes nothing is a BROKE).
#   F  guard_server itself, extracted from each of the 3 launchers: harness KILLed -> server gone
#   S  the call sites: each launcher calls guard_server on every server it starts (mutant: call removed)
#   R  the real harness run, KILLed: check_crux_serve_code (fake serve) and check_crux_ollama_in_lock
#      (ollama stub, and the apr stub that dogfood execs inside a child)
# TERM is not a row: bash runs the EXIT trap on TERM with no trap of ours, so there is no code to prove.
# INT is not driven: a background job of a non-interactive shell starts with SIGINT ignored.
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail
case "${1:-}" in -h|--help) printf 'usage: bash scripts/check_harness_server_cleanup.sh (no arguments)\n'; exit 0 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
for t in python3 flock curl pgrep; do command -v "$t" > /dev/null 2>&1 || { printf 'check_harness_server_cleanup: ENV - %s missing\n' "$t" >&2; exit 2; }; done
T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
FAIL=0
LAUNCHERS="check_crux_serve_code check_crux_ollama_in_lock crux_inference_dogfood"
ok() { printf '  ok   %s\n' "$1"; }
broke() { printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }
# mutant <src> <out> <sed args...>: a copy of <src> that must differ; ROOT is pinned to the real tree
mutant() {
  local src="$1" out="$2"; shift 2
  sed -e "$@" -e "s|^ROOT=.*|ROOT=\"$ROOT\"|" "$src" > "$out"
  if cmp -s "$src" "$out"; then broke "mutant of $(basename "$src") changed nothing (vacuous)"; return 1; fi
}
# gone <pid>: wait up to 20 s for it to exit; prints gone|leaked and reaps a leak
gone() { local i; for i in $(seq 1 100); do kill -0 "$1" 2> /dev/null || { echo gone; return; }; sleep 0.2; done; kill -KILL "$1" 2> /dev/null; echo leaked; }
verdict() { # <label> <state> <want>
  if [ "$2" = "$3" ]; then ok "$1 (server $2)"; else broke "$1: server $2, wanted $3"; fi
}
# F: guard_server alone, driven by a throwaway harness that is then KILLed
fn_row() { # <label> <launcher file> <want>
  local body="$T/fn.sh" h s st
  sed -n '/^guard_server() {/,/^}/p' "$2" > "$body"
  [ -s "$body" ] || { broke "$1: no guard_server() in $2"; return; }
  : > "$T/srv.pid"
  bash -c '. "$1"; sleep 300 & srv=$!; guard_server "$srv"; echo "$srv" > "$2"; sleep 300' x "$body" "$T/srv.pid" < /dev/null > /dev/null 2>&1 &
  h=$!
  for _ in $(seq 1 50); do [ -s "$T/srv.pid" ] && break; sleep 0.1; done
  s=$(cat "$T/srv.pid" 2> /dev/null)
  [ -n "$s" ] || { broke "$1: harness never started its server"; kill -KILL "$h" 2> /dev/null; return; }
  kill -KILL "$h"; wait "$h" 2> /dev/null
  st=$(gone "$s"); pkill -KILL -P "$h" 2> /dev/null
  verdict "$1" "$st" "$3"
}
for l in $LAUNCHERS; do
  f="$ROOT/scripts/$l.sh"
  fn_row "F $l: harness KILLed, guard_server ends the server" "$f" gone
  sed 's/^guard_server() {$/guard_server() { return 0/' "$f" > "$T/m-$l.sh"
  if cmp -s "$f" "$T/m-$l.sh"; then broke "F mutant of $l changed nothing"; else fn_row "F MUTANT $l: guard_server a no-op leaks" "$T/m-$l.sh" leaked; fi
done
# S: every server start in a launcher is followed by guard_server
for l in $LAUNCHERS; do
  f="$ROOT/scripts/$l.sh"
  n=$(grep -cE '(^|[;[:space:]])guard_server "' "$f")
  if [ "$n" -ge 1 ]; then ok "S $l calls guard_server on its server ($n site)"; else broke "S $l never calls guard_server"; fi
  if mutant "$f" "$T/s-$l.sh" 's/; guard_server "[^"]*"//'; then
    n=$(grep -cE '(^|[;[:space:]])guard_server "' "$T/s-$l.sh")
    if [ "$n" -eq 0 ]; then ok "S MUTANT $l: call removed -> the row goes red"; else broke "S MUTANT $l still has $n call(s)"; fi
  fi
done
# descendants <pid>: every live descendant, so a row never picks another session's server
descendants() { local c; for c in $(pgrep -P "$1" 2> /dev/null); do echo "$c"; descendants "$c"; done; }
# R: the real harness, KILLed. <label> <script> <process pattern of the server> <want>
real_row() {
  local label="$1" script="$2" pat="$3" want="$4" h s i st
  bash "$script" > "$T/out.log" 2>&1 < /dev/null &
  h=$!
  s=""
  for i in $(seq 1 600); do s=$(descendants "$h" | while read -r d; do tr "\0" " " < "/proc/$d/cmdline" 2> /dev/null | grep -aq -- "$pat" && echo "$d"; done | head -1); [ -n "$s" ] && break; kill -0 "$h" 2> /dev/null || break; sleep 0.1; done
  [ -n "$s" ] || { broke "$label: harness never started a server matching '$pat'"; kill -KILL "$h" 2> /dev/null; return; }
  # a cancelled job also takes the dogfood child (which would otherwise end its own server): kill it and its timeout
  for d in $(descendants "$h"); do tr "\0" " " < "/proc/$d/cmdline" 2> /dev/null | grep -aqE "crux_inference_dogfood|^timeout " && kill -KILL "$d" 2> /dev/null; done
  kill -KILL "$h"; wait "$h" 2> /dev/null
  st=$(gone "$s")
  verdict "$label" "$st" "$want"
  # the harness dies, but its other children may not: reap what this row started
  pkill -KILL -f "$T/" 2> /dev/null; sleep 0.3
}
S1="$ROOT/scripts/check_crux_serve_code.sh"
real_row "R serve_code KILLed: the watcher ends the fake serve" "$S1" 'crux_fake_serve.py --port' gone
mutant "$S1" "$T/r1.sh" 's/; guard_server "[^"]*"//' && real_row "R MUTANT serve_code without the call leaks" "$T/r1.sh" 'crux_fake_serve.py --port' leaked
S2="$ROOT/scripts/check_crux_ollama_in_lock.sh"
real_row "R ollama_in_lock KILLed: the watcher ends the ollama stub" "$S2" 'stub_server.py ollama' gone
mutant "$S2" "$T/r2.sh" 's/; guard_server "[^"]*"//' && real_row "R MUTANT ollama_in_lock without the call leaks" "$T/r2.sh" 'stub_server.py ollama' leaked
real_row "R ollama_in_lock KILLed: the watcher ends the apr stub" "$S2" 'stub_server.py apr' gone
# the apr stub dies on its own once its dogfood child is gone in this simulation, so the mutant cannot be shown to leak;
# the watcher is pinned structurally instead: its cmdline-checked kill must exist, and the mutant (block deleted) has none
n=$(grep -c "&& kill \"\$stub\"" "$S2")
if [ "$n" -ge 1 ]; then ok "S apr stub carries its cmdline-checked watcher ($n kill)"; else broke "S apr stub has no watcher kill"; fi
if mutant "$S2" "$T/r3.sh" '/^    ( sleep 1; id=\$(tr .* "\/proc\/\$stub\/cmdline"/,/kill "\$stub" 2> \/dev\/null ) < \/dev\/null/d'; then
  n=$(grep -c "&& kill \"\$stub\"" "$T/r3.sh"); if [ "$n" -eq 0 ]; then ok "S MUTANT apr stub without its watcher -> the row goes red"; else broke "S MUTANT apr stub still has $n watcher kill"; fi
fi
printf 'check_harness_server_cleanup: %d broke\n' "$FAIL"
[ "$FAIL" -eq 0 ]
