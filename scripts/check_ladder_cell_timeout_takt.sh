#!/usr/bin/env bash
# check_ladder_cell_timeout_takt.sh — H1 (hard per-cell timeout = FAIL) and H5 (takt tripwires) in the cells producer.
#
# WHY. Operator 2026-09-28 17:50Z: "H1 hard per-cell timeout. Timeout = FAIL, blocks FINAL, never a skip. Default
# 10 min; after sampling, per class = 4 x measured p95 (floor 5 min). The timeout value is recorded in every
# receipt." and "H5 Takt tripwires after rc.1: at +2 h, +4 h, +6 h, completed cells >= 90% of plan, or ANDON
# immediately." A hung 4B cell held lambda's GPU for hours, and the old subprocess timeout killed only `flock`:
# the apr grandchild kept the card after the cell was written down. So the case that matters is not "the timer
# fired" but "the process GROUP is gone and the row says FAIL with the timeout it ran under".
#
# Runs the SHIPPED scripts/lib/model_ladder_cells_produce.py (imported), then each mutant must turn a case RED.
# Exit: 0 all cases + mutants as expected · 1 a case or mutant landed wrong · 2 could not check.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
LIB=scripts/lib
[ -f "$LIB/model_ladder_cells_produce.py" ] || { echo "check_ladder_cell_timeout_takt: producer absent" >&2; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/cell-tmo.XXXXXX") || exit 2
trap 'rm -rf -- "${W:?}"' EXIT

cases() { # <libdir> [quiet] -> 0 when every case lands as expected
  python3 - "$1" "$W" "${2:-}" <<'PY'
import importlib, json, os, sys, time
lib, w, quiet = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, lib)
P = importlib.import_module("model_ladder_cells_produce")
bad = 0
def case(name, ok, why):
    global bad
    if not ok:
        bad += 1
    if not quiet or not ok:
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else ": " + why))

# H1-a: a hung apr whose grandchild holds on -- the whole group must be gone, rc 124, timed_out set.
apr = os.path.join(w, "hang-apr")
pidf = os.path.join(w, "grandchild.pid")
open(apr, "w").write(f"#!/bin/sh\nsleep 20 &\necho $! > {pidf}\nwait\n")
os.chmod(apr, 0o755)
R = P.Runner(apr, "", 5, 2)
t0 = time.monotonic(); rc, out, err = R.call(["run"]); el = time.monotonic() - t0
time.sleep(0.3)
gpid = int(open(pidf).read())
alive = os.path.exists(f"/proc/{gpid}") and "Z" not in open(f"/proc/{gpid}/stat").read().split()[2]
case("timeout-kills-the-group", rc == 124 and R.timed_out and not alive and el < 10,
     f"rc={rc} timed_out={R.timed_out} grandchild_alive={alive} elapsed={el:.1f}s")
if alive:
    os.kill(gpid, 9)

# H1-b: the row a timeout produces is a FAIL naming the timeout, and carries timeout_s.
row = P.stamp({"verdict": "pass", "reason": "answered"}, time.monotonic(), 600, True)
case("timeout-row-is-fail", row["verdict"] == "fail" and row["reason"].startswith("TIMEOUT") and row["timeout_s"] == 600
     and row["timed_out"] is True, json.dumps(row))
row = P.stamp({"verdict": "pass", "reason": "answered"}, time.monotonic(), 600, False)
case("no-timeout-row-keeps-verdict", row["verdict"] == "pass" and row["timeout_s"] == 600 and "wall_s" in row, json.dumps(row))

# H1-c: class timeouts floor at 5 min; no class -> the 10-min default.
class A: pass
a = A(); a.timeout = P.DEFAULT_CELL_TIMEOUT_S; a.class_timeouts = {"m.gguf|run": 100, "m.gguf|chat": 1200}
case("class-timeout-floor", P.cell_timeout(a, {"file": "m.gguf"}, "run") == 300, str(P.cell_timeout(a, {"file": "m.gguf"}, "run")))
case("class-timeout-used", P.cell_timeout(a, {"file": "m.gguf"}, "chat") == 1200, "")
case("default-timeout-10min", P.cell_timeout(a, {"file": "m.gguf"}, "serve") == 600 and P.DEFAULT_CELL_TIMEOUT_S == 600, "")

# H5: at +2 h 9/10 is met (90%), at +4 h 17/20 is an ANDON, +6 h not yet reached -> not judged.
plan = os.path.join(w, "plan.json"); log = os.path.join(w, "takt.jsonl")
json.dump({"start_epoch": 1000, "checkpoints": [{"at_s": 7200, "planned_cells": 10}, {"at_s": 14400, "planned_cells": 20},
                                                {"at_s": 21600, "planned_cells": 30}]}, open(plan, "w"))
now = [1000 + 7200]
T = P.Takt(plan, log, clock=lambda: now[0])
P.COMPLETED[0] = 9
r1 = T.tick()
case("takt-90pct-met", len(r1) == 1 and not r1[0]["andon"] and not os.path.exists(log + ".andon"), json.dumps(r1))
now[0] = 1000 + 14400 + 5; P.COMPLETED[0] = 17
r2 = T.tick()
andon = open(log + ".andon").read() if os.path.exists(log + ".andon") else ""
case("takt-below-90pct-andon", len(r2) == 1 and r2[0]["andon"] and "ANDON H5" in andon, json.dumps(r2) + " andon=" + andon)
case("takt-future-not-judged", T.fired == 2 and len(open(log).read().splitlines()) == 2, f"fired={T.fired}")
try:
    json.dump({"start_epoch": 1, "checkpoints": []}, open(plan, "w")); P.Takt(plan, log); empty_ok = False
except ValueError:
    empty_ok = True
case("takt-empty-plan-refused", empty_ok, "an empty plan was accepted")
sys.exit(1 if bad else 0)
PY
}

echo "== cases"
timeout 120 bash -c "$(declare -f cases); W=$W; cases $LIB"; bad=$?
echo "== mutants (each must turn a case RED)"
mutant() { # <name> <old> <new>
  local md="$W/mut-$1"; mkdir -p "$md"; cp "$LIB"/*.py "$md/"
  python3 - "$md/model_ladder_cells_produce.py" "$2" "$3" <<'PY' || { echo "FAIL  mutant $1 did not apply -- it proves nothing"; bad=1; return; }
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
sys.exit(1) if s.count(old) != 1 else open(p, "w").write(s.replace(old, new, 1))
PY
  if timeout 120 bash -c "$(declare -f cases); W=$W; cases $md quiet" > /dev/null 2>&1; then echo "FAIL  mutant $1 SURVIVED"; bad=1; else echo "ok    mutant $1 killed"; fi
}
mutant kill-only-leader 'os.killpg(p.pid, signal.SIGKILL)' 'p.kill()'
mutant no-new-session 'text=True, start_new_session=True)' 'text=True)'
mutant timeout-not-fail '        row["verdict"] = "fail"
' '        pass
'
mutant no-floor 'return max(int(t), TIMEOUT_FLOOR_S) if' 'return int(t) if'
mutant takt-always-ok 'ok = done >= self.RATIO * plan' 'ok = True'
mutant takt-no-andon-file '                with open(self.log + ".andon", "a") as f:
                    f.write(line + "\n")' '                pass'
[ "$bad" = 0 ] && { echo "PASS H1 timeout (group kill, FAIL, timeout_s) + H5 takt: all cases and 6 mutants"; exit 0; }
echo "RED"; exit 1
