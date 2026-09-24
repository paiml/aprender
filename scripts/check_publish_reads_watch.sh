#!/usr/bin/env bash
# check_publish_reads_watch.sh -- the publish step RE-READS the candidate watch's verdict (#4045 M8, shift-left).
#
# BEHAVIOURAL, not a grep (the check_tag_step_gated.sh pattern): watch_gate() and run_preflight() are EXTRACTED from
# scripts/release/autopilot.sh and RUN against stub verdicts, once per outcome, and the transcript is judged:
#     a fresh GREEN verdict for this release commit -> the preflight runs
#     no verdict / another sha / older than the bound / ANDON -> DIE, and the preflight never runs
# Then each rule is deleted in a copy of the autopilot and the row that names it must go RED.
#
#   check_publish_reads_watch.sh    (exit 0 every row and mutant landed, 1 not, 2 ENV)
set -uo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_publish_reads_watch.sh   (the case table; no arguments)"; exit 0 ;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/release/autopilot.sh"
T=$(mktemp -d) || exit 2
cleanup() { if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T"; fi; }
trap cleanup EXIT
MC=$(printf 'a%.0s' $(seq 1 40))

run() { # run <autopilot copy> <case> -> transcript (SAY/DIE/PREFLIGHT-RAN lines)
  local ap=$1 c=$2 w="$T/case-$2" fn
  mkdir -p "$w/state/0.70.0" "$w/scripts" "$w/ap"
  printf '#!/bin/bash\necho PREFLIGHT-RAN\nexit 0\n' > "$w/scripts/check_publish_preflight.sh"
  fn=$(awk '/^watch_gate\(\) \{/,/^\}/' "$ap"; awk '/^run_preflight\(\) \{/,/^\}/' "$ap")
  [ -n "$fn" ] || { echo "MISSING-FUNCTIONS"; return 2; }
  # a repo per case: C = the candidate the watch measured; S = the SQUASH merge of it (same tree, another sha, as a
  # squash-merged bump PR makes); D = a release commit whose CODE differs from the candidate
  local gg="git -C $w -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t" C S D mc
  $gg init -q && mkdir -p "$w/src" && echo code > "$w/src/lib.rs" && $gg add src && $gg commit -qm base
  echo candidate >> "$w/src/lib.rs" && $gg commit -qam candidate && C=$($gg rev-parse HEAD)
  S=$($gg commit-tree "$C^{tree}" -p "$C~1" -m squash)
  echo other >> "$w/src/lib.rs" && $gg commit -qam differs && D=$($gg rev-parse HEAD)
  mc=$S; [ "$c" = squash-code-differs ] && mc=$D
  python3 - "$w/state/0.70.0" "$c" "$mc" "$C" <<'PY'
import json, os, sys, time
d, case, mc, cand = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if case == "missing":
    sys.exit(0)
now = time.time()
w = {"schema": "apr-candidate-watch/v1", "version": "0.70.0", "sha": mc, "andon": False, "real_red": [], "bookkeeping_red": ["dogfood:bashrs"],
     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - (7 * 3600 if case in ("stale", "stale-fresh-mtime") else 60)))}
if case == "other-sha":
    w["sha"] = "b" * 40
if case in ("squash-same-tree", "squash-code-differs"):
    w["sha"] = cand   # the watch measured the candidate; the release commit is its squash
if case == "corrupt-newest":
    json.dump(w, open(os.path.join(d, "watch-a.json"), "w"))            # a fresh green ...
    open(os.path.join(d, "watch-b.json"), "w").write("{corrupt")        # ... beside an unreadable (maybe newer) verdict
    sys.exit(0)
if case == "older-green-newer-andon":
    newer = dict(w, andon=True, real_red=["dogfood:model-parity"], at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60)))
    json.dump(newer, open(os.path.join(d, "watch-b.json"), "w"))
    w["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 3600))   # older by its own clock ...
    p = os.path.join(d, "watch-a.json"); json.dump(w, open(p, "w"))
    os.utime(os.path.join(d, "watch-b.json"), (now - 7200, now - 7200))      # ... but copied LATER (newer mtime)
    sys.exit(0)
if case == "andon":
    w["andon"], w["real_red"] = True, ["dogfood:model-parity"]
p = os.path.join(d, "watch-20260923T000000Z.json")
json.dump(w, open(p, "w"))
if case == "stale":
    t = time.time() - 7 * 3600
    os.utime(p, (t, t))
# stale-fresh-mtime: the verdict is 7 h old by its own `at`, its FILE was just copied (fresh mtime)
PY
  ( cd "$w" && V=0.70.0 MC="$mc" AP="$w/ap" LOG="$w/log" STATUS="$w/status" CANDIDATE_WATCH_STATE="$w/state" bash -c '
      say() { echo "SAY $*"; }
      die() { echo "DIE $*"; exit 1; }
      '"$fn"'
      run_preflight' 2>&1; cat "$w/ap/preflight.log" 2> /dev/null )   # the stub writes PREFLIGHT-RAN there
}
table() { # table <autopilot> -> ok/FAIL <row> lines
  local ap=$1 out
  out=$(run "$ap" fresh)
  if grep -q '^SAY WATCH GREEN ' <<< "$out" && grep -q '^PREFLIGHT-RAN' <<< "$out"; then echo "ok    fresh-green-publishes"
  else echo "FAIL  fresh-green-publishes -- $(tr '\n' ' ' <<< "$out" | cut -c1-160)"; fi
  out=$(run "$ap" squash-same-tree)
  if grep -q '^SAY WATCH GREEN ' <<< "$out" && grep -q '^PREFLIGHT-RAN' <<< "$out"; then echo "ok    squash-same-tree-admitted"
  else echo "FAIL  squash-same-tree-admitted -- $(tr '\n' ' ' <<< "$out" | cut -c1-160)"; fi
  for c in missing other-sha squash-code-differs stale stale-fresh-mtime andon older-green-newer-andon corrupt-newest; do
    out=$(run "$ap" "$c")
    case "$c" in
      missing) needle="no candidate-watch verdict" ;; other-sha) needle="not the release commit" ;;
      squash-code-differs) needle="not the release commit" ;; older-green-newer-andon) needle="ANDON" ;;
      corrupt-newest) needle="cannot be read -- the newest cannot be told" ;;
      stale|stale-fresh-mtime) needle="h old (> 6 h)" ;; andon) needle="ANDON: REAL gate(s) red: dogfood:model-parity" ;;
    esac
    if grep -q "^DIE candidate watch: .*$needle" <<< "$out" && ! grep -q '^PREFLIGHT-RAN' <<< "$out"; then echo "ok    $c-refused"
    else echo "FAIL  $c-refused -- $(tr '\n' ' ' <<< "$out" | cut -c1-160)"; fi
  done
}
bad=0
out=$(table "$SUBJECT"); printf '%s\n' "$out"; grep -q '^FAIL' <<< "$out" && bad=1
mutant() { # mutant <label> <row> <old> <new>
  local m="$T/ap-$1.sh" mo
  python3 -c 'import sys; s=open(sys.argv[1]).read(); assert s.count(sys.argv[3])==1; open(sys.argv[2],"w").write(s.replace(sys.argv[3],sys.argv[4]))' \
    "$SUBJECT" "$m" "$3" "$4" 2> /dev/null || { echo "FAIL  mutant $1 did not apply"; bad=1; return; }
  mo=$(table "$m")
  if grep -q "^FAIL  $2 " <<< "$mo"; then echo "ok    mutant $1 killed by $2"; else echo "FAIL  mutant $1 SURVIVED $2"; bad=1; fi
}
mutant gate-uncalled missing-refused '    watch_gate "${CANDIDATE_WATCH_STATE:-$HOME/.local/state/aprender-candidate-watch}" "$V" "$MC" "${CANDIDATE_WATCH_MAX_AGE_H:-6}"' '    :'
mutant sha-unchecked other-sha-refused 'if w.get("schema") != "apr-candidate-watch/v1" or not same:' 'if w.get("schema") != "apr-candidate-watch/v1":'
mutant age-unchecked stale-refused 'if age > max_h or age < -0.1:' 'if False:'
mutant tree-equivalence-off squash-same-tree-admitted 'same = ws_sha == sha or (bool(ws_sha) and subprocess.run(' 'same = ws_sha == sha or (False and subprocess.run('
mutant corrupt-skipped corrupt-newest-refused '        print("the watch verdict %s cannot be read -- the newest cannot be told" % os.path.basename(f)); sys.exit(1)' '        pass'
mutant newest-by-mtime older-green-newer-andon-refused 'w = max(ws, key=at_of)' 'w = json.load(open(max(glob.glob(os.path.join(state, v, "watch-*.json")), key=os.path.getmtime)))'
mutant age-from-mtime stale-fresh-mtime-refused 'age = (time.time() - at_of(w)) / 3600.0' 'age = (time.time() - max(os.path.getmtime(f) for f in glob.glob(os.path.join(state, v, "watch-*.json")))) / 3600.0'
mutant andon-unchecked andon-refused 'if w.get("andon") or w.get("real_red"):' 'if False:'
echo "check_publish_reads_watch: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
exit "$bad"
