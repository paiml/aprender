#!/usr/bin/env bash
# check_qm11_shadow.sh -- the QM-11 (#4529) tier-router SHADOW can observe and never gate.
#
# The shadow step (ci/sections.yml, workspace-test-build) prints the tier router's
# decision beside today's tier. It is admitted to CI on one condition: it changes no
# gate. This guard makes that condition a check instead of a promise:
#   S1  the step exists in workspace-test-build, once;
#   S2  it is continue-on-error: true with a timeout-minutes of at most 10, a fault is turned
#       into a ::warning::, and every curl in the step script is bounded by --max-time;
#   S3  it runs AFTER the tier is staged (it cannot change what the shards and the
#       fan-in check read) and BEFORE the sigma-build upload (its file is published);
#   S4  it writes sigma-build/qm11-route.json, and no workflow or section reads that
#       file -- the day something does, the shadow has become a gate, and that is a
#       ruling (docs/audits/qm11-design-note.md), not a drive-by;
#   T1  tier_router.py's own case table and mutants still turn RED;
#   T2  tier_route_step.sh's own case table passes.
# --self-test mutates the wiring (sections.yml) AND the step script; every mutant must be RED.
#
# usage: check_qm11_shadow.sh [--self-test]
# exit:  0 PASS · 1 FAIL · 64 usage
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STEP_NAME='QM-11 shadow: the tier router'

# structure <sections.yml> <dir-to-scan-for-readers>...  -> prints FAIL lines; rc 1 on any
structure() {
  python3 - "$STEP_NAME" "$@" <<'PY'
import os, re, sys
name, sections, scan = sys.argv[1], sys.argv[2], sys.argv[3:]
text = open(sections).read().splitlines()
fails = []
# The workspace-test-build job: from its key to the next job key at the same indent.
try:
    start = text.index("  workspace-test-build:")
except ValueError:
    print("FAIL S1: no workspace-test-build job in", sections); sys.exit(1)
end = next((i for i in range(start + 1, len(text)) if re.match(r"^  [A-Za-z0-9_-]+:\s*$", text[i])), len(text))
job = text[start:end]
steps, lines, cur = [], [], None
for k, ln in enumerate(job, start):
    if re.match(r"^      - ", ln):
        cur = [ln]; steps.append(cur); lines.append([k])
    elif cur is not None and (ln.startswith("        ") or not ln.strip()):
        cur.append(ln); lines[-1].append(k)
    elif cur is not None and not ln.startswith("      #"):
        cur = None
blob = ["\n".join(s) for s in steps]
hits = [i for i, b in enumerate(blob) if name in b]
if len(hits) != 1:
    print(f"FAIL S1: {len(hits)} '{name}' step(s) in workspace-test-build (want 1)"); sys.exit(1)
i = hits[0]; s = blob[i]
if not re.search(r"^        continue-on-error: true\s*$", s, re.M):
    fails.append("S2: the shadow step is not continue-on-error: true")
m = re.search(r"^        timeout-minutes: (\d+)\s*$", s, re.M)
if not m or int(m.group(1)) > 10:
    fails.append("S2: the shadow step has no timeout-minutes <= 10 (a hung fetch would hold the job)")
if "|| echo \"::warning::qm11-shadow" not in s:
    fails.append("S2: a fault of the shadow step is not turned into a ::warning::")
stage = [k for k, b in enumerate(blob) if "Stage the tier this job decided" in b]
upload = [k for k, b in enumerate(blob) if "upload-artifact" in b and "name: sigma-build" in b]
if not stage or i < stage[0]:
    fails.append("S3: the shadow step does not run after 'Stage the tier this job decided'")
if not upload or i > upload[0]:
    fails.append("S3: the shadow step does not run before the sigma-build upload")
if "sigma-build/qm11-route.json" not in s:
    fails.append("S4: the shadow step does not write sigma-build/qm11-route.json")
own = set(lines[i])  # 0-based lines of the shadow step itself: the one writer
files = {os.path.realpath(sections): sections}
for d in scan:
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith((".yml", ".yaml")):
                files.setdefault(os.path.realpath(os.path.join(dp, f)), os.path.join(dp, f))
for real, p in sorted(files.items()):
    for k, ln in enumerate(open(p, errors="replace")):
        if "qm11-route.json" not in ln or ln.lstrip().startswith("#"):
            continue
        if real == os.path.realpath(sections) and k in own:
            continue
        fails.append(f"S4: {p}:{k + 1} reads qm11-route.json -- the shadow would gate")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
PY
}

# bounded_curl <step.sh>: every curl invocation line and the shared header array carry --max-time.
bounded_curl() {
  python3 - "$1" <<'PY'
import sys
lines = [l for l in open(sys.argv[1]) if not l.lstrip().startswith("#")]
hdr = [l for l in lines if "hdr=(" in l]
calls = [l for l in lines if "curl " in l]
ok = (len(hdr) == 1 and "--max-time" in hdr[0] and "--connect-timeout" in hdr[0]
      and calls and all('"${hdr[@]}"' in l for l in calls))
sys.exit(0 if ok else 1)
PY
}

run() {
  local rc=0 o
  command -v jq >/dev/null || { echo "FAIL T2: jq is not on PATH (input_set.sh needs it); NOT a skip"; return 1; }
  structure "$ROOT/ci/sections.yml" "$ROOT/ci" "$ROOT/.github/workflows" || rc=1
  bounded_curl "$ROOT/scripts/ci/tier_route_step.sh" \
    || { echo "FAIL S2: a curl in tier_route_step.sh is not bounded by --max-time"; rc=1; }
  o=$(mktemp); trap 'rm -f "${o:?}"' RETURN
  python3 "$ROOT/scripts/ci/tier_router.py" --self-test > "$o" 2>&1 || { tail -5 "$o"; echo "FAIL T1: tier_router.py --self-test"; rc=1; }
  tail -1 "$o"
  bash "$ROOT/scripts/ci/tier_route_step.sh" --self-test > "$o" 2>&1 || { grep FAIL "$o"; echo "FAIL T2: tier_route_step.sh --self-test"; rc=1; }
  tail -1 "$o"
  [ "$rc" -eq 0 ] && echo "PASS check_qm11_shadow: the shadow observes and gates nothing"
  return "$rc"
}

# Each mutant is one way the shadow could start to gate or stop reporting; each must be RED.
self_test() {
  local t fail=0 n=0
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' RETURN
  mkdir -p "$t/wf"
  structure "$ROOT/ci/sections.yml" "$t/wf" >/dev/null || { echo "FAIL the real sections.yml is not GREEN"; return 1; }
  mut() { # mut <name> <python expression over s>
    n=$((n + 1))
    python3 -c "import sys; s=open(sys.argv[1]).read(); s=$2; open(sys.argv[2],'w').write(s)" "$ROOT/ci/sections.yml" "$t/s.yml"
    if cmp -s "$ROOT/ci/sections.yml" "$t/s.yml"; then echo "FAIL  $1 (mutant did not apply)"; fail=$((fail + 1)); return; fi
    if structure "$t/s.yml" "$t/wf" >/dev/null; then echo "FAIL  $1 stayed GREEN"; fail=$((fail + 1)); else echo "RED   $1"; fi
  }
  mut "continue-on-error dropped" "s.replace('        continue-on-error: true\n        timeout-minutes: 5\n', '        timeout-minutes: 5\n', 1)"
  mut "fault no longer a warning" "s.replace('|| echo \"::warning::qm11-shadow', '|| exit 1 # ', 1)"
  mut "step renamed away (not wired)" "s.replace('QM-11 shadow: the tier router', 'QM-11 shadow: renamed', 1)"
  mut "step moved before the tier is staged" "(lambda a, b: s.replace(b, '').replace('      - name: \"Stage the tier this job decided', b + '      - name: \"Stage the tier this job decided', 1))(None, s[s.index('      - name: \"QM-11 shadow'):s.index('      - uses: actions/upload-artifact@v7\n        with:\n          name: ws-lib-archive')])"
  mut "a later step reads qm11-route.json" "s.replace('      - *ws_own\n', '      - name: gate on it\n        run: jq -e .router sigma-build/qm11-route.json\n      - *ws_own\n', 1)"
  # A reader in ANOTHER workflow file is caught too.
  n=$((n + 1))
  printf 'jobs:\n  x:\n    steps:\n      - run: cat sigma-build/qm11-route.json\n' > "$t/wf/other.yml"
  if structure "$ROOT/ci/sections.yml" "$t/wf" >/dev/null; then echo "FAIL  a reader in another workflow stayed GREEN"; fail=$((fail + 1)); else echo "RED   a reader in another workflow"; fi
  mut "timeout-minutes dropped" "s.replace('        continue-on-error: true\n        timeout-minutes: 5\n', '        continue-on-error: true\n', 1)"
  mut "timeout-minutes raised past 10" "s.replace('        continue-on-error: true\n        timeout-minutes: 5\n', '        continue-on-error: true\n        timeout-minutes: 60\n', 1)"
  # The step script: each mutant is one way it could lie; its own case table must turn RED.
  # QM11_BIN points the mutated copy at the real helpers beside the real script.
  local step="$ROOT/scripts/ci/tier_route_step.sh"
  smut() { # smut <key>: the mutant table below; one fixed-string replacement each
    n=$((n + 1))
    python3 - "$step" "$t/step.sh" "$1" <<'PY'
import sys
src, dst, key = sys.argv[1:4]
M = {
  # the pre-quorum shape: the ref pasted into the child script between single quotes
  "injection": ("""'bash "$BIN/input_set.sh" base "$W/input-set.json" . "$QM11_BASE" > "$W/base.log" 2>&1'""",
                """"bash \\"\\$BIN/input_set.sh\\" base \\"\\$W/input-set.json\\" . '$QM11_BASE' > \\"\\$W/base.log\\" 2>&1\""""),
  "owners-dropped": ('args+=(--owners "$work/owners.json"); ', ''),
  "stale-as-valid": ('args+=(--stale "$work/stale.txt")', 'args+=(--base-valid)'),
  "refusal-ignored": ('if ! why=$(in_env', 'if false && ! why=$(in_env'),
}
a, b = M[key]
s = open(src).read()
open(dst, "w").write(s.replace(a, b, 1))
PY
    if cmp -s "$step" "$t/step.sh"; then echo "FAIL  $1 (mutant did not apply)"; fail=$((fail + 1)); return; fi
    if QM11_BIN="$ROOT/scripts/ci" bash "$t/step.sh" --self-test >/dev/null 2>&1; then
      echo "FAIL  step mutant $1 stayed GREEN"; fail=$((fail + 1))
    else echo "RED   step mutant $1"; fi
  }
  smut injection; smut owners-dropped; smut stale-as-valid; smut refusal-ignored
  n=$((n + 1))
  sed 's/--connect-timeout 10 --max-time 60 //' "$step" > "$t/step.sh"
  if bounded_curl "$t/step.sh"; then echo "FAIL  unbounded curl stayed GREEN"; fail=$((fail + 1)); else echo "RED   unbounded curl"; fi
  n=$((n + 1))
  sed 's/-fsSL "\${hdr\[@\]}"/-fsSL/' "$step" > "$t/step.sh"
  if cmp -s "$step" "$t/step.sh"; then echo "FAIL  curl-without-hdr (mutant did not apply)"; fail=$((fail + 1))
  elif bounded_curl "$t/step.sh"; then echo "FAIL  a curl without the bounded header stayed GREEN"; fail=$((fail + 1))
  else echo "RED   a curl without the bounded header"; fi
  bounded_curl "$step" || { echo "FAIL the real step script is not bounded"; fail=$((fail + 1)); }
  echo "check_qm11_shadow self-test: $((n - fail))/$n mutants RED"
  [ "$fail" -eq 0 ]
}

case "${1:-}" in
  --self-test) self_test ;;
  -h|--help) sed -n '/^# usage:/,/^# exit:/p' "$0"; echo "  --self-test  every mutant of the wiring must turn RED" ;;
  "") run ;;
  *) echo "usage: check_qm11_shadow.sh [--self-test]" >&2; exit 64 ;;
esac
