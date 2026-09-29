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
#   S4  it writes sigma-build/qm11-route.json, and the ONE reader is the join-decide step
#       (J1) -- any other reader in any workflow or section is RED;
#   J1  join C (ruled 08:32Z): one "QM-11 join C: decide" step in workspace-test-build,
#       id join, no if:, no continue-on-error, after the shadow and before the uploads,
#       running qm11_join.sh decide on the shadow's file into ws-lib-archive + sigma-build;
#   J2  one "QM-11 join C: compile once" step there, if exactly join != 'none', before the
#       uploads; `steps.join` conditions NO other step -- the join cannot switch one off;
#   J3  one "QM-11 join C: the router's lib tests" step in workspace-test-shard, if exactly
#       tier == 'quick' && shards != 1, not continue-on-error, AFTER the quick junit copies
#       (it adds to the quick tier, it replaces none of it), running the archive with -E;
#   J4  the workspace-test fan-in (ci.yml) requires every shard's join to equal the build
#       job's, and compares the union of the join junits with the build job's list, rc=1;
#   J5  qm11_join.sh's decision table passes, and each of its mutants turns it RED;
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
dec = [k for k, b in enumerate(blob) if 'QM-11 join C: decide' in b]
if len(dec) != 1:
    fails.append(f"J1: {len(dec)} 'QM-11 join C: decide' step(s) in workspace-test-build (want 1)")
else:
    d = dec[0]; db = blob[d]
    own |= {k for k in lines[d] if "qm11_join.sh decide" in text[k]}  # ... and its one reader
    if not re.search(r"^        id: join\s*$", db, re.M):
        fails.append("J1: the decide step is not id: join")
    if re.search(r"^        (if|continue-on-error):", db, re.M):
        fails.append("J1: the decide step has an if: or continue-on-error: (it must always decide)")
    want = 'bash scripts/ci/qm11_join.sh decide "$TIER" sigma-build/qm11-route.json ws-lib-archive sigma-build'
    if want not in db:
        fails.append("J1: the decide step does not run: " + want)
    if d < i or not upload or d > upload[0] or not any(
            "upload-artifact" in b and "name: ws-lib-archive" in b and k > d for k, b in enumerate(blob)):
        fails.append("J1: the decide step is not after the shadow and before both uploads")
    comp = [k for k, b in enumerate(blob) if 'QM-11 join C: compile once' in b]
    if len(comp) != 1:
        fails.append(f"J2: {len(comp)} 'QM-11 join C: compile once' step(s) (want 1)")
    else:
        c = comp[0]
        if not re.search(r"^        if: steps\.join\.outputs\.join != 'none'\s*$", blob[c], re.M):
            fails.append("J2: the join compile step is not if: steps.join.outputs.join != 'none'")
        if c < d or c > upload[0]:
            fails.append("J2: the join compile step is not between the decide step and the uploads")
        if "join.list.json" not in blob[c] or "--archive-file /workspace/ws-lib-archive/lib.tar.zst" not in blob[c]:
            fails.append("J2: the join compile step does not archive the lib tests and list the join")
alltext = open(sections).read().splitlines()
jl = [k for k, ln in enumerate(alltext) if "steps.join" in ln and not ln.lstrip().startswith("#")]
if len(jl) != 1 or "if: steps.join.outputs.join != 'none'" not in alltext[jl[0]]:
    fails.append(f"J2: steps.join appears on {len(jl)} line(s); only the join compile step's if: may read it")
# J3: the shard half.
try:
    ss = text.index("  workspace-test-shard:", text.index("jobs:") if "jobs:" in text else 0)
except ValueError:
    ss = None
if ss is None:
    fails.append("J3: no workspace-test-shard job in " + sections)
else:
    se = next((k for k in range(ss + 1, len(text)) if re.match(r"^  [A-Za-z0-9_-]+:\s*$", text[k])), len(text))
    sh, cur = [], None
    for ln in text[ss:se]:
        if re.match(r"^      - ", ln):
            cur = [ln]; sh.append(cur)
        elif cur is not None and (ln.startswith("        ") or not ln.strip()):
            cur.append(ln)
        elif cur is not None and not ln.startswith("      #"):
            cur = None
    sb = ["\n".join(x) for x in sh]
    run_ = [k for k, b in enumerate(sb) if "QM-11 join C: the router's lib tests" in b]
    if len(run_) != 1:
        fails.append(f"J3: {len(run_)} join run step(s) in workspace-test-shard (want 1)")
    else:
        r = run_[0]; rb = sb[r]
        if not re.search(r"^        if: steps\.tier\.outputs\.tier == 'quick' && matrix\.shards != 1\s*$", rb, re.M):
            fails.append("J3: the join run step is not if: tier == 'quick' && matrix.shards != 1")
        if re.search(r"^        continue-on-error:", rb, re.M):
            fails.append("J3: the join run step is continue-on-error")
        if '--archive-file "$ARCHIVE"' not in rb or '-E "$FLT"' not in rb:
            fails.append("J3: the join run step does not run the archive under the shipped filter")
        copies = [k for k, b in enumerate(sb) if "keep the quick tree-reader junit" in b or "keep the quick selected-crates junit" in b]
        if len(copies) != 2 or r < max(copies):
            fails.append("J3: the join run step is not after both quick junit copies (it must add, not interleave)")
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

# fanin <ci.yml>: J4, the workspace-test fan-in checks the join (prints FAIL lines; rc 1 on any).
fanin() {
  python3 - "$1" <<'PY'
import sys
t = open(sys.argv[1]).read()
need = {
  "a missing build decision is RED": '[ -s "$d/sigma-build/join" ] || {',
  "a missing shard decision is RED": '[ -f "$d/sigma-shard-$n/join" ] || {',
  "a shard that applied another decision is RED": '[ "$js" = "$jb" ] || {',
  "the union is compared with the build job's list": '"${j[@]}" --list-json "$d/sigma-build/join.list.json" || rc=1',
}
fails = [k for k, v in need.items() if v not in t]
for f in fails:
    print("FAIL J4:", f)
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
  fanin "$ROOT/.github/workflows/ci.yml" || rc=1
  bounded_curl "$ROOT/scripts/ci/tier_route_step.sh" \
    || { echo "FAIL S2: a curl in tier_route_step.sh is not bounded by --max-time"; rc=1; }
  o=$(mktemp); trap 'rm -f "${o:?}"' RETURN
  python3 "$ROOT/scripts/ci/tier_router.py" --self-test > "$o" 2>&1 || { tail -5 "$o"; echo "FAIL T1: tier_router.py --self-test"; rc=1; }
  tail -1 "$o"
  bash "$ROOT/scripts/ci/tier_route_step.sh" --self-test > "$o" 2>&1 || { grep FAIL "$o"; echo "FAIL T2: tier_route_step.sh --self-test"; rc=1; }
  tail -1 "$o"
  bash "$ROOT/scripts/ci/qm11_join.sh" --self-test > "$o" 2>&1 || { grep FAIL "$o"; echo "FAIL J5: qm11_join.sh --self-test"; rc=1; }
  tail -1 "$o"
  [ "$rc" -eq 0 ] && echo "PASS check_qm11_shadow: the shadow gates nothing; its one reader, join C, only adds"
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
    if structure "$t/s.yml" "$t/wf" > "$t/why"; then echo "FAIL  $1 stayed GREEN"; fail=$((fail + 1)); else echo "RED   $1 -- $(head -1 "$t/why")"; fi
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
  # Join C: each mutant is one way the join could narrow today's tier or stop deciding.
  mut "decide step made continue-on-error" "s.replace('        id: join\n', '        id: join\n        continue-on-error: true\n', 1)"
  mut "decide step made conditional" "s.replace('        id: join\n', '        id: join\n        if: success()\n', 1)"
  mut "decide reads another file" "s.replace('decide \"\$TIER\" sigma-build/qm11-route.json', 'decide \"\$TIER\" /dev/null', 1)"
  mut "join compile runs only on none" "s.replace(\"if: steps.join.outputs.join != 'none'\", \"if: steps.join.outputs.join == 'none'\", 1)"
  mut "the join switches a quick step off" "s.replace(\"      - name: \\\"Quick tier: every test target that reads the tree (BSE-17)\\\"\n        if: steps.tier.outputs.tier == 'quick'\n\", \"      - name: \\\"Quick tier: every test target that reads the tree (BSE-17)\\\"\n        if: steps.tier.outputs.tier == 'quick' && steps.join.outputs.join == 'none'\n\", 1)"
  mut "shard join step made continue-on-error" "s.replace(\"lib tests from the compile-once archive (#4529)\\\"\n        if: steps.tier.outputs.tier == 'quick' && matrix.shards != 1\n\", \"lib tests from the compile-once archive (#4529)\\\"\n        if: steps.tier.outputs.tier == 'quick' && matrix.shards != 1\n        continue-on-error: true\n\", 1)"
  mut "shard join step narrowed to one shard" "s.replace(\"lib tests from the compile-once archive (#4529)\\\"\n        if: steps.tier.outputs.tier == 'quick' && matrix.shards != 1\n\", \"lib tests from the compile-once archive (#4529)\\\"\n        if: steps.tier.outputs.tier == 'quick' && matrix.shard == 1\n\", 1)"
  mut "shard join run drops the shipped filter" "s.replace(' -E \"\$FLT\" --partition', ' --partition', 1)"
  mut "shard join step moved before the quick junit copies" "(lambda b: s.replace(b, '').replace('      - name: \"Quick tier: every test target that reads the tree (BSE-17)\"', b + '      - name: \"Quick tier: every test target that reads the tree (BSE-17)\"', 1))(s[s.index('      # QM-11 (#4529) join C, the shard half'):s.index('      - name: \"Σ-executed (quick tier): the tests this job ran')])"
  local ci="$ROOT/.github/workflows/ci.yml"
  fmut() { # fmut <name> <from> <to>: one fan-in mutant
    n=$((n + 1))
    python3 -c "import sys; s=open(sys.argv[1]).read(); open(sys.argv[2],'w').write(s.replace(sys.argv[3], sys.argv[4], 1))" "$ci" "$t/ci.yml" "$2" "$3"
    if cmp -s "$ci" "$t/ci.yml"; then echo "FAIL  $1 (mutant did not apply)"; fail=$((fail + 1)); return; fi
    if fanin "$t/ci.yml" > "$t/why"; then echo "FAIL  fan-in $1 stayed GREEN"; fail=$((fail + 1)); else echo "RED   fan-in $1 -- $(head -1 "$t/why")"; fi
  }
  fanin "$ci" >/dev/null || { echo "FAIL the real ci.yml fan-in is not GREEN"; fail=$((fail + 1)); }
  fmut "shard decisions no longer compared" '[ "$js" = "$jb" ] || {' 'true || {'
  fmut "join union not compared" '--list-json "$d/sigma-build/join.list.json" || rc=1' '--list-json "$d/sigma-build/join.list.json" || true'
  fmut "missing build decision tolerated" '[ -s "$d/sigma-build/join" ] || {' '[ -s "$d/sigma-build/join" ] || true || {'
  # The decision table: each mutant makes the join narrow or splice; its own table must turn RED.
  local jn="$ROOT/scripts/ci/qm11_join.sh"
  jmut() { # jmut <name> <from> <to>
    n=$((n + 1))
    python3 -c "import sys; s=open(sys.argv[1]).read(); open(sys.argv[2],'w').write(s.replace(sys.argv[3], sys.argv[4], 1))" "$jn" "$t/join.sh" "$2" "$3"
    if cmp -s "$jn" "$t/join.sh"; then echo "FAIL  $1 (mutant did not apply)"; fail=$((fail + 1)); return; fi
    if bash "$t/join.sh" --self-test > "$t/why" 2>&1; then echo "FAIL  join mutant $1 stayed GREEN"; fail=$((fail + 1)); else echo "RED   join mutant $1 -- $(grep -m1 FAIL "$t/why")"; fi
  }
  jmut "the today guard dropped" 'if today != "quick":' 'if False:'
  jmut "package names spliced unchecked" 're.fullmatch(r"[A-Za-z0-9_-]+", p)' 'True'
  jmut "an unknown route falls to none" 'say("t3", "all()", f"router {tier!r}: every workspace lib test")' 'say("none", "", "x")'
  jmut "T3 becomes none" 'if tier == "T0":' 'if tier in ("T0", "T3"):'
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
