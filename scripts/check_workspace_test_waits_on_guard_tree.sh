#!/usr/bin/env bash
# check_workspace_test_waits_on_guard_tree.sh — a RED guard-tree stops the workspace-test shards
# before they take a runner, and the required check `workspace-test` still REPORTS (#3177).
#
# WHY. A merge group whose guard-tree was already red kept all three workspace-test shards running,
# and `gate` reported the red only after them: 2.8 h of clean-room time over the last 20 red merge
# groups (aprender-dd, 2026-09-24). ci.yml now gives workspace-test-shard `needs: [guard-tree]`.
# The danger of that edit is the REQUIRED check: branch protection and the merge queue wait on a
# check named `workspace-test`, and a skipped job whose name is required blocks nothing and says
# nothing, so a queue can wait on it forever. The fan-in job must therefore still RUN and go RED.
#
# It reads the SHIPPED .github/workflows/ci.yml, evaluates GitHub's job-status rules over it (a job
# with `needs:` and no `if:` runs only when every need succeeded; `if: always()` runs regardless),
# and EXECUTES the fan-in's own `run:` script with the resulting `needs.*.result` values:
#   guard-red        guard-tree failure   -> shards skipped, workspace-test runs, exits non-zero,
#                                            and its ::error:: line names guard-tree
#   guard-cancelled  guard-tree cancelled -> shards skipped, workspace-test runs and is red
#   guard-green      guard-tree success   -> shards run; workspace-test green when they pass
#   shard-red        guard green, a shard failed -> workspace-test red (the old behaviour holds)
#   required-name    the fan-in's check-run name is exactly `workspace-test`
# --self-test plants: the shards' needs removed · `if: always()` on the shards · the fan-in's
# `if: always()` removed · the fan-in's skipped arm exiting 0. Each must turn a row RED.
#
# Exit: 0 all as expected · 1 a row landed wrong · 2 could not check.
set -uo pipefail
CI=".github/workflows/ci.yml"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ci) [ $# -ge 2 ] || { echo "--ci needs a value" >&2; exit 2; }; CI="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_workspace_test_waits_on_guard_tree: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -f "$CI" ] || { echo "  cannot check: $CI not found" >&2; exit 2; }
python3 -c 'import yaml' 2>/dev/null || { echo "  cannot check: python3 yaml module missing" >&2; exit 2; }
T=$(mktemp -d -t check_wt_waits_guard.XXXXXXXX) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
cleanup() { case "${T:-}" in /tmp/?*) if [ -n "$T" ] && [ -d "$T" ]; then rm -rf -- "$T" || :; fi ;; esac; }
trap cleanup EXIT

# Prints one line per row: "ok <row>" or "FAIL <row>: <why>". Exit 2 when the file cannot be read.
run_rows() { # <ci.yml>
  python3 - "$1" "$T" <<'PY'
import os, subprocess, sys, yaml
ci, tmp = sys.argv[1], sys.argv[2]
try:
    jobs = yaml.safe_load(open(ci))["jobs"]
except Exception as e:  # noqa: BLE001
    print(f"cannot parse {ci}: {e}", file=sys.stderr); sys.exit(2)
SH, FAN, GT = "workspace-test-shard", "workspace-test", "guard-tree"
RELEASE_PUSH_SKIP = "!(github.event_name == 'push' && startsWith(github.ref, 'refs/heads/release/'))"
for j in (SH, FAN, GT):
    if j not in jobs:
        print(f"FAIL structure: job '{j}' is not in {ci}"); sys.exit(0)

def needs(j):
    n = jobs[j].get("needs", [])
    return [n] if isinstance(n, str) else list(n)

def runs(j, results):
    """GitHub: no `if:` means success(); always() runs whatever the needs concluded."""
    cond = str(jobs[j].get("if", "")).replace("${{", "").replace("}}", "").strip()
    # #4112 skips these jobs on a push to release/*; every row here is a pull_request / merge_group event, where
    # that clause is true. Only this EXACT clause is reduced: any other `if:` stays unmodelled and fails the row.
    for clause in (" && " + RELEASE_PUSH_SKIP, RELEASE_PUSH_SKIP):
        cond = cond.replace(clause, "")
    if cond == "always()":
        return True
    if cond not in ("", "success()"):
        return None  # a condition this table does not model
    return all(results.get(n) == "success" for n in needs(j))

def fan_in(results):
    """Execute the fan-in's shipped run: scripts with its env rendered from results."""
    out, rc = "", 0
    for st in jobs[FAN].get("steps", []):
        if "run" not in st:
            continue
        env = dict(os.environ)
        for k, v in (st.get("env") or {}).items():
            v = str(v)
            for n, r in results.items():
                v = v.replace("${{ needs.%s.result }}" % n, r)
            if "${{" in v:
                return None, f"env {k}={v!r} reads something this table does not render"
            env[k] = v
        p = subprocess.run(["bash", "-c", st["run"]], env=env, capture_output=True, text=True, cwd=tmp)
        out += p.stdout + p.stderr
        if p.returncode != 0:
            rc = p.returncode; break
    return rc, out

def row(name, guard, shard_outcome, want_shards_run, want_fan_green, want_in_error=None):
    results = {GT: guard}
    sr = runs(SH, results)
    if sr is None:
        print(f"FAIL {name}: {SH} has an `if:` this table does not model: {jobs[SH].get('if')!r}"); return
    if sr != want_shards_run:
        print(f"FAIL {name}: guard-tree {guard} -> shards {'RUN' if sr else 'SKIPPED'}, want {'run' if want_shards_run else 'skipped'}"); return
    results[SH] = shard_outcome if sr else "skipped"
    fr = runs(FAN, results)
    if fr is None:
        print(f"FAIL {name}: {FAN} has an `if:` this table does not model: {jobs[FAN].get('if')!r}"); return
    if not fr:
        print(f"FAIL {name}: the required check `{FAN}` is SKIPPED (shards {results[SH]}) -- a merge queue would wait on it forever"); return
    rc, out = fan_in(results)
    if rc is None:
        print(f"FAIL {name}: {out}"); return
    if (rc == 0) != want_fan_green:
        print(f"FAIL {name}: `{FAN}` exited {rc} with shards {results[SH]}, want {'green' if want_fan_green else 'red'}; output: {out.strip()[:200]}"); return
    if want_in_error and want_in_error not in out:
        print(f"FAIL {name}: `{FAN}`'s output does not name {want_in_error!r}: {out.strip()[:200]}"); return
    print(f"ok   {name}: guard-tree {guard} -> shards {results[SH]} -> `{FAN}` {'green' if rc == 0 else 'red'}")

row("guard-red", "failure", "success", False, False, "guard-tree was 'failure'")
row("guard-cancelled", "cancelled", "success", False, False, "guard-tree was 'cancelled'")
row("guard-green", "success", "success", True, True)
row("shard-red", "success", "failure", True, False)
name = jobs[FAN].get("name", FAN)
print(f"ok   required-name: the fan-in reports as `{name}`" if name == FAN
      else f"FAIL required-name: the fan-in reports as `{name}`, branch protection requires `{FAN}`")
PY
}

judge() { # <ci.yml> -> prints rows, returns 0 when every row is ok
  local out rc
  out=$(run_rows "$1"); rc=$?
  [ "$rc" = 2 ] && return 2
  printf '%s\n' "$out" | sed 's/^/  /'
  grep -q '^ok   required-name' <<< "$out" || return 1
  ! grep -q '^FAIL' <<< "$out"
}

if [ "$SELF_TEST" = 1 ]; then
  bad=0
  mutant() { # <label> <python-edit> <row...>
    local label="$1" edit="$2" m="$T/m-$1.yml" o c; shift 2
    python3 - "$CI" "$m" "$edit" <<'PY' || { echo "  FAIL  mutant $label did not apply"; bad=1; return; }
import sys, yaml
src, dst, edit = sys.argv[1], sys.argv[2], sys.argv[3]
w = yaml.safe_load(open(src)); j = w["jobs"]
before = yaml.safe_dump(w)
exec(edit)
if yaml.safe_dump(w) == before: sys.exit(1)
yaml.safe_dump(w, open(dst, "w"))
PY
    o=$(run_rows "$m" 2>&1) || true
    for c in "$@"; do
      grep -q "^FAIL $c:" <<< "$o" && printf '  ok    mutant %-18s killed by %s\n' "$label" "$c" \
        || { echo "  FAIL  mutant $label SURVIVED row $c"; bad=1; }
    done
  }
  mutant shards-no-needs "j['workspace-test-shard'].pop('needs', None)" guard-red guard-cancelled
  mutant shards-always "j['workspace-test-shard']['if'] = 'always()'" guard-red guard-cancelled
  mutant fan-in-no-always "j['workspace-test'].pop('if', None)" guard-red guard-cancelled shard-red
  mutant skipped-arm-green "st = j['workspace-test']['steps'][0]; st['run'] = st['run'].replace('fix the guard first)\"; exit 1', 'fix the guard first)\"; exit 0')" guard-red guard-cancelled
  mutant fan-in-renamed "j['workspace-test']['name'] = 'workspace-test (fan-in)'" required-name
  [ "$bad" = 0 ] && { echo "SELF-TEST OK"; exit 0; }
  echo "SELF-TEST FAIL"; exit 1
fi
echo "workspace-test waits on guard-tree, and still reports when it skips ($CI)"
judge "$CI"; rc=$?
[ "$rc" = 2 ] && { echo "  cannot check: $CI did not parse" >&2; exit 2; }
[ "$rc" = 0 ] && { echo "PASS"; exit 0; }
echo "FAIL"; exit 1
