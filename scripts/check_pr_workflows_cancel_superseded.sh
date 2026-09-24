#!/usr/bin/env bash
# check_pr_workflows_cancel_superseded.sh -- every pull_request workflow cancels
# the run a newer push to the same PR has superseded (#3676).
#
# WHY THIS EXISTS
#   book-contracts.yml had no concurrency group at all, so every push to a PR ran
#   the ~18 min chapter suite to completion even after the next push had started
#   its own run. A 24 h sample to 2026-09-24 19:15Z put ~1650 of its ~3700
#   pull_request wall-minutes after a newer push had started. book.yml was the
#   opposite mistake: ONE global group "pages" (cancel false) held every PR's
#   mdBook check in a single repo-wide queue. ci.yml has always done it right.
#
# THE RULE, for every workflow that triggers on pull_request:
#   a workflow-level `concurrency` whose `group` is keyed to the PR
#   (github.event.pull_request.number) and whose `cancel-in-progress` is true
#   or an expression that is true on pull_request.
#   EXEMPT, by structure and never by name: a workflow with a job-level
#   `perf-<host>` group. PP-19 (check_perf_concurrency_groups.sh) forbids
#   cancelling a measurement mid-window, and the two rules must not fight.
#
#   bash scripts/check_pr_workflows_cancel_superseded.sh             # gate
#   bash scripts/check_pr_workflows_cancel_superseded.sh --dir DIR   # gate a fixture dir
#   bash scripts/check_pr_workflows_cancel_superseded.sh --self-test # case table
# EXIT: 0 every PR workflow holds; 1 a violation; 2 the workflows cannot be read.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2

scan() { # scan DIR -> rc 0/1/2
    python3 - "$1" <<'PY'
import glob, sys
try:
    import yaml
except ImportError:
    print("ENV   PyYAML is not importable; the workflows could not be parsed"); sys.exit(2)
files = sorted(glob.glob(sys.argv[1] + "/*.yml") + glob.glob(sys.argv[1] + "/*.yaml"))
if not files:
    print("ENV   no workflow files in %s -- cannot judge, not a pass" % sys.argv[1]); sys.exit(2)
bad = n = 0
for f in files:
    name = f.rsplit("/", 1)[-1]
    try:
        d = yaml.safe_load(open(f, encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print("FAIL  %s does not parse: %s" % (name, str(e).splitlines()[0])); bad = 1; continue
    on = d.get(True, d.get("on")) or {}   # PyYAML reads the bare key `on` as True
    if isinstance(on, str): on = {on: None}
    if isinstance(on, list): on = {k: None for k in on}
    if "pull_request" not in on: continue
    n += 1
    jobs = d.get("jobs") or {}
    perf = [j for j, v in jobs.items() if isinstance(v, dict) and isinstance(v.get("concurrency"), dict)
            and str(v["concurrency"].get("group", "")).startswith("perf-")]
    if perf:
        print("ok    %-34s exempt: job %s holds a perf-<host> group (PP-19 forbids cancelling it)" % (name, perf[0])); continue
    c = d.get("concurrency")
    if not isinstance(c, dict):
        print("FAIL  %-34s triggers on pull_request with no workflow-level concurrency: a superseded push runs to completion" % name); bad = 1; continue
    g, cip = str(c.get("group", "")), c.get("cancel-in-progress", False)
    if "github.event.pull_request.number" not in g:
        print("FAIL  %-34s group %r is not keyed to the PR: every PR shares one queue" % (name, g)); bad = 1; continue
    if not (cip is True or (isinstance(cip, str) and "pull_request" in cip)):
        print("FAIL  %-34s cancel-in-progress is %r: a superseded PR run is never cancelled" % (name, cip)); bad = 1; continue
    print("ok    %-34s group %s, cancel-in-progress %s" % (name, g, cip))
if n == 0:
    print("ENV   no workflow triggers on pull_request -- cannot judge, not a pass"); sys.exit(2)
sys.exit(bad)
PY
}

case "${1:-}" in -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--dir" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --dir DIR" >&2; exit 1; }
    scan "$2"; exit $?
fi

if [ "${1:-}" = "--self-test" ]; then
    echo "=== PR workflows cancel superseded runs: case table ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/pr-cancel.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    hdr='on:\n  pull_request:\n    branches: [main]\n'
    job='jobs:\n  a:\n    runs-on: [self-hosted]\n    steps:\n      - run: "true"\n'
    good='concurrency:\n  group: x-${{ github.event.pull_request.number || github.ref }}\n  cancel-in-progress: ${{ github.event_name == '"'"'pull_request'"'"' }}\n'
    mk() { mkdir -p "$d/$1"; printf "$2" > "$d/$1/w.yml"; }
    mk good         "$hdr$good$job"
    mk literal_true "${hdr}concurrency:\n  group: x-\${{ github.event.pull_request.number }}\n  cancel-in-progress: true\n$job"
    mk none         "$hdr$job"
    mk global_group "${hdr}concurrency:\n  group: \"pages\"\n  cancel-in-progress: false\n$job"
    mk cancel_false "${hdr}concurrency:\n  group: x-\${{ github.event.pull_request.number || github.ref }}\n  cancel-in-progress: false\n$job"
    mk perf_exempt  "${hdr}jobs:\n  a:\n    runs-on: [self-hosted]\n    concurrency:\n      group: perf-gx10\n      cancel-in-progress: false\n    steps:\n      - run: \"true\"\n"
    mk push_only    "on:\n  push:\n    branches: [main]\n$job"
    mkdir -p "$d/empty"
    bad=0
    while read -r case want; do
        [ -n "$case" ] || continue
        rc=0; scan "$d/$case" > "$d/out" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    %-13s -> rc=%s\n' "$case" "$rc"
        else printf 'FAIL  %-13s wanted rc=%s, got rc=%s: %s\n' "$case" "$want" "$rc" "$(tail -1 "$d/out")"; bad=1; fi
    done <<'ROWS'
good          0
literal_true  0
none          1
global_group  1
cancel_false  1
perf_exempt   0
push_only     2
empty         2
ROWS
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== every pull_request workflow cancels a superseded run (check_pr_workflows_cancel_superseded.sh, #3676) ==="
scan "$REPO_ROOT/.github/workflows"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
