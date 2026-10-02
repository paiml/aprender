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
#   (a `${{ }}` expression reading github.event.pull_request.number, never a
#   literal) and whose `cancel-in-progress` is `true` or exactly
#   `${{ github.event_name == 'pull_request' }}` -- no looser match, since
#   `!= 'pull_request'` would never cancel on a PR and still contains the word.
#   pull_request_target counts as pull_request. A trigger whose `types:` omit
#   `synchronize` is skipped by structure: no push can supersede its run (the
#   default types include synchronize, so an untyped trigger is checked).
#   ALLOW below is the only exemption, each entry with its reason. An entry
#   whose workflow no longer needs it (or no longer exists) is RED, so the
#   list can only shrink. A job-level perf-<host> group (PP-19) does NOT
#   exempt a workflow: ci.yml holds both, and they are independent.
#
#   bash scripts/check_pr_workflows_cancel_superseded.sh             # gate
#   bash scripts/check_pr_workflows_cancel_superseded.sh --dir DIR   # gate a fixture dir
#   bash scripts/check_pr_workflows_cancel_superseded.sh --self-test # case table
# EXIT: 0 every PR workflow holds; 1 a violation; 2 the workflows cannot be read.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2

scan() { # scan DIR -> rc 0/1/2
    python3 - "$1" "${2:-fixture}" <<'PY'
import glob, re, sys
try:
    import yaml
except ImportError:
    print("ENV   PyYAML is not importable; the workflows could not be parsed"); sys.exit(2)
files = sorted(glob.glob(sys.argv[1] + "/*.yml") + glob.glob(sys.argv[1] + "/*.yaml"))
if not files:
    print("ENV   no workflow files in %s -- cannot judge, not a pass" % sys.argv[1]); sys.exit(2)
# name -> reason. Shrink-only: a stale entry is RED.
ALLOW = {
    "pr-review-quorum.yml": "base-owned receipt gate held to its exact shape by "
        "check_receipt_gate_base_owned.sh B1..B4; it runs in seconds, so it is "
        "not a CI-cost surface worth a change to a merge-gate workflow",
}
GROUP = re.compile(r"\$\{\{[^}]*github\.event\.pull_request\.number[^}]*\}\}")
CANCEL = "${{ github.event_name == 'pull_request' }}"
bad = n = 0
seen = set()
for f in files:
    name = f.rsplit("/", 1)[-1]
    try:
        d = yaml.safe_load(open(f, encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print("FAIL  %s does not parse: %s" % (name, str(e).splitlines()[0])); bad = 1; continue
    on = d.get(True, d.get("on")) or {}   # PyYAML reads the bare key `on` as True
    if isinstance(on, str): on = {on: None}
    if isinstance(on, list): on = {k: None for k in on}
    trig = [on[k] for k in ("pull_request", "pull_request_target") if k in on]
    if not trig: continue
    if all(isinstance(t, dict) and t.get("types") and "synchronize" not in t["types"] for t in trig):
        print("ok    %-34s skipped: its types omit synchronize, so no push supersedes a run" % name); continue
    n += 1
    c = d.get("concurrency")
    g = str(c.get("group", "")) if isinstance(c, dict) else ""
    cip = c.get("cancel-in-progress", False) if isinstance(c, dict) else False
    if not isinstance(c, dict):
        why = "has no workflow-level concurrency: a superseded push runs to completion"
    elif not GROUP.search(g):
        why = "group %r is not a ${{ }} expression keyed to the PR number: PRs share one queue" % g
    elif not (cip is True or (isinstance(cip, str) and " ".join(cip.split()) == CANCEL)):
        why = "cancel-in-progress is %r, not true or %s: a superseded PR run is not cancelled" % (cip, CANCEL)
    else:
        why = None
    if name in ALLOW:
        seen.add(name)
        if why is None:
            print("FAIL  %-34s is in ALLOW but now passes: remove the stale entry" % name); bad = 1
        else:
            print("ok    %-34s allowed: %s" % (name, ALLOW[name]))
        continue
    if why:
        print("FAIL  %-34s %s" % (name, why)); bad = 1
    else:
        print("ok    %-34s group %s, cancel-in-progress %s" % (name, g, cip))
for name in sorted(set(ALLOW) - seen):
    if any(f.endswith("/" + name) for f in files) or sys.argv[2] == "live":
        print("FAIL  %-34s is in ALLOW but is not a pull_request workflow here: remove the stale entry" % name); bad = 1
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
    mk inverted     "${hdr}concurrency:\n  group: x-\${{ github.event.pull_request.number }}\n  cancel-in-progress: \${{ github.event_name != '"'"'pull_request'"'"' }}\n$job"
    mk literal_grp  "${hdr}concurrency:\n  group: x-github.event.pull_request.number\n  cancel-in-progress: true\n$job"
    mk pr_target    "on:\n  pull_request_target:\n    branches: [main]\n$job"
    mk open_only    "on:\n  pull_request_target:\n    types: [opened, reopened]\n$job"
    mk perf_nogrp   "${hdr}jobs:\n  a:\n    runs-on: [self-hosted]\n    concurrency:\n      group: perf-gx10\n      cancel-in-progress: false\n    steps:\n      - run: \"true\"\n"
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
inverted      1
literal_grp   1
pr_target     1
perf_nogrp    1
open_only     2
push_only     2
empty         2
ROWS
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== every pull_request workflow cancels a superseded run (check_pr_workflows_cancel_superseded.sh, #3676) ==="
scan "$REPO_ROOT/.github/workflows" live; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
