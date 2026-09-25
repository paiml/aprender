#!/usr/bin/env python3
"""ci_guard_steps.py -- the guard steps of a ci.yml job, read from ci.yml itself.

One reader serves two consumers, so neither keeps its own list of guards:

  check-run-all  (#4415, lever 0) -- count the guard steps that are FAIL-FAST,
                 i.e. would be skipped when an earlier guard step fails. GitHub
                 Actions skips every later step of a job once one step fails
                 unless that step's `if:` names a status function
                 (`!cancelled()`, `always()`, `failure()`). On RC PR #4318
                 every red guard job reported exactly ONE failure and skipped
                 15-71 of guard-cargo's 77 steps (median 55).
  run            (#4416, lever 2) -- execute the SAME `run:` blocks locally,
                 all of them, and print one row per step. The command list is
                 ci.yml; there is no second copy to drift (F1, #2640).
  list           -- print the guard steps (index, fail-fast?, name).
  check-coverage (#4416) -- every guard script (scripts/check_*.sh|py) that a
                 NON-guard job runs must be acknowledged, with a reason, in
                 scripts/ci_guards_local_uncovered.txt: `run` cannot reach it,
                 so a new one is drift until someone says why. A stale row
                 (the job no longer runs it) fails too, so the list only shrinks.

THE GUARD JOBS are read from ci.yml too: every `guard-*` job that the `gate`
job needs. A new guard job runs locally with no edit here.

THE SETUP BOUNDARY. A job's leading steps (ownership restore, checkout, the
origin/main fetch, target-dir creation) are setup, not guards: when setup
fails, running the guards anyway only paints dozens of red rows that say
nothing. The boundary is the step whose `id:` is `guard-setup`. Until
ci.yml carries that id, the boundary falls back to the LAST leading step
that is a `uses:` step, a `git fetch`, or creates the job's target dir, and
the output says which rule it used.

Exit codes: 0 ok, 1 finding (fail-fast over baseline / a step FAILED),
2 usage or unreadable input -- fail closed, never a vacuous 0.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

try:
    import yaml
except ImportError:  # fail closed: a missing parser is not a pass
    print("ci_guard_steps: python3 yaml module missing", file=sys.stderr)
    sys.exit(2)

STATUS_FN = re.compile(r"\b(cancelled|always|failure)\s*\(\s*\)")
SETUP_RUN = re.compile(r"git fetch|target dir|GUARD_TARGET_DIR.*mkdir|mkdir.*GUARD_TARGET_DIR")
EXPR = re.compile(r"\$\{\{.*?\}\}")
GUARD_SCRIPT = re.compile(r"scripts/(check_[\w.-]+\.(?:sh|py))")


def load_wf(workflow):
    try:
        with open(workflow, encoding="utf-8") as fh:
            wf = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        print(f"ci_guard_steps: cannot read {workflow}: {exc}", file=sys.stderr)
        sys.exit(2)
    return (wf or {}).get("jobs") or {}


def guard_jobs(workflow):
    """The `guard-*` jobs the gate needs, in ci.yml order. Never empty."""
    jobs = load_wf(workflow)
    needs = (jobs.get("gate") or {}).get("needs") or []
    needs = [needs] if isinstance(needs, str) else needs
    found = [j for j in jobs if j.startswith("guard-") and j in needs]
    if not found:
        print(f"ci_guard_steps: no guard-* job in the gate's needs in {workflow}",
              file=sys.stderr)
        sys.exit(2)
    return found


def load_job(workflow, job):
    jobs = load_wf(workflow)
    if job not in jobs:
        print(f"ci_guard_steps: job '{job}' not in {workflow}", file=sys.stderr)
        sys.exit(2)
    steps = jobs[job].get("steps") or []
    if not steps:
        print(f"ci_guard_steps: job '{job}' has no steps", file=sys.stderr)
        sys.exit(2)
    return jobs[job], steps


def setup_end(steps):
    """Index of the last setup step, and which rule found it."""
    for i, s in enumerate(steps):
        if s.get("id") == "guard-setup":
            return i, "id: guard-setup"
    end = -1
    for i, s in enumerate(steps[:8]):
        text = (s.get("name") or "") + "\n" + (s.get("run") or "")
        if "uses" in s or SETUP_RUN.search(text):
            end = i
    return end, "fallback: last leading uses/fetch/target-dir step"


def fail_fast(step):
    cond = step.get("if")
    return cond is None or not STATUS_FN.search(str(cond))


def guard_steps(steps):
    end, _ = setup_end(steps)
    return [(i, s) for i, s in enumerate(steps) if i > end]


def cmd_list(args):
    for job in args.jobs:
        _, steps = load_job(args.workflow, job)
        end, rule = setup_end(steps)
        print(f"== {job}: setup ends at step {end} ({rule})")
        for i, s in guard_steps(steps):
            tag = "FAIL-FAST" if fail_fast(s) else "runs-on-red"
            print(f"{i:3d}  {tag:11s}  {s.get('name') or s.get('uses') or '(unnamed)'}")
    return 0


def read_baseline(path):
    base = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    job, n = line.split()
                    base[job] = int(n)
    except (OSError, ValueError) as exc:
        print(f"ci_guard_steps: bad baseline {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    return base


def cmd_check(args):
    base = read_baseline(args.baseline)
    rc = 0
    for job in args.jobs:
        _, steps = load_job(args.workflow, job)
        n = sum(1 for _, s in guard_steps(steps) if fail_fast(s))
        if job not in base:
            print(f"FAIL {job}: no baseline row in {args.baseline}")
            rc = 1
            continue
        if n > base[job]:
            print(f"FAIL {job}: {n} fail-fast guard steps > baseline {base[job]} "
                  f"-- give new guard steps `if: ${{{{ !cancelled() && "
                  f"steps.guard-setup.outcome == 'success' }}}}`")
            rc = 1
        elif n < base[job]:
            print(f"ok   {job}: {n} fail-fast guard steps < baseline {base[job]} "
                  f"-- tighten {args.baseline} to {n}")
        else:
            print(f"ok   {job}: {n} fail-fast guard steps (baseline {base[job]})")
    return rc


def read_ack(path):
    ack = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                body, _, reason = line.partition("#")
                if not body.strip():
                    continue
                parts = body.split()
                if len(parts) != 2 or not reason.strip():
                    print(f"ci_guard_steps: {path}:{n}: want `<job> <script>  # <reason>`",
                          file=sys.stderr)
                    sys.exit(2)
                ack[tuple(parts)] = reason.strip()
    except OSError as exc:
        print(f"ci_guard_steps: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    return ack


def cmd_coverage(args):
    jobs = load_wf(args.workflow)
    local = set(guard_jobs(args.workflow))
    seen = set()
    for job, d in jobs.items():
        if job in local:
            continue
        for s in d.get("steps") or []:
            for script in GUARD_SCRIPT.findall(str(s.get("run") or "")):
                seen.add((job, script))
    ack = read_ack(args.ack)
    rc = 0
    for job, script in sorted(seen - set(ack)):
        print(f"FAIL {job} runs scripts/{script}, which the local runner cannot reach "
              f"-- move it into a guard job, or acknowledge it in {args.ack} with a reason")
        rc = 1
    for job, script in sorted(set(ack) - seen):
        print(f"FAIL stale row in {args.ack}: {job} no longer runs scripts/{script} -- delete it")
        rc = 1
    if rc == 0:
        print(f"ok   {len(local)} guard jobs run locally ({', '.join(sorted(local))}); "
              f"{len(seen)} guard scripts in other jobs, all acknowledged")
    return rc


def local_env(job_env, root):
    """The job's env with ${{ }} expressions resolved to local stand-ins."""
    env = dict(os.environ)
    env.setdefault("GITHUB_WORKSPACE", root)
    env.setdefault("PR_OR_REF", "local")
    scratch = os.environ.get("CI_GUARDS_SCRATCH") or os.path.join(root, "target", "ci-guards-local")
    os.makedirs(scratch, exist_ok=True)
    env.setdefault("GITHUB_PATH", os.path.join(scratch, "github_path"))
    env.setdefault("GITHUB_OUTPUT", os.path.join(scratch, "github_output"))
    env.setdefault("GITHUB_STEP_SUMMARY", os.path.join(scratch, "step_summary"))
    env.setdefault("GITHUB_EVENT_NAME", "local")
    for k, v in (job_env or {}).items():
        if k in os.environ:
            continue
        v = str(v)
        env[k] = EXPR.sub("local", v) if EXPR.search(v) else v
    for k in ("GUARD_TARGET_DIR", "GUARD_CARGO_HOME"):
        if k in env and k not in os.environ:
            env[k] = os.path.join(scratch, k.lower())
    return env


def skip_reason(step, env):
    cond = str(step.get("if") or "")
    if "event_name" in cond:
        return "if: needs a CI event"
    if "uses" in step:
        return "uses: action"
    run = step.get("run") or ""
    if EXPR.search(run):
        return "run: has a ${{ }} expression"
    if "docker run" in run or "$IMAGE" in run or "${IMAGE}" in run:
        image = env.get("IMAGE", "")
        probe = subprocess.run(["docker", "image", "inspect", image],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False) if image else None
        if probe is None or probe.returncode != 0:
            return f"needs docker image {image or '$IMAGE'}"
    return None


def cmd_run(args):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                          text=True, check=False).stdout.strip()
    if not root:
        print("ci_guard_steps: not in a git work tree", file=sys.stderr)
        return 2
    only = re.compile(args.only) if args.only else None
    rows, failed = [], 0

    def emit(row):
        # One row per step AS IT FINISHES: a full guard-tree run is minutes long,
        # and a table printed only at the end showed nothing when it was killed.
        rows.append(row)
        job, i, st, sec, name = row
        print(f"{job:12s} {i:4d} {st:7s} {sec:6.1f}  {name}", flush=True)

    print(f"{'job':12s} {'step':>4s} {'result':7s} {'sec':>6s}  name", flush=True)
    for job in args.jobs:
        job_def, steps = load_job(args.workflow, job)
        env = local_env(job_def.get("env"), root)
        for i, s in guard_steps(steps):
            name = s.get("name") or s.get("uses") or "(unnamed)"
            if only and not only.search(name):
                continue
            reason = skip_reason(s, env)
            if reason:
                emit((job, i, "SKIP", 0.0, f"{name}  [{reason}]"))
                continue
            senv = dict(env)
            for k, v in (s.get("env") or {}).items():
                if k not in os.environ and not EXPR.search(str(v)):
                    senv[k] = str(v)
            t0 = time.monotonic()
            log = os.path.join(senv["GITHUB_PATH"] + f".{job}.{i}.log")
            with open(log, "w", encoding="utf-8") as out:
                # Own process group, so a timeout kills the step's children too
                # (cargo, sleep): killing bash alone orphans them.
                proc = subprocess.Popen(
                    ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", s["run"]],
                    cwd=root, env=senv, stdout=out, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    rc = proc.wait(timeout=args.step_timeout or None)
                    status = "PASS" if rc == 0 else "FAIL"
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    rc, status = 124, "TIMEOUT"
            failed += rc != 0
            emit((job, i, status, time.monotonic() - t0,
                  name if rc == 0 else f"{name}  [rc={rc}, log {log}]"))
    ran = sum(1 for r in rows if r[2] != "SKIP")
    skipped = len(rows) - ran
    print(f"SUMMARY: {failed} failed / {ran} ran / {skipped} skipped")
    if ran == 0:
        print("ci_guard_steps: nothing ran -- refusing a vacuous pass", file=sys.stderr)
        return 2
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--workflow", default=".github/workflows/ci.yml")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "check-run-all", "run", "check-coverage"):
        sp = sub.add_parser(name)
        if name != "check-coverage":
            sp.add_argument("jobs", nargs="*")
        if name == "check-run-all":
            sp.add_argument("--baseline", default="scripts/guard_fail_fast_baseline.txt")
        if name == "run":
            sp.add_argument("--only", help="regex on step name")
            sp.add_argument("--step-timeout", type=float, default=900.0,
                            help="seconds per step; 0 = none (a timeout is a failure)")
        if name == "check-coverage":
            sp.add_argument("--ack", default="scripts/ci_guards_local_uncovered.txt")
    args = p.parse_args(argv)
    if getattr(args, "jobs", None) == []:
        args.jobs = guard_jobs(args.workflow)
    return {"list": cmd_list, "check-run-all": cmd_check, "run": cmd_run,
            "check-coverage": cmd_coverage}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
