#!/usr/bin/env python3
"""ci_guard_steps.py -- run CI's guard steps, all of them, in CI and locally.

ONE runner, called through scripts/ci_guards.sh, serves both places (#4415,
operator amendment A4, 2026-09-25): CI's guard jobs run it as a single step,
and `make guards-local` / the pre-push hook run the same file. Both print the
same `ci_guards: sha256 ...` line, so "the same script ran" can be checked.

WHERE THE STEPS LIVE. The steps of guard job J are the steps of the job
`J-steps` in ci.yml, a MANIFEST job whose `if: false` means GitHub never runs
it. They stay in ci.yml, verbatim, so every guard that reads ci.yml's text
(check_guards_are_wired.sh, guard_tree.sh, check_model_tests_wired.sh, the
contracts' `test:` greps) still finds each invocation. A manifest step is
plain: `name`, `run`, and optionally `env`. The runner cannot honour `if:`,
`uses:`, `id:`, `shell:`, `timeout-minutes`, `continue-on-error` or
`working-directory`, so check-run-all refuses them.

WHY ONE STEP. GitHub Actions skips every later step of a job after one step
fails. On RC PR #4318 every red guard job reported exactly ONE failure, and
guard-cargo skipped 15-71 of its 77 steps (median 55). The runner never stops:
every step runs, each failure gets an `::error` annotation, and one table
(stdout plus $GITHUB_STEP_SUMMARY) lists them all.

Subcommands
  run            execute the steps. Rows stream as each step finishes.
  check-run-all  the wiring guard (scripts/check_guard_steps_run_all.sh).
  check-coverage guard scripts that ci.yml runs OUTSIDE the guard jobs must
                 be acknowledged in scripts/ci_guards_uncovered.txt (#4416).
  list           the steps each job runs.
  sha            the sha256 line.

Exit codes: 0 ok, 1 finding (a step FAILED or TIMED OUT, wiring broken),
2 usage or unreadable input -- fail closed, never a vacuous 0.
"""

import argparse
import hashlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

try:
    import yaml
except ImportError:  # fail closed: a missing parser is not a pass
    print("ci_guard_steps: python3 yaml module missing", file=sys.stderr)
    sys.exit(2)

STATUS_FN = re.compile(r"\b(cancelled|always|failure)\s*\(\s*\)")
EXPR = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
GUARD_SCRIPT = re.compile(r"scripts/((?:check|guard)_[A-Za-z0-9_]+\.sh)")
MANIFEST_SUFFIX = "-steps"
PLAIN_KEYS = {"name", "run", "env"}
RUNNER = "scripts/ci_guards.sh"
# The only ${{ }} a manifest step may use in `env:`, and where each is found at
# run time. The runner step in ci.yml exports GITHUB_TOKEN and GH_TOKEN.
KNOWN_EXPR = {
    "github.token": ("GITHUB_TOKEN", "GH_TOKEN"),
    "runner.temp": ("RUNNER_TEMP",),
}
HERE = os.path.dirname(os.path.abspath(__file__))
SELF_FILES = (os.path.join(HERE, "..", "ci_guards.sh"), os.path.abspath(__file__))


def in_ci():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def load_jobs(workflow):
    try:
        with open(workflow, encoding="utf-8") as fh:
            wf = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        print(f"ci_guard_steps: cannot read {workflow}: {exc}", file=sys.stderr)
        sys.exit(2)
    return (wf or {}).get("jobs") or {}


def guard_jobs(jobs, workflow):
    """The `guard-*` jobs the gate needs, in ci.yml order. Never empty."""
    needs = (jobs.get("gate") or {}).get("needs") or []
    needs = [needs] if isinstance(needs, str) else needs
    found = [j for j in jobs if j.startswith("guard-") and j in needs]
    if not found:
        print(f"ci_guard_steps: no guard-* job in the gate's needs in {workflow}", file=sys.stderr)
        sys.exit(2)
    return found


def job_steps(jobs, job, workflow):
    if job not in jobs:
        print(f"ci_guard_steps: job '{job}' not in {workflow}", file=sys.stderr)
        sys.exit(2)
    steps = jobs[job].get("steps") or []
    if not steps:
        print(f"ci_guard_steps: job '{job}' has no steps", file=sys.stderr)
        sys.exit(2)
    return steps


def setup_end(steps):
    """Index of the setup step (`id: guard-setup`); -1 when the job has none."""
    for i, s in enumerate(steps):
        if s.get("id") == "guard-setup":
            return i
    return -1


def fail_fast(step):
    cond = step.get("if")
    return cond is None or not STATUS_FN.search(str(cond))


def calls_runner(step, job):
    return re.search(rf"(^|\s){re.escape(RUNNER)}\s+{re.escape(job)}(\s|$)", str(step.get("run") or "")) is not None


def sha_line(jobs, names):
    h = hashlib.sha256()
    for f in SELF_FILES:
        with open(f, "rb") as fh:
            h.update(fh.read())
    m = hashlib.sha256()
    for job in names:
        m.update(yaml.safe_dump(jobs.get(job + MANIFEST_SUFFIX), sort_keys=True).encode())
    return f"ci_guards: sha256 {h.hexdigest()[:16]} manifest {m.hexdigest()[:16]} ({' '.join(names)})"


# ── check-run-all ───────────────────────────────────────────────────────────


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


def manifest_findings(jobs, job):
    """Every way the manifest of `job` could be skipped or mis-run."""
    out = []
    man = jobs.get(job + MANIFEST_SUFFIX)
    if man is None:
        # A runner step with no manifest runs only the steps left behind it --
        # green over nothing it was meant to run (lane b, #4415).
        if any(calls_runner(s, job) for s in jobs[job].get("steps") or []):
            out.append(f"{job}: a step runs `bash {RUNNER} {job}` but there is no {job}{MANIFEST_SUFFIX} job")
        return out
    if man.get("if") is not False:
        out.append(f"{job}{MANIFEST_SUFFIX}: a manifest job needs `if: false`, or GitHub runs its steps a second time")
    for i, s in enumerate(man.get("steps") or []):
        name = s.get("name") or f"step {i}"
        extra = sorted(set(s) - PLAIN_KEYS)
        if extra:
            out.append(f"{job}{MANIFEST_SUFFIX} '{name}': {', '.join(extra)} -- the runner cannot honour it")
        if not s.get("run"):
            out.append(f"{job}{MANIFEST_SUFFIX} '{name}': no run:")
        if EXPR.search(str(s.get("run") or "")):
            out.append(f"{job}{MANIFEST_SUFFIX} '{name}': ${{{{ }}}} in run: -- put it in env:")
        for k, v in (s.get("env") or {}).items():
            for e in EXPR.findall(str(v)):
                if e not in KNOWN_EXPR:
                    out.append(f"{job}{MANIFEST_SUFFIX} '{name}': env {k} uses ${{{{ {e} }}}}, "
                               f"which the runner cannot resolve (known: {', '.join(KNOWN_EXPR)})")
    if not man.get("steps"):
        out.append(f"{job}{MANIFEST_SUFFIX}: no steps")
    if not any(calls_runner(s, job) for s in jobs[job].get("steps") or []):
        out.append(f"{job}: no step runs `bash {RUNNER} {job}`, so {job}{MANIFEST_SUFFIX} never runs")
    return out


def cmd_check(args):
    jobs = load_jobs(args.workflow)
    base = read_baseline(args.baseline)
    names = args.jobs or guard_jobs(jobs, args.workflow)
    rc = 0
    for job in names:
        steps = job_steps(jobs, job, args.workflow)
        end = setup_end(steps)
        if end < 0:
            print(f"FAIL {job}: no step has `id: guard-setup` -- the runner step's if: reads "
                  f"steps.guard-setup.outcome, so without it the guards are skipped and {job} is green")
            rc = 1
        n = sum(1 for s in steps[end + 1:] if fail_fast(s))
        if job not in base:
            print(f"FAIL {job}: no baseline row in {args.baseline}")
            rc = 1
        elif n > base[job]:
            print(f"FAIL {job}: {n} fail-fast steps after the setup step > baseline {base[job]} -- a guard "
                  f"step goes in {job}{MANIFEST_SUFFIX}; a step that must stay in {job} needs "
                  f"`if: ${{{{ !cancelled() && steps.guard-setup.outcome == 'success' }}}}`")
            rc = 1
        else:
            print(f"ok   {job}: {n} fail-fast steps after setup (baseline {base[job]})"
                  + (f" -- tighten {args.baseline} to {n}" if n < base[job] else ""))
        for f in manifest_findings(jobs, job):
            print(f"FAIL {f}")
            rc = 1
    for j, d in jobs.items():  # an orphan manifest runs nowhere
        if j.endswith(MANIFEST_SUFFIX) and d.get("if") is False and j[: -len(MANIFEST_SUFFIX)] not in jobs:
            print(f"FAIL {j}: a manifest with no job {j[: -len(MANIFEST_SUFFIX)]} to run it")
            rc = 1
    return rc


# ── check-coverage (#4416) ─────────────────────────────────────────────────


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
                    print(f"ci_guard_steps: {path}:{n}: want `<job> <script>  # <reason>`", file=sys.stderr)
                    sys.exit(2)
                ack[tuple(parts)] = reason.strip()
    except OSError as exc:
        print(f"ci_guard_steps: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    return ack


def cmd_coverage(args):
    jobs = load_jobs(args.workflow)
    local = set(guard_jobs(jobs, args.workflow))
    local |= {j + MANIFEST_SUFFIX for j in local}
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
        print(f"FAIL {job} runs scripts/{script}, which `make guards-local` cannot reach "
              f"-- move it into a guard manifest, or acknowledge it in {args.ack} with a reason")
        rc = 1
    for job, script in sorted(set(ack) - seen):
        print(f"FAIL stale row in {args.ack}: {job} no longer runs scripts/{script} -- delete it")
        rc = 1
    if rc == 0:
        print(f"ok   guard scripts outside the guard jobs: {len(seen)}, all acknowledged")
    return rc


# ── run ─────────────────────────────────────────────────────────────────────


def base_env(root, scratch):
    """CI: the runner step's own environment, untouched. Locally: stand-ins for
    what the job env and the runner would provide."""
    env = dict(os.environ)
    if in_ci():
        return env
    env.setdefault("GITHUB_WORKSPACE", root)
    env.setdefault("GITHUB_EVENT_NAME", "local")
    env.setdefault("PR_OR_REF", "local")
    env.setdefault("RUNNER_TEMP", os.path.join(scratch, "runner_temp"))
    os.makedirs(env["RUNNER_TEMP"], exist_ok=True)
    return env


def local_job_env(job_env, env, scratch):
    """Locally, the job's own env: block (CI already exported it)."""
    for k, v in (job_env or {}).items():
        if k in os.environ:
            continue
        v = str(v)
        env[k] = os.path.join(scratch, k.lower()) if k in ("GUARD_TARGET_DIR", "GUARD_CARGO_HOME") \
            else EXPR.sub("local", v)


def resolve(value, env):
    """${{ e }} -> its runtime value, or None when it cannot be resolved."""
    missing = []

    def sub(m):
        for name in KNOWN_EXPR.get(m.group(1), ()):
            if env.get(name):
                return env[name]
        missing.append(m.group(1))
        return ""

    out = EXPR.sub(sub, str(value))
    return None if missing else out


def skip_reason(step, env):
    """Why a step cannot run here. In CI nothing is skipped: check-run-all has
    already refused every manifest step the runner cannot honour."""
    if in_ci():
        return None
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
        probe = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False) if image else None
        if probe is None or probe.returncode != 0:
            return f"needs docker image {image or '$IMAGE'}"
    return None


def read_github_env(path):
    """KEY=VALUE and KEY<<DELIM ... DELIM lines, as the runner reads them."""
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    i = 0
    while i < len(lines):
        line = lines[i]
        if "<<" in line and ("=" not in line or line.index("<<") < line.index("=")):
            key, delim = line.split("<<", 1)
            body = []
            i += 1
            while i < len(lines) and lines[i] != delim:
                body.append(lines[i])
                i += 1
            out[key] = "\n".join(body)
        elif "=" in line:
            key, val = line.split("=", 1)
            out[key] = val
        i += 1
    return out


def steps_for(jobs, job, workflow):
    """(index label, step) for every step `run` executes for `job`: its manifest,
    then (locally only) the job's own steps after setup, minus the runner step."""
    out = []
    man = jobs.get(job + MANIFEST_SUFFIX)
    if man is not None:
        out += [(f"m{i}", s) for i, s in enumerate(man.get("steps") or [])]
    if man is None or not in_ci():
        steps = job_steps(jobs, job, workflow)
        out += [(str(i), s) for i, s in enumerate(steps)
                if i > setup_end(steps) and not calls_runner(s, job)
                and "always" not in str(s.get("if") or "")]
    return out


def run_step(cmd, env, cwd, log, timeout, stream):
    """Run one step in its own process group, so a timeout kills its children
    (cargo, docker, sleep) too, not only bash. Returns (rc, status)."""
    out = None if stream else open(log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", cmd],
                                cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout or None)
            return rc, "PASS" if rc == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 124, "TIMEOUT"
    finally:
        if out:
            out.close()


def cmd_run(args):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                          text=True, check=False).stdout.strip()
    if not root:
        print("ci_guard_steps: not in a git work tree", file=sys.stderr)
        return 2
    jobs = load_jobs(args.workflow)
    names = args.jobs or guard_jobs(jobs, args.workflow)
    print(sha_line(jobs, names), flush=True)
    missing = [j for j in names if in_ci() and j + MANIFEST_SUFFIX not in jobs]
    if missing:
        print(f"ci_guard_steps: no manifest job for {', '.join(missing)} -- nothing to run", file=sys.stderr)
        return 2
    scratch = os.environ.get("CI_GUARDS_SCRATCH") or (
        tempfile.mkdtemp(prefix="ci-guards-") if in_ci() else os.path.join(root, "target", "ci-guards-local"))
    os.makedirs(scratch, exist_ok=True)
    only = re.compile(args.only) if args.only else None
    stream = in_ci() if args.stream is None else args.stream
    timeout = args.step_timeout
    rows = []

    def emit(row):
        # One row per step AS IT FINISHES: a table printed only at the end shows
        # nothing when the job is killed.
        rows.append(row)
        job, i, st, sec, name = row
        print(f"{st:7s} {job}#{i} {sec:6.1f}s  {name}", flush=True)
        if st in ("FAIL", "TIMEOUT") and in_ci():
            print(f"::error title={job}: guard step {st.lower()}::{name}", flush=True)

    for job in names:
        env = base_env(root, scratch)
        if not in_ci():
            local_job_env(jobs[job].get("env"), env, scratch)
        for idx, s in steps_for(jobs, job, args.workflow):
            name = s.get("name") or s.get("uses") or "(unnamed)"
            if only and not only.search(name):
                continue
            reason = skip_reason(s, env)
            senv = dict(env)
            for k, v in (s.get("env") or {}).items():
                r = resolve("" if v is None else v, env)
                if r is None and in_ci():
                    reason = reason or f"FAIL: env {k} cannot be resolved"
                elif r is not None and (k not in os.environ or in_ci()):
                    senv[k] = r
            if reason and reason.startswith("FAIL"):
                emit((job, idx, "FAIL", 0.0, f"{name}  [{reason}]"))
                continue
            if reason:
                emit((job, idx, "SKIP", 0.0, f"{name}  [{reason}]"))
                continue
            tag = f"{job}.{idx}"
            for var in ("GITHUB_PATH", "GITHUB_ENV", "GITHUB_OUTPUT"):
                senv[var] = os.path.join(scratch, f"{var.lower()}.{tag}")
                open(senv[var], "w", encoding="utf-8").close()
            senv.setdefault("GITHUB_STEP_SUMMARY", os.path.join(scratch, "step_summary"))
            log = os.path.join(scratch, f"{tag}.log")
            if stream:
                print(f"::group::{job}#{idx} {name}", flush=True)
            t0 = time.monotonic()
            rc, status = run_step(str(s["run"]), senv, root, log, timeout, stream)
            if stream:
                print("::endgroup::", flush=True)
            # what the step handed to later steps, as the Actions runner would apply it
            with open(senv["GITHUB_PATH"], encoding="utf-8") as fh:
                for p in reversed([ln.strip() for ln in fh if ln.strip()]):
                    env["PATH"] = p + os.pathsep + env.get("PATH", "")
            env.update(read_github_env(senv["GITHUB_ENV"]))
            detail = name if rc == 0 else f"{name}  [rc={rc}" + ("]" if stream else f", log {log}]")
            emit((job, idx, status, time.monotonic() - t0, detail))

    failed = [r for r in rows if r[2] in ("FAIL", "TIMEOUT")]
    ran = sum(1 for r in rows if r[2] != "SKIP")
    skipped = len(rows) - ran
    print(f"SUMMARY: {len(failed)} failed / {ran} ran / {skipped} skipped")
    for job, i, st, _, name in failed:
        print(f"  {st:7s} {job}#{i} {name}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if in_ci() and summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Guard steps: {len(failed)} failed / {ran} ran\n\n| job | step | result | sec |\n|---|---|---|---|\n")
            for job, i, st, sec, name in rows:
                fh.write(f"| {job} | {name.split('  [')[0]} | {st} | {sec:.0f} |\n")
    if ran == 0:
        print("ci_guard_steps: nothing ran -- refusing a vacuous pass", file=sys.stderr)
        return 2
    return 1 if failed else 0


def cmd_list(args):
    jobs = load_jobs(args.workflow)
    for job in args.jobs or guard_jobs(jobs, args.workflow):
        print(f"== {job}")
        for idx, s in steps_for(jobs, job, args.workflow):
            print(f"{idx:>4s}  {s.get('name') or s.get('uses') or '(unnamed)'}")
    return 0


def cmd_sha(args):
    jobs = load_jobs(args.workflow)
    print(sha_line(jobs, args.jobs or guard_jobs(jobs, args.workflow)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--workflow", default=".github/workflows/ci.yml")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "check-run-all", "check-coverage", "run", "sha"):
        sp = sub.add_parser(name)
        sp.add_argument("jobs", nargs="*")
        if name == "check-run-all":
            sp.add_argument("--baseline", default="scripts/guard_fail_fast_baseline.txt")
        if name == "check-coverage":
            sp.add_argument("--ack", default="scripts/ci_guards_uncovered.txt")
        if name == "run":
            sp.add_argument("--only", help="regex on step name")
            sp.add_argument("--step-timeout", type=int,
                            default=int(os.environ.get("CI_GUARDS_STEP_TIMEOUT", "1200")),
                            help="seconds per step; a TIMEOUT is red (default 1200)")
            sp.add_argument("--stream", dest="stream", action="store_true", default=None,
                            help="step output to stdout in ::group:: blocks (default in CI)")
            sp.add_argument("--no-stream", dest="stream", action="store_false")
    args = p.parse_args(argv)
    return {"list": cmd_list, "check-run-all": cmd_check, "check-coverage": cmd_coverage,
            "run": cmd_run, "sha": cmd_sha}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
