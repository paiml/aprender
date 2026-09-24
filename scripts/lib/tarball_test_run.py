#!/usr/bin/env python3
"""tarball_test_run.py BUILD_JSON RUN_DIR: run every test binary the tarball build compiled, as `cargo test` would (#4175).

BUILD_JSON is the `cargo build --workspace --tests --message-format json` stream of the tarball
workspace. Every compiler-artifact with profile.test and an executable is one test binary. It runs in
its crate's unpacked directory with the environment `cargo test` gives it, because a bare exe is not
a `cargo test` and each gap below measured a false RED in the 0.69.1 run:
  B1  CARGO_BIN_EXE_<bin>   the crate's own bin targets, from the same stream (profile.test false)
  B2  LD_LIBRARY_PATH       `rustc --print sysroot`/lib and <target>/debug/deps (proc-macro and dylib exes)
  B3  RUSTUP_TOOLCHAIN      the workspace's toolchain. env!("CARGO") is the toolchain's OWN cargo, not
                            the rustup proxy, so it does not set it; a test that runs cargo then gets
                            the `rustc` proxy resolving in a registry crate's dir (no
                            rust-toolchain.toml) to the host default, and every workspace rlib is E0514.
      CARGO_TARGET_DIR      the tarball build's target, so such a test reuses it instead of a cold one
  plus CARGO, CARGO_MANIFEST_DIR, CARGO_PKG_NAME, CARGO_PKG_VERSION and cwd = the manifest dir.
  B4  bounded memory        --jobs binaries at a time, each with --test-threads (the 0.69.1 run at
                            4 x 8 peaked at 60.8 GiB); the caller may also cap the whole run's RSS.

Options: --jobs N (2) --test-threads N (4) --timeout S (1800, per binary) --skip-crate NAME (repeatable:
compile-only crates, printed) --cargo PATH --toolchain NAME --sysroot DIR --target-dir DIR.
Prints one RUN line per failing binary naming its failing tests, and a total.
Exit: 0 every binary passed · 1 a binary failed (named) · 2 could not check (no binary: vacuous; bad input).
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import resource
import signal
import subprocess
import sys
import time
import tomllib


def manifest_pkg(manifest_path, cache):
    if manifest_path not in cache:
        pkg = tomllib.loads(pathlib.Path(manifest_path).read_text())["package"]
        cache[manifest_path] = (pkg["name"], pkg["version"])
    return cache[manifest_path]


def collect(stream):
    """-> (tests, bins): tests = [(pkg, version, kind, target, exe, mdir)], bins = {pkg: {bin: exe}}"""
    tests, bins, seen, cache = [], {}, set(), {}
    for line in stream:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") != "compiler-artifact" or not m.get("executable"):
            continue
        name, version = manifest_pkg(m["manifest_path"], cache)
        t, exe = m["target"], m["executable"]
        if m["profile"].get("test"):
            if exe not in seen:
                seen.add(exe)
                tests.append((name, version, t["kind"][0], t["name"], exe, str(pathlib.Path(m["manifest_path"]).parent)))
        elif "bin" in t["kind"]:
            bins.setdefault(name, {})[t["name"]] = exe
    return tests, bins


def test_env(base, a, pkg, version, mdir, bins):
    env = dict(base)
    env.update(CARGO_MANIFEST_DIR=mdir, CARGO_PKG_NAME=pkg, CARGO_PKG_VERSION=version,
               CARGO=a.cargo, RUSTUP_TOOLCHAIN=a.toolchain, CARGO_TARGET_DIR=a.target_dir)
    for b, exe in bins.get(pkg, {}).items():
        env["CARGO_BIN_EXE_" + b] = exe
    libs = [os.path.join(a.sysroot, "lib"), os.path.join(a.target_dir, "debug", "deps")]
    if base.get("LD_LIBRARY_PATH"):
        libs.append(base["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env


FAILED_TEST = re.compile(r"^---- (\S+) stdout ----$")


def run_one(row, a, bins, run_dir):
    pkg, version, kind, target, exe, mdir = row
    log = pathlib.Path(run_dir) / ("%s__%s__%s.log" % (pkg, kind, target))
    t0 = time.monotonic()
    with open(log, "wb") as out:
        p = subprocess.Popen([exe, "--test-threads=%d" % a.test_threads], cwd=mdir, stdout=out, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=test_env(os.environ, a, pkg, version, mdir, bins),
                             start_new_session=True)
        try:
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)  # its OWN process group, by the pid we started: never a pattern
            p.wait()
            rc = "timeout"
    failed = [m.group(1) for m in map(FAILED_TEST.match, log.read_text(errors="replace").splitlines()) if m]
    return row, rc, time.monotonic() - t0, failed, log


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("build_json")
    ap.add_argument("run_dir")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--test-threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--skip-crate", action="append", default=[])
    ap.add_argument("--cargo", required=True)
    ap.add_argument("--toolchain", required=True)
    ap.add_argument("--sysroot", required=True)
    ap.add_argument("--target-dir", required=True)
    a = ap.parse_args(argv[1:])
    if a.jobs < 1 or a.test_threads < 1 or a.timeout < 1:
        print("  cannot check: --jobs, --test-threads and --timeout must be >= 1", file=sys.stderr)
        return 2
    try:
        with open(a.build_json) as f:
            tests, bins = collect(f)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        print("  cannot check: unreadable build stream %s: %s" % (a.build_json, e), file=sys.stderr)
        return 2
    skipped = sorted({r[0] for r in tests if r[0] in a.skip_crate})
    run = [r for r in tests if r[0] not in a.skip_crate]
    if not run:
        print("  cannot check: the build stream names no test binary to run (vacuous)", file=sys.stderr)
        return 2
    for s in skipped:
        n = sum(1 for r in tests if r[0] == s)
        print("COMPILE-ONLY %s: %d test binary(ies) compiled, not run (needs a GPU toolchain)" % (s, n))
    os.makedirs(a.run_dir, exist_ok=True)
    print("running %d test binary(ies), %d at a time, --test-threads=%d, timeout %ds each (RUSTUP_TOOLCHAIN=%s)"
          % (len(run), a.jobs, a.test_threads, a.timeout, a.toolchain))
    fails = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for row, rc, dt, failed, log in ex.map(lambda r: run_one(r, a, bins, a.run_dir), run):
            if rc != 0:
                fails.append((row, rc, dt, failed, log))
    peak_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576
    for (pkg, version, kind, target, _exe, _m), rc, dt, failed, log in fails:
        names = ", ".join(failed[:8]) + (" (+%d more)" % (len(failed) - 8) if len(failed) > 8 else "")
        print("RUN FAIL  %s-%s %s %s rc=%s %.0fs: %s  [log %s]"
              % (pkg, version, kind, target, rc, dt, names or "no failing test named (see log)", log))
    print("RUN: %d of %d test binary(ies) passed; peak single-binary RSS %.1f GiB"
          % (len(run) - len(fails), len(run), peak_gib))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
