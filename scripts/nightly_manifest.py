#!/usr/bin/env python3
"""nightly_manifest.py -- the gate, the manifest and the publish plan of the
aprender nightly (#4189). nightly.yml is a thin driver; every decision is made
here, so the case table (--self-test) can pin it.

Operator, verbatim (relayed by the cop, 2026-09-24): "nightly binary built is
hard requirement for soveriegn stack, but intelligent: don't build if no work,
only build working and passing CI and flag broken as tickets: this is arbiter
job" -- and: "we need nightly binaries (or latest release..whatever is newer)
on fleet from nightly build; the end. fix".

  gate     --sha S [--prev F] [--targets T,..] [--checks-json F]
           Prints ONE decision on stdout:
             reused      every target already holds a verdict for S: no work
             build       every required check is `success` on S
             red-ci      a required check completed and is not success on S
             ci-pending  a required check is absent or still running on S
           Exit 0 on a decision, 3 when the checks could not be read.
  record   --target T --sha S --bins a,b --bin-dir D --dist D [--build-outcome X]
           One target's fragment: sha256 of each tarball AND of each
           executable, its --version line, and the SHA it prints. A printed
           SHA that is not S, a failing --version, a missing artifact or a
           failed build makes the target red.
  merge    --prev F --fragments D --sha S --decision X [--run-id N] [--run-url U]
           The next manifest. A target with a green fragment takes it; every
           other target keeps its last green_sha/tools and records why not.
  publish  --manifest F --dist D --sha S --repo R   (token in $GH_TOKEN)
           Moves the `nightly` tag to S, replaces the assets of every target
           whose green_sha is S, and uploads the manifest LAST. It never
           deletes the release, so one red arch never unpublishes the other.
  --self-test

Schema aprender-nightly-manifest/v1, agreed 2026-09-24 with aprender-49 (the
resolvers, #4186) and infra-8d (the fleet installer). Consumers accept a tool
only when sha256(executable) == bin_sha256 and, if version_sha is non-null,
`--version` prints a prefix of it. Tickets for red SHAs are the arbiter's:
this workflow holds `contents: write` only (issues would need `issues: write`).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

SCHEMA = "aprender-nightly-manifest/v1"
MANIFEST_ASSET = "nightly-manifest.json"
STAGED = "staged."  # prefix of an upload not yet swapped in
REQUIRED = ["ci / gate", "workspace-test"]
TARGETS = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "aarch64-apple-darwin"]
# A SHA printed in parentheses -- `apr 0.69.0 (aa7c6ef03)`. `pv 0.69.0
# (aprender provable-contracts verifier)` prints none, and neither does a
# tarball build's `(v0.69.1+no-git)`.
VERSION_SHA = re.compile(r"\(([0-9a-f]{7,40})[\s),]")
# A BUILD verdict on S means S has been handled. red-ci / ci-pending do not:
# CI can be rerun green, and then S is work again.
BUILD_VERDICTS = {"build-failed", "version-mismatch", "version-failed", "missing-artifact"}


class GateError(Exception):
    pass


def api(method, url, token=None, data=None, ctype="application/json", tries=3):
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(url, method=method, data=data)
        req.add_header("Accept", "application/vnd.github+json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if data is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries:
                raise
        except urllib.error.URLError:
            if attempt == tries:
                raise
        time.sleep(5 * attempt)


# ---------------------------------------------------------------- gate

def fetch_check_runs(repo, sha, token):
    # The repo is public. The workflow token carries `contents: write` only --
    # `checks: read` would be a permission change -- so a token the API refuses
    # is retried anonymously (60/h, and this makes one call a night).
    runs, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{repo}/commits/{sha}/check-runs?per_page=100&page={page}"
        body = None
        for tok in ([token] if token else []) + [None]:
            try:
                body = api("GET", url, tok)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (401, 403) or tok is None:
                    raise GateError(f"check-runs for {sha[:9]}: HTTP {e.code}") from e
            except Exception as e:  # noqa: BLE001 -- anything else is "could not read"
                raise GateError(f"check-runs for {sha[:9]} unreadable: {e}") from e
        batch = body.get("check_runs", [])
        runs += batch
        if len(batch) < 100:
            return runs
        page += 1


def gate(sha, prev, targets, check_runs, required=REQUIRED):
    tg = (prev or {}).get("targets", {})

    def handled(t):
        e = tg.get(t) or {}
        red = e.get("red") or {}
        return e.get("green_sha") == sha or (red.get("sha") == sha and red.get("reason") in BUILD_VERDICTS)

    if prev and all(handled(t) for t in targets):
        return "reused", "every target already holds a verdict for this SHA"
    pending, red = [], []
    for name in required:
        mine = [c for c in check_runs if c.get("name") == name]
        if not mine:
            pending.append(f"{name}: absent")
            continue
        latest = max(mine, key=lambda c: c.get("id", 0))  # a rerun supersedes
        if latest.get("status") != "completed":
            pending.append(f"{name}: {latest.get('status')}")
        elif latest.get("conclusion") != "success":
            red.append(f"{name}: {latest.get('conclusion')}")
    if red:
        return "red-ci", "; ".join(red)
    if pending:
        return "ci-pending", "; ".join(pending)
    return "build", "required checks success: " + ", ".join(required)


# ---------------------------------------------------------------- record

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_version(exe):
    # Empty cwd, no stdin, a hard timeout: every shipped bin answers --version
    # (clap), and a bin that cannot is not "working".
    with tempfile.TemporaryDirectory() as cwd:
        try:
            p = subprocess.run([exe, "--version"], cwd=cwd, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, str(e)
    out = (p.stdout or p.stderr).strip().splitlines()
    return p.returncode, (out[0] if out else "")


def record(target, sha, bins, bin_dir, dist, build_outcome="success", probe=probe_version, version=None):
    if build_outcome != "success":
        # A cancelled build proved nothing about the code, so it is not a build
        # verdict: the gate retries the SHA instead of calling it handled.
        reason = "build-failed" if build_outcome == "failure" else "build-cancelled"
        return {"target": target, "sha": sha, "status": "red", "tools": {},
                "red": {"sha": sha, "reason": reason, "detail": f"build step: {build_outcome}"}}
    tools, red = {}, None
    for b in bins:
        exe, tar = os.path.join(bin_dir, b), os.path.join(dist, f"{b}-{target}.tar.gz")
        if not (os.path.isfile(exe) and os.path.isfile(tar)):
            red = red or {"sha": sha, "reason": "missing-artifact", "detail": b}
            continue
        rc, line = probe(exe)
        m = VERSION_SHA.search(line + " ")
        vsha = m.group(1) if m else None
        # Only apr prints the commit, so a bin is bound to its SHA by bin_sha256,
        # not by --version: a MISSING sha is fine, a DIFFERENT one is not.
        if rc != 0 or (version and not re.search(rf"(?<![\d.]){re.escape(version)}(?![\d])", line)):
            red = red or {"sha": sha, "reason": "version-failed",
                          "detail": f"{b}: rc={rc}, wanted {version or 'any version'}: {line}"}
        elif vsha and not sha.startswith(vsha):
            red = red or {"sha": sha, "reason": "version-mismatch",
                          "detail": f"{b} prints {vsha}, built at {sha[:9]}"}
        tools[b] = {
            "asset": os.path.basename(tar),
            "sha256": sha256_file(tar),
            "bin_sha256": sha256_file(exe),
            "version_sha": sha if vsha and sha.startswith(vsha) else None,
            "version_output": line,
        }
    return {"target": target, "sha": sha, "status": "red" if red else "green", "red": red, "tools": tools}


# ---------------------------------------------------------------- merge

def merge(prev, fragments, sha, decision, targets, run_id, run_url, now):
    old = (prev or {}).get("targets", {})
    out = {}
    for t in targets:
        before = dict(old.get(t) or {"status": "red", "green_sha": None, "built_at": None,
                                     "built_run_id": None, "red": None, "tools": {}})
        frag = fragments.get(t)
        if decision == "build" and frag and frag["status"] == "green":
            out[t] = {"status": "green", "green_sha": sha, "built_at": now,
                      "built_run_id": run_id, "red": None, "tools": frag["tools"]}
        elif decision == "build":
            # No fragment: the runner died before recording, OR the download of a
            # good one failed. Red, never old-green -- but `no-fragment` is not a
            # BUILD verdict, so the gate retries this SHA on the next run.
            red = (frag or {}).get("red") or {"sha": sha, "reason": "no-fragment",
                                              "detail": "runner died or artifact download failed"}
            before.update(status="red", red=dict(red, run_url=run_url, ticket=None))
            out[t] = before  # the last good build stays served
        elif decision == "red-ci":
            before.update(status="red", red={"sha": sha, "reason": "red-ci", "run_url": run_url, "ticket": None})
            out[t] = before
        else:  # reused, ci-pending: nothing is wrong with this target
            out[t] = before
    if decision == "build":
        final = "built" if all(out[t]["green_sha"] == sha for t in targets) else "build-failed"
    else:
        final = decision
    return {"schema": SCHEMA, "source": "nightly", "repo": "paiml/aprender", "generated_at": now,
            "run_id": run_id, "run_url": run_url, "trigger_sha": sha, "decision": final,
            "targets": out}


# ---------------------------------------------------------------- publish

def publish_plan(manifest, sha, dist_files):
    """Asset names to (re)upload, in order. Only targets built at `sha` BY THIS
    RUN -- a reused night is green at `sha` too, but its assets are already
    published and its dist dir is empty. The manifest last, so a reader never
    sees a manifest ahead of its assets."""
    names = []
    for t in sorted(manifest["targets"]):
        e = manifest["targets"][t]
        if e["status"] != "green" or e["green_sha"] != sha or e.get("built_run_id") != manifest["run_id"]:
            continue
        for tool in sorted(e["tools"].values(), key=lambda x: x["asset"]):
            for n in (tool["asset"], tool["asset"] + ".sha256"):
                if n not in dist_files:
                    raise ValueError(f"manifest names {n}, absent from the dist dir")
                names.append(n)
    return names + [MANIFEST_ASSET]


def publish(manifest_path, dist, sha, repo, token):
    with open(manifest_path) as f:
        man = json.load(f)
    plan = publish_plan(man, sha, set(os.listdir(dist)) | {MANIFEST_ASSET})
    A = f"https://api.github.com/repos/{repo}"
    try:
        rel = api("GET", f"{A}/releases/tags/nightly", token)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        rel = None
    if plan[:-1]:  # something was built at sha: the tag follows it
        try:
            api("PATCH", f"{A}/git/refs/tags/nightly", token, json.dumps({"sha": sha, "force": True}).encode())
        except urllib.error.HTTPError as e:
            if e.code not in (404, 422):
                raise
            api("POST", f"{A}/git/refs", token, json.dumps({"ref": "refs/tags/nightly", "sha": sha}).encode())
    if rel is None:
        rel = api("POST", f"{A}/releases", token, json.dumps({
            "tag_name": "nightly", "target_commitish": sha, "name": "Nightly Build",
            "prerelease": True, "make_latest": "false"}).encode())
    existing = {a["name"]: a["id"] for a in api("GET", f"{A}/releases/{rel['id']}/assets?per_page=100", token)}
    for n, i in existing.items():  # a staged upload an earlier run died holding
        if n.startswith(STAGED):
            api("DELETE", f"{A}/releases/assets/{i}", token)
    # Phase 1 stages EVERY upload; a failure here leaves the release untouched.
    # Phase 2 only deletes and renames, in plan order, manifest last.
    staged = {}
    for n in plan:
        src = manifest_path if n == MANIFEST_ASSET else os.path.join(dist, n)
        with open(src, "rb") as f:
            staged[n] = api("POST", f"https://uploads.github.com/repos/{repo}/releases/{rel['id']}/assets"
                            f"?name={STAGED}{n}", token, f.read(), "application/octet-stream")["id"]
    for n in plan:
        if n in existing:
            api("DELETE", f"{A}/releases/assets/{existing[n]}", token)
        api("PATCH", f"{A}/releases/assets/{staged[n]}", token, json.dumps({"name": n}).encode())
        print(f"published {n}")
    rows = "\n".join(
        f"| `{t}` | {e['status']} | `{(e['green_sha'] or '-')[:9]}` | {', '.join(sorted(e['tools'])) or '-'} | "
        f"{(e.get('red') or {}).get('reason', '')} |" for t, e in sorted(man["targets"].items()))
    body = ("Automated nightly build of `main`, one arch independent of the other (#4189).\n\n"
            f"Decision for `{sha[:9]}`: **{man['decision']}** ({man['run_url']})\n\n"
            "| target | status | green_sha | tools | red reason |\n|---|---|---|---|---|\n" + rows +
            f"\n\nVerify against `{MANIFEST_ASSET}` (schema `{SCHEMA}`): sha256 of the executable must equal "
            "`bin_sha256`. This is a prerelease; for stable, see the latest tagged version.\n")
    api("PATCH", f"{A}/releases/{rel['id']}", token, json.dumps({"body": body}).encode())


# ---------------------------------------------------------------- self-test

def self_test():
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, want {want!r}"))
        if not ok:
            fails.append(name)

    S, OLD, T = "a" * 40, "b" * 40, TARGETS
    run = lambda n, s, c, i=1: {"name": n, "status": s, "conclusion": c, "id": i}  # noqa: E731
    green = [run("ci / gate", "completed", "success"), run("workspace-test", "completed", "success")]
    man = lambda tg: {"targets": tg}  # noqa: E731
    gt = lambda sha: {"status": "green", "green_sha": sha, "tools": {}}  # noqa: E731

    print("gate:")
    check("no prev manifest, CI green -> build", gate(S, None, T, green)[0], "build")
    check("HEAD == green_sha on every target -> reused (no work, no build)",
          gate(S, man({t: gt(S) for t in T}), T, [])[0], "reused")
    check("HEAD == green_sha on ONE target only -> build (the other arch still needs it)",
          gate(S, man({T[0]: gt(S), T[1]: gt(OLD)}), T, green)[0], "build")
    check("HEAD already build-failed on the other target -> reused (no rebuild loop)",
          gate(S, man({T[0]: gt(S), **{t: {"green_sha": OLD, "red": {"sha": S, "reason": "build-failed"}} for t in T[1:]}}), T, [])[0],
          "reused")
    check("HEAD was red-ci, CI rerun green -> build",
          gate(S, man({t: {"green_sha": OLD, "red": {"sha": S, "reason": "red-ci"}} for t in T}), T, green)[0], "build")
    check("a required check failed -> red-ci", gate(S, None, T, [run("ci / gate", "completed", "failure"), green[1]])[0],
          "red-ci")
    check("cancelled is not success -> red-ci",
          gate(S, None, T, [run("ci / gate", "completed", "cancelled"), green[1]])[0], "red-ci")
    check("a required check still running -> ci-pending",
          gate(S, None, T, [run("ci / gate", "in_progress", None), green[1]])[0], "ci-pending")
    check("a required check absent -> ci-pending, never build", gate(S, None, T, green[:1])[0], "ci-pending")
    check("rerun: failure then success -> build (latest run decides)",
          gate(S, None, T, [run("ci / gate", "completed", "failure", 1),
                            run("ci / gate", "completed", "success", 2), green[1]])[0], "build")
    check("rerun: success then failure -> red-ci (latest run decides)",
          gate(S, None, T, [run("ci / gate", "completed", "success", 1),
                            run("ci / gate", "completed", "failure", 2), green[1]])[0], "red-ci")
    check("a look-alike check name does not satisfy a required one",
          gate(S, None, T, [run("gate", "completed", "success"), green[1]])[0], "ci-pending")

    print("record:")
    with tempfile.TemporaryDirectory() as d:
        for b in ("apr", "pv"):
            open(os.path.join(d, b), "wb").write(b"exe-" + b.encode())
            open(os.path.join(d, f"{b}-{T[0]}.tar.gz"), "wb").write(b"tar-" + b.encode())
        fake = lambda out: (lambda exe: out[os.path.basename(exe)])  # noqa: E731
        good = {"apr": (0, f"apr 0.69.0 ({S[:9]})"), "pv": (0, "pv 0.69.0 (aprender provable-contracts verifier)")}
        ok = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good))
        check("every bin runs and prints its build SHA -> green", ok["status"], "green")
        check("pv prints no SHA -> version_sha null", ok["tools"]["pv"]["version_sha"], None)
        check("apr's short SHA is recorded as the full build SHA", ok["tools"]["apr"]["version_sha"], S)
        check("bin_sha256 hashes the EXECUTABLE, not the tarball",
              ok["tools"]["pv"]["bin_sha256"], hashlib.sha256(b"exe-pv").hexdigest())
        mm = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, apr=(0, f"apr 0.69.0 ({OLD[:9]})"))))
        check("a binary printing ANOTHER SHA -> version-mismatch", (mm["status"], mm["red"]["reason"]),
              ("red", "version-mismatch"))
        ng = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, apr=(0, "apr 0.69.1 (v0.69.1+no-git)"))))
        check("a +no-git build is not a mismatch; its version_sha is null",
              (ng["status"], ng["tools"]["apr"]["version_sha"]), ("green", None))
        vf = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(101, "panicked"))))
        check("--version failing -> version-failed", (vf["status"], vf["red"]["reason"]), ("red", "version-failed"))
        bf = record(T[0], S, ["apr", "pv"], d, d, build_outcome="failure", probe=fake(good))
        check("the build step failed -> build-failed, no tools", (bf["red"]["reason"], bf["tools"]), ("build-failed", {}))
        bc = record(T[0], S, ["apr", "pv"], d, d, build_outcome="cancelled", probe=fake(good))
        check("a cancelled build -> build-cancelled, and the gate retries it",
              (bc["red"]["reason"], gate(S, man({T[0]: gt(S), T[1]: {"green_sha": OLD, "red": bc["red"]}}), T, green)[0]),
              ("build-cancelled", "build"))
        vv = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good), version="0.69.0")
        check("every bin prints the crate version -> green (only apr prints a SHA)", vv["status"], "green")
        vw = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(0, "pv 0.68.0"))), version="0.69.0")
        check("a bin printing another crate version -> version-failed", (vw["status"], (vw.get("red") or {}).get("reason")),
              ("red", "version-failed"))
        vp = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(0, "pv 10.69.0"))), version="0.69.0")
        check("the version must match whole, not as a substring (10.69.0 != 0.69.0)", vp["status"], "red")
        os.remove(os.path.join(d, f"pv-{T[0]}.tar.gz"))
        ma = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good))
        check("a missing tarball -> missing-artifact", (ma["status"], ma["red"]["reason"]), ("red", "missing-artifact"))

    print("merge:")
    now = "2026-09-24T12:00:00Z"
    prev = {"targets": {t: {"status": "green", "green_sha": OLD, "tools": {"apr": {"asset": f"apr-{t}.tar.gz"}},
                            "red": None} for t in T}}
    gfrag = lambda t: {"target": t, "status": "green", "red": None,  # noqa: E731
                       "tools": {"apr": {"asset": f"apr-{t}.tar.gz"}, "pv": {"asset": f"pv-{t}.tar.gz"}}}
    rfrag = lambda t: {"target": t, "status": "red", "red": {"sha": S, "reason": "build-failed"}, "tools": {}}  # noqa: E731
    m = merge(prev, {t: gfrag(t) for t in T}, S, "build", T, 7, "u", now)
    check("every target green -> built, all at S", (m["decision"], [m["targets"][t]["green_sha"] for t in T]), ("built", [S] * len(T)))
    check("source is nightly (the installer compares it with the latest release)", m["source"], "nightly")
    m = merge(prev, {T[0]: gfrag(T[0]), T[1]: rfrag(T[1])}, S, "build", T, 7, "u", now)
    check("per-arch: x86_64 moves to S while aarch64 is red",
          (m["decision"], m["targets"][T[0]]["green_sha"], m["targets"][T[1]]["green_sha"], m["targets"][T[1]]["status"]),
          ("build-failed", S, OLD, "red"))
    check("the red target keeps serving its last green tools", list(m["targets"][T[1]]["tools"]), ["apr"])
    nf = merge(prev, {T[0]: gfrag(T[0])}, S, "build", T, 7, "u", now)
    check("a target with NO fragment is red (no-fragment), not silently old-green",
          (nf["targets"][T[1]]["status"], nf["targets"][T[1]]["red"]["reason"], nf["decision"]),
          ("red", "no-fragment", "build-failed"))
    check("...and the gate RETRIES it next run (a failed download is not a build verdict)",
          gate(S, nf, T, green)[0], "build")
    check("a red fragment for ANOTHER sha does not count (fragments are keyed by sha upstream)",
          merge(prev, {}, S, "build", T, 7, "u", now)["decision"], "build-failed")
    m = merge(prev, {}, S, "red-ci", T, 7, "u", now)
    check("red-ci: nothing built, reason per target, old green kept",
          (m["decision"], m["targets"][T[0]]["red"]["reason"], m["targets"][T[0]]["green_sha"]), ("red-ci", "red-ci", OLD))
    m = merge(prev, {}, S, "ci-pending", T, 7, "u", now)
    check("ci-pending: targets untouched", m["targets"] == prev["targets"], True)
    m = merge(prev, {}, S, "reused", T, 7, "u", now)
    check("reused: targets untouched, generated_at refreshed", (m["targets"] == prev["targets"], m["generated_at"]),
          (True, now))

    print("publish plan:")
    m = merge(prev, {T[0]: gfrag(T[0]), T[1]: rfrag(T[1])}, S, "build", T, 7, "u", now)
    files = {f"{b}-{t}.tar.gz{s}" for b in ("apr", "pv") for t in T for s in ("", ".sha256")}
    plan = publish_plan(m, S, files)
    check("only the green arch's assets are replaced; manifest last",
          plan, [f"apr-{T[0]}.tar.gz", f"apr-{T[0]}.tar.gz.sha256", f"pv-{T[0]}.tar.gz", f"pv-{T[0]}.tar.gz.sha256",
                 MANIFEST_ASSET])
    check("red-ci uploads the manifest only", publish_plan(merge(prev, {}, S, "red-ci", T, 7, "u", now), S, files),
          [MANIFEST_ASSET])
    both = merge(prev, {t: gfrag(t) for t in T}, S, "build", T, 7, "u", now)
    check("the next night is reused, and reused uploads the manifest only (dist is empty)",
          (gate(S, both, T, [])[0], publish_plan(merge(both, {}, S, "reused", T, 8, "u", now), S, set())),
          ("reused", [MANIFEST_ASSET]))
    check("ci-pending: arches still green at an OLDER sha are not re-uploaded",
          publish_plan(merge(prev, {}, S, "ci-pending", T, 7, "u", now), S, files), [MANIFEST_ASSET])
    try:
        publish_plan(m, S, files - {f"pv-{T[0]}.tar.gz.sha256"})
        got = "no error"
    except ValueError:
        got = "error"
    check("a manifest naming an absent asset refuses to publish", got, "error")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failing row(s)")
    return 1 if fails else 0


# ---------------------------------------------------------------- cli

def load(path):
    if path and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            doc = json.load(f)
        return doc
    return None


def load_manifest(path):
    doc = load(path)
    # An unknown or foreign manifest proves nothing about "no work".
    return doc if doc and doc.get("schema") == SCHEMA else None


def main(argv):
    if argv[1:2] == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(prog="nightly_manifest.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--sha", required=True)
    g.add_argument("--prev")
    g.add_argument("--targets", default=",".join(TARGETS))
    g.add_argument("--checks-json")
    g.add_argument("--repo", default="paiml/aprender")
    r = sub.add_parser("record")
    for a in ("--target", "--sha", "--bins", "--bin-dir", "--dist"):
        r.add_argument(a, required=True)
    r.add_argument("--build-outcome", default="success")
    r.add_argument("--version", help="the crate version every bin's --version must print")
    m = sub.add_parser("merge")
    m.add_argument("--prev")
    m.add_argument("--fragments", required=True)
    m.add_argument("--sha", required=True)
    m.add_argument("--decision", required=True, choices=["build", "reused", "red-ci", "ci-pending"])
    m.add_argument("--targets", default=",".join(TARGETS))
    m.add_argument("--run-id", type=int, default=0)
    m.add_argument("--run-url", default="")
    m.add_argument("--now")
    p = sub.add_parser("publish")
    for a in ("--manifest", "--dist", "--sha"):
        p.add_argument(a, required=True)
    p.add_argument("--repo", default="paiml/aprender")
    a = ap.parse_args(argv[1:])

    if a.cmd == "gate":
        try:
            runs = load(a.checks_json)["check_runs"] if a.checks_json else \
                fetch_check_runs(a.repo, a.sha, os.environ.get("GH_TOKEN"))
        except GateError as e:
            print(f"::error::{e}", file=sys.stderr)
            return 3
        decision, why = gate(a.sha, load_manifest(a.prev), a.targets.split(","), runs)
        print(decision)
        print(f"gate {a.sha[:9]}: {decision} -- {why}", file=sys.stderr)
        return 0
    if a.cmd == "record":
        print(json.dumps(record(a.target, a.sha, a.bins.split(","), a.bin_dir, a.dist, a.build_outcome,
                                version=a.version), indent=1))
        return 0
    if a.cmd == "merge":
        frags = {}
        if os.path.isdir(a.fragments):
            for n in sorted(os.listdir(a.fragments)):
                fr = load(os.path.join(a.fragments, n)) if n.startswith("fragment-") and n.endswith(".json") else None
                if fr and fr.get("sha") == a.sha:
                    frags[fr["target"]] = fr
        now = a.now or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = merge(load_manifest(a.prev), frags, a.sha, a.decision, a.targets.split(","), a.run_id, a.run_url, now)
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    if a.cmd == "publish":
        publish(a.manifest, a.dist, a.sha, a.repo, os.environ["GH_TOKEN"])
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
