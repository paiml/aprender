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
  bins     --metadata F [--format list|cargo]
           The shipped set, DERIVED from `cargo metadata --no-deps`: every
           workspace [[bin]], never a hand list (#4189 -- the nightly shipped 1
           of 28 while a list said "apr,pv"). `list` prints the comma-joined
           bin names (NIGHTLY_BINS); `cargo` prints the `-p P --bin B` args,
           plus `--features P/F` for each required feature. A bin name that two
           packages define (the root facade re-exports `apr`) ships once, from
           the member crate.
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
TARGETS = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
# Recorded and published like TARGETS, but never part of the verdict (#4292):
# darwin builds on mini, one 16 GB box that is sometimes offline. Its red or
# missing fragment must not fail the Linux nightly, and must not stop the gate
# from answering `reused` (a missing fragment is not a build verdict, so a
# required target would be rebuilt every run while mini is down).
ADVISORY_TARGETS = ["aarch64-apple-darwin"]
# A SHA printed in parentheses -- `apr 0.69.0 (aa7c6ef03)`. `pv 0.69.0
# (aprender provable-contracts verifier)` prints none, and neither does a
# tarball build's `(v0.69.1+no-git)`.
VERSION_SHA = re.compile(r"\(([0-9a-f]{7,40})[\s),]")
# A BUILD verdict on S means S has been handled. red-ci / ci-pending do not:
# CI can be rerun green, and then S is work again.
BUILD_VERDICTS = {"build-failed", "version-mismatch", "version-no-sha", "version-failed", "missing-artifact",
                  "variant-feature-missing"}
# The string a feature leaves in the executable. `libcuda.so` is the runtime
# loader name, present only under --features cuda (1 vs 0 on 0.66.0; the same
# proof binary-release.yml's cuda lane runs).
VARIANT_MARKERS = {"cuda": b"libcuda.so"}


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
    # (clap), and a bin that cannot is not "working". The workflow passes a
    # RELATIVE --bin-dir; resolved against the empty cwd it names nothing, and
    # every bin read as version-failed rc=127 -- so resolve it first.
    exe = os.path.abspath(exe)
    with tempfile.TemporaryDirectory() as cwd:
        try:
            p = subprocess.run([exe, "--version"], cwd=cwd, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, str(e)
    out = (p.stdout or p.stderr).strip().splitlines()
    return p.returncode, (out[0] if out else "")


def record(target, sha, bins, bin_dir, dist, build_outcome="success", probe=probe_version, version=None,
           variants=()):
    if build_outcome != "success":
        # A cancelled build proved nothing about the code, so it is not a build
        # verdict: the gate retries the SHA instead of calling it handled.
        reason = "build-failed" if build_outcome == "failure" else "build-cancelled"
        return {"target": target, "sha": sha, "status": "red", "tools": {},
                "red": {"sha": sha, "reason": reason, "detail": f"build step: {build_outcome}"}}
    # dist=None is the release-commit smoke (`smoke`, #4189): the same verdicts
    # on the built executables, with no tarball to require or hash.
    # A variant `bin:feature` (#4326: apr:cuda) is a second build of one bin with a
    # cargo feature, at <bin_dir>/<feature>/<bin>, shipped as <bin>-<target>-<feature>
    # -- the name binary-release.yml gives it, so a resolver asks one name of the
    # nightly, an rc and a release. It is judged as every bin is, and must also
    # carry the feature's marker: a cuda build that lost --features is a CPU apr
    # under a cuda name, which would regress every GPU verb on the host installing it.
    items = [(b, b, os.path.join(bin_dir, b), f"{b}-{target}.tar.gz", None) for b in bins]
    for v in variants:
        b, feat = v.split(":", 1)
        items.append((f"{b}-{feat}", b, os.path.join(bin_dir, feat, b), f"{b}-{target}-{feat}.tar.gz",
                       VARIANT_MARKERS.get(feat)))
    tools, red = {}, None
    for key, b, exe, tar_name, marker in items:
        tar = os.path.join(dist, tar_name) if dist is not None else None
        if not (os.path.isfile(exe) and (tar is None or os.path.isfile(tar))):
            red = red or {"sha": sha, "reason": "missing-artifact", "detail": key}
            continue
        if marker is not None:
            with open(exe, "rb") as f:
                if marker not in f.read():
                    red = red or {"sha": sha, "reason": "variant-feature-missing",
                                  "detail": f"{key}: no {marker.decode()} in the executable"}
        rc, line = probe(exe)
        m = VERSION_SHA.search(line + " ")
        vsha = m.group(1) if m else None
        # #4219: every [[bin]] prints its build SHA via aprender-build-sha, so a
        # bin that prints NONE (a +no-git build, or one that never adopted the
        # crate) is as red as one printing ANOTHER sha -- the fleet could not
        # tell which commit it is running.
        if rc != 0 or (version and not re.search(rf"(?<![\d.]){re.escape(version)}(?![\d])", line)):
            red = red or {"sha": sha, "reason": "version-failed",
                          "detail": f"{b}: rc={rc}, wanted {version or 'any version'}: {line}"}
        elif vsha and not sha.startswith(vsha):
            red = red or {"sha": sha, "reason": "version-mismatch",
                          "detail": f"{b} prints {vsha}, built at {sha[:9]}"}
        elif not vsha:
            red = red or {"sha": sha, "reason": "version-no-sha",
                          "detail": f"{b}: --version names no build SHA: {line}"}
        tools[key] = {
            "asset": os.path.basename(tar) if tar else None,
            "sha256": sha256_file(tar) if tar else None,
            "bin_sha256": sha256_file(exe),
            "version_sha": sha if vsha and sha.startswith(vsha) else None,
            "version_output": line,
        }
    return {"target": target, "sha": sha, "status": "red" if red else "green", "red": red, "tools": tools}


# ---------------------------------------------------------------- bins

def workspace_bins(meta):
    """[(bin, package, required_features)] for every workspace [[bin]], one per
    bin name, sorted. A name defined twice resolves to the package NOT at the
    workspace root (the facade), so one cargo build never has two outputs
    fighting over target/<t>/release/<bin>."""
    root = os.path.join(meta["workspace_root"], "Cargo.toml")
    members = set(meta.get("workspace_members", []))
    by_name = {}
    for p in meta["packages"]:
        if members and p["id"] not in members:
            continue
        for t in p["targets"]:
            if "bin" not in t["kind"]:
                continue
            cand = (t["name"], p["name"], tuple(t.get("required-features", [])), p["manifest_path"] == root)
            cur = by_name.get(t["name"])
            if cur is None or (cur[3] and not cand[3]):
                by_name[t["name"]] = cand
            elif not cur[3] and not cand[3] and cur[1] != cand[1]:
                raise ValueError(f"bin {t['name']} is defined by both {cur[1]} and {cand[1]}")
    return [(b, pkg, list(f)) for b, pkg, f, _ in sorted(by_name.values())]


def cargo_args(bins):
    args, pkgs = [], []
    for b, pkg, feats in bins:
        if pkg not in pkgs:
            pkgs.append(pkg)
        args += ["--bin", b] + [x for f in feats for x in ("--features", f"{pkg}/{f}")]
    return [x for p in pkgs for x in ("-p", p)] + args


# ---------------------------------------------------------------- merge

def merge(prev, fragments, sha, decision, targets, run_id, run_url, now, advisory=()):
    old = (prev or {}).get("targets", {})
    out = {}
    for t in list(targets) + [x for x in advisory if x not in targets]:
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
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "rel"))
        with open(os.path.join(d, "rel", "fake"), "w") as f:
            f.write("#!/bin/sh\necho 'fake 0.1.0 (abc123def)'\n")
        os.chmod(os.path.join(d, "rel", "fake"), 0o755)
        here = os.getcwd()
        os.chdir(d)
        try:
            got = probe_version(os.path.join("rel", "fake"))
        finally:
            os.chdir(here)
    check("probe_version runs a RELATIVE bin path (the workflow's --bin-dir)",
          got, (0, "fake 0.1.0 (abc123def)"))
    check("no prev manifest, CI green -> build", gate(S, None, T, green)[0], "build")
    check("HEAD == green_sha on every target -> reused (no work, no build)",
          gate(S, man({t: gt(S) for t in T}), T, [])[0], "reused")
    check("HEAD == green_sha on ONE target only -> build (the other arch still needs it)",
          gate(S, man({T[0]: gt(S), T[1]: gt(OLD)}), T, green)[0], "build")
    check("HEAD already build-failed on the other target -> reused (no rebuild loop)",
          gate(S, man({T[0]: gt(S), T[1]: {"green_sha": OLD, "red": {"sha": S, "reason": "build-failed"}}}), T, [])[0],
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

    print("bins:")
    def pkg(name, bins, root=False, feats=None):
        return {"id": name, "name": name,
                "manifest_path": "/w/Cargo.toml" if root else f"/w/crates/{name}/Cargo.toml",
                "targets": [{"name": "lib" + name, "kind": ["lib"]}] +
                           [{"name": b, "kind": ["bin"], **({"required-features": feats} if feats else {})}
                            for b in bins]}
    meta = {"workspace_root": "/w", "packages": [
        pkg("aprender", ["apr"], root=True), pkg("apr-cli", ["apr", "apr-corpus-ingest"]),
        pkg("contracts-cli", ["pv"]), pkg("db", ["aprender-db"], feats=["server"]), pkg("core", [])]}
    meta["workspace_members"] = [p["id"] for p in meta["packages"]]
    wb = workspace_bins(meta)
    check("every workspace [[bin]] ships, a lib-only crate adds none",
          [b for b, _, _ in wb], ["apr", "apr-corpus-ingest", "aprender-db", "pv"])
    check("a bin the facade re-exports ships from the member crate", dict((b, p) for b, p, _ in wb)["apr"], "apr-cli")
    check("required-features become --features pkg/feat",
          cargo_args(wb), ["-p", "apr-cli", "-p", "db", "-p", "contracts-cli", "--bin", "apr",
                           "--bin", "apr-corpus-ingest", "--bin", "aprender-db", "--features", "db/server",
                           "--bin", "pv"])
    ext = dict(meta, packages=meta["packages"] + [pkg("vendored", ["tool"])])
    check("a package outside workspace_members ships nothing", [b for b, _, _ in workspace_bins(ext)],
          ["apr", "apr-corpus-ingest", "aprender-db", "pv"])
    dup = dict(meta, packages=meta["packages"] + [pkg("other", ["pv"])])
    dup["workspace_members"] = [p["id"] for p in dup["packages"]]
    try:
        workspace_bins(dup)
        check("two member crates defining one bin name -> refused", "accepted", "refused")
    except ValueError:
        check("two member crates defining one bin name -> refused", "refused", "refused")

    print("record:")
    with tempfile.TemporaryDirectory() as d:
        for b in ("apr", "pv"):
            open(os.path.join(d, b), "wb").write(b"exe-" + b.encode())
            open(os.path.join(d, f"{b}-{T[0]}.tar.gz"), "wb").write(b"tar-" + b.encode())
        fake = lambda out: (lambda exe: out[os.path.basename(exe)])  # noqa: E731
        good = {"apr": (0, f"apr 0.69.0 ({S[:9]})"), "pv": (0, f"pv 0.69.0 ({S[:9]}) (aprender provable-contracts verifier)")}
        ok = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good))
        check("every bin runs and prints its build SHA -> green", ok["status"], "green")
        check("pv's short SHA is recorded as the full build SHA", ok["tools"]["pv"]["version_sha"], S)
        ns = record(T[0], S, ["apr", "pv"], d, d,
                    probe=fake(dict(good, pv=(0, "pv 0.69.0 (aprender provable-contracts verifier)"))))
        check("a bin printing NO SHA -> version-no-sha (#4219)", (ns["status"], (ns["red"] or {}).get("reason"),
              ns["tools"]["pv"]["version_sha"]), ("red", "version-no-sha", None))
        check("apr's short SHA is recorded as the full build SHA", ok["tools"]["apr"]["version_sha"], S)
        check("bin_sha256 hashes the EXECUTABLE, not the tarball",
              ok["tools"]["pv"]["bin_sha256"], hashlib.sha256(b"exe-pv").hexdigest())
        mm = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, apr=(0, f"apr 0.69.0 ({OLD[:9]})"))))
        check("a binary printing ANOTHER SHA -> version-mismatch", (mm["status"], mm["red"]["reason"]),
              ("red", "version-mismatch"))
        ng = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, apr=(0, "apr 0.69.1 (v0.69.1+no-git)"))))
        check("a +no-git build names no SHA -> version-no-sha; its version_sha is null",
              (ng["status"], (ng["red"] or {}).get("reason"), ng["tools"]["apr"]["version_sha"]), ("red", "version-no-sha", None))
        vf = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(101, "panicked"))))
        check("--version failing -> version-failed", (vf["status"], vf["red"]["reason"]), ("red", "version-failed"))
        bf = record(T[0], S, ["apr", "pv"], d, d, build_outcome="failure", probe=fake(good))
        check("the build step failed -> build-failed, no tools", (bf["red"]["reason"], bf["tools"]), ("build-failed", {}))
        bc = record(T[0], S, ["apr", "pv"], d, d, build_outcome="cancelled", probe=fake(good))
        check("a cancelled build -> build-cancelled, and the gate retries it",
              (bc["red"]["reason"], gate(S, man({T[0]: gt(S), T[1]: {"green_sha": OLD, "red": bc["red"]}}), T, green)[0]),
              ("build-cancelled", "build"))
        vv = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good), version="0.69.0")
        check("every bin prints the crate version -> green and its build SHA", vv["status"], "green")
        vw = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(0, "pv 0.68.0"))), version="0.69.0")
        check("a bin printing another crate version -> version-failed", (vw["status"], (vw.get("red") or {}).get("reason")),
              ("red", "version-failed"))
        vp = record(T[0], S, ["apr", "pv"], d, d, probe=fake(dict(good, pv=(0, "pv 10.69.0"))), version="0.69.0")
        check("the version must match whole, not as a substring (10.69.0 != 0.69.0)", vp["status"], "red")
        os.remove(os.path.join(d, f"pv-{T[0]}.tar.gz"))
        ma = record(T[0], S, ["apr", "pv"], d, d, probe=fake(good))
        check("a missing tarball -> missing-artifact", (ma["status"], ma["red"]["reason"]), ("red", "missing-artifact"))
        sk = record("release-commit", S, ["apr", "pv"], d, None, probe=fake(good), version="0.69.0")
        check("smoke (dist=None): no tarball is required -> green", sk["status"], "green")
        os.remove(os.path.join(d, "pv"))
        sm = record("release-commit", S, ["apr", "pv"], d, None, probe=fake(good), version="0.69.0")
        check("smoke: a bin that did not build -> missing-artifact",
              (sm["status"], (sm["red"] or {}).get("reason")), ("red", "missing-artifact"))
        sv = record("release-commit", S, ["apr"], d, None, probe=fake(dict(good, apr=(0, "apr 0.69.0"))), version="0.69.0")
        check("smoke: a bin printing no SHA -> version-no-sha",
              (sv["status"], (sv["red"] or {}).get("reason")), ("red", "version-no-sha"))

    print("record, variant (#4326: apr:cuda):")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "cuda"))
        cu_exe, cu_tar = os.path.join(d, "cuda", "apr"), os.path.join(d, f"apr-{T[0]}-cuda.tar.gz")
        open(os.path.join(d, "apr"), "wb").write(b"exe-apr")
        open(os.path.join(d, f"apr-{T[0]}.tar.gz"), "wb").write(b"tar-apr")
        open(cu_exe, "wb").write(b"exe-apr\0libcuda.so\0")
        open(cu_tar, "wb").write(b"tar-apr-cuda")
        by_path = lambda out: (lambda exe: out.get(exe, (0, f"apr 0.69.0 ({S[:9]})")))  # noqa: E731
        vg = record(T[0], S, ["apr"], d, d, probe=by_path({}), variants=["apr:cuda"])
        check("a cuda variant carrying libcuda.so, printing S -> green",
              (vg["status"], sorted(vg["tools"])), ("green", ["apr", "apr-cuda"]))
        check("the variant ships under binary-release.yml's name, apr-<target>-cuda.tar.gz",
              vg["tools"]["apr-cuda"]["asset"], f"apr-{T[0]}-cuda.tar.gz")
        check("the variant's bin_sha256 is ITS executable, not the CPU apr",
              vg["tools"]["apr-cuda"]["bin_sha256"], hashlib.sha256(b"exe-apr\0libcuda.so\0").hexdigest())
        check("no variants asked -> the manifest is unchanged (no apr-cuda row)",
              sorted(record(T[0], S, ["apr"], d, d, probe=by_path({}))["tools"]), ["apr"])
        vm = record(T[0], S, ["apr"], d, d, probe=by_path({cu_exe: (0, f"apr 0.69.0 ({OLD[:9]})")}),
                    variants=["apr:cuda"])
        check("the variant printing ANOTHER SHA -> version-mismatch",
              (vm["status"], (vm["red"] or {}).get("reason")), ("red", "version-mismatch"))
        open(cu_exe, "wb").write(b"exe-apr-cpu-only")
        vn = record(T[0], S, ["apr"], d, d, probe=by_path({}), variants=["apr:cuda"])
        check("a 'cuda' build with no libcuda.so (lost --features) -> variant-feature-missing",
              (vn["status"], (vn["red"] or {}).get("reason")), ("red", "variant-feature-missing"))
        check("variant-feature-missing is a BUILD verdict (the gate does not rebuild S)",
              "variant-feature-missing" in BUILD_VERDICTS, True)
        os.remove(cu_tar)
        vt = record(T[0], S, ["apr"], d, d, probe=by_path({}), variants=["apr:cuda"])
        check("the variant's tarball missing -> missing-artifact, naming apr-cuda",
              (vt["status"], (vt["red"] or {}).get("reason"), (vt["red"] or {}).get("detail")),
              ("red", "missing-artifact", "apr-cuda"))
        mv = {"run_id": 7, "targets": {T[0]: {"status": "green", "green_sha": S, "built_run_id": 7,
                                              "tools": vg["tools"]}}}
        names = {f"apr-{T[0]}{x}.tar.gz{y}" for x in ("", "-cuda") for y in ("", ".sha256")}
        check("publish plan uploads the cuda tarball and its .sha256",
              {f"apr-{T[0]}-cuda.tar.gz", f"apr-{T[0]}-cuda.tar.gz.sha256"} <= set(publish_plan(mv, S, names)), True)

    print("merge:")
    now = "2026-09-24T12:00:00Z"
    prev = {"targets": {t: {"status": "green", "green_sha": OLD, "tools": {"apr": {"asset": f"apr-{t}.tar.gz"}},
                            "red": None} for t in T}}
    gfrag = lambda t: {"target": t, "status": "green", "red": None,  # noqa: E731
                       "tools": {"apr": {"asset": f"apr-{t}.tar.gz"}, "pv": {"asset": f"pv-{t}.tar.gz"}}}
    rfrag = lambda t: {"target": t, "status": "red", "red": {"sha": S, "reason": "build-failed"}, "tools": {}}  # noqa: E731
    m = merge(prev, {T[0]: gfrag(T[0]), T[1]: gfrag(T[1])}, S, "build", T, 7, "u", now)
    check("both green -> built, both at S", (m["decision"], [m["targets"][t]["green_sha"] for t in T]), ("built", [S, S]))
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
    D = ADVISORY_TARGETS[0]
    check("darwin is advisory, not a required target", (ADVISORY_TARGETS, D in T), (["aarch64-apple-darwin"], False))
    tg = lambda m: m["targets"].get(D) or {}  # noqa: E731 -- absent reads as a FAIL row, not a crash
    both = {T[0]: gfrag(T[0]), T[1]: gfrag(T[1])}
    m = merge(prev, dict(both, **{D: gfrag(D)}), S, "build", T, 7, "u", now, advisory=[D])
    check("advisory green -> recorded green at S, decision built",
          (m["decision"], tg(m).get("status"), tg(m).get("green_sha")), ("built", "green", S))
    m = merge(prev, both, S, "build", T, 7, "u", now, advisory=[D])
    check("advisory with NO fragment (mini offline) -> red no-fragment, decision STILL built",
          (m["decision"], tg(m).get("status"), (tg(m).get("red") or {}).get("reason")), ("built", "red", "no-fragment"))
    check("...and the gate answers reused next run: a down mini does not force a rebuild",
          gate(S, m, T, [])[0], "reused")
    pd = dict(prev, targets=dict(prev["targets"], **{D: {"status": "green", "green_sha": OLD, "red": None,
                                                       "tools": {"apr": {"asset": f"apr-{D}.tar.gz"}}}}))
    m = merge(pd, dict(both, **{D: rfrag(D)}), S, "build", T, 7, "u", now, advisory=[D])
    check("advisory red -> decision built, darwin keeps serving its last green",
          (m["decision"], tg(m).get("green_sha"), list(tg(m).get("tools") or {})), ("built", OLD, ["apr"]))
    m = merge(prev, dict(both, **{D: gfrag(D)}), S, "build", T, 7, "u", now, advisory=[D])
    dfiles = {f"{b}-{t}.tar.gz{x}" for b in ("apr", "pv") for t in T + [D] for x in ("", ".sha256")}
    check("publish plan uploads the green darwin tarball",
          f"apr-{D}.tar.gz" in publish_plan(m, S, dfiles), True)
    with tempfile.TemporaryDirectory() as d:  # the CLI the workflow calls: --advisory defaults to darwin
        for t in T + [D]:
            with open(os.path.join(d, f"fragment-{t}.json"), "w") as f:
                json.dump(dict(gfrag(t), sha=S), f)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "merge", "--fragments", d, "--sha", S,
                            "--decision", "build", "--now", now], capture_output=True, text=True)
        cli = json.loads(p.stdout) if p.returncode == 0 else {"targets": {}}
        check("CLI merge records darwin by default, decision from the required targets",
              (cli.get("decision"), tg(cli).get("green_sha")), ("built", S))

    print("publish plan:")
    m = merge(prev, {T[0]: gfrag(T[0]), T[1]: rfrag(T[1])}, S, "build", T, 7, "u", now)
    files = {f"{b}-{t}.tar.gz{s}" for b in ("apr", "pv") for t in T for s in ("", ".sha256")}
    plan = publish_plan(m, S, files)
    check("only the green arch's assets are replaced; manifest last",
          plan, [f"apr-{T[0]}.tar.gz", f"apr-{T[0]}.tar.gz.sha256", f"pv-{T[0]}.tar.gz", f"pv-{T[0]}.tar.gz.sha256",
                 MANIFEST_ASSET])
    check("red-ci uploads the manifest only", publish_plan(merge(prev, {}, S, "red-ci", T, 7, "u", now), S, files),
          [MANIFEST_ASSET])
    both = merge(prev, {T[0]: gfrag(T[0]), T[1]: gfrag(T[1])}, S, "build", T, 7, "u", now)
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
    r.add_argument("--variants", default="", help="bin:feature,.. built at <bin-dir>/<feature>/<bin> (#4326)")
    m = sub.add_parser("merge")
    m.add_argument("--prev")
    m.add_argument("--fragments", required=True)
    m.add_argument("--sha", required=True)
    m.add_argument("--decision", required=True, choices=["build", "reused", "red-ci", "ci-pending"])
    m.add_argument("--targets", default=",".join(TARGETS))
    m.add_argument("--advisory", default=",".join(ADVISORY_TARGETS), help="recorded, never in the decision (#4292)")
    m.add_argument("--run-id", type=int, default=0)
    m.add_argument("--run-url", default="")
    m.add_argument("--now")
    bn = sub.add_parser("bins")
    bn.add_argument("--metadata", required=True, help="`cargo metadata --no-deps --format-version 1` output")
    bn.add_argument("--format", default="list", choices=["list", "cargo"])
    sm = sub.add_parser("smoke", help="record's verdicts on built bins, no tarballs; rc 1 when red")
    for a in ("--sha", "--bins", "--bin-dir", "--version"):
        sm.add_argument(a, required=True)
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
                                version=a.version, variants=[v for v in a.variants.split(",") if v]), indent=1))
        return 0
    if a.cmd == "merge":
        frags = {}
        if os.path.isdir(a.fragments):
            for n in sorted(os.listdir(a.fragments)):
                fr = load(os.path.join(a.fragments, n)) if n.startswith("fragment-") and n.endswith(".json") else None
                if fr and fr.get("sha") == a.sha:
                    frags[fr["target"]] = fr
        now = a.now or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = merge(load_manifest(a.prev), frags, a.sha, a.decision, a.targets.split(","), a.run_id, a.run_url, now,
                    advisory=[x for x in a.advisory.split(",") if x])
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    if a.cmd == "bins":
        wb = workspace_bins(load(a.metadata))
        print(",".join(b for b, _, _ in wb) if a.format == "list" else " ".join(cargo_args(wb)))
        return 0
    if a.cmd == "smoke":
        out = record("release-commit", a.sha, a.bins.split(","), a.bin_dir, None, version=a.version)
        print(json.dumps(out, indent=1))
        return 1 if out["status"] != "green" else 0
    if a.cmd == "publish":
        publish(a.manifest, a.dist, a.sha, a.repo, os.environ["GH_TOKEN"])
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
