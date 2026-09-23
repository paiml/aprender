#!/usr/bin/env python3
"""facade_registry_pin.py ROOT INDEX_BASE: can each crates/facades crate in the tree actually ship? (#4111)

A facade's own version lives on its own line (aprender#2546), while its upstream requirement tracks
the aprender version. Until #4111 nothing moved the own version, so after 0.4.0 was published every
cascade found "provable-contracts 0.4.0 already exists" and uploaded nothing. The facades stayed
frozen at aprender-contracts ^0.64.0 through 0.69.1, and the publish dry-run passed because it only
WARNS about an existing version.

For each facade package (a crates/facades/*/Cargo.toml with a [package]) this reads its version and
its path dependencies' version requirements from the TREE, then reads the sparse index at INDEX_BASE:
  OK    <crate> <ver> not on the index yet             the cascade uploads it
  OK    <crate> <ver> on the index with the same reqs  nothing to upload, and nothing is lost
  FAIL  <crate> <ver> on the index with other reqs     the tree's pin can never ship: bump the facade
                                                      version (scripts/bump-version.sh does)
  ERR   the index could not be read                    no verdict; the caller refuses
Exit: 0 all OK, 1 any FAIL, 2 could not answer (and no FAIL seen).
INDEX_BASE is https://index.crates.io in production; the preflight self-test serves file:// fixtures.
"""
import json
import pathlib
import sys
import tomllib
import urllib.error
import urllib.request


def index_path(name):
    n = name.lower()
    if len(n) <= 2:
        return "%d/%s" % (len(n), n)
    if len(n) == 3:
        return "3/%s/%s" % (n[0], n)
    return "%s/%s/%s" % (n[:2], n[2:4], n)


def cargo_req(req):
    """A manifest's `version = "0.69.0"` is published as the requirement "^0.69.0"."""
    req = req.strip()
    return "^" + req if req[:1].isdigit() else req


def tree_facades(root):
    fdir = root / "crates" / "facades"
    ws = tomllib.loads((fdir / "Cargo.toml").read_text())
    ws_ver = ws.get("workspace", {}).get("package", {}).get("version")
    out = []
    for man in sorted(fdir.glob("*/Cargo.toml")):
        m = tomllib.loads(man.read_text())
        pkg = m.get("package")
        if not pkg:
            continue
        ver = pkg.get("version")
        if isinstance(ver, dict):
            ver = ws_ver
        reqs = {}
        for key, dep in (m.get("dependencies") or {}).items():
            if isinstance(dep, dict) and "path" in dep and "version" in dep:
                reqs[dep.get("package", key)] = cargo_req(dep["version"])
        out.append((pkg["name"], ver, reqs))
    return out


def index_entries(base, name):
    """-> list of index records, [] when the crate is absent (404 / no file), None when unreadable."""
    url = "%s/%s" % (base.rstrip("/"), index_path(name))
    req = urllib.request.Request(url, headers={"User-Agent": "aprender-publish-preflight R8 (#4111)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        return [] if e.code == 404 else None
    except FileNotFoundError:
        return []
    except (urllib.error.URLError, OSError, ValueError):
        return None
    recs = []
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            return None
    return recs


def main(argv):
    if len(argv) != 3:
        print("usage: facade_registry_pin.py ROOT INDEX_BASE", file=sys.stderr)
        return 2
    root, base = pathlib.Path(argv[1]), argv[2]
    if not (root / "crates" / "facades" / "Cargo.toml").is_file():
        print("OK    no crates/facades workspace in this tree")
        return 0
    try:
        facades = tree_facades(root)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print("ERR   cannot read the crates/facades manifests: %s" % e)
        return 2
    if not facades:
        print("ERR   crates/facades holds no package (vacuous)")
        return 2
    rc = 0
    for name, ver, reqs in facades:
        recs = index_entries(base, name)
        if recs is None:
            print("ERR   %s: the index at %s could not be read" % (name, base))
            rc = max(rc, 2) if rc != 1 else 1
            continue
        live = next((r for r in recs if r.get("vers") == ver), None)
        if live is None:
            print("OK    %s %s is not on the index yet: the cascade uploads it (reqs %s)" % (name, ver, reqs or "none"))
            continue
        published = {d.get("package") or d.get("name"): d.get("req") for d in live.get("deps", [])}
        drift = {p: (published.get(p), want) for p, want in reqs.items() if published.get(p) != want}
        if drift:
            detail = ", ".join("%s: published %s, tree %s" % (p, got, want) for p, (got, want) in sorted(drift.items()))
            print("FAIL  %s %s is ALREADY on crates.io with other requirements (%s). The upload would be refused and "
                  "the tree's pin would never ship: bump the facade version (scripts/bump-version.sh)" % (name, ver, detail))
            rc = 1
        else:
            print("OK    %s %s is on the index with the same requirements (%s): nothing to upload" % (name, ver, reqs or "none"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
