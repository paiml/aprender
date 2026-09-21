#!/usr/bin/env python3
"""The set of crates a release cascade must ship — read from EVERY workspace.

WHY THIS FILE EXISTS (aprender#2559)
------------------------------------
`scripts/cascade-publish.sh` hand-maintains a TIERS[] table, and
`scripts/cascade-drain.sh` recovered its universe by grepping that same table.
MEASURED on c22fe88ef: TIERS[] listed exactly the 70 publishable crates of the
ROOT workspace — a perfect match, zero drift, and therefore no signal that
anything was missing.

    $ comm -13 <TIERS> <root publishable>   -> empty
    $ comm -23 <TIERS> <root publishable>   -> empty

But `crates/facades/` is a SECOND workspace, `exclude`d from the root, holding
three publishable crates (`provable-contracts`, `-macros`, `-cli`). None of
them appeared anywhere in the cascade, and the cascade's FINAL VERIFICATION
loop iterates TIERS[] — so it printed "ALL crates at $TARGET" while three
crates with 57K downloads between them had never been uploaded. Absence read
as success. That is the "guard's universe built from the wrong side" defect:
the loop cannot iterate what it cannot see, and a universe derived from one
workspace is complete *with respect to itself*.

The remedy is to stop deriving the universe from a hand-written list or from a
single `cargo metadata`, and to derive it from EVERY workspace this repository
publishes out of. Every consumer — the cascade, the drain, the coverage guard,
the publish-safety scan — reads it from here, so they cannot disagree about
what "the release" contains.

TWO PROPERTIES THE CONSUMERS NEED AND CANNOT GET FROM A BARE NAME
----------------------------------------------------------------
1. MANIFEST PATH. `cargo publish -p provable-contracts` from the repo root is
   not merely wrong, it is impossible — MEASURED:

       $ cargo publish -p provable-contracts --dry-run --no-verify
       error: package ID specification `provable-contracts` did not match any packages
       rc=101

   so adding the name to TIERS[] without also carrying its manifest path would
   have produced a cascade that fails on every pass. Excluded crates must be
   published with `--manifest-path`.

2. VERSION. The facades version INDEPENDENTLY of the aprender version line
   (0.4.0 vs 0.63.0) and that independence is deliberate and documented
   (aprender#2546): these crate names have no 0.63.0 history. A cascade that
   compares every crate against one `$TARGET_VERSION` would judge the facades
   permanently behind, never reach N/N, and report a false failure on an
   append-only registry — the single most dangerous thing the drain can say.
   So the expected version travels WITH the crate, from its own workspace.

Output is TSV, one row per publishable crate, sorted by name:

    <name>\t<version>\t<absolute manifest path>\t<workspace root>

Usage:
    cascade_universe.py <repo-root>            # all workspaces
    cascade_universe.py --names <repo-root>    # names only, one per line
    cascade_universe.py --order <repo-root>    # the PUBLISH ORDER: same rows, dependencies first
    cascade_universe.py --order --names <repo-root>
    cascade_universe.py --edges <repo-root>    # "<crate>\t<dep>": every must-precede edge inside the
                                               # universe (what check_cascade_covers_all_crates.sh
                                               # checks a publish sequence against)

THE PUBLISH ORDER (#3462). cascade-publish.sh used to hand-maintain TIERS[], and MEASURED at
v0.68.1 it was not a dependency order: 47 (crate -> non-dev workspace dep) pairs sat with the
dep in a LATER tier (aprender-core in T2 needs aprender-compute in T6; apr-cli in T10 needs eight
T13 crates). It only ever published because cascade-drain.sh re-ran it until the deferrals
stopped. Under a stop-on-first-non-zero rule (operator, 0.68.1) it stops at crate 13. The order
is now DERIVED here, from the same metadata as the universe: crate B must be on the registry
before crate A when A depends on B through a normal or build dependency, or through a VERSIONED
dev-dependency. cargo keeps a versioned dev-dependency in the published manifest and resolves it
on the registry (PMAT-955: 0.65.0 stuck at 48/74 on aprender-core <-> aprender-test-lib); a
path-only dev-dependency (req "*") is stripped at publish and orders nothing. Ties break by
(workspace, name): root-workspace crates first, then the excluded workspaces' crates (the
facades), so the order is deterministic and the facades land LAST. A cycle is exit 2, naming
the crates left on it: no order exists, and pretending one does is the defect.

A crate carrying `publish = false` is not in the universe: it is never
uploaded, so the cascade must not wait for it.
"""

import json
import os
import subprocess
import sys

# Every workspace this repository publishes out of, as a path relative to the
# repo root. `crates/facades` is `exclude`d from the root workspace on purpose
# (two primary packages sharing one lib name collide on the uplifted rlib —
# rust-lang/cargo#6313), which is exactly why it has to be named HERE: cargo
# will never volunteer it.
#
# Adding a workspace to this list is the whole maintenance burden. If a third
# excluded workspace ever appears, scripts/check_cascade_covers_all_crates.sh
# fails until it is listed and tiered.
WORKSPACES = (
    ".",
    "crates/facades",
)

# A universe smaller than this means the enumeration broke, not that the repo
# shrank. A vacuous universe would let every consumer report a clean pass over
# nothing, which is the failure mode this file exists to remove.
MIN_CRATES = 70


def metadata(repo_root, ws):
    manifest = os.path.join(repo_root, ws, "Cargo.toml")
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1",
         "--manifest-path", manifest],
        capture_output=True, check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        sys.stderr.write(
            f"cascade_universe: cargo metadata failed for {manifest}\n"
            f"{out.stderr.decode('utf-8', 'replace')}\n"
        )
        return None
    return json.loads(out.stdout)


# The dependency names each crate needs ON THE REGISTRY before it can be published (see above).
DEPS = {}


def must_precede(dep):
    if dep.get("kind") == "dev":
        return dep.get("req", "*") != "*"
    return True  # normal (kind null) and build


def rows(repo_root):
    seen = {}
    DEPS.clear()
    for ws in WORKSPACES:
        doc = metadata(repo_root, ws)
        if doc is None:
            return None
        ws_root = os.path.normpath(os.path.join(os.path.abspath(repo_root), ws))
        for pkg in doc["packages"]:
            if pkg.get("publish") == []:  # `publish = false`
                continue
            # A name appearing in two workspaces would make "which version is
            # the release" ambiguous. Say so rather than picking one.
            if pkg["name"] in seen and seen[pkg["name"]][1] != pkg["manifest_path"]:
                sys.stderr.write(
                    f"cascade_universe: `{pkg['name']}` is publishable from two "
                    f"workspaces:\n  {seen[pkg['name']][1]}\n  {pkg['manifest_path']}\n"
                )
                return None
            seen[pkg["name"]] = (pkg["version"], pkg["manifest_path"], ws_root)
            DEPS[pkg["name"]] = {d["name"] for d in pkg.get("dependencies", []) if must_precede(d)}
    return sorted((n, v, m, w) for n, (v, m, w) in seen.items())


def publish_order(got, repo_root):
    """got: rows(). -> the same rows in a topological publish order, or None (exit 2) on a cycle."""
    import heapq
    names = {r[0] for r in got}
    by = {r[0]: r for r in got}
    root = os.path.normpath(os.path.abspath(repo_root))
    rank = {n: (0 if os.path.normpath(by[n][3]) == root else 1) for n in names}
    for n in names:
        bad = sorted(d for d in DEPS.get(n, ()) if d in names and rank[d] > rank[n])
        if bad:
            sys.stderr.write(f"cascade_universe: root-workspace crate `{n}` depends on excluded-workspace crate(s) {bad}; "
                             f"the facades must stay leaves of the publish order\n")
            return None
    pending = {n: {d for d in DEPS.get(n, ()) if d in names and d != n} for n in names}
    users = {n: set() for n in names}
    for n, ds in pending.items():
        for d in ds:
            users[d].add(n)
    heap = [(rank[n], n) for n in names if not pending[n]]
    heapq.heapify(heap)
    out = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(by[n])
        for u in sorted(users[n]):
            pending[u].discard(n)
            if not pending[u]:
                heapq.heappush(heap, (rank[u], u))
    if len(out) != len(names):
        stuck = sorted(n for n in names if pending[n])
        sys.stderr.write(f"cascade_universe: the publish graph has a CYCLE; no order exists. On it or behind it: {stuck}\n")
        return None
    return out


def main(argv):
    names_only = "--names" in argv[1:]
    ordered = "--order" in argv[1:]
    edges = "--edges" in argv[1:]
    args = [a for a in argv[1:] if not a.startswith("--")]
    repo_root = args[0] if args else "."

    got = rows(repo_root)
    if got is None:
        return 2
    if len(got) < MIN_CRATES:
        sys.stderr.write(
            f"cascade_universe: enumerated only {len(got)} publishable crate(s), "
            f"expected at least {MIN_CRATES}. The ENUMERATION is broken, not the repo.\n"
        )
        return 2
    if edges:
        names = {r[0] for r in got}
        for n in sorted(names):
            for d in sorted(DEPS.get(n, ())):
                if d in names and d != n:
                    print(f"{n}\t{d}")
        return 0
    if ordered:
        got = publish_order(got, repo_root)
        if got is None:
            return 2

    for name, version, manifest, ws_root in got:
        if names_only:
            print(name)
        else:
            print(f"{name}\t{version}\t{manifest}\t{ws_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
