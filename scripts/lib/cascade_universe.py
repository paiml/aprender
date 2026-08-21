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


def rows(repo_root):
    seen = {}
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
    return sorted((n, v, m, w) for n, (v, m, w) in seen.items())


def main(argv):
    names_only = "--names" in argv[1:]
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

    for name, version, manifest, ws_root in got:
        if names_only:
            print(name)
        else:
            print(f"{name}\t{version}\t{manifest}\t{ws_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
