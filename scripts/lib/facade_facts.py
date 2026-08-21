#!/usr/bin/env python3
"""Structural facts about the crates.io compatibility facades.

Consumes two `cargo metadata --no-deps --format-version 1` documents — the
aprender workspace and the facade workspace — and checks the four structural
properties a facade must hold. Everything here is read from cargo's own
resolution, never from a hand-rolled TOML parse, so it cannot disagree with
what cargo will actually build or publish.

  R1 LIB NAME     each facade's lib/bin target name equals the target name of
                  the crate it fronts. A facade whose lib is called anything
                  else does not answer `use provable_contracts::…`, which is
                  the entire reason it exists.
  R2 NOT A MEMBER no facade appears in the aprender workspace. Two primary
                  packages sharing one lib name collide on the uplifted rlib
                  ("output filename collision … may become a hard error",
                  rust-lang/cargo#6313), and the facade `pv` would overwrite
                  the real `pv` in the shared target dir.
  R3 VERSION PIN  each RE-EXPORTING facade's dependency on its upstream
                  requires exactly the aprender workspace version. The facade
                  manifests are OUTSIDE the workspace, so `cargo set-version`
                  does not reach them: without this check a release bumps the
                  workspace and leaves the facades pinned to a version that no
                  longer exists.
                  For a SIGNPOST facade the rule INVERTS: it must have NO
                  dependency on the crate it fronts. See R7.
  R4 NON-VACUITY  the facade fronting `aprender-contracts` carries at least
                  MIN_CORPUS example targets. A gate that passes on n=0 is a
                  fail mode; if the compat corpus is unwired this says so
                  instead of reporting a clean run over nothing.
  R7 NO BINARIES  no facade declares a `[[bin]]` target. aprender#2558: FOUR
                  crates declared a bin named `pv` (this repo owned two of
                  them) and `cargo install` overwrites ~/.cargo/bin/pv without
                  warning. The facades yield the name; `aprender-contracts-cli`
                  keeps it. Enforced here AND, across both workspaces, by
                  scripts/check_duplicate_bin_names.sh.

The read-only accessors share this file so the guard never has to embed a
`python3 -c '…'` one-liner: bashrs parses the quoting inside one as shell and
reports a spurious SC1078, and scripts/shell_lint_baseline.txt is a shrink-only
count, so a false positive is as blocking as a real one.

Usage:
    facade_facts.py <root-metadata.json> <facade-metadata.json> [min_corpus]
    facade_facts.py --target-dir <root-metadata.json>
    facade_facts.py --version-of <root-metadata.json> <package-name>
    facade_facts.py --has-bin <metadata.json> <package-name> <bin-name>

Prints one `ok`/`FAIL` row per property and exits non-zero if any FAILed.
"""

import json
import sys

# facade package -> (crate it fronts, expected target name, target kind, kind of facade)
#
# RE-EXPORT facades depend on their upstream and forward its surface, so R3
# pins the version. The SIGNPOST facade forwards nothing: it is a lib-only
# tombstone that names the replacement, so R3 inverts to "must NOT depend".
#
# provable-contracts-cli became a signpost in aprender#2558. Two reasons, and
# either alone would have been sufficient:
#   1. NAME COLLISION. Four crates declared a bin named `pv`; the CLI facade is
#      463 downloads against the library's 57K, and it is the only one of the
#      three that collides. It yields.
#   2. IT COULD NOT COMPILE WHEN PUBLISHED. Its main.rs called
#      `aprender_contracts_cli::run()`, but the published 0.63.0 is BIN-ONLY
#      (crates.io API: has_lib false) — a registry consumer got E0433. Dropping
#      the dependency dissolves that; a lib-only facade calls run() from
#      nowhere.
REEXPORT = "re-export"
SIGNPOST = "signpost"
FACADES = {
    "provable-contracts": (
        "aprender-contracts",
        "provable_contracts",
        "lib",
        REEXPORT,
    ),
    "provable-contracts-macros": (
        "aprender-contracts-macros",
        "provable_contracts_macros",
        "lib",
        REEXPORT,
    ),
    "provable-contracts-cli": (
        "aprender-contracts-cli",
        "provable_contracts_cli",
        "lib",
        SIGNPOST,
    ),
}

DEFAULT_MIN_CORPUS = 27


def packages(doc):
    return {p["name"]: p for p in doc.get("packages", [])}


def target_names(pkg, kind):
    out = []
    for t in pkg.get("targets", []):
        kinds = t.get("kind", [])
        if kind in kinds or (kind == "lib" and "proc-macro" in kinds):
            out.append(t["name"])
    return out


class Report:
    """Accumulating ok/FAIL printer. A class rather than a closure so each rule
    below is its own top-level function: `main` carried every rule inline and was
    over the complexity gate (cyclomatic 20, cognitive 90) BEFORE this change, so
    the refactor is a debt payment, not a workaround for the rows added here.
    """

    def __init__(self):
        self.fails = 0

    def row(self, ok, text):
        if ok:
            print(f"ok    {text}")
        else:
            print(f"FAIL  {text}")
            self.fails += 1


def accessor(argv):
    """The read-only `--flag` forms. Returns an exit code, or None if argv is not
    one of them."""
    if len(argv) >= 3 and argv[1] == "--target-dir":
        with open(argv[2], encoding="utf-8") as fh:
            print(json.load(fh)["target_directory"])
        return 0
    if len(argv) >= 5 and argv[1] == "--has-bin":
        # Exit 0 iff <package> declares a bin target named <bin-name>. Used to
        # prove the CLI facade's redirect points at a tool that exists rather
        # than at a name someone deleted (check_facade_compat.sh row C3).
        return has_bin(argv[2], argv[3], argv[4])
    if len(argv) >= 4 and argv[1] == "--version-of":
        return version_of(argv[2], argv[3])
    return None


def has_bin(path, package, binary):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    for pkg in doc.get("packages", []):
        if pkg["name"] == package and binary in target_names(pkg, "bin"):
            return 0
    return 1


def version_of(path, package):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    for pkg in doc.get("packages", []):
        if pkg["name"] == package:
            print(pkg["version"])
            return 0
    return 1


def check_r3(rep, name, pkg, upstream, shape, rp):
    """R3 — the upstream version pin, INVERTED for a signpost facade."""
    reqs = [d["req"] for d in pkg.get("dependencies", []) if d.get("name") == upstream]
    if shape == SIGNPOST:
        # Inverted on purpose. A signpost that depends on the crate it replaces
        # drags that whole crate into every consumer's build for nothing, and
        # re-couples its publishability to the cascade order — which is exactly
        # the E0433 that made the bin form unpublishable.
        rep.row(
            reqs == [],
            f"R3 {name}: is a SIGNPOST facade and must NOT depend on "
            f"{upstream}; found {reqs or '[none]'}",
        )
        return
    if upstream not in rp:
        rep.row(False, f"R3 {name}: upstream {upstream} not found in the aprender workspace")
        return
    want_req = f"^{rp[upstream]['version']}"
    rep.row(
        reqs == [want_req],
        f"R3 {name}: requires {upstream} {reqs or '[none]'}, "
        f"workspace is at {rp[upstream]['version']} (want {want_req})",
    )


def check_one_facade(rep, name, spec, rp, fp):
    """R0/R1/R2/R3/R7 for a single facade package."""
    upstream, want_target, kind, shape = spec
    if name not in fp:
        rep.row(False, f"R0 {name}: absent from the facade workspace")
        return
    pkg = fp[name]

    got = target_names(pkg, kind)
    rep.row(
        want_target in got,
        f"R1 {name}: {kind} target is {got or '[none]'}, "
        f"must include `{want_target}` to front {upstream}",
    )
    rep.row(
        name not in rp,
        f"R2 {name}: must NOT be an aprender workspace member "
        f"(rlib/bin name collision; rust-lang/cargo#6313)",
    )
    check_r3(rep, name, pkg, upstream, shape, rp)
    bins = target_names(pkg, "bin")
    rep.row(
        bins == [],
        f"R7 {name}: declares bin target(s) {bins or '[none]'}; no facade may "
        f"declare a binary (aprender#2558 — four crates claimed `pv`)",
    )


def check_corpus(rep, fp, min_corpus):
    """R4 — a gate that passes on n=0 is a fail mode."""
    if "provable-contracts" not in fp:
        rep.row(False, "R4 provable-contracts absent; corpus size unknown")
        return
    n = len(target_names(fp["provable-contracts"], "example"))
    rep.row(
        n >= min_corpus,
        f"R4 provable-contracts: {n} compat example target(s) wired, "
        f"minimum {min_corpus} — a corpus of nothing passes vacuously",
    )


def main(argv):
    code = accessor(argv)
    if code is not None:
        return code
    if len(argv) < 3:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        root = json.load(fh)
    with open(argv[2], encoding="utf-8") as fh:
        facade = json.load(fh)
    min_corpus = int(argv[3]) if len(argv) > 3 else DEFAULT_MIN_CORPUS

    rp, fp = packages(root), packages(facade)
    rep = Report()
    # R3 needs the workspace version. Take it from the upstream package cargo
    # actually resolved rather than from `[workspace.package]` text.
    for name, spec in sorted(FACADES.items()):
        check_one_facade(rep, name, spec, rp, fp)
    check_corpus(rep, fp, min_corpus)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
