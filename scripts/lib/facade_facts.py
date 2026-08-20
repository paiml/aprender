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
  R3 VERSION PIN  each facade's dependency on its upstream requires exactly the
                  aprender workspace version. The facade manifests are OUTSIDE
                  the workspace, so `cargo set-version` does not reach them:
                  without this check a release bumps the workspace and leaves
                  the facades pinned to a version that no longer exists.
  R4 NON-VACUITY  the facade fronting `aprender-contracts` carries at least
                  MIN_CORPUS example targets. A gate that passes on n=0 is a
                  fail mode; if the compat corpus is unwired this says so
                  instead of reporting a clean run over nothing.

Two read-only accessors share this file so the guard never has to embed a
`python3 -c '…'` one-liner: bashrs parses the quoting inside one as shell and
reports a spurious SC1078, and scripts/shell_lint_baseline.txt is a shrink-only
count, so a false positive is as blocking as a real one.

Usage:
    facade_facts.py <root-metadata.json> <facade-metadata.json> [min_corpus]
    facade_facts.py --target-dir <root-metadata.json>
    facade_facts.py --version-of <root-metadata.json> <package-name>

Prints one `ok`/`FAIL` row per property and exits non-zero if any FAILed.
"""

import json
import sys

# facade package -> (upstream package, expected target name, target kind)
FACADES = {
    "provable-contracts": ("aprender-contracts", "provable_contracts", "lib"),
    "provable-contracts-macros": (
        "aprender-contracts-macros",
        "provable_contracts_macros",
        "lib",
    ),
    "provable-contracts-cli": ("aprender-contracts-cli", "pv", "bin"),
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


def main(argv):
    if len(argv) >= 3 and argv[1] == "--target-dir":
        print(json.load(open(argv[2], encoding="utf-8"))["target_directory"])
        return 0
    if len(argv) >= 4 and argv[1] == "--version-of":
        doc = json.load(open(argv[2], encoding="utf-8"))
        for pkg in doc.get("packages", []):
            if pkg["name"] == argv[3]:
                print(pkg["version"])
                return 0
        return 1
    if len(argv) < 3:
        print(__doc__)
        return 2
    root = json.load(open(argv[1], encoding="utf-8"))
    facade = json.load(open(argv[2], encoding="utf-8"))
    min_corpus = int(argv[3]) if len(argv) > 3 else DEFAULT_MIN_CORPUS

    rp, fp = packages(root), packages(facade)
    fails = 0

    def row(ok, text):
        nonlocal fails
        if ok:
            print(f"ok    {text}")
        else:
            print(f"FAIL  {text}")
            fails += 1

    # R3 needs the workspace version. Take it from the upstream package cargo
    # actually resolved rather than from `[workspace.package]` text.
    for name, (upstream, want_target, kind) in sorted(FACADES.items()):
        if name not in fp:
            row(False, f"R0 {name}: absent from the facade workspace")
            continue
        pkg = fp[name]

        # R1
        got = target_names(pkg, kind)
        row(
            want_target in got,
            f"R1 {name}: {kind} target is {got or '[none]'}, "
            f"must include `{want_target}` to front {upstream}",
        )

        # R2
        row(
            name not in rp,
            f"R2 {name}: must NOT be an aprender workspace member "
            f"(rlib/bin name collision; rust-lang/cargo#6313)",
        )

        # R3
        if upstream not in rp:
            row(False, f"R3 {name}: upstream {upstream} not found in the aprender workspace")
        else:
            want_req = f"^{rp[upstream]['version']}"
            reqs = [
                d["req"]
                for d in pkg.get("dependencies", [])
                if d.get("name") == upstream
            ]
            row(
                reqs == [want_req],
                f"R3 {name}: requires {upstream} {reqs or '[none]'}, "
                f"workspace is at {rp[upstream]['version']} (want {want_req})",
            )

    # R4
    if "provable-contracts" in fp:
        n = len(target_names(fp["provable-contracts"], "example"))
        row(
            n >= min_corpus,
            f"R4 provable-contracts: {n} compat example target(s) wired, "
            f"minimum {min_corpus} — a corpus of nothing passes vacuously",
        )
    else:
        row(False, "R4 provable-contracts absent; corpus size unknown")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
