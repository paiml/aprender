#!/usr/bin/env python3
"""Every `[[bin]] name` in this repository, across ALL of its workspaces.

WHY THIS EXISTS (aprender#2558)
-------------------------------
Nothing detected two crates declaring the same binary name. Measured on
crates.io 2026-08-21, FOUR things claimed the name `pv` — the crates.io pipe
viewer, `/usr/bin/pv`, `aprender-contracts-cli`, and the `provable-contracts-cli`
facade — and this repository owned two of them. All four write
`~/.cargo/bin/pv`, and `cargo install` overwrites it with no warning. A shadowed
binary is worse than a missing one: edits look effective and change nothing.

Consumes one or more `cargo metadata --no-deps --format-version 1` documents,
each labelled with the workspace it came from, and reports every bin name
claimed by more than one package.

  W1 COVERAGE     at least two workspace documents. crates/facades is
                  `exclude`d from the root workspace, so a scan of the root
                  workspace ALONE cannot see the facade — inert by
                  construction, which is the exact pre-filter defect that made
                  a whole guard class dead.
  W2 FACADE SEEN  a `provable-contracts*` package is present in the scan. W1
                  counts documents; W2 proves the second document is the one
                  that matters. Without it, passing two copies of the root
                  metadata satisfies W1 and checks nothing new.
  W3 NON-VACUITY  the scan found at least MIN_BINS bin targets. A metadata read
                  that silently returned nothing would report zero duplicates
                  and read exactly like a clean tree.
  D  DUPLICATES   every bin name claimed by 2+ packages must appear in the
                  allowlist with EXACTLY that set of packages. `apr` is
                  declared twice on purpose (crates/apr-cli's src/main.rs and
                  the root `aprender` facade's src/bin/apr.rs), so intent is
                  modelled rather than special-cased: an undeclared duplicate
                  FAILS, a declared one passes, and a THIRD claimant appearing
                  on a declared name FAILS because the sets no longer match.
  A  STALE        an allowlist entry whose duplicate no longer exists FAILS.
                  An allowlist that only ever grows stops describing the tree.

Allowlist format, one entry per line, `#` comments and blanks ignored:

    <bin-name> <package> <package> [...]   # reason

Usage:
    bin_names.py <allowlist> <label>=<metadata.json> [<label>=<metadata.json> ...]
    bin_names.py --list <label>=<metadata.json> [...]

Prints one `ok`/`FAIL` row per property and exits non-zero if any FAILed.
"""

import json
import sys

# Floor for W3. The root workspace declares dozens of bins; a scan returning
# fewer than this is broken, not clean. Deliberately far below the true count
# so it never becomes a maintenance tax — it exists to catch ZERO, not to track
# the number.
MIN_BINS = 5


def load(spec):
    """`label=path` -> (label, parsed document)."""
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"bin_names.py: expected <label>=<metadata.json>, got {spec!r}")
    with open(path, encoding="utf-8") as fh:
        return label, json.load(fh)


def bins_in(doc):
    """[(bin name, package name)] for every bin target in one metadata doc."""
    out = []
    for pkg in doc.get("packages", []):
        for tgt in pkg.get("targets", []):
            if "bin" in tgt.get("kind", []):
                out.append((tgt["name"], pkg["name"]))
    return out


def read_allowlist(path):
    """bin name -> (frozenset of packages, reason)."""
    entries = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line, _, reason = raw.partition("#")
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise SystemExit(
                    f"{path}: an allowlist entry needs a bin name and 2+ packages: {raw!r}"
                )
            entries[fields[0]] = (frozenset(fields[1:]), reason.strip())
    return entries


class Report:
    """Accumulating ok/FAIL printer. A class rather than a closure so each rule
    below is a small top-level function the complexity gate can see separately —
    one 25-branch `main` was over the threshold and, more to the point, unreadable.
    """

    def __init__(self):
        self.fails = 0

    def row(self, ok, text):
        if ok:
            print(f"ok    {text}")
        else:
            print(f"FAIL  {text}")
            self.fails += 1


def claims_in(docs):
    """bin name -> {packages declaring it}, merged across every document."""
    claims = {}
    for _label, doc in docs:
        for name, pkg in bins_in(doc):
            claims.setdefault(name, set()).add(pkg)
    return claims


def check_coverage(rep, docs, claims):
    """W1/W2/W3 — the scan must be wide enough and non-empty to mean anything."""
    rep.row(
        len(docs) >= 2,
        f"W1 {len(docs)} workspace document(s) scanned, need 2+ — crates/facades is "
        f"`exclude`d from the root workspace and is invisible to a root-only scan",
    )
    all_pkgs = {p["name"] for _label, doc in docs for p in doc.get("packages", [])}
    facade_pkgs = sorted(n for n in all_pkgs if n.startswith("provable-contracts"))
    rep.row(
        bool(facade_pkgs),
        f"W2 facade workspace present in the scan: {facade_pkgs or '[NONE]'} — W1 counts "
        f"documents, this proves the second one is the excluded workspace",
    )
    total = sum(len(v) for v in claims.values())
    rep.row(
        total >= MIN_BINS,
        f"W3 {total} bin target(s) found across {len(docs)} workspace(s), "
        f"minimum {MIN_BINS} — a scan of nothing reports no duplicates",
    )


def undeclared_row(name, pkgs, listed, allow_path):
    return (
        f"D  `{name}` is declared by {len(pkgs)} packages ({listed}) and is NOT "
        f"declared intentional. `cargo install` writes ~/.cargo/bin/{name} for "
        f"each of them and overwrites without warning. Either rename one, or "
        f"add a line to {allow_path} saying why both must keep the name."
    )


def check_duplicates(rep, dups, allow, allow_path):
    """D — every duplicate must be declared, with the SAME claimant set."""
    if not dups:
        print("ok    D  no bin name is claimed by two packages")
    for name in sorted(dups):
        pkgs = dups[name]
        listed = ", ".join(sorted(pkgs))
        if name not in allow:
            rep.row(False, undeclared_row(name, pkgs, listed, allow_path))
            continue
        want, reason = allow[name]
        matched = want == pkgs
        tail = f" — {reason}" if matched else " — the claimant set CHANGED"
        rep.row(
            matched,
            f"D  `{name}` is claimed by {{{listed}}}, declared intentional for "
            f"{{{', '.join(sorted(want))}}}{tail}",
        )


def check_stale(rep, dups, allow, claims, allow_path):
    """A — an allowlist that only ever grows stops describing the tree."""
    for name in sorted(allow):
        if name in dups:
            continue
        now = sorted(claims.get(name, [])) or "nothing"
        rep.row(
            False,
            f"A  {allow_path} declares `{name}` as an intentional duplicate, but it is "
            f"no longer claimed by two packages (now: {now}). Remove the entry — a "
            f"stale allowlist stops describing the tree.",
        )


def list_bins(specs):
    """`--list`: every bin target, one `name<TAB>package<TAB>workspace` per line."""
    for spec in specs:
        label, doc = load(spec)
        for name, pkg in sorted(bins_in(doc)):
            print(f"{name}\t{pkg}\t{label}")
    return 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "--list":
        return list_bins(argv[2:])
    # Deliberately accepts a SINGLE document: W1 must be able to FAIL on a
    # one-workspace scan rather than exiting 2 on a usage error, which would be
    # indistinguishable from a broken invocation.
    if len(argv) < 3:
        print(__doc__)
        return 2

    allow_path = argv[1]
    allow = read_allowlist(allow_path)
    docs = [load(spec) for spec in argv[2:]]
    claims = claims_in(docs)
    dups = {n: pkgs for n, pkgs in claims.items() if len(pkgs) > 1}

    rep = Report()
    check_coverage(rep, docs, claims)
    check_duplicates(rep, dups, allow, allow_path)
    check_stale(rep, dups, allow, claims, allow_path)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
