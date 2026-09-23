"""lock_contained.py -- is a release cut's Cargo.lock change CONTAINED in main? (preflight R4, 0.69.1)

argv: <lock at merge-base> <lock at the cut> <lock at main>   (file paths)
stdout: one reason per line the cut's change is NOT on main; nothing when it is. exit 0 contained, 1 not,
2 unreadable.

main's merge queue squashes, and main moves on after a cut (a dedupe renames `zune-core 0.5.1` to
`zune-core`, a textual change with no dependency meaning), so the lock is compared SEMANTICALLY, through
the canonical dependency delta ladder_equiv.lock_delta already computes (never a second canonicaliser):
every package the cut adds is on main, every package it removes is gone from main, every dependency
edge it adds/removes is added/removed on main, and every external version/source/checksum the cut
moved to is on main. A main that REVERTED the change (back to the merge-base state) is not containing it.
"""
import sys
import tomllib

import ladder_equiv as E


def reasons(mb, cut, main):
    delta, _ = E.lock_delta(mb, cut)
    M, C = E._lock_shape(main), E._lock_shape(cut)
    out = []
    for p in delta["packages_added"]:
        if p not in M:
            out.append(f"the cut adds package {p}; main does not have it")
            continue
        main_attrs = {(e[0], e[1], e[2]) for e in M[p]}
        for a in {(e[0], e[1], e[2]) for e in C[p]} - main_attrs:
            out.append(f"the cut adds {p} {a[0]} ({(a[2] or '')[:12]}); main has it only at another version/source")
    for p in delta["packages_removed"]:
        if p in M:
            out.append(f"the cut removes package {p}; main still has it")
    for name, ch in sorted(delta["changed"].items()):
        if name not in M:
            out.append(f"the cut changes {name}; main has no {name} at all")
            continue
        edges = {d for entry in M[name] for d in entry[3]}
        for e in ch["edges_added"]:
            if e not in edges:
                out.append(f"the cut adds dependency {name} -> {e}; main does not have it")
        for e in ch["edges_removed"]:
            if e in edges:
                out.append(f"the cut removes dependency {name} -> {e}; main still has it")
        if ch["attrs"] is not None:
            main_attrs = {(e[0], e[1], e[2]) for e in M[name]}
            for a in {(e[0], e[1], e[2]) for e in C[name]} - main_attrs:
                out.append(f"the cut moves {name} to {a[0]} ({(a[2] or '')[:12]}); main does not have that")
    return out


def main():
    try:
        texts = [open(p, encoding="utf-8").read() for p in sys.argv[1:4]]
        rs = reasons(*texts)
    except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"Cargo.lock unreadable: {exc}")
        sys.exit(2)
    for r in rs:
        print(r)
    sys.exit(1 if rs else 0)


if __name__ == "__main__":
    main()
