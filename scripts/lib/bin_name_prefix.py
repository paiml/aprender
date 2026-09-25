#!/usr/bin/env python3
"""Every shipped [[bin]] is named aprender-* (#4430). See check_bin_names_aprender.sh.

    bin_name_prefix.py PENDING_FOLD label=metadata.json ...   # check: 0 ok, 1 red, 2 ENV
    bin_name_prefix.py --list label=metadata.json ...         # the shipped bin set, one per line
"""
import json
import re
import sys

EXEMPT = {"apr", "pv"}  # the two user-facing names (apr-mono-binary-rule-v1)
NAME = re.compile(r"^aprender-[a-z0-9]+(-[a-z0-9]+)*$")


def load(args):
    """[(label, [(bin, package, manifest)])] from label=path metadata documents."""
    docs = []
    for a in args:
        label, sep, path = a.partition("=")
        if not sep:
            raise ValueError(f"not label=path: {a}")
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        docs.append((label, [(t["name"], p["name"], p["manifest_path"])
                             for p in meta["packages"] for t in p["targets"]
                             if "bin" in t["kind"]]))
    return docs


def read_pending(path):
    """{bin: package} from `<bin> <package>  # why` rows."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].split()
            if not line:
                continue
            if len(line) != 2:
                raise ValueError(f"{path}:{n}: want `<bin> <package>`, got {line}")
            rows[line[0]] = line[1]
    return rows


def check(pending_path, args):
    try:
        pending = read_pending(pending_path)
        docs = load(args)
    except (OSError, ValueError, KeyError) as e:
        print(f"ENV: {e}")
        return 2
    bad = 0
    labels = [d[0] for d in docs]
    bins = [b for _, bs in docs for b in bs]
    # Vacuity: both workspaces, and bins in them -- the facades are `exclude`d
    # from the root workspace, so a root-only scan cannot see their [[bin]]s.
    if "facades" not in labels:
        print(f"W1 facade workspace not in the scan (scanned: {', '.join(labels) or 'none'})")
        bad = 1
    if not any(b[0] == "apr" for b in bins):
        print(f"W2 {len(bins)} bin target(s) found and none is `apr`: the scan is not this tree")
        bad = 1
    seen = {}
    for name, pkg, manifest in bins:
        seen.setdefault(name, set()).add(pkg)
        if name in EXEMPT or NAME.match(name):
            continue
        if pending.get(name) == pkg:
            print(f"ok   `{name}` ({pkg}) is pending fold into `apr`, not a rename")
            continue
        print(f"N  `{name}` ({pkg}, {manifest}) is not aprender-*: rename it, "
              f"or list it in the pending-fold file if it is being folded into apr")
        bad = 1
    for name, pkg in sorted(pending.items()):
        if pkg not in seen.get(name, set()):
            print(f"P  pending-fold row `{name} {pkg}` names no bin in the tree: "
                  f"the fold landed or the row is wrong -- delete the row")
            bad = 1
    if not bad:
        print(f"ok   {len(set(seen))} bin name(s) across {len(docs)} workspace(s): "
              f"every one is aprender-*, apr/pv, or pending fold ({len(pending)})")
    return bad


def main(argv):
    if argv[:1] == ["--list"]:
        try:
            names = sorted({b[0] for _, bs in load(argv[1:]) for b in bs})
        except (OSError, ValueError, KeyError) as e:
            print(f"ENV: {e}", file=sys.stderr)
            return 2
        print("\n".join(names))
        return 0 if names else 2
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    return check(argv[0], argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
