#!/usr/bin/env python3
"""Emit the resolved transitive dependency closure of a workspace crate.

Helper for ``scripts/check_format_sovereignty.sh`` (issue #2231). Reads
``cargo metadata --all-features`` JSON on stdin and prints, one per line, the
names of every crate in the transitive normal-dependency closure of the crate
named in argv[1]. Kept as a separate file (not an inline heredoc) so the shell
guard lints cleanly under ``bashrs`` — embedded Python confuses the shell linter.
"""
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: format_sovereignty_closure.py <crate-name>\n")
        return 2
    crate = sys.argv[1]
    data = json.load(sys.stdin)
    id_to_name = {p["id"]: p["name"] for p in data["packages"]}
    nodes = {n["id"]: n for n in data["resolve"]["nodes"]}
    roots = [p["id"] for p in data["packages"] if p["name"] == crate]
    if not roots:
        sys.stderr.write("crate not found in workspace: %s\n" % crate)
        return 3
    seen: set = set()
    stack = [roots[0]]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes.get(node_id)
        if node:
            for dep in node["deps"]:
                stack.append(dep["pkg"])
    for node_id in sorted(seen):
        print(id_to_name[node_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
