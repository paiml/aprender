#!/usr/bin/env python3
"""Print the explicit `-p <crate>` list that scopes a `cargo llvm-cov report` (#4023).

`cargo llvm-cov report` takes its package scope from the CURRENT package. This repo's root
Cargo.toml is also a package (the `apr` facade), so an unscoped report covers only the facade
and comes out empty: measured twice (the Makefile's single-phase note, and coverage-nightly
run 35892421393). The older cargo-llvm-cov rejected `report --workspace`, and 0.9.0 rejects
`report --exclude`. An explicit `-p` list works on both, so every report is scoped with it.

The list is DERIVED from `cargo metadata --no-deps` (the workspace members), never kept by
hand, so a new crate is reported the day it is added.

Usage: coverage_report_scope.py [--exclude NAME ...]
"""
import json
import subprocess
import sys


def main(argv):
    exclude = set()
    args = iter(argv[1:])
    for a in args:
        if a == "--exclude":
            exclude.add(next(args, ""))
        else:
            sys.exit(__doc__)
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    names = sorted(p["name"] for p in meta["packages"] if p["name"] not in exclude)
    unknown = exclude - {p["name"] for p in meta["packages"]}
    if unknown:
        sys.exit(f"coverage_report_scope: --exclude names no workspace member: {sorted(unknown)}")
    if not names:
        sys.exit("coverage_report_scope: no workspace members left to report")
    print(" ".join(f"-p {n}" for n in names))


if __name__ == "__main__":
    main(sys.argv)
