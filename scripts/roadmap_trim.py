#!/usr/bin/env python3
"""roadmap_trim.py — collapse a `$PMAT work add`/`work complete` roadmap.yaml
re-serialisation back to the base's bytes, one command (PMAT-980, G-6, #2874).

    python3 scripts/roadmap_trim.py [--base <ref>] [--file <path>]

is exactly:

    python3 scripts/lib/roadmap_diff.py trim --base <ref> --file <path> --write

--base defaults to `git merge-base origin/main HEAD` (the driver's own
convention in scripts/check_roadmap_diff_additive.sh) and --file defaults to
docs/roadmaps/roadmap.yaml.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lib"))

import roadmap_diff  # noqa: E402  (path set up above)


def _default_base() -> str:
    try:
        proc = subprocess.run(
            ["git", "merge-base", "origin/main", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as e:
        raise roadmap_diff.UsageError(f"git merge-base failed to run: {e}") from e
    if proc.returncode != 0:
        raise roadmap_diff.UsageError(
            f"git merge-base origin/main HEAD failed: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadmap_trim.py")
    ap.add_argument("--base", default=None)
    ap.add_argument("--file", default="docs/roadmaps/roadmap.yaml")
    args = ap.parse_args(argv)

    try:
        base = args.base if args.base is not None else _default_base()
        return roadmap_diff.main(
            ["trim", "--base", base, "--file", args.file, "--write"]
        )
    except roadmap_diff.UsageError as e:
        sys.stderr.write(f"USAGE ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
