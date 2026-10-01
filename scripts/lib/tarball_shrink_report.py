#!/usr/bin/env python3
"""tarball_shrink_report.py PACKAGE_LOG WS_DIR: what the published tarballs do NOT test (#4114, #4129, #4130).

A green tarball build must never hide a shrinking test surface. Two mechanisms shrink it, and both
are counted here instead of disappearing in silence:
  NOT SHIPPED   integration test targets cargo DROPS from a crate because their file is excluded,
                read from cargo's own warning in PACKAGE_LOG, per crate:
                  warning: ignoring test `x` as `tests/x.rs` is not included in the published package
  SKIP SITES    call sites of a `*_or_skip(` helper in the shipped sources under WS_DIR/pkgs. This is
                the #4048/#4129/#4130 pattern: a test that reads a workspace file at run time and
                SKIPs by name out of tree. The build compiles and does not run, so it cannot see a
                run-time SKIP. It counts every place one can happen.
Prints one line per crate with a nonzero count, then the totals. Always exits 0: this is a report,
and the verdict is the build's. Exits 2 when an input is missing, because a report over nothing
would read as "nothing shrank".
"""
import pathlib
import re
import sys
from collections import OrderedDict

PACKAGING = re.compile(r"^\s+Packaging (\S+) v(\S+)")
IGNORED = re.compile(r"^warning: ignoring test `[^`]+` as `([^`]+)` is not included in the published package")
SKIP_CALL = re.compile(r"\b\w+_or_skip\(")
SKIP_DEF = re.compile(r"\bfn\s+\w+_or_skip\s*\(")


def main(argv):
    if len(argv) != 3:
        print("usage: tarball_shrink_report.py PACKAGE_LOG WS_DIR", file=sys.stderr)
        return 2
    log, ws = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    if not log.is_file() or not (ws / "pkgs").is_dir():
        print("tarball_shrink_report: missing %s or %s/pkgs" % (log, ws), file=sys.stderr)
        return 2
    not_shipped, crate = OrderedDict(), None
    for ln in log.read_text(errors="replace").splitlines():
        m = PACKAGING.match(ln)
        if m:
            crate = m.group(1)
            continue
        m = IGNORED.match(ln)
        if m:
            not_shipped.setdefault(crate or "?", []).append(m.group(1))
    skips = OrderedDict()
    for d in sorted((ws / "pkgs").iterdir()):
        n = 0
        for f in d.rglob("*.rs"):
            for line in f.read_text(errors="replace").splitlines():
                if SKIP_CALL.search(line) and not SKIP_DEF.search(line):
                    n += 1
        if n:
            skips[d.name] = n
    for c, files in not_shipped.items():
        shown = ", ".join(files[:5]) + (", ..." if len(files) > 5 else "")
        print("NOT SHIPPED  %-32s %3d integration test target(s): %s" % (c, len(files), shown))
    for d, n in skips.items():
        print("SKIP SITES   %-32s %3d run-time *_or_skip( call site(s) in shipped code" % (d, n))
    print("SHRINK: %d integration test target(s) not shipped across %d crate(s); %d run-time skip site(s) across %d crate(s)"
          % (sum(len(v) for v in not_shipped.values()), len(not_shipped), sum(skips.values()), len(skips)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
