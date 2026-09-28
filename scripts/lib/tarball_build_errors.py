#!/usr/bin/env python3
"""tarball_build_errors.py LOG: attribute a tarball-workspace build's errors to the crates that own them (#4114).

Reads `cargo build --tests --keep-going --message-format short` output. Prints one block per failing crate:
  RED   <crate> (<target>): N error(s)
          <file>:<line>:<col>: error: <message>
Errors are matched to crates through their `pkgs/<name>-<version>/` path, and "could not compile"
lines through the crate name cargo prints. Exits 1 when an error is attributed to a crate, 0 when
there is no error at all, and 3 when the only errors belong to no crate (cargo could not even start:
a bad argument or an unresolvable workspace), and 4 when the BUILD HOST failed (disk full, a write
failure, a killed compiler): then no line of the log is evidence about any crate. Measured: gx10 at
ENOSPC turned every crate's compile into "could not compile" and the first draft reported RED. The
caller reports 3 and 4 as "could not check", never as a defect of a crate. It exits 2 when the log
is missing or empty.
"""
import re
import sys
from collections import OrderedDict

DIAG = re.compile(r"^(?:.*/)?pkgs/([^/]+)/(\S+?):(\d+):(\d+): error(?:\[\w+\])?: (.*)$")
FAILED = re.compile(r"^error: could not compile `([^`]+)` \(([^)]+)\)")
HOST = re.compile(r"No space left on device|os error 28|failed to write `|Disk quota exceeded|"
                  r"signal: 9, SIGKILL|\(signal: 9\)|Cannot allocate memory")


def main(argv):
    if len(argv) != 2:
        print("usage: tarball_build_errors.py LOG", file=sys.stderr)
        return 2
    try:
        lines = open(argv[1], encoding="utf-8", errors="replace").read().splitlines()
    except OSError as e:
        print("tarball_build_errors: cannot read %s: %s" % (argv[1], e), file=sys.stderr)
        return 2
    if not lines:
        print("tarball_build_errors: empty build log", file=sys.stderr)
        return 2
    host = [ln for ln in lines if HOST.search(ln)]
    if host:
        print("HOST  the build host failed, so no crate verdict is possible (%d line(s)):" % len(host))
        for ln in host[:5]:
            print("        %s" % ln)
        return 4
    by_dir, failed = OrderedDict(), OrderedDict()
    other = []
    for ln in lines:
        m = DIAG.match(ln)
        if m:
            by_dir.setdefault(m.group(1), []).append("%s:%s:%s: error: %s" % (m.group(2), m.group(3), m.group(4), m.group(5)))
            continue
        f = FAILED.match(ln)
        if f:
            failed.setdefault(f.group(1), []).append(f.group(2))
        elif ln.startswith("error"):
            other.append(ln)
    if not by_dir and not failed and not other:
        return 0
    for d, errs in by_dir.items():
        # the dir is <name>-<version>; a version may itself hold '-' (0.70.0-rc.1), so match the
        # longest crate name cargo reported that prefixes the dir, never a split on the last '-'
        name = max((n for n in failed if d.startswith(n + "-")), key=len, default=None)
        targets = ", ".join(failed.pop(name, [])) if name else "?"
        print("RED   %s (%s): %d error(s) in the published tarball" % (d, targets, len(errs)))
        for e in errs[:20]:
            print("        %s" % e)
    for name, targets in failed.items():
        print("RED   %s (%s): could not compile (no diagnostic attributed to a tarball path)" % (name, ", ".join(targets)))
    for ln in other[:10]:
        print("ERROR %s" % ln)
    return 1 if (by_dir or failed) else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
