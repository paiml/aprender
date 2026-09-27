#!/usr/bin/env python3
"""ladder_meter.py — run one ladder command and record what it cost the host (#4520 step 4).

Usage:  ladder_meter.py <verb> -- <command> [args...]

Runs <command> with this process's stdin/stdout/stderr, waits for it with wait4(2), and
appends ONE JSON line to $LADDER_METER:
  {"cell": $LADDER_METER_CELL, "file_bytes": $LADDER_METER_FILE_BYTES, "verb": <verb>,
   "rc": <exit>, "bytes_read": ru_inblock*512, "peak_rss_bytes": ru_maxrss*1024, "wall_s": <s>}

bytes_read is STORAGE bytes (the kernel's per-task read_bytes, summed over the waited-for tree):
what the ladder pulled off the disk, which is what took lambda down. A read served from the page
cache costs the host nothing and is not counted -- by design, not by accident.
peak_rss_bytes is the largest RSS of any one process in the tree (getrusage semantics).

SIGTERM/SIGINT/SIGHUP are forwarded to the child, which is still waited for and recorded: the
serve probe stops its server by signalling this wrapper, and a server's cost is still a cost.

Exit: the child's status (128+N for a signal) · 2 usage, nothing ran. No $LADDER_METER: runs
the command unrecorded (a hand call outside the ladder).
"""
import json
import os
import signal
import subprocess
import sys
import time


def main(argv):
    if len(argv) < 4 or argv[2] != "--":
        print("ladder_meter: usage: ladder_meter.py <verb> -- <command> [args...]", file=sys.stderr)
        return 2
    verb, cmd = argv[1], argv[3:]
    t0 = time.monotonic()
    try:
        child = subprocess.Popen(cmd)
    except OSError as e:
        print(f"ladder_meter: cannot run {cmd[0]!r}: {e}", file=sys.stderr)
        return 127
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda n, _f: child.send_signal(n))
    while True:
        try:
            _, status, ru = os.wait4(child.pid, 0)
            break
        except InterruptedError:
            continue
    wall = time.monotonic() - t0
    rc = os.waitstatus_to_exitcode(status)
    rc = 128 - rc if rc < 0 else rc
    out = os.environ.get("LADDER_METER")
    if out:
        fb = os.environ.get("LADDER_METER_FILE_BYTES", "")
        rec = {"cell": os.environ.get("LADDER_METER_CELL") or None,
               "file_bytes": int(fb) if fb.isdigit() else None,
               "verb": verb, "rc": rc,
               "bytes_read": ru.ru_inblock * 512,
               "peak_rss_bytes": ru.ru_maxrss * 1024,
               "wall_s": round(wall, 2)}
        with open(out, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
