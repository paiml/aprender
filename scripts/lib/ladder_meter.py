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

RSS CAP. With $LADDER_METER_RSS_FACTOR, $LADDER_METER_RSS_SLACK, $LADDER_METER_RSS_KILL and a file
size, the cap is kill x (factor x file_bytes + slack) -- a multiple of the judge's own limit, so a
call a little over budget finishes (and is judged RED), while a runaway is stopped. The tree is
polled every POLL_S and, the first time any one process's RSS passes the cap, the whole tree gets
SIGKILL and the record carries "rss_capped": <bytes seen>. Overshoot is bounded by one poll of
growth; the box's MemoryMax (ladder_box.sh) is the backstop for what a poll can miss. Without the
cap a call over its budget ran on until the box OOM-killed it, an hour later, receipt and all.

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

POLL_S = 0.25


def rss_cap():
    try:
        fb = int(os.environ["LADDER_METER_FILE_BYTES"])
        budget = float(os.environ["LADDER_METER_RSS_FACTOR"]) * fb + int(os.environ["LADDER_METER_RSS_SLACK"])
        return int(float(os.environ["LADDER_METER_RSS_KILL"]) * budget)
    except (KeyError, ValueError):
        return None


def tree(root):
    """root and every live descendant, from one pass over /proc."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def rss_bytes(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


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
    cap, capped = rss_cap(), None
    while True:
        try:
            pid, status, ru = os.wait4(child.pid, os.WNOHANG if cap else 0)
        except InterruptedError:
            continue
        if pid:
            break
        if cap and capped is None:
            procs = tree(child.pid)
            seen = max((rss_bytes(p) for p in procs), default=0)
            if seen > cap:
                capped = seen
                print(f"ladder_meter: {verb}: RSS {seen} B passed the budget cap {cap} B -- killing the call", file=sys.stderr)
                for p in reversed(procs):
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
        time.sleep(POLL_S)
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
        if capped is not None:
            rec["rss_capped"] = capped
            rec["peak_rss_bytes"] = max(rec["peak_rss_bytes"], capped)
        with open(out, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
