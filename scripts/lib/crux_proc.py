"""A stopped CRUX driver takes its engine with it (#3952; measured by aprender-dd on gx10, 2026-09-23).

SIGTERM to `crux_engine_vllm.sh gen-batch` left VLLM::EngineCore reparented to init, holding 58928 MiB after
the GPU lock dropped, so the next holder found an occupied card. Two causes, two fixes:
  1. `uv run` did not forward SIGTERM to python. The wrappers now `uv sync --frozen` and then `exec` the
     environment's python, so the signal reaches this process.
  2. vLLM's engine core is a CHILD process (and `vllm serve` / `transformers serve` run in their own session),
     so python exiting is not enough. install() makes SIGTERM/SIGINT/SIGHUP kill every descendant — found by
     walking /proc, which reaches children that changed session or process group — then exit 143.
Stdlib only; Linux /proc.
"""

from __future__ import annotations

import os
import signal
import sys
import time


def descendants(root: int) -> list[int]:
    """Every live descendant of `root`, deepest first, from /proc/<pid>/stat."""
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="utf-8", errors="replace") as f:
                stat = f.read()
        except OSError:
            continue
        # the command field can hold spaces and parens: the ppid is the 2nd field after the LAST ')'
        ppid = int(stat[stat.rfind(")") + 2:].split()[1])
        children.setdefault(ppid, []).append(int(name))
    out: list[int] = []

    def walk(pid: int) -> None:
        for c in children.get(pid, []):
            walk(c)
            out.append(c)

    walk(root)
    return out


def self_pid() -> int:
    """This process's pid AS /proc NUMBERS IT. Under a pid namespace whose /proc is the host's (a sandbox; also
    `unshare -Urpf`), os.getpid() is the namespace's pid (1) while /proc lists host pids, so a walk rooted at
    getpid() finds no children and kills nothing (quorum lane 2, 2026-09-23; reproduced unprivileged: getpid 1,
    /proc/self 1762161). /proc/self resolves in /proc's own namespace."""
    try:
        return int(os.readlink("/proc/self"))
    except (OSError, ValueError):
        return os.getpid()


def _nspids(proc_pid) -> list[int]:
    """The pid at every namespace level, outermost first (/proc/<pid>/status `NSpid:`), or [] if unreadable."""
    try:
        with open(f"/proc/{proc_pid}/status", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    return [int(x) for x in line.split()[1:]]
    except OSError:
        pass
    return []


def signal_pid(proc_pid: int) -> int | None:
    """The pid kill() must be given for a process /proc numbers `proc_pid`. /proc can be ANOTHER namespace's (a
    sandbox with the host's /proc): kill() takes pids in THIS process's namespace, so a host pid signals nothing,
    or worse, a different process that happens to have that number here. Translated through NSpid at this
    process's own nesting level; None if the process is not visible from here."""
    level = len(_nspids("self")) - 1
    ids = _nspids(proc_pid)
    if level < 0:
        return proc_pid  # no NSpid support: /proc and kill() agree
    return ids[level] if len(ids) > level else None


def kill_descendants(grace: float = 5.0) -> list[int]:
    """SIGTERM every descendant, wait up to `grace` seconds, SIGKILL whatever is left. Returns the pids signalled."""
    pids = descendants(self_pid())
    targets = {p: signal_pid(p) for p in pids}
    for p, t in targets.items():
        if t is None:
            continue
        try:
            os.kill(t, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.1)
    for p, t in targets.items():
        if t is not None and _alive(p):
            try:
                os.kill(t, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return pids


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"  # a zombie holds no memory


def install() -> None:
    def on_signal(signum, _frame):
        signal.signal(signum, signal.SIG_IGN)  # one cleanup, even if the signal repeats
        kill_descendants()
        sys.exit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
