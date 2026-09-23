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


def kill_descendants(grace: float = 5.0) -> list[int]:
    """SIGTERM every descendant, wait up to `grace` seconds, SIGKILL whatever is left. Returns the pids signalled."""
    pids = descendants(os.getpid())
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.1)
    for p in pids:
        if _alive(p):
            try:
                os.kill(p, signal.SIGKILL)
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
