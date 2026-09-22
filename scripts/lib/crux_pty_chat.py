#!/usr/bin/env python3
"""crux_pty_chat.py: drive an interactive chat CLI through a pseudo-terminal (#3739, the `chat` verb).

WHY. Measured on lambda: fed a pipe, llama.cpp's chat CLI loops on empty prompts
and ollama's interactive mode reads the whole pipe as ONE prompt. Neither takes a
multi-turn conversation on stdin, so they are driven the way a person drives them:
wait for the prompt marker, type one turn, capture the reply up to the next marker.

It starts nothing but the argv it is given, opens no socket and reads no clock
for any measurement (the per-turn timeout is a bound, not a metric). Output is
the row-contract JSON the judge already reads for plugin engines:
  {"text": <final reply>, "turns": [<every reply>], "reported": {"device": ...}}
Exit: 0 every turn answered · 3 a turn timed out or the process died (the JSON
still says which turn and why) · 2 usage.

Usage: crux_pty_chat.py --marker REGEX --turns FILE.json --out FILE.json
           [--exit-line TEXT] [--device LABEL] [--turn-timeout S] [--start-timeout S]
           [--strip REGEX]... [--answer-after REGEX] -- CMD [ARGS...]
"""
import argparse
import json
import os
import pty
import re
import select
import signal
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][AB012]|\r")


def clean(text):
    return ANSI.sub("", text)


class Session:
    def __init__(self, argv):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "dumb"
            try:
                os.execvp(argv[0], argv)
            finally:
                os._exit(127)
        self.buf = ""

    def read_until(self, marker, since, bound):
        """Read until `marker` matches the cleaned output after offset `since`."""
        deadline = time.monotonic() + bound
        while True:
            m = marker.search(clean(self.buf)[since:])
            if m:
                return since + m.start()
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            r, _, _ = select.select([self.fd], [], [], min(left, 0.5))
            if r:
                try:
                    chunk = os.read(self.fd, 65536)
                except OSError:
                    return None
                if not chunk:
                    return None
                self.buf += chunk.decode("utf-8", "replace")

    def send(self, line):
        os.write(self.fd, (line + "\r").encode("utf-8"))

    def close(self, exit_line):
        if exit_line:
            try:
                self.send(exit_line)
            except OSError:
                pass
        for _ in range(20):
            done, _ = os.waitpid(self.pid, os.WNOHANG)
            if done:
                return
            time.sleep(0.25)
        try:
            os.kill(self.pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except OSError:
            pass


def main(argv):
    if "--" not in argv:
        sys.stderr.write("usage: crux_pty_chat.py [options] -- CMD [ARGS...]\n")
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exit-line", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--turn-timeout", type=float, default=300)
    ap.add_argument("--start-timeout", type=float, default=300)
    ap.add_argument("--strip", action="append", default=[])
    ap.add_argument("--answer-after", default="",
                    help="the reply starts after the LAST match of this regex (an echo the terminal breaks up)")
    a = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1:]
    if not cmd:
        sys.stderr.write("no command after --\n")
        return 2
    turns = json.load(open(a.turns))
    marker = re.compile(a.marker)
    strips = [re.compile(s) for s in a.strip]
    s = Session(cmd)
    out = {"text": None, "turns": [], "reported": {"device": a.device or None, "interface": "pty"}}
    rc = 0
    at = s.read_until(marker, 0, a.start_timeout)
    if at is None:
        out["error"] = "no prompt marker within %.0fs of start" % a.start_timeout
        rc = 3
    else:
        pos = at + 1
        for i, turn in enumerate(turns):
            s.send(turn)
            end = s.read_until(marker, pos, a.turn_timeout)
            if end is None:
                out["error"] = "turn %d: no prompt marker within %.0fs" % (i + 1, a.turn_timeout)
                rc = 3
                break
            reply = clean(s.buf)[pos:end]
            if a.answer_after:
                last = None
                for last in re.finditer(a.answer_after, reply, re.M):
                    pass
                if last is not None:
                    reply = reply[last.end():]
            else:
                k = reply.find(turn)      # the terminal echoes what was typed; the reply follows it
                if k >= 0:
                    reply = reply[k + len(turn):]
            for rx in strips:
                reply = rx.sub("", reply)
            out["turns"].append(reply.strip())
            pos = end + 1
    s.close(a.exit_line)
    if out["turns"] and rc == 0:
        out["text"] = out["turns"][-1]
    out["transcript_tail"] = clean(s.buf)[-1500:]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
