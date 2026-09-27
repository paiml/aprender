#!/usr/bin/env python3
"""crux_apr_code.py: the CRUX `code` verb for apr (#3962), one prompt -> one row-contract JSON.

`apr code` is an agent, not a sampler, and three of its properties are recorded rather
than papered over:
  controls  FEATURE-DETECTED from `apr code --help`. With #3978 (`--no-gpu`/`--gpu`,
            `--max-tokens`, `--thinking`), the lane's backend, the prompt's max_tokens
            and `--thinking off` are PASSED, and the row records them as controlled.
            Without #3978, apr code has none of them: it always spawns
            `apr serve run <model> --gpu`. The row then records each control as
            `none`, and the cpu lane refuses the cell (the harness asks
            `crux_apr_code.py controls`). Either way the cell is judged by ground truth:
            its python block is EXECUTED against the prompt's asserts
            (scripts/lib/crux_oracles.py code_tests).
  prompt    apr code adds its own agent system prompt, loaded from --project. The
            project is a fresh, EMPTY tmpdir, so no APR.md or CLAUDE.md from the tree
            leaks into it, and --max-turns 1 stops a tool loop.

The reply is the `result` field of `apr code -p --output-format json`, copied RAW:
think blocks, prose and fences as apr printed them. The oracle extracts the last
python block itself.

protocol_fault (the cell ran and the verb broke, so it is RED whatever `text` says):
  exit_<rc>         apr code exited non-zero
  not_json          stdout is not the JSON envelope
  is_error          the envelope says is_error / status != ok
  empty_text        the envelope's result is empty

Usage: crux_apr_code.py --apr BIN --model M --prompt-file P.json --max-tokens N
                        [--backend gpu|cpu] [--thinking off] [--timeout 600] --out O.json
       crux_apr_code.py controls --apr BIN    prints `yes` when apr code has the #3978 flags
Exit: 0 an answer was read · 3 protocol fault · 2 usage.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile


def clip_head(s, n):
    """`s` cut to its first `n` chars, SAYING how many were cut (#4046: a silent cut reads as the whole text)."""
    return s if len(s) <= n else f"{s[:n]} … and {len(s) - n} more chars"


def clip_tail(s, n):
    """`s` cut to its last `n` chars, SAYING how many were dropped (#4046)."""
    return s if len(s) <= n else f"[{len(s) - n} earlier chars dropped] {s[-n:]}"


def has_controls(apr):
    """Does this apr's `code` take the #3978 flags? Read from its own --help, never assumed."""
    try:
        h = subprocess.run([apr, "code", "--help"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return all(f in h for f in ("--no-gpu", "--max-tokens", "--thinking"))


def main(argv):
    if argv[:1] == ["controls"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--apr", required=True)
        print("yes" if has_controls(ap.parse_args(argv[1:]).apr) else "no")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--apr", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True, help="one v2 prompt entry (a single user message)")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--serve-pid-file", default="",
                    help="append the pid of the `apr serve` child apr code launched, for the cell's teardown")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prompt = json.load(open(a.prompt_file))
    users = [m["content"] for m in prompt["messages"] if m.get("role") == "user"]
    ctl = has_controls(a.apr)
    flags = ["--%s" % ("no-gpu" if a.backend == "cpu" else "gpu"), "--max-tokens", str(a.max_tokens),
             "--thinking", a.thinking] if ctl else []
    out = {"text": None, "protocol_fault": None, "refused": None,
           "reported": {"interface": "apr code -p --output-format json --max-turns 1 " + " ".join(flags),
                        "backend_control": ("passed: %s" % flags[0]) if ctl else
                        "none: apr code spawns `apr serve run <model> --gpu` (agent/driver/apr_serve.rs); "
                        "it has no backend flag before #3978",
                        "max_tokens_requested": a.max_tokens,
                        "max_tokens_control": "passed: --max-tokens" if ctl else "none: apr code has no --max-tokens",
                        "thinking_requested": a.thinking,
                        "thinking_control": "passed: --thinking" if ctl else "none: apr code has no thinking switch",
                        "envelope": None, "stderr_tail": None}}

    def finish(rc):
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        return rc

    if a.backend == "cpu" and not ctl:
        out["refused"] = ("apr code has no backend flag: it always spawns `apr serve run --gpu` "
                          "(agent/driver/apr_serve.rs), so it cannot be run on the cpu lane (#3978)")
        return finish(4)
    if len(users) != 1:
        out["refused"] = "the code verb drives one user turn; prompt %s has %d" % (prompt.get("id"), len(users))
        return finish(4)
    project = tempfile.mkdtemp(prefix="crux-code-project-")
    try:
        try:
            p = subprocess.run([a.apr, "code", "-p", "--model", a.model, "--project", project,
                                "--max-turns", "1", "--output-format", "json", *flags, "--", users[0]],
                               capture_output=True, text=True, timeout=a.timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            out["protocol_fault"] = "timeout: apr code did not finish in %ss" % a.timeout
            err = exc.stderr or b""
            out["reported"]["stderr_tail"] = clip_tail(err.decode("utf-8", "replace") if isinstance(err, bytes) else err, 600)
            return finish(3)
    finally:
        shutil.rmtree(project, ignore_errors=True)
    out["reported"]["stderr_tail"] = clip_tail(p.stderr, 600)
    # apr code's own `apr serve` child is invisible to the cell otherwise. It dies with
    # its parent (PR_SET_PDEATHSIG), but the cell's teardown PROVES that, off the GPU
    # too, before the lock drops.
    served = re.findall(r"Launched apr serve on port \d+ \(pid (\d+)", p.stderr)
    out["reported"]["serve_pids"] = [int(x) for x in served]
    if a.serve_pid_file and served:
        with open(a.serve_pid_file, "a") as fh:
            fh.write("".join(x + "\n" for x in served))
    if p.returncode != 0:
        out["protocol_fault"] = "exit_%d: apr code exited non-zero" % p.returncode
        return finish(3)
    try:
        env = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
    except ValueError:
        env = None
    if not isinstance(env, dict):
        out["protocol_fault"] = "not_json: stdout is not the --output-format json envelope: %s" % clip_head(repr(p.stdout), 160)
        return finish(3)
    out["reported"]["envelope"] = {k: env.get(k) for k in ("type", "subtype", "status", "is_error", "num_turns",
                                                            "tokens_in", "tokens_out", "session_id")}
    if env.get("is_error") or env.get("status") not in (None, "ok"):
        out["protocol_fault"] = "is_error: the envelope reports status=%r is_error=%r" % (env.get("status"),
                                                                                         env.get("is_error"))
        return finish(3)
    if not env.get("result"):
        out["protocol_fault"] = "empty_text: the envelope's result is empty"
        return finish(3)
    out["text"] = env["result"]
    return finish(0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
