#!/usr/bin/env python3
"""crux_apr_code.py: the CRUX `code` verb for apr (#3962), one prompt -> one row-contract JSON.

`apr code` is an agent, not a sampler, and three of its properties are recorded rather
than papered over:
  backend   apr code has NO backend flag. It spawns `apr serve run <model> --gpu`
            (crates/aprender-orchestrate/src/agent/driver/apr_serve.rs) and talks to
            that server. On the gpu lane the row says so. On the cpu lane there is no way
            to ask it for the CPU, so the harness does not run it and the row is
            REFUSED with that reason. An apr cell that did not run is never green.
  sampling  apr code takes no --max-tokens, --temperature or thinking switch. The row
            records the prompt's max_tokens and thinking as REQUESTED and the control as
            `none`. So the cell is judged by ground truth only: its python block
            is EXECUTED against the prompt's asserts (scripts/lib/crux_oracles.py
            code_tests), never compared token-for-token.
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
                        [--thinking off] [--timeout 600] --out O.json
Exit: 0 an answer was read · 3 protocol fault · 2 usage.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apr", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True, help="one v2 prompt entry (a single user message)")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prompt = json.load(open(a.prompt_file))
    users = [m["content"] for m in prompt["messages"] if m.get("role") == "user"]
    out = {"text": None, "protocol_fault": None, "refused": None,
           "reported": {"interface": "apr code -p --output-format json --max-turns 1",
                        "backend_control": "none: apr code spawns `apr serve run <model> --gpu` "
                                           "(agent/driver/apr_serve.rs); it has no backend flag",
                        "max_tokens_requested": a.max_tokens, "max_tokens_control": "none: apr code has no --max-tokens",
                        "thinking_requested": a.thinking, "thinking_control": "none: apr code has no thinking switch",
                        "envelope": None, "stderr_tail": None}}

    def finish(rc):
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        return rc

    if len(users) != 1:
        out["refused"] = "the code verb drives one user turn; prompt %s has %d" % (prompt.get("id"), len(users))
        return finish(4)
    project = tempfile.mkdtemp(prefix="crux-code-project-")
    try:
        try:
            p = subprocess.run([a.apr, "code", "-p", "--model", a.model, "--project", project,
                                "--max-turns", "1", "--output-format", "json", "--", users[0]],
                               capture_output=True, text=True, timeout=a.timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            out["protocol_fault"] = "timeout: apr code did not finish in %ss" % a.timeout
            out["reported"]["stderr_tail"] = (exc.stderr or b"")[-600:].decode("utf-8", "replace") \
                if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-600:]
            return finish(3)
    finally:
        shutil.rmtree(project, ignore_errors=True)
    out["reported"]["stderr_tail"] = p.stderr[-600:]
    if p.returncode != 0:
        out["protocol_fault"] = "exit_%d: apr code exited non-zero" % p.returncode
        return finish(3)
    try:
        env = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
    except ValueError:
        env = None
    if not isinstance(env, dict):
        out["protocol_fault"] = "not_json: stdout is not the --output-format json envelope: %r" % p.stdout[:160]
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
