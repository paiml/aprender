#!/usr/bin/env python3
"""Judge one `apr code` edit-and-verify run from its artifacts (#3719).

scripts/apr_code_edit_verify.sh runs the agent and leaves an artifact
directory behind. This module reads only those artifacts: the working copy,
the independent test re-run, the python-shim log of what the agent's shell
ran, the final answer, and the serve child's own output. It writes
<out>/cell.json and prints one summary line.

The mechanisms are checked in pipeline order and the FIRST one that fails is
the cell's mechanism, using the names from #3719 done_when 1:

  serve child did not load -> fell back to CPU -> (serve child refused the
  forward) -> tool call not parsed -> wrong edit -> test not run
  -> wrong final answer

"serve child refused the forward" is not in #3719's list: it names a child
that loaded every layer on CUDA and then answered the completion with an HTTP
error, which none of the listed names describes.

What each mechanism reads, and why:
  * The --emit-trace file is NOT evidence of tool calls. It holds exactly four
    records and the assistant turn is one text block, whatever the agent did
    (crates/aprender-orchestrate/src/agent/code.rs emit_ccpa_trace). A judge
    reading tool_use blocks from it could never pass.
  * "tool call not parsed" needs a call the model emitted and the driver did
    not execute: stats.py unchanged while the final answer still carries the
    markup the driver parses (<tool_call> or a ```json block, see
    agent/driver/realizar.rs parse_tool_calls).
  * "test not run" reads the python shims' log: the agent's shell tool runs
    `sh -c` with the inherited PATH, and the shims are first on it.

Exit: 0 PASS, 1 FAIL, 2 decline (the GPU lock was never acquired).
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# What the serve child prints on its CUDA path, and on each way it can end up
# on the CPU instead (crates/apr-cli/src/commands/serve/).
#
# `CUDA optimized model ready` is a LOAD-time line and is not evidence of a CUDA
# forward: measured on 856009cc9, the child printed it for Qwen3.5-4B after
# `Model ready: 0 layers` and `gpu-layers: requested=all resolved=0 total=0`,
# and the first request then failed with HTTP 500 (#3571 step 1). So the
# backend is `cuda` only when every layer is resident on CUDA and a completion
# actually came back; the residency line is the evidence cited.
CUDA_READY = re.compile(r"CUDA optimized model ready")
CPU_MARKERS = re.compile(
    r"\[GPU->CPU FALLBACK\]|Using CPU inference|Q4K CPU inference ready"
    r"|layers resident on the CPU"
)
# The generic GGUF route: a layer count, then the gpu-layers residency line.
MODEL_LAYERS = re.compile(r"Model ready: (\d+) layers")
GPU_LAYERS = re.compile(r"gpu-layers: requested=\S+ resolved=(\d+) total=(\d+) \(backend=(\w+)\)")
# The Qwen35Session route (#3571 step 2, aprender-c7) states both in one line:
# `Model ready: Qwen3.5 hybrid, N layers resident on the GPU|CPU, declared context C tokens`.
HYBRID_READY = re.compile(r"Model ready: .*?(\d+) layers resident on the (GPU|CPU)")
# The driver prints this once the child answers its health check.
SERVE_READY = re.compile(r"apr serve ready \(")
# The driver's error when the child answers a completion with an HTTP error.
SERVE_HTTP_ERROR = re.compile(r"apr serve HTTP (\d+): (.*)")
# The thinking mode the serve child renders (the Qwen35Session route prints
# `chat template: Qwen3NoThink (thinking off)`). The driver strips <think>
# blocks before parsing, so nothing else can show it. The template name varies
# with the shared selection (#3755), so only the `(thinking ...)` part is read.
# Its value is `off`, `on` or `the model's choice`; only on/off key a ladder
# row, and anything else is kept raw and keys onto no cell.
THINKING_LINE = re.compile(r"chat template:.*\(thinking ([^)]+)\)")
# A per-request prompt size printed by the child. session_end.tokens_in in the
# trace is summed over turns (agent/result.rs accumulate), so it is not one.
PROMPT_TOKENS_LINE = re.compile(r"\bprompt_tokens[=: ]+(\d+)")
# Tool-call markup the driver parses. Left in the final answer, it is a call
# the model made and the driver never executed.
UNPARSED_CALL = re.compile(r"<tool_call>|```json|<function=")

# The ladder's row keys (#3712 scripts/lib/model_ladder_cells.py, #3715): a
# row is `4k` only if one request's measured prompt reached the rung. A
# smaller task is keyed `task`, which no rung owes.
RUNG_4K_TOKENS = 4096
# The driver caps every completion at min(manifest 4096, 1024)
# (crates/aprender-orchestrate/src/agent/driver/apr_serve.rs).
DRIVER_MAX_TOKENS = 1024

EDITED_FILE = "stats.py"
TEST_FILE = "test_stats.py"
IGNORED_DIRS = {"__pycache__"}


def read(path):
    try:
        return ANSI.sub("", Path(path).read_text(errors="replace"))
    except FileNotFoundError:
        return ""


def last_lines(text, n=6):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def first_match(text, pattern):
    for ln in text.splitlines():
        if pattern.search(ln):
            return ln.strip()
    return ""


def project_files(root):
    root = Path(root)
    out = set()
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if p.is_file():
            out.add(str(rel))
    return out


def one_line_edit(before, after):
    """True iff `after` differs from `before` by exactly one replaced line."""
    a, b = before.splitlines(), after.splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b).get_opcodes() if op[0] != "equal"]
    return (len(ops) == 1 and ops[0][0] == "replace"
            and ops[0][2] - ops[0][1] == 1 and ops[0][4] - ops[0][3] == 1)


def envelope(stdout):
    """The `--output-format json` result envelope, or None."""
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    except json.JSONDecodeError:
        pass
    for ln in reversed(stdout.splitlines()):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def residency(child):
    """(layers, resident_on_cuda, evidence line) from either serve route."""
    hybrid = first_match(child, HYBRID_READY)
    if hybrid:
        n, where = HYBRID_READY.search(hybrid).groups()
        return int(n), where == "GPU" and int(n) > 0, hybrid
    layers_line = first_match(child, MODEL_LAYERS)
    layers = int(MODEL_LAYERS.search(layers_line).group(1)) if layers_line else None
    gpu_line = first_match(child, GPU_LAYERS)
    if not gpu_line:
        return layers, False, layers_line
    resolved, total, backend = GPU_LAYERS.search(gpu_line).groups()
    resident = (backend == "cuda" and int(resolved) == int(total) > 0
                and bool(CUDA_READY.search(child)))
    return layers, resident, "\n".join(x for x in (layers_line, gpu_line) if x)


def judge(a):
    out = Path(a.out)
    fixture = Path(a.fixture) / "project"
    work = out / "project"
    # gpu-q prints its queue position to stderr while it waits; those lines are
    # the queue talking, not apr, and are kept out of every evidence excerpt.
    stderr = "\n".join(ln for ln in read(out / "stderr.txt").splitlines()
                       if not ln.startswith("gpu-q: "))
    stdout = read(out / "stdout.json")
    child = read(out / "serve-child.stdout") + "\n" + read(out / "serve-child.stderr")

    cell = {
        "verb": "code",
        "host": a.host,
        "file": Path(a.model).name,
        "sha256": read(out / "model.sha256").strip(),
        "apr_version": read(out / "apr-version.txt").strip(),
        "hostname": read(out / "hostname.txt").strip(),
        "started": read(out / "started.txt").strip(),
        "finished": read(out / "finished.txt").strip(),
        "rc": a.rc,
        "test_rc": a.test_rc,
        "task": "tests/fixtures/apr-code-edit-verify",
    }

    if not (out / "lock-acquired").exists():
        cell.update(verdict="decline", mechanism="gpu lock not acquired",
                    reason=f"/tmp/apr-gpu.lock not acquired within {a.lock_wait}s; nothing ran",
                    backend="none", fallback=False, evidence="")
        return cell, 2

    timed_out = a.rc == 124
    cpu_line = first_match(child, CPU_MARKERS)
    layers, resident, residency_line = residency(child)
    http_error = SERVE_HTTP_ERROR.search(stderr)
    env = envelope(stdout)
    result = str(env.get("result", "")) if env else ""
    answered = env is not None

    if cpu_line:
        backend = "cpu"
    elif resident and answered:
        backend = "cuda"
    else:
        backend = "unknown"
    cell.update(backend=backend, fallback=bool(cpu_line),
                evidence=cpu_line or (residency_line if backend == "cuda" else ""),
                model_layers=layers, answer_chars=len(result))

    thinking = THINKING_LINE.search(child)
    raw = thinking.group(1).strip() if thinking else ""
    cell["thinking"] = raw if raw in ("on", "off") else "unknown"
    cell["thinking_raw"] = raw
    sizes = [int(n) for n in PROMPT_TOKENS_LINE.findall(child)]
    cell["prompt_tokens"] = max(sizes) if sizes else None
    cell["context"] = "4k" if sizes and max(sizes) >= RUNG_4K_TOKENS else "task"
    cell["max_tokens"] = DRIVER_MAX_TOKENS

    calls = [ln for ln in read(out / "agent-python.log").splitlines() if ln.strip()]
    cell["agent_python_calls"] = calls[:20]

    def fail(mechanism, reason, evidence=None):
        if timed_out:
            reason = f"{reason} (apr code timed out after {a.timeout}s)"
        cell.update(verdict="fail", mechanism=mechanism, reason=reason)
        if evidence is not None:
            cell["evidence"] = evidence
        return cell, 1

    # 1. serve child did not load (a model with no layers is not loaded, even
    #    when the child calls itself ready)
    if not SERVE_READY.search(stderr):
        return fail("serve child did not load",
                    "the driver never reported `apr serve ready`",
                    last_lines(child) or last_lines(stderr))
    if layers == 0:
        detail = (f"; first request: HTTP {http_error.group(1)} {http_error.group(2)[:200]}"
                  if http_error else "")
        return fail("serve child did not load",
                    f"the child reported ready with 0 layers{detail}", residency_line)

    # 2. fell back to CPU, or never showed every layer resident on CUDA
    if cpu_line:
        return fail("fell back to CPU", "serve child printed a CPU path", cpu_line)
    if not resident:
        return fail("fell back to CPU", "serve child never showed every layer resident on CUDA",
                    residency_line or last_lines(child))

    # 2b. the child loaded on CUDA but refused the completion
    if http_error and not answered:
        return fail("serve child refused the forward",
                    f"HTTP {http_error.group(1)}: {http_error.group(2)[:300]}", residency_line)

    # 3. tool call not parsed: no edit landed, and the answer still carries
    #    the markup of a call the driver should have executed
    src_before, src_after = read(fixture / EDITED_FILE), read(work / EDITED_FILE)
    unparsed = UNPARSED_CALL.search(result)
    if src_after == src_before and unparsed:
        return fail("tool call not parsed",
                    f"{EDITED_FILE} unchanged and the final answer carries `{unparsed.group(0)}`",
                    result[max(0, unparsed.start() - 80):unparsed.start() + 220])

    # 4. wrong edit
    before_files, after_files = project_files(fixture), project_files(work)
    cell["files_added"] = sorted(after_files - before_files)
    cell["files_removed"] = sorted(before_files - after_files)
    diff = "".join(difflib.unified_diff(src_before.splitlines(True), src_after.splitlines(True),
                                        EDITED_FILE, EDITED_FILE))
    (out / "stats.diff").write_text(diff)
    cell["edit"] = [ln for ln in diff.splitlines()
                    if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    if read(work / TEST_FILE) != read(fixture / TEST_FILE):
        return fail("wrong edit", f"{TEST_FILE} was modified; the task forbids it")
    if cell["files_added"] or cell["files_removed"]:
        return fail("wrong edit",
                    f"files added {cell['files_added']} / removed {cell['files_removed']}")
    if src_after == src_before:
        return fail("wrong edit", f"{EDITED_FILE} was not changed and no unparsed call was left",
                    result[:300] or last_lines(stderr))
    if not one_line_edit(src_before, src_after):
        return fail("wrong edit", f"{EDITED_FILE} changed by more than one line", diff[:600])
    if a.test_rc != 0:
        return fail("wrong edit", "one-line edit made, but the independent test re-run still fails",
                    last_lines(read(out / "unittest.txt")))

    # 5. test not run (by the agent; the harness's re-run is not in the log)
    ran = [c for c in calls if "unittest" in c or "test_stats" in c]
    cell["test_call"] = ran[0] if ran else ""
    if not ran:
        return fail("test not run", "the agent's shell never ran python on the fixture's test",
                    "; ".join(calls[:3]) or "no python invocation at all")

    # 6. wrong final answer
    if env is None or env.get("subtype") != "success":
        return fail("wrong final answer", "no `type: result, subtype: success` envelope on stdout",
                    last_lines(stdout) or last_lines(stderr))
    if not re.search(r"\b(pass(es|ed|ing)?|OK)\b", result, re.IGNORECASE):
        return fail("wrong final answer", "the final answer does not report the passing test",
                    result[:300])
    if a.rc != 0:
        return fail("wrong final answer", f"task done but apr code exited {a.rc}")

    cell.update(verdict="pass", mechanism="", reason="")
    return cell, 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--rc", type=int, required=True)
    p.add_argument("--test-rc", type=int, required=True)
    p.add_argument("--lock-wait", type=int, default=0)
    p.add_argument("--timeout", type=int, default=0)
    a = p.parse_args()
    cell, code = judge(a)
    Path(a.out, "cell.json").write_text(json.dumps(cell, indent=1) + "\n")
    why = f"{cell['mechanism']}: {cell['reason']}" if cell.get("mechanism") else "task completed"
    print(f"{cell['verdict'].upper():7} {cell['host']:7} {cell['file']}  "
          f"backend={cell['backend']}  {why}")
    if cell.get("evidence"):
        print(f"        evidence: {cell['evidence'].splitlines()[0][:200]}")
    return code


if __name__ == "__main__":
    sys.exit(main())
