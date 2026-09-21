#!/usr/bin/env python3
"""Judge one `apr code` edit-and-verify run from its artifacts (#3719).

scripts/apr_code_edit_verify.sh runs the agent and leaves an artifact
directory behind. This module reads only those artifacts: the working copy,
the independent test re-run, the --emit-trace records, and the serve child's
own output. It writes <out>/cell.json and prints one summary line.

The mechanisms are checked in pipeline order and the FIRST one that fails is
the cell's mechanism, using the names from #3719 done_when 1:

  serve child did not load -> fell back to CPU -> (serve child refused the
  forward) -> tool call not parsed -> wrong edit -> test not run
  -> wrong final answer

"serve child refused the forward" is not in #3719's list: it names a child
that loaded every layer on CUDA and then answered the completion with an HTTP
error, which none of the listed names describes.

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
# actually came back; the gpu-layers line is the evidence cited.
CUDA_READY = re.compile(r"CUDA optimized model ready")
CPU_MARKERS = re.compile(
    r"\[GPU->CPU FALLBACK\]|Using CPU inference|Q4K CPU inference ready"
)
MODEL_LAYERS = re.compile(r"Model ready: (\d+) layers")
GPU_LAYERS = re.compile(r"gpu-layers: requested=\S+ resolved=(\d+) total=(\d+) \(backend=(\w+)\)")
# The driver prints this once the child answers its health check.
SERVE_READY = re.compile(r"apr serve ready \(")
# The driver's error when the child answers a completion with an HTTP error.
SERVE_HTTP_ERROR = re.compile(r"apr serve HTTP (\d+): (.*)")
# The thinking mode the serve child renders. The driver strips <think> blocks
# before parsing, so the trace cannot show it; only the child's own line can.
THINKING_LINE = re.compile(r"chat template:.*\(thinking (on|off)\)")
# A per-request prompt size printed by the child. session_end.tokens_in in the
# trace is summed over turns (agent/result.rs accumulate), so it is not one.
PROMPT_TOKENS_LINE = re.compile(r"\bprompt_tokens[=: ]+(\d+)")

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


def trace_records(path):
    records, bad = [], 0
    for ln in read(path).splitlines():
        if not ln.strip():
            continue
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    return records, bad


def tool_uses(records):
    uses = []
    for r in records:
        if r.get("kind") != "assistant_turn":
            continue
        for b in r.get("blocks") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                uses.append(b)
    return uses


def ran_the_test(uses):
    """A shell tool call whose input names the fixture's test."""
    for u in uses:
        name = str(u.get("name", "")).lower()
        blob = json.dumps(u.get("input", {}))
        if ("shell" in name or "bash" in name) and (
            "unittest" in blob or "test_stats" in blob
        ):
            return blob[:200]
    return ""


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


def judge(a):
    out = Path(a.out)
    fixture = Path(a.fixture) / "project"
    work = out / "project"
    stderr = read(out / "stderr.txt")
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
    cuda_line = first_match(child, CUDA_READY)
    cpu_line = first_match(child, CPU_MARKERS)
    layers_line = first_match(child, MODEL_LAYERS)
    layers = int(MODEL_LAYERS.search(layers_line).group(1)) if layers_line else None
    gpu_line = first_match(child, GPU_LAYERS)
    resolved, total, gpu_backend = GPU_LAYERS.search(gpu_line).groups() if gpu_line else (None, None, None)
    all_resident = (gpu_backend == "cuda" and resolved is not None
                    and int(resolved) == int(total) and int(total) > 0)
    http_error = SERVE_HTTP_ERROR.search(stderr)

    records, bad = trace_records(out / "trace.jsonl")
    uses = tool_uses(records)
    turns = [r for r in records if r.get("kind") == "assistant_turn"]
    cell.update(trace_records=len(records), trace_bad_lines=bad, tool_calls=len(uses),
                tools_used=sorted({str(u.get("name")) for u in uses}))

    if cpu_line:
        backend = "cpu"
    elif cuda_line and all_resident and turns:
        backend = "cuda"
    else:
        backend = "unknown"
    cell.update(backend=backend, fallback=bool(cpu_line),
                evidence=cpu_line or (gpu_line if backend == "cuda" else ""),
                model_layers=layers, gpu_layers=gpu_line)

    thinking = THINKING_LINE.search(child)
    cell["thinking"] = thinking.group(1) if thinking else "unknown"
    sizes = [int(n) for n in PROMPT_TOKENS_LINE.findall(child)]
    cell["prompt_tokens"] = max(sizes) if sizes else None
    cell["context"] = "4k" if sizes and max(sizes) >= RUNG_4K_TOKENS else "task"
    cell["max_tokens"] = DRIVER_MAX_TOKENS
    ends = [r for r in records if r.get("kind") == "session_end"]
    cell["tokens_in_total"] = ends[-1].get("tokens_in") if ends else None

    env = envelope(stdout)
    result = str(env.get("result", "")) if env else ""
    cell["answer_chars"] = len(result)

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
        detail = f"; first request: HTTP {http_error.group(1)} {http_error.group(2)[:200]}" if http_error else ""
        return fail("serve child did not load",
                    f"the child reported ready with 0 layers{detail}",
                    "\n".join(x for x in (layers_line, gpu_line) if x))

    # 2. fell back to CPU, or never showed every layer resident on CUDA
    if cpu_line:
        return fail("fell back to CPU", "serve child printed a CPU path", cpu_line)
    if not (cuda_line and all_resident):
        return fail("fell back to CPU",
                    "serve child never showed every layer resident on CUDA",
                    gpu_line or last_lines(child))

    # 2b. the child loaded on CUDA but refused the completion
    if http_error and not turns:
        return fail("serve child refused the forward",
                    f"HTTP {http_error.group(1)}: {http_error.group(2)[:300]}",
                    gpu_line)

    # 3. tool call not parsed
    if not uses:
        return fail("tool call not parsed",
                    f"the trace holds {len(records)} records and no tool_use block",
                    result[:300] or last_lines(stderr))

    # 4. wrong edit
    before_files, after_files = project_files(fixture), project_files(work)
    cell["files_added"] = sorted(after_files - before_files)
    cell["files_removed"] = sorted(before_files - after_files)
    src_before, src_after = read(fixture / EDITED_FILE), read(work / EDITED_FILE)
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
        return fail("wrong edit", f"{EDITED_FILE} was not changed")
    if not one_line_edit(src_before, src_after):
        return fail("wrong edit", f"{EDITED_FILE} changed by more than one line", diff[:600])
    if a.test_rc != 0:
        return fail("wrong edit", "one-line edit made, but the independent test re-run still fails",
                    last_lines(read(out / "unittest.txt")))

    # 5. test not run
    ran = ran_the_test(uses)
    cell["test_call"] = ran
    if not ran:
        return fail("test not run", "no shell tool call ran the fixture's test")

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
