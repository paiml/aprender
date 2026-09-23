"""The vLLM CRUX engine (#3952).

Run through ``scripts/crux_engine_vllm.sh`` — ``uv run --frozen`` over the committed ``uv.lock`` beside this
file, never an ambient python — as ``probe | gen``. Row contract v1 (aprender-76, #3739 comment 5765991210):
``gen`` appends exactly ONE JSONL row to ``$CRUX_MANIFEST`` and writes its artifacts under
``$CRUX_WORK/<model_sha256[:12]>/<verb>/``. A cell vLLM cannot run is still a row — ``rc: null`` and
``refused`` carrying the reason — never an absence and never a guess.

WHAT vLLM COMPARES AGAINST. vLLM 0.30.0 cannot load a local GGUF: ``maybe_override_with_speculators`` hands
the ``.gguf`` to transformers' ``get_config_dict`` as JSON (measured on lambda, 2026-09-23, #3952). So a cell
runs the model's SOURCE weights at a pinned revision (``--source-repo``/``--source-revision``, the same
sidecar the hf engine uses) and every row says so: apr's quantized file against the source weights, never
"the same file".

``tok``, ``tmpl`` and ``greedy`` are ``none`` for this engine: vLLM tokenizes and renders through the same
transformers tokenizer the hf engine already reports, so a vLLM row there would be the hf row twice.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "uv.lock"
ENGINE = "vllm"
COMPARES = "apr quantized file vs source weights (vLLM 0.30.0 cannot load the GGUF, #3952)"


def die(msg: str, code: int = 2) -> None:
    print(f"crux_engine_vllm: {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ninja_path() -> str | None:
    """vLLM shells out to `ninja` at engine init. The wrapper puts the venv's bin on PATH; this is the check
    that it worked, so a missing ninja refuses BY NAME before a model is loaded."""
    return shutil.which("ninja")


# ── probe ──────────────────────────────────────────────────────────────────
def probe() -> None:
    try:
        import torch
        import transformers
        import vllm
    except Exception as e:  # the environment itself is the refusal
        die(f"the locked environment does not import: {type(e).__name__}: {e}", 3)
    ninja = ninja_path()
    if not ninja:
        die("`ninja` is not on PATH — vLLM's engine init shells out to it (FileNotFoundError: 'ninja')", 3)
    if not torch.cuda.is_available():
        # The pinned wheel is the CUDA build; it has no CPU backend, so without a device nothing can run.
        die(f"torch.cuda.is_available() is False (torch {torch.__version__}, built for CUDA {torch.version.cuda})", 3)
    parts = [
        f"vllm={vllm.__version__}",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"device={torch.cuda.get_device_name(0).replace(' ', '_')}",
        f"capability=sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}",
        f"ninja={ninja}",
        f"lock_sha256={sha256_file(LOCK)}",
    ]
    print(" ".join(parts))


# ── shared ─────────────────────────────────────────────────────────────────
def env_paths() -> tuple[Path, Path]:
    manifest, work = os.environ.get("CRUX_MANIFEST"), os.environ.get("CRUX_WORK")
    if not manifest:
        die("CRUX_MANIFEST is unset")
    if not work:
        die("CRUX_WORK is unset")
    return Path(manifest), Path(work)


def workdir(work: Path, sha: str, sub: str) -> Path:
    d = work / sha[:12] / sub.replace(" ", "-")
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_row(manifest: Path, row: dict) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def refusal(e: BaseException, *logs: Path) -> str:
    """The exception's own words — plus, when vLLM's engine core died in its subprocess, the ROOT error that
    process printed. The parent only ever sees "Engine core initialization failed. See root cause above.",
    which names no cause (measured on gx10: the cause was a memory-profiling AssertionError)."""
    text = f"{type(e).__name__}: {e}"
    root = next((r for r in (root_error(p) for p in logs) if r), None)
    if root and root not in text:
        text = f"{text} — engine core: {root}"
    return text[:4000]


def root_error(log: Path) -> str | None:
    """The last `<Name>Error: …` line the engine processes wrote, with vLLM's log prefix stripped."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = re.search(r"\b([A-Za-z_]*(?:Error|Exception)): (.*)$", line)
        if m and "Engine core initialization failed" not in line:
            return f"{m.group(1)}: {m.group(2).strip()}"
    return None


@contextlib.contextmanager
def fd2_to(path: Path):
    """Point file descriptor 2 at `path` for the duration, so vLLM's engine-core SUBPROCESS (which inherits the
    descriptor, not sys.stderr) writes its traceback where the row can quote it. Restored afterwards."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(path, "w", encoding="utf-8") as f:
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def check_sha(sha: str) -> None:
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        die("--model-sha256 must be 64 lowercase hex characters")


def thinking_kwargs(thinking: str) -> dict:
    return {"on": {"enable_thinking": True}, "off": {"enable_thinking": False}}.get(thinking, {})


def split_think(text: str) -> tuple[str, str]:
    """(answer, reasoning). A <think> with no </think> ran out inside its reasoning: the answer is EMPTY."""
    s, e = text.find("<think>"), text.find("</think>")
    if s < 0:
        return text.strip(), ""
    if e < 0:
        return "", text[s + 7 :].strip()
    return text[e + 8 :].strip(), text[s + 7 : e].strip()


def load_messages(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["messages"]


def gpu_memory_utilization() -> float:
    """vLLM reserves this fraction of the device up front. The 4090 is shared with other measurements and
    GB10 memory is unified with the host, so the default is a fraction, and the row records it."""
    return float(os.environ.get("CRUX_VLLM_GPU_MEMORY_UTILIZATION", "0.5"))


def preflight(a) -> None:
    """Every reason a cell cannot run that is knowable before a model loads, each by name."""
    if a.backend != "gpu":
        raise RuntimeError(
            "the pinned vLLM wheel is the CUDA build and has no CPU backend; a cpu-lane cell cannot run on it"
        )
    if not a.source_repo:
        raise RuntimeError(
            "vLLM 0.30.0 cannot load a local GGUF (maybe_override_with_speculators reads the .gguf as a JSON "
            "config, #3952); this cell needs --source-repo/--source-revision to run the source weights"
        )
    if not ninja_path():
        raise RuntimeError("`ninja` is not on PATH — vLLM's engine init shells out to it")


def device_label() -> str:
    import torch

    return f"cuda:0 {torch.cuda.get_device_name(0)}"


# ── gen ────────────────────────────────────────────────────────────────────
def conversation_turns(messages: list, respond) -> tuple[str, list, list]:
    """Drives a `chat` cell exactly as the hf engine does: every user turn is answered with the conversation
    so far, and the ANSWER — reasoning split off — is appended as the assistant turn before the next one."""
    convo, answers, raws = [], [], []
    for m in messages:
        convo.append(m)
        if m.get("role") != "user":
            continue
        raw = respond(convo)
        answer, _ = split_think(raw)
        convo.append({"role": "assistant", "content": answer})
        answers.append(answer)
        raws.append(raw)
    if not raws:
        raise ValueError("--messages holds no user turn to answer")
    return raws[-1], answers, raws


def gen_generate(a, messages):
    """`run`: one reply. `chat`: the conversation driven turn by turn. The engine is built ONCE either way."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.source_repo, revision=a.source_revision, tokenizer_revision=a.source_revision,
        dtype=a.dtype, max_model_len=a.context, seed=a.seed,
        gpu_memory_utilization=gpu_memory_utilization(), enforce_eager=True,
    )
    # Greedy whatever --temperature says, as the protocol asks (the hf engine does the same).
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=a.seed)
    counts = {"prompt_tokens": 0, "completion_tokens": 0}

    def respond(convo):
        out = llm.chat(convo, params, use_tqdm=False, chat_template_kwargs=thinking_kwargs(a.thinking) or None)[0]
        counts["prompt_tokens"] = len(out.prompt_token_ids)  # the FINAL turn's prompt holds the whole conversation
        counts["completion_tokens"] += len(out.outputs[0].token_ids)
        return out.outputs[0].text

    if a.verb == "chat":
        raw, answers, _ = conversation_turns(messages, respond)
        return raw, {**counts, "device": device_label()}, answers
    return respond(messages), {**counts, "device": device_label()}, None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gen_serve(a, messages, d: Path, stem: str):
    """`vllm serve`, the pinned version's own OpenAI-compatible server, one request, then stopped."""
    port = free_port()
    cmd = [
        str(Path(sys.executable).parent / "vllm"), "serve", a.source_repo, "--revision", a.source_revision,
        "--tokenizer-revision", a.source_revision, "--host", "127.0.0.1", "--port", str(port),
        "--dtype", a.dtype, "--max-model-len", str(a.context), "--seed", str(a.seed),
        "--gpu-memory-utilization", str(gpu_memory_utilization()), "--enforce-eager",
    ]
    log = open(d / f"{stem}.server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 600  # a first start compiles kernels
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited {proc.returncode} before answering (log: {stem}.server.log)")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("vllm serve did not answer /v1/models within 600 s")
                time.sleep(0.5)
        body = {"model": a.source_repo, "messages": messages, "max_tokens": a.max_tokens, "temperature": 0.0,
                "seed": a.seed}
        kw = thinking_kwargs(a.thinking)
        if kw:
            body["chat_template_kwargs"] = kw
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        (d / f"{stem}.resp.json").write_text(json.dumps(resp, indent=1), encoding="utf-8")
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            raw = f"<think>{reasoning}</think>{raw}"
        usage = resp.get("usage") or {}
        return raw, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                     "device": f"{device_label()} (vllm serve)"}
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()


def gen(a) -> None:
    manifest, work = env_paths()
    check_sha(a.model_sha256)
    d = workdir(work, a.model_sha256, a.verb)
    stem = f"{ENGINE}-{a.prompt_id}-{a.thinking}"
    out, err = d / f"{stem}.json", d / f"{stem}.err"
    row = {
        "kind": "gen", "engine": ENGINE, "model_sha256": a.model_sha256, "host": a.host, "verb": a.verb,
        "thinking": a.thinking, "backend": a.backend, "prompt_id": a.prompt_id,
        "rc": 0, "stdout": str(out), "stderr": str(err), "refused": None,
        # The comparison this row is: apr's file (model_sha256) against these SOURCE weights, never the file.
        "source": {"repo": a.source_repo, "revision": a.source_revision, "dtype": a.dtype, "compares": COMPARES},
        "gpu_memory_utilization": gpu_memory_utilization(),
    }
    engine_log = d / f"{stem}.engine.log"
    row["engine_log"] = str(engine_log)
    try:
        preflight(a)
        messages = load_messages(a.messages)
        turns = None
        with fd2_to(engine_log):
            if a.verb in ("run", "chat"):
                raw, counts, turns = gen_generate(a, messages)
            else:
                raw, counts = gen_serve(a, messages, d, stem)
        answer, reasoning = split_think(raw)
        doc = {"text": answer}
        if turns is not None:
            doc["turns"] = turns  # every assistant answer, in order; `text` is the last of them
        if reasoning:
            doc["reasoning"] = reasoning
        doc["reported"] = {
            "interface": "LLM.chat" if a.verb in ("run", "chat") else "vllm serve",
            "thinking_requested": a.thinking, "thinking_emitted": bool(reasoning), **counts,
        }
        out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        err.write_text("", encoding="utf-8")
    except Exception as e:
        reason = refusal(e, engine_log, d / f"{stem}.server.log")
        err.write_text(reason, encoding="utf-8")
        row.update(rc=None, stdout=None, refused=reason)
    append_row(manifest, row)


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> None:
    if argv and argv[0] in ("tok", "tmpl", "greedy"):
        die(f"`{argv[0]}` is `none` for vLLM: it tokenizes and renders through the transformers tokenizer the "
            "hf engine already reports")
    p = argparse.ArgumentParser(prog="crux_engine_vllm.sh")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")

    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--model-sha256", required=True)
    g.add_argument("--verb", required=True, choices=["run", "chat", "serve run", "code"])
    g.add_argument("--prompt-id", required=True)
    g.add_argument("--messages", required=True)
    g.add_argument("--prompt-file")
    g.add_argument("--thinking", required=True, choices=["on", "off", "unset"])
    g.add_argument("--backend", required=True, choices=["gpu", "cpu"])
    g.add_argument("--host", required=True)
    g.add_argument("--max-tokens", type=int, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--temperature", type=float, required=True)
    g.add_argument("--context", type=int, required=True)
    g.add_argument("--source-repo")
    g.add_argument("--source-revision")
    g.add_argument("--dtype", default="bfloat16")

    a = p.parse_args(argv)
    if a.cmd == "gen" and a.source_repo and not a.source_revision:
        die("gen: --source-repo needs --source-revision (a moving branch is not a pin)")
    {"probe": lambda _: probe(), "gen": gen}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
