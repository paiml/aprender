"""The vLLM CRUX engine (#3952).

Run through ``scripts/crux_engine_vllm.sh`` — ``uv run --frozen`` over the committed ``uv.lock`` beside this
file, never an ambient python — as ``probe | gen | gen-batch``. Row contract v1 (aprender-76, #3739 comment 5765991210):
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

# Loopback only: every URL this file opens is a local server it just started. An http_proxy/HTTP_PROXY in the
# environment (sandboxed runners set one) would otherwise route 127.0.0.1 through the proxy (quorum lane 2).
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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


def prompt_opens_think(rendered: str) -> bool:
    """Does the rendered prompt END inside an opened think block? Qwen3.5's official thinking-ON template prefills
    `<think>\\n` (measured, #3990), so the generation starts INSIDE the block with no opening tag, and split_think
    — which looks for `<think>` — would hand the whole reasoning back as the answer, and an unclosed block as a
    complete one. item_doc restores the opener before splitting when this is true."""
    return rendered.rstrip().endswith("<think>")


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


# ── cache integrity (#3971): shared with the hf engine ─────────────────────────
sys.path.insert(0, str(HERE.parent / "lib"))
import crux_hf_verify  # noqa: E402
from crux_hf_verify import verified_source  # noqa: E402
from crux_sse import parse_sse  # noqa: E402


def device_label() -> str:
    import torch

    return f"cuda:0 {torch.cuda.get_device_name(0)}"


# ── gen ────────────────────────────────────────────────────────────────────
# BATCH MODE (cop, 2026-09-23): engine start-up dominated every sweep — vLLM built an engine per PROMPT. Now one
# engine per call serves every item of a batch (`gen-batch`), and `gen` is a batch of one through the same code.
# Row contract v1 is unchanged: exactly one row per item. A batch holds ONE interface — in-process (run, chat)
# or `vllm serve` (serve run, code) — because the second would be a second engine on the same card; items of
# the other interface are refused by name, so a producer splits them into two calls.
INPROC = ("run", "chat")
SERVE = ("serve run", "serve stream", "code")
VERBS = INPROC + SERVE
THINKING = ("on", "off", "unset")


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


def load_inproc(a):
    """ONE in-process engine for the whole batch. Returns (respond, interface, device), where
    respond(convo, thinking, max_tokens) -> (text, prompt_tokens, completion_tokens)."""
    from vllm import LLM, SamplingParams

    src = verified_source(a.source_repo, a.source_revision)
    llm = LLM(
        model=str(src), tokenizer=str(src), served_model_name=a.source_repo,
        dtype=a.dtype, max_model_len=a.context, seed=a.seed,
        gpu_memory_utilization=gpu_memory_utilization(), enforce_eager=True,
    )

    tok = llm.get_tokenizer()

    def respond(convo, thinking, max_tokens):
        # Greedy whatever --temperature says, as the protocol asks (the hf engine does the same).
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=a.seed)
        out = llm.chat(convo, params, use_tqdm=False, chat_template_kwargs=thinking_kwargs(thinking) or None)[0]
        rendered = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, **thinking_kwargs(thinking))
        return (out.outputs[0].text, len(out.prompt_token_ids), len(out.outputs[0].token_ids),
                prompt_opens_think(rendered))

    return respond, "LLM.chat", device_label()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve_session(a, log_path: Path):
    """ONE `vllm serve` (the pinned version's own OpenAI-compatible server) for the whole batch, stopped at the
    end. Yields (respond, interface, device) like load_inproc; respond also saves the raw response when given
    a path."""
    src = verified_source(a.source_repo, a.source_revision)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(src))  # the server renders with this same template: ask it what it rendered
    port = free_port()
    cmd = [
        str(Path(sys.executable).parent / "vllm"), "serve", str(src), "--served-model-name", a.source_repo,
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", a.dtype, "--max-model-len", str(a.context), "--seed", str(a.seed),
        "--gpu-memory-utilization", str(gpu_memory_utilization()), "--enforce-eager",
    ]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 600  # a first start compiles kernels
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited {proc.returncode} before answering (log: {log_path.name})")
            try:
                NO_PROXY_OPENER.open(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("vllm serve did not answer /v1/models within 600 s")
                time.sleep(0.5)

        def respond(convo, thinking, max_tokens, resp_path=None, stream=False):
            body = {"model": a.source_repo, "messages": convo, "max_tokens": max_tokens, "temperature": 0.0,
                    "seed": a.seed}
            kw = thinking_kwargs(thinking)
            if kw:
                body["chat_template_kwargs"] = kw
            if stream:
                # `serve stream` (#3962): the same server, stream=true; usage rides the last chunk
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            body_bytes = NO_PROXY_OPENER.open(req, timeout=1800).read()
            if stream:
                sse = body_bytes.decode("utf-8", "replace")
                if resp_path is not None:
                    resp_path.with_suffix(".sse.txt").write_text(sse, encoding="utf-8")
                raw, usage, _ = parse_sse(sse, terminal="done")  # vLLM always ends a stream with [DONE]
                return (raw, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        prompt_opens_think(tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True,
                                                                   **thinking_kwargs(thinking))))
            resp = json.loads(body_bytes)
            if resp_path is not None:
                resp_path.write_text(json.dumps(resp, indent=1), encoding="utf-8")
            msg = resp["choices"][0]["message"]
            raw = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                raw = f"<think>{reasoning}</think>{raw}"
            usage = resp.get("usage") or {}
            return (raw, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    prompt_opens_think(tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True,
                                                               **thinking_kwargs(thinking))))

        yield respond, "vllm serve", f"{device_label()} (vllm serve)"
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


def item_doc(it: dict, respond, interface: str, device: str, resp_path: Path | None) -> dict:
    """One item through an already-loaded engine: `run` is one reply, `chat` the conversation turn by turn."""
    messages = load_messages(it["messages"])
    counts = {"prompt_tokens": 0, "completion_tokens": 0}

    opened = []

    def one(convo):
        if interface == "vllm serve":
            res = respond(convo, it["thinking"], it["max_tokens"], resp_path, stream=it["verb"] == "serve stream")
        else:
            res = respond(convo, it["thinking"], it["max_tokens"])
        text, n_prompt, n_completion = res[:3]
        if len(res) > 3 and res[3]:
            # the prompt opened the think block (#3990): restore the opener so the split sees the block
            opened.append(True)
            text = "<think>" + text
        counts["prompt_tokens"] = n_prompt  # the FINAL turn's prompt holds the whole conversation
        if n_completion is not None and counts["completion_tokens"] is not None:
            counts["completion_tokens"] += n_completion
        else:
            counts["completion_tokens"] = None
        return text

    turns = None
    if it["verb"] == "chat":
        raw, turns, _ = conversation_turns(messages, one)
    else:
        raw = one(messages)
    answer, reasoning = split_think(raw)
    doc = {"text": answer, "raw_text": raw}
    if turns is not None:
        doc["turns"] = turns  # every assistant answer, in order; `text` is the last of them
    if reasoning:
        doc["reasoning"] = reasoning
    if it["verb"] == "serve stream":
        interface = interface + " (stream)"
    doc["reported"] = {"interface": interface, "thinking_requested": it["thinking"],
                       "thinking_emitted": bool(reasoning), "prompt_opens_think": bool(opened), **counts,
                       "device": device}
    return doc


def item_error(it: dict) -> str | None:
    """Why an item cannot be run at all, by name, before any engine is involved."""
    for k in ("prompt_id", "verb", "messages", "thinking", "max_tokens"):
        if k not in it:
            return f"batch item has no {k!r}"
    if it["verb"] not in VERBS:
        return f"batch item verb {it['verb']!r} is not one of {', '.join(VERBS)}"
    if it["thinking"] not in THINKING:
        return f"batch item thinking {it['thinking']!r} is not one of {', '.join(THINKING)}"
    if not isinstance(it["max_tokens"], int) or it["max_tokens"] < 1:
        return f"batch item max_tokens {it['max_tokens']!r} is not a positive integer"
    return None


def run_batch(a, items: list) -> None:
    """Every item gets exactly one row, in order; the engine is loaded at most ONCE for the batch."""
    manifest, work = env_paths()
    check_sha(a.model_sha256)
    batch_id = f"{os.getpid()}-{time.time_ns()}"
    slots = []
    for it in items:
        verb = it.get("verb") if it.get("verb") in VERBS else "invalid"
        d = workdir(work, a.model_sha256, verb)
        stem = f"{ENGINE}-{it.get('prompt_id', 'noid')}-{it.get('thinking', 'unset')}"
        out, err = d / f"{stem}.json", d / f"{stem}.err"
        row = {
            "kind": "gen", "engine": ENGINE, "model_sha256": a.model_sha256, "host": a.host, "verb": it.get("verb"),
            "thinking": it.get("thinking"), "backend": a.backend, "prompt_id": it.get("prompt_id"),
            "rc": 0, "stdout": str(out), "stderr": str(err), "refused": None,
            # The comparison this row is: apr's file (model_sha256) against these SOURCE weights, never the file.
            "source": {"repo": a.source_repo, "revision": a.source_revision, "dtype": a.dtype, "compares": COMPARES},
            "gpu_memory_utilization": gpu_memory_utilization(),
            "batch": {"id": batch_id, "size": len(items)},
        }
        slots.append({"it": it, "row": row, "d": d, "stem": stem, "out": out, "err": err, "reason": item_error(it)})

    def refuse(slot, reason):
        slot["err"].write_text(reason, encoding="utf-8")
        slot["row"].update(rc=None, stdout=None, refused=reason)

    live = [s for s in slots if s["reason"] is None]
    for s in slots:
        if s["reason"] is not None:
            refuse(s, s["reason"])
    inproc = [s for s in live if s["it"]["verb"] in INPROC]
    serve = [s for s in live if s["it"]["verb"] in SERVE]
    if inproc and serve:
        for s in serve:
            refuse(s, "this batch also holds in-process items (run, chat), and a second interface would be a second "
                      "engine on the same card; send serve/code items in their own gen-batch call")
        serve = []
    group = inproc or serve
    engine_log = work / a.model_sha256[:12] / f"{ENGINE}-batch-{batch_id}.engine.log"
    server_log = work / a.model_sha256[:12] / f"{ENGINE}-batch-{batch_id}.server.log"
    for s in group:
        s["row"]["engine_log"] = str(engine_log)
    if group:
        try:
            preflight(a)
        except Exception as e:
            for s in group:
                refuse(s, refusal(e))
            group = []
    if group:
        with fd2_to(engine_log), contextlib.ExitStack() as stack:
            try:
                if group is inproc:
                    respond, interface, device = load_inproc(a)
                else:
                    respond, interface, device = stack.enter_context(serve_session(a, server_log))
            except Exception as e:
                # The load itself failed: every item of the batch is refused with the engine's own root cause.
                reason = refusal(e, engine_log, server_log)
                for s in group:
                    refuse(s, reason)
            else:
                for s in group:
                    try:
                        resp_path = s["d"] / f"{s['stem']}.resp.json" if interface == "vllm serve" else None
                        doc = item_doc(s["it"], respond, interface, device, resp_path)
                        s["out"].write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                        s["err"].write_text("", encoding="utf-8")
                    except Exception as e:
                        refuse(s, refusal(e))
    for s in slots:
        append_row(manifest, s["row"])


def gen(a) -> None:
    """One cell: a batch of one, through exactly the code a batch uses."""
    run_batch(a, [{"prompt_id": a.prompt_id, "verb": a.verb, "messages": a.messages, "thinking": a.thinking,
                   "max_tokens": a.max_tokens}])


def gen_batch(a) -> None:
    """`--batch <jsonl>`: one item per line — {"prompt_id", "verb", "messages": <path>, "thinking",
    "max_tokens"} — all against ONE engine load."""
    items = []
    with open(a.batch, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except ValueError as e:
                    die(f"--batch line {n} is not JSON: {e}")
    if not items:
        die("--batch holds no items")
    run_batch(a, items)


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> None:
    import crux_proc  # scripts/lib is on sys.path (see the verify import above)

    crux_proc.install()  # a stopped driver takes its engine children with it (#3952, measured on gx10)
    if argv and argv[0] in ("tok", "tmpl", "greedy"):
        die(f"`{argv[0]}` is `none` for vLLM: it tokenizes and renders through the transformers tokenizer the "
            "hf engine already reports")
    p = argparse.ArgumentParser(prog="crux_engine_vllm.sh")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")

    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--model-sha256", required=True)
    g.add_argument("--verb", required=True, choices=list(VERBS))
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

    b = sub.add_parser("gen-batch")
    b.add_argument("--batch", required=True, help="JSONL: one {prompt_id, verb, messages, thinking, max_tokens} per line")
    b.add_argument("--model", required=True)
    b.add_argument("--model-sha256", required=True)
    b.add_argument("--backend", required=True, choices=["gpu", "cpu"])
    b.add_argument("--host", required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--temperature", type=float, required=True)
    b.add_argument("--context", type=int, required=True)
    b.add_argument("--source-repo")
    b.add_argument("--source-revision")
    b.add_argument("--dtype", default="bfloat16")

    a = p.parse_args(argv)
    if a.cmd in ("gen", "gen-batch") and a.source_repo and not a.source_revision:
        die(f"{a.cmd}: --source-repo needs --source-revision (a moving branch is not a pin)")
    {"probe": lambda _: probe(), "gen": gen, "gen-batch": gen_batch}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
