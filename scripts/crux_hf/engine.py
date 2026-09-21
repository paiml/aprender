"""The Hugging Face transformers CRUX engine (#3739, PMAT-3778).

Run through ``scripts/crux_engine_hf.sh`` — ``uv run --frozen`` over the committed ``uv.lock`` beside this
file, never an ambient python — as ``probe | gen | tok | tmpl | greedy``. Row contract v1 (aprender-76,
#3739 comment 5765991210): every subcommand but ``probe`` appends exactly ONE JSONL row to
``$CRUX_MANIFEST`` and writes its artifacts under ``$CRUX_WORK/<model_sha256[:12]>/<verb|kind>/``. A cell
HF cannot run is still a row — ``rc: null`` and ``refused`` carrying the exception's own words — never an
absence and never a guess.

Generation is greedy (``do_sample=False``) whatever ``--temperature`` says, as the protocol asks; no rate
is computed from a clock (``reported`` carries token counts only, and whatever the server itself reports).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "uv.lock"


def die(msg: str, code: int = 2) -> None:
    print(f"crux_engine_hf: {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── probe ──────────────────────────────────────────────────────────────────
def probe() -> None:
    try:
        import jinja2
        import tokenizers
        import torch
        import transformers
    except Exception as e:  # the environment itself is the refusal
        die(f"the locked environment does not import: {type(e).__name__}: {e}", 3)
    cuda = torch.cuda.is_available()
    parts = [
        f"transformers={transformers.__version__}",
        f"torch={torch.__version__}",
        f"tokenizers={tokenizers.__version__}",
        f"jinja2={jinja2.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={str(cuda).lower()}",
    ]
    if cuda:
        parts.append(f"device={torch.cuda.get_device_name(0).replace(' ', '_')}")
        parts.append(f"capability=sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
        parts.append(f"arch_list={','.join(torch.cuda.get_arch_list())}")
    parts.append(f"lock_sha256={sha256_file(LOCK)}")
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


def refusal(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:4000]


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


def source_of(a: argparse.Namespace) -> tuple[str, str | None]:
    """What HF loads: the source repo at its revision when given (a GGUF cell runs the bf16 source), else --model."""
    if a.source_repo:
        return a.source_repo, a.source_revision
    return a.model, None


def device_of(backend: str):
    import torch

    if backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"torch.cuda.is_available() is False (torch {torch.__version__}, built for CUDA {torch.version.cuda})"
            )
        return "cuda"
    return "cpu"


def device_label(model) -> str:
    """The device the weights are ON, read from the model — `cpu`, or `cuda:<i> <name>`. The judge holds a row
    to its lane with it (aprender-76: required; a row on the wrong device is no answer)."""
    import torch

    d = next(model.parameters()).device
    if d.type == "cuda":
        return f"cuda:{d.index} {torch.cuda.get_device_name(d)}"
    return d.type


# ── gen ────────────────────────────────────────────────────────────────────
def conversation_turns(messages: list, respond) -> tuple[str, list, list]:
    """Drives a `chat` cell (aprender-76, #3739): `messages` holds the USER turns (a system message, if any,
    kept in place); every user turn is answered with the conversation so far, and the ANSWER — reasoning
    split off — is appended as the assistant turn before the next user turn. Returns (raw final reply, the
    answer of every turn in order, the raw reply of every turn)."""
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


def gen_generate(a, messages, device):
    """`run`: one reply to the messages as given. `chat`: the conversation driven turn by turn. The model is
    loaded ONCE either way."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, rev = source_of(a)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    # device_map, not .to(): loaded straight onto the lane's device (see main() for why CUDA is hidden on cpu).
    model = AutoModelForCausalLM.from_pretrained(src, revision=rev, dtype=getattr(torch, a.dtype), device_map=device)
    counts = {"prompt_tokens": 0, "completion_tokens": 0}

    def respond(convo):
        torch.manual_seed(a.seed)
        inputs = tok.apply_chat_template(
            convo, add_generation_prompt=True, return_tensors="pt", return_dict=True, **thinking_kwargs(a.thinking)
        ).to(device)
        n_prompt = int(inputs["input_ids"].shape[1])
        out = model.generate(**inputs, max_new_tokens=a.max_tokens, do_sample=False)
        new = out[0][n_prompt:]
        counts["prompt_tokens"] = n_prompt  # the FINAL turn's prompt, which holds the whole conversation
        counts["completion_tokens"] += int(new.shape[0])
        return tok.decode(new, skip_special_tokens=True)

    if a.verb == "chat":
        raw, answers, _ = conversation_turns(messages, respond)
        return raw, {**counts, "device": device_label(model)}, answers
    return respond(messages), {**counts, "device": device_label(model)}, None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gen_serve(a, messages, device, d: Path, stem: str):
    """`transformers serve`, the pinned version's own server, one request, then stopped."""
    src, rev = source_of(a)
    port = free_port()
    cmd = [
        str(Path(sys.executable).parent / "transformers"), "serve", "--host", "127.0.0.1", "--port", str(port),
        "--device", device, "--dtype", a.dtype, "--default-seed", str(a.seed), "--no-enable-cors",
    ]
    log = open(d / f"{stem}.server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 180
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"transformers serve exited {proc.returncode} before answering (log: {stem}.server.log)")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("transformers serve did not answer /v1/models within 180 s")
                time.sleep(0.5)
        model_id = f"{src}@{rev}" if rev else src
        body = {"model": model_id, "messages": messages, "max_tokens": a.max_tokens, "temperature": 0.0, "seed": a.seed}
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
        if msg.get("reasoning_content"):
            raw = f"<think>{msg['reasoning_content']}</think>{raw}"
        usage = resp.get("usage") or {}
        # The server is a separate process: its device is the one it was TOLD, and on the cpu lane CUDA is
        # hidden from it too (inherited environment), so "cpu" there is the only device it can have used.
        import torch

        label = f"cuda:0 {torch.cuda.get_device_name(0)}" if device == "cuda" else "cpu"
        return raw, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                     "device": f"{label} (transformers serve --device {device})"}
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
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
    stem = f"hf-{a.prompt_id}-{a.thinking}"
    out, err = d / f"{stem}.json", d / f"{stem}.err"
    row = {
        "kind": "gen", "engine": "hf", "model_sha256": a.model_sha256, "host": a.host, "verb": a.verb,
        "thinking": a.thinking, "backend": a.backend, "prompt_id": a.prompt_id,
        "rc": 0, "stdout": str(out), "stderr": str(err), "refused": None,
        "source": {"repo": a.source_repo, "revision": a.source_revision, "dtype": a.dtype},
    }
    try:
        messages = load_messages(a.messages)
        device = device_of(a.backend)
        turns = None
        if a.verb in ("run", "chat"):
            raw, counts, turns = gen_generate(a, messages, device)
        else:
            raw, counts = gen_serve(a, messages, device, d, stem)
        answer, reasoning = split_think(raw)
        doc = {"text": answer}
        if turns is not None:
            doc["turns"] = turns  # every assistant answer, in order; `text` is the last of them
        if reasoning:
            doc["reasoning"] = reasoning
        doc["reported"] = {
            "interface": "generate" if a.verb in ("run", "chat") else "transformers serve",
            "thinking_requested": a.thinking, "thinking_emitted": bool(reasoning), **counts,
        }
        out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        err.write_text("", encoding="utf-8")
    except Exception as e:
        err.write_text(refusal(e), encoding="utf-8")
        row.update(rc=None, stdout=None, refused=refusal(e))
    append_row(manifest, row)


# ── tok / tmpl ─────────────────────────────────────────────────────────────
def tok(a) -> None:
    manifest, work = env_paths()
    check_sha(a.model_sha256)
    d = workdir(work, a.model_sha256, "tok")
    ids_path = d / f"hf-{a.prompt_id}.json"
    row = {
        "kind": "tok", "engine": "hf", "model_sha256": a.model_sha256, "host": a.host, "prompt_id": a.prompt_id,
        "input": str(Path(a.input).resolve()), "ids": str(ids_path), "add_special": a.add_special, "refused": None,
    }
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(a.source_repo, revision=a.source_revision)
        text = Path(a.input).read_bytes().decode("utf-8")
        ids = t.encode(text, add_special_tokens=a.add_special)
        ids_path.write_text(json.dumps({"tokens": ids}), encoding="utf-8")
    except Exception as e:
        row.update(ids=None, refused=refusal(e))
    append_row(manifest, row)


def tmpl(a) -> None:
    manifest, work = env_paths()
    check_sha(a.model_sha256)
    d = workdir(work, a.model_sha256, "tmpl")
    rendered = d / f"hf-tmpl-{a.prompt_id}-{a.thinking}.txt"
    row = {
        "kind": "tmpl", "engine": "hf", "model_sha256": a.model_sha256, "host": a.host, "prompt_id": a.prompt_id,
        "thinking": a.thinking, "messages": str(Path(a.messages).resolve()), "rendered": str(rendered),
        "refused": None,
        # #3755's render_chat_template_reference.py is the named producer once it is on main; until then the
        # same call, inline — named here so a reader knows which produced the bytes.
        "producer": "apply_chat_template(tokenize=False, add_generation_prompt=True) inline",
    }
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(a.source_repo, revision=a.source_revision)
        text = t.apply_chat_template(
            load_messages(a.messages), tokenize=False, add_generation_prompt=True, **thinking_kwargs(a.thinking)
        )
        rendered.write_bytes(text.encode("utf-8"))
    except Exception as e:
        row.update(rendered=None, refused=refusal(e))
    append_row(manifest, row)


# ── greedy ─────────────────────────────────────────────────────────────────
def greedy(a) -> None:
    manifest, work = env_paths()
    check_sha(a.model_sha256)
    d = workdir(work, a.model_sha256, "greedy")
    tokens_path = d / f"hf-greedy-{a.prompt_id}.json"
    logits_path = d / f"hf-greedy-{a.prompt_id}.logits.npy"
    row = {
        "kind": "greedy", "engine": "hf", "model_sha256": a.model_sha256, "host": a.host, "prompt_id": a.prompt_id,
        "steps": a.steps, "tokens": str(tokens_path), "logits": str(logits_path) if a.logits else None,
        "refused": None,
    }
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device_of(a.backend)
        t = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=device)
        if a.messages:
            inputs = t.apply_chat_template(
                load_messages(a.messages), add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:
            inputs = t(Path(a.prompt_file).read_text(encoding="utf-8"), return_tensors="pt")
        inputs = inputs.to(device)
        n_prompt = int(inputs["input_ids"].shape[1])
        out = model.generate(
            # No min_new_tokens: forcing N steps suppresses EOS and makes HF continue past the answer
            # (measured: "4." became "4.000000"), which would put the first divergence after EOS.
            **inputs, max_new_tokens=a.steps, do_sample=False,
            output_logits=a.logits, return_dict_in_generate=True,
        )
        gen_ids = out.sequences[0][n_prompt:].tolist()
        doc = {"prompt_ids": inputs["input_ids"][0].tolist(), "generated_ids": gen_ids, "device": device_label(model)}
        if a.logits:
            steps = torch.stack([l[0] for l in out.logits]).float().cpu()
            np.save(logits_path, steps.numpy())
            # A degenerate answer has a cause the ids cannot show: NaN logits make argmax 0 at every step,
            # which decodes as "!!!!" (measured on the GPU lane, #3739). Reported per step, never judged.
            doc["step_stats"] = [
                {"nan": int(torch.isnan(s).sum()), "min": float(torch.nan_to_num(s).min()),
                 "max": float(torch.nan_to_num(s).max())}
                for s in steps
            ]
        tokens_path.write_text(json.dumps(doc), encoding="utf-8")
    except Exception as e:
        row.update(tokens=None, logits=None, refused=refusal(e))
    append_row(manifest, row)


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="crux_engine_hf.sh")
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

    t = sub.add_parser("tok")
    t.add_argument("--model-sha256", required=True)
    t.add_argument("--source-repo", required=True)
    t.add_argument("--source-revision")
    t.add_argument("--prompt-id", required=True)
    t.add_argument("--input", required=True)
    t.add_argument("--add-special", action="store_true")
    t.add_argument("--host", required=True)

    m = sub.add_parser("tmpl")
    m.add_argument("--model-sha256", required=True)
    m.add_argument("--source-repo", required=True)
    m.add_argument("--source-revision")
    m.add_argument("--prompt-id", required=True)
    m.add_argument("--messages", required=True)
    m.add_argument("--thinking", required=True, choices=["on", "off", "unset"])
    m.add_argument("--host", required=True)

    r = sub.add_parser("greedy")
    r.add_argument("--model", required=True)
    r.add_argument("--model-sha256", required=True)
    r.add_argument("--prompt-id", required=True)
    r.add_argument("--prompt-file")
    r.add_argument("--messages")
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--logits", action="store_true")
    r.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--host", required=True)

    a = p.parse_args(argv)
    # THE CPU LANE MUST NOT TOUCH THE GPU. It takes no GPU lock (gpu-q rule clause 2), and with CUDA merely
    # VISIBLE, transformers 5.17 loaded a `--backend cpu` model through a path that reached the 4090 and
    # produced token 0 at every step (aprender-76's cpu lane, #3739). So CUDA is hidden from this process
    # BEFORE torch is imported — for cpu cells and for tok/tmpl, which never need a device.
    if getattr(a, "backend", "cpu") == "cpu" and a.cmd != "probe":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.cmd == "gen" and a.source_repo and not a.source_revision:
        die("gen: --source-repo needs --source-revision (a moving branch is not a pin)")
    if a.cmd == "greedy" and not (a.messages or a.prompt_file):
        die("greedy: --messages or --prompt-file is required")
    {"probe": lambda _: probe(), "gen": gen, "tok": tok, "tmpl": tmpl, "greedy": greedy}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
