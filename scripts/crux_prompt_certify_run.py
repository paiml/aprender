#!/usr/bin/env python3
"""crux_prompt_certify_run: produce the rows crux_prompt_certify.py admits prompts from (#3962).

One invocation is ONE (model, leg), meant to run inside ONE `gpu-q` hold, so the lock is taken per
leg, never for a whole certification and never per prompt:

  --leg ggml   llama-server on a GGUF (the BF16 GGUF for ggml@bf16, the quantized one for ggml@quant),
               driven through scripts/lib/crux_openai_client.py, the client every CRUX serve cell uses.
               A multi-turn prompt is answered turn by turn with the history so far, exactly as the hf
               and vLLM drivers do (conversation_turns). `--reasoning-format none` keeps the think block
               in the reply, so an unclosed one reaches the oracle as unclosed.
  --leg hf     scripts/crux_engine_hf.sh gen   (aprender-83's driver; it appends its own row)
  --leg vllm   scripts/crux_engine_vllm.sh gen (likewise), both on the PINNED source weights.

Every cell is greedy: temperature 0, seed 0, max_tokens from the prompt's own `max_tokens[thinking]`,
enable_thinking explicit. Rows are CRUX row contract v1, appended to --manifest. A cell whose row is
already in the manifest with rc 0 is skipped, so an interrupted leg resumes where it stopped.

Exit: 0 every cell produced a row (a refused row is still a row) · 2 usage/ENV.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "scripts" / "lib" / "crux_openai_client.py"


def die(msg: str) -> None:
    print(f"crux_prompt_certify_run: {msg}", file=sys.stderr)
    sys.exit(2)


def done_cells(manifest: Path, engine: str, key: str) -> set:
    got = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ident = r.get("model_sha256") if engine == "llama.cpp" else (r.get("source") or {}).get("revision")
            if r.get("kind") == "gen" and r.get("engine") == engine and ident == key and r.get("rc") == 0:
                got.add((r["prompt_id"], r["thinking"]))
    return got


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(a, log: Path):
    port = free_port()
    argv = [a.llama_server, "-m", a.gguf, "--port", str(port), "--host", "127.0.0.1", "-ngl", "999",
            "-c", str(a.context), "--jinja", "--reasoning-format", "none", "--seed", "0", "-np", "1"]
    proc = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        if proc.poll() is not None:
            die(f"llama-server exited {proc.returncode} while loading; see {log}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return proc, url, argv
        except Exception:
            pass
        time.sleep(1)
    os.killpg(proc.pid, signal.SIGKILL)
    die(f"llama-server never became healthy; see {log}")


def ggml_leg(a, prompts, thinking_modes, manifest: Path, work: Path) -> None:
    skip = done_cells(manifest, "llama.cpp", a.gguf_sha256)
    todo = [(p, t) for t in thinking_modes for p in prompts if (p["id"], t) not in skip]
    if not todo:
        print("ggml: nothing to do")
        return
    d = work / a.gguf_sha256[:12] / "ggml"
    d.mkdir(parents=True, exist_ok=True)
    proc, url, argv = start_server(a, d / "llama-server.log")
    version = subprocess.run([a.llama_server, "--version"], capture_output=True, text=True).stderr.strip()
    try:
        for p, thinking in todo:
            stem = f"llama.cpp-{p['id']}-{thinking}"
            out, err = d / f"{stem}.json", d / f"{stem}.err"
            row = {"kind": "gen", "engine": "llama.cpp", "model_sha256": a.gguf_sha256, "host": a.host,
                   "verb": "serve run", "thinking": thinking, "backend": "gpu", "prompt_id": p["id"],
                   "rc": 0, "stdout": str(out), "stderr": str(err), "refused": None,
                   "server": {"argv": argv, "version": version}}
            convo, turns, why = [], [], None
            extra = json.dumps({"chat_template_kwargs": {"enable_thinking": thinking == "on"}})
            for m in p["messages"]:
                convo.append(m)
                if m["role"] != "user":
                    continue
                mfile, tout = d / f"{stem}.t{len(turns)}.messages.json", d / f"{stem}.t{len(turns)}.json"
                mfile.write_text(json.dumps({"messages": convo}), encoding="utf-8")
                r = subprocess.run([sys.executable, str(CLIENT), "--url", url, "--model", "certify",
                                    "--messages", str(mfile), "--max-tokens", str(p["max_tokens"][thinking]),
                                    "--temperature", "0", "--seed", "0", "--extra", extra, "--out", str(tout)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    why = f"client exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}"
                    break
                text = json.loads(tout.read_text(encoding="utf-8"))["text"]
                turns.append(text)
                convo.append({"role": "assistant", "content": text})
            if why:
                err.write_text(why, encoding="utf-8")
                row.update(rc=None, stdout=None, refused=why)
            else:
                out.write_text(json.dumps({"text": turns[-1], "turns": turns}, ensure_ascii=False), encoding="utf-8")
                err.write_text("", encoding="utf-8")
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"ggml {p['id']} thinking={thinking}: {'refused' if why else 'ok'}", flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def driver_leg(a, prompts, thinking_modes, manifest: Path, work: Path) -> None:
    engine = a.leg
    script = Path(a.drivers) / "scripts" / f"crux_engine_{engine}.sh"
    if not script.exists():
        die(f"{script} not found (the {engine} driver lives on aprender-83's branch until it lands)")
    skip = done_cells(manifest, engine, a.source_revision)
    env = dict(os.environ, CRUX_MANIFEST=str(manifest), CRUX_WORK=str(work))
    d = work / "messages"
    d.mkdir(parents=True, exist_ok=True)
    for thinking in thinking_modes:
        for p in prompts:
            if (p["id"], thinking) in skip:
                continue
            mfile = d / f"{p['id']}.json"
            mfile.write_text(json.dumps({"messages": p["messages"]}), encoding="utf-8")
            r = subprocess.run(["bash", str(script), "gen", "--model", a.gguf, "--model-sha256", a.gguf_sha256,
                                "--verb", "chat", "--prompt-id", p["id"], "--messages", str(mfile),
                                "--thinking", thinking, "--backend", "gpu", "--host", a.host,
                                "--max-tokens", str(p["max_tokens"][thinking]), "--seed", "0", "--temperature", "0",
                                "--context", str(a.context), "--source-repo", a.source_repo,
                                "--source-revision", a.source_revision], env=env, capture_output=True, text=True)
            print(f"{engine} {p['id']} thinking={thinking}: driver exit {r.returncode}"
                  + (f" {r.stderr.strip()[-200:]}" if r.returncode else ""), flush=True)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="crux_prompt_certify_run.py")
    ap.add_argument("--leg", required=True, choices=["ggml", "hf", "vllm"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gguf", required=True, help="ggml: the GGUF served; hf/vllm: the quant the cell joins on")
    ap.add_argument("--gguf-sha256", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--thinking", default="on,off")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--only", default="", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--llama-server", default="llama-server")
    ap.add_argument("--drivers", default=str(ROOT), help="a checkout holding scripts/crux_engine_{hf,vllm}.sh")
    ap.add_argument("--source-repo")
    ap.add_argument("--source-revision")
    a = ap.parse_args(argv)
    if a.leg != "ggml" and not (a.source_repo and a.source_revision):
        die(f"--leg {a.leg} needs --source-repo and --source-revision (a moving branch is not a pin)")
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))["prompts"]
    if a.only:
        keep = set(a.only.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    modes = [t for t in a.thinking.split(",") if t]
    manifest, work = Path(a.manifest).resolve(), Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (ggml_leg if a.leg == "ggml" else driver_leg)(a, prompts, modes, manifest, work)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
