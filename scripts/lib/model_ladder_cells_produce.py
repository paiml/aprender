#!/usr/bin/env python3
"""The PRODUCER of the model ladder receipt's `cells[]` (#3712 row B, ont:release/row for #3715).

scripts/lib/model_ladder_cells.py JUDGES cells; until this file nothing WROTE them, so the judge sat
NOT ARMED on every receipt ever made (measured 2026-09-22: 0.68.1/0.68.2/0.69.1, no `cells` key). This
module closes that loop. It is driven by scripts/model_ladder.sh and has two halves:

  enrich   every inventory row gains the terms the judge derives the owed set from -- arch,
           context_length, the chat template's thinking evidence, and the memory terms -- read from
           the FILE HEADER (`apr inspect --json`), never from the file name. Cheap: runs every time.
  measure  every owed (model, verb, thinking, rung) cell is RUN on the accelerator and recorded as one
           row. Expensive (a 148k-token prefill per long cell), so model_ladder.sh runs it only under
           --cells / MODEL_LADDER_CELLS=1: nightly and release-train scale ("anything huge, must be
           nightly only", operator 2026-09-23).

THE OWED SET IS THE JUDGE'S. `owed_rungs` and `expected_modes` are imported from the judge's module,
not re-derived, so the producer can never measure a set the judge does not demand (or skip one it does).

WHAT A ROW CLAIMS IS WHAT WAS MEASURED, and an unmeasured field is null, never a guess:
  * backend/fallback come from apr's own envelope (`backend.ran`/`backend.fell_back` for run, the chat
    epilogue for chat, `used_gpu` in a serve body). `apr code` reports no backend at all, so its rows
    carry backend null -- the judge reads that as "not a pass" and the gap stays visible (it is apr's
    gap, not the gate's).
  * prompt_tokens is apr's count. chat and code print none, so those rows carry the count `apr run`
    measured for the IDENTICAL prompt text, named in `prompt_tokens_source`; a chat template only
    ever ADDS tokens to that, so it is a floor, never an overstatement.
  * a pass needs the answer to contain a NEEDLE planted at the START of the prompt. A model that
    ignored the first 147k tokens cannot pass a 148k rung by answering "Paris".
  * a pre-load capacity refusal (`GPU capacity refused: ... = N MiB, against F MiB free of T MiB`) is
    a `refused` row carrying apr's OWN arithmetic as required_bytes/available_bytes; the judge then
    decides whether it is an honest "cannot fit this host" or a co-tenant (RED).
"""
import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ladder_cells as J  # noqa: E402  (the judge: one owed-set enumeration, not two)

MIB = 1024 * 1024
NEEDLE_WORD = "TANGERINE-4417"
QUESTION = ("Question: what is the passphrase stated in the very first line of this message? "
            "Reply with the passphrase only.")
FILLER = "Log entry {n:06d}: the ferry left the harbour on time and the weather stayed calm.\n"
CAPACITY_RE = re.compile(r"=\s*([0-9.]+)\s*MiB,\s*against\s*([0-9.]+)\s*MiB free of\s*([0-9.]+)\s*MiB")


# ---------------------------------------------------------------------------------------- enrich
def _num(md, key):
    v = md.get(key)
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def thinking_evidence(tmpl):
    """-> (markers, generation_opens_think). #3723's inputs, from the template text itself."""
    if tmpl is None:
        return None, False
    markers = [m for m in ("enable_thinking", "<think>", "</think>") if m in tmpl]
    gen = tmpl.split("add_generation_prompt", 1)[1] if "add_generation_prompt" in tmpl else ""
    # The generation prompt OPENS a think block unconditionally when it emits <think> and no
    # enable_thinking switch exists anywhere to turn it off.
    return markers, ("<think>" in gen and "enable_thinking" not in tmpl)


def kv_bytes_per_token_f32(md, arch):
    """capacity.rs's term: 2 (K,V) x kv heads x head dim x 4 B, over the layers that HAVE a KV cache
    (a hybrid's full-attention layers only). None when any number is absent from the header."""
    p = f"{arch}."
    layers, kvh = _num(md, p + "block_count"), _num(md, p + "attention.head_count_kv")
    heads, emb = _num(md, p + "attention.head_count"), _num(md, p + "embedding_length")
    kl = _num(md, p + "attention.key_length") or (emb // heads if emb and heads else None)
    vl = _num(md, p + "attention.value_length") or kl
    interval = _num(md, p + "full_attention_interval")
    if None in (layers, kvh, kl, vl):
        return None
    attn_layers = layers // interval if interval else layers
    return attn_layers * kvh * (kl + vl) * 4


def inventory_terms(inspect, file_bytes):
    """The fields model_ladder_cells.judge derives the owed set from, from `apr inspect --json`."""
    md = (inspect or {}).get("metadata") or {}
    arch = md.get("general.architecture")
    if not arch and (inspect or {}).get("architecture") not in (None, "", "unknown"):
        arch = inspect["architecture"]
    tmpl = md.get("tokenizer.chat_template")
    markers, opens = thinking_evidence(tmpl)
    modes = J.expected_modes(markers, opens)
    return {
        "arch": arch or None,
        "quant": md.get("general.file_type"),
        "context_length": _num(md, f"{arch}.context_length") if arch else None,
        "thinking_markers": markers,
        "generation_opens_think": opens,
        "thinking_modes": sorted(modes) if modes else None,
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if tmpl is not None else None,
        "weights_bytes": file_bytes,
        "kv_bytes_per_token": kv_bytes_per_token_f32(md, arch) if arch else None,
        "kv_dtype": "f32",
        # NOT measured: apr computes its workspace inside the load path and prints it only in a
        # refusal. Null keeps every cell owed in full (the judge's rule), so apr's own pre-load
        # arithmetic -- recorded on a refused row -- decides fit, never a number made up here.
        "workspace_bytes": None,
    }


def gpu_memory():
    """-> (total_bytes, free_bytes). A discrete GPU per nvidia-smi; a unified-memory GPU (GB10, where
    nvidia-smi says [N/A]) per /proc/meminfo, as capacity.rs's DeviceMemory::Unified does."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        t, f = (int(x.strip()) * MIB for x in out.split(","))
        return t, f
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        pass
    try:
        mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        kb = lambda k: int(mi[k].split()[0]) * 1024  # noqa: E731
        return kb("MemTotal"), kb("MemAvailable")
    except (OSError, KeyError, ValueError):
        return None, None


# --------------------------------------------------------------------------------------- measure
def build_prompt(tokens, chars_per_token):
    """A prompt of at least `tokens` tokens at the given density: the needle FIRST, then filler, then
    the question LAST, so the answer is only reachable across the whole context."""
    head = f"The passphrase is {NEEDLE_WORD}. Remember it.\n"
    body, n = [head], 0
    size, want = len(head) + len(QUESTION), int(tokens * chars_per_token * 1.01)
    while size < want:
        line = FILLER.format(n=n)
        body.append(line); size += len(line); n += 1
    body.append(QUESTION)
    return "".join(body)


def split_thinking(text):
    """-> (answer, think_closed). think_closed is None when no think block was opened at all."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    if "<think>" in text:
        return "", False
    return text.strip(), None


def refusal(stderr):
    """apr's own pre-load arithmetic, if this was a capacity refusal -> (required, available, total)."""
    if "capacity refused" not in stderr:
        return None
    m = CAPACITY_RE.search(stderr)
    if not m:
        return (None, None, None)
    return tuple(int(float(x) * MIB) for x in m.groups())


class Runner:
    """Runs apr under the fleet GPU lock: `flock -E 75 -w WAIT LOCK choom -n 1000 -- apr ...`."""

    def __init__(self, apr, lock, wait, timeout):
        self.apr, self.lock, self.wait, self.timeout = apr, lock, wait, timeout

    def argv(self, args):
        pre = ["flock", "-E", "75", "-w", str(self.wait), self.lock, "choom", "-n", "1000", "--"] if self.lock else []
        return pre + [self.apr] + args

    def call(self, args, stdin=None):
        try:
            p = subprocess.run(self.argv(args), input=stdin, capture_output=True, text=True, timeout=self.timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {self.timeout} s"


def base_row(item, verb, mode, rid, max_tokens):
    return {"sha256": item["sha256"], "file": item["file"], "verb": verb, "thinking": mode, "context": rid,
            "prompt_tokens": None, "max_tokens": max_tokens, "think_closed": None, "answer_chars": 0,
            "ttft_ms": None, "required_bytes": None, "available_bytes": None, "verdict": "fail",
            "backend": None, "fallback": None, "rc": None, "reason": ""}


def finish(row, rc, text, stderr):
    """Shared verdict: refusal, failure, or a pass that must contain the needle."""
    row["rc"] = rc
    ref = refusal(stderr)
    if rc != 0 and ref is not None:
        row["verdict"], row["required_bytes"], row["available_bytes"] = "refused", ref[0], ref[1]
        row["reason"] = stderr.strip().splitlines()[-1] if stderr.strip() else "capacity refused"
        return row
    answer, closed = split_thinking(text or "")
    row["answer_chars"], row["think_closed"] = len(answer), closed
    if rc != 0:
        row["reason"] = f"rc {rc}: " + (stderr.strip().splitlines()[-1] if stderr.strip() else "no stderr")
    elif closed is False:
        row["reason"] = f"thinking never closed within max_tokens {row['max_tokens']}: no </think> in the output"
    elif NEEDLE_WORD not in answer:
        row["reason"] = f"answer does not contain the needle planted at token 0: {answer[:120]!r}" + (f" ... and {len(answer) - 120} more chars" if len(answer) > 120 else "")
    else:
        row["verdict"], row["reason"] = "pass", "ok"
    return row


def measure_run(R, path, prompt_file, mode, max_tokens, row):
    rc, out, err = R.call(["run", path, "--input", prompt_file, "--thinking", mode, "--format", "json",
                           "--max-tokens", str(max_tokens), "--gpu"])
    env = {}
    try:
        env = json.loads(out) if rc == 0 else {}
    except json.JSONDecodeError:
        err += " | stdout was not the --format json envelope"
        rc = rc or 3
    be = env.get("backend") or {}
    row["prompt_tokens"] = env.get("prompt_tokens")
    row["backend"] = {"gpu": "cuda", "cpu": "cpu"}.get(be.get("ran"))
    row["fallback"] = be.get("fell_back")
    return finish(row, rc, env.get("text", ""), err)


def measure_chat(R, path, prompt, mode, max_tokens, row):
    one_line = " ".join(prompt.split("\n"))  # chat reads ONE turn per line
    rc, out, err = R.call(["chat", path, "--json", "--max-tokens", str(max_tokens), "--thinking", mode, "--gpu"],
                          stdin=one_line + "\n/exit\n")
    text, env = [], {}
    for line in out.splitlines():
        try:
            d = json.loads(line)
            if isinstance(d, dict) and "backend" in d:
                env = d; continue
        except json.JSONDecodeError:
            pass
        text.append(line)
    be = env.get("backend") or {}
    row["backend"] = {"gpu": "cuda", "cpu": "cpu"}.get(be.get("ran"))
    row["fallback"] = be.get("fell_back")
    return finish(row, rc, "\n".join(text), err)


def measure_code(R, path, prompt, mode, max_tokens, row):
    rc, out, err = R.call(["code", "--model", path, "-p", "--output-format", "json", "--thinking", mode,
                           "--max-tokens", str(max_tokens), "--gpu"], stdin=prompt)
    try:
        text = json.loads(out).get("result", "") if rc == 0 else ""
    except json.JSONDecodeError:
        text, rc = "", rc or 3
    # backend stays None: `apr code` reports none (its serve child's backend is not in the envelope).
    row["reason_backend"] = "apr code's envelope carries no backend; not established"
    return finish(row, rc, text, err)


class Serve:
    """One `apr serve run` per model; killed by its RECORDED pid (its own process group), never by pattern."""

    def __init__(self, R, path, ceiling_s):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        self.log = open(os.devnull, "w")
        self.proc = subprocess.Popen(R.argv(["serve", "run", path, "--port", str(self.port), "--gpu"]),
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.up = self._wait(ceiling_s)

    def _wait(self, ceiling_s):
        end = time.time() + ceiling_s
        while time.time() < end and self.proc.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2); return True
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        return False

    def ask(self, prompt, mode, max_tokens, timeout):
        body = json.dumps({"model": "apr", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                           "stream": False, "chat_template_kwargs": {"enable_thinking": mode == "on"}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return 0, json.loads(r.read()), ""
        except urllib.error.HTTPError as e:
            return e.code, {}, e.read().decode(errors="replace")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            return 1, {}, str(e)

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait(10)
        self.log.close()


def measure_serve(S, prompt, mode, max_tokens, row, timeout):
    if not S.up:
        row["rc"], row["reason"] = 1, "apr serve never answered /health"
        return row
    rc, body, err = S.ask(prompt, mode, max_tokens, timeout)
    msg = ((body.get("choices") or [{}])[0].get("message") or {})
    text = msg.get("content") or ""
    if msg.get("reasoning_content"):
        text = f"<think>{msg['reasoning_content']}</think>{text}"
    row["prompt_tokens"] = (body.get("usage") or {}).get("prompt_tokens")
    ug = body.get("used_gpu")
    row["backend"] = None if ug is None else ("cuda" if ug else "cpu")
    row["fallback"] = None if ug is None else (not ug)
    return finish(row, rc, text, err)


def cells_for(item, L, rungs_doc):
    """[(rid, tokens, mode, verb)] this item owes -- the judge's own enumeration."""
    C = L.get("cells") or {}
    rungs, consumer_max, _ = J.load_rungs(rungs_doc, lambda _m: None)
    modes = J.expected_modes(item.get("thinking_markers"), bool(item.get("generation_opens_think")))
    modes = sorted(modes or {"on", "off"})
    out = []
    for rid, tok in J.owed_rungs(item, rungs, C.get("long_rungs_for") or {}, consumer_max):
        for verb in C.get("verbs") or []:
            for mode in modes:
                out.append((rid, tok, mode, verb))
    return out


def measure_item(R, item, path, L, rungs_doc, a):
    rows, density = [], a.chars_per_token
    owed = cells_for(item, L, rungs_doc)
    by_rung = {}
    for rid, tok, mode, verb in owed:
        by_rung.setdefault((rid, tok), []).append((mode, verb))
    serve = None
    try:
        for (rid, tok), jobs in sorted(by_rung.items(), key=lambda kv: kv[0][1] or 0):
            if tok is None:
                for mode, verb in jobs:
                    r = base_row(item, verb, mode, rid, a.max_tokens); r["reason"] = "rung has no token count"; rows.append(r)
                continue
            ctx = item.get("context_length")
            # A prompt must leave room to answer: at the `declared` rung (tok == context_length) the
            # prompt is context_length - budget - 1, which the judge's `prompt_tokens >= tok` refuses.
            # That conflict is the judge's to rule on (reported on #3712), not this file's to hide.
            target = min(tok, int(ctx) - a.max_tokens_thinking - 1) if ctx else tok
            prompt = build_prompt(target, density)
            pf = os.path.join(a.work, f"prompt-{rid}.txt")
            open(pf, "w").write(prompt)
            measured = {}
            for mode, verb in sorted(jobs, key=lambda j: j[1] != "run"):  # run first: it measures the count
                budget = a.max_tokens_thinking if mode == "on" else a.max_tokens
                row = base_row(item, verb, mode, rid, budget)
                if verb == "run":
                    for _attempt in range(3):  # the first rung's density is a guess; apr's count corrects it
                        row = measure_run(R, path, pf, mode, budget, base_row(item, verb, mode, rid, budget))
                        pt = row["prompt_tokens"]
                        if not pt:
                            break
                        density = max(1.0, len(prompt) / pt)
                        if pt >= target:
                            break
                        prompt = build_prompt(target, density)
                        open(pf, "w").write(prompt)
                    if row["prompt_tokens"]:
                        measured[mode] = row["prompt_tokens"]
                elif verb == "serve":
                    serve = serve or Serve(R, path, a.serve_ceiling)
                    row = measure_serve(serve, prompt, mode, budget, row, R.timeout)
                else:
                    row = (measure_chat if verb == "chat" else measure_code)(R, path, prompt, mode, budget, row)
                if row["prompt_tokens"] is None and measured:
                    row["prompt_tokens"] = measured.get(mode) or max(measured.values())
                    row["prompt_tokens_source"] = "apr run, identical prompt text (this verb prints no count)"
                rows.append(row)
    finally:
        if serve:
            serve.close()
    return rows


# ------------------------------------------------------------------------------------------ main
def cmd_enrich(a):
    """inventory.jsonl + models (file|path) -> enriched inventory.jsonl (same rows, same order)."""
    paths = dict(l.rstrip("\n").split("|", 1) for l in open(a.models) if "|" in l)
    R = Runner(a.apr, None, 0, 120)  # inspect reads a header: no GPU, no lock
    rows = [json.loads(l) for l in open(a.inventory) if l.strip()]
    with open(a.out, "w") as f:
        for it in rows:
            rc, out, _ = R.call(["inspect", paths.get(it["file"], ""), "--json"])
            try:
                ins = json.loads(out) if rc == 0 else {}
            except json.JSONDecodeError:
                ins = {}
            it.update(inventory_terms(ins, it.get("bytes")))
            f.write(json.dumps(it) + "\n")
    return 0


def cmd_measure(a):
    import yaml
    L = yaml.safe_load(open(a.ladder))["ladder"]
    rungs_doc = json.load(open(a.rungs))
    paths = dict(l.rstrip("\n").split("|", 1) for l in open(a.models) if "|" in l)
    R = Runner(a.apr, a.lock, a.lock_wait, a.timeout)
    rows = []
    for l in open(a.inventory):
        if l.strip():
            it = json.loads(l)
            if a.only and a.only not in (it["file"], "inv:" + it["file"]):
                continue
            rows += measure_item(R, it, paths[it["file"]], L, rungs_doc, a)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"cells: {len(rows)} row(s), {sum(r['verdict'] == 'pass' for r in rows)} pass, "
          f"{sum(r['verdict'] == 'refused' for r in rows)} refused")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpumem")
    for name in ("enrich", "measure"):
        s = sub.add_parser(name)
        s.add_argument("--apr", required=True)
        s.add_argument("--inventory", required=True)
        s.add_argument("--models", required=True, help="file|path lines, the ladder's INVENTORY")
        s.add_argument("--out", required=True)
    m = sub.choices["measure"]
    m.add_argument("--ladder", required=True)
    m.add_argument("--rungs", required=True)
    m.add_argument("--work", required=True)
    m.add_argument("--lock", default="")
    m.add_argument("--lock-wait", type=int, default=1800)
    m.add_argument("--timeout", type=int, default=3600, help="per apr call, seconds")
    m.add_argument("--serve-ceiling", type=int, default=600)
    m.add_argument("--max-tokens", type=int, default=64)
    m.add_argument("--max-tokens-thinking", type=int, default=1024, help="a think block must have room to CLOSE")
    m.add_argument("--chars-per-token", type=float, default=4.5)
    m.add_argument("--only", default="")
    a = p.parse_args(argv)
    if a.cmd == "gpumem":
        t, f = gpu_memory()
        print(json.dumps({"gpu_mem_total_bytes": t, "gpu_mem_free_bytes": f}))
        return 0
    return cmd_enrich(a) if a.cmd == "enrich" else cmd_measure(a)


if __name__ == "__main__":
    sys.exit(main())
