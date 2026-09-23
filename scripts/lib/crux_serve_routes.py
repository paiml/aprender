#!/usr/bin/env python3
"""crux_serve_routes.py: the CRUX serve verb over EVERY route `apr serve` mounts (#3962).

WHERE THE ROUTES COME FROM. The running server's own `GET /`. The realizar router
builds that index from the same table it mounts (router.rs `route_table` ->
`advertised_routes`), so what is listed is what is served, for this file format,
on this build. Nothing here restates the route list. model_ladder.sh used to grep
router.rs and then filter it through a hand-written regex of four paths, so a
fifth generation route was never probed. Here the universe is the index, and
every entry in it must be CLASSIFIED below:
  GENERATION      a route that turns a prompt into text. It is driven with every
                  CRUX prompt of the verb, in each mode its wire supports.
  NOT_GENERATION  a route that returns no model text (health, tokenize, embeddings,
                  metrics...). It is listed in the receipt with its reason and is
                  never counted as a green serve cell.
An indexed route that is in NEITHER table is RED `unclassified_route`, so a new
endpoint cannot ship unprobed just because nobody added it to a list. This is the
must-RED case scripts/check_crux_serve_code.sh plants. A server whose `GET /`
returns no route index is RED `no_route_index`: its surface cannot be derived, so
it cannot be covered. Measured at fc942f6be, that is apr's APR-CPU fallback
router and its safetensors-inspection router.

WHAT A SERVE CELL IS. One (route, mode, prompt). The mode is `nonstream` (verb key
`serve run`) or `stream` (verb key `serve stream`). The reply is written as the
row contract v1 JSON the judge reads with parse_engine_json:
  {"text": <final turn, raw>, "turns": [...]?, "protocol_fault": str|null,
   "refused": str|null,
   "reported": {"route", "mode", "http", "chunks", "terminal", "finish_reason",
                "rendered_prompts", "thinking_requested", "thinking_control"}}
`text` is RAW, think blocks included. The oracle (scripts/lib/crux_oracles.py)
strips those itself, so an unclosed one comes out as `think_unclosed`, never as
an empty answer.

`protocol_fault` is non-null when the wire itself broke, and the cell is then RED
whatever the text says (#3957 F4c, quorum Q6):
  http_<code>          not 200
  empty_text           200 with no text in the route's reply field
  stream_zero_deltas   a stream that delivered no content chunk
  stream_truncated     a stream that ended without its terminal event: OpenAI SSE
                       `data: [DONE]`, realizar SSE `event: done`, or an NDJSON
                       object with `"done": true`
  stream_no_finish     an OpenAI SSE stream where no chunk carried finish_reason
  unparseable          a body or chunk that is not the route's JSON
  unreachable          the connection failed or timed out
A protocol fault is a MEASUREMENT, not a refusal: the cell ran and the route broke.

MULTI-TURN. A prompt with N user turns is answered one turn at a time. Each
request carries the history so far, including the server's own earlier replies,
exactly as the vLLM driver's conversation_turns does. Every reply goes in `turns`.

RAW-PROMPT ROUTES (/generate, /v1/completions, /stream/generate, ...) apply no
chat template, so sending them the bare user text would ask a different question
from the one the chat routes get. The conversation is therefore RENDERED by the
reference renderer, llama.cpp's `/apply-template` on the same GGUF, with
enable_thinking set explicitly. Each rendered string's sha256 goes into the row
(quorum Q5: the rendered prompt string is recorded per engine). With no renderer,
those routes are REFUSED with the reason; they are never sent unrendered text.

THINKING. apr serve reads no per-request thinking toggle on any of its routers
(measured: none reads enable_thinking, chat_template_kwargs or think). So
`thinking_control` records what was requested and that apr cannot be told it.
The Qwen3 template pre-fills an empty think block, which makes apr serve
thinking-OFF by construction.

NO CLOCK IS READ for a rate. TTFT and decode rate belong to
`apr test llm bench` (PERF-009).

CLI
  crux_serve_routes.py plan  (--url U | --index-file F)   prints the classified universe as one JSON
  crux_serve_routes.py drive --url U --route 'POST /x' --mode nonstream|stream
                             --prompt-file P.json --max-tokens N [--temperature 0] [--seed 0]
                             [--render-url R] [--thinking off|on] [--device D] --out O.json
Exit codes:
  drive  0  an answer was read with no protocol fault
         3  protocol fault (the JSON names it)
         4  refused, not measured (no renderer; route not a generation route;
            mode the wire lacks)
         2  usage
  plan   0  every indexed route classified
         1  unclassified route(s), or no index
         2  usage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request

# kind -> how the request is built and where the reply's text is read from.
#   chat_messages   OpenAI chat: messages in; choices[].message.content, or delta.content when streaming
#   text_prompt     OpenAI completions: rendered prompt in; choices[].text
#   raw_generate    realizar GenerateRequest: rendered prompt in; `text`
#   raw_sse         realizar /stream/generate: always SSE, `event: token` then `event: done`
#   raw_batch       prompts: [rendered] in; results[0].text
#   ollama_chat     messages in; message.content, NDJSON when streaming
#   ollama_generate prompt in (the server applies the template); `response`, NDJSON when streaming
# `modes` are the modes the WIRE supports: a route that cannot stream is never asked to.
GENERATION = {
    "POST /v1/chat/completions":        {"kind": "chat_messages", "modes": ("nonstream", "stream")},
    "POST /v1/chat/completions/stream": {"kind": "chat_messages", "modes": ("stream",)},
    "POST /v1/completions":             {"kind": "text_prompt", "modes": ("nonstream", "stream")},
    "POST /generate":                   {"kind": "raw_generate", "modes": ("nonstream",)},
    "POST /stream/generate":            {"kind": "raw_sse", "modes": ("stream",)},
    "POST /realize/generate":           {"kind": "raw_sse", "modes": ("stream",)},
    "POST /batch/generate":             {"kind": "raw_batch", "modes": ("nonstream",)},
    "POST /realize/batch":              {"kind": "raw_batch", "modes": ("nonstream",)},
    "POST /v1/batch/completions":       {"kind": "raw_batch", "modes": ("nonstream",)},
    "POST /api/chat":                   {"kind": "ollama_chat", "modes": ("nonstream", "stream")},
    "POST /api/generate":               {"kind": "ollama_generate", "modes": ("nonstream", "stream")},
}
NOT_GENERATION = {
    "GET /": "the route index itself",
    "GET /health": "liveness", "GET /health/live": "liveness", "GET /health/ready": "readiness",
    "GET /ready": "readiness", "GET /models": "model listing", "GET /v1/models": "model listing",
    "POST /tokenize": "token ids, no generation", "POST /batch/tokenize": "token ids, no generation",
    "POST /realize/embed": "embedding vector", "POST /v1/embeddings": "embedding vector",
    "POST /api/embeddings": "embedding vector",
    "GET /realize/model": "model metadata", "POST /realize/reload": "model reload",
    "GET /metrics": "telemetry", "GET /metrics/dispatch": "telemetry",
    "POST /metrics/dispatch/reset": "telemetry reset", "GET /v1/metrics": "telemetry",
    "GET /v1/effective-config": "server configuration",
    "POST /v1/predict": "classical-model prediction (503 for an LLM)",
    "POST /v1/explain": "classical-model explanation (503 for an LLM)",
    "GET /v1/audit/:request_id": "audit record lookup",
    "POST /v1/gpu/warmup": "GPU warmup", "GET /v1/gpu/status": "GPU status",
    "GET /api/tags": "ollama model listing", "POST /api/show": "ollama model metadata",
    "GET /api/version": "ollama version",
    "POST /v1/logprobs": "per-token logprobs of GIVEN text, no generation",
    "POST /v1/perplexity": "perplexity of GIVEN text, no generation",
}
RAW_KINDS = ("text_prompt", "raw_generate", "raw_sse", "raw_batch")


def classify(index):
    """Split a route index into generation / not_generation / unclassified."""
    gen, other, unclassified = [], [], []
    for r in index:
        if r in GENERATION:
            gen.append({"route": r, "kind": GENERATION[r]["kind"], "modes": list(GENERATION[r]["modes"])})
        elif r in NOT_GENERATION:
            other.append({"route": r, "why": NOT_GENERATION[r]})
        else:
            unclassified.append(r)
    return {"generation": gen, "not_generation": other, "unclassified": unclassified}


def index_from_doc(doc):
    routes = doc.get("routes") if isinstance(doc, dict) else None
    if not isinstance(routes, list) or not routes or not all(isinstance(r, str) for r in routes):
        return None, "GET / carries no `routes` list: the router serves no route index"
    return routes, None


def fetch_index(url, timeout=30):
    """The server's own route index from GET /, or (None, why)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return None, "GET / answered HTTP %s: the router serves no route index" % exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, "GET / failed: %s" % exc
    try:
        doc = json.loads(body)
    except ValueError:
        return None, "GET / is not JSON (%r): the router serves no route index" % body[:80]
    return index_from_doc(doc)


class Fault(Exception):
    """The wire broke. `meta` and `text` carry whatever WAS read, so the row shows the evidence."""

    def __init__(self, why, meta=None, text=None):
        super().__init__(why)
        self.meta, self.text = meta or {}, text


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise Fault("http_%s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200]),
                    {"http": exc.code})
    except (urllib.error.URLError, OSError) as exc:
        raise Fault("unreachable: %s" % exc)


def _json(raw, what):
    try:
        return json.loads(raw)
    except ValueError:
        raise Fault("unparseable: %s is not JSON: %r" % (what, raw[:120]))


def _stream_verdict(parts, meta, terminal_what):
    """Shared stream rules: the terminal event must have arrived, and some content must have too."""
    text = "".join(parts)
    if meta["terminal"] is None:
        raise Fault("stream_truncated: %s after %d chunk(s)" % (terminal_what, meta["chunks"]), meta, text)
    if not text:
        raise Fault("stream_zero_deltas: %d chunk(s), none with content" % meta["chunks"], meta, text)
    return text, meta


def read_openai_sse(lines, field):
    """OpenAI SSE: `data: {chunk}` lines, a chunk that carries finish_reason, then `data: [DONE]`."""
    parts, meta = [], {"chunks": 0, "terminal": None, "finish_reason": None}
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            meta["terminal"] = "[DONE]"
            break
        chunk = _json(data, "an SSE chunk")
        meta["chunks"] += 1
        for ch in chunk.get("choices") or []:
            parts.append(((ch.get("delta") or {}).get("content") if field == "delta" else ch.get("text")) or "")
            if ch.get("finish_reason"):
                meta["finish_reason"] = ch["finish_reason"]
    text, meta = _stream_verdict(parts, meta, "no `data: [DONE]`")
    if meta["finish_reason"] is None:
        raise Fault("stream_no_finish: no chunk carried finish_reason", meta, text)
    return text, meta


def read_named_sse(lines):
    """realizar SSE: `event: token` + `data: {"text": ...}`, then the terminal `event: done`."""
    parts, meta, event = [], {"chunks": 0, "terminal": None, "finish_reason": None}, None
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = _json(line[5:].strip(), "an SSE data line")
            if event == "token":
                meta["chunks"] += 1
                parts.append(data.get("text") or "")
            elif event == "done":
                meta["terminal"] = "event: done"
                break
    return _stream_verdict(parts, meta, "no `event: done`")


def read_ndjson(lines, field):
    """ollama NDJSON: one object per line; the terminal object has `"done": true`."""
    parts, meta = [], {"chunks": 0, "terminal": None, "finish_reason": None}
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        obj = _json(line, "an NDJSON line")
        meta["chunks"] += 1
        parts.append(((obj.get("message") or {}).get("content") if field == "message" else obj.get("response")) or "")
        if obj.get("done") is True:
            meta["terminal"], meta["finish_reason"] = "done:true", obj.get("done_reason")
            break
    return _stream_verdict(parts, meta, 'no object with `"done": true`')


def request_body(kind, stream, messages, rendered, a):
    body = _base_body(kind, stream, messages, rendered, a)
    body.update(json.loads(getattr(a, "extra", "") or "{}"))
    return body


def _base_body(kind, stream, messages, rendered, a):
    samp = {"max_tokens": a.max_tokens, "temperature": a.temperature, "seed": a.seed}
    opts = {"temperature": a.temperature, "seed": a.seed, "num_predict": a.max_tokens}
    model = getattr(a, "model", "default") or "default"
    return {
        "chat_messages": lambda: {"model": model, "messages": messages, "stream": stream, **samp},
        "text_prompt": lambda: {"model": model, "prompt": rendered, "stream": stream, **samp},
        "raw_generate": lambda: {"prompt": rendered, "strategy": "greedy", **samp},
        "raw_sse": lambda: {"prompt": rendered, "strategy": "greedy", **samp},
        "raw_batch": lambda: {"prompts": [rendered], "strategy": "greedy", **samp},
        "ollama_chat": lambda: {"model": model, "messages": messages, "stream": stream, "options": opts},
        "ollama_generate": lambda: {"model": model, "prompt": messages[-1]["content"], "stream": stream,
                                    "options": opts},
    }[kind]()


def read_nonstream(kind, doc):
    """(text, finish_reason) from a route's one JSON reply."""
    if kind in ("chat_messages", "text_prompt"):
        ch = (doc.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") if kind == "chat_messages" else ch.get("text")
        return text, ch.get("finish_reason")
    if kind == "raw_generate":
        return doc.get("text"), None
    if kind == "raw_batch":
        return ((doc.get("results") or [{}])[0]).get("text"), None
    if kind == "ollama_chat":
        return (doc.get("message") or {}).get("content"), doc.get("done_reason")
    return doc.get("response"), doc.get("done_reason")


def one_request(base, route, kind, mode, messages, rendered, a):
    """One turn through one route. Returns (text, meta); raises Fault when the wire breaks."""
    stream = mode == "stream"
    resp = _post(base.rstrip("/") + route.split(" ", 1)[1], request_body(kind, stream, messages, rendered, a),
                 a.timeout)
    meta = {"http": resp.status}
    with resp:
        if stream:
            try:
                if kind == "chat_messages":
                    text, m = read_openai_sse(resp, "delta")
                elif kind == "text_prompt":
                    text, m = read_openai_sse(resp, "text")
                elif kind == "raw_sse":
                    text, m = read_named_sse(resp)
                else:
                    text, m = read_ndjson(resp, "message" if kind == "ollama_chat" else "response")
            except Fault as f:
                f.meta = {**meta, **f.meta}
                raise
            except (OSError, ValueError) as exc:
                raise Fault("stream_truncated: the connection broke mid-stream: %s" % exc, meta)
            return text, {**meta, **m}
        doc = _json(resp.read().decode("utf-8", "replace"), "the response body")
    text, finish = read_nonstream(kind, doc)
    meta.update({"chunks": None, "terminal": None, "finish_reason": finish})
    if not text:
        raise Fault("empty_text: HTTP 200 and no text in the route's reply field", meta, text)
    return text, meta


def render(render_url, messages, thinking, timeout):
    """llama.cpp's own rendering of the conversation, generation prompt included."""
    resp = _post(render_url.rstrip("/") + "/apply-template",
                 {"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking == "on"}}, timeout)
    with resp:
        doc = _json(resp.read().decode("utf-8", "replace"), "the /apply-template body")
    if not isinstance(doc.get("prompt"), str) or not doc["prompt"]:
        raise Fault("the renderer returned no prompt")
    return doc["prompt"]


def drive(a):
    prompt = json.load(open(a.prompt_file))
    spec = GENERATION.get(a.route)
    rep = {"route": a.route, "mode": a.mode, "device": a.device or None,
           "thinking_requested": a.thinking,
           "thinking_control": "none: apr serve reads no per-request thinking toggle on any router",
           "http": None, "chunks": None, "terminal": None, "finish_reason": None, "rendered_prompts": []}
    out = {"text": None, "protocol_fault": None, "refused": None, "reported": rep}

    def finish(rc):
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        return rc

    if spec is None:
        out["refused"] = "route %s is not classified as a generation route" % a.route
        return finish(4)
    if a.mode not in spec["modes"]:
        out["refused"] = "route %s does not support mode %s (its wire: %s)" % (a.route, a.mode, "/".join(spec["modes"]))
        return finish(4)
    raw = spec["kind"] in RAW_KINDS
    if raw and not a.render_url:
        out["refused"] = ("route %s takes a raw prompt and no reference renderer was given: "
                          "unrendered text would ask a different question" % a.route)
        return finish(4)
    users = [m for m in prompt["messages"] if m.get("role") == "user"]
    history = [m for m in prompt["messages"] if m.get("role") == "system"]
    turns = []
    try:
        for u in users:
            history.append(u)
            rendered = None
            if raw:
                try:
                    rendered = render(a.render_url, history, a.thinking, a.timeout)
                except Fault as exc:
                    out["refused"] = "the reference renderer failed: %s" % exc
                    return finish(4)
                rep["rendered_prompts"].append(hashlib.sha256(rendered.encode()).hexdigest())
            text, meta = one_request(a.url, a.route, spec["kind"], a.mode, history, rendered, a)
            rep.update(meta)
            turns.append(text)
            history.append({"role": "assistant", "content": text})
    except Fault as f:
        rep.update(f.meta)
        out["protocol_fault"] = str(f)
        if f.text:
            out["partial_text"] = f.text
        if turns:
            out["turns"] = turns
        return finish(3)
    out["text"] = turns[-1]
    if len(users) > 1:
        out["turns"] = turns
    return finish(0)


def plan(a):
    if a.url:
        index, why = fetch_index(a.url)
    else:
        try:
            index, why = index_from_doc(json.load(open(a.index_file)))
        except (OSError, ValueError) as exc:
            index, why = None, "index file unreadable: %s" % exc
    if index is None:
        print(json.dumps({"index": None, "no_route_index": why}))
        return 1
    p = classify(index)
    p["index"] = index
    print(json.dumps(p))
    return 1 if p["unclassified"] else 0


MODE_VERB = {"nonstream": "serve run", "stream": "serve stream"}


def slug(route):
    return "".join(c if c.isalnum() else "_" for c in route).strip("_")


def sweep(a):
    """Every (route, mode, prompt) of one server, written as one JSON per cell plus plan.json.

    The route set is the server's own index unless --routes pins it (a comparator: llama-server
    and ollama are asked only their OpenAI chat route, which is what they share with apr).
    A cell whose driver dies still leaves no silent gap: `rows` turns a missing file into a row.
    """
    if a.routes:
        p = classify(a.routes.split(","))
        p["index"], p["pinned"] = None, True
    else:
        index, why = fetch_index(a.url)
        if index is None:
            p = {"index": None, "no_route_index": why, "generation": [], "not_generation": [], "unclassified": []}
        else:
            p = classify(index)
            p["index"] = index
    cells = []
    for pid, verbs in (json.loads(x) for x in open(a.prompt_list) if x.strip()):
        for g in p["generation"]:
            for mode in g["modes"]:
                if MODE_VERB[mode] not in verbs:
                    continue
                out = "%s/%s-%s-%s.json" % (a.out_dir, slug(g["route"]), mode, pid)
                ns = argparse.Namespace(url=a.url, route=g["route"], mode=mode,
                                        prompt_file="%s/prompt-%s.json" % (a.prompt_dir, pid),
                                        max_tokens=int(open("%s/maxtok-%s.txt" % (a.prompt_dir, pid)).read()),
                                        temperature=a.temperature, seed=a.seed, thinking=a.thinking,
                                        render_url=a.render_url, device=a.device, timeout=a.timeout, out=out,
                                        extra=a.extra, model=a.model)
                try:
                    rc = drive(ns)
                except Exception as exc:  # a driver bug must still leave a row, never a gap
                    rc = 3
                    json.dump({"text": None, "refused": None, "reported": {"route": g["route"], "mode": mode},
                               "protocol_fault": "driver_crash: %s: %s" % (type(exc).__name__, exc)}, open(out, "w"))
                cells.append({"route": g["route"], "mode": mode, "prompt_id": pid, "out": out, "rc": rc})
    p["cells"] = cells
    p["prompts"] = [json.loads(x) for x in open(a.prompt_list) if x.strip()]
    json.dump(p, open("%s/plan.json" % a.out_dir, "w"), indent=1)
    return 0


def rows(a):
    """Manifest rows (row contract v1 + `route`, `mode`) from one sweep's plan.json.

    RED BY CONSTRUCTION, never by omission. Every serve prompt × both serve verbs gets a row
    naming the fault when the surface itself could not be covered:
      no plan at all        the sweep never ran (the cell was not had, or it died): a refused row
      no_route_index        GET / gave no index: rc 3, protocol_fault, text null
      unclassified_route    an indexed route no table knows: rc 3, protocol_fault, text null
    so a judge that ignores `protocol_fault` still sees no text, and cannot score it right.
    """
    prompts = [json.loads(x) for x in open(a.prompt_list) if x.strip()]
    base = {"kind": "gen", "engine": a.engine, "model_sha256": a.sha, "host": a.host,
            "thinking": a.thinking, "backend": a.backend}
    emitted = []

    def emit(pid, verb, mode, route, rc, stdout, refused):
        emitted.append({**base, "prompt_id": pid, "verb": verb, "mode": mode, "route": route,
                        "rc": rc, "stdout": stdout, "stderr": None, "refused": refused})

    def fault_file(route, mode, pid, why):
        f = "%s/%s-%s-%s.fault.json" % (a.out_dir, slug(route), mode, pid)
        json.dump({"text": None, "refused": None, "protocol_fault": why,
                   "reported": {"route": route, "mode": mode}}, open(f, "w"))
        return f

    try:
        p = json.load(open("%s/plan.json" % a.out_dir))
    except (OSError, ValueError):
        why = a.cell_why or "the serve sweep wrote no plan.json (it did not run, or died before the end)"
        for pid, verbs in prompts:
            for mode, verb in MODE_VERB.items():
                if verb in verbs:
                    emit(pid, verb, mode, None, None, None, why)
        p = None
    if p is not None:
        for pid, verbs in prompts:
            for mode, verb in MODE_VERB.items():
                if verb not in verbs:
                    continue
                if p.get("no_route_index"):
                    emit(pid, verb, mode, "GET /", 3,
                         fault_file("GET /", mode, pid, "no_route_index: " + p["no_route_index"]), None)
                for r in p.get("unclassified") or []:
                    emit(pid, verb, mode, r, 3, fault_file(r, mode, pid,
                         "unclassified_route: %s is mounted (GET / lists it) and no CRUX table knows its wire; "
                         "classify it in scripts/lib/crux_serve_routes.py" % r), None)
        for c in p.get("cells") or []:
            refused = None
            if c["rc"] == 4:
                try:
                    refused = json.load(open(c["out"])).get("refused") or "refused"
                except (OSError, ValueError):
                    refused = "refused"
            emit(c["prompt_id"], MODE_VERB[c["mode"]], c["mode"], c["route"],
                 None if refused else c["rc"], None if refused else c["out"], refused)
    with open(a.manifest, "a") as fh:
        for r in emitted:
            fh.write(json.dumps(r) + "\n")
        if p is not None and a.engine == "apr":
            fh.write(json.dumps({"kind": "serve_routes", "engine": a.engine, "model_sha256": a.sha, "host": a.host,
                                 "backend": a.backend, "index": p.get("index"),
                                 "no_route_index": p.get("no_route_index"),
                                 "generation": p.get("generation"), "not_generation": p.get("not_generation"),
                                 "unclassified": p.get("unclassified")}) + "\n")
    return 0


def add_sampling(sp):
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--thinking", choices=["off", "on"], default="off")
    sp.add_argument("--render-url", default="")
    sp.add_argument("--device", default="")
    sp.add_argument("--timeout", type=float, default=600)
    sp.add_argument("--model", default="default", help="the request's `model` field (ollama needs its import name)")
    sp.add_argument("--extra", default="{}", help="JSON merged into every request body (keep_alive, chat_template_kwargs)")


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url")
    g.add_argument("--index-file")
    d = sub.add_parser("drive")
    d.add_argument("--url", required=True)
    d.add_argument("--route", required=True)
    d.add_argument("--mode", required=True, choices=["nonstream", "stream"])
    d.add_argument("--prompt-file", required=True, help="one v2 prompt entry (needs `messages`)")
    d.add_argument("--max-tokens", type=int, required=True)
    d.add_argument("--out", required=True)
    add_sampling(d)
    s = sub.add_parser("sweep")
    s.add_argument("--url", required=True)
    s.add_argument("--routes", default="", help="comma list pinning the routes (comparators); default: GET / index")
    s.add_argument("--prompt-list", required=True, help='JSONL of [prompt_id, [verbs]]')
    s.add_argument("--prompt-dir", required=True, help="holds prompt-<id>.json and maxtok-<id>.txt")
    s.add_argument("--out-dir", required=True)
    add_sampling(s)
    r = sub.add_parser("rows")
    for f in ("--out-dir", "--prompt-list", "--manifest", "--engine", "--sha", "--host", "--backend"):
        r.add_argument(f, required=True)
    r.add_argument("--thinking", default="off")
    r.add_argument("--cell-why", default="")
    a = ap.parse_args(argv)
    return {"plan": plan, "drive": drive, "sweep": sweep, "rows": rows}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
