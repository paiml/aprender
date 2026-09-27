#!/usr/bin/env python3
"""crux_fake_serve.py: a FIXTURE server for scripts/check_crux_serve_code.sh (#3962). Never a comparator.

It speaks the wire of each apr serve generation route and can break any of them
on demand, so every protocol fault crux_serve_routes.py claims to catch can be
planted and seen to go RED. The answer it gives is always `<answer>4</answer>`.

  --index normal   GET / lists the routes below
  --index extra    ... plus `POST /v1/brand-new`, a route no CRUX table knows
  --index none     GET / is plain text (the APR-CPU fallback router's shape)
  --fault NAME     break every stream/body the named way:
                   ok | no_done | zero_deltas | no_finish | empty_text | http_500 | bad_json
Also serves POST /apply-template (the reference renderer), echoing a rendered prompt.
Usage: crux_fake_serve.py --port P [--index normal] [--fault ok]
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUTES = ["GET /", "GET /health", "POST /v1/chat/completions", "POST /v1/chat/completions/stream",
          "POST /v1/completions", "POST /generate", "POST /stream/generate", "POST /api/chat",
          "POST /api/generate", "POST /tokenize"]
ANSWER = "<answer>4</answer>"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, "{}")
        if self.path == "/":
            if A.index == "none":
                return self.send(200, "APR v2 Inference Server - POST /v1/completions", "text/plain")
            routes = ROUTES + (["POST /v1/brand-new"] if A.index == "extra" else [])
            return self.send(200, json.dumps({"service": "apr serve", "routes": routes}))
        self.send(404, "{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/apply-template":
            return self.send(200, json.dumps({"prompt": "<|user|>" + body["messages"][-1]["content"] + "<|assistant|>"}))
        if A.fault == "http_500":
            return self.send(500, '{"error":"planted"}')
        text = "" if A.fault in ("empty_text", "zero_deltas") else ANSWER
        p = self.path
        stream = bool(body.get("stream")) or p in ("/v1/chat/completions/stream", "/stream/generate")
        if not stream:
            if A.fault == "bad_json":
                return self.send(200, "not json at all")
            doc = {"/v1/chat/completions": {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
                   "/v1/completions": {"choices": [{"text": text, "finish_reason": "stop"}]},
                   "/generate": {"text": text},
                   "/api/chat": {"message": {"content": text}, "done": True},
                   "/api/generate": {"response": text, "done": True}}.get(p)
            return self.send(200, json.dumps(doc)) if doc is not None else self.send(404, "{}")
        pieces = [text[:8], text[8:]] if text else ["", ""]
        out = []
        if p in ("/v1/chat/completions", "/v1/chat/completions/stream", "/v1/completions"):
            key = (lambda t: {"delta": {"content": t}}) if "chat" in p else (lambda t: {"text": t})
            for t in pieces:
                out.append("data: " + json.dumps({"choices": [{**key(t), "finish_reason": None}]}))
            if A.fault != "no_finish":
                out.append("data: " + json.dumps({"choices": [{**key(""), "finish_reason": "stop"}]}))
            if A.fault == "bad_json":
                out.append("data: {truncated")
            if A.fault != "no_done":
                out.append("data: [DONE]")
            return self.send(200, "\n\n".join(out) + "\n\n", "text/event-stream")
        if p == "/stream/generate":
            for t in pieces:
                out.append("event: token\ndata: " + json.dumps({"token_id": 1, "text": t}))
            if A.fault != "no_done":
                out.append("event: done\ndata: " + json.dumps({"num_generated": 2}))
            return self.send(200, "\n\n".join(out) + "\n\n", "text/event-stream")
        if p in ("/api/chat", "/api/generate"):
            f = (lambda t: {"message": {"content": t}}) if p == "/api/chat" else (lambda t: {"response": t})
            for t in pieces:
                out.append(json.dumps({**f(t), "done": False}))
            if A.fault != "no_done":
                out.append(json.dumps({**f(""), "done": True, "done_reason": "stop"}))
            return self.send(200, "\n".join(out) + "\n", "application/x-ndjson")
        self.send(404, "{}")


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--index", default="normal", choices=["normal", "extra", "none"])
ap.add_argument("--fault", default="ok")
A = ap.parse_args(sys.argv[1:])
ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
