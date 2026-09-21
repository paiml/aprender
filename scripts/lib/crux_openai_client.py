#!/usr/bin/env python3
"""crux_openai_client.py: the ONE OpenAI client every CRUX `serve run` cell uses (#3739, the serve verb).

The issue, verbatim: "serve: `apr serve` · `llama-server` · `ollama serve`, all through
the SAME OpenAI `/v1/chat/completions` client, streaming + non-streaming". This is that
client: one request, the same body shape for every server, and the answer written as
the row-contract JSON the judge reads for every other engine:
  {"text": <answer>, "reported": {"device", "stream", "finish_reason", "usage", "chunks"}}

It measures NOTHING about speed: it reads no clock and computes no rate. TTFT and
decode rate for the serve verb come from `apr test llm bench`, the canonical client
(PERF-009). `usage` is transcribed as the server reported it, and never judged.

Exit: 0 an answer was read · 3 the server refused, errored or sent no content (the
JSON says why) · 2 usage.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--messages", required=True, help='JSON file: {"messages": [...]}')
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--device", default="")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--extra", default="{}", help="JSON merged into the request body (e.g. keep_alive)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    body = {"model": a.model, "messages": json.load(open(a.messages))["messages"],
            "max_tokens": a.max_tokens, "temperature": a.temperature, "seed": a.seed,
            "stream": bool(a.stream)}
    body.update(json.loads(a.extra))
    out = {"text": None, "reported": {"device": a.device or None, "stream": bool(a.stream),
                                       "finish_reason": None, "usage": None, "chunks": 0}}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    rc = 0
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            if a.stream:
                parts = []
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    out["reported"]["chunks"] += 1
                    if chunk.get("usage"):
                        out["reported"]["usage"] = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        parts.append((ch.get("delta") or {}).get("content") or "")
                        if ch.get("finish_reason"):
                            out["reported"]["finish_reason"] = ch["finish_reason"]
                out["text"] = "".join(parts)
            else:
                doc = json.loads(resp.read().decode("utf-8", "replace"))
                ch = (doc.get("choices") or [{}])[0]
                out["text"] = (ch.get("message") or {}).get("content")
                out["reported"]["finish_reason"] = ch.get("finish_reason")
                out["reported"]["usage"] = doc.get("usage")
    except urllib.error.HTTPError as exc:
        out["error"] = "HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:300])
        rc = 3
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        rc = 3
    if rc == 0 and not out["text"]:
        out["error"] = "the server returned no content"
        rc = 3
    if rc != 0:
        out["text"] = None
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
