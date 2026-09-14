#!/usr/bin/env python3
"""PERF-060 calibration probe: one W1 request, fully instrumented.

Sends a single W1 corpus prompt to an OpenAI-compatible /v1/chat/completions
endpoint and reports TTFT, per-token ITL, decode rate and total latency.

Deliberately NOT the APR-PERF-GATE-001 4.4 harness: this is calibration, and
every replicate is printed rather than summarised. See calibration.md 6 for the
list of what this run was not.

    probe.py <url> <prompts.jsonl> <line-index> stream|nonstream
"""
import json
import sys
import time
import urllib.request

TIMEOUT_S = 900


def build_request(url, prompt, stream):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": stream,
    }
    return urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )


def chunk_content(payload):
    """The delta text of one SSE `data:` payload, or None if it carries none."""
    if payload == "[DONE]":
        return None, None
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    choices = d.get("choices") or [{}]
    return (choices[0].get("delta") or {}).get("content"), d.get("usage")


def read_stream(req):
    """Per-token arrival times, relative to just before the request was sent."""
    t0 = time.time()
    stamps, usage, head = [], None, ""
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            content, u = chunk_content(line[6:])
            if content:
                stamps.append(time.time() - t0)
                head += content
            if u:
                usage = u
    return time.time() - t0, stamps, usage, head[:200]


def read_once(req):
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        d = json.loads(resp.read().decode())
    head = (d["choices"][0]["message"]["content"] or "")[:200]
    return time.time() - t0, [], d.get("usage"), head


def timing(stamps):
    """TTFT, median inter-token gap and decode rate from arrival times."""
    if not stamps:
        return {}
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    span = stamps[-1] - stamps[0]
    return {
        "content_chunks": len(stamps),
        "ttft_s": round(stamps[0], 3),
        "itl_p50_ms": round((gaps[len(gaps) // 2] if gaps else 0.0) * 1000, 3),
        "decode_tok_s": round((len(stamps) - 1) / span, 3) if span > 0 else None,
    }


def main():
    url, corpus, idx, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    with open(corpus) as fh:
        obj = json.loads(fh.readlines()[idx])
    prompt = obj["prompt"]
    stream = mode == "stream"

    req = build_request(url, prompt, stream)
    total, stamps, usage, head = (read_stream if stream else read_once)(req)

    out = {
        "prompt_index": idx,
        "prompt_chars": len(prompt),
        "total_s": round(total, 3),
        "usage": usage,
        "text_head": head,
    }
    out.update(timing(stamps))
    if stamps:
        out["tail_s"] = round(total - stamps[-1], 3)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
