"""OpenAI-compatible Server-Sent-Events, read the way CRUX judges a stream (#3952, for #3962's `serve stream`).

Stdlib only; shared by the hf and vllm drivers. The text is the concatenated `delta.content`; reasoning deltas
(`reasoning_content` / `reasoning`) are kept apart and returned as a <think> block, as the non-streaming path
does. A stream that ends without its terminal event is TRUNCATED, and is an error, never an answer (#3957 Q6:
"require the stream's terminal event, so a truncated stream is RED").

THE TERMINAL EVENT IS THE SERVER'S OWN, and it is named by the caller, never guessed:
  "done"           `data: [DONE]` — the OpenAI protocol; vLLM 0.30.0 always sends it (measured, lambda 2026-09-23)
  "finish_reason"  a chunk whose choice carries a non-null finish_reason — transformers serve 5.17.0 NEVER sends
                   [DONE] (measured: 17 events, the last `finish_reason: "stop"` + usage), so for it that chunk is
                   the terminal event
A server that sends [DONE] is not excused by a finish_reason: accepting either for every server would let a
vLLM stream cut after its last content chunk pass.
"""

from __future__ import annotations

import json


class TruncatedStream(RuntimeError):
    pass


def parse_sse(text: str, terminal: str = "done") -> tuple[str, dict, int]:
    """-> (raw text with any reasoning as a leading <think> block, usage dict, number of content chunks)."""
    if terminal not in ("done", "finish_reason"):
        raise ValueError(f"unknown terminal event {terminal!r}")
    content, reasoning, usage, chunks, done, finished = [], [], {}, 0, False, None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            done = True
            break
        try:
            doc = json.loads(payload)
        except ValueError as e:
            raise RuntimeError(f"stream chunk is not JSON: {e}: {payload[:120]!r}") from e
        if isinstance(doc.get("usage"), dict):
            usage = doc["usage"]
        for ch in doc.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
                chunks += 1
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning.append(r)
            if ch.get("finish_reason"):
                finished = ch["finish_reason"]
    if terminal == "done" and not done:
        raise TruncatedStream(f"the stream ended without its terminal `data: [DONE]` event after {chunks} content "
                              "chunk(s): truncated, so it is no answer")
    if terminal == "finish_reason" and not finished:
        raise TruncatedStream(f"the stream ended without a chunk carrying finish_reason (this server's terminal "
                              f"event) after {chunks} content chunk(s): truncated, so it is no answer")
    raw = "".join(content)
    if reasoning:
        raw = f"<think>{''.join(reasoning)}</think>{raw}"
    return raw, usage, chunks
