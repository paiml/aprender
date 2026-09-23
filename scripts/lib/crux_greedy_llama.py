#!/usr/bin/env python3
"""crux_greedy_llama.py: greedy token rows for #3957 F9 (aprender-36), from a running llama-server.

  gen  --url U --messages F --thinking on|off --max-tokens N --seed S --out O
       llama.cpp itself: render the chat template with enable_thinking EXPLICIT (/apply-template), generate
       greedily (temperature 0, top_k 1) with the generated ids returned (/completion return_tokens), and
       decode those ids WITH special tokens (/detokenize), so <think>/</think> are visible in the text.
  apr  --url U --apr-json F --max-tokens N --out O
       apr's own greedy ids (`apr run --format json` "tokens"), decoded with special tokens through the SAME
       server's /detokenize: one tokenizer for both engines' text, and the row says whose ids and whose decode.

Output (the `raw` object aprender-36's judge reads): {"generated_ids", "generated_text", "greedy": true,
"special": true, "max_tokens"} plus provenance. A failure is written as {"error": ...} with exit 3 — the caller
turns it into a refused row, never an absence. Stdlib only.
"""
import argparse
import json
import sys
import urllib.request


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def detok(url, ids):
    return post(url, "/detokenize", {"tokens": ids})["content"]


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", required=True)
    g.add_argument("--messages", required=True)
    g.add_argument("--thinking", required=True, choices=["on", "off"])
    g.add_argument("--max-tokens", type=int, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True)
    r = sub.add_parser("apr")
    r.add_argument("--url", required=True)
    r.add_argument("--apr-json", required=True)
    r.add_argument("--max-tokens", type=int, required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "gen":
            msgs = json.load(open(a.messages))["messages"]
            rendered = post(a.url, "/apply-template", {"messages": msgs,
                                                       "chat_template_kwargs": {"enable_thinking": a.thinking == "on"}})["prompt"]
            prompt_ids = post(a.url, "/tokenize", {"content": rendered, "add_special": True,
                                                   "parse_special": True})["tokens"]
            resp = post(a.url, "/completion", {"prompt": prompt_ids, "n_predict": a.max_tokens, "temperature": 0.0,
                                               "top_k": 1, "seed": a.seed, "return_tokens": True,
                                               "cache_prompt": False})
            ids = resp.get("tokens")
            if not isinstance(ids, list):
                raise RuntimeError("llama-server /completion returned no `tokens` list (return_tokens unsupported?)")
            doc = {"generated_ids": ids, "generated_text": detok(a.url, ids), "greedy": True, "special": True,
                   "max_tokens": a.max_tokens, "rendered_prompt": rendered, "prompt_ids": prompt_ids,
                   "thinking": a.thinking, "stop_type": resp.get("stop_type"),
                   "decoded_by": "llama.cpp /detokenize (special tokens kept)"}
        else:
            apr = json.load(open(a.apr_json))
            ids = apr.get("tokens")
            if not isinstance(ids, list):
                raise RuntimeError("apr --format json carried no `tokens` list")
            doc = {"generated_ids": ids, "generated_text": detok(a.url, ids), "greedy": True, "special": True,
                   "max_tokens": a.max_tokens, "apr_text": apr.get("text"), "finish_reason": apr.get("finish_reason"),
                   "backend": apr.get("backend"),
                   "decoded_by": "llama.cpp /detokenize of apr's own ids (one tokenizer for both engines' text)"}
    except Exception as e:  # noqa: BLE001 — every failure is a named refusal
        json.dump({"error": f"{type(e).__name__}: {e}"}, open(a.out, "w"))
        return 3
    json.dump(doc, open(a.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
