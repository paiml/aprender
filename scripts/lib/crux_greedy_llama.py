#!/usr/bin/env python3
"""crux_greedy_llama.py: greedy token rows for #3957 F9 (aprender-36), from a running llama-server.

  gen  --url U --messages F --thinking on|off --max-tokens N --seed S --out O [--apr-stderr E]
       llama.cpp itself: generate greedily (temperature 0, top_k 1) with the generated ids returned (/completion
       return_tokens), and decode those ids WITH special tokens (/detokenize), so <think>/</think> are visible.
       THE PROMPT IS APR'S OWN TOKEN IDS when apr ran (`apr run -v` prints `encoded N tokens: [...]` on stderr),
       so both engines start from IDENTICAL tokens by construction (aprender-36: "llama.cpp must get that
       IDENTICAL rendered prompt"). llama.cpp's own rendering (/apply-template with enable_thinking explicit) is
       recorded beside it with `prompt_ids_equal`, so a template mismatch is visible, never silent. Without an
       apr prompt, llama.cpp's own rendering is the prompt, and `prompt_source` says so.
  apr  --url U --apr-json F --max-tokens N --out O [--apr-stderr E]
       apr's own greedy ids (`apr run --format json` "tokens"), decoded with special tokens through the SAME
       server's /detokenize: one tokenizer for both engines' text, and the row says whose ids and whose decode.

Output (the `raw` object aprender-36's judge reads): {"generated_ids", "generated_text", "greedy": true,
"special": true, "max_tokens"} plus provenance. A failure is written as {"error": ...} with exit 3 — the caller
turns it into a refused row, never an absence. Stdlib only.
"""
import argparse
import json
import re
import sys
import urllib.request


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def detok(url, ids):
    return post(url, "/detokenize", {"tokens": ids})["content"]


def apr_prompt(stderr_path):
    """(prompt ids, formatted_prompt as apr printed it — Rust-escaped) from `apr run -v` stderr, or (None, None)."""
    if not stderr_path:
        return None, None
    try:
        err = open(stderr_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None, None
    ids = re.search(r"encoded (\d+) tokens: \[([0-9, ]*)\]", err)
    fp = re.search(r'formatted_prompt="((?:[^"\\]|\\.)*)"', err)
    return ([int(x) for x in ids.group(2).replace(" ", "").split(",") if x] if ids else None,
            fp.group(1) if fp else None)


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
    g.add_argument("--apr-stderr")
    r = sub.add_parser("apr")
    r.add_argument("--url", required=True)
    r.add_argument("--apr-json", required=True)
    r.add_argument("--max-tokens", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--apr-stderr")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "gen":
            msgs = json.load(open(a.messages))["messages"]
            rendered = post(a.url, "/apply-template", {"messages": msgs,
                                                       "chat_template_kwargs": {"enable_thinking": a.thinking == "on"}})["prompt"]
            own_ids = post(a.url, "/tokenize", {"content": rendered, "add_special": True,
                                                "parse_special": True})["tokens"]
            apr_ids, apr_rendered = apr_prompt(a.apr_stderr)
            prompt_ids = apr_ids if apr_ids else own_ids
            resp = post(a.url, "/completion", {"prompt": prompt_ids, "n_predict": a.max_tokens, "temperature": 0.0,
                                               "top_k": 1, "seed": a.seed, "return_tokens": True,
                                               "cache_prompt": False})
            ids = resp.get("tokens")
            if not isinstance(ids, list):
                raise RuntimeError("llama-server /completion returned no `tokens` list (return_tokens unsupported?)")
            doc = {"generated_ids": ids, "generated_text": detok(a.url, ids), "greedy": True, "special": True,
                   "max_tokens": a.max_tokens, "prompt_ids": prompt_ids,
                   "prompt_source": "apr's own prompt ids (identical tokens)" if apr_ids else
                                    "llama.cpp /apply-template (no apr prompt ids to reuse)",
                   "llama_template_rendered": rendered, "llama_template_prompt_ids": own_ids,
                   "apr_rendered_prompt": apr_rendered,
                   "prompt_ids_equal": (own_ids == apr_ids) if apr_ids else None,
                   "thinking": a.thinking, "stop_type": resp.get("stop_type"),
                   "decoded_by": "llama.cpp /detokenize (special tokens kept)"}
        else:
            apr = json.load(open(a.apr_json))
            apr_ids, apr_rendered = apr_prompt(a.apr_stderr)
            ids = apr.get("tokens")
            if not isinstance(ids, list):
                raise RuntimeError("apr --format json carried no `tokens` list")
            doc = {"generated_ids": ids, "generated_text": detok(a.url, ids), "greedy": True, "special": True,
                   "max_tokens": a.max_tokens, "apr_text": apr.get("text"), "finish_reason": apr.get("finish_reason"),
                   "prompt_ids": apr_ids, "rendered_prompt": apr_rendered,
                   "backend": apr.get("backend"),
                   "decoded_by": "llama.cpp /detokenize of apr's own ids (one tokenizer for both engines' text)"}
    except Exception as e:  # noqa: BLE001 — every failure is a named refusal
        json.dump({"error": f"{type(e).__name__}: {e}"}, open(a.out, "w"))
        return 3
    json.dump(doc, open(a.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
