"""llama.cpp temp-0 oracle for #4261: official GGUF template -> ids -> /completion, keyed per prompt."""
import json, sys, urllib.request
U = "http://127.0.0.1:18261"
def post(path, body):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=2400))
for p in sys.argv[1:]:
    text = open(f"{p}.txt").read()
    rendered = post("/apply-template", {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    ids = post("/tokenize", {"content": rendered, "add_special": False, "parse_special": True})["tokens"]
    c = post("/completion", {"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1,
                             "cache_prompt": False, "return_tokens": True, "seed": 0})
    json.dump({"prompt": p, "rendered_tail": rendered[-120:], "prompt_ids": ids, "prompt_tokens": len(ids),
               "tokens": c["tokens"], "text": c["content"], "timings": c.get("timings")},
              open(f"out/llama-9b-{p}.json", "w"))
    print(p, len(ids), len(c["tokens"]), flush=True)
