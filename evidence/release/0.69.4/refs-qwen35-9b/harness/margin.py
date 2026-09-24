"""#4261: llama.cpp top-5 probs at the first apr/llama divergence, fed the shared prefix as ids."""
import json, urllib.request
U = "http://127.0.0.1:18261"
def post(path, body):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
res = {}
for p, i in (("p850", 13), ("p4k", 2)):
    l = json.load(open(f"out/llama-9b-{p}.json"))
    ids = l["prompt_ids"] + l["tokens"][:i]
    c = post("/completion", {"prompt": ids, "n_predict": 1, "temperature": 0, "top_k": 1, "n_probs": 5,
                             "cache_prompt": False, "post_sampling_probs": False})
    top = c["completion_probabilities"][0]["top_logprobs"]
    res[p] = {"pos": i, "top": [(t["id"], t["token"], t["logprob"]) for t in top]}
    print(p, res[p], flush=True)
json.dump(res, open("out/margins.json", "w"), indent=1)
