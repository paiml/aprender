"""0.69.0's MEASURED evidence, carried into apr-model-ladder-receipt/v2 without inventing a measurement.

Inputs (all measured at the 0.69.0 release, apr 0.69.0 (225b2a9ab)):
  ladder   rel-069-state/ladder/0.69.0/<host>.json  (v1; sha256 per rung, measured on the host)
  sweep    q4k-sweep-<host>.jsonl                    (every Q4_K file the sweep reached; NO sha256)
  holds    inventory listing of the ladder contract's inventory dirs (file + bytes; NO sha256)
Rules:
  * inventory = what the host holds. sha256 ONLY where the 0.69.0 ladder measured that exact file on that host;
    otherwise absent (pv names it `unmeasuredModel`: the release never identified it, so never proved it).
  * arch only from the ladder rung; context_length, thinking, memory arithmetic: absent (0.69.0 measured none).
  * cells: the ladder row becomes the (run, think-off, 4k) row -- verdict pass iff the row was green, fallback
    from backends.cuda -- with prompt_tokens and answer_chars ABSENT, because 0.69.0 ran golden prompts and
    recorded neither. A sweep row for a file with no measured sha is carried UNKEYED (pv: `unkeyedRow`, verdict).
  * chat / serve / code / think-on / every other rung: no row, because 0.69.0 never ran them.
"""
import json, sys
AP, SW, SP, OUT = sys.argv[1:5]
REL = "225b2a9abbd171b3249794722f8da5fc8027e565"
for host in ["lambda", "gx10"]:
    lad = json.load(open(f"{AP}/ladder/0.69.0/{host}.json"))
    ladder_contract = {"qwen2-1.5b-q4km": ("qwen2.5-1.5b-instruct-q4_k_m.gguf", "qwen2"),
                       "qwen3-1.7b-q4km": ("Qwen3-1.7B-Q4_K_M.gguf", "qwen3"),
                       "qwen3-8b-q4km": ("Qwen3-8B-Q4_K_M.gguf", "qwen3"),
                       "qwen35-0.8b-q4km": ("Qwen3.5-0.8B-Q4_K_M.gguf", "qwen35"),
                       "qwen35-2b-q4km": ("Qwen3.5-2B-Q4_K_M.gguf", "qwen35"),
                       "qwen35-4b-q4km": ("Qwen3.5-4B-Q4_K_M.gguf", "qwen35"),
                       "qwen35-9b-q4km": ("Qwen3.5-9B-Q4_K_M.gguf", "qwen35"),
                       "qwen35-27b-q4km": ("Qwen3.5-27B-Q4_K_M.gguf", "qwen35")}
    by_file = {}
    for r in lad["rungs"]:
        f, arch = ladder_contract[r["id"]]
        by_file[f] = (r, arch)
    holds = json.load(open(f"{SP}/inv-{host}.json"))
    inventory, cells = [], []
    for h in holds:
        item = {"file": h["file"], "bytes": h["bytes"]}
        if h["file"] in by_file and by_file[h["file"]][0].get("sha256"):
            r, arch = by_file[h["file"]]
            item["sha256"] = r["sha256"]; item["arch"] = arch
            cuda = (r.get("backends") or {}).get("cuda") or {}
            cells.append({"sha256": r["sha256"], "file": h["file"], "verb": "run", "thinking": "off", "context": "4k",
                          "verdict": "pass" if r.get("green") else "fail", "backend": "cuda",
                          "fallback": bool(cuda.get("fallback")), "rc": cuda.get("rc"),
                          "reason": "0.69.0 ladder row (golden prompts; prompt_tokens and answer_chars not recorded)"
                                    + ("" if r.get("green") else ": " + str((r.get("golden_output") or {}).get("message", ""))[:120])})
        inventory.append(item)
    for l in open(f"{SW}/q4k-sweep-{host}.jsonl"):
        s = json.loads(l)
        if s["model"] in by_file:
            continue  # the ladder row, with its measured sha, already carries it
        cells.append({"file": s["model"], "verb": "run", "thinking": "off", "context": "4k",
                      "verdict": "pass" if s["green"] else "fail", "backend": "cuda",
                      "fallback": bool(s.get("gpu_fallback")), "rc": s.get("gpu_run_rc"),
                      "reason": "0.69.0 Q4_K sweep row (no sha256 recorded): "
                                + str((s.get("golden_output") or {}).get("message", ""))[:120]})
    out = {"schema": "apr-model-ladder-receipt/v2", "host": host, "version": "0.69.0", "sha": lad["sha"],
           "apr_sha": REL, "apr_version": lad.get("apr_version"), "cc": lad["cc"], "gpu": lad.get("gpu"),
           "translated_from": [f"rel-069-state/ladder/0.69.0/{host}.json", f"q4k-sweep-{host}.jsonl",
                               "inventory listing of ~/models, ~/.apr/models, ~/.cache/apr/models (2026-09-21T16:20:36Z)"],
           "inventory": inventory, "cells": cells, "rungs": []}
    json.dump(out, open(f"{OUT}/models/{host}.json", "w"), indent=1)
    print(host, len(inventory), "held,", sum(1 for i in inventory if "sha256" in i), "hashed,", len(cells), "rows")
