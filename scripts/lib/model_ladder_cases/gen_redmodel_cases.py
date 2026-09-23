#!/usr/bin/env python3
"""Generate the #3957 F9/F10 case fixtures (RED-MODEL, RED-UNSUPPORTED) from `green-inventory-beyond-ladder`.

Run from the repo root:  python3 scripts/lib/model_ladder_cases/gen_redmodel_cases.py
Every case is written from THIS file, so a fixture and the rule it proves cannot drift apart
silently: regenerate, then `git diff` shows exactly what moved. Each must-RED case varies ONE
fact from its green twin, so a mutant deleting one rule is killed by exactly the case for it.
"""

import copy
import json
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "green-inventory-beyond-ladder")
PREFIXES = ("green-red-model-", "green-red-unsupported-", "red-model-", "red-unsupported-")

D, D_SHA = "Qwen3.5-0.8B-IQ4_XS.gguf", "d" * 64          # the defective file (F9)
C, C_SHA = "Qwen3.5-4B-Q4_K_M.gguf", "c" * 64            # its positive-control sibling
M, M_SHA = "Qwen3.5-35B-A3B-UD-IQ4_XS.gguf", "3" * 64    # the unsupported architecture (F10)
IDS = [151667, 198, 32313, 11, 1077, 594, 1490]
PIDS = [248045, 846, 198, 3838, 374, 220, 17, 10, 17, 30, 248046, 198, 248045, 74455, 198]   # apr's rendering
OPIDS = PIDS + [248068, 198]                                                               # the official ON form opens <think>
# Verbatim, as `apr run --gpu` printed it on lambda (apr 0.69.1 (7b8aa7e32), rc 12).
REFUSAL = ("error: Not implemented: this build has no CUDA forward for architecture 'qwen35moe': Qwen3.5 MoE is a hybrid "
           "(Gated DeltaNet / SSM layers + mixture-of-experts), and the qwen3moe CUDA forward (#3714) does not run SSM "
           "layers. Re-run without --gpu to use the CPU path deliberately. This is a refusal, not a fallback: nothing "
           "was loaded and nothing was generated.")
VERBS = ("run", "chat", "serve run", "code")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh) if p.endswith(".json") else yaml.safe_load(fh)


def row_like(base_row, f, sha):
    x = copy.deepcopy(base_row)
    x.update({"id": f"inv:{f}", "file": f, "sha256": sha, "architecture": "qwen35"})
    return x


def crux_cell(sha, host, verb, thinking, verdict, control=False):
    return {"key": {"model_sha256": sha, "host": host, "verb": verb, "thinking": thinking, "rung": "golden",
                    "prompt_id": "think-control" if control else "fixture-control"},
            "verdict": verdict, **({"positive_control": True} if control else {})}


def raw(ids, text, prompt_ids=None, template_ids=None):
    r = {"generated_ids": list(ids), "generated_text": text, "greedy": True, "special": True, "max_tokens": 256,
         "prompt_ids": list(prompt_ids or PIDS)}
    if template_ids is not None:
        r["template_prompt_ids"] = list(template_ids)
    return r


def greedy(sha, host, apr_ids, oracle_ids, oracle_text, apr_text=None, extra=None):
    """A thinking-ON greedy entry: the PARITY row (apr and llama.cpp on apr's ids) and the llama.cpp@official
    row (the model's own template, #3990), which is where the defect must show."""
    g = {"key": {"model_sha256": sha, "host": host, "prompt_id": "think-2plus2", "thinking": "on"},
         "engines": ["apr", "llama.cpp", "llama.cpp@official"],
         "apr": {"raw": raw(apr_ids, apr_text if apr_text is not None else oracle_text)},
         "llama.cpp": {"raw": raw(oracle_ids, oracle_text), "version": "b10987"},
         "llama.cpp@official": {"raw": raw(oracle_ids, oracle_text, OPIDS, OPIDS), "version": "b10987"}}
    g.update(extra or {})
    return g


def build_f9(defect="think_never_closed", bf16=False):
    """lambda holds D (defective, golden_output RED) and C (its green control). -> (ladder, receipts, crux)."""
    L = load(os.path.join(BASE, "ladder.yaml"))
    rec = {h: load(os.path.join(BASE, "receipts", f"{h}.json")) for h in ("lambda", "gx10")}
    crux = {n[:-5]: load(os.path.join(BASE, "crux", n)) for n in sorted(os.listdir(os.path.join(BASE, "crux")))}
    R = rec["lambda"]
    tmpl = next(x for x in R["rungs"] if x.get("file") == "Extra-Q4_K_M.gguf")
    d = row_like(tmpl, D, D_SHA)
    msg = ("thinking-ON leg: the <think> block never closed within 2048 tokens" if defect == "think_never_closed"
           else "thinking-ON leg: <think> closed EMPTY within 2048 tokens")
    d.update({"green": False, "golden_output": {"passed": False, "skipped": False, "message": msg},
              "qa_rc": 1, "gates_failed": ["golden_output"], "gates_account_for_rc": True})
    R["rungs"].extend([d, row_like(tmpl, C, C_SHA)])
    R["inventory"].extend([{"file": D, "sha256": D_SHA, "bytes": 1}, {"file": C, "sha256": C_SHA, "bytes": 1}])
    R["red"] = sum(1 for x in R["rungs"] if not x.get("green"))
    L["ladder"]["inventory"]["red_model"] = {D: {"ticket": "#3951", "defect": defect, "control": C, "thinking": "on",
                                                 **({"bf16_reproduces": True} if bf16 else {})}}
    closed = "<think>\nThe sum of two and two is four.\n</think>\n\n<answer>4</answer>"
    defect_text = ("<think>\nOkay, let me think about this. Okay, let me think about this." if defect == "think_never_closed"
                   else "<think>\n.\n</think>\n\n<answer>4</answer>")
    for lane in ("cpu", "gpu"):
        X = crux[f"lambda-{lane}"]
        for v in VERBS:
            X["cells"] += [crux_cell(D_SHA, "lambda", v, "off", "GREEN"), crux_cell(D_SHA, "lambda", v, "on", "RED"),
                           crux_cell(C_SHA, "lambda", v, "off", "GREEN")]
        X["cells"].append(crux_cell(C_SHA, "lambda", "run", "on", "GREEN", control=True))
        extra = {"hf": {"raw": raw([1], defect_text), "version": "4.57.1"}} if bf16 else None
        X["greedy"] = [greedy(D_SHA, "lambda", IDS, IDS, defect_text, extra=extra),
                       greedy(C_SHA, "lambda", IDS[:3], IDS[:3], closed)]
    return L, rec, crux


def build_f10(arch="qwen35moe", key_arch="qwen35moe"):
    L = load(os.path.join(BASE, "ladder.yaml"))
    rec = {h: load(os.path.join(BASE, "receipts", f"{h}.json")) for h in ("lambda", "gx10")}
    crux = {n[:-5]: load(os.path.join(BASE, "crux", n)) for n in sorted(os.listdir(os.path.join(BASE, "crux")))}
    R = rec["lambda"]
    tmpl = next(x for x in R["rungs"] if x.get("file") == "Extra-Q4_K_M.gguf")
    m = row_like(tmpl, M, M_SHA)
    m["architecture"] = arch
    be = m["backends"]["cuda"]
    be.update({"ran": False, "rc": 1, "fallback": False})
    be["verbs"]["run"] = {"ran": False, "rc": 1, "refusal": REFUSAL.replace("qwen35moe'", f"{arch}'"),
                          "stdout_bytes": 178, "generated_bytes": 0}   # 178 = apr's `verbose:` preamble, measured
    be["verbs"]["chat"] = {"ran": False, "rc": 1}
    be["verbs"]["serve"] = {"probed": False, "why": "apr serve --gpu refused the architecture", "routes": {}, "teardown": "clean"}
    m.update({"green": False, "qa_rc": 1, "gates_failed": ["capability_match"], "gates_account_for_rc": True,
              "capability_match": {"passed": False, "skipped": False, "message": REFUSAL}})
    R["rungs"].append(m)
    R["inventory"].append({"file": M, "sha256": M_SHA, "bytes": 1})
    R["red"] = sum(1 for x in R["rungs"] if not x.get("green"))
    L["ladder"]["inventory"]["red_unsupported"] = {"Qwen3.5-35B-A3B*.gguf": {"architecture": key_arch, "ticket": "#3977"}}
    return L, rec, crux


W, W_SHA = "Qwen3.5-0.8B-UD-IQ2_XXS.gguf", "a" * 64      # the wrong-answer file (F9 wrong_answer)
K, K_SHA = "Qwen3.5-0.8B-Q4_K_M.gguf", "b" * 64            # its higher-quant control


def wa_raw(ids, text, prompt_ids=OPIDS, top2=None, template_ids=None):
    r = raw(ids, text, prompt_ids, template_ids)
    if top2 is not None:
        r["top2_logits"] = top2
    return r


def wa_greedy(sha, apr_ids, apr_text, ref_ids, ref_text, pid="golden-2plus2"):
    """A thinking-OFF greedy entry on the OFFICIAL template ids (#3990): apr, and llama.cpp@official on
    this lane (the CPU lane's is the reference, the GPU lane's is the oracle's CUDA leg)."""
    return {"key": {"model_sha256": sha, "host": "lambda", "prompt_id": pid, "thinking": "off"},
            "apr": {"raw": wa_raw(apr_ids, apr_text, OPIDS)},
            "llama.cpp@official": {"raw": wa_raw(ref_ids, ref_text, OPIDS, None, OPIDS)}}


def div(step):
    """IDS with the token at `step` changed (None: IDS unchanged)."""
    return list(IDS) if step is None else IDS[:step] + [99 + step] + IDS[step + 1:]


def build_wa(apr_div=None, ref_div=None):
    """lambda holds W (golden_output RED: 2+2 answered wrong) and K, its higher-quant control. apr
    first diverges from the llama.cpp CPU reference at `apr_div`, the oracle's CUDA leg at `ref_div`."""
    L = load(os.path.join(BASE, "ladder.yaml"))
    rec = {h: load(os.path.join(BASE, "receipts", f"{h}.json")) for h in ("lambda", "gx10")}
    crux = {n[:-5]: load(os.path.join(BASE, "crux", n)) for n in sorted(os.listdir(os.path.join(BASE, "crux")))}
    R = rec["lambda"]
    tmpl = next(x for x in R["rungs"] if x.get("file") == "Extra-Q4_K_M.gguf")
    w = row_like(tmpl, W, W_SHA)
    w.update({"green": False, "golden_output": {"passed": False, "skipped": False, "message": "expected '4' in 'What is 2+2?'"},
              "qa_rc": 1, "gates_failed": ["golden_output"], "gates_account_for_rc": True})
    R["rungs"].extend([w, row_like(tmpl, K, K_SHA)])
    R["inventory"].extend([{"file": W, "sha256": W_SHA, "bytes": 1}, {"file": K, "sha256": K_SHA, "bytes": 1}])
    R["red"] = sum(1 for x in R["rungs"] if not x.get("green"))
    L["ladder"]["inventory"]["red_model"] = {W: {"ticket": "#4004", "defect": "wrong_answer", "thinking": "off",
                                                 "expect": "4", "prompts": ["golden-2plus2"], "control": K}}
    wrong = "<think>\n\n</think>\n\nThe answer is five."
    right = "<think>\n\n</think>\n\nThe answer is 4."
    for lane in ("cpu", "gpu"):
        X = crux[f"lambda-{lane}"]
        for v in VERBS:
            X["cells"] += [crux_cell(W_SHA, "lambda", v, "off", "RED"), crux_cell(W_SHA, "lambda", v, "on", "GREEN"),
                           crux_cell(K_SHA, "lambda", v, "off", "GREEN")]
        ref = IDS if lane == "cpu" else div(ref_div)
        X["greedy"] = [wa_greedy(W_SHA, div(apr_div), wrong, ref, wrong),
                       wa_greedy(K_SHA, IDS[:3], right, IDS[:3], right)]
    return L, rec, crux


def write(name, L, rec, crux, rc, must, must_not=None):
    d = os.path.join(HERE, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "receipts"))
    os.makedirs(os.path.join(d, "crux"))
    with open(os.path.join(d, "ladder.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(L, fh, sort_keys=True)
    for h, R in rec.items():
        with open(os.path.join(d, "receipts", f"{h}.json"), "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1)
    for n, X in crux.items():
        # every apr greedy row records the backend it RAN on (#3957 F9); a case overrides it to plant a fallback
        lane = X.get("backend")
        for gg in X.get("greedy", []):
            if isinstance(gg.get("apr"), dict) and isinstance(gg["apr"].get("raw"), dict):
                gg["apr"]["raw"].setdefault("backend", {"requested": lane, "ran": lane, "fell_back": False})
        with open(os.path.join(d, "crux", f"{n}.json"), "w", encoding="utf-8") as fh:
            json.dump(X, fh, indent=1)
    shutil.copy(os.path.join(BASE, "version"), os.path.join(d, "version"))
    for fn, val in (("expected_rc", str(rc)), ("must_match", must), ("must_not_match", must_not)):
        if val is not None:
            with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
                fh.write(val + "\n")


def lam(rec):
    return rec["lambda"]


def row(rec, f):
    return next(x for x in lam(rec)["rungs"] if x.get("file") == f)


def main():
    for n in os.listdir(HERE):
        if n.startswith(PREFIXES) and os.path.isdir(os.path.join(HERE, n)):
            shutil.rmtree(os.path.join(HERE, n))
    # ---------------------------------------------------------------- F9 RED-MODEL
    L, rec, crux = build_f9()
    write("green-red-model-proven", L, rec, crux, 0, r"RED-MODEL lambda +Qwen3.5-0.8B-IQ4_XS.gguf +think_never_closed",
          r"FAIL|every required rung green")
    L, rec, crux = build_f9("think_empty")
    write("green-red-model-empty-proven", L, rec, crux, 0, r"RED-MODEL lambda +Qwen3.5-0.8B-IQ4_XS.gguf +think_empty", r"FAIL")
    L, rec, crux = build_f9(bf16=True)
    write("green-red-model-bf16-proven", L, rec, crux, 0, r"RED-MODEL lambda +Qwen3.5-0.8B-IQ4_XS.gguf", r"FAIL")

    L, rec, crux = build_f9()   # the operator's must-RED: llama.cpp CLOSES the block on this run -> apr's fault
    crux["lambda-gpu"]["greedy"][0]["llama.cpp@official"]["raw"]["generated_text"] = "<think>\nTwo plus two is four.\n</think>\n\n<answer>4</answer>"
    write("red-model-oracle-closes", L, rec, crux, 1, r"llama.cpp on the OFFICIAL template does NOT reproduce think_never_closed")

    L, rec, crux = build_f9()   # #3990: the defect shown ONLY on apr's rendering -- no official-template row
    for X in crux.values():
        for gg in X.get("greedy", []):
            gg.pop("llama.cpp@official", None)
    write("red-model-official-missing", L, rec, crux, 1, r"no llama.cpp@official row")

    L, rec, crux = build_f9()   # #3990: an "official" row that actually ran on apr's ids launders a template defect
    crux["lambda-gpu"]["greedy"][0]["llama.cpp@official"]["raw"]["prompt_ids"] = PIDS
    write("red-model-official-not-official", L, rec, crux, 1, r"did not run on the official template's ids")

    L, rec, crux = build_f9()   # the parity row: the engines ran on different prompt ids
    crux["lambda-gpu"]["greedy"][0]["llama.cpp"]["raw"]["prompt_ids"] = OPIDS
    write("red-model-parity-prompt-differs", L, rec, crux, 1, r"did not run on the same prompt ids")

    L, rec, crux = build_f9()   # must-RED: the key is present and no oracle ran
    for X in crux.values():
        X["greedy"] = [g for g in X.get("greedy", []) if g["key"]["model_sha256"] != D_SHA]
    write("red-model-no-oracle", L, rec, crux, 1, r"no oracle ran on this sweep")

    L, rec, crux = build_f9()   # must-RED: apr CPU != GPU
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = IDS[:-1] + [9]
    write("red-model-cpu-ne-gpu", L, rec, crux, 1, r"apr CPU and GPU DIFFER at step 6")

    L, rec, crux = build_f9()   # apr's ids differ from llama.cpp's on the identical GGUF
    crux["lambda-gpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = IDS[:2] + [7] + IDS[3:]
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = IDS[:2] + [7] + IDS[3:]
    write("red-model-apr-ne-oracle", L, rec, crux, 1, r"DIFFER from llama.cpp's on the identical GGUF at step 2")

    L, rec, crux = build_f9()   # the equal FLAG is not read: a planted equal=true with different ids stays RED
    g = crux["lambda-gpu"]["greedy"][0]
    g["apr"]["raw"]["generated_ids"] = IDS[::-1]
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = IDS[::-1]
    g["llama.cpp"].update({"first_divergence": None, "equal": True})
    write("red-model-equal-flag-not-trusted", L, rec, crux, 1, r"DIFFER from llama.cpp's on the identical GGUF at step 0")

    L, rec, crux = build_f9()   # the positive control does not close under llama.cpp: the instrument is blind
    crux["lambda-gpu"]["greedy"][1]["llama.cpp@official"]["raw"]["generated_text"] = "<think>\nThe sum of two and two"
    write("red-model-control-blind", L, rec, crux, 1, r"the instrument is blind")

    L, rec, crux = build_f9()   # the control is the defective file itself
    L["ladder"]["inventory"]["red_model"][D]["control"] = D
    write("red-model-control-is-self", L, rec, crux, 1, r"is the defective file itself")

    L, rec, crux = build_f9()   # the control's positive-control CRUX cell is RED
    for X in crux.values():
        for c in X["cells"]:
            if c.get("positive_control") and c["key"]["model_sha256"] == C_SHA:
                c["verdict"] = "RED"
    write("red-model-control-cell-red", L, rec, crux, 1, r"positive-control CRUX cells are not all GREEN")

    L, rec, crux = build_f9()   # a key matching no held file (#3880)
    L["ladder"]["inventory"]["red_model"]["Qwen9-0.1B-*.gguf"] = dict(L["ladder"]["inventory"]["red_model"][D])
    write("red-model-key-no-file", L, rec, crux, 1, r"red_model\['Qwen9-0.1B-\*.gguf'\] matches NO held file")

    L, rec, crux = build_f9()   # a key with no ticket
    del L["ladder"]["inventory"]["red_model"][D]["ticket"]
    write("red-model-no-ticket", L, rec, crux, 1, r"names no `#NNNN` ticket")

    L, rec, crux = build_f9()   # RED-MODEL cannot hide a second failure on the same row
    row(rec, D)["backends"]["cuda"]["verbs"]["serve"]["routes"]["/api/chat|stream=false"]["output_bad"] = "gibberish (fragment 'zombie')"
    write("red-model-residual", L, rec, crux, 1, r"excuses only the declared think_never_closed defect, and the row ALSO fails")

    L, rec, crux = build_f9()   # the row went GREEN: the key is stale
    d = row(rec, D)
    d.update({"green": True, "golden_output": {"passed": True, "skipped": False, "message": "ok"}, "qa_rc": 0, "gates_failed": []})
    lam(rec)["red"] = sum(1 for x in lam(rec)["rungs"] if not x.get("green"))
    for X in crux.values():
        for c in X["cells"]:
            c["verdict"] = "GREEN"
    write("red-model-stale", L, rec, crux, 1, r"is STALE: the row is GREEN")

    L, rec, crux = build_f9()   # the key covers thinking ON only: a RED thinking-OFF cell still blocks
    crux["lambda-gpu"]["cells"] = [dict(c, verdict="RED") if (c["key"]["model_sha256"] == D_SHA and c["key"]["thinking"] == "off"
                                                             and c["key"]["verb"] == "chat") else c for c in crux["lambda-gpu"]["cells"]]
    write("red-model-thinking-off-red", L, rec, crux, 1, r"FAIL  cell Qwen3.5-0.8B-IQ4_XS.gguf .*verb=chat -- CRUX: 1 RED")

    L, rec, crux = build_f9()   # ... and an unmeasured thinking-OFF axis blocks too
    for X in crux.values():
        X["cells"] = [c for c in X["cells"] if not (c["key"]["model_sha256"] == D_SHA and c["key"]["thinking"] == "off")]
    write("red-model-thinking-off-missing", L, rec, crux, 1, r"no verdict on the other axis proves the rest")

    L, rec, crux = build_f9()   # the key must say thinking: on
    L["ladder"]["inventory"]["red_model"][D]["thinking"] = "off"
    write("red-model-axis-not-on", L, rec, crux, 1, r"a think-block defect is a thinking-ON verdict")

    L, rec, crux = build_f9(bf16=True)   # the key says the MODEL is at fault; the bf16 leg closes, so it is not
    crux["lambda-gpu"]["greedy"][0]["hf"]["raw"]["generated_text"] = "<think>\nTwo and two make four.\n</think>\n4"
    write("red-model-bf16-not-reproduced", L, rec, crux, 1, r"the bf16 hf leg does NOT reproduce")

    L, rec, crux = build_f9("think_empty")   # think_empty: a closed block WITH content under llama.cpp is not empty
    crux["lambda-gpu"]["greedy"][0]["llama.cpp@official"]["raw"]["generated_text"] = "<think>\nFour.\n</think>\n4"
    write("red-model-empty-oracle-has-content", L, rec, crux, 1, r"llama.cpp on the OFFICIAL template does NOT reproduce think_empty")

    L, rec, crux = build_f9()   # lambda's receipt is REJECTED (stale sha): the key is NOT JUDGED, never "matches nothing"
    lam(rec)["apr_sha"] = "5" * 40
    write("red-model-receipt-rejected-not-judged", L, rec, crux, 1, r"not judged, because no receipt was accepted from \['lambda'\]",
          r"matches NO held file")

    L, rec, crux = build_f9()   # `prompts`: the claim covers the named prompt only; a closing prompt beside it is not evidence
    L["ladder"]["inventory"]["red_model"][D]["prompts"] = ["think-2plus2"]
    for lane in ("cpu", "gpu"):
        g = copy.deepcopy(crux[f"lambda-{lane}"]["greedy"][0])
        g["key"]["prompt_id"] = "fact-capital-france"
        for eng in ("llama.cpp", "llama.cpp@official"):
            g[eng]["raw"]["generated_text"] = "<think>\nParis is the capital.\n</think>\n<answer>Paris</answer>"
        crux[f"lambda-{lane}"]["greedy"].append(g)
    write("green-red-model-named-prompts", L, rec, crux, 0, r"RED-MODEL lambda +Qwen3.5-0.8B-IQ4_XS.gguf", r"FAIL")

    L, rec, crux = build_f9()   # a named prompt this sweep did not measure
    L["ladder"]["inventory"]["red_model"][D]["prompts"] = ["think-2plus2", "arith-17x23"]
    write("red-model-named-prompt-unmeasured", L, rec, crux, 1, r"claims the defect on prompt\(s\) \['arith-17x23'\]")

    # ---------------------------------------------------------------- F9 wrong_answer (cop rulings 2026-09-23)
    L, rec, crux = build_wa()
    write("green-red-model-wrong-answer-identical", L, rec, crux, 0,
          r"RED-MODEL lambda +Qwen3.5-0.8B-UD-IQ2_XXS.gguf +wrong_answer .*apr identical to the llama.cpp CPU reference", r"FAIL")
    L, rec, crux = build_wa(apr_div=4, ref_div=2)   # #4004's measurement: apr step 4 >= the oracle's own step 2
    write("green-red-model-wrong-answer-calibrated", L, rec, crux, 0,
          r"RED-MODEL lambda +Qwen3.5-0.8B-UD-IQ2_XXS.gguf +wrong_answer .*first diverges at step 4, not earlier than the oracle's own CPU/CUDA divergence at step 2", r"FAIL")
    L, rec, crux = build_wa(apr_div=2, ref_div=4)   # must-RED: apr diverges EARLIER than the oracle's self-divergence
    write("red-model-wrong-answer-apr-earlier", L, rec, crux, 1, r"apr diverges from the llama.cpp CPU reference at step 2.*EARLIER than the reference's own CUDA leg does \(step 4\)")
    L, rec, crux = build_wa(apr_div=4, ref_div=None)   # must-RED: the oracle agrees with itself fully while apr diverges
    write("red-model-wrong-answer-oracle-agrees", L, rec, crux, 1, r"the oracle's CPU and CUDA legs agree on every token, and apr diverges from them at step 4")
    L, rec, crux = build_wa(apr_div=4, ref_div=2)   # must-RED: the oracle's CUDA leg is missing
    crux["lambda-gpu"]["greedy"][0].pop("llama.cpp@official")
    write("red-model-wrong-answer-cuda-leg-missing", L, rec, crux, 1, r"the oracle's CUDA leg \(llama.cpp@official, GPU lane\) is MISSING")
    L, rec, crux = build_wa()   # must-RED: llama.cpp (CPU reference) answers correctly
    crux["lambda-cpu"]["greedy"][0]["llama.cpp@official"]["raw"]["generated_text"] = "<think>\n\n</think>\n\n2+2 = 4"
    write("red-model-wrong-answer-llama-correct", L, rec, crux, 1, r"llama.cpp CPU answers CORRECTLY")
    L, rec, crux = build_wa()   # must-RED: the higher-quant control is missing
    for X in crux.values():
        X["greedy"] = [g for g in X.get("greedy", []) if g["key"]["model_sha256"] != K_SHA]
    write("red-model-wrong-answer-control-missing", L, rec, crux, 1, r"has no apr \+ llama.cpp@official greedy record")
    L, rec, crux = build_wa()   # the control answers wrong too: the prompt, not the quant
    crux["lambda-gpu"]["greedy"][1]["apr"]["raw"]["generated_text"] = "The answer is five."
    write("red-model-wrong-answer-control-wrong", L, rec, crux, 1, r"does not answer '4' in both engines")
    L, rec, crux = build_wa()   # apr on a non-official prompt: the wrong answer may be the template
    crux["lambda-gpu"]["greedy"][0]["apr"]["raw"]["prompt_ids"] = PIDS
    write("red-model-wrong-answer-not-official", L, rec, crux, 1, r"apr did not run on the model's OFFICIAL template")
    L, rec, crux = build_wa()   # apr CPU != GPU while the oracle's own backends agree fully
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = div(5)
    write("red-model-wrong-answer-cpu-ne-gpu", L, rec, crux, 1, r"apr CPU and GPU diverge at step 5 while the oracle's CPU and CUDA legs agree on every token")
    L, rec, crux = build_wa(apr_div=None, ref_div=4)   # cop ruling (b) must-RED: apr's CPU/GPU split EARLIER than the oracle's
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = div(2)
    write("red-model-wrong-answer-cpu-gpu-earlier", L, rec, crux, 1, r"apr CPU and GPU diverge at step 2, EARLIER than the oracle's own CPU/CUDA split \(step 4\)")
    L, rec, crux = build_wa(apr_div=None, ref_div=2)   # #4004's shape: apr splits LATER than the oracle -> admitted
    crux["lambda-cpu"]["greedy"][0]["apr"]["raw"]["generated_ids"] = div(5)
    write("green-red-model-wrong-answer-cpu-gpu-calibrated", L, rec, crux, 0, r"apr CPU/GPU split at step 5, not earlier than the oracle's own at step 2", r"FAIL")
    L, rec, crux = build_wa()   # the apr "GPU" row fell back to the CPU (measured on IQ2_XXS, 165578f17)
    crux["lambda-gpu"]["greedy"][0]["apr"]["raw"]["backend"] = {"requested": "gpu", "ran": "cpu", "fell_back": True}
    write("red-model-wrong-answer-gpu-fell-back", L, rec, crux, 1, r"the apr GPU-lane row did NOT run on the GPU \(ran='cpu', fell_back=True\)")
    L, rec, crux = build_f9()   # ... and the think-block class holds its GPU leg to the same rule
    crux["lambda-gpu"]["greedy"][0]["apr"]["raw"]["backend"] = {"requested": "gpu", "ran": "cpu", "fell_back": True}
    write("red-model-gpu-fell-back", L, rec, crux, 1, r"the apr GPU-lane row did NOT run on the GPU")
    L, rec, crux = build_wa()   # a wrong_answer key must say what the right answer is
    del L["ladder"]["inventory"]["red_model"][W]["expect"]
    write("red-model-wrong-answer-no-expect", L, rec, crux, 1, r"is a wrong_answer key with no `expect`")

    # ---------------------------------------------------------------- F10 RED-UNSUPPORTED
    L, rec, crux = build_f10()
    write("green-red-unsupported-proven", L, rec, crux, 0,
          r"RED-UNSUPPORTED lambda +Qwen3.5-35B-A3B-UD-IQ4_XS.gguf +qwen35moe", r"FAIL|every required rung green")

    L, rec, crux = build_f10()   # the operator's must-RED: the model RUNS (refusal removed) -> RED
    be = row(rec, M)["backends"]["cuda"]
    be.update({"ran": True, "rc": 0})
    be["verbs"]["run"] = {"ran": True, "rc": 0, "refusal": None, "stdout_bytes": 490, "generated_bytes": 312}
    write("red-unsupported-model-runs", L, rec, crux, 1, r"declared RED-UNSUPPORTED .* NOT OBSERVED")

    L, rec, crux = build_f10()   # isolated: exit 0 alone (refusal text and zero bytes still present)
    row(rec, M)["backends"]["cuda"].update({"ran": True, "rc": 0})
    write("red-unsupported-ran", L, rec, crux, 1, r"exited 0 with ran=True -- the model ran")

    L, rec, crux = build_f10()   # isolated: generated output alone
    row(rec, M)["backends"]["cuda"]["verbs"]["run"]["generated_bytes"] = 42
    write("red-unsupported-stdout", L, rec, crux, 1, r"generated 42 bytes of output")

    L, rec, crux = build_f10()   # the operator's must-RED: it falls back
    row(rec, M)["backends"]["cuda"]["fallback"] = True
    write("red-unsupported-fell-back", L, rec, crux, 1, r"FELL BACK instead of refusing")

    L, rec, crux = build_f10()   # the operator's must-RED: no refusal observed
    row(rec, M)["backends"]["cuda"]["verbs"]["run"]["refusal"] = "error: CUDA out of memory"
    write("red-unsupported-no-refusal", L, rec, crux, 1, r"no refusal BY NAME was observed")

    L, rec, crux = build_f10(arch="qwen3moe")   # the header disagrees with the key
    write("red-unsupported-arch-mismatch", L, rec, crux, 1, r"the file's header says 'qwen3moe', the key says 'qwen35moe'")

    L, rec, crux = build_f10(arch="qwen35", key_arch="qwen35")   # the operator's must-RED: a supported architecture
    lam(rec)["rungs"][-2]["architecture"] = "qwen35"   # Extra-Q4_K_M.gguf, GREEN on cuda, of the same architecture
    write("red-unsupported-supported-arch", L, rec, crux, 1, r"the architecture HAS a CUDA path")

    L, rec, crux = build_f10()   # a key matching no held file
    L["ladder"]["inventory"]["red_unsupported"]["Qwen4-*-A9B*.gguf"] = {"architecture": "qwen4moe", "ticket": "#3977"}
    write("red-unsupported-key-no-file", L, rec, crux, 1, r"red_unsupported\['Qwen4-\*-A9B\*.gguf'\] matches NO held file")


if __name__ == "__main__":
    main()
