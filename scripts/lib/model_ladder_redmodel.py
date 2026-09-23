"""model_ladder_redmodel.py -- the ladder judge's two NAMED RED verdicts (#3957 F9, F10).

Two operator rulings (2026-09-23, on #3957) keep a held file in the capability report as an
explicit RED cell that does not fail the release:

  F9  RED-MODEL        "proceed with (a)": the FILE is defective, proven by llama.cpp.
                       Qwen3.5-0.8B-IQ4_XS never closes its think block (#3951);
                       Qwen3.5-0.8B-Q4_K_M closes it EMPTY (#3948 quorum).
  F10 RED-UNSUPPORTED  "c": apr has no CUDA forward for the architecture and refuses it by
                       name. Qwen3.5-35B-A3B (qwen35moe) until the 0.70.0 work.

Neither is a deferral in disguise. Each verdict is a CLAIM ABOUT CAUSE, and a claim about
cause is admitted only when THIS sweep re-proves it. A declaration alone buys nothing, the
same rule as the #3880 amnesty check on `inventory.deferred`. When a proof is missing or
disagrees, the row is plain FAIL and blocks, and the line names what failed.

RED-MODEL is proven, per (host, file), only when ALL of these hold on this sweep. The
evidence comes from the CRUX receipts bound to the cut (cop ruling 2026-09-23: one mechanism
answers "what does llama.cpp do on this file", not a second runner in the ladder):
  1. ENGINE PARITY ON apr's IDS. A thinking-ON greedy entry for this file's sha256 in a
     gpu-lane CRUX receipt carries RAW records for apr AND llama.cpp that ran on the SAME
     prompt ids, both greedy, the oracle printing special tokens, with the same max_tokens,
     and whose `generated_ids` are EQUAL. This judge compares the lists itself; it reads no
     `equal` or `first_divergence` flag.
  2. THE DEFECT ON THE OFFICIAL TEMPLATE (#3990, cop ruling 2026-09-23). apr's rendering is
     not the model's own chat template (measured by aprender-83: the official Qwen3.5 ON form
     opens `<think>\n`). An oracle that only ever saw apr's prompt would launder an apr
     TEMPLATE defect into a model verdict. So a `llama.cpp@official` row must have run on the
     official template's ids (`prompt_ids == template_prompt_ids`), and its `generated_text`
     must show the defect the key names: `think_never_closed` means no </think>, and
     `think_empty` means closed with no word inside. If it closes, the fault is apr's: plain RED.
  3. POSITIVE CONTROL ON A SIBLING. The key names a sibling file, held on the same host with
     a different sha. llama.cpp's greedy output for it closes the think block with content,
     and its thinking-ON positive-control CRUX cells are GREEN (at least one, none RED).
     This proves the instrument can see a working think block.
  4. apr CPU == GPU. apr's raw ids in the cpu-lane receipt equal those in the gpu-lane
     receipt for the same prompt.
  5. WHEN THE KEY SAYS `bf16_reproduces: true` (a defect of the MODEL, not of the quant),
     every hf/vLLM bf16 leg in that greedy entry, and at least one, also shows the defect.
     Those legs read other weights, so they are judged on their text, never on ids.
  5b. WHEN THE KEY NAMES `prompts`, the defect is claimed on exactly those prompts: each one
     must have a greedy entry, and entries for other prompts are not part of the claim (the
     2B loops on 4 of 7 thinking-ON prompts, measured by aprender-dd; one that closes is not
     evidence against a claim made on the other four). Without `prompts`, every entry is.
  6. THE AXIS IS thinking-ON ONLY. The key covers `thinking: on`; the file's thinking-OFF
     CRUX cells must still be GREEN, and model_ladder_crux enforces that.
  7. THE DEFECT IS THE WHOLE FAILURE. With golden_output neutralised (and qa_rc too, when
     golden_output is the only failed gate), the row must be green. RED-MODEL excuses the
     model's own think-block defect, never a serve route or a fallback beside it.

RED-UNSUPPORTED is proven, per (host, file), only when ALL of these hold:
  1. The row records the architecture from the file header (`architecture`), and it equals
     the key's.
  2. On cuda, `apr run --gpu` was OBSERVED refusing by name. rc != 0, no fallback on any
     verb, `generated_bytes == 0` (nothing generated: stdout minus apr's `verbose:` preamble), and a `refusal` text naming
     "no CUDA forward for architecture '<arch>'" plus "This is a refusal, not a fallback"
     (aprender-serve capability::no_cuda_forward_reason).
  3. The architecture has NO CUDA path, as measured on this sweep. A row of that
     architecture that is green on cuda, on any host, refuses the key.

Key hygiene (both kinds): each entry needs a `#NNNN` ticket. A key that matches no held file
on any host is REFUSED (#3880). A key that matches a green row is STALE: the cause is gone,
so delete the key. A file two keys cover is ambiguous and refused.
"""

import fnmatch
import glob
import json
import os
import re

import model_ladder_crux as crux

TICKET = re.compile(r"#\d+")
DEFECTS = {"think_never_closed": "never_closed", "think_empty": "empty", "wrong_answer": None}
AXES = {"on": {"on"}, "off": {"off"}, "any": {"on", "off"}}


def answer_of(text):
    """The final answer of a completion: the text after the last </think>, or all of it when there is
    no think block. An unclosed block has no answer (None)."""
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return None if "<think>" in text else text


def answers(text, expect):
    """True when `expect` appears as a whole token in the final answer (a number, a word)."""
    a = answer_of(text)
    return a is not None and re.search(r"(?<![\w.])" + re.escape(expect) + r"(?![\w])", a) is not None


def top2_margin(raw, step):
    """top-1 minus top-2 logit at generated `step`, from raw["top2_logits"][step] = [top1, top2]."""
    t = raw.get("top2_logits") if isinstance(raw, dict) else None
    try:
        a, b = t[step]
        return float(a) - float(b)
    except (TypeError, ValueError, IndexError, KeyError):
        return None
REFUSAL_CLASS = "This is a refusal, not a fallback"


def think_state(text):
    """-> 'never_closed' | 'empty' | 'ok' for a decoded thinking-ON completion.

    A Qwen3.5 thinking-ON template ends the prompt inside the block, so a completion may
    carry only the closing tag. Empty means no word of two or more characters inside, so a
    lone '.' or a whitespace run counts as empty (#3948 quorum, the single-token bypass)."""
    if not isinstance(text, str) or "</think>" not in text:
        return "never_closed"
    inside = text.split("</think>", 1)[0].replace("<think>", "")
    return "ok" if re.search(r"\w{2,}", inside) else "empty"


def _raw(g, engine):
    """An engine's raw greedy record, or None unless it carries a non-empty list of int ids."""
    r = ((g or {}).get(engine) or {}).get("raw")
    if not isinstance(r, dict):
        return None
    ids = r.get("generated_ids")
    if not (isinstance(ids, list) and ids and all(isinstance(i, int) and not isinstance(i, bool) for i in ids)):
        return None
    return r


def _margins(a, b, step):
    """Evidence only (cop ruling: logits are recorded, not the bar): the top-2 margins at `step`."""
    if step is None:
        return ""
    ma, mb = top2_margin(a, step), top2_margin(b, step)
    if ma is None and mb is None:
        return ""
    fmt = lambda m: "n/a" if m is None else f"{m:.3f}"
    return f" (top-2 margins there: apr {fmt(ma)}, reference {fmt(mb)})"


def gpu_leg_problem(raw):
    """#3957 F9: an apr GPU-lane row must PROVE it ran on the GPU. aprender-6c [3ada9a] measured a cuda
    build that FALLS BACK to the CPU on Qwen3.5-0.8B-UD-IQ2_XXS (IQ3_XXS attn_gate not admitted), so a
    "GPU" row from it is a CPU row with a GPU label -- and CPU == GPU would then compare a leg with
    itself. The row's own `backend` record (apr run --format json: requested/ran/fell_back) is read;
    an absent record is not a GPU run. -> None, or the reason."""
    be = raw.get("backend") if isinstance(raw, dict) else None
    if not isinstance(be, dict):
        return "the apr GPU-lane row records no `backend` -- nothing shows it ran on the GPU"
    if be.get("fell_back") is not False or str(be.get("ran") or "").lower() not in ("gpu", "cuda"):
        return (f"the apr GPU-lane row did NOT run on the GPU (ran={be.get('ran')!r}, fell_back={be.get('fell_back')!r}) "
                f"-- it is a CPU row with a GPU label, so CPU == GPU would compare a leg with itself")
    return None


def _ids(v):
    return isinstance(v, list) and bool(v) and all(isinstance(i, int) and not isinstance(i, bool) for i in v)


def _first_diff(a, b):
    i = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)
    return min(len(a), len(b)) if i is None and len(a) != len(b) else i


def neutralise_golden(x):
    """The row with its think-block defect taken out: golden_output passes, and qa_rc is 0 when
    golden_output is the only gate that failed. Everything else stays as measured."""
    y = dict(x)
    y["golden_output"] = {"passed": True, "skipped": False, "message": "neutralised for the RED-MODEL residual"}
    gf = set(x.get("gates_failed") or [])
    if isinstance(x.get("qa_rc"), int) and x.get("qa_rc") != 0 and gf and gf <= {"golden_output"}:
        y["qa_rc"] = 0
    return y


class RedVerdicts:
    def __init__(self, L, out):
        inv = L.get("inventory") or {}
        self.out = out
        self.failed = False
        self.tables = {"red_model": {}, "red_unsupported": {}}
        self.keys = []        # (kind, pattern) of every admitted key
        self.proven = {}      # (host, file) -> "RED-MODEL:<thinking axis>" | "RED-UNSUPPORTED"
        self.greedy = {}      # (sha, host, lane) -> [thinking-ON greedy entry]
        self.ctl_cells = {}   # (sha, host) -> [verdict of each thinking-ON positive-control cell]
        for kind in self.tables:
            src = inv.get(kind)
            if src is None:
                continue
            if not isinstance(src, dict):
                self._fail(f"inventory.{kind} is not a mapping of file glob -> entry (#3957 F9/F10)")
                continue
            for pat, e in src.items():
                bad = self._entry_problem(kind, e)
                if bad:
                    self._fail(f"inventory.{kind}[{pat!r}] {bad} -- the key is refused (#3957 F9/F10)")
                    continue
                self.tables[kind][pat] = e
                self.keys.append((kind, pat))

    def _fail(self, msg):
        self.out("FAIL  " + msg)
        self.failed = True

    @staticmethod
    def _entry_problem(kind, e):
        if not isinstance(e, dict):
            return "is not a mapping"
        if not TICKET.search(str(e.get("ticket") or "")):
            return "names no `#NNNN` ticket carrying the evidence"
        if kind == "red_model":
            if e.get("defect") not in DEFECTS:
                return f"declares defect {e.get('defect')!r}, not one of {sorted(DEFECTS)}"
            if not isinstance(e.get("control"), str) or not e["control"].strip():
                return "names no sibling `control` file for the oracle's positive control"
            if e["defect"] == "wrong_answer":
                if e.get("thinking") not in AXES:
                    return f"covers thinking {e.get('thinking')!r}; a wrong_answer key names its axis: on, off or any"
                if not isinstance(e.get("expect"), str) or not e["expect"].strip():
                    return "is a wrong_answer key with no `expect` -- a wrong answer needs the right one to be judged against"
            elif e.get("thinking") != "on":
                return f"covers thinking {e.get('thinking')!r}; a think-block defect is a thinking-ON verdict (`thinking: on`)"
            if e.get("bf16_reproduces", False) not in (True, False):
                return "has a `bf16_reproduces` that is not a boolean"
            ps = e.get("prompts")
            if ps is not None and not (isinstance(ps, list) and ps and all(isinstance(p, str) and p for p in ps)):
                return "has a `prompts` that is not a non-empty list of prompt ids"
        elif not isinstance(e.get("architecture"), str) or not e["architecture"].strip():
            return "names no `architecture`"
        return None

    # ------------------------------------------------------------ CRUX evidence
    def load_crux(self, crux_dir, cut, equiv):
        """Index the thinking-ON greedy entries and positive-control cells of every CRUX receipt bound
        to the cut that did not DECLINE. model_ladder_crux reports the unbound and declined ones."""
        if not self.tables["red_model"] or not crux_dir:
            return
        for f in sorted(glob.glob(os.path.join(crux_dir, "*.json"))):
            try:
                with open(f, encoding="utf-8") as fh:
                    R = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(R, dict) or R.get("schema") != "crux-inference-receipt/v1":
                continue
            asha = crux.apr_sha_of(R)
            if not (asha and crux.HEX40.fullmatch(asha)) or (asha != cut and asha not in equiv):
                continue
            if (R.get("summary") or {}).get("verdict") == "DECLINE":
                continue
            lane = R.get("backend")
            for g in R.get("greedy") or []:
                k = g.get("key") or {}
                if k.get("thinking") in ("on", "off"):
                    self.greedy.setdefault((k.get("model_sha256"), k.get("host") or R.get("host"), lane), []).append(g)
            for c in R.get("cells") or []:
                k = c.get("key") or {}
                if c.get("positive_control") and k.get("thinking") == "on":
                    self.ctl_cells.setdefault((k.get("model_sha256"), k.get("host") or R.get("host")), []).append(c.get("verdict"))

    # ------------------------------------------------------------ per row
    def classify(self, host, f, x, why, residual_of, sha_of):
        """-> None when no key covers `f`, else (blocking, line). A proven named RED does not block;
        everything else is FAIL. `residual_of(row)` is the ladder's why_of for that row, and
        `sha_of(host, file)` is a held file's sha256 on that host."""
        hits = self._hits(f)
        if not hits:
            return None
        tag = f"{host:7} {f:22}"
        if len(hits) > 1:
            return True, f"FAIL  {tag} is covered by {len(hits)} red-verdict keys {[p for _, p in hits]} -- one cause per file (#3957 F9/F10)"
        kind, pat = hits[0]
        e = self.tables[kind][pat]
        if not why:
            return True, (f"FAIL  {tag} inventory.{kind}[{pat!r}] is STALE: the row is GREEN, so the declared cause is gone. "
                          f"Delete the key; a green file is reported green (#3957 F9/F10, #3880)")
        if kind == "red_model":
            probs, evidence = self._prove_model(host, f, x, e, sha_of)
            resid = residual_of(neutralise_golden(x))
            if resid:
                probs.append(f"RED-MODEL excuses only the declared {e['defect']} defect, and the row ALSO fails: " + "; ".join(resid))
            if probs:
                return True, (f"FAIL  {tag} declared RED-MODEL ({e['defect']}, {e['ticket']}) but NOT RE-PROVEN on this sweep, "
                              f"so it is plain RED: " + "; ".join(probs) + " -- was: " + "; ".join(why))
            # The axis travels with the verdict: model_ladder_crux sets aside that axis's cells only.
            self.proven[(host, f)] = "RED-MODEL:" + ",".join(sorted(AXES.get(e["thinking"], {"on"})))
            return False, f"RED-MODEL {tag} {e['defect']} ({e['ticket']}) -- the FILE is defective, never green: {evidence}"
        probs, refusal = self._prove_unsupported(x, e)
        if probs:
            return True, (f"FAIL  {tag} declared RED-UNSUPPORTED ({e['architecture']}, {e['ticket']}) but NOT OBSERVED on this "
                          f"sweep, so it is plain RED: " + "; ".join(probs) + " -- was: " + "; ".join(why))
        self.proven[(host, f)] = "RED-UNSUPPORTED"
        return False, (f"RED-UNSUPPORTED {tag} {e['architecture']} ({e['ticket']}) -- no CUDA forward, refused by name, "
                       f"never green: {refusal!r}")

    def _hits(self, f):
        return [(kind, p) for kind, p in self.keys if fnmatch.fnmatch(str(f).lower(), str(p).lower())]

    def _greedy_on(self, sha, host, lane, axis=frozenset({"on"})):
        return [g for g in self.greedy.get((sha, host, lane)) or [] if (g.get("key") or {}).get("thinking") in axis]

    def _prove_model(self, host, f, x, e, sha_of):
        if e["defect"] == "wrong_answer":
            return self._prove_wrong_answer(host, f, x, e, sha_of)
        sha = x.get("sha256") or sha_of(host, f)
        if not (isinstance(sha, str) and crux.HEX64.fullmatch(sha)):
            return ["the row has no 64-hex sha256, so no oracle can be joined to it"], ""
        gpu = self._greedy_on(sha, host, "gpu")
        named = e.get("prompts")
        if named:
            gpu = [g for g in gpu if (g.get("key") or {}).get("prompt_id") in named]
            have = {(g.get("key") or {}).get("prompt_id") for g in gpu}
            gone = [p for p in named if p not in have]
            if gone:
                return [f"the key claims the defect on prompt(s) {gone}, and this sweep has no thinking-ON greedy entry "
                        f"for them on {host} (gpu lane) -- a claim nothing measured (#3957 F9)"], ""
        if not gpu:
            return [f"no thinking-ON greedy entry for sha {sha[:12]} on {host} in a gpu-lane CRUX receipt bound to the cut -- "
                    f"no oracle ran on this sweep (#3957 F9)"], ""
        cpu = {(g.get("key") or {}).get("prompt_id"): g for g in self._greedy_on(sha, host, "cpu")}
        want = DEFECTS[e["defect"]]
        probs, shown = [], []
        for g in gpu:
            pid = (g.get("key") or {}).get("prompt_id")
            a, o = _raw(g, "apr"), _raw(g, "llama.cpp")
            if a is None or o is None:
                probs.append(f"{pid}: no raw generated_ids for " + " and ".join(n for n, v in (("apr", a), ("llama.cpp", o)) if v is None))
                continue
            p = []
            gp = gpu_leg_problem(a)
            if gp:
                p.append(f"{pid}: {gp}")
            if not (_ids(a.get("prompt_ids")) and a.get("prompt_ids") == o.get("prompt_ids")):
                p.append(f"{pid}: the parity row's engines did not run on the same prompt ids -- a divergence would be the "
                         f"prompt's, not the engine's")
            if not (a.get("greedy") is True and o.get("greedy") is True and o.get("special") is True):
                p.append(f"{pid}: the runs are not both greedy with the oracle printing special tokens")
            if a.get("max_tokens") != o.get("max_tokens"):
                p.append(f"{pid}: max_tokens differ (apr {a.get('max_tokens')!r}, llama.cpp {o.get('max_tokens')!r})")
            if a["generated_ids"] != o["generated_ids"]:
                p.append(f"{pid}: apr's greedy ids DIFFER from llama.cpp's on the identical GGUF at step "
                         f"{_first_diff(a['generated_ids'], o['generated_ids'])} -- the output is apr's, not the file's")
            off = _raw(g, "llama.cpp@official")
            if off is None:
                p.append(f"{pid}: no llama.cpp@official row -- the defect was never shown on the model's own template, only "
                         f"on apr's rendering (#3990)")
            elif not (_ids(off.get("template_prompt_ids")) and off.get("prompt_ids") == off.get("template_prompt_ids")):
                p.append(f"{pid}: the llama.cpp@official row did not run on the official template's ids -- it proves "
                         f"nothing about the model under its own template (#3990)")
            elif think_state(off.get("generated_text")) != want:
                p.append(f"{pid}: llama.cpp on the OFFICIAL template does NOT reproduce {e['defect']} (its think block is "
                         f"{think_state(off.get('generated_text'))}) -- the defect is apr's")
            if e.get("bf16_reproduces"):
                legs = {n: g.get(n) for n in ("hf", "vllm") if isinstance(g.get(n), dict)}
                texts = {n: ((v.get("raw") or {}).get("generated_text")) for n, v in legs.items()}
                if not texts:
                    p.append(f"{pid}: the key says the bf16 model reproduces it, and no hf/vLLM leg ran")
                for n, t in sorted(texts.items()):
                    if think_state(t) != want:
                        p.append(f"{pid}: the bf16 {n} leg does NOT reproduce {e['defect']} (its think block is {think_state(t)}) "
                                 f"-- the model is not at fault")
            ca = _raw(cpu.get(pid), "apr")
            if ca is None:
                p.append(f"{pid}: no apr CPU leg with raw ids in a cpu-lane receipt -- CPU == GPU is unmeasured")
            elif ca["generated_ids"] != a["generated_ids"]:
                p.append(f"{pid}: apr CPU and GPU DIFFER at step {_first_diff(ca['generated_ids'], a['generated_ids'])}")
            probs.extend(p)
            if not p:
                shown.append(f"{pid}: {len(o['generated_ids'])} ids equal")
        cfile = e["control"]
        csha = sha_of(host, cfile)
        if not (isinstance(csha, str) and crux.HEX64.fullmatch(csha)):
            probs.append(f"the positive-control sibling {cfile} is not held on {host}")
        elif csha == sha or cfile == f:
            probs.append(f"the positive control {cfile} is the defective file itself, so it controls nothing")
        else:
            co = [_raw(g, "llama.cpp@official") for g in self._greedy_on(csha, host, "gpu")]
            co = [r if r is not None and _ids(r.get("template_prompt_ids")) and r.get("prompt_ids") == r.get("template_prompt_ids")
                  else None for r in co]
            if not co or any(r is None for r in co):
                probs.append(f"no llama.cpp@official thinking-ON greedy output on the official template for the control "
                             f"{cfile} on {host} -- the instrument was not shown to see a working think block")
            elif any(think_state(r.get("generated_text")) != "ok" for r in co):
                probs.append(f"the control {cfile} does not close its think block with content under llama.cpp -- the "
                             f"instrument is blind, so the attribution means nothing")
            vs = self.ctl_cells.get((csha, host)) or []
            if not vs:
                probs.append(f"the control {cfile} has no thinking-ON positive-control CRUX cell on {host}")
            elif any(v != "GREEN" for v in vs):
                probs.append(f"the control {cfile}'s positive-control CRUX cells are not all GREEN ({vs})")
        return probs, (f"apr == llama.cpp on apr's ids ({', '.join(shown)}); llama.cpp shows it on the official template; "
                       f"control {cfile} closes and answers; apr CPU == GPU")

    def _prove_wrong_answer(self, host, f, x, e, sha_of):
        """#3957 F9 `wrong_answer` (cop rulings 2026-09-23, within operator ruling (a)): the FILE answers
        wrong, proven by llama.cpp. PARITY IS CALIBRATED PER SWEEP against the oracle's own noise
        (aprender-6c [3ada9a], #4004: llama.cpp CPU vs llama.cpp CUDA, same build, identical official
        ids, first-diverge at step 2 -- so no fixed margin separates an apr fault from the reference's
        backend noise). Per greedy entry on the key's thinking axis, all on the OFFICIAL template ids:
          - reference = llama.cpp@official on the CPU lane; oracle CUDA leg = llama.cpp@official on
            the GPU lane; apr = apr on the GPU lane (with apr CPU == GPU required separately);
          - apr ran the official ids (its prompt_ids == the template's);
          - apr's first divergence step from the reference is >= the oracle CUDA leg's own first
            divergence step from the reference (apr is at least as close to the CPU reference as the
            reference's own GPU backend). apr identical to the reference passes; the oracle's two
            backends agreeing FULLY while apr diverges is plain RED; a missing CUDA leg is plain RED;
          - apr and both llama.cpp backends all answer WRONG (`expect` absent from the final answer);
          - apr's own CPU/GPU first divergence is >= the oracle's CPU/CUDA one (cop ruling (b));
        and the `control` (the same architecture at a higher quant) answers CORRECTLY on the same
        prompt in BOTH engines. raw.top2_logits, when present, are reported as evidence, never the bar."""
        sha = x.get("sha256") or sha_of(host, f)
        if not (isinstance(sha, str) and crux.HEX64.fullmatch(sha)):
            return ["the row has no 64-hex sha256, so no oracle can be joined to it"], ""
        axis = AXES[e["thinking"]]
        expect = e["expect"].strip()
        gpu = self._greedy_on(sha, host, "gpu", axis)
        named = e.get("prompts")
        if named:
            gpu = [g for g in gpu if (g.get("key") or {}).get("prompt_id") in named]
            gone = [p for p in named if p not in {(g.get("key") or {}).get("prompt_id") for g in gpu}]
            if gone:
                return [f"the key claims the wrong answer on prompt(s) {gone}, and this sweep has no greedy entry for them"], ""
        if not gpu:
            return [f"no thinking-{e['thinking']} greedy entry for sha {sha[:12]} on {host} in a gpu-lane CRUX receipt "
                    f"bound to the cut -- no oracle ran on this sweep (#3957 F9)"], ""

        def by_pid(entries):
            return {((g.get("key") or {}).get("prompt_id"), (g.get("key") or {}).get("thinking")): g for g in entries}
        cpu = by_pid(self._greedy_on(sha, host, "cpu", axis))
        cfile = e["control"]
        csha = sha_of(host, cfile)
        ctl = by_pid(self._greedy_on(csha, host, "gpu", axis)) if isinstance(csha, str) else {}
        probs, shown = [], []
        if not (isinstance(csha, str) and crux.HEX64.fullmatch(csha)):
            probs.append(f"the higher-quant control {cfile} is not held on {host}")
        elif csha == sha or cfile == f:
            probs.append(f"the control {cfile} is the defective file itself, so it controls nothing")
        for g in gpu:
            k = g.get("key") or {}
            pid, th = k.get("prompt_id"), k.get("thinking")
            c = cpu.get((pid, th))
            a, cuda, ref, ca = _raw(g, "apr"), _raw(g, "llama.cpp@official"), _raw(c, "llama.cpp@official"), _raw(c, "apr")
            if ref is None:
                probs.append(f"{pid}/{th}: no llama.cpp@official CPU-lane record -- there is no reference to measure against")
                continue
            if cuda is None:
                probs.append(f"{pid}/{th}: the oracle's CUDA leg (llama.cpp@official, GPU lane) is MISSING -- the "
                             f"reference's own backend noise is unmeasured, so apr cannot be calibrated against it")
                continue
            if a is None:
                probs.append(f"{pid}/{th}: no raw apr greedy record on the GPU lane")
                continue
            p = []
            gp = gpu_leg_problem(a)
            if gp:
                p.append(f"{pid}/{th}: {gp}")
            for n, r in (("CPU", ref), ("CUDA", cuda)):
                if not (_ids(r.get("template_prompt_ids")) and r.get("prompt_ids") == r.get("template_prompt_ids")):
                    p.append(f"{pid}/{th}: the llama.cpp@official {n} leg did not run on the official template's ids (#3990)")
            if cuda.get("prompt_ids") != ref.get("prompt_ids"):
                p.append(f"{pid}/{th}: the oracle's CPU and CUDA legs did not run on identical ids")
            if a.get("prompt_ids") != ref.get("template_prompt_ids"):
                p.append(f"{pid}/{th}: apr did not run on the model's OFFICIAL template, so its wrong answer may be apr's "
                         f"prompt, not the model (#3990)")
            d_apr = _first_diff(a["generated_ids"], ref["generated_ids"])
            d_ref = _first_diff(cuda["generated_ids"], ref["generated_ids"])
            ev = _margins(a, ref, d_apr)
            if d_apr is None:
                shown.append(f"{pid}/{th}: apr identical to the llama.cpp CPU reference ({len(a['generated_ids'])} ids)")
            elif d_ref is None:
                p.append(f"{pid}/{th}: the oracle's CPU and CUDA legs agree on every token, and apr diverges from them at "
                         f"step {d_apr}{ev} -- the divergence is apr's, not noise the reference shares")
            elif d_apr < d_ref:
                p.append(f"{pid}/{th}: apr diverges from the llama.cpp CPU reference at step {d_apr}{ev}, EARLIER than the "
                         f"reference's own CUDA leg does (step {d_ref}) -- apr is further from the reference than its noise")
            else:
                shown.append(f"{pid}/{th}: apr first diverges at step {d_apr}{ev}, not earlier than the oracle's own "
                             f"CPU/CUDA divergence at step {d_ref}")
            for n, r in (("apr", a), ("llama.cpp CPU", ref), ("llama.cpp CUDA", cuda)):
                if answers(r.get("generated_text"), expect):
                    p.append(f"{pid}/{th}: {n} answers CORRECTLY ({expect!r}) -- the file can answer, so the wrong answer "
                             f"is not the model's")
            # Cop ruling (b), 2026-09-23: apr's own CPU/GPU split is calibrated like parity -- it must come
            # no earlier than the oracle's own CPU/CUDA split on the same ids (6c: apr splits at step 31 on a
            # 0.015 margin, the oracle at step 2).
            d_cg = None if ca is None else _first_diff(ca["generated_ids"], a["generated_ids"])
            if ca is None:
                p.append(f"{pid}/{th}: no apr CPU leg with raw ids in a cpu-lane receipt -- apr CPU vs GPU is unmeasured")
            elif d_cg is None:
                shown.append(f"{pid}/{th}: apr CPU == GPU")
            elif d_ref is None:  # the oracle never split: apr CPU/GPU split is its own
                p.append(f"{pid}/{th}: apr CPU and GPU diverge at step {d_cg}{_margins(ca, a, d_cg)} while the oracle's CPU "
                         f"and CUDA legs agree on every token -- apr's split is its own, not shared noise")
            elif d_cg < d_ref:
                p.append(f"{pid}/{th}: apr CPU and GPU diverge at step {d_cg}, EARLIER than the oracle's own CPU/CUDA split "
                         f"(step {d_ref}) -- apr's backends disagree more than the reference's do")
            else:
                shown.append(f"{pid}/{th}: apr CPU/GPU split at step {d_cg}{_margins(ca, a, d_cg)}, not earlier than the "
                             f"oracle's own at step {d_ref}")
            cc = ctl.get((pid, th))
            ca2, co2 = _raw(cc, "apr"), _raw(cc, "llama.cpp@official")
            if ca2 is None or co2 is None:
                p.append(f"{pid}/{th}: the higher-quant control {cfile} has no apr + llama.cpp@official greedy record for "
                         f"this prompt -- nothing shows the architecture CAN answer it")
            elif not (answers(ca2.get("generated_text"), expect) and answers(co2.get("generated_text"), expect)):
                p.append(f"{pid}/{th}: the higher-quant control {cfile} does not answer {expect!r} in both engines -- the "
                         f"prompt, not the quant, may be at fault")
            probs.extend(p)
        return probs, (f"wrong answer in apr and both llama.cpp backends on the official template: {'; '.join(shown)}; "
                       f"control {cfile} answers {e['expect']!r} in both engines")

    @staticmethod
    def _prove_unsupported(x, e):
        arch = e["architecture"]
        probs = []
        got = x.get("architecture")
        if not isinstance(got, str) or not got:
            probs.append("the row records no header `architecture`, so the declared one is unverified")
        elif got != arch:
            probs.append(f"the file's header says {got!r}, the key says {arch!r}")
        be = (x.get("backends") or {}).get("cuda")
        if not isinstance(be, dict):
            return probs + ["cuda was not measured, so no refusal was observed"], None
        run = (be.get("verbs") or {}).get("run") or {}
        refusal = run.get("refusal")
        if be.get("fallback") or any(isinstance(v, dict) and v.get("fallback") for v in (be.get("verbs") or {}).values()):
            probs.append("apr FELL BACK instead of refusing")
        if be.get("rc") in (0, None) or be.get("ran") is not False:
            probs.append(f"`apr run --gpu` exited {be.get('rc')!r} with ran={be.get('ran')!r} -- the model ran, so it is not unsupported")
        if run.get("generated_bytes") != 0:
            probs.append(f"`apr run --gpu` generated {run.get('generated_bytes')!r} bytes of output -- a refusal generates nothing")
        needle = f"no CUDA forward for architecture '{arch}'"
        if not isinstance(refusal, str) or needle not in refusal or REFUSAL_CLASS not in refusal:
            probs.append(f"no refusal BY NAME was observed (want {needle!r} and {REFUSAL_CLASS!r}, got {refusal!r})")
        return probs, refusal

    # ------------------------------------------------------------ after every host
    def finish(self, good, held, required, green_on_cuda):
        """Refuse keys that match no held file, and red_unsupported keys whose architecture a green row
        proves to have a CUDA path. `held` is {host: [file]} from the ACCEPTED receipts, `required`
        the required host ids, and `green_on_cuda(row)` the ladder's own verdict. -> failed.

        "Matches no held file" is judged only when every required host's receipt was accepted: a
        rejected receipt (stale, unreadable) held files nobody read, and claiming the key matched
        nothing there would be a false statement (measured: the stale 0.69.1 receipts at 7b8aa7e32)."""
        missing = sorted(set(required) - set(held))
        for kind, pat in self.keys:
            n = sum(1 for files in held.values() for f in files if (kind, pat) in self._hits(f))
            if n:
                continue
            if missing:
                self.out(f"note  inventory.{kind}[{pat!r}] matches no file in the ACCEPTED receipts; not judged, "
                         f"because no receipt was accepted from {missing} (the FAIL above stands)")
            else:
                self._fail(f"inventory.{kind}[{pat!r}] matches NO held file on any host -- a verdict for a file nobody "
                           f"holds is amnesty, not evidence (#3957 F9/F10, #3880)")
        for pat, e in self.tables["red_unsupported"].items():
            arch = e["architecture"]
            for host, R in sorted(good.items()):
                for x in R.get("rungs") or []:
                    if x.get("present") and x.get("architecture") == arch and green_on_cuda(x):
                        self._fail(f"inventory.red_unsupported[{pat!r}] declares {arch!r} unsupported, but {host} "
                                   f"{x.get('file')} of that architecture is GREEN on cuda -- the architecture HAS a CUDA "
                                   f"path, so the key is refused (#3957 F10)")
        if self.proven:
            n_m = sum(1 for v in self.proven.values() if v.startswith("RED-MODEL"))
            self.out(f"RED   {n_m} RED-MODEL + {len(self.proven) - n_m} RED-UNSUPPORTED cell(s): counted RED, never green. "
                     f"They do not block only because this sweep re-proved each cause (#3957 F9/F10)")
        return self.failed
