"""model_ladder_crux.py -- the ladder judge's EXTERNAL ORACLE join (#3957 F4, F8).

The ladder receipt says what apr did: it ran, on which backend, without falling back,
and whether the output looked degenerate. None of that says the output was RIGHT.
The CRUX receipts (scripts/crux_inference_dogfood.sh -> scripts/lib/crux_inference_judge.py)
are where a verb's output is judged against engines that are not apr. This module
is where the release gate reads them. A cell is:

    (model, format, quant, host, backend, verb)

and it is PROVEN only when the CRUX receipts bound to the cut judged it GREEN. There
is no third state. A cell no CRUX receipt covers is RED "no oracle", never skipped:
the gate cannot tell an unmeasured verb from a working one, and #3828/#3957 exist
because absence used to read as conformance.

FORMAT DECIDES THE ORACLE (cop ruling on #3957, 2026-09-23). GGUF and SafeTensors
cells are judged by CRUX directly: the same-representation engines are the ggml family
on the identical GGUF, and hf/vLLM on the identical SafeTensors weights. No engine
other than apr reads .apr, so a .apr cell reaches an outside oracle only through a
CHAIN (F8, amended by quorum finding Q4). The chain is PROVEN only when all of these hold:
  1. the row records its conversion source: file, 64-hex sha256;
  2. `.apr` == source tensor by tensor, ELEMENT-WISE (`apr diff`, not a summary
     statistic): method "elementwise", at least one tensor compared, zero differing;
  3. at runtime, apr on the .apr gives the same greedy answer as apr on the source,
     on every claimed backend (Q4: tensors at rest do not prove layout, deserialization
     or kernels);
  4. the source's own cell for this (host, backend, verb) is CRUX GREEN.
If any link is missing, the cell is RED "no external oracle".

A CRUX receipt is evidence only when (a) it is bound to the cut by a 40-hex apr sha,
by the same rule as the ladder receipts (#3957 F2), and (b) it did not DECLINE. A
declined receipt means its harness could not certify its own lane (no positive
control, or a negative control it could not see), so none of its cells count.
"""

import glob
import json
import os
import re

#: ladder verb -> the verb string a CRUX cell key carries
VERB_TO_CRUX = {"run": "run", "chat": "chat", "serve": "serve run", "code": "code"}
#: ladder backend -> the CRUX receipt's lane (`backend` field)
LANE = {"cuda": "gpu", "gpu": "gpu", "cpu": "cpu"}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
QUANT = re.compile(r"(iq\d_[a-z]+|q\d_k(?:_[a-z]+)?|q\dk|q\d_\d|bf16|fp16|f16|f32)", re.I)


def fmt_of(fname):
    f = (fname or "").lower()
    if f.endswith(".gguf"):
        return "gguf"
    if f.endswith(".apr"):
        return "apr"
    if f.endswith(".safetensors") or f.endswith("-st") or "safetensors" in f:
        return "safetensors"
    return "unknown"


def quant_of(fname):
    m = QUANT.search(os.path.basename(fname or ""))
    return m.group(1).upper() if m else "?"


def apr_sha_of(receipt):
    a = receipt.get("apr")
    sha = (a.get("sha") if isinstance(a, dict) else None) or receipt.get("apr_sha")
    return sha if isinstance(sha, str) else None


def load_crux(crux_dir, cut, equiv, out):
    """-> ({(sha, host, lane, crux_verb): [verdict, ...]}, failed)."""
    index, failed = {}, False
    files = sorted(glob.glob(os.path.join(crux_dir or "", "*.json"))) if crux_dir else []
    if not files:
        out(f"FAIL  no CRUX receipt under {crux_dir!r} -- no verb of any model has been judged against an "
            f"outside engine, so every cell below is unproven (#3957 F4)")
        return index, True
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                R = json.load(fh)
        except (OSError, ValueError) as exc:
            out(f"FAIL  CRUX receipt {f} unreadable: {exc}")
            failed = True
            continue
        if not isinstance(R, dict) or R.get("schema") != "crux-inference-receipt/v1":
            out(f"FAIL  {f} is not a crux-inference-receipt/v1 (schema {R.get('schema') if isinstance(R, dict) else None!r})")
            failed = True
            continue
        asha = apr_sha_of(R)
        if not (asha and HEX40.fullmatch(asha)) or (asha != cut and asha not in equiv):
            out(f"FAIL  CRUX receipt {os.path.basename(f)} is bound to apr sha {asha!r}, not the cut {cut[:12]} -- "
                f"its cells are not evidence for this build (#3957 F2)")
            failed = True
            continue
        summ = R.get("summary") or {}
        if summ.get("verdict") == "DECLINE":
            out(f"FAIL  CRUX receipt {os.path.basename(f)} DECLINED ({summ.get('declined_because')}) -- a lane that "
                f"could not certify its own controls vouches for nothing (#3957 F6)")
            failed = True
            continue
        lane = R.get("backend")
        for c in R.get("cells") or []:
            k = c.get("key") or {}
            key = (k.get("model_sha256"), k.get("host") or R.get("host"), lane, k.get("verb"))
            index.setdefault(key, []).append(c.get("verdict"))
    return index, failed


def crux_cell(index, sha, host, backend, verb):
    """-> (proven, reason) for one (model sha, host, backend, verb)."""
    got = index.get((sha, host, LANE.get(backend, backend), VERB_TO_CRUX.get(verb, verb)))
    if not got:
        return False, "no CRUX verdict: no outside engine has judged this cell, so it is not proven"
    bad = [v for v in got if v != "GREEN"]
    if bad:
        counts = {v: got.count(v) for v in sorted(set(got))}
        return False, "CRUX: " + ", ".join(f"{n} {v}" for v, n in counts.items())
    return True, f"CRUX GREEN ({len(got)} cell(s))"


def apr_chain(x, backends):
    """The F8 links that live on the .apr row itself. -> (source sha or None, [reasons])."""
    src = x.get("source")
    if not isinstance(src, dict) or not src.get("file") or not HEX64.fullmatch(str(src.get("sha256") or "")):
        return None, ["no external oracle: the .apr records no conversion source (file + 64-hex sha256), "
                      "and no engine but apr reads .apr (#3957 F8)"]
    why = []
    td = src.get("tensor_diff")
    if not isinstance(td, dict) or td.get("method") != "elementwise" or not isinstance(td.get("tensors_compared"), int) \
            or td.get("tensors_compared") < 1:
        why.append("no element-wise tensor diff against the source -- a summary statistic cannot see a scrambled "
                   "tensor, and an absent diff proves nothing (#3957 F8)")
    elif td.get("tensors_differing") != 0:
        why.append(f"the .apr DIFFERS from its source in {td.get('tensors_differing')!r} of "
                   f"{td.get('tensors_compared')} tensor(s) (max |d| {td.get('max_abs_diff')}) (#3957 F8)")
    ge = src.get("greedy_equal")
    for b in backends:
        e = (ge or {}).get(b) if isinstance(ge, dict) else None
        if not isinstance(e, dict) or "equal" not in e:
            why.append(f"{b}: apr-on-.apr vs apr-on-source greedy output NOT MEASURED -- tensors at rest do not prove "
                       f"layout, deserialization or kernels (#3957 Q4)")
        elif e.get("equal") is not True:
            why.append(f"{b}: apr on the .apr answered {e.get('apr_on_apr')!r}, apr on the source {e.get('apr_on_source')!r} "
                       f"-- the runtime diverges from the file it was converted from (#3957 Q4)")
    return src["sha256"], why


def judge(L, good, crux_dir, cut, equiv, out):
    """Print one line per cell and a per-format summary. -> True when any cell is not proven."""
    verbs = list((L.get("cells") or {}).get("verbs") or ["run", "chat", "serve", "code"])
    inv_backends = list((L.get("inventory") or {}).get("backends") or [])
    rung_by_file = {r.get("gguf"): r for r in L.get("rungs") or []}
    index, failed = load_crux(crux_dir, cut, equiv, out)
    tally = {}
    for host in sorted(good):
        R = good[host]
        inv_sha = {i.get("file"): i.get("sha256") for i in R.get("inventory") or [] if isinstance(i, dict)}
        for x in R.get("rungs") or []:
            if not x.get("present"):
                continue
            f = x.get("file") or ""
            r = rung_by_file.get(f)
            if r is not None and r.get("hosts") and host not in r["hosts"]:
                continue
            backends = list(r.get("backends") or []) if r is not None else inv_backends
            sha = x.get("sha256") or inv_sha.get(f) or (r or {}).get("sha256")
            fmt, q = fmt_of(f), quant_of(f)
            src_sha, chain_why = (apr_chain(x, backends) if fmt == "apr" else (None, []))
            for b in backends:
                for v in verbs:
                    label = f"{f} fmt={fmt} quant={q} host={host} backend={b} verb={v}"
                    t = tally.setdefault(fmt, [0, 0])
                    t[0] += 1
                    if not (isinstance(sha, str) and HEX64.fullmatch(sha)):
                        ok, why = False, "the row has no 64-hex sha256, so no oracle can be joined to it"
                    elif fmt == "apr":
                        per_b = [w for w in chain_why if not w.startswith(tuple(f"{o}:" for o in backends if o != b))]
                        if src_sha is None:
                            ok, why = False, per_b[0]
                        else:
                            s_ok, s_why = crux_cell(index, src_sha, host, b, v)
                            reasons = per_b + ([] if s_ok else [f"source {src_sha[:12]} is not proven: {s_why} (#3957 F8)"])
                            ok, why = (not reasons), ("; ".join(reasons) if reasons else f"chain to source {src_sha[:12]}: {s_why}")
                    else:
                        ok, why = crux_cell(index, sha, host, b, v)
                    if ok:
                        t[1] += 1
                        out(f"ok    cell {label} -- {why}")
                    else:
                        failed = True
                        out(f"FAIL  cell {label} -- {why}")
    out("cells by format: " + ", ".join(f"{k} {v[1]}/{v[0]} proven" for k, v in sorted(tally.items()))
        if tally else "FAIL  no cell was owed -- a release that measured no cell proved nothing")
    return failed or not tally
