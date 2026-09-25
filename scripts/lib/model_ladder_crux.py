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


#: `apr --version`'s own line, as crux_inference_dogfood.sh records it: "apr 0.69.1 (d8a6df53a)"
APR_VERSION_SHA = re.compile(r"^apr \S+ \(([0-9a-f]{7,40})\)$")


def _git_resolve(short):
    """A short sha -> the full commit sha in this checkout, or None (unknown or ambiguous)."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", short + "^{commit}"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    full = r.stdout.strip()
    return full if r.returncode == 0 and HEX40.fullmatch(full) else None


def apr_sha_of(receipt, resolve=None):
    """The FULL sha of the apr binary a receipt measured, or None.

    An explicit apr.sha / apr_sha wins. Otherwise the sha comes from the BINARY'S OWN version line (the
    producer, crux_inference_dogfood.sh, records only `apr.version_line`; harness.sha is the harness
    checkout, never the binary). Every cell's engines.apr.version must name the same binary, since a
    receipt mixing binaries binds to none. The short sha is resolved in this checkout and must be
    unambiguous. A dirty or unparseable line -> None, so the receipt fails closed (#3957 F2; aprender-3a
    2026-09-23: the real receipt's only binding was the version line, and the judge read a field no
    producer writes)."""
    a = receipt.get("apr")
    sha = (a.get("sha") if isinstance(a, dict) else None) or receipt.get("apr_sha")
    if sha is not None:
        return sha if isinstance(sha, str) else None
    line = a.get("version_line") if isinstance(a, dict) else None
    m = APR_VERSION_SHA.match(line.strip()) if isinstance(line, str) else None
    if not m:
        return None
    for c in receipt.get("cells") or []:
        v = ((c.get("engines") or {}).get("apr") or {}).get("version") if isinstance(c, dict) else None
        if v is not None and (not isinstance(v, str) or v.strip() != line.strip()):
            return None
    return (resolve or _git_resolve)(m.group(1))


def load_crux(crux_dir, cut_sha, equiv, out):
    """-> ({(sha, host, lane, crux_verb): [verdict, ...]}, failed)."""
    index, failed = {}, False
    files = sorted(f for f in glob.glob(os.path.join(crux_dir or "", "*.json"))
                   if not os.path.basename(f).startswith("prompt-certification")) if crux_dir else []
    if not files:
        out(f"FAIL  no CRUX receipt under {crux_dir!r} -- no verb of any model has been judged against an "
            f"outside engine, so every cell below is unproven (#3957 F4)")
        return index, True
    for f in files:
        # The prompt certification and its inventory live BESIDE the receipts (evidence/crux/<v>/, #3962);
        # they are inputs to the coverage rule, not receipts. Every OTHER json here must be a receipt.
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
        if not (asha and HEX40.fullmatch(asha)) or (asha != cut_sha and asha not in equiv):
            out(f"FAIL  CRUX receipt {os.path.basename(f)} is bound to apr sha {asha!r}, not the cut {cut_sha[:12]} -- "
                f"its cells are not evidence for this build (#3957 F2)")
            failed = True
            continue
        # #4004: a GREEDY-ONLY receipt (crux_inference_dogfood --greedy-only) carries raw greedy rows for the
        # F9 judge and NO cells, so its collect verdict is DECLINE ("no cell was measured"). It vouches for
        # no cell and is not a lane verdict: skipped here, read by model_ladder_redmodel only. It must
        # still be bound to the cut (above). One that carries cells claims both roles and is refused.
        if R.get("greedy_only") is True:
            if R.get("cells"):
                out(f"FAIL  CRUX receipt {os.path.basename(f)} is marked greedy_only and carries {len(R['cells'])} cell(s) -- "
                    f"a receipt is either a lane verdict or greedy evidence, never both (#4004)")
                failed = True
            else:
                out(f"note  CRUX receipt {os.path.basename(f)} is greedy-only: no cells, evidence for the F9 judge only (#4004)")
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
            index.setdefault(key, []).append((c.get("verdict"), k.get("thinking")))
    return index, failed


def crux_cell(index, sha, host, backend, verb, red_model=None):
    """-> (proven, reason) for one (model sha, host, backend, verb).

    `red_model` (#3957 F9): the file carries a RED-MODEL verdict re-proven on this sweep, which
    covers the thinking-ON axis only. Its thinking-ON verdicts are set aside, and the cell is
    proven by the thinking-OFF verdicts alone, which must exist and be GREEN."""
    both = index.get((sha, host, LANE.get(backend, backend), VERB_TO_CRUX.get(verb, verb))) or []
    axis = set(red_model or ())
    got = [v for v, t in both if not (red_model and t in axis)]
    if red_model and not got and axis >= {"on", "off"}:
        return True, f"RED-MODEL on every thinking axis ({len(both)} cell(s) set aside)"
    if red_model and not got:
        return False, (f"RED-MODEL covers thinking {sorted(axis)} only, and no verdict on the other axis proves the rest "
                       f"of this cell (#3957 F9)")
    if not got:
        return False, "no CRUX verdict: no outside engine has judged this cell, so it is not proven"
    bad = [v for v in got if v != "GREEN"]
    if bad:
        counts = {v: got.count(v) for v in sorted(set(got))}
        return False, "CRUX: " + ", ".join(f"{n} {v}" for v, n in counts.items())
    if red_model:
        return True, (f"CRUX GREEN off the RED-MODEL axis ({len(got)} cell(s)); thinking {sorted(axis)} is RED-MODEL "
                      f"({len(both) - len(got)} cell(s) set aside)")
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


def load_certified(cert_p, out):
    """#3710 ruling 1: the models CRUX must cover are the CERTIFIED ones -- the keys of the prompt
    certification's per-model admissions (admitted_by_sha / admitted_by_sha_thinking), read from the
    receipt, never listed here. -> (set of sha256, failed). An absent or unreadable receipt is RED, and
    returns None: every held model stays CRUX-required, so a missing receipt can never RELAX the gate."""
    try:
        with open(cert_p, encoding="utf-8") as fh:
            C = json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        out(f"FAIL  no prompt-certification receipt at {cert_p!r} ({exc}) -- which models CRUX must cover is unknown, "
            f"so every held model stays CRUX-required (#3710 ruling 1)")
        return None, True
    keys = set()
    for field in ("admitted_by_sha", "admitted_by_sha_thinking"):
        v = C.get(field)
        if isinstance(v, dict):
            keys |= {k for k in v if HEX64.fullmatch(str(k))}
    if not keys:
        out(f"FAIL  prompt-certification {cert_p!r} certifies no model -- CRUX coverage cannot be scoped (#3710 ruling 1)")
        return None, True
    return keys, False


def judge(L, good, crux_dir, cut, equiv, out, red=None, cert_p=None):
    """Print one line per cell and a per-format summary. -> True when any cell is not proven.

    `red` (#3957 F9/F10): {(host, file): "RED-MODEL:<axis>" | "RED-UNSUPPORTED"}, holding ONLY the verdicts
    model_ladder_redmodel re-proved on this sweep. A RED-UNSUPPORTED file's cuda cells print
    RED-UNSUPPORTED and do not block; a RED-MODEL file's cells are proven on thinking OFF alone."""
    red = red or {}
    verbs = list((L.get("cells") or {}).get("verbs") or ["run", "chat", "serve", "code"])
    inv_backends = list((L.get("inventory") or {}).get("backends") or [])
    rung_by_file = {r.get("gguf"): r for r in L.get("rungs") or []}
    certified, failed = (load_certified(cert_p, out) if cert_p else (None, False))
    held = set()
    for R in good.values():
        inv = {i.get("file"): i.get("sha256") for i in R.get("inventory") or [] if isinstance(i, dict)}
        for x in R.get("rungs") or []:
            if x.get("present"):
                held.add(x.get("sha256") or inv.get(x.get("file")) or (rung_by_file.get(x.get("file")) or {}).get("sha256"))
    need = certified is None or bool(held & certified)
    if need or (crux_dir and glob.glob(os.path.join(crux_dir, "*.json"))):
        index, lfail = load_crux(crux_dir, cut, equiv, out)
        failed = failed or (lfail and need)
    else:
        index = {}
        out("note  no CRUX receipt needed: no held model is CRUX-certified; every model is proven by the ladder (#3710 ruling 1)")
    for model_sha in sorted(certified or ()):
        if model_sha not in held:
            failed = True
            out(f"FAIL  certified model {model_sha[:12]} is held by no required host -- CRUX must prove every certified model (#3710 ruling 1)")
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
                    t = tally.setdefault(fmt, [0, 0, 0, 0])
                    t[0] += 1
                    named = red.get((host, f))
                    if named == "RED-UNSUPPORTED" and b in ("cuda", "gpu"):
                        t[2] += 1
                        out(f"RED-UNSUPPORTED cell {label} -- apr refused this architecture by name on this sweep; counted RED, never green (#3957 F10)")
                        continue
                    if not (isinstance(sha, str) and HEX64.fullmatch(sha)):
                        ok, why = False, "the row has no 64-hex sha256, so no oracle can be joined to it"
                    elif fmt == "apr":
                        per_b = [w for w in chain_why if not w.startswith(tuple(f"{o}:" for o in backends if o != b))]
                        if src_sha is None:
                            ok, why = False, per_b[0]
                        elif certified is not None and src_sha not in certified:
                            # ruling 1: the chain's tensor + runtime links still hold; the source itself is
                            # proven by the ladder, not by CRUX, since it is not certified
                            ok, why = (not per_b), ("; ".join(per_b) if per_b else
                                       f"chain to source {src_sha[:12]} (source not CRUX-certified; proven by the ladder, #3710 ruling 1)")
                        else:
                            s_ok, s_why = crux_cell(index, src_sha, host, b, v)
                            reasons = per_b + ([] if s_ok else [f"source {src_sha[:12]} is not proven: {s_why} (#3957 F8)"])
                            ok, why = (not reasons), ("; ".join(reasons) if reasons else f"chain to source {src_sha[:12]}: {s_why}")
                    elif certified is not None and sha not in certified:
                        # #3710 ruling 1: CRUX GREEN is required only for the certified models; this one is
                        # proven by the ladder's golden oracle (why_of above), which still refuses any red row.
                        t[3] += 1
                        out(f"note  cell {label} -- not CRUX-certified: proven by the ladder's golden oracle, not by CRUX (#3710 ruling 1)")
                        continue
                    else:
                        axis = named.split(":", 1)[1].split(",") if named and named.startswith("RED-MODEL") else None
                        ok, why = crux_cell(index, sha, host, b, v, red_model=axis)
                    if ok and named and named.startswith("RED-MODEL"):
                        # proven on thinking OFF, RED on thinking ON: a RED cell, never counted proven (#3957 F9)
                        t[2] += 1
                        out(f"RED-MODEL cell {label} -- {why}")
                    elif ok:
                        t[1] += 1
                        out(f"ok    cell {label} -- {why}")
                    else:
                        failed = True
                        out(f"FAIL  cell {label} -- {why}")
    out("cells by format: " + ", ".join(f"{k} {v[1]}/{v[0]} proven" + (f", {v[2]} named RED (F9/F10)" if v[2] else "")
                                        + (f", {v[3]} ladder-only (not CRUX-certified)" if v[3] else "")
                                        for k, v in sorted(tally.items()))
        if tally else "FAIL  no cell was owed -- a release that measured no cell proved nothing")
    return failed or not tally
