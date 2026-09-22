"""The cells half of scripts/check_model_ladder.sh (#3712 row B).

Operator 2026-09-21: "lets ensure this release has MORE than enough tokens for our needs (overdo it),
and our verbs are dogfooded, i.e chat with and without thinking, ditto run, ditto code, etc". A cell is
(model, host, verb, thinking, context rung). This module decides which cells a host OWES and judges the
receipt's `cells[]` against that set. Nothing here trusts a verdict string on its own:

  * the owed set is derived from the MEASURED inventory (arch, context_length, thinking_modes and the
    memory terms), never from a list the producer chose;
  * thinking_modes must agree with the template evidence the receipt carries (#3723's derivation:
    `enable_thinking` -> both modes; a generation prompt that always opens <think> -> ON only; else OFF);
  * a cell fits a host iff weights + kv_bytes_per_token * tokens + workspace <= gpu_mem_total_bytes
    (operator decision "a", #3710). A fitting cell must PASS; a non-fitting cell must be an honest
    pre-load `refused` whose required_bytes exceeds the host's TOTAL (a co-tenant refusal is RED); an
    unmeasured term keeps the cell owed in full;
  * every owed (model, verb, thinking, rung) must PASS on at least one required host;
  * a missing row, or two rows for one cell, is a violation naming the cell.

The rung ids and token counts are NOT declared here or in the ladder contract: they live in ONE file,
evidence/release/context-rungs.json (schema apr-release-context-rungs/v1, #3715), which pv reads too. A
second declaration is how two readers drift. consumer-max is derived there: the max over
consumers[].max_prompt_tokens, and a consumer without a number is a violation, never a smaller max.

judge(ladder, receipts, rungs_doc, out) -> 0 green | 1 red. `receipts` maps host id -> receipt;
`rungs_doc` is the parsed context-rungs file, or None when it is missing or unreadable (a named FAIL).
"""
RUNGS_SCHEMA = "apr-release-context-rungs/v1"
MAX_LINES_PER_MODEL = 12


def _tokens(rung, ctx, consumer_max):
    if rung.get("tokens") is not None:
        return int(rung["tokens"])
    if rung["id"] == "consumer-max":
        return consumer_max
    if rung["id"] == "declared":
        return ctx
    return None


def expected_modes(markers, gen_opens_think):
    """#3723's derivation, from the evidence the receipt carries."""
    if markers is None:
        return None
    if "enable_thinking" in markers:
        return {"on", "off"}
    if gen_opens_think:
        return {"on"}
    return {"off"}


def owes_long(item, long_for):
    arch = item.get("arch")
    if arch is None:
        return True, "no general.architecture measured: owes the long rungs in full"
    if arch in (long_for.get("families") or []):
        return True, f"family {arch}"
    rep = (long_for.get("representatives") or {}).get(arch)
    if rep == item.get("file"):
        return True, f"representative for arch {arch}"
    return False, f"arch {arch} represented by {rep}" if rep else f"arch {arch} has no representative"


def fits(item, total, tokens):
    terms = [item.get("weights_bytes"), item.get("kv_bytes_per_token"), item.get("workspace_bytes"), total]
    if any(t is None for t in terms):
        return True, None  # an unmeasured term keeps the cell owed in full
    need = int(item["weights_bytes"]) + int(item["kv_bytes_per_token"]) * tokens + int(item["workspace_bytes"])
    return need <= int(total), need


def load_rungs(doc, out):
    """-> (rungs, consumer_max, rc). A file that sizes nothing is a named FAIL, never an empty universe."""
    if doc is None:
        out("FAIL  evidence/release/context-rungs.json is missing or unreadable -- no context rung is sized, so no cell can be judged (#3715)")
        return [], None, 1
    if doc.get("schema") != RUNGS_SCHEMA:
        out(f"FAIL  context rungs schema {doc.get('schema')!r} is not {RUNGS_SCHEMA}")
        return [], None, 1
    rc = 0
    rungs = [r for r in doc.get("rungs") or [] if r.get("id")]
    if not rungs:
        out("FAIL  the context rungs file names no rung -- it owes nothing, which is not a pass")
        rc = 1
    maxima = []
    for c in doc.get("consumers") or []:
        n = c.get("max_prompt_tokens")
        if not isinstance(n, int) or n <= 0:
            out(f"FAIL  consumer {c.get('name')!r} has no max_prompt_tokens -- consumer-max cannot be derived without it")
            rc = 1
        else:
            maxima.append(n)
    consumer_max = max(maxima) if maxima and rc == 0 else None
    return rungs, consumer_max, rc


def judge(L, receipts, rungs_doc, out, rungs_main=None):
    C = L.get("cells")
    if not C:
        return 0
    required = {h["id"] for h in L.get("hosts") or [] if h.get("required")}
    receipts = {hid: R for hid, R in receipts.items() if hid in required}  # a non-required host proves nothing here
    verbs = list(C.get("verbs") or [])
    long_for = C.get("long_rungs_for") or {}
    rungs, consumer_max, rc = load_rungs(rungs_doc, out)
    if rungs_main is not None:  # the context-rungs file at origin/main is a floor, like the ladder itself
        for what, key in (("rung ids", "rungs"), ("consumers", "consumers")):
            name = "id" if key == "rungs" else "name"
            gone = {x.get(name) for x in rungs_main.get(key) or []} - {x.get(name) for x in (rungs_doc or {}).get(key) or []}
            if gone:
                out(f"FAIL  context rungs {what} DROPPED vs origin/main: {sorted(map(str, gone))} -- the token bar may rise, never fall"); rc = 1
    long_ids = {r["id"] for r in rungs if r.get("long")}
    if not verbs:
        out("FAIL  the ladder's cells block names no verbs -- it owes nothing, which is not a pass")
        return 1
    passed_anywhere = {}  # (sha, verb, mode, rung) -> bool
    failed_somewhere = set()  # keys already reported by a per-host FAIL
    for hid, R in receipts.items():
        total = R.get("gpu_mem_total_bytes")
        rows = {}
        for c in R.get("cells") or []:
            k = (c.get("file"), c.get("verb"), c.get("thinking"), c.get("context"))
            rows.setdefault(k, []).append(c)
        if not R.get("cells"):
            out(f"FAIL  {hid:7} receipt carries no cells -- no verb was dogfooded at any context (#3712)")
            rc = 1
        owed = passed = refused = 0
        host_rc = rc  # an unsized rung set is red for every host, never an 'ok 0 owed'
        for item in R.get("inventory") or []:
            f, sha = item.get("file"), item.get("sha256")
            fails = []
            ctx = item.get("context_length")
            if ctx is None:
                fails.append(f"no declared context_length measured (apr inspect drops the arch keys, #3733) -- the declared rung cannot be sized")
            modes = set(item.get("thinking_modes") or [])
            want = expected_modes(item.get("thinking_markers"), bool(item.get("generation_opens_think")))
            if not modes:
                modes = {"on", "off"}
                fails.append("no thinking_modes measured: owes both modes")
            elif not modes <= {"on", "off"}:
                fails.append(f"thinking_modes {sorted(modes)} is not a subset of on/off")
            elif want is not None and modes != want:
                fails.append(f"thinking_modes {sorted(modes)} disagrees with its template evidence (markers {item.get('thinking_markers')}) which derives {sorted(want)}")
            if want is not None:
                modes = want  # the evidence decides what is owed, not the producer's own claim
            lo, why_long = owes_long(item, long_for)
            for rung in rungs:
                rid = rung["id"]
                if rid in long_ids and not lo:
                    continue
                tok = _tokens(rung, ctx, consumer_max)
                if tok is None:
                    fails.append(f"rung {rid} has no token count ({'consumer-max is not derived' if rid == 'consumer-max' else 'no context_length'}) -- owed, unmeasurable")
                    continue
                if ctx is not None and tok > int(ctx):
                    continue  # above the model's own declared context: not owed
                fit, need = fits(item, total, tok)
                for verb in verbs:
                    for mode in sorted(modes):
                        owed += 1
                        k = (f, verb, mode, rid)
                        got = rows.get(k, [])
                        key = (sha, verb, mode, rid)
                        passed_anywhere.setdefault(key, False)
                        label = f"{verb}/{mode}/{rid}"
                        n_before = len(fails)
                        if not got:
                            fails.append(f"{label} MISSING"); failed_somewhere.add(key); continue
                        if len(got) > 1:
                            fails.append(f"{label} has {len(got)} rows -- ambiguous"); failed_somewhere.add(key); continue
                        c = got[0]
                        v = c.get("verdict")
                        if v == "pass":
                            bad = []
                            if c.get("backend") != "cuda": bad.append(f"backend {c.get('backend')!r}")
                            if c.get("fallback") is not False: bad.append("fell back")
                            if c.get("rc") != 0: bad.append(f"rc {c.get('rc')}")
                            if int(c.get("prompt_tokens") or 0) < tok: bad.append(f"prompt_tokens {c.get('prompt_tokens')} < {tok}")
                            if int(c.get("answer_chars") or 0) <= 0: bad.append("no answer")
                            if mode == "on" and c.get("think_closed") is not True: bad.append("thinking never closed")
                            if not fit:
                                bad.append(f"the declared arithmetic says it cannot fit ({need} B > {total} B total) -- a pass here means the arithmetic that decides what is owed is wrong (#3596)")
                            if bad:
                                fails.append(f"{label} 'pass' is not a pass: " + ", ".join(bad))
                            else:
                                passed += 1; passed_anywhere[key] = True
                        elif v == "refused":
                            req = c.get("required_bytes")
                            if fit and need is not None:
                                fails.append(f"{label} REFUSED but it fits: {need} B <= {total} B total -- a co-tenant or a wrong arithmetic, not the host's capacity")
                            elif total is None or req is None or int(req) <= int(total):
                                fails.append(f"{label} REFUSED without an honest arithmetic (required {req} vs total {total})")
                            else:
                                refused += 1
                        else:
                            fails.append(f"{label} {v}: {str(c.get('reason', ''))[:80]}")
                        if len(fails) > n_before:
                            failed_somewhere.add(key)
            for i, w in enumerate(fails):
                if i == MAX_LINES_PER_MODEL:
                    out(f"FAIL  {hid:7} {f}: ... and {len(fails) - i} more"); break
                out(f"FAIL  {hid:7} {f}: {w}")
            if fails:
                rc = host_rc = 1
        out(f"{'ok  ' if host_rc == 0 else 'info'}  {hid:7} cells: {owed} owed, {passed} pass, {refused} honest refusal(s)")
    # Long rungs are expensive, so they are owed by the long-rung families plus ONE representative per other
    # arch (cop 2026-09-21). An arch with no representative, or one no required host holds, is never tested
    # long at all -- that is a hole, not a saving.
    held = {it.get("file") for R in receipts.values() for it in R.get("inventory") or []}
    archs = {it.get("arch") for R in receipts.values() for it in R.get("inventory") or [] if it.get("arch")}
    fams = set(long_for.get("families") or [])
    reps = long_for.get("representatives") or {}
    for arch in sorted(archs - fams):
        r = reps.get(arch)
        if not r:
            out(f"FAIL  arch {arch} has no long-rung representative in the contract -- no {arch} model is ever tested past the short rungs"); rc = 1
        elif r not in held:
            out(f"FAIL  the long-rung representative for arch {arch}, {r}, is held by no required host"); rc = 1
    for key, ok in sorted(passed_anywhere.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        sha, verb, mode, rid = key
        if not ok and key not in failed_somewhere:  # honest refusals everywhere: the rung passes nowhere
            out(f"FAIL  cell {verb}/{mode}/{rid} of model {str(sha)[:12]} PASSES on no required host -- every declared rung must pass somewhere (#3710 'a')")
            rc = 1
    return rc
