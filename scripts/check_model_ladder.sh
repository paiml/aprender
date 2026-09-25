#!/usr/bin/env bash
# check_model_ladder.sh — the T-2 gate over contracts/model-capability-ladder-v1.yaml.
#
# Reads one receipt per REQUIRED host (written by scripts/model_ladder.sh) for
# the version being cut and is green only when every required rung is present
# and green on every one of them, AND every model the host HOLDS is too.
#
# #3712 (operator 2026-09-21: "you must ensure all models Q4_K CUDA work; the end",
# and publishing with "most working" is a "p0 tire fire"). Three rules, no exemptions
# and no thresholds:
#   * No Q4_K rung is optional. `required: false` on a Q4_K rung is REFUSED, and so is
#     a Q4_K rung that does not claim cuda.
#   * The universe is the host's measured inventory. A receipt must be schema v2 and
#     carry a non-empty `inventory`. Every inventory model must appear in the run and be
#     green on CUDA, or it is a FAIL naming it.
#   * A skipped capability_match or golden_output, a fallback line, or a run rc != 0 is RED. It is declared in Cargo.toml
# [package.metadata.dogfood] so scripts/dogfood.sh runs it in every phase; a
# missing receipt is FAIL, not DEFER — a dev build can measure this, no
# published crate is needed.
#
# Discrimination: `--self-test` runs the case table in scripts/lib/model_ladder_cases/
# (each case is a receipt set + expected exit) and exits 1 if any case lands on
# the wrong verdict. A validator that has only ever seen valid input is
# indistinguishable from `exit 0` (#2696 lesson).
#
# Anti-shrink: the ladder as it exists at origin/main is the floor for rungs,
# hosts and backends (check_multiplatform_dogfood.sh layer 2). Growing is free.
#
# Exit: 0 green · 1 red · 2 not green, not red: decline (unreadable ladder / no required
#       host / unresolvable cut commit) or DEFER (a declared deferral was used, #3957 F1) ·
#       self-test: 0 all cases as expected, 1 otherwise.
set -uo pipefail

LADDER="contracts/model-capability-ladder-v1.yaml"
# Overridable so the floor below can be PROVEN against a planted case in a temp dir
# rather than by planting a permanently-failing case in the real table (#3887).
CASES_DIR="${MODEL_LADDER_CASES_DIR:-scripts/lib/model_ladder_cases}"
NIGHTLY_ROOT_DIR=""; EQUIV_ONLY=0; SELF_TEST=0; ONLY_CASE=""; RECEIPT_DIR=""; LADDER_MAIN_OVERRIDE=""; CUT_COMMIT=""; CRUX_DIR=""; SCOPE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) SELF_TEST=1; shift ;;
    --case) [ $# -ge 2 ] || { echo "--case needs a value" >&2; exit 2; }; ONLY_CASE="$2"; shift 2 ;;
    --receipts) [ $# -ge 2 ] || { echo "--receipts needs a value" >&2; exit 2; }; RECEIPT_DIR="$2"; shift 2 ;;
    --ladder) [ $# -ge 2 ] || { echo "--ladder needs a value" >&2; exit 2; }; LADDER="$2"; shift 2 ;;
    --ladder-main) [ $# -ge 2 ] || { echo "--ladder-main needs a value" >&2; exit 2; }; LADDER_MAIN_OVERRIDE="$2"; shift 2 ;;
    --version) [ $# -ge 2 ] || { echo "--version needs a value" >&2; exit 2; }; VERSION_OVERRIDE="$2"; shift 2 ;;
    --cut-commit) [ $# -ge 2 ] || { echo "--cut-commit needs a value" >&2; exit 2; }; CUT_COMMIT="$2"; shift 2 ;;
    --crux) [ $# -ge 2 ] || { echo "--crux needs a value" >&2; exit 2; }; CRUX_DIR="$2"; shift 2 ;;
    --equiv-only) EQUIV_ONLY=1; shift ;;   # self-test hook: print equiv_lines for --receipts/--crux/--cut-commit
    --nightly) [ $# -ge 2 ] || { echo "--nightly needs a value" >&2; exit 2; }; NIGHTLY_ROOT_DIR="$2"; shift 2 ;;
    --scope) [ $# -ge 2 ] || { echo "--scope needs a value" >&2; exit 2; }; SCOPE="$2"; shift 2 ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_model_ladder: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"   # before the cd: the mutants copy this file
# The root is derived from this file's path, never from `git rev-parse` (it dies in the CI container,
# aprender#3581) or from the caller's cwd. MODEL_LADDER_ROOT is how a mutant copy, which lives in a
# temp dir, is told the tree it judges.
cd "${MODEL_LADDER_ROOT:-$(dirname "$SELF")/..}" || exit 2

# ---------------------------------------------------------------- the judge
# judge <ladder> <ladder_at_main_or_empty> <receipt_dir> <version> <context-rungs.json> <its origin/main copy>
#       <cut commit, 40-hex> <file of apr_shas proven equal to the cut modulo evidence/>
#       <CRUX receipt dir>  → exit 0/1/2
judge() {
  python3 - "$1" "$2" "$3" "$4" "${5:-}" "${6:-}" "${7:-}" "${8:-}" "${9:-}" "${10:-}" <<'PY'
import fnmatch, json, os, sys, yaml
ladder_p, main_p, rdir, version, rungs_p, rungs_main_p, cut, equiv_p, crux_dir, cert_p = sys.argv[1:11]
cut_sha = cut  # the cut IS a 40-hex commit sha; named so, a [:12] of it is an identifier prefix (#4046)
# A mutant copy of a module, in --self-test: each directory is searched first when set.
sys.path.insert(0, "scripts/lib")
for _lib in (os.environ.get("MODEL_LADDER_CRUX_LIB"), os.environ.get("MODEL_LADDER_CELLS_LIB"),
             os.environ.get("MODEL_LADDER_REDMODEL_LIB")):
    if _lib: sys.path.insert(0, _lib)
import model_ladder_cells
import model_ladder_crux
import model_ladder_redmodel
try:
    L = yaml.safe_load(open(ladder_p))["ladder"]
except Exception as e:
    print(f"decline: ladder unreadable: {e}"); sys.exit(2)
hosts = [h for h in L.get("hosts", []) if h.get("required")]
rungs = L.get("rungs", [])
if not hosts: print("decline: ladder names no required host"); sys.exit(2)
if not rungs: print("decline: ladder has no rungs"); sys.exit(2)
rc = 0
import re
# #3957 F2: A RECEIPT IS BOUND TO THE COMMIT, NOT TO THE VERSION STRING. `version` is a
# label every build between two bumps shares: the 0.69.1 receipts were measured at
# 9b7739951 and the cut was 28c24207a, 7 commits later, and the only record of that was
# `inventory.apr_sha_drift` -- prose no code read. A receipt is evidence for the cut only
# when its `apr_sha` IS the cut, or is a commit whose tree equals the cut's outside
# evidence/ (the receipts-commit rule of release-readiness-v1 `fresh`, R7), which the
# caller proves with git and passes in. A short or absent sha binds to nothing.
if not re.fullmatch(r"[0-9a-f]{40}", cut or ""):
    print(f"decline: the cut commit {cut!r} is not a full 40-hex sha — a receipt cannot be bound to it (#3957 F2)"); sys.exit(2)
equiv, equiv_proof = set(), {}
if equiv_p and os.path.exists(equiv_p):
    # one line per sha that binds the cut: "<sha>" or "<sha>\t<proof>" (#3710 ruling 3 records the proof)
    for ln in open(equiv_p):
        parts = ln.rstrip("\n").split("\t", 1)
        if re.fullmatch(r"[0-9a-f]{40}", parts[0].strip()):
            equiv.add(parts[0].strip())
            if len(parts) > 1:
                equiv_proof[parts[0].strip()] = parts[1].strip()
if "apr_sha_drift" in (L.get("inventory") or {}) or "apr_sha_drift" in L:
    print("FAIL  the ladder carries `apr_sha_drift` — a prose declaration no code reads cannot excuse a receipt measured at another commit; "
          "the gate binds receipts by apr_sha. Delete the key and re-measure at the cut (#3957 F2)"); rc = 1
def is_q4k(r):  # a Q4_K model, by its file or its id (#3712)
    return bool(re.search(r"q4_?k", f"{r.get('gguf', '')} {r.get('id', '')}", re.I))
# #3872: DO NOT CAP. A length cap is structurally wrong for this field: the classification
# ("this is a refusal, not a fallback"), the scope caveat and the diagnostic instruction are
# whatever the author added LAST, so any cap removes exactly the part a cap looks harmless
# for keeping. Measured: the first draft of this fix capped at 200 and case
# red-reason-survives-truncation caught it, because the classification starts at char 256.
# A red line prints its whole reason; a message too long to read is a defect in the message.
def _disp(msg):
    return str(msg or "")

def _semver(v):
    import re as _re
    m = _re.match(r"(\d+)\.(\d+)\.(\d+)", str(v or ""))
    return tuple(int(g) for g in m.groups()) if m else None

def timing_required(v):  # #4051: the contract's `timing_required_from` (0.70.0) on -- per-call stamps
    frm = _semver((L.get("timing") or {}).get("required_from"))
    cur = _semver(v)
    return frm is not None and (cur is None or cur >= frm)

def timing_gaps(x):  # -> every reason the row's #4051 stamps do not account for its apr calls
    st = x.get("timing")
    if not isinstance(st, list):
        return ["the row records no `timing` -- no apr call it reports was stamped"]
    out = []
    need = [("qa", None), ("inspect", None), ("code", None)] + [(v, b) for b in sorted(x.get("backends") or {}) for v in ("run", "chat", "serve")]
    for verb, be in need:
        got = [s for s in st if isinstance(s, dict) and s.get("verb") == verb and (s.get("backend") or None) == be]
        if not got:
            out.append("no stamp for `%s`%s" % (verb, " on %s" % be if be else ""))
    for s in st:
        if not isinstance(s, dict):
            out.append("a stamp is not an object: %r" % (s,)); continue
        t0, t1, lw, lk = s.get("t_start"), s.get("t_end"), s.get("lock_wait_s"), s.get("lock")
        tag = "%s%s" % (s.get("verb"), " on %s" % s.get("backend") if s.get("backend") else "")
        if not all(isinstance(t, (int, float)) for t in (t0, t1)):
            out.append("`%s` stamp lacks t_start/t_end: %r" % (tag, [t0, t1]))
        elif t1 < t0:
            out.append("`%s` stamp has t_end < t_start (%.3f < %.3f)" % (tag, t1, t0))
        if lk not in ("gpu", "none") or not isinstance(lw, (int, float)) or lw < 0 or (lk == "none" and lw != 0):
            out.append("`%s` stamp has no valid lock wait (lock %r, lock_wait_s %r)" % (tag, lk, lw))
    return out

def why_of(x, backends):  # every reason a measured row is not green on the claimed backends
    why = []
    cm, go = x.get("capability_match") or {}, x.get("golden_output") or {}
    claims_gpu = bool({"cuda", "gpu"} & set(backends))
    cap_ok = (cm.get("passed") and not cm.get("skipped")) or (cm.get("skipped") and not claims_gpu)
    if not cap_ok: why.append("capability_match " + ("SKIPPED" if cm.get("skipped") else "FAIL") + ": " + _disp(cm.get("message", "")))
    if not (go.get("passed") and not go.get("skipped")): why.append("golden_output " + ("SKIPPED" if go.get("skipped") else "FAIL") + ": " + _disp(go.get("message", "")))
    # #3898: THE RECEIPT RECORDED THE FAILURE AND NOTHING READ IT.
    #
    # The producer runs `apr qa`, captures its exit code in `qa_rc`, records WHICH gates
    # failed in `gates_failed`, and runs the #3830-class check that the rc is explained
    # (`gates_account_for_rc`). Three correctly populated fields describing a real
    # validation failure — and this function consulted none of them, because a row's
    # `green` is `capability_match AND golden_output` only.
    #
    # Measured instance, evidence/dogfood/models/0.69.1/lambda.json @ ab4ba54ec:
    #     inv:Qwen3.5-4B-UD-Q4_K_XL.gguf   green=True  qa_rc=5
    #                                      gates_failed=['tensor_contract']
    #                                      gates_account_for_rc=True
    # and the gate printed `ok lambda inv:Qwen3.5-4B-UD-Q4_K_XL.gguf green on cuda`.
    # That is the Q4_K UD model — inside Rule A's scope, and the "no model left behind"
    # model most likely to be cited as proof.
    #
    # KEYED ON qa_rc, NEVER ON A NAMED GATE. Naming `tensor_contract` reproduces the
    # defect at n+1 the moment a different gate matters, which is exactly why
    # `gates_account_for_rc` is an invariant over WHICHEVER gate goes missing rather than
    # over a listed one. Ten gates are in scope and a future regression in any of them
    # could not fail the release: tensor_contract, metadata_plausibility,
    # performance_regression, throughput, format_parity, ptx_parity, gpu_state_isolation,
    # classifier_head, ollama_parity, gpu_speedup.
    #
    # An accepted refusal is DECLARED, not ignored: a row matching an `inventory.deferred`
    # reason is printed DEFERRED and never counted green (#3846/#3880). A gate that cannot
    # tell an accepted refusal from a new regression is not reading either.
    qa_rc = x.get("qa_rc")
    # #3957 F3: `not in (0, None)` let a row with NO qa_rc through as if apr qa had exited 0.
    # An absent field is never the passing value (release-readiness-v1 cell_pass precondition);
    # the producer writes qa_rc on every row it builds, so absence means the row was not built
    # by it, or was built from nothing.
    if "qa_rc" not in x or not isinstance(qa_rc, int) or isinstance(qa_rc, bool):
        why.append(f"`qa_rc` is MISSING or not an integer ({qa_rc!r}) — a row that cannot say how apr qa exited has not shown that it passed (#3957 F3)")
    elif qa_rc != 0:
        gf = x.get("gates_failed") or []
        named = ", ".join(str(g) for g in gf) if gf else "NO GATE NAMED"
        why.append(f"apr qa exited {qa_rc} — gates_failed: {named} (#3898)")
    # `gates_account_for_rc: false` is worse than a named failure: the recorded gates do
    # not explain the rc, so MORE failed than the row accounts for. Refused on its own,
    # independently of the rc, because a row can be rc=0 and still not add up.
    if x.get("gates_account_for_rc") is False:
        why.append("`gates_account_for_rc` is FALSE — the recorded gates do not explain `qa_rc`, so more failed than this row accounts for (#3898)")
    be = x.get("backends") or {}
    for b in backends:
        v = be.get(b)
        if v is None: why.append(f"{b}: not measured")
        elif v.get("fallback"): why.append(f"{b}: FELL BACK — the claimed backend did not run")
        elif v.get("escaped_special"): why.append(f"{b}: the formatted prompt carries a zero-width-escaped special token — templated twice (#3743)")
        elif not v.get("ran"): why.append(f"{b}: did not run (rc={v.get('rc')})")
        # #3828: the release matrix claims verbs {run, chat, serve, code}. The producer used
        # to measure `run` alone, so three columns were computed over nothing and a live
        # /api/chat defect passed every gate we own. A verb ABSENT from a receipt is refused
        # here by name: absence must never read as conformance, which is this epic's whole
        # premise (#3712/#3715) applied to the instrument rather than to the models.
        vb = (v or {}).get("verbs")
        if vb is None:
            why.append(f"{b}: receipt records no `verbs` object — the release matrix claims {{run, chat, serve, code}} and this rung measured only `run` (#3828)")
        else:
            for verb in ("run", "chat", "code"):
                r = vb.get(verb)
                if r is None: why.append(f"{b}: verb `{verb}` is MISSING from the receipt — not measured is not passed (#3828)")
                elif not r.get("ran"): why.append(f"{b}: verb `{verb}` did not run (rc={r.get('rc')})")
                # #3957 F4a: the producer's verb_ok reads `output_bad` and this judge did not, so a
                # verb that exited 0 while printing gibberish was ok HERE -- the #3897 asymmetry in
                # the other direction. The judge is the decision surface; it reads what the
                # producer read.
                if r is not None and r.get("output_bad"):
                    why.append(f"{b}: verb `{verb}` produced bad output: {_disp(r.get('output_bad'))} (#3957 F4a)")
            sv = vb.get("serve")
            if sv is None:
                why.append(f"{b}: verb `serve` is MISSING from the receipt — no rung has ever asked apr serve to load a model (#3571, #3828)")
            elif not sv.get("probed"):
                why.append(f"{b}: verb `serve` was NOT PROBED: {sv.get('why', 'no reason recorded')} (#3828)")
            else:
                rts = sv.get("routes") or {}
                if not rts:
                    why.append(f"{b}: verb `serve` probed but recorded NO routes — an empty route set is the vacuous pass this gate exists to refuse (#3828)")
                # The ollama-compat wire has its own translation layer and has diverged from
                # the OpenAI-compat one twice independently (#3825, and the #3571 hybrid
                # defect), so its coverage is never inherited from a representative route.
                if not any(k.startswith("/api/chat") for k in rts):
                    why.append(f"{b}: verb `serve` recorded no /api/chat probe — the ollama-compat route is the one real Ollama harnesses hit and it cannot inherit /v1 coverage (alfredodeza, #3715; #3828)")
                for rk, rv in sorted(rts.items()):
                    if not rv.get("ok"): why.append(f"{b}: serve route {rk} returned http {rv.get('http')} (#3828)")
                    if rv.get("output_bad"): why.append(f"{b}: serve route {rk} produced bad output: {_disp(rv.get('output_bad'))} (#3957 F4a)")
            # #3838: the teardown is part of the measurement, and the JUDGE has to read it.
            # The producer marks its own cell red on a failed teardown, but that enforces the
            # rule only in the code that happened to observe the failure -- a receipt written
            # by any other path would carry the field with nothing reading it, and the decision
            # surface for a receipt is this judge. `failed` means the server outlived its
            # launcher and could not be attributed to the run's own tree, so the route results
            # above were read off a process that was still running when they were taken.
            # A receipt with NO teardown key predates #3838 and is refused by name rather than
            # tolerated: absence scored as conformance is the shape this whole gate exists for.
            if sv is not None:
                td = sv.get("teardown")
                if td is None:
                    why.append(f"{b}: verb `serve` records no `teardown` — a probe that does not say whether its server died is not a completed measurement (#3838)")
                elif td == "failed":
                    why.append(f"{b}: verb `serve` teardown FAILED — the server outlived its launcher and could not be proven to belong to this tree, so the route results were taken from a process still running (#3838)")
                elif td == "undetermined":
                    why.append(f"{b}: verb `serve` teardown UNDETERMINED — the process tree could not be resolved, so nothing proves the server died; a cell cannot claim a clean teardown from an observation that does not discriminate (#3943)")
                elif td not in ("clean", "escalated"):
                    why.append(f"{b}: verb `serve` teardown is {td!r}, which is not one of clean/escalated/failed/undetermined (#3838, #3943)")
    return why
# #3712: no Q4_K rung is optional, and every one claims cuda. The key is refused, not tolerated.
for r in rungs:
    if is_q4k(r) and r.get("required") is not True:
        print(f"FAIL  rung {r['id']} is a Q4_K rung with required: {r.get('required')!r} — no Q4_K model is optional; every one must be green on CUDA (#3712)"); rc = 1
    if is_q4k(r) and "cuda" not in (r.get("backends") or []):
        print(f"FAIL  rung {r['id']} is a Q4_K rung that does not claim cuda — every Q4_K model must be green on CUDA (#3712)"); rc = 1
inv_backends = list((L.get("inventory") or {}).get("backends") or [])
# #3957 F9/F10: the two NAMED RED verdicts, RED-MODEL and RED-UNSUPPORTED. Each is a claim about
# CAUSE and is admitted only when THIS sweep re-proves it; otherwise the row is plain FAIL. The
# rules live in scripts/lib/model_ladder_redmodel.py, and the case table proves each one RED first.
RV = model_ladder_redmodel.RedVerdicts(L, print)
if RV.failed: rc = 1
RV.load_crux(crux_dir, cut, equiv)
held_sha = {}  # host -> {file: sha256}, the host's measured inventory (for the F9 control sibling)
def sha_of(host, f): return (held_sha.get(host) or {}).get(f)
def named_red(host, f, x, why, backends):
    """-> True when a key covers the row (the line is printed and rc is set); None otherwise."""
    gpu = {"cuda", "gpu"}
    others = [b for b in backends if b not in gpu]
    def residual(y):  # RED-MODEL: the ladder's own verdict over the row with the defect neutralised
        return why_of(y, backends)
    v = RV.classify(host, f, x, why, residual, sha_of)
    if v is None: return None
    blocking, line = v
    # RED-UNSUPPORTED excuses the CUDA backend only: every other claimed backend is judged as usual.
    if not blocking and RV.proven.get((host, f)) == "RED-UNSUPPORTED" and others:
        rest = [w for w in why_of(x, others) if any(w.startswith(f"{o}:") for o in others)]
        if rest:
            RV.proven.pop((host, f), None)
            blocking, line = True, f"FAIL  {host:7} {f:22} RED-UNSUPPORTED excuses cuda only, and " + "; ".join(rest)
    print(line)
    global rc
    if blocking: rc = 1
    return True
# #3846: the DECLARED inventory deferrals (glob -> reason). Empty when absent, so a
# contract without the key defers nothing and every failing row is refused as before.
inv_deferred = dict((L.get("inventory") or {}).get("deferred") or {})
# #3880: a deferral that nothing retires is an amnesty. Two counters per key, across ALL
# hosts: how many held files the key MATCHED at all, and how many rows it actually
# DEFERRED. They separate the two ways a key can go unused, and only one is a defect:
#   matched 0  -> not applicable here (no such file on any host). NOT stale: the "*A3B*"
#                 key is legitimately unused on a host holding no MoE model.
#   matched >0, deferred 0 -> every file it covers is GREEN. The capability it excuses
#                 arrived and the key outlived it. STALE, and refused below.
# Self-retiring: nothing to date, no issue to look up, no network. Once support lands the
# rows go green, the key stops deferring, and the next run demands its deletion.
defer_matched = dict((k, 0) for k in inv_deferred)
defer_used    = dict((k, 0) for k in inv_deferred)
if not (L.get("inventory") or {}).get("patterns") or "cuda" not in inv_backends:
    print("FAIL  the ladder declares no inventory (patterns + backends incl. cuda) — the universe cannot be the host's measured Q4_K models (#3712)"); rc = 1
# anti-shrink vs origin/main
if main_p and os.path.exists(main_p):
    try:
        M = yaml.safe_load(open(main_p))["ladder"]
        mr = {r["id"]: set(r.get("backends", [])) for r in M.get("rungs", [])}
        hr = {r["id"]: set(r.get("backends", [])) for r in rungs}
        for rid, mb in mr.items():
            if rid not in hr: print(f"FAIL  rung DROPPED vs origin/main: {rid} — the ladder may grow, never shrink"); rc = 1
            elif not mb <= hr[rid]: print(f"FAIL  backends DROPPED on {rid} vs origin/main: {sorted(mb - hr[rid])}"); rc = 1
        # per-rung hosts: is a floor too. A rung listed for every host at origin/main (no hosts: key) may not
        # be narrowed to one host — that is dropping a required (rung, host) pair by another spelling.
        def hostset(r, allh): return set(r.get("hosts") or allh)
        allh = {h["id"] for h in L.get("hosts", []) if h.get("required")}
        mrh = {r["id"]: hostset(r, allh) for r in M.get("rungs", []) if r.get("required")}
        for r in rungs:
            if r.get("required") and r["id"] in mrh and not mrh[r["id"]] <= hostset(r, allh):
                print(f"FAIL  hosts DROPPED on {r['id']} vs origin/main: {sorted(mrh[r['id']] - hostset(r, allh))} — a rung may gain hosts, never lose one"); rc = 1
        mc, hc = M.get("cells") or {}, L.get("cells") or {}
        if mc and not hc:
            print("FAIL  the cells block DROPPED vs origin/main -- verbs x thinking x context would owe nothing"); rc = 1
        elif mc:
            # #3828: a removal may be DECLARED, never silent. #3817 deliberately withdrew
            # qwen3moe (this build has no CUDA forward for the architecture at all), and the
            # floor refused it -- two correct rules in tension. A withdrawal is admitted only
            # when `cells.withdrawn.<what>` names the key AND carries a non-empty reason, and
            # it is printed every run so it cannot decay into a silent shrink. An UNDECLARED
            # removal still fails, which is the floor's whole point.
            wd = hc.get("withdrawn") or {}
            def declared(what_key, key):
                d = (wd.get(what_key) or {})
                why = d.get(key) if isinstance(d, dict) else None
                return why if isinstance(why, str) and why.strip() else None
            for what, wkey, a, b in (("verbs", "verbs", mc.get("verbs"), hc.get("verbs")),
                               ("long-rung families", "families", (mc.get("long_rungs_for") or {}).get("families"), (hc.get("long_rungs_for") or {}).get("families")),
                               ("long-rung representatives", "representatives", list((mc.get("long_rungs_for") or {}).get("representatives") or {}), list((hc.get("long_rungs_for") or {}).get("representatives") or {}))):
                gone = set(a or []) - set(b or [])
                for g in sorted(gone):
                    why = declared(wkey, g)
                    if why:
                        print(f"note  cells {what} WITHDRAWN (declared): {g} — {why}")
                    else:
                        print(f"FAIL  cells {what} DROPPED vs origin/main: {g} — a removal must be declared in cells.withdrawn.{wkey} with a reason, or it is a silent shrink"); rc = 1
        mh = {h["id"] for h in M.get("hosts", []) if h.get("required")}
        for hid in mh - {h["id"] for h in hosts}:
            print(f"FAIL  required host DROPPED vs origin/main: {hid}"); rc = 1
        print(f"ok    ladder covers every rung/host/backend at origin/main ({len(mr)} rungs, {len(mh)} hosts)")
    except Exception as e:
        print(f"FAIL  ladder at origin/main unreadable: {e}"); rc = 1
else:
    print("!     BOOTSTRAP: no ladder at origin/main yet; the anti-shrink floor arms when this lands")
# #3907 (operator ruling via the release cop aprender-cf, 2026-09-23): a KNOWN RED that ships with its
# ticket. `ladder.known_red` pins ONE model (rung id or file, AND its sha256) to ONE failure clause (a
# regex over the row's reason) and ONE ticket. A row is covered only when EVERY reason it fails is
# matched by an entry pinned to that same model: the same model failing another clause, or another
# model failing this clause, is still RED. An entry that covers nothing on this sweep is refused as
# STALE, the #3880 no-amnesty rule, so a known red cannot outlive its defect.
KR = []
for i, e in enumerate(L.get("known_red") or []):
    e = e if isinstance(e, dict) else {}
    missing = [k for k in ("sha256", "clause", "ticket", "ruling") if not str(e.get(k) or "").strip()]
    if not (e.get("rung") or e.get("file")):
        missing.append("rung|file")
    if missing:
        print(f"FAIL  known_red entry {i} has no {', '.join(missing)} -- a known red names its model, its sha256, its clause, its ticket and its ruling"); rc = 1
        continue
    # `clause` is one regex or a LIST: a single defect can surface as more than one reason on a row
    # (#3907's golden failure also makes `apr qa` exit non-zero naming that gate, #3898). Each regex is
    # pinned to this model; a reason none of them matches still makes the row RED.
    try:
        e["_re"] = [re.compile(c) for c in (e["clause"] if isinstance(e["clause"], list) else [e["clause"]])]
    except re.error as exc:
        print(f"FAIL  known_red entry {i} ({e['ticket']}) clause is not a valid regex: {exc}"); rc = 1
        continue
    e["_used"] = 0
    KR.append(e)

def known_red(row_key, f, sha, why):
    """-> the covering entries when EVERY reason is matched by an entry pinned to this model, else None."""
    if not why or not KR:
        return None
    hits = []
    for w in why:
        e = next((e for e in KR if (e.get("rung") == row_key or e.get("file") == f) and e["sha256"] == sha
                  and any(r_.search(w) for r_ in e["_re"])), None)
        if e is None:
            return None
        hits.append(e)
    return hits

good = {}  # host id -> a receipt that passed the host-level checks; the cells judge reads only these
for h in hosts:
    f = os.path.join(rdir, f"{h['id']}.json")
    if not os.path.exists(f):
        print(f"FAIL  {h['id']:7} no receipt at {f} — run scripts/model_ladder.sh on {h['id']} ({h.get('gpu')}, {h.get('cc')})"); rc = 1; continue
    try:
        R = json.load(open(f))
    except Exception as e:
        print(f"FAIL  {h['id']:7} receipt unreadable: {e}"); rc = 1; continue
    asha = R.get("apr_sha")
    if R.get("version") != version:
        # #4040: a nightly measured BEFORE the release's version bump carries the old version. It stands only
        # when the #4037 carry proved the whole diff to the cut harmless and named the bump as version-only.
        vproof = equiv_proof.get(asha, "") if isinstance(asha, str) and asha != cut else ""
        if "CARRIED FORWARD (#4037)" in vproof and "a workspace version bump only" in vproof:
            print(f"ok    {h['id']:7} receipt is for {R.get('version')!r}, the cut is {version!r}: carried across a version-only bump (#4040)")
        else:
            print(f"FAIL  {h['id']:7} receipt is for {R.get('version')!r}, this cut is {version!r} — STALE"); rc = 1; continue
    if not (isinstance(asha, str) and re.fullmatch(r"[0-9a-f]{40}", asha)):
        print(f"FAIL  {h['id']:7} receipt carries no 40-hex apr_sha ({asha!r}) — it names a version, and a version is not a build (#3957 F2)"); rc = 1; continue
    if asha != cut and asha not in equiv:
        print(f"FAIL  {h['id']:7} receipt measured at apr_sha {asha[:12]}, cut is {cut_sha[:12]}, and the trees differ outside evidence/ — STALE BY SHA: re-measure at the cut (#3957 F2)"); rc = 1; continue
    if asha != cut and asha in equiv_proof:
        print(f"ok    {h['id']:7} receipt apr_sha {asha[:12]} binds cut {cut_sha[:12]}: {equiv_proof[asha]}")
    if int(R.get("executed", 0)) < 1:
        print(f"FAIL  {h['id']:7} receipt executed=0 — a receipt that measured nothing is not evidence"); rc = 1; continue
    inv = R.get("inventory")
    if R.get("schema") != "apr-model-ladder-receipt/v2" or not isinstance(inv, list):
        print(f"FAIL  {h['id']:7} receipt carries no measured inventory (schema {R.get('schema')!r}) — the universe is what the host HOLDS, not a list (#3712)"); rc = 1; continue
    if not inv:
        print(f"FAIL  {h['id']:7} measured inventory is EMPTY — a host holding no Q4_K model proved nothing (#3712)"); rc = 1; continue
    # #3842: RECONCILE THE SCALAR AGAINST THE ROWS. A RED THAT HIDES.
    # gx10's 0.69.1 receipt said `red: 3` and carried only 2 non-green rows: the failed
    # REQUIRED rung qwen35-27b-q4km was absent from its own receipt entirely. `apr qa
    # --json` wrote 0 bytes for that cell (after 78 s of successful GPU work and a
    # PASSED F2 guard), the ladder appended an empty line to rows.jsonl, the assembler
    # dropped it -- and `red` was counted on a separate path. A consumer reading `rows`
    # would see 13/15 green plus two known refusals and could not learn that a required
    # rung failed at all. Every other guard here looks for a false GREEN; this one looks
    # for a MISSING RED, which no amount of per-row judging can find.
    declared_red = R.get("red")
    if isinstance(declared_red, int):
        rows_red = sum(1 for x in R.get("rungs", []) if not x.get("green"))
        if declared_red != rows_red:
            # Report and KEEP JUDGING. An inconsistent counter is an ADDITIONAL finding,
            # not a reason to stop reading the rows -- a `continue` here would suppress
            # every real per-row refusal behind it, which is the same suppression the
            # check exists to expose.
            print(f"FAIL  {h['id']:7} receipt says red={declared_red} but carries {rows_red} non-green row(s) — a red that is counted and not recorded is a red nobody can read (#3842)"); rc = 1
    # #4051: from 0.70.0 every measured row carries a stamp for every apr call it records -- qa, code, and
    # run/chat/serve on each backend -- with t_end >= t_start and an explicit lock wait (0 only for a lane
    # that takes no lock). Every #4033 lever is judged against these, so a missing one is RED, never zero.
    if timing_required(version):
        for x in R.get("rungs", []):
            if not x.get("present") or x.get("row_synthesized"):
                continue
            for why in timing_gaps(x):
                print(f"FAIL  {h['id']:7} {x.get('id') or x.get('file')}: TIMING {why} (#4051)"); rc = 1
    good[h["id"]] = R
    held_sha[h["id"]] = {i.get("file"): i.get("sha256") for i in inv if isinstance(i, dict)}
    by = {r.get("id"): r for r in R.get("rungs", [])}
    by_file = {x.get("file"): x for x in R.get("rungs", []) if x.get("file")}
    ladder_files = {r.get("gguf") for r in rungs}
    inv_green = 0
    for item in inv:
        f = item.get("file")
        x = by_file.get(f)
        if x is None or not x.get("present"):  # held by the host, absent from the run
            print(f"FAIL  {h['id']:7} inventory model {f} is MISSING from the run — the host holds it, so the release must prove it (#3712)"); rc = 1; continue
        if f in ladder_files:
            continue  # a ladder rung: judged, required, in the rung loop below
        why = why_of(x, inv_backends)
        if named_red(h["id"], f, x, why, inv_backends): continue
        kr = known_red(None, f, x.get("sha256") or item.get("sha256"), why)
        if kr:
            for e in kr: e["_used"] += 1
            print(f"KNOWN-RED {h['id']:7} inv:{f} ships with {', '.join(sorted({e['ticket'] for e in kr}))} -- " + "; ".join(why)); continue
        # #3846 / operator ruling 2026-09-22 ("we kick the MoE work to later"): a DECLARED
        # deferral. `inventory.deferred` maps a filename glob to a REASON, and a failing row
        # whose file matches one is printed DEFERRED with that reason instead of refusing.
        # Three properties make this a mechanism rather than an escape hatch:
        #   - it needs a non-empty reason, so an undocumented glob defers nothing;
        #   - it is NOT counted in inv_green, so a deferral can never read as a pass;
        #   - a row that fails and matches NOTHING here is still refused (case table).
        # It only ever applies to a row that ALREADY failed: a matching row that is green is
        # reported green, because deferring a passing model would hide a working capability.
        defer_why = None
        # #3880: MATCH is counted for every held file the key covers, green or not -- that is
        # what makes "no such file here" distinguishable from "every such file now passes".
        for pat, reason in (inv_deferred or {}).items():
            if fnmatch.fnmatch(f.lower(), str(pat).lower()):
                defer_matched[pat] = defer_matched.get(pat, 0) + 1
        if why:
            for pat, reason in (inv_deferred or {}).items():
                if fnmatch.fnmatch(f.lower(), str(pat).lower()) and isinstance(reason, str) and reason.strip():
                    defer_why = reason.strip(); defer_used[pat] = defer_used.get(pat, 0) + 1; break
        if why and defer_why:
            print(f"DEFER {h['id']:7} inv:{f:22} {defer_why} — was: " + "; ".join(why))
        elif why: print(f"FAIL  {h['id']:7} inv:{f:22} " + "; ".join(why)); rc = 1
        else:   inv_green += 1; print(f"ok    {h['id']:7} inv:{f:22} green on {','.join(inv_backends)}")
    print(f"ok    {h['id']:7} inventory: {len(inv)} Q4_K model(s) held, every one in the run")
    for r in rungs:
        rid = r["id"]; req = bool(r.get("required")) or is_q4k(r)
        x = by.get(rid)
        tag = "required" if req else "optional"
        listed = r.get("hosts")
        if listed and h["id"] not in listed:
            print(f"skip  {h['id']:7} {rid:22} not listed for this host (hosts: {','.join(listed)}) — not a claim here, not a pass either"); continue
        if x is None or not x.get("present"):
            if req: print(f"FAIL  {h['id']:7} {rid:22} ABSENT — a required rung the host does not hold is unmeasured, not passed"); rc = 1
            else:   print(f"skip  {h['id']:7} {rid:22} absent ({tag})")
            continue
        # #3957 F3: `is False` passed a row with no sha_ok at all. Only an explicit True is a match.
        if "sha_ok" not in x:
            print(f"FAIL  {h['id']:7} {rid:22} `sha_ok` is MISSING — a row that does not say its file matched the pinned sha256 has not shown it measured this rung (#3957 F3)"); rc = 1; continue
        if x.get("sha_ok") is not True:
            print(f"FAIL  {h['id']:7} {rid:22} sha256 mismatch — a different file is a different measurement"); rc = 1; continue
        why = why_of(x, r.get("backends", []))
        if named_red(h["id"], r.get("gguf") or x.get("file") or rid, x, why, r.get("backends", [])): continue
        kr = known_red(rid, r.get("gguf") or x.get("file"), x.get("sha256") or r.get("sha256"), why)
        if kr:
            for e in kr: e["_used"] += 1
            print(f"KNOWN-RED {h['id']:7} {rid} ships with {', '.join(sorted({e['ticket'] for e in kr}))} -- " + "; ".join(why)); continue
        if why:
            if req: print(f"FAIL  {h['id']:7} {rid:22} " + "; ".join(why)); rc = 1
            else:   print(f"warn  {h['id']:7} {rid:22} ({tag}) " + "; ".join(why))
        else:
            print(f"ok    {h['id']:7} {rid:22} green on {','.join(r.get('backends', []))} ({R.get('gpu') or 'no-gpu'}, sha {R.get('sha')})")
# ---- cells: verb x thinking x context rung, per model per host (#3712 row B)
rungs_doc = None
if rungs_p and os.path.exists(rungs_p):
    try:
        rungs_doc = json.load(open(rungs_p))
    except Exception as e:
        print(f"FAIL  context rungs {rungs_p} unreadable: {e}")
rungs_main = None
if rungs_main_p and os.path.exists(rungs_main_p) and os.path.getsize(rungs_main_p) > 0:
    try:
        rungs_main = json.load(open(rungs_main_p))
    except Exception as e:
        print(f"FAIL  context rungs at origin/main unreadable: {e}"); rc = 1
# #3880: refuse a deferral that deferred nothing while covering something. Runs after EVERY
# host, because a key may be idle on one host and load-bearing on another -- judging it
# per-host would refuse a live key the moment one host held no matching file.
for pat in inv_deferred:
    if defer_matched.get(pat, 0) > 0 and defer_used.get(pat, 0) == 0:
        print(f"FAIL  deferral {pat!r} is STALE: it covers {defer_matched[pat]} held model(s) and deferred NOTHING — "
              f"every file it excuses is green, so the capability arrived and the key outlived it. Delete it (#3880).")
        rc = 1
if RV.finish(good, {k: list(v) for k, v in held_sha.items()}, [h["id"] for h in hosts], lambda y: not why_of(y, ["cuda"])):
    rc = 1
# #3907: after EVERY host, a known-red entry that covered nothing is stale; one that covered rows is
# summarised so the run cannot then claim "every required rung green".
for e in KR:
    if e["_used"] == 0:
        print(f"FAIL  known-red entry {e.get('rung') or e.get('file')} {e['ticket']} covered NOTHING on this sweep -- "
              f"the clause no longer fails (or the model changed): delete the entry, a known red must not outlive its defect"); rc = 1
if any(e["_used"] for e in KR):
    print(f"KNOWN-RED {sum(e['_used'] for e in KR)} row(s) ship RED with their tickets: "
          + ", ".join(sorted({e['ticket'] for e in KR if e['_used']})) + " -- counted RED, never green")
if model_ladder_cells.judge(L, good, rungs_doc, print, rungs_main):
    rc = 1
# #3957 F4/F8: every (model, format, quant, host, backend, verb) cell must be PROVEN by an outside
# oracle -- the CRUX receipts bound to the cut -- or, for .apr, by the chain to its source.
if model_ladder_crux.judge(L, good, crux_dir, cut, equiv, print, RV.proven, cert_p, timing_required(version)):
    rc = 1
# #3957 F1: a DEFERRED row is not green. DEFER is `Unknown(NotRun)` in the fleet vocabulary
# (crates/aprender-contracts/src/ontology/verdict.rs FLEET_LABELS), the same element as
# `decline` -- exit 2. It used to fall through to exit 0, so a run whose only non-green rows
# were deferred printed "every required rung green". FAIL still dominates: rc 1 stays 1.
# The line below must NEVER start `DEFERRED: ` (with the colon): scripts/dogfood.sh reads
# that exact prefix as "this gate cannot be measured pre-publish" and marks it DEFER instead
# of FAIL. Pinned by must_not_match in the defer-* cases.
deferred_rows = sum(defer_used.values())
if deferred_rows and rc == 0:
    print(f"DEFERRED {deferred_rows} row(s) — declared, not proven: exit 2, never green (#3957 F1)")
    rc = 2
sys.exit(rc)
PY
}

# ---------------------------------------------------------------- self-test
# ---------------------------------------------------------------- the GPU lock
# Cop ruling 2026-09-21 (#3712): every apr call scripts/model_ladder.sh makes runs under the fleet GPU
# lock (flock, bounded wait) and choom -n 1000, through ONE function, apr_locked. Two halves:
#   lock_audit  (static, also in the REAL run): no apr subcommand is invoked on "$APR" directly.
#   lock_probe  (behavioural, self-test): a fake apr, called through --lock-probe, must see the lock
#               held and its own oom_score_adj at 1000; with the lock held elsewhere the call must
#               decline (exit 2) within the bounded wait, naming the holder's pid.
# lock_audit <producer> -> prints FAIL lines, exit 1 on any raw call
# ---------------------------------------------------------------- receipt equivalence
# equiv_lines <ladder> <receipt dir> <crux dir> <cut> <scratch prefix> -> "sha<TAB>proof" per receipt sha that
# binds the cut without being the cut. Run from the repo root (git, scripts/lib). A function so the
# self-test drives the SAME code on a scratch repository (`--equiv-only`).
equiv_lines() {
  # A receipt measured at another commit still binds when that commit's tree equals the cut's
  # outside evidence/ -- committing the receipts is itself a commit. Proven here, per sha, with git.
  # The shas: every ladder receipt's apr_sha and every CRUX receipt's binary (apr_sha_of, #3957 F2).
  python3 -c 'import glob, json, os, sys
sys.path.insert(0, "scripts/lib"); import model_ladder_crux
for f in glob.glob(sys.argv[1] + "/*.json"):
    try: print(json.load(open(f)).get("apr_sha") or "")
    except Exception: pass
for f in glob.glob(sys.argv[2] + "/*.json"):
    if os.path.basename(f).startswith("prompt-certification"): continue
    try: print(model_ladder_crux.apr_sha_of(json.load(open(f))) or "")
    except Exception: pass' "$2" "$3" 2> /dev/null | sort -u | while read -r s; do
    [ -n "$s" ] && [ -n "$4" ] || continue
    git -c safe.directory="$PWD" cat-file -e "$s^{commit}" 2> /dev/null || continue
    # #3957 F2 + #3710 ruling 3: evidence-only, or the contract's scoped hotfix -- scripts/lib/ladder_equiv.py.
    git -c safe.directory="$PWD" diff --name-only "$s" "$4" 2> /dev/null > "$5.paths" || continue
    git -c safe.directory="$PWD" show "$s:Cargo.lock" > "$5.lock_a" 2> /dev/null || : > "$5.lock_a"
    git -c safe.directory="$PWD" show "$4:Cargo.lock" > "$5.lock_b" 2> /dev/null || : > "$5.lock_b"
    python3 -c 'import sys, yaml
sys.path.insert(0, "scripts/lib"); import ladder_equiv
scope = (yaml.safe_load(open(sys.argv[3]))["ladder"] or {}).get("hotfix_scope") or {}
paths = [ln.strip() for ln in open(sys.argv[4]) if ln.strip()]
kind, proof = ladder_equiv.classify(sys.argv[1], sys.argv[2], paths, scope, open(sys.argv[5]).read(), open(sys.argv[6]).read())
if kind: print(sys.argv[1] + "\t" + proof)
sys.exit(0 if kind else 3)' "$s" "$4" "$1" "$5.paths" "$5.lock_a" "$5.lock_b" && continue
    # #4037: CARRY-FORWARD -- no path in the diff can reach the apr binary's inference (closure derived from
    # both endpoints' crate graphs, embedded and measurement inputs derived from their sources). Each
    # endpoint is checked out detached in a throwaway worktree. Any failure does not carry.
    if proof=$(python3 scripts/lib/ladder_carry.py "$PWD" "$s" "$4" 2> /dev/null); then
      printf '%s\tCARRIED FORWARD (#4037) -- %s\n' "$s" "$proof"
    fi
  done
  rm -f "$5.paths" "$5.lock_a" "$5.lock_b"
}

lock_audit() {
  python3 - "$1" <<'LOCKPY'
import re, sys
bad = 0
for n, line in enumerate(open(sys.argv[1]), 1):
    code = re.sub(r"(^|\s)#.*$", "", line)          # a call in a comment is not a call
    for m in re.finditer(r'(?<!\\)"?\$\{?APR\}?"?[ \t]+(?!--version\b)([a-z][a-z-]*)\b', code):
        print(f"FAIL  {sys.argv[1]}:{n} calls \"$APR\" {m.group(1)} directly -- every GPU apr call goes through apr_locked (the fleet lock + choom 1000)")
        bad = 1
sys.exit(bad)
LOCKPY
}
# exclusive_audit <producer> -> the GPU run leg goes through gpu_exclusive_run and a CONTENDED
# refusal declines (#3964). Static: the helper's behaviour is its own 12-row self-test
# (scripts/lib/gpu_exclusive_run_selftest.sh); this proves the ladder is wired to it.
exclusive_audit() {
  python3 - "$1" <<'EXCLPY'
import re, sys
src = open(sys.argv[1]).read()
code = "\n".join(re.sub(r"(^|\s)#.*$", "", l) for l in src.splitlines())
bad = 0
body = re.search(r"apr_exclusive\(\) \{(.*?)\n\}", code, re.S)
if not body or 'bash scripts/lib/gpu_exclusive_run.sh "$APR" "$@"' not in body.group(1):
    print("FAIL  apr_exclusive does not run apr through scripts/lib/gpu_exclusive_run.sh"); bad = 1
elif "GPU_OWNED_PREFIX=" not in body.group(1) or 'GPU_LOCK="$GPU_LOCK"' not in body.group(1):
    print("FAIL  apr_exclusive does not pass GPU_OWNED_PREFIX and the ladder's GPU_LOCK"); bad = 1
if not re.search(r'if \[ "\$flag" = --gpu \].*\n(?:\s*#[^\n]*\n)*\s*apr_exclusive run "\$path"', code):
    print("FAIL  the --gpu run leg does not go through apr_exclusive -- a foreign GPU process goes unseen"); bad = 1
if not re.search(r"gpu_exclusive_run: CONTENDED' \"\$run_e\"; then\n[^\n]*decline: ENV the GPU was not exclusive[^\n]*\n\s*exit 2", code):
    print("FAIL  a CONTENDED GPU leg does not decline (exit 2) -- a shared-card result would be judged"); bad = 1
sys.exit(bad)
EXCLPY
}
# lock_probe <producer> <work dir> -> prints ok/FAIL lines, exit 1 on any failure
lock_probe() {
  local prod=$1 w=$2 out rc hp bad=0
  mkdir -p "$w"; : > "$w/lock"
  printf '#!/usr/bin/env bash\nif flock -n "$FAKE_LOCK" true; then l=UNLOCKED; else l=LOCKED; fi\necho "fake-apr $1 lock=$l oom=$(cat /proc/self/oom_score_adj)"\n' > "$w/apr"
  chmod +x "$w/apr"
  out=$(FAKE_LOCK="$w/lock" MODEL_LADDER_ROOT="$PWD" MODEL_LADDER_GPU_LOCK="$w/lock" DOGFOOD_ALLOW_UNPINNED=1 APR="$w/apr" timeout 60 bash "$prod" --lock-probe qa probe 2>&1); rc=$?
  if [ "$rc" = 0 ] && grep -q 'lock=LOCKED oom=1000' <<< "$out"; then echo "ok    lock: an apr call runs holding the lock, at oom_score_adj 1000"
  else echo "FAIL  lock: the probe call did not run holding the lock at oom 1000 (rc=$rc): $out"; bad=1; fi
  python3 -c 'import fcntl, sys, time; f = open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); time.sleep(60)' "$w/lock" &
  hp=$!
  sleep 0.5
  out=$(FAKE_LOCK="$w/lock" MODEL_LADDER_ROOT="$PWD" MODEL_LADDER_GPU_LOCK="$w/lock" MODEL_LADDER_LOCK_WAIT=1 DOGFOOD_ALLOW_UNPINNED=1 APR="$w/apr" timeout 8 bash "$prod" --lock-probe run probe 2>&1); rc=$?
  kill "$hp" 2> /dev/null; wait "$hp" 2> /dev/null
  if [ "$rc" = 2 ] && grep -q 'was not free after 1s' <<< "$out" && grep -q "holder: pid $hp" <<< "$out"; then echo "ok    lock: a held lock declines (exit 2) in the bounded wait, naming the holder's pid"
  else echo "FAIL  lock: a held lock did not decline in the bounded wait naming its holder (rc=$rc): $out"; bad=1; fi
  return "$bad"
}

CASE_CUT=ca5eca5eca5eca5eca5eca5eca5eca5eca5eca5e
if [ "$EQUIV_ONLY" = 1 ]; then
  # bashrs SEC010: $t is this run's own mktemp.
  # bashrs disable-next-line=SEC010
  t=$(mktemp); equiv_lines "$LADDER" "$RECEIPT_DIR" "$CRUX_DIR" "$CUT_COMMIT" "$t"; rm -f "$t"; exit 0
fi
if [ "$SELF_TEST" = 1 ]; then
  n=0; bad=0
  for c in "$CASES_DIR"/*/; do
    name=$(basename "$c")
    [ -z "$ONLY_CASE" ] || [ "$name" = "$ONLY_CASE" ] || continue
    [ -f "$c/expected_rc" ] || { echo "FAIL  case $name has no expected_rc"; bad=$((bad+1)); continue; }
    want=$(cat "$c/expected_rc")
    # #3887 THE FLOOR. A case expecting a REFUSAL must assert WHY. Judged on rc alone it
    # cannot tell "the gate refused for my planted reason" from "the gate was already
    # refusing for some other reason", so it passes for the wrong reason, permanently and
    # invisibly. Measured: proving #3880's stale-deferral check, a mutant disabling the
    # refusal left red-deferral-stale at rc=1 anyway -- from an unrelated pre-existing
    # failure in the same fixture -- and `expected_rc` alone would have let that mutant
    # SURVIVE. Only the must_match on the STALE text killed it.
    if [ "$want" != "0" ] && [ ! -f "$c/must_match" ]; then
      echo "FAIL  case $name expects rc=$want and carries no must_match -- a case that cannot say WHY it is red has no floor under it (#3887)"
      bad=$((bad+1)); continue
    fi
    lad="$c/ladder.yaml"; [ -f "$lad" ] || lad="$LADDER"
    main=""; [ -f "$c/ladder_main.yaml" ] && main="$c/ladder_main.yaml"
    # #3957 F2: every case is judged against a cut commit. CASE_CUT is the table's convention
    # (every fixture receipt carries it as apr_sha); a case overrides it with a `cut_commit` file,
    # and lists shas proven equal to the cut in `equivalent_shas`.
    cut=$(cat "$c/cut_commit" 2>/dev/null || echo "$CASE_CUT")
    out=$(judge "$lad" "$main" "$c/receipts" "$(cat "$c/version" 2>/dev/null || echo 0.0.0-case)" "$c/context-rungs.json" "$c/context-rungs_main.json" "$cut" "$c/equivalent_shas" "$c/crux" "$c/certification.json"); got=$?
    n=$((n+1))
    # #3887 THE OTHER POLARITY. A case that exists to prove a gate stays QUIET rests on rc
    # alone otherwise, and that works only because an over-eager check happens to flip rc.
    # A check printing a spurious line WITHOUT changing rc would pass it. Not required of
    # every green case -- most assert normal operation, not silence -- but honoured wherever
    # a case declares one. Here-strings throughout: never a pipe into grep -q.
    if [ "$got" = "$want" ] \
       && { [ ! -f "$c/must_match" ] || grep -qE "$(cat "$c/must_match")" <<< "$out"; } \
       && { [ ! -f "$c/must_not_match" ] || ! grep -qE "$(cat "$c/must_not_match")" <<< "$out"; }; then
      printf 'ok    case %-28s rc=%s\n' "$name" "$got"
    else
      printf 'FAIL  case %-28s rc=%s want=%s%s%s\n' "$name" "$got" "$want" \
        "$([ -f "$c/must_match" ] && printf ' must_match=/%s/' "$(cat "$c/must_match")")" \
        "$([ -f "$c/must_not_match" ] && printf ' must_not_match=/%s/' "$(cat "$c/must_not_match")")"
      printf '%s\n' "$out" | sed 's/^/        /'; bad=$((bad+1))
    fi
  done
  if [ "$n" -lt 6 ] && [ -z "$ONLY_CASE" ]; then echo "FAIL  only $n case(s) ran; the table needs >= 6 to discriminate"; bad=$((bad+1)); fi
  if [ -n "$ONLY_CASE" ] && [ "$n" -eq 0 ]; then echo "FAIL  no case named $ONLY_CASE under $CASES_DIR -- a case that did not run is not a pass"; bad=$((bad+1)); fi
  # Mutants (#3712): each refusal is deleted in a copy of this script, and the case that
  # names it must go RED under the copy. A rule no case can tell from its absence is theater.
  if [ -z "$ONLY_CASE" ]; then
    mdir=$(mktemp -d "${TMPDIR:-/tmp}/ladder-mut.XXXXXX") || exit 2
    mutant() { # mutant <label> <case that must kill it> <sed expression deleting the rule>
      local m="$mdir/$1.sh"
      sed "$3" "$SELF" > "$m"
      if cmp -s "$SELF" "$m"; then echo "FAIL  mutant $1 did not apply -- case $2 proves nothing"; bad=$((bad+1)); return; fi
      if MODEL_LADDER_ROOT="$PWD" bash "$m" --self-test --case "$2" >/dev/null 2>&1; then echo "FAIL  mutant $1 SURVIVED: case $2 stays ok with the rule deleted"; bad=$((bad+1))
      else printf 'ok    mutant %-28s killed by case %s\n' "$1" "$2"; fi
    }
    # #3887: the FLOOR itself, proven against a PLANTED case rather than by planting a
    # permanently-failing case in the real table. A red-expecting case with no must_match
    # must be refused; a red-expecting case WITH one must still run. Both directions,
    # because a floor that refuses everything is as useless as one that refuses nothing.
    floor_case() { # floor_case <dir> <expected_rc> [must_match text]
      mkdir -p "$1/receipts"
      printf '%s\n' "$2" > "$1/expected_rc"
      [ -z "${3:-}" ] || printf '%s\n' "$3" > "$1/must_match"
    }
    fdir="$mdir/floor"; mkdir -p "$fdir"
    floor_case "$fdir/red-no-reason" 2
    if MODEL_LADDER_CASES_DIR="$fdir" MODEL_LADDER_ROOT="$PWD" bash "$SELF" --self-test --case red-no-reason >/dev/null 2>&1; then
      echo "FAIL  the #3887 floor did not refuse a red-expecting case with no must_match"; bad=$((bad+1))
    else
      printf 'ok    floor: a red-expecting case with no must_match is refused (#3887)\n'
    fi
    floor_case "$fdir/red-with-reason" 2 'decline'
    out_fr=$(MODEL_LADDER_CASES_DIR="$fdir" MODEL_LADDER_ROOT="$PWD" bash "$SELF" --self-test --case red-with-reason 2>&1)
    if grep -qE 'carries no must_match' <<< "$out_fr"; then
      echo "FAIL  the #3887 floor refused a red-expecting case that DOES carry a must_match -- it refuses everything"; bad=$((bad+1))
    else
      printf 'ok    floor: a red-expecting case WITH a must_match is not refused by the floor (#3887)\n'
    fi
    mutant q4k-required-false red-q4k-required-false 's/if is_q4k(r) and r.get("required") is not True:/if False:/'
    mutant q4k-without-cuda   red-q4k-rung-cpu-only  's/if is_q4k(r) and "cuda" not in (r.get("backends") or \[\]):/if False:/'
    mutant inventory-missing  red-inventory-model-missing 's/if x is None or not x.get("present"):  # held by the host, absent from the run/if False:/'
    # #3898: the rule that reads `qa_rc`. Deleting it must break the case that names it.
    mutant qa-rc-ignored      red-qa-rc-nonzero-refused 's/    elif qa_rc != 0:/    elif False:/'
    mutant gates-unaccounted  red-gates-do-not-account-for-rc 's|if x.get("gates_account_for_rc") is False:|if False:|'
    # #3957 F4a: the judge reads `output_bad` on chat, code and every serve route.
    # #3957 F3: an ABSENT qa_rc / sha_ok is a FAIL, never the passing value.
    mutant qa-rc-absent       red-qa-rc-missing           's/if "qa_rc" not in x or not isinstance(qa_rc, int) or isinstance(qa_rc, bool):/if False:/'
    mutant sha-ok-absent      red-sha-ok-missing          's/        if "sha_ok" not in x:/        if False:/'
    # #3907: a KNOWN-RED is pinned to ONE model, ONE clause and ONE ticket, and goes stale when it covers nothing.
    mutant kr-any-model      red-known-red-other-model  's/(e.get("rung") == row_key or e.get("file") == f) and e\["sha256"\] == sha/True/'
    mutant kr-any-reason     red-known-red-extra-clause 's/^            return None$/            continue/'
    mutant kr-any-clause     red-known-red-other-clause 's/ and any(r_.search(w) for r_ in e\["_re"\])//'
    mutant kr-never-stale    red-known-red-stale        's/^    if e\["_used"\] == 0:$/    if False:/'
    mutant kr-no-ticket-ok   red-known-red-no-ticket    's/("sha256", "clause", "ticket", "ruling")/("sha256", "clause", "ruling")/'
    # #3957 F1: a deferred row exits 2 (Unknown/NotRun), never 0.
    mutant defer-exits-0      defer-inventory-declared    's/if deferred_rows and rc == 0:/if False:/'
    mutant defer-exits-0-qa   defer-qa-rc-declared        's/if deferred_rows and rc == 0:/if False:/'
    mutant defer-colon        defer-inventory-declared    's/DEFERRED {deferred_rows} row(s)/DEFERRED: {deferred_rows} row(s)/'
    # #3957 F2: receipts bind to the cut commit's sha; the prose drift key is refused.
    mutant sha-stale          red-receipt-sha-stale       's/    if asha != cut and asha not in equiv:/    if False:/'
    mutant sha-missing        red-receipt-sha-missing     's/    if not (isinstance(asha, str) and re.fullmatch(r"\[0-9a-f\]{40}", asha)):/    if False:/'
    mutant version-drift-any-carry red-version-differs-not-a-bump 's/        if "CARRIED FORWARD (#4037)" in vproof and "a workspace version bump only" in vproof:/        if "CARRIED FORWARD (#4037)" in vproof:/'
    mutant sha-equiv-ignored  green-receipt-sha-equivalent 's/    if asha != cut and asha not in equiv:/    if asha != cut:/'
    # #4051: the timing rule, each clause deleted; the case that names the clause must go RED.
    mutant timing-rule-off     red-timing-missing          's/^    if timing_required(version):$/    if False:/'
    mutant timing-verb-unread  red-timing-verb-unstamped   's/^            out.append("no stamp for /            pass  # out.append("no stamp for /'
    mutant timing-inspect-unrequired red-timing-inspect-unstamped 's/^    need = \[("qa", None), ("inspect", None), ("code", None)\]/    need = [("qa", None), ("code", None)]/'
    mutant timing-order-unread red-timing-end-before-start 's/^        elif t1 < t0:$/        elif False:/'
    if python3 -c 'import sys, yaml; t = (yaml.safe_load(open(sys.argv[1]))["ladder"].get("timing") or {}); sys.exit(0 if str(t.get("required_from") or "") == "0.70.0" else 1)' "$LADDER"; then
      echo "ok    timing: the shipped contract requires per-call stamps from 0.70.0 (#4051)"
    else echo "FAIL  timing: $LADDER does not set ladder.timing.required_from 0.70.0 -- the #4051 rule is off for real receipts"; bad=$((bad+1)); fi
    mutant sha-drift-key      red-ladder-apr-sha-drift-key 's/^if "apr_sha_drift" in (L.get("inventory") or {}) or "apr_sha_drift" in L:/if False:/'
    mutant verb-output-bad    red-chat-output-bad-rc0     's/if r is not None and r.get("output_bad"):/if False:/'
    mutant verb-output-bad-code red-code-output-bad-rc0   's/if r is not None and r.get("output_bad"):/if False:/'
    mutant route-output-bad   red-serve-route-output-bad  's/if rv.get("output_bad"): why.append/if False: why.append/'
    # AND THE CROSS-CHECK THAT MAKES THE MUTANT MEAN SOMETHING (#3898).
    # A mutant killed by its own case only proves the case reads the rule. It does NOT
    # prove the rule covers a surface nothing else covers — and #3842's reconcile is the
    # nearest neighbour, close enough that "we already had that" is the obvious objection.
    # So: delete the qa_rc rule and require the #3842 case to stay GREEN. If #3842 went red
    # too, the two rules would be redundant and this one would be unjustified.
    qa_mut="$mdir/cross-qa-rc.sh"
    sed 's/    elif qa_rc != 0:/    elif False:/' "$SELF" > "$qa_mut"
    if cmp -s "$SELF" "$qa_mut"; then
      echo "FAIL  cross-check mutant did not apply -- the #3842 independence claim proves nothing"; bad=$((bad+1))
    elif MODEL_LADDER_ROOT="$PWD" bash "$qa_mut" --self-test --case red-receipt-red-not-recorded >/dev/null 2>&1; then
      printf 'ok    cross: #3842 reconcile stays GREEN with the qa_rc rule deleted — #3898 covers a surface it cannot\n'
    else
      echo "FAIL  cross: deleting the qa_rc rule also broke red-receipt-red-not-recorded -- the two rules are not independent, so #3898 is not the separate surface it claims"; bad=$((bad+1))
    fi
    # The lock: the real producer passes both halves; each producer mutant must fail at least one.
    prod=scripts/model_ladder.sh
    if lock_audit "$prod" > "$mdir/audit.out"; then echo "ok    lock: $prod makes no GPU apr call outside apr_locked"
    else cat "$mdir/audit.out"; bad=$((bad+1)); fi
    lock_probe "$prod" "$mdir/probe" || bad=$((bad+1))
    pmutant() { # pmutant <label> <sed expression breaking the lock in a copy of the producer>
      local m="$mdir/p-$1.sh"
      sed "$2" "$prod" > "$m"
      if cmp -s "$prod" "$m"; then echo "FAIL  producer mutant $1 did not apply -- the lock checks prove nothing"; bad=$((bad+1)); return; fi
      if lock_audit "$m" > /dev/null && lock_probe "$m" "$mdir/probe-$1" > /dev/null; then echo "FAIL  producer mutant $1 SURVIVED the lock checks"; bad=$((bad+1))
      else printf 'ok    producer mutant %-14s killed by the lock checks\n' "$1"; fi
    }
    pmutant raw-apr-call 's/apr_locked qa "\$path"/"$APR" qa "$path"/'
    pmutant no-lock      's/^  flock -E "\$LOCK_BUSY" -w "\$LOCK_WAIT" "\$GPU_LOCK" sh -c/  sh -c/'
    pmutant no-choom     's/exec choom -n 1000 -- "\$@"/exec "$@"/'
    # #4051: the stamp an apr call writes. stamp_probe holds the lock ~2 s, then makes one call through
    # --lock-probe; the stamp must exist, be ordered, and show the wait. Each mutant names its row.
    stamp_probe() { # stamp_probe <producer> <work dir> -> ok/FAIL lines naming the row
      local prod=$1 w=$2 hp rc
      mkdir -p "$w"; : > "$w/lock"; : > "$w/stamps.jsonl"
      printf '#!/usr/bin/env bash\necho fake-apr\n' > "$w/apr"; chmod +x "$w/apr"
      python3 -c 'import fcntl, sys, time; f = open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); time.sleep(2)' "$w/lock" &
      hp=$!; sleep 0.3
      LADDER_STAMPS="$w/stamps.jsonl" MODEL_LADDER_ROOT="$PWD" MODEL_LADDER_GPU_LOCK="$w/lock" MODEL_LADDER_LOCK_WAIT=30 \
        DOGFOOD_ALLOW_UNPINNED=1 APR="$w/apr" timeout 60 bash "$prod" --lock-probe run probe > /dev/null 2>&1; rc=$?
      wait "$hp" 2> /dev/null
      python3 - "$w/stamps.jsonl" "$rc" <<'STP'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
x = rows[0] if len(rows) == 1 else {}
t0, ta, t1, lw = x.get("t_start"), x.get("t_acquired"), x.get("t_end"), x.get("lock_wait_s")
checks = [
    ("stamp-written", len(rows) == 1 and sys.argv[2] == "0", "one stamp line for one apr call (got %d, rc %s)" % (len(rows), sys.argv[2])),
    ("stamp-ordered", all(isinstance(v, float) for v in (t0, ta, t1)) and t0 <= ta <= t1, "t_start <= t_acquired <= t_end: %r" % [t0, ta, t1]),
    ("stamp-lock-wait", isinstance(lw, float) and lw >= 1.2 and x.get("lock") == "gpu", "a ~1.7 s held lock shows in lock_wait_s: %r (lock %r)" % (lw, x.get("lock"))),
    ("stamp-verb", x.get("verb") == "run", "the verb is recorded: %r" % x.get("verb")),
]
bad = 0
for name, ok, what in checks:
    print(("ok    stamp " if ok else "FAIL  stamp ") + name + " -- " + what)
    bad |= not ok
sys.exit(bad)
STP
    }
    if stamp_probe "$prod" "$mdir/stamp"; then :; else bad=$((bad+1)); fi
    # the join: a row takes exactly its own cell's stamps (by rid), and a failed join leaves `timing` ABSENT
    join_probe() { # join_probe <producer> <work dir>
      local prod=$1 w=$2 out
      mkdir -p "$w"
      printf '%s\n' '{"rid": "r1", "verb": "qa", "t_start": 1.0, "t_end": 2.0}' '{"rid": "r2", "verb": "qa", "t_start": 3.0, "t_end": 4.0}' \
        '{"rid": "r1", "verb": "code", "t_start": 5.0, "t_end": 6.0}' > "$w/stamps.jsonl"
      out=$(MODEL_LADDER_ROOT="$PWD" DOGFOOD_ALLOW_UNPINNED=1 APR=/bin/true bash "$prod" --timing-join '{"id": "a"}' "$w/stamps.jsonl" r1 2> /dev/null)
      python3 - "$out" <<'JP'
import json, sys
try: r = json.loads(sys.argv[1])
except ValueError: r = {}
t = r.get("timing")
ok = isinstance(t, list) and [x.get("verb") for x in t] == ["qa", "code"] and all("rid" not in x for x in t)
print(("ok    stamp " if ok else "FAIL  stamp ") + "stamp-joined -- the row takes its own cell's 2 stamps, not the other cell's: %r" % (t,))
sys.exit(0 if ok else 1)
JP
      out=$(MODEL_LADDER_ROOT="$PWD" DOGFOOD_ALLOW_UNPINNED=1 APR=/bin/true bash "$prod" --timing-join '{"id": "a"}' "$w/absent.jsonl" r1 2> /dev/null)
      if python3 -c 'import json, sys; sys.exit(0 if "timing" not in json.loads(sys.argv[1]) else 1)' "$out" 2> /dev/null; then
        echo "ok    stamp stamp-join-fails-absent -- an unreadable stamp file leaves timing ABSENT (the judge refuses it)"
      else echo "FAIL  stamp stamp-join-fails-absent -- got: $out"; return 1; fi
    }
    if join_probe "$prod" "$mdir/join"; then :; else bad=$((bad+1)); fi
    jmutant() { # jmutant <label> <row> <sed on a copy of the producer>
      local m="$mdir/j-$1.sh" out
      sed "$3" "$prod" > "$m"
      if cmp -s "$prod" "$m"; then echo "FAIL  join mutant $1 did not apply"; bad=$((bad+1)); return; fi
      out=$(join_probe "$m" "$mdir/join-$1")
      if printf '%s\n' "$out" | grep -q "^FAIL  stamp $2 "; then printf 'ok    join mutant %-16s killed by %s\n' "$1" "$2"
      else echo "FAIL  join mutant $1 SURVIVED row $2"; bad=$((bad+1)); fi
    }
    # #4051/#4035: apr qa's per-gate duration_ms reaches the row (the qa JSON dies with $WORK, so the row is the
    # only place it survives). A fixture report through the SHIPPED row builder (--qa-row).
    qa_duration_probe() { # qa_duration_probe <producer> <work dir>
      local prod=$1 w=$2 out; mkdir -p "$w"
      printf '%s' '{"gates": [{"name": "golden_output", "passed": true, "duration_ms": 1234}, {"name": "tensor_contract", "passed": true, "duration_ms": 56}]}' > "$w/qa.json"
      out=$(MODEL_LADDER_ROOT="$PWD" DOGFOOD_ALLOW_UNPINNED=1 APR=/bin/true bash "$prod" --qa-row "$w/qa.json" 2> /dev/null)
      if python3 -c 'import json,sys; r=json.loads(sys.argv[1]); g=r["gates"]; sys.exit(0 if r["golden_output"]["duration_ms"] == 1234 and g["tensor_contract"]["duration_ms"] == 56 else 1)' "$out" 2> /dev/null; then
        echo "ok    stamp qa-duration -- apr qa's per-gate duration_ms reaches the row (1234 ms, 56 ms)"
      else echo "FAIL  stamp qa-duration -- got: $out"; return 1; fi
    }
    if qa_duration_probe "$prod" "$mdir/qad"; then :; else bad=$((bad+1)); fi
    m="$mdir/qad-mut.sh"; sed 's/^            "duration_ms": x.get("duration_ms")}$/            }/' "$prod" > "$m"
    if cmp -s "$prod" "$m"; then echo "FAIL  qa-duration mutant did not apply"; bad=$((bad+1))
    elif qa_duration_probe "$m" "$mdir/qad-m" > /dev/null; then echo "FAIL  qa-duration mutant duration-dropped SURVIVED"; bad=$((bad+1))
    else echo "ok    stamp mutant duration-dropped killed by qa-duration"; fi
    jmutant any-rid       stamp-joined           's/^    if x.get("rid") == rid:$/    if True:/'
    jmutant absent-empty  stamp-join-fails-absent 's/^  python3 - "\$1" "\$2" "\$3" <<.PY. 2> \/dev\/null || printf .%s. "\$1"$/  python3 - "$1" "$2" "$3" <<'"'"'PY'"'"' 2> \/dev\/null || python3 -c "import json,sys; r=json.loads(sys.argv[1]); r[\"timing\"]=[]; print(json.dumps(r))" "$1"/'

    smutant_p() { # smutant_p <label> <row that must go RED> <sed on a copy of the producer>
      local m="$mdir/st-$1.sh" out
      sed "$3" "$prod" > "$m"
      if cmp -s "$prod" "$m"; then echo "FAIL  stamp mutant $1 did not apply"; bad=$((bad+1)); return; fi
      out=$(stamp_probe "$m" "$mdir/stamp-$1")
      if printf '%s\n' "$out" | grep -q "^FAIL  stamp $2 "; then printf 'ok    stamp mutant %-16s killed by %s\n' "$1" "$2"
      else echo "FAIL  stamp mutant $1 SURVIVED row $2"; bad=$((bad+1)); fi
    }
    smutant_p no-stamp       stamp-written   's/^  stamp_call gpu "\$t0" /  : stamp_call gpu "$t0" /'
    smutant_p acq-is-start   stamp-lock-wait 's/^  stamp_call gpu "\$t0" "\$(cat "\$acq" 2> \/dev\/null)"/  stamp_call gpu "$t0" "$t0"/'
    smutant_p end-before     stamp-ordered   's/^  t1=\$(date +%s.%N)$/  t1=$t0/'
    pmutant unbounded    's/ -w "\$LOCK_WAIT"//'
    if exclusive_audit "$prod" > "$mdir/excl.out"; then echo "ok    exclusive: $prod runs its GPU leg through gpu_exclusive_run and declines CONTENDED (#3964)"
    else cat "$mdir/excl.out"; bad=$((bad+1)); fi
    emutant() { # emutant <label> <sed expression breaking the exclusive GPU leg in a copy of the producer>
      local m="$mdir/e-$1.sh"
      sed "$2" "$prod" > "$m"
      if cmp -s "$prod" "$m"; then echo "FAIL  exclusive mutant $1 did not apply -- the check proves nothing"; bad=$((bad+1)); return; fi
      if exclusive_audit "$m" > /dev/null; then echo "FAIL  exclusive mutant $1 SURVIVED the exclusive check"; bad=$((bad+1))
      else printf 'ok    exclusive mutant %-16s killed\n' "$1"; fi
    }
    emutant gpu-leg-locked    's/^      apr_exclusive run "$path"/      apr_locked run "$path"/'
    emutant helper-bypassed   's/bash scripts\/lib\/gpu_exclusive_run.sh "$APR" "$@"/"$APR" "$@"/'
    emutant contended-judged  '/decline: ENV the GPU was not exclusive/{n;s/exit 2/:/}'
    # The cells module (scripts/lib/model_ladder_cells.py): each rule deleted in a copy, imported through
    # MODEL_LADDER_CELLS_LIB, and the case that names the rule must go RED under the copy.
    cmutant() { # cmutant <label> <case that must kill it> <sed expression deleting the rule>
      local md="$mdir/c-$1"; mkdir -p "$md"
      sed "$3" scripts/lib/model_ladder_cells.py > "$md/model_ladder_cells.py"
      if cmp -s scripts/lib/model_ladder_cells.py "$md/model_ladder_cells.py"; then echo "FAIL  cells mutant $1 did not apply -- case $2 proves nothing"; bad=$((bad+1)); return; fi
      if MODEL_LADDER_CELLS_LIB="$md" bash "$SELF" --self-test --case "$2" > /dev/null 2>&1; then echo "FAIL  cells mutant $1 SURVIVED: case $2 stays ok with the rule deleted"; bad=$((bad+1))
      else printf 'ok    cells mutant %-17s killed by case %s\n' "$1" "$2"; fi
    }
    cmutant missing-cell    red-cells-missing-cell          's/fails.append(f"{label} MISSING"); failed_somewhere.add(key); continue/continue/'
    cmutant cotenant        red-cells-cotenant-refusal      's/if fit and need is not None:/if False:/'
    cmutant passes-nowhere  red-cells-declared-passes-nowhere 's/if not ok and key not in failed_somewhere:/if False:/'
    cmutant think-closed    red-cells-thinking-never-closed 's/if mode == "on" and c.get("think_closed") is not True:/if False:/'
    cmutant fell-back       red-cells-pass-fell-back        's/if c.get("fallback") is not False:/if False:/'
    cmutant prompt-short    red-cells-prompt-under-rung     's/if int(c.get("prompt_tokens") or 0) < tok:/if False:/'
    cmutant modes-evidence  red-cells-thinking-modes-disagree-with-template 's/elif want is not None and modes != want:/elif False:/'
    cmutant no-representative red-cells-arch-without-representative 's/        if not r:/        if False:/'
    cmutant pass-beyond-fit red-cells-pass-beyond-its-arithmetic 's/    if not fit:/    if False:/'
    cmutant family-long     red-cells-missing-cell          's/    if arch in (long_for.get("families") or \[\]):/    if False:/'
    cmutant rungs-floor     red-cells-rung-dropped-vs-main  's/        if gone:/        if False:/'
    cmutant not-armed       red-cells-no-cells              's/    if receipts and not carrying:/    if receipts:/'
    # #3957 F4/F8: the CRUX join (scripts/lib/model_ladder_crux.py), each rule deleted in a copy
    # imported through MODEL_LADDER_CRUX_LIB; the case that names the rule must go RED.
    xmutant() { # xmutant <label> <case that must kill it> <sed expression deleting the rule>
      local md="$mdir/x-$1"; mkdir -p "$md"
      sed "$3" scripts/lib/model_ladder_crux.py > "$md/model_ladder_crux.py"
      if cmp -s scripts/lib/model_ladder_crux.py "$md/model_ladder_crux.py"; then echo "FAIL  crux mutant $1 did not apply -- case $2 proves nothing"; bad=$((bad+1)); return; fi
      if MODEL_LADDER_CRUX_LIB="$md" bash "$SELF" --self-test --case "$2" > /dev/null 2>&1; then echo "FAIL  crux mutant $1 SURVIVED: case $2 stays ok with the rule deleted"; bad=$((bad+1))
      else printf 'ok    crux mutant %-18s killed by case %s\n' "$1" "$2"; fi
    }
    xmutant no-receipts       red-crux-missing        's/^    if not files:/    if False:/'
    xmutant cell-red-ignored  red-crux-cell-red       's/^    if bad:/    if False:/'
    xmutant cell-absent-ok    red-crux-verb-absent    's/^    if not got:/    if False and not got:/'
    xmutant unbound-receipt   red-crux-unbound        's/^        if not (asha and HEX40.fullmatch(asha)) or (asha != cut_sha and asha not in equiv):/        if False:/'
    xmutant declined-receipt  red-crux-declined       's/^        if summ.get("verdict") == "DECLINE":/        if False:/'
    xmutant apr-no-source     red-apr-no-source       's/^    if not isinstance(src, dict) or not src.get("file") or not HEX64.fullmatch(str(src.get("sha256") or "")):/    if False:/'
    xmutant apr-tensor-diff   red-apr-tensor-perturbed 's/^    elif td.get("tensors_differing") != 0:/    elif False:/'
    xmutant apr-summary-diff  red-apr-summary-diff-only 's/td.get("method") != "elementwise" or //'
    xmutant apr-greedy        red-apr-greedy-differs  's/^        elif e.get("equal") is not True:/        elif False:/'
    xmutant apr-source-proven red-apr-source-red      's/s_ok, s_why = crux_cell(index, src_sha, host, b, v)/s_ok, s_why = True, "mutant"/'
    # #3957 F9/F10: the named RED verdicts (scripts/lib/model_ladder_redmodel.py). Each rule deleted in a copy
    # imported through MODEL_LADDER_REDMODEL_LIB; the case that isolates the rule must go RED under the copy.
    rmutant() { # rmutant <label> <case that must kill it> <sed expression deleting the rule>
      local md="$mdir/r-$1"; mkdir -p "$md"
      sed "$3" scripts/lib/model_ladder_redmodel.py > "$md/model_ladder_redmodel.py"
      if cmp -s scripts/lib/model_ladder_redmodel.py "$md/model_ladder_redmodel.py"; then echo "FAIL  redmodel mutant $1 did not apply -- case $2 proves nothing"; bad=$((bad+1)); return; fi
      if MODEL_LADDER_REDMODEL_LIB="$md" bash "$SELF" --self-test --case "$2" > /dev/null 2>&1; then echo "FAIL  redmodel mutant $1 SURVIVED: case $2 stays ok with the rule deleted"; bad=$((bad+1))
      else printf 'ok    redmodel mutant %-19s killed by case %s\n' "$1" "$2"; fi
    }
    rmutant oracle-closes     red-model-oracle-closes        's/^            elif think_state(off.get("generated_text")) != want:/            elif False:/'
    rmutant empty-has-content red-model-empty-oracle-has-content 's/^            elif think_state(off.get("generated_text")) != want:/            elif False:/'
    rmutant official-missing  red-model-official-missing     's/^            if off is None:/            if False:/'
    rmutant official-ids      red-model-official-not-official 's/^            elif not (_ids(off.get("template_prompt_ids")) and off.get("prompt_ids") == off.get("template_prompt_ids")):/            elif False:/'
    rmutant parity-prompt     red-model-parity-prompt-differs 's/^            if not (_ids(a.get("prompt_ids")) and a.get("prompt_ids") == o.get("prompt_ids")):/            if False:/'
    rmutant no-oracle         red-model-no-oracle            's/^        if not gpu:/        if False:/'
    rmutant ids-differ        red-model-apr-ne-oracle        's/^            if a\["generated_ids"\] != o\["generated_ids"\]:/            if False:/'
    rmutant equal-flag        red-model-equal-flag-not-trusted 's/^            if a\["generated_ids"\] != o\["generated_ids"\]:/            if False:/'
    rmutant cpu-ne-gpu        red-model-cpu-ne-gpu           's/^            elif ca\["generated_ids"\] != a\["generated_ids"\]:/            elif False:/'
    rmutant control-blind     red-model-control-blind        's/^            elif any(think_state(r.get("generated_text")) != "ok" for r in co):/            elif False:/'
    rmutant control-self      red-model-control-is-self      's/^        elif csha == sha or cfile == f:/        elif False:/'
    rmutant control-cell      red-model-control-cell-red     's/^            elif any(v != "GREEN" for v in vs):/            elif False:/'
    rmutant bf16-leg          red-model-bf16-not-reproduced  's/^                    if think_state(t) != want:/                    if False:/'
    rmutant named-filter      green-red-model-named-prompts  's/^        if named:/        if False:/'
    rmutant named-prompt      red-model-named-prompt-unmeasured 's/^            if gone:/            if False:/'
    # #3957 F9 wrong_answer (cop ruling 2026-09-23): each admission condition deleted, its must-RED case goes green.
    rmutant wa-apr-earlier    red-model-wrong-answer-apr-earlier  's/^            elif d_apr < d_ref:/            elif False:/'
    rmutant wa-oracle-agrees  red-model-wrong-answer-oracle-agrees 's/^            elif d_ref is None:/            elif False:/'
    rmutant wa-cuda-missing   red-model-wrong-answer-cuda-leg-missing 's/^            if cuda is None:/            if False:/'
    rmutant wa-cpu-gpu        red-model-wrong-answer-cpu-ne-gpu   's/^            elif d_ref is None:  # the oracle never split: apr CPU\/GPU split is its own$/            elif False:/'
    rmutant wa-cpu-gpu-early  red-model-wrong-answer-cpu-gpu-earlier 's/^            elif d_cg < d_ref:/            elif False:/'
    rmutant gpu-fell-back     red-model-wrong-answer-gpu-fell-back 's/^            if gp:/            if False:/'
    rmutant gpu-fell-back-f9  red-model-gpu-fell-back        's/^            if gp:/            if False:/'
    rmutant wa-llama-correct  red-model-wrong-answer-llama-correct 's/^                if answers(r.get("generated_text"), expect):/                if False:/'
    rmutant wa-control-gone   red-model-wrong-answer-control-missing 's/^            if ca2 is None or co2 is None:/            if False:/'
    rmutant wa-control-wrong  red-model-wrong-answer-control-wrong 's/^            elif not (answers(ca2.get("generated_text"), expect) and answers(co2.get("generated_text"), expect)):/            elif False:/'
    rmutant wa-official       red-model-wrong-answer-not-official 's/^            if a.get("prompt_ids") != ref.get("template_prompt_ids"):/            if False:/'
    rmutant wa-expect         red-model-wrong-answer-no-expect    's/^                if not isinstance(e.get("expect"), str) or not e\["expect"\].strip():/                if False:/'
    rmutant greedy-only-read  green-red-model-wrong-answer-greedy-only-cpu 's/^            greedy_only = R.get("greedy_only") is True and not R.get("cells")$/            greedy_only = False/'
    rmutant axis-on           red-model-axis-not-on          's/^            elif e.get("thinking") != "on":/            elif False:/'
    rmutant residual          red-model-residual             's/^            if resid:/            if False:/'
    rmutant stale             red-model-stale                's/^        if not why:/        if False:/'
    rmutant ticket            red-model-no-ticket            's/^        if not TICKET.search/        if False and not TICKET.search/'
    rmutant no-file           red-model-key-no-file          's/^            if n:$/            if True:/'
    rmutant no-file-unsup     red-unsupported-key-no-file    's/^            if n:$/            if True:/'
    rmutant not-judged-honest red-model-receipt-rejected-not-judged 's/^            if missing:/            if False:/'
    rmutant unsup-ran         red-unsupported-ran            's/^        if be.get("rc") in (0, None) or be.get("ran") is not False:/        if False:/'
    rmutant unsup-stdout      red-unsupported-stdout         's/^        if run.get("generated_bytes") != 0:/        if False:/'
    rmutant unsup-fell-back   red-unsupported-fell-back      's/^        if be.get("fallback") or/        if False and be.get("fallback") or/'
    rmutant unsup-by-name     red-unsupported-no-refusal     's/^        if not isinstance(refusal, str) or needle not in refusal or REFUSAL_CLASS not in refusal:/        if False:/'
    rmutant unsup-arch-header red-unsupported-arch-mismatch  's/^        elif got != arch:/        elif False:/'
    rmutant unsup-has-path    red-unsupported-supported-arch 's/^                    if x.get("present") and x.get("architecture") == arch and green_on_cuda(x):/                    if False:/'
    # The crux-join half of F9/F10: the thinking-OFF axis stays owed, and a proven verdict is what lets a green case pass.
    xmutant greedy-only-skip  green-red-model-wrong-answer-greedy-only-cpu 's/^        if R.get("greedy_only") is True:$/        if False:/'
    xmutant greedy-only-cells red-crux-greedy-only-with-cells 's/^            if R.get("cells"):$/            if False:/'
    xmutant red-model-off-owed red-model-thinking-off-missing 's/^    if red_model and not got:/    if False:/'
    xmutant unsup-cell-named  green-red-unsupported-proven  's/^                    if named == "RED-UNSUPPORTED" and b in ("cuda", "gpu"):/                    if False:/'
    # #3710 ruling 3 / #4022: the scoped-hotfix receipt binding (scripts/lib/ladder_equiv.py), a pure
    # classifier driven by a case table: in scope binds; one stray path, a dependency change in
    # Cargo.lock, or a receipt not at the pinned receipts_at does NOT.
    equiv_table() { # equiv_table <lib dir> -> 0 when every row lands, 1 otherwise
      python3 - "$1" <<'EQ'
import sys; sys.path.insert(0, sys.argv[1]); import ladder_equiv as L
A, B = "a" * 40, "b" * 40
S = {"receipts_at": A, "paths": ["crates/aprender-mcp/**", "crates/apr-cli/Cargo.toml", "Cargo.lock", "docs/**"], "ruling": "r3"}
lk = '[[package]]\nname = "apr-cli"\nversion = "0.69.1"\n\n[[package]]\nname = "serde"\nversion = "1.0.1"\nsource = "registry+x"\nchecksum = "c"\n'
rows = [
  ("in-scope hotfix binds", ("hotfix",), (A, B, ["crates/aprender-mcp/src/lib.rs", "docs/r.md", "crates/apr-cli/Cargo.toml"], S)),
  ("a stray path is NOT equivalent", (None, "outside its scope"), (A, B, ["crates/aprender-mcp/src/lib.rs", "crates/aprender-serve/src/x.rs"], S)),
  ("a receipt not at receipts_at is NOT equivalent", (None, "pinned receipts_at"), ("c" * 40, B, ["docs/r.md"], S)),
  ("a workspace version bump in Cargo.lock binds", ("hotfix",), (A, B, ["Cargo.lock"], S, lk, lk.replace('"0.69.1"', '"0.69.2"'))),
  ("an external dependency change in Cargo.lock is NOT equivalent", (None, "changes dependencies"), (A, B, ["Cargo.lock"], S, lk, lk.replace("1.0.1", "1.0.2"))),
  ("evidence-only binds without the scope", ("evidence",), ("c" * 40, B, ["evidence/x.json"], S)),
]
# #4022 OPERATOR OVERRIDE, pinned to EXACTLY one lock delta: aprender-mcp drops its `anyhow` edge.
def lock(mcp_deps, cli_deps=("aprender-mcp",), extra=""):
    q = lambda ds: ", ".join('"%s"' % d for d in ds)
    return ('[[package]]\nname = "anyhow"\nversion = "1.0.0"\nsource = "registry+x"\nchecksum = "a"\n\n'
            '[[package]]\nname = "serde"\nversion = "1.0.1"\nsource = "registry+x"\nchecksum = "c"\n\n'
            '[[package]]\nname = "aprender-mcp"\nversion = "0.69.1"\ndependencies = [%s]\n\n'
            '[[package]]\nname = "apr-cli"\nversion = "0.69.1"\ndependencies = [%s]\n%s' % (q(mcp_deps), q(cli_deps), extra))
BASE = lock(["anyhow", "serde"], ("aprender-mcp", "anyhow"))
PINNED = lock(["serde"], ("aprender-mcp", "anyhow"))
_, dsha = L.lock_delta(BASE, PINNED)
SO = dict(S, lock_overrides=[{"delta_sha256": dsha, "ticket": "#4022", "date": "2026-09-23", "ruling": "q"}])
rows += [
  ("the pinned lock delta binds as OPERATOR OVERRIDE", ("override", "OPERATOR OVERRIDE (#4022 lock delta)"), (A, B, ["Cargo.lock"], SO, BASE, PINNED)),
  ("an ADDED edge is not the pinned delta: RED", (None, "no operator override pinned"), (A, B, ["Cargo.lock"], SO, BASE, lock(["serde", "anyhow", "extra"], ("aprender-mcp", "anyhow")))),
  ("a NEW package is not the pinned delta: RED", (None, "no operator override pinned"), (A, B, ["Cargo.lock"], SO, BASE, lock(["serde"], ("aprender-mcp", "anyhow"), '\n[[package]]\nname = "nix"\nversion = "0.29.0"\nsource = "registry+x"\nchecksum = "n"\n'))),
  ("ANOTHER crate's edges change too: RED", (None, "no operator override pinned"), (A, B, ["Cargo.lock"], SO, BASE, lock(["serde"], ("aprender-mcp",)))),
  ("a DIFFERENT removal (changed delta hash): RED", (None, "no operator override pinned"), (A, B, ["Cargo.lock"], SO, BASE, lock(["anyhow"], ("aprender-mcp", "anyhow")))),
  ("the same delta with no override pinned stays RED", (None, "no operator override pinned"), (A, B, ["Cargo.lock"], S, BASE, PINNED)),
  ("an override without its recorded ruling is not an override: RED", (None, "no operator override pinned"),
   (A, B, ["Cargo.lock"], dict(S, lock_overrides=[{"delta_sha256": dsha, "ticket": "#4022", "date": "2026-09-23"}]), BASE, PINNED)),
]
# #4032 (X2 = X + one commit): its two files EXACTLY, never a directory glob.
S2 = dict(S, paths=S["paths"] + ["crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs",
                                  "crates/aprender-serve/src/gguf/inference/forward/qwen35_known_issue.rs"])
rows += [
  ("X2's two #4032 files bind with the #4022 ones", ("hotfix",), (A, B, ["crates/aprender-mcp/src/lib.rs",
      "crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs",
      "crates/aprender-serve/src/gguf/inference/forward/qwen35_known_issue.rs"], S2)),
  ("a THIRD aprender-serve file beside #4032's is RED", (None, "outside its scope"), (A, B, [
      "crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs",
      "crates/aprender-serve/src/gguf/inference/forward/forward_qwen3.rs"], S2)),
]
bad = 0
for name, want, args in rows:
    kind, proof = L.classify(*args)
    ok = kind == want[0] and (len(want) == 1 or want[1] in proof)
    print(("ok    equiv " if ok else "FAIL  equiv ") + name + ("" if ok else " -> %r %r" % (kind, proof)))
    bad |= not ok
sys.exit(bad)
EQ
    }
    if equiv_table scripts/lib; then printf 'ok    equiv: the scoped-hotfix table lands on the shipped classifier\n'
    else equiv_table scripts/lib; bad=$((bad+1)); fi
    emutant() { # emutant <label> <sed deleting the rule>
      local md="$mdir/e-$1"; mkdir -p "$md"
      sed "$2" scripts/lib/ladder_equiv.py > "$md/ladder_equiv.py"
      if cmp -s scripts/lib/ladder_equiv.py "$md/ladder_equiv.py"; then echo "FAIL  equiv mutant $1 did not apply"; bad=$((bad+1)); return; fi
      if equiv_table "$md" > /dev/null 2>&1; then echo "FAIL  equiv mutant $1 SURVIVED the table"; bad=$((bad+1))
      else printf 'ok    equiv mutant %-16s killed by the table\n' "$1"; fi
    }
    emutant stray-allowed  's/^    if stray:$/    if False:/'
    emutant deps-allowed   's/^        if why:$/        if False:/'
    emutant any-receipt    's/^    if receipt_sha != scope.get("receipts_at"):$/    if False:/'
    emutant override-any-delta 's/o.get("delta_sha256") == delta_sha/True/'
    emutant override-no-ruling 's/and all(str(o.get(k) or "").strip() for k in ("ticket", "ruling", "date"))/and True/'
    # 0.69.1 OPERATOR EMERGENCY SCOPE (scripts/lib/crux_smoke_scope.py): a hermetic table over generated
    # receipts. Every must-RED names its reason; the green row is the control.
    smoke_table() { # smoke_table <lib dir> -> 0 when every row lands
      python3 - "$1" <<'SM'
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1]); import crux_smoke_scope as C
CUT = "d" * 40
# the checkout's `git rev-parse` stand-in: only the cut's own short sha resolves
C.model_ladder_crux._git_resolve = lambda short: CUT if CUT.startswith(short) else ("e" * 40 if ("e" * 40).startswith(short) else None)
SH = ["1" * 64, "2" * 64, "3" * 64]
L = {"emergency_scopes": [{"name": "crux-smoke", "release": "0.69.1", "date": "2026-09-23", "quote": "q",
                            "hosts": ["lambda", "gx10"], "thinking": ["off", "on"]}],
     "release_gate": {"from": "0.70.0", "ruling": "r", "crux_smoke": {"hosts": ["lambda", "gx10"], "thinking": ["off", "on"]}}}
# 2 = SH[1] is admitted with thinking OFF only (its ON leg loops at greedy, the real 2B's shape)
ADMIT = {s: {"off": ["ctl"], "on": ["ctl"]} for s in SH}
ADMIT[SH[1]] = {"off": ["ctl"], "on": []}
def build(d, hosts=("lambda", "gx10"), drop=None, ctl="GREEN", noctl=False, sha=CUT, cell="GREEN", admit=None, dropmode=None, onlymode=None,
          real=None, cellver=None, meta=False, junk=False, controlonly=False):
    # real=<version line>: the shape crux_inference_dogfood.sh ACTUALLY writes (copied from lambda's X2 shard
    # evidence/crux/0.69.1/d8a6df53a/shards/00fe7986ff5f-off/lambda-gpu.json): no apr.sha, a SHORT harness.sha,
    # and the binary named only by `apr --version`'s line, repeated in every cell's engines.apr.version.
    json.dump({"schema": "crux-prompt-certification/v1", "admitted_by_sha": {s: ["ctl"] for s in SH},
               "admitted_by_sha_thinking": admit or ADMIT}, open(os.path.join(d, "prompt-certification.json"), "w"))
    for h in hosts:
        cells = []
        for s in SH:
            if drop == (h, s):
                continue
            for t in (onlymode.get((h, s), ("off", "on")) if onlymode else ("off", "on")):
                if dropmode == (h, s, t):
                    continue
                cells.append({"key": {"model_sha256": s, "host": h, "thinking": t, "verb": "run"},
                              "verdict": cell if (h, s, t) == ("lambda", "3" * 64, "on") else "GREEN", "positive_control": False,
                              "engines": {"apr": {"answered": True, "version": (cellver or {}).get((h, s, t), real)}} if real else {}})
                if controlonly:
                    cells[-1]["positive_control"] = True
                elif not noctl:
                    cells.append({"key": {"model_sha256": s, "host": h, "thinking": t, "verb": "run"}, "verdict": ctl, "positive_control": True})
        json.dump({"schema": "crux-inference-receipt/v1", "host": h, "backend": "gpu",
                   **({"apr": {"version_line": real}, "harness": {"sha": real.split("(")[-1].rstrip(")")[:9],
                       "driver": "scripts/crux_inference_dogfood.sh"}} if real else {"apr": {"sha": sha}}),
                   "cells": cells, "summary": {"verdict": "PASS"}}, open(os.path.join(d, h + "-gpu.json"), "w"))
        if meta:   # crux_sweep_shards.sh's merge meta, real keys: no schema, no cells
            json.dump({"version": "0.69.1", "host": h, "backend": "gpu", "apr": {"version_line": "apr 0.69.1 (ddddddddd)"},
                       "harness": {"sha": "ddddddddd"}, "models": []}, open(os.path.join(d, h + "-gpu.meta.json"), "w"))
    if junk:
        json.dump({"note": "not a receipt"}, open(os.path.join(d, "stray.json"), "w"))
rows = [
  ("RELEASE GATE: the normal smoke from 0.70.0 on, no per-release entry", False, "RELEASE GATE (normal, #4045)", {"_scope": "release"}, "0.70.0"),
  ("RELEASE GATE: a later release uses it too", False, "RELEASE GATE (normal, #4045)", {"_scope": "release"}, "0.71.3"),
  ("RELEASE GATE: before its `from` it is refused", True, "the normal release gate applies from 0.70.0", {"_scope": "release"}, "0.69.2"),
  ("RELEASE GATE: its smoke rules are the same (a host missing is RED)", True, "host gx10 has no CRUX receipt", {"_scope": "release", "hosts": ("lambda",)}, "0.70.0"),
  ("RELEASE GATE: a release_gate without its ruling is refused", True, "ladder.release_gate is missing ['ruling']", {"_scope": "release", "_noruling": True}, "0.70.0"),
  ("the sweep's merge meta beside a receipt is skipped by name", False, "lambda-gpu.meta.json is the sweep's merge meta", {"meta": True}, "0.69.1"),
  ("any OTHER non-receipt beside the receipts still FAILs", True, "stray.json is not a crux-inference-receipt/v1", {"meta": True, "junk": True}, "0.69.1"),
  ("a control-only smoke says so instead of claiming a separate control check", False, "(control-only smoke)", {"controlonly": True}, "0.69.1"),
  ("green: both hosts, 3 certified models x off/on, controls GREEN", False, "OPERATOR EMERGENCY SCOPE: CRUX smoke only", {}, "0.69.1"),
  ("a host missing is RED", True, "host gx10 has no CRUX receipt", {"hosts": ("lambda",)}, "0.69.1"),
  ("a certified model missing on a host is RED", True, "no CRUX cell", {"drop": ("gx10", "2" * 64)}, "0.69.1"),
  ("the positive control RED is RED", True, "not GREEN", {"ctl": "RED"}, "0.69.1"),
  ("a non-control cell RED beside a GREEN control is RED", True, "1 of 2 cell(s) not GREEN", {"cell": "RED"}, "0.69.1"),
  ("no positive control is RED", True, "positive control is missing", {"noctl": True}, "0.69.1"),
  ("a receipt at a sha other than the cut is RED", True, "not the release binary", {"sha": "e" * 40}, "0.69.1"),
  ("the scope used for another release is RED", True, "for release 0.69.1 ONLY", {}, "0.70.0"),
  ("an ADMITTED mode missing on a host is RED", True, "thinking=off: no CRUX cell", {"dropmode": ("gx10", "2" * 64, "off")}, "0.69.1"),
  ("a certified model with ZERO admitted modes is RED", True, "has NO admitted thinking mode",
   {"admit": dict(ADMIT, **{"3" * 64: {"off": [], "on": []}})}, "0.69.1"),
  ("an UNADMITTED mode's green cells never count toward the pass", True, "thinking=off: no CRUX cell",
   {"onlymode": {("lambda", "2" * 64): ("on",)}}, "0.69.1"),
  ("a REAL-shaped receipt (version line only, short harness sha) binds to the cut", False,
   "OPERATOR EMERGENCY SCOPE: CRUX smoke only", {"real": "apr 0.69.1 (ddddddddd)"}, "0.69.1"),
  ("a real-shaped receipt from another binary is RED", True, "not the release binary", {"real": "apr 0.69.1 (eeeeeeeee)"}, "0.69.1"),
  ("a real-shaped receipt whose short sha resolves to nothing is RED", True, "not the release binary",
   {"real": "apr 0.69.1 (abcdef012)"}, "0.69.1"),
  ("a DIRTY binary's version line binds to nothing and is RED", True, "not the release binary",
   {"real": "apr 0.69.1 (ddddddddd-dirty)"}, "0.69.1"),
  ("a cell measured by a different binary unbinds the whole receipt", True, "not the release binary",
   {"real": "apr 0.69.1 (ddddddddd)", "cellver": {("gx10", "1" * 64, "on"): "apr 0.69.1 (eeeeeeeee)"}}, "0.69.1"),
  ("the admitted matrix is printed", False, "2222222222 -> off;", {}, "0.69.1"),
  ("a mode the certification does NOT admit needs no cells (the real 2B shape: OFF only)", False,
   "ok    gx10 certified model 222222222222 thinking=off",
   {"onlymode": {("lambda", "2" * 64): ("off",), ("gx10", "2" * 64): ("off",)}}, "0.69.1"),
]
bad = 0
for name, want_fail, needle, kw, version in rows:
    with tempfile.TemporaryDirectory() as d:
        kw = dict(kw); scope = kw.pop("_scope", "crux-smoke")
        LL = L if not kw.pop("_noruling", False) else dict(L, release_gate={k: v for k, v in L["release_gate"].items() if k != "ruling"})
        build(d, **kw)
        lines = []
        got = C.judge(LL, version, d, os.path.join(d, "prompt-certification.json"), CUT, scope, lines.append)
        ok = got == want_fail and any(needle in ln for ln in lines)
        print(("ok    smoke " if ok else "FAIL  smoke ") + name + ("" if ok else " -> failed=%s %s%s" % (got, "[%d earlier lines dropped] " % (len(lines) - 3) if len(lines) > 3 else "", lines[-3:])))
        bad |= not ok
sys.exit(bad)
SM
    }
    if smoke_table scripts/lib; then printf 'ok    smoke: the emergency-scope table lands on the shipped module\n'
    else smoke_table scripts/lib; bad=$((bad+1)); fi
    smutant() { # smutant <label> <sed deleting the rule>
      local md="$mdir/s-$1"; mkdir -p "$md"
      cp scripts/lib/model_ladder_crux.py "$md/"
      sed "$2" scripts/lib/crux_smoke_scope.py > "$md/crux_smoke_scope.py"
      if cmp -s scripts/lib/crux_smoke_scope.py "$md/crux_smoke_scope.py"; then echo "FAIL  smoke mutant $1 did not apply"; bad=$((bad+1)); return; fi
      if smoke_table "$md" > /dev/null 2>&1; then echo "FAIL  smoke mutant $1 SURVIVED the table"; bad=$((bad+1))
      else printf 'ok    smoke mutant %-18s killed by the table\n' "$1"; fi
    }
    smutant host-optional     's/^        if h not in seen_hosts:$/        if False:/'
    smutant model-optional    's/^                if not got:$/                if False:/'
    smutant red-cells-ok      's/^                if red:$/                if False:/'
    smutant control-optional  's/^                elif not ctl or any(v != "GREEN" for v in ctl):$/                elif False:/'
    smutant any-sha           's/^        if asha != cut_sha:$/        if False:/'
    smutant any-release       's/^    if str(version) != str(entry\["release"\]):$/    if False:/'
    smutant zero-modes-ok     's/^        if not m:$/        if False:/'
    smutant meta-not-skipped  's/^                out(f"note  {os.path.basename(f)} is the sweep.s merge meta, not a receipt -- skipped"); continue$/                pass/'
    smutant any-json-skipped  's/^            if os.path.basename(f).endswith(".meta.json") and isinstance(R, dict) and "schema" not in R and "cells" not in R:$/            if True:/'
    smutant control-only-hidden 's/^                elif all(pc for _, pc in got):$/                elif False:/'
    smutant release-before-from 's/^        if frm is None or cur is None or cur < frm:$/        if frm is None or cur is None:/'
    smutant release-unrecorded 's/^        missing = \[k for k in ("from", "ruling") if not g.get(k)\]/        missing = [k for k in ("from",) if not g.get(k)]/'
    smutant all-modes-counted 's/^            for mode in matrix\[sha\]:$/            for mode in ("off", "on"):/'
    vmutant() { # vmutant <label> <sed deleting a binding rule in model_ladder_crux.apr_sha_of> -- the smoke table must go RED
      local md="$mdir/v-$1"; mkdir -p "$md"
      cp scripts/lib/crux_smoke_scope.py "$md/"
      sed "$2" scripts/lib/model_ladder_crux.py > "$md/model_ladder_crux.py"
      if cmp -s scripts/lib/model_ladder_crux.py "$md/model_ladder_crux.py"; then echo "FAIL  binding mutant $1 did not apply"; bad=$((bad+1)); return; fi
      if smoke_table "$md" > /dev/null 2>&1; then echo "FAIL  binding mutant $1 SURVIVED the table"; bad=$((bad+1))
      else printf 'ok    binding mutant %-16s killed by the table\n' "$1"; fi
    }
    vmutant version-line-unread 's/^    if not m:$/    if True:/'
    vmutant short-sha-unresolved 's/^    return (resolve or _git_resolve)(m.group(1))$/    return m.group(1)/'
    vmutant dirty-accepted      's/(\[0-9a-f\]{7,40})\\)\$/([0-9a-f]{7,40})/'
    vmutant mixed-cells-ok      's/^        if v is not None and (not isinstance(v, str) or v.strip() != line.strip()):$/        if False:/'
    # #4037 CARRY-FORWARD, wired: equiv_lines on a scratch cargo workspace (a root facade with an `apr` bin
    # and a path dependency `core`). The table drives THIS script (`--equiv-only`), so a mutant of the
    # wiring is a sed copy of the script; a mutant of the rule is a copy of ladder_carry.py.
    carry_wiring() { # carry_wiring <script> <lib dir> -> 0 when every row lands
      local r; r=$(mktemp -d) || return 1
      python3 - "$r" "$1" "$2" "$PWD/scripts/lib" <<'CW'
import json, os, shutil, subprocess, sys
r, script, lib, reallib = sys.argv[1:5]
def w(p, t):
    os.makedirs(os.path.dirname(os.path.join(r, p)) or r, exist_ok=True); open(os.path.join(r, p), "w").write(t)
def git(*a):
    return subprocess.run(["git", "-C", r, "-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()
w("Cargo.toml", '[workspace]\nmembers = ["core"]\n[package]\nname = "facade"\nversion = "0.1.0"\nedition = "2021"\n'
                '[dependencies]\ncore = { path = "core" }\n[[bin]]\nname = "apr"\npath = "src/bin/apr.rs"\n')
w("src/bin/apr.rs", "fn main() {}\n"); w("core/Cargo.toml", '[package]\nname = "core"\nversion = "0.1.0"\nedition = "2021"\n')
w("core/src/lib.rs", "pub fn f() {}\n"); w("docs/a.md", "a\n")
w("contracts/ladder.yaml", "ladder:\n  hotfix_scope: {}\n")
git("init", "-q"); git("add", "-A"); git("commit", "-qm", "A"); A = git("rev-parse", "HEAD")
w("docs/a.md", "b\n"); git("commit", "-qam", "docs"); DOCS = git("rev-parse", "HEAD")
git("checkout", "-q", A); w("core/src/lib.rs", "pub fn f() { let _ = 1; }\n"); git("commit", "-qam", "core"); CORE = git("rev-parse", "HEAD")
git("checkout", "-q", A); w("docs/a.md", "c\n"); git("commit", "-qam", "A2"); A2 = git("rev-parse", "HEAD")
w("docs/a.md", "d\n"); git("commit", "-qam", "docs2"); DOCS2 = git("rev-parse", "HEAD")
# a release's version bump: every workspace version moves, nothing else (#4040)
git("checkout", "-q", A)
for f in ("Cargo.toml", "core/Cargo.toml"):
    p = os.path.join(r, f); t = open(p).read(); open(p, "w").write(t.replace('version = "0.1.0"', 'version = "0.2.0"'))
git("commit", "-qam", "bump"); BUMP = git("rev-parse", "HEAD")
# a bump that ALSO adds a dependency to the apr facade is not a version-only diff
p = os.path.join(r, "Cargo.toml"); t = open(p).read(); open(p, "w").write(t.replace('[dependencies]\n', '[dependencies]\nlibc = "0.2"\n'))
assert "libc" in open(p).read() and 'version = "0.2.0"' in open(p).read(), "the scratch bump fixtures did not apply"
git("commit", "-qam", "bump+dep"); BUMPDEP = git("rev-parse", "HEAD")
# the gate's libs, untracked in the scratch repo (never part of any diff); the rule under test from `lib`
os.makedirs(os.path.join(r, "scripts/lib"))
for f in ("ladder_equiv.py", "model_ladder_crux.py"):
    shutil.copy(os.path.join(reallib, f), os.path.join(r, "scripts/lib", f))
shutil.copy(os.path.join(lib, "ladder_carry.py"), os.path.join(r, "scripts/lib/ladder_carry.py"))
def rec(sha_ladder=None, sha_crux=None):
    d = os.path.join(r, "rc"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/receipts"); os.makedirs(d + "/crux")
    if sha_ladder:
        json.dump({"apr_sha": sha_ladder}, open(d + "/receipts/x.json", "w"))
    if sha_crux:   # the REAL producer's shape: only `apr --version`'s line names the binary
        json.dump({"schema": "crux-inference-receipt/v1", "apr": {"version_line": "apr 0.1.0 (%s)" % sha_crux[:9]}},
                  open(d + "/crux/lambda-gpu.json", "w"))
    return d
def lines(d, cut):
    out = subprocess.run(["bash", script, "--equiv-only", "--ladder", "contracts/ladder.yaml", "--receipts", d + "/receipts",
                          "--crux", d + "/crux", "--cut-commit", cut], cwd=r, capture_output=True, text=True,
                         env=dict(os.environ, MODEL_LADDER_ROOT=r)).stdout
    return {ln.split("\t")[0]: ln.split("\t", 1)[1] for ln in out.splitlines() if "\t" in ln}
rows = [
  ("carry-docs-only-diff binds by CARRIED FORWARD", lambda: "CARRIED FORWARD (#4037)" in lines(rec(A), DOCS).get(A, "")),
  ("carry-closure-diff MUST NOT bind (must-RED)", lambda: A not in lines(rec(A), CORE)),
  ("carry-crux-receipt sha enumerated and carried", lambda: "CARRIED FORWARD" in lines(rec(None, A2), DOCS2).get(A2, "")),
  ("carry-unknown-sha never binds", lambda: not lines(rec("f" * 40), DOCS)),
  ("carry-version-bump binds across a version-only bump", lambda: "a workspace version bump only" in lines(rec(A), BUMP).get(A, "")),
  ("carry-bump-with-dep MUST NOT bind", lambda: A not in lines(rec(A), BUMPDEP)),
]
bad = 0
for name, fn in rows:
    ok = fn()
    print(("ok    carry " if ok else "FAIL  carry ") + name)
    bad |= not ok
sys.exit(bad)
CW
      # bashrs SEC011: $r is this case's own mktemp -d.
      # bashrs disable-next-line=SEC011
      local rc=$?; rm -rf -- "$r"; return $rc
    }
    if carry_wiring "$SELF" scripts/lib; then printf 'ok    carry: the #4037 wiring table lands on the shipped script\n'
    else carry_wiring "$SELF" scripts/lib; bad=$((bad+1)); fi
    if python3 scripts/lib/ladder_carry_cases.py --mutants > /dev/null 2>&1; then printf 'ok    carry: ladder_carry_cases.py rows + mutants (#4037)\n'
    else python3 scripts/lib/ladder_carry_cases.py --mutants | grep -E 'FAIL|SURVIVED|ANCHOR'; bad=$((bad+1)); fi
    cwmutant() { # cwmutant <label> <row that must go RED> <sed on the script | -> <sed on ladder_carry.py | ->
      local md="$mdir/cw-$1"; mkdir -p "$md"; cp scripts/lib/ladder_carry.py "$md/"; cp "$SELF" "$md/gate.sh"
      [ "$3" = - ] || sed -i "$3" "$md/gate.sh"
      [ "$4" = - ] || sed -i "$4" "$md/ladder_carry.py"
      if cmp -s "$SELF" "$md/gate.sh" && cmp -s scripts/lib/ladder_carry.py "$md/ladder_carry.py"; then echo "FAIL  carry mutant $1 did not apply"; bad=$((bad+1)); return; fi
      local out; out=$(carry_wiring "$md/gate.sh" "$md")   # never through a pipe: the table's own rc is 1 here
      if printf '%s\n' "$out" | grep -q "^FAIL  carry $2"; then printf 'ok    carry mutant %-18s killed by %s\n' "$1" "$2"
      else echo "FAIL  carry mutant $1 SURVIVED row $2"; bad=$((bad+1)); fi
    }
    cwmutant carry-unwired   carry-docs-only-diff 's/^    if proof=$(python3 scripts\/lib\/ladder_carry.py /    if false \&\& proof=$(python3 scripts\/lib\/ladder_carry.py /' -
    cwmutant crux-unlisted   carry-crux-receipt   's/^for f in glob.glob(sys.argv\[2\] + "\/\*.json"):$/for f in []:/' -
    cwmutant src-test-only   carry-closure-diff   - 's/^TEST_ONLY_DIRS = ("tests", "benches", "examples")$/TEST_ONLY_DIRS = ("tests", "benches", "examples", "src")/'
    cwmutant bump-not-derived carry-version-bump - 's/^        bumped = {p for p in paths if posixpath.basename(p) in ("Cargo.toml", "Cargo.lock")$/        bumped = set() and {p for p in paths if posixpath.basename(p) in ("Cargo.toml", "Cargo.lock")/'
    cwmutant carry-ignores-rc carry-closure-diff   's/^    if proof=$(python3 scripts\/lib\/ladder_carry.py "$PWD" "$s" "$4" 2> \/dev\/null); then$/    if proof=$(python3 scripts\/lib\/ladder_carry.py "$PWD" "$s" "$4" 2> \/dev\/null; true); then/' -
    # #4040: the nightly admission rule (its own table + mutants), and the --nightly wiring: an empty nightly
    # root must refuse BY NAME before any receipt is judged. The wiring mutant is a sed copy of this script.
    if python3 scripts/lib/nightly_admission_cases.py --mutants > "$mdir/nadm.out" 2>&1; then printf 'ok    nightly: nightly_admission_cases.py rows + mutants (#4040)\n'
    else grep -E 'FAIL|SURVIVED|ANCHOR' "$mdir/nadm.out"; bad=$((bad+1)); fi
    nightly_wiring() { # nightly_wiring <script> -> 0 when an empty nightly root is refused by name
      local e out; e=$(mktemp -d)
      out=$(MODEL_LADDER_ROOT="$PWD" bash "$1" --nightly "$e" --cut-commit "$(git rev-parse HEAD)" 2>&1); local rc=$?
      rmdir "$e" 2> /dev/null
      [ "$rc" = 1 ] && grep -q '^FAIL  NIGHTLY lambda: no nightly at all (#4040)' <<< "$out" && grep -q '^RED   no admissible nightly' <<< "$out"
    }
    if bash scripts/check_certify_nightly.sh > "$mdir/cn.out" 2>&1; then printf 'ok    nightly: check_certify_nightly.sh -- the driver table + %s mutants (#4040)\n' "$(grep -c '^ok    mutant' "$mdir/cn.out")"
    else grep -E '^FAIL' "$mdir/cn.out"; bad=$((bad+1)); fi
    # #4045: --scope release without --nightly is refused by name (the smoke alone is not a release)
    out=$(MODEL_LADDER_ROOT="$PWD" bash "$SELF" --scope release --cut-commit "$(git rev-parse HEAD)" 2>&1); r=$?
    if [ "$r" = 2 ] && grep -q '^decline: --scope release needs --nightly' <<< "$out"; then echo "ok    release: --scope release without --nightly declines by name"
    else echo "FAIL  release: --scope release without --nightly did not decline by name (rc $r)"; bad=$((bad+1)); fi
    sed 's/^  if \[ "\$SCOPE" = release \] \&\& \[ -z "\$NIGHTLY_ROOT_DIR" \]; then$/  if false; then/' "$SELF" > "$mdir/rs-mut.sh"
    if cmp -s "$SELF" "$mdir/rs-mut.sh"; then echo "FAIL  release mutant smoke-alone did not apply"; bad=$((bad+1))
    else out=$(MODEL_LADDER_ROOT="$PWD" bash "$mdir/rs-mut.sh" --scope release --cut-commit "$(git rev-parse HEAD)" 2>&1)
      if grep -q '^decline: --scope release needs --nightly' <<< "$out"; then echo "FAIL  release mutant smoke-alone SURVIVED"; bad=$((bad+1))
      else echo "ok    release mutant smoke-alone killed by the decline row"; fi
    fi
    # #4040 POSITIVE CONTROL + must-REDs, end to end through the real gate (quorum lane, Fable: "the green path was
    # never exercised"). A nightly root built from the `green` case's receipts, re-bound to a real commit.
    nightly_e2e() { # nightly_e2e <script> <sha> <drop-cpu 0|1> [case] -> prints the gate's output, returns its rc
      local n c="scripts/lib/model_ladder_cases/${4:-green}" rc; n=$(mktemp -d)
      python3 - "$c" "$n" "$2" "$3" <<'NE'
import json, os, sys, time
c, n, sha, dropcpu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
for h in ("lambda", "gx10"):
    d = os.path.join(n, sha, h); os.makedirs(d + "/ladder"); os.makedirs(d + "/crux")
    R = json.load(open(os.path.join(c, "receipts", h + ".json"))); R["apr_sha"] = sha
    json.dump(R, open(d + "/ladder/%s.json" % h, "w"))
    lanes = {}
    for lane in ("gpu", "cpu"):
        X = json.load(open(os.path.join(c, "crux", "%s-%s.json" % (h, lane))))
        if isinstance(X.get("apr"), dict): X["apr"]["sha"] = sha
        p = d + "/crux/%s-%s.json" % (h, lane)
        if not (dropcpu and lane == "cpu"):
            json.dump(X, open(p, "w"))
        lanes[lane] = {"receipt": p, "verdict": (X.get("summary") or {}).get("verdict")}
    json.dump({"schema": "apr-nightly-certification/v1", "sha": sha, "host": h, "version": "1.2.3", "t_start": time.time() - 3600,
               "t_end": time.time() - 60, "ladder": {"rc": 0, "receipt": d + "/ladder/%s.json" % h}, "crux": {"rc": 0, "lanes": lanes},
               "certification": {"path": os.path.abspath(os.path.join(c, "certification.json"))}, "green": True, "why": []},
              open(d + "/verdict.json", "w"))
NE
      # shellcheck disable=SC2086
      MODEL_LADDER_ROOT="$PWD" bash "$1" --nightly "$n" --cut-commit "$(git rev-parse HEAD)" --ladder "${NE_LADDER:-$c/ladder.yaml}" \
        --ladder-main "${NE_LADDER:-$c/ladder.yaml}" --version 1.2.3 ${NE_EXTRA_ARGS:-} 2>&1; rc=$?
      rm -rf -- "$n"; return "$rc"
    }
    head=$(git rev-parse HEAD)
    out=$(nightly_e2e "$SELF" "$head" 0); r=$?
    if [ "$r" = 0 ] && grep -q '^ok    NIGHTLY gx10: GREEN' <<< "$out" && grep -q '^ok    every required rung green on every required host' <<< "$out"; then
      echo "ok    nightly e2e-green -- a GREEN nightly at the cut is admitted and the release judges green on it (positive control)"
    else echo "FAIL  nightly e2e-green -- rc $r: $(grep -E '^(FAIL|RED)' <<< "$out" | head -3 | tr '\n' ' ')"; bad=$((bad+1)); fi
    out=$(nightly_e2e "$SELF" "$head" 1); r=$?
    if [ "$r" = 1 ] && grep -q "says green but its CRUX cpu receipt is missing or not PASS" <<< "$out"; then
      echo "ok    nightly e2e-cpu-lane-missing -- a nightly without the cpu CRUX lane is refused by name"
    else echo "FAIL  nightly e2e-cpu-lane-missing -- rc $r"; bad=$((bad+1)); fi
    # the same path with the #4051 timing rule ON (the case ladders carry timing.required_from): a stamped nightly is
    # admitted and judged green; an unstamped one is RED by name through --nightly (degraded quorum, Fable)
    out=$(nightly_e2e "$SELF" "$head" 0 green-timing); r=$?
    if [ "$r" = 0 ] && grep -q '^ok    every required rung green on every required host' <<< "$out"; then
      echo "ok    nightly e2e-timing-green -- a stamped nightly is admitted and judged green with the timing rule on"
    else echo "FAIL  nightly e2e-timing-green -- rc $r: $(grep -E '^(FAIL|RED)' <<< "$out" | head -3 | tr '\n' ' ')"; bad=$((bad+1)); fi
    out=$(nightly_e2e "$SELF" "$head" 0 red-timing-missing); r=$?
    if [ "$r" = 1 ] && grep -q "TIMING the row records no \`timing\`" <<< "$out"; then
      echo "ok    nightly e2e-timing-unstamped -- an unstamped nightly is RED by name through --nightly"
    else echo "FAIL  nightly e2e-timing-unstamped -- rc $r"; bad=$((bad+1)); fi
    anc=$(git log -n 1 --format=%H -- scripts/model_ladder.sh); anc=$(git rev-parse "$anc^" 2> /dev/null)
    if [ -n "$anc" ]; then
      out=$(nightly_e2e "$SELF" "$anc" 0); r=$?
      if [ "$r" = 1 ] && grep -q 'STALE BY SHA' <<< "$out"; then
        echo "ok    nightly e2e-ancestor-stale -- a nightly at an ancestor whose delta touches the measurement (model_ladder.sh) is STALE, never carried"
      else echo "FAIL  nightly e2e-ancestor-stale -- rc $r"; bad=$((bad+1)); fi
    fi
    sed 's/^  RECEIPT_DIR="\$NIGHTLY_OUT\/receipts"; CRUX_DIR="\$NIGHTLY_OUT\/crux"$/  :/' "$SELF" > "$mdir/ne-mut.sh"
    if cmp -s "$SELF" "$mdir/ne-mut.sh"; then echo "FAIL  nightly mutant dirs-unlinked did not apply"; bad=$((bad+1))
    else out=$(nightly_e2e "$mdir/ne-mut.sh" "$head" 0); r=$?
      if [ "$r" = 0 ]; then echo "FAIL  nightly mutant dirs-unlinked SURVIVED e2e-green"; bad=$((bad+1))
      else echo "ok    nightly mutant dirs-unlinked killed by e2e-green"; fi
    fi
    # #4045 end to end: --scope release = CRUX smoke on the release binary AND the admitted nightly. A GREEN smoke
    # with a GREEN nightly releases; a RED smoke with the SAME green nightly does not (the nightly cannot stand in).
    release_e2e() { # release_e2e <script> <GREEN|RED> -> the gate's output, its rc
      local sm lad rc; sm=$(mktemp -d); lad="$sm/ladder.yaml"
      python3 - "scripts/lib/model_ladder_cases/green/ladder.yaml" "$lad" "$sm" "$(git rev-parse HEAD)" "$2" <<'RE'
import json, os, sys, yaml
src, lad, sm, cut, want = sys.argv[1:6]
L = yaml.safe_load(open(src))
L["ladder"]["release_gate"] = {"from": "1.0.0", "ruling": "fixture", "crux_smoke": {"hosts": ["lambda", "gx10"], "thinking": ["off", "on"]}}
yaml.safe_dump(L, open(lad, "w"))
SH = ["0" * 64, "1" * 64]
json.dump({"schema": "crux-prompt-certification/v1", "admitted_by_sha": {s: ["fixture-control"] for s in SH},
           "admitted_by_sha_thinking": {s: {"off": ["fixture-control"], "on": []} for s in SH}},
          open(os.path.join(sm, "prompt-certification.json"), "w"))
for h in ("lambda", "gx10"):
    cells = [{"key": {"model_sha256": s, "host": h, "thinking": "off", "verb": "run"}, "positive_control": True,
              "verdict": "RED" if (want == "RED" and h == "gx10" and s == SH[1]) else "GREEN"} for s in SH]
    json.dump({"schema": "crux-inference-receipt/v1", "host": h, "backend": "gpu", "apr": {"sha": cut}, "cells": cells,
               "summary": {"verdict": "PASS"}}, open(os.path.join(sm, h + "-gpu.json"), "w"))
RE
      NE_LADDER="$lad" NE_EXTRA_ARGS="--scope release --crux $sm" nightly_e2e "$1" "$(git rev-parse HEAD)" 0; rc=$?
      # bashrs SEC011: $sm is this case's own mktemp -d.
      # bashrs disable-next-line=SEC011
      rm -rf -- "$sm"; return "$rc"
    }
    out=$(release_e2e "$SELF" GREEN); r=$?
    if [ "$r" = 0 ] && grep -q '^ok    RELEASE GATE (normal, #4045): CRUX smoke on the release binary GREEN' <<< "$out"; then
      echo "ok    release e2e-release-green -- smoke GREEN + nightly GREEN releases"
    else echo "FAIL  release e2e-release-green -- rc $r: $(grep -E '^(FAIL|RED|decline)' <<< "$out" | head -3 | tr '\n' ' ')"; bad=$((bad+1)); fi
    out=$(release_e2e "$SELF" RED); r=$?
    if [ "$r" = 1 ] && grep -q "^RED   RELEASE GATE: the release binary's CRUX smoke is RED" <<< "$out" && grep -q '^ok    NIGHTLY gx10: GREEN' <<< "$out"; then
      echo "ok    release e2e-smoke-red-nightly-green -- a RED smoke is not rescued by a GREEN nightly"
    else echo "FAIL  release e2e-smoke-red-nightly-green -- rc $r"; bad=$((bad+1)); fi
    sed 's/^if \[ -n "\$SMOKE_RC" \] \&\& \[ "\$SMOKE_RC" != 0 \]; then$/if false; then/' "$SELF" > "$mdir/sr-mut.sh"
    if cmp -s "$SELF" "$mdir/sr-mut.sh"; then echo "FAIL  release mutant smoke-folded-out did not apply"; bad=$((bad+1))
    else out=$(release_e2e "$mdir/sr-mut.sh" RED); r=$?
      # killed = the row's expectation (rc 1 naming the RED smoke) no longer holds under the mutant
      if [ "$r" = 1 ] && grep -q "^RED   RELEASE GATE: the release binary's CRUX smoke is RED" <<< "$out"; then
        echo "FAIL  release mutant smoke-folded-out SURVIVED e2e-smoke-red-nightly-green"; bad=$((bad+1))
      else echo "ok    release mutant smoke-folded-out killed by e2e-smoke-red-nightly-green (rc $r)"; fi
    fi
    if nightly_wiring "$SELF"; then echo "ok    nightly: --nightly with no nightly refuses by name, before any receipt is judged"
    else echo "FAIL  nightly: --nightly with an empty root did not refuse by name"; bad=$((bad+1)); fi
    sed 's/^  if ! python3 scripts\/lib\/nightly_admission.py /  if false \&\& python3 scripts\/lib\/nightly_admission.py /' "$SELF" > "$mdir/nw-mut.sh"
    if cmp -s "$SELF" "$mdir/nw-mut.sh"; then echo "FAIL  nightly mutant admission-unwired did not apply"; bad=$((bad+1))
    elif nightly_wiring "$mdir/nw-mut.sh"; then echo "FAIL  nightly mutant admission-unwired SURVIVED"; bad=$((bad+1))
    else echo "ok    nightly mutant admission-unwired killed by the empty-root refusal"; fi
    # #3710 ruling 1: CRUX coverage scoped to the certified models (model_ladder_crux.py).
    xmutant uncertified-owes-crux green-uncertified-no-crux 's/^                    elif certified is not None and sha not in certified:$/                    elif False:/'
    xmutant no-cert-relaxes   red-certification-missing 's/^        return None, True$/        return set(), False/'
    xmutant certified-unheld  red-certified-not-held    's/^        if model_sha not in held:$/        if False:/'
    xmutant cert-read-as-receipt green-cert-beside-crux-receipts 's/                   if not os.path.basename(f).startswith("prompt-certification")) if crux_dir else \[\]/                   ) if crux_dir else []/'
    xmutant crux-timing-unchecked red-crux-timing-unrequired 's/^        if timing_required and (R.get("timing") or {}).get("required") is not True:$/        if False:/'
    xmutant certified-as-none red-certified-missing-crux 's/^    need = certified is None or bool(held \& certified)$/    need = False; certified = set()/'
    if [ -n "$mdir" ] && [ "$mdir" != "/" ] && [ -d "$mdir" ]; then rm -rf -- "$mdir"; fi
  fi
  echo "self-test: $n case(s), $bad bad"
  [ "$bad" -eq 0 ]; exit $?
fi

# ---------------------------------------------------------------- real run
VERSION="${VERSION_OVERRIDE:-$(cargo metadata --no-deps --offline --format-version 1 --manifest-path Cargo.toml 2>/dev/null | python3 -c '
import json, os, sys
m = json.load(sys.stdin)
root = os.path.realpath("Cargo.toml")
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' 2>/dev/null)}"
[ -n "$VERSION" ] || { echo "decline: the version being cut cannot be resolved"; exit 2; }
[ -n "$RECEIPT_DIR" ] || RECEIPT_DIR="evidence/dogfood/models/$VERSION"
MAIN_LADDER=""
TMP_LADDER=""
_rm_tmp_ladder() { if [ -n "${TMP_LADDER:-}" ] && [ -f "$TMP_LADDER" ]; then rm -f "$TMP_LADDER"; fi; }
trap _rm_tmp_ladder EXIT
if [ -n "$LADDER_MAIN_OVERRIDE" ]; then MAIN_LADDER="$LADDER_MAIN_OVERRIDE"
else
  TMP_LADDER=$(mktemp)
  if git show "origin/main:$LADDER" > "$TMP_LADDER" 2>/dev/null && [ -s "$TMP_LADDER" ]; then MAIN_LADDER="$TMP_LADDER"; fi
fi
printf -- '--- model capability ladder receipts for %s (%s) ---------------------\n' "$VERSION" "$RECEIPT_DIR"
TMP_RUNGS=$(mktemp)
git show "origin/main:evidence/release/context-rungs.json" > "$TMP_RUNGS" 2> /dev/null || : > "$TMP_RUNGS"   # absent at main: the bootstrap
# #3957 F2: the cut. `--cut-commit`, else HEAD. safe.directory because a CI container's checkout
# is owned by another uid and plain rev-parse dies there (#3581). Unresolvable -> the judge declines.
[ -n "$CUT_COMMIT" ] || CUT_COMMIT=$(git -c safe.directory="$PWD" rev-parse HEAD 2> /dev/null || true)
# 0.69.1 OPERATOR EMERGENCY SCOPE (scripts/lib/crux_smoke_scope.py): `--scope crux-smoke` judges CRUX smoke
# receipts from the release binary INSTEAD of the ladder, only for the release its contract entry names.
SMOKE_RC=""
if [ -n "$SCOPE" ]; then
  # #4045: `release` is the NORMAL gate from 0.70.0 -- CRUX smoke on the release binary (below) AND the nightly
  # long certification admitted and bound to the cut (--nightly, further down). Neither alone is a release.
  if [ "$SCOPE" = release ] && [ -z "$NIGHTLY_ROOT_DIR" ]; then
    echo "decline: --scope release needs --nightly <root> -- the normal release gate is CRUX smoke on the release binary AND the nightly long certification (#4045)"
    exit 2
  fi
  [ -n "$CRUX_DIR" ] || CRUX_DIR="evidence/crux/$VERSION"
  python3 -c 'import sys, yaml
sys.path.insert(0, "scripts/lib"); import crux_smoke_scope
L = yaml.safe_load(open(sys.argv[1]))["ladder"]
sys.exit(1 if crux_smoke_scope.judge(L, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], print) else 0)' \
    "$LADDER" "$VERSION" "$CRUX_DIR" "${CRUX_CERT:-$CRUX_DIR/prompt-certification.json}" "$CUT_COMMIT" "$SCOPE"
  rc=$?
  if [ "$SCOPE" = release ]; then
    SMOKE_RC=$rc
    if [ "$rc" = 0 ]; then echo "ok    RELEASE GATE: CRUX smoke on the release binary satisfied -- now the nightly long certification"
    else echo "RED   RELEASE GATE: CRUX smoke on the release binary NOT satisfied (see FAIL rows) -- the nightly is still judged, and cannot rescue it"; fi
    CRUX_DIR=""   # the admitted nightly supplies the long CRUX receipts
  elif [ "$rc" = 0 ]; then
    echo "ok    OPERATOR EMERGENCY SCOPE: CRUX smoke only -- satisfied. The model ladder was NOT run for this release (nightly only); this is not \"every rung green\""
  else
    echo "RED   OPERATOR EMERGENCY SCOPE: CRUX smoke only -- NOT satisfied (see FAIL rows)"; rc=1
  fi
  [ "$SCOPE" = release ] || exit "$rc"
fi
# #4040: --nightly <root> judges the release on last night's long certification (scripts/certify_nightly.sh)
# instead of receipts measured for this cut. Admission (scripts/lib/nightly_admission.py): per REQUIRED host,
# the newest GREEN nightly, at most the contract's ladder.release_gate.nightly.max_age_h hours old, at the cut or an ancestor of it. Its
# receipts then bind to the cut like any other -- through equivalence / the #4037 carry-forward, or STALE.
if [ -n "$NIGHTLY_ROOT_DIR" ]; then
  NIGHTLY_OUT=$(mktemp -d)
  nightly_hosts=$(python3 -c 'import sys, yaml; print(" ".join(h["id"] for h in yaml.safe_load(open(sys.argv[1]))["ladder"]["hosts"] if h.get("required")))' "$LADDER") \
    || { echo "decline: the required hosts cannot be read from $LADDER"; exit 2; }
  # shellcheck disable=SC2086
  if ! python3 scripts/lib/nightly_admission.py "$NIGHTLY_ROOT_DIR" "$CUT_COMMIT" "$NIGHTLY_OUT" $nightly_hosts; then
    echo "RED   no admissible nightly for this cut -- a release is judged on a GREEN nightly within ladder.release_gate.nightly.max_age_h at an ancestor, or re-measured in full (#4040)"
    exit 1
  fi
  RECEIPT_DIR="$NIGHTLY_OUT/receipts"; CRUX_DIR="$NIGHTLY_OUT/crux"
fi
TMP_EQUIV=$(mktemp)
[ -n "$CRUX_DIR" ] || CRUX_DIR="evidence/crux/$VERSION"
equiv_lines "$LADDER" "$RECEIPT_DIR" "$CRUX_DIR" "$CUT_COMMIT" "$TMP_EQUIV" > "$TMP_EQUIV"
TMP_OUT=$(mktemp)
judge "$LADDER" "$MAIN_LADDER" "$RECEIPT_DIR" "$VERSION" evidence/release/context-rungs.json "$TMP_RUNGS" "$CUT_COMMIT" "$TMP_EQUIV" "$CRUX_DIR" "$CRUX_DIR/prompt-certification.json" > "$TMP_OUT"; rc=$?
cat "$TMP_OUT"
# #3957 F9/F10: a run whose only non-green cells are re-proven RED-MODEL / RED-UNSUPPORTED exits 0,
# and must not then claim "every required rung green". Read from the file, never through a pipe.
named_red=0; grep -q '^RED   [0-9]* RED-MODEL' "$TMP_OUT" && named_red=1
# #3907: a shipped KNOWN-RED is likewise RED, never green -- the summary must not say otherwise.
grep -q '^KNOWN-RED [0-9]* row(s) ship RED' "$TMP_OUT" && named_red=1
# #4022: a receipt bound to the cut by an OPERATOR OVERRIDE is not "every rung green" either.
override=0; grep -q 'OPERATOR OVERRIDE (' "$TMP_OUT" && override=1
rm -f "$TMP_OUT"
rm -f "$TMP_EQUIV" "$TMP_EQUIV.paths" "$TMP_EQUIV.lock_a" "$TMP_EQUIV.lock_b"
[ -n "$TMP_RUNGS" ] && [ -f "$TMP_RUNGS" ] && rm -f "$TMP_RUNGS"
# The producer that writes these receipts must not bypass the fleet GPU lock (#3712): RED, not a decline.
# #3957 F1: exit 2 now also means DEFER, so a raw GPU call must not hide behind it -- always RED.
if ! lock_audit scripts/model_ladder.sh; then rc=1; fi
# #4045: under --scope release a RED smoke fails the release whatever the nightly said.
if [ -n "$SMOKE_RC" ] && [ "$SMOKE_RC" != 0 ]; then
  echo "RED   RELEASE GATE: the release binary's CRUX smoke is RED -- the nightly cannot stand in for the binary being released (#4045)"
  rc=1
fi
case $rc in
  0) [ "$SCOPE" = release ] && echo "ok    RELEASE GATE (normal, #4045): CRUX smoke on the release binary GREEN, and the nightly long certification bound to the cut"
     if [ "$named_red" = 1 ]; then
       echo "ok    no blocking cell: every required rung is green, or RED-MODEL / RED-UNSUPPORTED re-proven on this sweep, or a KNOWN-RED shipping with its ticket (counted RED above, never green)"
     elif [ "$override" = 1 ]; then
       echo "ok    every required rung green -- against receipts bound by an OPERATOR OVERRIDE (#4022 lock delta, see above), not by the default rule"
     else
       echo "ok    every required rung green on every required host"
     fi
     [ "$override" = 1 ] && [ "$named_red" = 1 ] && echo "note  receipts bound by OPERATOR OVERRIDE (#4022 lock delta) -- see the binding lines above" ;;
  1) echo "RED   the release claims a capability no receipt proves — see FAIL rows (EPIC #3477)" ;;
  2) echo "NO-GO the verdict is not green: see DEFER / decline lines above (#3957 F1)" ;;
esac
exit $rc
