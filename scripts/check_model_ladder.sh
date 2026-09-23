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
# Exit: 0 green · 1 red · 2 decline (unreadable ladder / no required host) ·
#       self-test: 0 all cases as expected, 1 otherwise.
set -uo pipefail

LADDER="contracts/model-capability-ladder-v1.yaml"
# Overridable so the floor below can be PROVEN against a planted case in a temp dir
# rather than by planting a permanently-failing case in the real table (#3887).
CASES_DIR="${MODEL_LADDER_CASES_DIR:-scripts/lib/model_ladder_cases}"
SELF_TEST=0; ONLY_CASE=""; RECEIPT_DIR=""; LADDER_MAIN_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) SELF_TEST=1; shift ;;
    --case) [ $# -ge 2 ] || { echo "--case needs a value" >&2; exit 2; }; ONLY_CASE="$2"; shift 2 ;;
    --receipts) [ $# -ge 2 ] || { echo "--receipts needs a value" >&2; exit 2; }; RECEIPT_DIR="$2"; shift 2 ;;
    --ladder) [ $# -ge 2 ] || { echo "--ladder needs a value" >&2; exit 2; }; LADDER="$2"; shift 2 ;;
    --ladder-main) [ $# -ge 2 ] || { echo "--ladder-main needs a value" >&2; exit 2; }; LADDER_MAIN_OVERRIDE="$2"; shift 2 ;;
    --version) [ $# -ge 2 ] || { echo "--version needs a value" >&2; exit 2; }; VERSION_OVERRIDE="$2"; shift 2 ;;
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
# judge <ladder> <ladder_at_main_or_empty> <receipt_dir> <version> [context-rungs.json] [its origin/main copy]  → exit 0/1/2
# #3940 anti-vacuity, as a function so the case table can call it directly. A run that
# cannot name the cut it is judging must stop BEFORE reporting on it: inside judge() an
# empty value means "this case did not opt in", so if that state ever reached the real
# path the binding would silently not run.
cut_sha_or_die() { # cut_sha_or_die <resolved-sha>
  if [ -z "${1:-}" ]; then
    echo "FAIL  ladder  cannot determine the cut being judged (git rev-parse HEAD failed) -- the apr_sha binding would not run, and a ladder that cannot name its own cut does not pass vacuously (#3940)"
    return 1
  fi
  return 0
}

judge() {
  python3 - "$1" "$2" "$3" "$4" "${5:-}" "${6:-}" <<'PY'
import fnmatch, json, os, subprocess, sys, yaml
ladder_p, main_p, rdir, version, rungs_p, rungs_main_p = sys.argv[1:7]
sys.path.insert(0, os.environ.get("MODEL_LADDER_CELLS_LIB") or "scripts/lib")  # a mutant copy of the module, in --self-test
import model_ladder_cells
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
    if qa_rc not in (0, None):
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
                elif td not in ("clean", "escalated"):
                    why.append(f"{b}: verb `serve` teardown is {td!r}, which is not one of clean/escalated/failed (#3838)")
    return why
# #3712: no Q4_K rung is optional, and every one claims cuda. The key is refused, not tolerated.
for r in rungs:
    if is_q4k(r) and r.get("required") is not True:
        print(f"FAIL  rung {r['id']} is a Q4_K rung with required: {r.get('required')!r} — no Q4_K model is optional; every one must be green on CUDA (#3712)"); rc = 1
    if is_q4k(r) and "cuda" not in (r.get("backends") or []):
        print(f"FAIL  rung {r['id']} is a Q4_K rung that does not claim cuda — every Q4_K model must be green on CUDA (#3712)"); rc = 1
inv_backends = list((L.get("inventory") or {}).get("backends") or [])
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
# --- #3940: WHICH BINARY PRODUCED THE ROW ---------------------------------------
# This gate already binds two axes of staleness: the model file's sha256 ("a different
# file is a different measurement") and the release version (STALE, below). It has never
# bound the third -- the `apr` that produced the receipt. Every tag decision this cycle
# leaned on "receipt apr_sha == tagged HEAD", and that was cop and operator discipline
# with no mechanism under it. The producer has always written the field; the judge never
# read it.
#
# The version binding cannot cover this: within one release every candidate build reports
# the same `version`. That is the live case -- the capability-mirror fix CHANGED the
# binary (capability.rs `include_str!`s the packaged mirror, so the quant whitelist is
# compiled in) while the ladders measured with the older build. Both say 0.69.1.
#
# WHY THIS IS NOT AN EQUALITY CHECK. A receipt is always produced BEFORE the fold that
# lands it, so `apr_sha == HEAD` is false on every legitimate run -- measured, the two
# committed receipts sit 291 commits behind their own release head. A rule that reds a
# correct run is not a gate (CLAUDE.md: a gate never green on its own target is not a
# gate). So drift is REPORTED always and REFUSED only when the delta touches what the
# binary is compiled from.
CUT_SHA = os.environ.get("LADDER_CUT_SHA", "").strip()
BINARY_PATHS = ("crates/", "src/", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")

def _git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True)
        return r.returncode, r.stdout
    except Exception:
        return 127, ""

def _binary_delta(a, b):
    """Paths changed between two commits that the apr binary is compiled from."""
    code, out = _git("diff", "--name-only", f"{a}..{b}")
    if code != 0:
        return None
    return sorted({p for p in out.split("\n") if p.startswith(BINARY_PATHS)})

# A declaration lives in the CONTRACT, not the receipt: a generated file that declares
# its own exemption is self-certification. Same shape as `withdrawn:` and `deferred:`.
apr_drift_declared = ((L.get("inventory") or {}).get("apr_sha_drift") or "").strip()
apr_drift_used = 0

def judge_apr_sha(hid, R):
    """0 clean, 1 refused. Reports on EVERY run so drift cannot decay into an absence."""
    global apr_drift_used
    if not CUT_SHA:
        # NOT EVALUATED -- and this is not the vacuity hole it looks like. The real
        # invocation refuses BEFORE judging when git cannot name the cut (see the
        # LADDER_CUT_SHA block at the bottom of this file), so an empty value here is
        # reachable only from the case harness, where a case opts in with a `cut_sha`
        # file. The door is held shut by `cut_sha_or_die`, whose two rows in the
        # self-test below refuse an empty cut before judging ever starts.
        return 0
    apr_sha = (R.get("apr_sha") or "").strip()
    if not apr_sha:
        # The script's own :172 rule, applied to a second missing key: "absence scored as
        # conformance is the shape this whole gate exists for." DELIBERATELY after the
        # opt-in check: ordered first, it refused all 63 pre-#3940 fixtures, which carry no
        # such key and never claimed to. The case table caught that, which is what it is for.
        print(f"FAIL  {hid:7} receipt carries no `apr_sha` — which binary measured this is unrecorded, and an unrecorded instrument is not evidence (#3940)")
        return 1
    if _git("cat-file", "-e", apr_sha + "^{commit}")[0] != 0:
        print(f"FAIL  {hid:7} apr_sha {apr_sha[:9]} is not a commit in this repository — the measurement is not of this history (#3940)")
        return 1
    if _git("merge-base", "--is-ancestor", apr_sha, CUT_SHA)[0] != 0:
        print(f"FAIL  {hid:7} apr_sha {apr_sha[:9]} is NOT an ancestor of the cut {CUT_SHA[:9]} — measured on a commit this cut does not contain (#3940)")
        return 1
    if apr_sha == CUT_SHA:
        print(f"ok    {hid:7} apr_sha {apr_sha[:9]} == the cut — the receipt is from this exact binary")
        return 0
    delta = _binary_delta(apr_sha, CUT_SHA)
    count = (_git("rev-list", "--count", f"{apr_sha}..{CUT_SHA}")[1] or "?").strip()
    if delta is None:
        print(f"FAIL  {hid:7} apr_sha {apr_sha[:9]} differs from the cut and the delta could not be computed — an unknown delta is not a small one (#3940)")
        return 1
    if not delta:
        print(f"ok    {hid:7} apr_sha {apr_sha[:9]} is {count} commit(s) behind the cut, none touching what apr is built from")
        return 0
    if apr_drift_declared:
        apr_drift_used += 1
        print(f"ok    {hid:7} apr_sha {apr_sha[:9]} is {count} commit(s) behind and {len(delta)} binary path(s) changed — DECLARED: {apr_drift_declared}")
        return 0
    print(f"FAIL  {hid:7} apr_sha {apr_sha[:9]} is {count} commit(s) behind the cut and {len(delta)} path(s) apr is built from changed since — the row was measured by a DIFFERENT binary (#3940)")
    for pth in delta[:5]:
        print(f"               {pth}")
    if len(delta) > 5:
        print(f"               ... and {len(delta) - 5} more path(s)")
    print(f"               Re-measure at the cut, or declare it in the ladder contract as `inventory.apr_sha_drift` with a reason.")
    return 1

def judge_apr_drift_declaration():
    """#3880's distinction, applied to this declaration. A key that excused nothing is
    not harmless: it reads as coverage, and the next real drift hides behind it."""
    if apr_drift_declared and apr_drift_used == 0:
        print("FAIL  ladder  `inventory.apr_sha_drift` is declared but no receipt needed it — the drift it excused is gone and the key outlived it; remove it (#3940)")
        return 1
    return 0

good = {}  # host id -> a receipt that passed the host-level checks; the cells judge reads only these
for h in hosts:
    f = os.path.join(rdir, f"{h['id']}.json")
    if not os.path.exists(f):
        print(f"FAIL  {h['id']:7} no receipt at {f} — run scripts/model_ladder.sh on {h['id']} ({h.get('gpu')}, {h.get('cc')})"); rc = 1; continue
    try:
        R = json.load(open(f))
    except Exception as e:
        print(f"FAIL  {h['id']:7} receipt unreadable: {e}"); rc = 1; continue
    if R.get("version") != version:
        print(f"FAIL  {h['id']:7} receipt is for {R.get('version')!r}, this cut is {version!r} — STALE"); rc = 1; continue
    # #3842's rule applies: report and KEEP JUDGING. A `continue` here would suppress
    # every real per-row refusal behind it, which is the suppression this file exists to
    # expose. A wrong binary is an ADDITIONAL finding, not a reason to stop reading.
    if judge_apr_sha(h["id"], R):
        rc = 1
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
    good[h["id"]] = R
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
        if x.get("sha_ok") is False:
            print(f"FAIL  {h['id']:7} {rid:22} sha256 mismatch — a different file is a different measurement"); rc = 1; continue
        why = why_of(x, r.get("backends", []))
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
if model_ladder_cells.judge(L, good, rungs_doc, print, rungs_main):
    rc = 1
rc = judge_apr_drift_declaration() or rc
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
    # #3940: a case opts into the apr_sha binding by carrying a `cut_sha` file. Without
    # one the binding is not evaluated, so the 60+ cases written before it keep their
    # meaning instead of all reddening on a key they never had.
    if [ -f "$c/cut_sha" ]; then LADDER_CUT_SHA=$(cat "$c/cut_sha"); else LADDER_CUT_SHA=""; fi
    export LADDER_CUT_SHA
    out=$(judge "$lad" "$main" "$c/receipts" "$(cat "$c/version" 2>/dev/null || echo 0.0.0-case)" "$c/context-rungs.json" "$c/context-rungs_main.json"); got=$?
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
  # #3940 anti-vacuity, checked directly rather than through a fixture: judge() treats an
  # empty cut as "this case did not opt in", so the ONLY thing stopping that state from
  # reaching a real run and silently skipping the binding is this guard.
  if [ -z "$ONLY_CASE" ]; then
    if cut_sha_or_die "" > /dev/null 2>&1; then
      echo "FAIL  cut_sha_or_die accepted an EMPTY cut -- the apr_sha binding would not run and nothing would say so (#3940)"; bad=$((bad+1))
    else
      echo "ok    cut_sha_or_die  refuses an empty cut"
    fi
    if cut_sha_or_die "2f9fe2744d8b7f1429f38041806997a59654b109" > /dev/null 2>&1; then
      echo "ok    cut_sha_or_die  accepts a resolved cut"
    else
      echo "FAIL  cut_sha_or_die rejected a real sha -- a guard that refuses every run is not a guard"; bad=$((bad+1))
    fi
  fi
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
    mutant qa-rc-ignored      red-qa-rc-nonzero-refused 's/if qa_rc not in (0, None):/if False:/'
    mutant gates-unaccounted  red-gates-do-not-account-for-rc 's|if x.get("gates_account_for_rc") is False:|if False:|'
    # AND THE CROSS-CHECK THAT MAKES THE MUTANT MEAN SOMETHING (#3898).
    # A mutant killed by its own case only proves the case reads the rule. It does NOT
    # prove the rule covers a surface nothing else covers — and #3842's reconcile is the
    # nearest neighbour, close enough that "we already had that" is the obvious objection.
    # So: delete the qa_rc rule and require the #3842 case to stay GREEN. If #3842 went red
    # too, the two rules would be redundant and this one would be unjustified.
    qa_mut="$mdir/cross-qa-rc.sh"
    sed 's/if qa_rc not in (0, None):/if False:/' "$SELF" > "$qa_mut"
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
    pmutant no-lock      's/^apr_locked() { flock -E "\$LOCK_BUSY" -w "\$LOCK_WAIT" "\$GPU_LOCK" choom/apr_locked() { choom/'
    pmutant no-choom     's/ choom -n 1000 -- "\$APR" "\$@"/ "$APR" "$@"/'
    pmutant unbounded    's/ -w "\$LOCK_WAIT"//'
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
    cmutant pass-beyond-fit red-cells-pass-beyond-its-arithmetic 's/                            if not fit:/                            if False:/'
    cmutant family-long     red-cells-missing-cell          's/    if arch in (long_for.get("families") or \[\]):/    if False:/'
    cmutant rungs-floor     red-cells-rung-dropped-vs-main  's/            if gone:/            if False:/'
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
# #3940: the cut this run is judging. Exported, not passed positionally, because the
# self-test calls judge() with six arguments in every case and a seventh would rewrite
# all 63 of them.
#
# THIS IS THE ANTI-VACUITY GUARD for the apr_sha binding, and it lives here rather than
# in the judge because this is the only place that knows the run is REAL. Inside judge(),
# an unset value means "this case did not opt in"; out here it means git could not name
# what we are judging, and a ladder run that cannot name its own cut must not proceed to
# report on it.
: "${LADDER_CUT_SHA:=$(git rev-parse HEAD 2>/dev/null || echo "")}"
cut_sha_or_die "$LADDER_CUT_SHA" || exit 1
export LADDER_CUT_SHA
judge "$LADDER" "$MAIN_LADDER" "$RECEIPT_DIR" "$VERSION" evidence/release/context-rungs.json "$TMP_RUNGS"; rc=$?
[ -n "$TMP_RUNGS" ] && [ -f "$TMP_RUNGS" ] && rm -f "$TMP_RUNGS"
# The producer that writes these receipts must not bypass the fleet GPU lock (#3712): RED, not a decline.
if ! lock_audit scripts/model_ladder.sh; then [ "$rc" = 2 ] || rc=1; fi
case $rc in
  0) echo "ok    every required rung green on every required host" ;;
  1) echo "RED   the release claims a capability no receipt proves — see FAIL rows (EPIC #3477)" ;;
esac
exit $rc
