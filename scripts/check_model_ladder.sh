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
CASES_DIR="scripts/lib/model_ladder_cases"
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
judge() {
  python3 - "$1" "$2" "$3" "$4" "${5:-}" "${6:-}" <<'PY'
import json, os, sys, yaml
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
def why_of(x, backends):  # every reason a measured row is not green on the claimed backends
    why = []
    cm, go = x.get("capability_match") or {}, x.get("golden_output") or {}
    claims_gpu = bool({"cuda", "gpu"} & set(backends))
    cap_ok = (cm.get("passed") and not cm.get("skipped")) or (cm.get("skipped") and not claims_gpu)
    if not cap_ok: why.append("capability_match " + ("SKIPPED" if cm.get("skipped") else "FAIL") + ": " + str(cm.get("message", ""))[:60])
    if not (go.get("passed") and not go.get("skipped")): why.append("golden_output " + ("SKIPPED" if go.get("skipped") else "FAIL") + ": " + str(go.get("message", ""))[:60])
    be = x.get("backends") or {}
    for b in backends:
        v = be.get(b)
        if v is None: why.append(f"{b}: not measured")
        elif v.get("fallback"): why.append(f"{b}: FELL BACK — the claimed backend did not run")
        elif v.get("escaped_special"): why.append(f"{b}: the formatted prompt carries a zero-width-escaped special token — templated twice (#3743)")
        elif not v.get("ran"): why.append(f"{b}: did not run (rc={v.get('rc')})")
    return why
# #3712: no Q4_K rung is optional, and every one claims cuda. The key is refused, not tolerated.
for r in rungs:
    if is_q4k(r) and r.get("required") is not True:
        print(f"FAIL  rung {r['id']} is a Q4_K rung with required: {r.get('required')!r} — no Q4_K model is optional; every one must be green on CUDA (#3712)"); rc = 1
    if is_q4k(r) and "cuda" not in (r.get("backends") or []):
        print(f"FAIL  rung {r['id']} is a Q4_K rung that does not claim cuda — every Q4_K model must be green on CUDA (#3712)"); rc = 1
inv_backends = list((L.get("inventory") or {}).get("backends") or [])
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
            for what, a, b in (("verbs", mc.get("verbs"), hc.get("verbs")),
                               ("long-rung families", (mc.get("long_rungs_for") or {}).get("families"), (hc.get("long_rungs_for") or {}).get("families")),
                               ("long-rung representatives", list((mc.get("long_rungs_for") or {}).get("representatives") or {}), list((hc.get("long_rungs_for") or {}).get("representatives") or {}))):
                gone = set(a or []) - set(b or [])
                if gone: print(f"FAIL  cells {what} DROPPED vs origin/main: {sorted(gone)}"); rc = 1
        mh = {h["id"] for h in M.get("hosts", []) if h.get("required")}
        for hid in mh - {h["id"] for h in hosts}:
            print(f"FAIL  required host DROPPED vs origin/main: {hid}"); rc = 1
        print(f"ok    ladder covers every rung/host/backend at origin/main ({len(mr)} rungs, {len(mh)} hosts)")
    except Exception as e:
        print(f"FAIL  ladder at origin/main unreadable: {e}"); rc = 1
else:
    print("!     BOOTSTRAP: no ladder at origin/main yet; the anti-shrink floor arms when this lands")
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
    if int(R.get("executed", 0)) < 1:
        print(f"FAIL  {h['id']:7} receipt executed=0 — a receipt that measured nothing is not evidence"); rc = 1; continue
    inv = R.get("inventory")
    if R.get("schema") != "apr-model-ladder-receipt/v2" or not isinstance(inv, list):
        print(f"FAIL  {h['id']:7} receipt carries no measured inventory (schema {R.get('schema')!r}) — the universe is what the host HOLDS, not a list (#3712)"); rc = 1; continue
    if not inv:
        print(f"FAIL  {h['id']:7} measured inventory is EMPTY — a host holding no Q4_K model proved nothing (#3712)"); rc = 1; continue
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
        if why: print(f"FAIL  {h['id']:7} inv:{f:22} " + "; ".join(why)); rc = 1
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
if model_ladder_cells.judge(L, good, rungs_doc, print, rungs_main):
    rc = 1
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
    lad="$c/ladder.yaml"; [ -f "$lad" ] || lad="$LADDER"
    main=""; [ -f "$c/ladder_main.yaml" ] && main="$c/ladder_main.yaml"
    out=$(judge "$lad" "$main" "$c/receipts" "$(cat "$c/version" 2>/dev/null || echo 0.0.0-case)" "$c/context-rungs.json" "$c/context-rungs_main.json"); got=$?
    n=$((n+1))
    if [ "$got" = "$want" ] && { [ ! -f "$c/must_match" ] || grep -qE "$(cat "$c/must_match")" <<< "$out"; }; then
      printf 'ok    case %-28s rc=%s\n' "$name" "$got"
    else
      printf 'FAIL  case %-28s rc=%s want=%s%s\n' "$name" "$got" "$want" "$([ -f "$c/must_match" ] && printf ' must_match=/%s/' "$(cat "$c/must_match")")"
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
    mutant q4k-required-false red-q4k-required-false 's/if is_q4k(r) and r.get("required") is not True:/if False:/'
    mutant q4k-without-cuda   red-q4k-rung-cpu-only  's/if is_q4k(r) and "cuda" not in (r.get("backends") or \[\]):/if False:/'
    mutant inventory-missing  red-inventory-model-missing 's/if x is None or not x.get("present"):  # held by the host, absent from the run/if False:/'
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
judge "$LADDER" "$MAIN_LADDER" "$RECEIPT_DIR" "$VERSION" evidence/release/context-rungs.json "$TMP_RUNGS"; rc=$?
[ -n "$TMP_RUNGS" ] && [ -f "$TMP_RUNGS" ] && rm -f "$TMP_RUNGS"
# The producer that writes these receipts must not bypass the fleet GPU lock (#3712): RED, not a decline.
if ! lock_audit scripts/model_ladder.sh; then [ "$rc" = 2 ] || rc=1; fi
case $rc in
  0) echo "ok    every required rung green on every required host" ;;
  1) echo "RED   the release claims a capability no receipt proves — see FAIL rows (EPIC #3477)" ;;
esac
exit $rc
