#!/usr/bin/env bash
# check_model_ladder.sh — the T-2 gate over contracts/model-capability-ladder-v1.yaml.
#
# Reads one receipt per REQUIRED host (written by scripts/model_ladder.sh) for
# the version being cut and is green only when every required rung is present
# and green on every one of them. It is declared in Cargo.toml
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
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "check_model_ladder: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 2

# ---------------------------------------------------------------- the judge
# judge <ladder> <ladder_at_main_or_empty> <receipt_dir> <version>  → exit 0/1/2
judge() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys, yaml
ladder_p, main_p, rdir, version = sys.argv[1:5]
try:
    L = yaml.safe_load(open(ladder_p))["ladder"]
except Exception as e:
    print(f"decline: ladder unreadable: {e}"); sys.exit(2)
hosts = [h for h in L.get("hosts", []) if h.get("required")]
rungs = L.get("rungs", [])
if not hosts: print("decline: ladder names no required host"); sys.exit(2)
if not rungs: print("decline: ladder has no rungs"); sys.exit(2)
rc = 0
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
        mh = {h["id"] for h in M.get("hosts", []) if h.get("required")}
        for hid in mh - {h["id"] for h in hosts}:
            print(f"FAIL  required host DROPPED vs origin/main: {hid}"); rc = 1
        print(f"ok    ladder covers every rung/host/backend at origin/main ({len(mr)} rungs, {len(mh)} hosts)")
    except Exception as e:
        print(f"FAIL  ladder at origin/main unreadable: {e}"); rc = 1
else:
    print("!     BOOTSTRAP: no ladder at origin/main yet; the anti-shrink floor arms when this lands")
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
    by = {r.get("id"): r for r in R.get("rungs", [])}
    for r in rungs:
        rid = r["id"]; req = bool(r.get("required"))
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
        why = []
        cm, go = x.get("capability_match") or {}, x.get("golden_output") or {}
        claims_gpu = bool({"cuda", "gpu"} & set(r.get("backends", [])))
        cap_ok = (cm.get("passed") and not cm.get("skipped")) or (cm.get("skipped") and not claims_gpu)
        if not cap_ok: why.append("capability_match " + ("SKIPPED" if cm.get("skipped") else "FAIL") + ": " + str(cm.get("message",""))[:60])
        if not (go.get("passed") and not go.get("skipped")): why.append("golden_output " + ("SKIPPED" if go.get("skipped") else "FAIL") + ": " + str(go.get("message",""))[:60])
        be = x.get("backends") or {}
        for b in r.get("backends", []):
            v = be.get(b)
            if v is None: why.append(f"{b}: not measured")
            elif v.get("fallback"): why.append(f"{b}: FELL BACK — the claimed backend did not run")
            elif v.get("escaped_special"): why.append(f"{b}: the formatted prompt carries a zero-width-escaped special token — templated twice (#3743)")
            elif not v.get("ran"): why.append(f"{b}: did not run (rc={v.get('rc')})")
        if why:
            if req: print(f"FAIL  {h['id']:7} {rid:22} " + "; ".join(why)); rc = 1
            else:   print(f"warn  {h['id']:7} {rid:22} ({tag}) " + "; ".join(why))
        else:
            print(f"ok    {h['id']:7} {rid:22} green on {','.join(r.get('backends', []))} ({R.get('gpu') or 'no-gpu'}, sha {R.get('sha')})")
sys.exit(rc)
PY
}

# ---------------------------------------------------------------- self-test
if [ "$SELF_TEST" = 1 ]; then
  n=0; bad=0
  for c in "$CASES_DIR"/*/; do
    name=$(basename "$c")
    [ -z "$ONLY_CASE" ] || [ "$name" = "$ONLY_CASE" ] || continue
    [ -f "$c/expected_rc" ] || { echo "FAIL  case $name has no expected_rc"; bad=$((bad+1)); continue; }
    want=$(cat "$c/expected_rc")
    lad="$c/ladder.yaml"; [ -f "$lad" ] || lad="$LADDER"
    main=""; [ -f "$c/ladder_main.yaml" ] && main="$c/ladder_main.yaml"
    out=$(judge "$lad" "$main" "$c/receipts" "$(cat "$c/version" 2>/dev/null || echo 0.0.0-case)"); got=$?
    n=$((n+1))
    if [ "$got" = "$want" ] && { [ ! -f "$c/must_match" ] || grep -qE "$(cat "$c/must_match")" <<< "$out"; }; then
      printf 'ok    case %-28s rc=%s\n' "$name" "$got"
    else
      printf 'FAIL  case %-28s rc=%s want=%s%s\n' "$name" "$got" "$want" "$([ -f "$c/must_match" ] && printf ' must_match=/%s/' "$(cat "$c/must_match")")"
      printf '%s\n' "$out" | sed 's/^/        /'; bad=$((bad+1))
    fi
  done
  if [ "$n" -lt 6 ] && [ -z "$ONLY_CASE" ]; then echo "FAIL  only $n case(s) ran; the table needs >= 6 to discriminate"; bad=$((bad+1)); fi
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
judge "$LADDER" "$MAIN_LADDER" "$RECEIPT_DIR" "$VERSION"; rc=$?
case $rc in
  0) echo "ok    every required rung green on every required host" ;;
  1) echo "RED   the release claims a capability no receipt proves — see FAIL rows (EPIC #3477)" ;;
esac
exit $rc
