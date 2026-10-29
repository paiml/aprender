#!/usr/bin/env bash
# falsify_promote_scheduler.sh -- the case table for receipts_from_night.sh and promote_scheduler.sh (N26, PMAT-5306)
#
#   bash scripts/release/falsify_promote_scheduler.sh
#
# Proves a red, missing, wrong-sha or unsuccessful input writes NO receipt / starts NO promotion, and that the one green
# row does. Throwaway origin + clone under mktemp; gh and the promote script are stubs; nothing real is touched.
# Exit 0 = every row held, 1 = a row failed.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); printf 'FAIL: %s\n' "$1" >&2; }
expect() { # name want_rc got_rc
    if [ "$2" = "$3" ]; then ok; else bad "$1: want rc $2, got $3"; fi
}
SHA=$(printf 'a%.0s' $(seq 40)); OTHER=$(printf 'b%.0s' $(seq 40))

# ---- receipts_from_night.sh rows ----
mk_models() { # dir state rc golast
    mkdir -p "$1"
    printf 'measure=crux\nt1_rc=%s\nstate=%s\n' "$3" "$2" > "$1/verdict"
    printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: judge passed\n' "${4:-${SHA:0:9}}" > "$1/models-t1.log"
}
mk_cr() { # file sha conclusion
    jq -n --arg s "$2" --arg c "$3" '{run_id:"123",job:"clean-room (aprender)",conclusion:$c,tested_shas:[$s]}' > "$1"
}
run_r() { # name want_rc want_files(count) -- args via vars
    local n=$1 want=$2 files=$3 o="$TMP/o-$1" rc
    rm -rf "${o:?}"
    bash "$HERE/receipts_from_night.sh" --sha "$SHA" --out "$o" --models-dir "$MD" --cleanroom-json "$CR" --dryrun-rc "$DRC" --dryrun-commit "$DCM" > /dev/null 2>&1; rc=$?
    expect "$n rc" "$want" "$rc"
    [ "$(find "$o" -name '*.json' | wc -l)" = "$files" ] && ok || bad "$n: expected $files receipt files"
}
MD="$TMP/md-green"; mk_models "$MD" green 0; CR="$TMP/cr-green.json"; mk_cr "$CR" "$SHA" success; DRC=0 DCM=$SHA
run_r green 0 3
MD="$TMP/md-red"; mk_models "$MD" red 1; run_r models-red 1 2
MD="$TMP/md-redrc"; mk_models "$MD" green 1; run_r models-green-state-rc1 1 2
MD="$TMP/md-stale"; mk_models "$MD" green 0 cafef00d1; run_r models-go-other-sha 1 2
MD="$TMP/md-missing"; run_r models-missing 1 2
MD="$TMP/md-fail"; mk_models "$MD" green 0; printf 'MODELS   FAIL  gx10 cell\n' >> "$MD/models-t1.log"; run_r models-fail-line 1 2
MD="$TMP/md-green"; CR="$TMP/cr-fail.json"; mk_cr "$CR" "$SHA" failure; run_r cleanroom-failure 1 2
CR="$TMP/cr-other.json"; mk_cr "$CR" "$OTHER" success; run_r cleanroom-other-sha 1 2
CR="$TMP/cr-none.json"; run_r cleanroom-missing 1 2
CR="$TMP/cr-green.json"; DRC=1; run_r dryrun-red 1 2
DRC=0 DCM=$OTHER; run_r dryrun-other-sha 1 2
DRC="" DCM=""; run_r dryrun-unset 1 2

# ---- promote_scheduler.sh rows ----
O="$TMP/origin.git"; W="$TMP/work"; EV="$TMP/ev"; ST="$TMP/state"
git init -q --bare "$O"; git clone -q "$O" "$W" 2> /dev/null
g() { git -C "$W" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
g checkout -q -b main; printf x > "$W/f"; g add f; g commit -q -m c1; g push -q origin main
RS=$(g rev-parse HEAD)
g checkout -q -b nightly-evidence; mkdir -p "$W/models-crux/$RS"; printf 'state=green\nt1_rc=0\n' > "$W/models-crux/$RS/verdict"
printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: ok\n' "${RS:0:9}" > "$W/models-crux/$RS/models-t1.log"
g add -A; g commit -q -m ev; g push -q origin nightly-evidence; g checkout -q main
git clone -q "$O" "$TMP/ev-src" 2> /dev/null; git -C "$TMP/ev-src" checkout -q nightly-evidence; EV="$TMP/ev-src"

cat > "$TMP/gh" <<'SH'
#!/usr/bin/env bash
# stub gh: records dispatches; run list/view answer from FX_CR (green|none)
echo "$*" >> "$FX_CALLS"
case "$1 $2" in
  "run list") [ "$FX_CR" = none ] || echo 777 ;;
  "run view") if [ "$3" = --job ]; then echo "tested-sha: $FX_SHA"; else echo "success 888"; fi ;;
  "workflow run") exit 0 ;;
esac
SH
cat > "$TMP/promote" <<'SH'
#!/usr/bin/env bash
echo "PROMOTE $*" >> "$FX_CALLS"
[ "$FX_PROMOTE_RC" = 0 ] || { echo "promote_nightly: REFUSE -- stub" >&2; exit 1; }
exit 0
SH
printf '#!/usr/bin/env bash\nexit "${FX_GATE_RC:-0}"\n' > "$TMP/gate"
chmod +x "$TMP/gh" "$TMP/promote" "$TMP/gate"
sched() { # name want_rc want_promote_calls want_execute_calls args...
    local n=$1 wr=$2 wp=$3 we=$4 rc; shift 4
    : > "$TMP/calls"; rm -rf "${ST:?}"
    FX_CALLS="$TMP/calls" FX_SHA="$RS" GH="$TMP/gh" PROMOTE="$TMP/promote" DRYRUN_GATE="$TMP/gate" EVIDENCE_DIR="$EV/" \
        bash "$HERE/promote_scheduler.sh" --repo "$W" --state "$ST" "$@" > "$TMP/out-$n" 2>&1; rc=$?
    expect "$n rc" "$wr" "$rc"
    [ "$(grep -c '^PROMOTE' "$TMP/calls")" = "$wp" ] && ok || bad "$n: want $wp promote call(s), got $(grep -c '^PROMOTE' "$TMP/calls")"
    [ "$(grep -c '^PROMOTE.*--execute' "$TMP/calls")" = "$we" ] && ok || bad "$n: want $we --execute call(s)"
}
# nightly-evidence is read through EVIDENCE_DIR (a checkout); fetch targets the clone's origin
export FX_CR=green FX_PROMOTE_RC=0 FX_GATE_RC=0
sched dry-run-all-green 0 1 0
sched execute-all-green 0 2 1 --execute
FX_PROMOTE_RC=1 sched promote-dry-refuses-no-execute 1 1 0 --execute
FX_PROMOTE_RC=0 FX_GATE_RC=1 sched publish-dryrun-red 1 0 0 --execute
FX_GATE_RC=0 FX_CR=none sched cleanroom-missing-no-dispatch-flag 1 0 0 --execute
FX_CR=none sched cleanroom-missing-dispatches-pending 3 0 0 --execute --dispatch-cleanroom
grep -q 'workflow run clean-room.yml' "$TMP/calls" && ok || bad "dispatch not issued"
FX_CR=green sched too-early 0 0 0 --execute --not-before 2999-01-01T00:00:00Z
sed -i 's/state=green/state=red/' "$EV/models-crux/$RS/verdict"
FX_CR=green sched models-red-no-candidate 1 0 0 --execute

printf 'falsify_promote_scheduler: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
