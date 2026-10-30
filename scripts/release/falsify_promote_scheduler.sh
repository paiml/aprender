#!/usr/bin/env bash
# falsify_promote_scheduler.sh -- the case table and mutants for receipts_from_night.sh + promote_scheduler.sh (N26, PMAT-5306)
#
#   bash scripts/release/falsify_promote_scheduler.sh            # every row
#   bash scripts/release/falsify_promote_scheduler.sh --mutants  # the table, then one mutant per check: each must turn it red
#
# Proves a red, missing, wrong-sha, unverified or unmeasured input starts NO promotion, and that the green row does.
# The scheduler under test runs from a SANDBOX COPY with a .promote-test-seams file beside it (the only way to stub gh,
# the promote script and the verifiers); the installed script never honours env or seams. Throwaway origin + clone under
# mktemp; nothing real is touched.  Exit 0 = every row held (and every mutant caught), 1 = not.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${SCHED_SRC:-$HERE/promote_scheduler.sh}"
if [ "${1:-}" = --mutants ]; then
    T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
    bash "$0" > "$T/base.log" 2>&1 || { tail -n 5 "$T/base.log"; echo "mutants: the unmutated table is not green"; exit 1; }
    miss=0 n=0
    mut() { # name sed-expr
        n=$((n + 1)); sed "$2" "$SRC" > "$T/m.sh"
        if cmp -s "$SRC" "$T/m.sh"; then printf 'MUTANT %s did not apply\n' "$1" >&2; miss=$((miss + 1)); return; fi
        if SCHED_SRC="$T/m.sh" bash "$0" > "$T/m.log" 2>&1; then printf 'MUTANT %s SURVIVED\n' "$1" >&2; miss=$((miss + 1)); fi
    }
    mut dirty-check-dropped 's/^\[ -z "\$(G status --porcelain.*$/true/'
    mut not-at-main-dropped 's/^\[ "\$(G rev-parse HEAD.*$/true/'
    mut models-always-ok 's/^    \[ "\$(sed -n .s\/\^state=\/\/p. "\$go".*$/    true/'
    mut train-verdict-ignored 's/--exit-verdict > "\$d\/train.log" 2>&1 || {/--exit-verdict > "$d\/train.log" 2>\&1 || true || {/'
    mut no-ff-when-behind 's/ || G merge -q --ff-only .*$//'
    mut seams-in-repo-allowed 's/^    case "\$(realpath -e.*$/    :/'
    mut rate-floor-dropped 's/-lt 1000 \]; then/-lt 0 ]; then/'
    mut dispatch-time-filter-dropped 's/select(.createdAt >= [^)]*)/select(true)/'
    mut sandbox-dropped 's/"\${SANDBOX\[@\]}" bash "\$TRAIN"/bash "$TRAIN"/'
    mut already-promoted-check-dropped 's/ls-remote origin "refs\/tags\/v\$ver"/ls-remote origin "refs\/tags\/none-$ver"/'
    mut title-preference-dropped 's/(map(select([^+]*)) + \.)/./'
    mut red-marker-blocks-all 's/return 2 ;;/fail "red" ;;/'
    mut pending-silent 's/inbox_line "PENDING clean-room run \$id on[^;]*;//'
    mut lock-silent 's/inbox_line "SKIP lock held[^;]*;//'
    mut cleanroom-red-pending 's/^            completed\*) .*$/            completed*) say pending; exit 3 ;;/'
    mut seams-read-from-env 's|^\[ ! -f "\$HERE/.promote-test-seams" \] .*$|[ -z "${GH_SEAMS:-}" ] \|\| . "$GH_SEAMS"|'
    mut flock-dropped 's/^flock -n 9 .*$/true/'
    mut age-cap-dropped 's/\[ "\$ct" -ge "\$cutoff" \] || continue/true/'
    mut pushtags-unchecked 's/ \\$/ ;/; /not installed\/verified in/d'
    mut dryrun-skipped-on-execute 's/^    if \[ "\$exe" != 1 \]; then say.*$/    :/'
    printf 'falsify_promote_scheduler --mutants: %d mutants, %d not caught\n' "$n" "$miss"
    [ "$miss" = 0 ]; exit
fi

TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); printf 'FAIL: %s\n' "$1" >&2; }
expect() { if [ "$2" = "$3" ]; then ok; else bad "$1: want $2, got $3"; fi; }
SHA=$(printf 'a%.0s' $(seq 40)); OTHER=$(printf 'b%.0s' $(seq 40))

# ---- receipts_from_night.sh rows ----
mk_models() { # dir state rc golast commit
    mkdir -p "$1"
    printf 'commit=%s\nmeasure=crux\nt1_rc=%s\nstate=%s\n' "${5:-$SHA}" "$3" "$2" > "$1/verdict"
    printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: judge passed\n' "${4:-${SHA:0:9}}" > "$1/models-t1.log"
}
mk_cr() { jq -n --arg s "$2" --arg c "$3" '{run_id:"123",job:"clean-room (aprender)",conclusion:$c,tested_shas:[$s]}' > "$1"; }
run_r() { # name want_rc want_files
    local n=$1 o="$TMP/o-$1" rc
    rm -rf "${o:?}"
    bash "$HERE/receipts_from_night.sh" --sha "$SHA" --out "$o" --models-dir "$MD" --cleanroom-json "$CR" --dryrun-rc "$DRC" --dryrun-commit "$DCM" > /dev/null 2>&1; rc=$?
    expect "$n rc" "$2" "$rc"
    expect "$n files" "$3" "$(find "$o" -name '*.json' | wc -l)"
}
MD="$TMP/md-green"; mk_models "$MD" green 0; CR="$TMP/cr-green.json"; mk_cr "$CR" "$SHA" success; DRC=0 DCM=$SHA
run_r green 0 3
MD="$TMP/md-red"; mk_models "$MD" red 1; run_r models-red 1 2
MD="$TMP/md-redrc"; mk_models "$MD" green 1; run_r models-green-state-rc1 1 2
MD="$TMP/md-stale"; mk_models "$MD" green 0 cafef00d1; run_r models-go-other-sha 1 2
MD="$TMP/md-missing"; run_r models-missing 1 2
MD="$TMP/md-fail"; mk_models "$MD" green 0; printf 'MODELS   FAIL  gx10 cell\n' >> "$MD/models-t1.log"; run_r models-fail-line 1 2
MD="$TMP/md-dupkey"; mk_models "$MD" red 0; printf 'state=green\n' >> "$MD/verdict"; run_r models-later-state-green-first-is-red 1 2
MD="$TMP/md-commit"; mk_models "$MD" green 0 "" "$OTHER"; run_r models-verdict-other-commit 1 2
MD="$TMP/md-green"; CR="$TMP/cr-fail.json"; mk_cr "$CR" "$SHA" failure; run_r cleanroom-failure 1 2
CR="$TMP/cr-other.json"; mk_cr "$CR" "$OTHER" success; run_r cleanroom-other-sha 1 2
CR="$TMP/cr-none.json"; run_r cleanroom-missing 1 2
CR="$TMP/cr-green.json"; DRC=1; run_r dryrun-red 1 2
DRC=0 DCM=$OTHER; run_r dryrun-other-sha 1 2
DRC="" DCM=""; run_r dryrun-unset 1 2

# ---- promote_scheduler.sh rows: origin with two main commits (R1 older, R2 newest) and an evidence branch ----
O="$TMP/origin.git"; W="$TMP/work"; ST="$TMP/state"; INB="$TMP/inbox.md"
git init -q --bare "$O"; git clone -q "$O" "$W" 2> /dev/null
g() { git -C "$W" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
g checkout -q -b main; printf w > "$W/f"; printf '[workspace.package]\nversion = "1.2.3"\n' > "$W/Cargo.toml"; g add f Cargo.toml; T0=$(( $(date +%s) - 864000 )); GIT_AUTHOR_DATE="@$T0" GIT_COMMITTER_DATE="@$T0" g commit -q -m c0; R0=$(g rev-parse HEAD) # bashrs disable-line=DET002
printf x > "$W/f"; g add f; T1=$(( $(date +%s) - 7200 )); GIT_AUTHOR_DATE="@$T1" GIT_COMMITTER_DATE="@$T1" g commit -q -m c1; R1=$(g rev-parse HEAD) # bashrs disable-line=DET002
printf y > "$W/f"; g commit -q -am c2; R2=$(g rev-parse HEAD); g push -q origin main
g checkout -q -b nightly-evidence
for r in $R0 $R1 $R2; do
    mkdir -p "$W/models-crux/$r"; printf 'commit=%s\nmeasure=crux\nt1_rc=0\nstate=green\n' "$r" > "$W/models-crux/$r/verdict"
    printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: ok\n' "${r:0:9}" > "$W/models-crux/$r/models-t1.log"
done
g add -A; g commit -q -m ev; g push -q origin nightly-evidence; g checkout -q main
EV="$TMP/ev-src"; git clone -q "$O" "$EV" 2> /dev/null; git -C "$EV" checkout -q nightly-evidence

BIN="$TMP/bin"; mkdir -p "$BIN"; cp "$SRC" "$BIN/promote_scheduler.sh"; cp "$HERE/receipts_from_night.sh" "$BIN/"
cat > "$TMP/gh" <<'SH'
#!/usr/bin/env bash
echo "GH $*" >> "$FX_CALLS"
repo=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = --repo ] && { j=$((i+1)); repo=${!j}; }; done
case "$*" in *rate_limit*) echo "${FX_RATE:-5000}"; exit 0 ;; esac
case "$*" in
  *"workflow run"*) exit 0 ;;
  *"--json databaseId,createdAt"*) fx_j='[{"databaseId":111,"createdAt":"2000-01-01T00:00:00Z","displayTitle":""}]'; [ "${FX_STALE_RUN:-0}" = 1 ] || fx_j="[{\"databaseId\":998,\"createdAt\":\"2999-01-01T00:00:00Z\",\"displayTitle\":\"other\"},{\"databaseId\":999,\"createdAt\":\"2999-01-01T00:00:00Z\",\"displayTitle\":\"clean-room ${FX_SHA:0:9}\"},{\"databaseId\":111,\"createdAt\":\"2000-01-01T00:00:00Z\",\"displayTitle\":\"\"}]"
      fx_f=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = --jq ] && { j=$((i+1)); fx_f=${!j}; }; done; printf '%s' "$fx_j" | jq -r "$fx_f" ;;
  *"run list"*) [ "$FX_CR" = none ] || echo 777 ;;
  *"--job"*) echo "tested-sha: $FX_SHA" ;;
  *"status,conclusion"*) echo "$FX_RUN_STATE" ;;
  *"run view"*) echo "success 888" ;;
esac
SH
cat > "$TMP/promote" <<'SH'
#!/usr/bin/env bash
echo "PROMOTE $*" >> "$FX_CALLS"
[ "$FX_PROMOTE_RC" = 0 ] || { echo "promote_nightly: REFUSE -- stub" >&2; exit 1; }
SH
printf '#!/usr/bin/%s\nexit "${FX_GATE_RC:-0}"\n' env\ bash > "$TMP/gate"
cat > "$TMP/models" <<'SH'
#!/usr/bin/env bash
# stub verifier: state per sha
sha=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = --commit ] && { j=$((i+1)); sha=${!j}; }; done
st=green; case " ${FX_MODELS_RED_SHA:-none} " in *" $sha "*) st=red ;; esac; [ "${FX_MODELS:-green}" = green ] || st=$FX_MODELS
printf 'state=%s\n' "$st" >> "$GITHUB_OUTPUT"
SH
cat > "$TMP/pt" <<'SH'
#!/usr/bin/env bash
echo "PT $*" >> "$FX_CALLS"
case "$1" in --install) [ "${FX_PT_INSTALL_FAILS:-0}" = 1 ] || touch "$FX_PT_FLAG" ;; --verify) [ "${FX_PT_OK:-1}" = 1 ] || [ -e "$FX_PT_FLAG" ] ;; esac
SH
cat > "$TMP/train" <<'SH'
#!/usr/bin/env bash
echo "TRAIN $*" >> "$FX_CALLS"
[ ! -e "$HOME/.cargo/credentials.toml" ] || [ -n "${FX_HIDDEN:-}" ] || { echo "no-token: a credentials file is reachable" >&2; exit 2; }
sha=""; out=""; ev=0
for ((i=1;i<=$#;i++)); do case "${!i}" in --commit) j=$((i+1)); sha=${!j} ;; --out) j=$((i+1)); out=${!j} ;; --exit-verdict) ev=1 ;; esac; done
[ -n "$out" ] && [ "$ev" = 1 ] && [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "train: bad args" >&2; exit 3; }
[ "$FX_LANES" = none ] && { echo "NOT RELEASABLE: ci-main not_measured"; exit 1; }
[ -z "${FX_RED_JOB:-}" ] || { echo "NOT RELEASABLE: $FX_RED_JOB red"; exit 1; }
[ "$sha" = "${FX_TRAIN_RED_SHA:-none}" ] && { echo "NOT RELEASABLE: attempt 2"; exit 1; }
mkdir -p "$out"; echo "RELEASABLE"
SH
chmod +x "$TMP/train"; mkdir -p "$TMP/h/.cargo"; : > "$TMP/h/.cargo/credentials.toml"  # a FAKE file in the sandbox test home
chmod +x "$TMP/gh" "$TMP/promote" "$TMP/gate" "$TMP/models" "$TMP/pt"
cat > "$BIN/.promote-test-seams" <<EOF
GH="$TMP/gh"; PROMOTE="$TMP/promote"; RECEIPTS="$BIN/receipts_from_night.sh"; DRYRUN_GATE="$TMP/gate"
MODELS_NIGHTLY="${FX_REAL_MODELS_PATH:-$TMP/models}"; PUSHTAGS="$TMP/pt"; EVIDENCE_DIR="$EV"; DISPATCH_WAIT=0
SANDBOX=(env "HOME=$TMP/h" FX_HIDDEN=1)
TRAIN="\${FX_TRAIN_PATH:-$TMP/train}"
[ -z "\${FX_REAL_MODELS:-}" ] || MODELS_NIGHTLY="$HERE/models_nightly.sh"
EOF
sched() { # name want_rc want_promote want_exec [args...]   (FX_* in the environment)
    local n=$1 wr=$2 wp=$3 we=$4 rc; shift 4
    : > "$TMP/calls"; : > "$INB"; rm -rf "${ST:?}" "${TMP:?}/pt-flag"
    if [ "${FX_PRESEED:-0}" = 1 ]; then mkdir -p "$ST/$R2"; echo 999 > "$ST/$R2/cleanroom-run-id"; fi
    FX_CALLS="$TMP/calls" FX_PT_FLAG="$TMP/pt-flag" FX_R1=$R1 FX_R2=$R2 FX_R0=$R0 \
        bash "${SCHED_BIN:-$BIN/promote_scheduler.sh}" --repo "$W" --state "$ST" --inbox "$INB" "$@" > "$TMP/out-$n" 2>&1; rc=$?
    expect "$n rc" "$wr" "$rc"
    expect "$n promote calls" "$wp" "$(grep -c '^PROMOTE' "$TMP/calls")"
    expect "$n execute calls" "$we" "$(grep -c '^PROMOTE.*--execute' "$TMP/calls")"
}
inbox_has() { grep -q "$2" "$INB" && ok || bad "$1: inbox lacks '$2'"; }
export FX_CR=green FX_PROMOTE_RC=0 FX_GATE_RC=0 FX_LANES=green FX_SHA=$R2 FX_RUN_STATE="completed success"
sched dry-run-all-green 0 1 0
sched execute-all-green 0 2 1 --execute; inbox_has execute-all-green DONE
grep -q "PT --verify" "$TMP/calls" && ok || bad "execute did not run pre-push-tags --verify"
FX_PT_OK=0 sched pushtags-installed-then-verified 0 2 1 --execute
grep -q "PT --install" "$TMP/calls" && ok || bad "pushtags --install not run"
FX_PT_OK=0 FX_PT_INSTALL_FAILS=1 sched pushtags-cannot-install 1 1 0 --execute; inbox_has pushtags-cannot-install RED
FX_PROMOTE_RC=1 sched promote-dry-refuses 1 1 0 --execute; inbox_has promote-dry-refuses RED
FX_GATE_RC=1 sched publish-dryrun-red 1 0 0 --execute; inbox_has publish-dryrun-red RED
FX_MODELS=red sched models-not-green 1 0 0 --execute; inbox_has models-not-green RED
FX_MODELS=not_measured sched models-not-measured 1 0 0 --execute
FX_REAL_MODELS=1 sched real-models-nightly-weak-bundle 1 0 0 --execute
FX_RED_JOB=preflight sched release-day-lane-red 1 0 0 --execute
FX_RED_JOB=cleanroom-gpu sched cleanroom-gpu-lane-red 1 0 0 --execute
FX_RED_JOB=ci-main sched ci-main-red 1 0 0 --execute
FX_TRAIN_PATH=/nonexistent/train sched train-unavailable-fails-closed 1 0 0 --execute; inbox_has train-unavailable RED
FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 sched train-red-newest-falls-back 0 2 1 --execute
grep -q "TRAIN --out .* --commit $R1 --exit-verdict" "$TMP/calls" || grep -q "TRAIN.*--commit $R1.*--exit-verdict\|TRAIN.*--exit-verdict.*--commit $R1" "$TMP/calls" && ok || bad "train not called with --commit/--exit-verdict"
FX_RATE=999 sched rate-floor 0 0 0 --execute; inbox_has rate-floor SKIP
FX_RATE=x sched rate-unreadable 0 0 0 --execute
FX_LANES=none sched lane-evidence-missing-fails-closed 1 0 0 --execute; inbox_has lane-evidence-missing RED
FX_CR=none sched cleanroom-missing-no-flag 1 0 0 --execute
FX_CR=none sched cleanroom-missing-dispatches-pending 3 0 0 --execute --dispatch-cleanroom
grep -q 'workflow run clean-room.yml' "$TMP/calls" && ok || bad "dispatch not issued"
inbox_has dispatch PENDING
expect "run id recorded (named for the sha, not unrelated 998)" 999 "$(cat "$ST/$R2/cleanroom-run-id" 2> /dev/null)"
FX_CR=none FX_PRESEED=1 FX_RUN_STATE="in_progress " sched cleanroom-running-pending 3 0 0 --execute --dispatch-cleanroom; inbox_has cleanroom-running PENDING
FX_CR=none FX_STALE_RUN=1 sched dispatch-only-older-runs-not-adopted 1 0 0 --execute --dispatch-cleanroom
FX_CR=none FX_PRESEED=1 FX_RUN_STATE="completed failure" sched cleanroom-ended-red-not-pending 1 0 0 --execute; inbox_has cleanroom-ended-red RED
FX_CR=none FX_PRESEED=1 FX_RUN_STATE="" sched cleanroom-unreadable 1 0 0 --execute
FX_PRESEED=1 FX_RUN_STATE="completed failure" FX_SHA=$R1 sched red-marker-does-not-block-older 0 2 1 --execute
grep -q "PROMOTE --sha $R1" "$TMP/calls" && ok || bad "older candidate not promoted after red marker"
sched too-early 0 0 0 --execute --not-before 2999-01-01T00:00:00Z
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 sched newest-red-falls-back-to-last-green 0 2 1 --execute; inbox_has newest-red RED
grep -q "PROMOTE --sha $R1" "$TMP/calls" && ok || bad "fallback did not promote the older green sha"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R2 sched newest-red-no-clean-room-on-older 1 0 0 --execute
FX_MODELS_RED_SHA="$R2 $R1" FX_SHA=$R0 sched age-cap-skips-old-green-sha 1 0 0 --execute
# checkout hygiene
printf z >> "$W/f"; sched dirty-checkout 1 0 0 --execute; git -C "$W" checkout -q -- f
g commit -q --allow-empty -m local; sched checkout-not-at-origin-main 1 0 0 --execute; g reset -q --hard origin/main
# idempotence: the version tag is already on origin -> exit 0, no promote, no older candidate
g push -q origin "$R2:refs/tags/v1.2.3"
sched already-promoted-tag-on-origin 0 0 0 --execute
grep -q 'already promoted' "$INB" && ok || bad "already-promoted inbox line"
g push -q origin ":refs/tags/v1.2.3"
# liveness: a clean checkout BEHIND origin/main fast-forwards and runs
printf n > "$W/f2"; g add f2; g commit -q -m c3; g push -q origin main; g reset -q --hard HEAD~1
sched behind-origin-fast-forwards 0 1 0
expect "ff reached origin/main" "$(g rev-parse origin/main)" "$(g rev-parse HEAD)"
# a seams file inside the repo checkout is refused
mkdir -p "$W/sb"; cp "$BIN/promote_scheduler.sh" "$BIN/receipts_from_night.sh" "$BIN/.promote-test-seams" "$W/sb/"
SCHED_BIN="$W/sb/promote_scheduler.sh" sched seams-inside-repo-refused 1 0 0 --execute; inbox_has seams-inside-repo RED
rm -rf -- "${W:?}/sb"
# env never honoured: a sandbox copy WITHOUT the seams file, hostile env
mkdir -p "$TMP/bin2"; cp "$SRC" "$TMP/bin2/promote_scheduler.sh"
GH="$TMP/gh" PROMOTE="$TMP/promote" RECEIPTS="$BIN/receipts_from_night.sh" DRYRUN_GATE="$TMP/gate" MODELS_NIGHTLY="$TMP/models" PUSHTAGS="$TMP/pt" EVIDENCE_DIR="$EV" \
    SCHED_BIN="$TMP/bin2/promote_scheduler.sh" sched env-overrides-ignored 1 0 0 --execute
expect "env: no stub gh call" 0 "$(grep -c '^GH' "$TMP/calls")"
# lock: a second run while the first holds it does nothing
mkdir -p "$ST"; : > "$TMP/calls"
( exec 8> "$ST/.lock"; flock -n 8; FX_CALLS="$TMP/calls" bash "$BIN/promote_scheduler.sh" --repo "$W" --state "$ST" --inbox "$INB" --execute > "$TMP/out-lock" 2>&1; echo $? > "$TMP/lock-rc" )
expect "lock rc" 0 "$(cat "$TMP/lock-rc")"; grep -q 'holds the lock' "$TMP/out-lock" && ok || bad "lock message"
expect "lock: nothing ran" 0 "$(wc -l < "$TMP/calls")"
inbox_has lock-held SKIP
printf 'falsify_promote_scheduler: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
