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
    mut rate-floor-dropped 's/-lt "\$floor" \]; then/-lt 0 ]; then/'
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
    mut fallback-silent 's/^    inbox_line "\$head_\$why"$/    :/'
    mut fallback-names-cut 's/snames% }/snames%% *}/'
    mut cut-marker-room-not-reserved 's/room=\$((room - 24))/room=$((room - 0))/'
    mut fallback-already-promoted-silent 's/^            fallback_line "already-promoted at" "\$sha"$/            :/'
    mut cleanroom-fail-label-dropped 's/FL="no clean-room at" fail/fail/'
    mut fallback-cleanroom-dispatched-silent 's/fallback_line "clean-room dispatched at" "\$sha"; //'
    mut fallback-cleanroom-pending-silent 's/fallback_line "clean-room pending at" "\$sha"; //'
    mut cleanroom-red-second-line 's/completed\*) crnote=/completed*) inbox_line "RED clean-room run $id on ${sha:0:9}"; crnote=/'
    mut dryrun-skipped-on-execute 's/^    if \[ "\$exe" != 1 \]; then fallback_line.*$/    :/'
    mut fallback-already-promoted-behind-marker 's/^            fallback_line "already-promoted at" "\$sha"$/            [ -e "$d\/already-promoted" ] || fallback_line "already-promoted at" "$sha"/'
    mut fail-drops-skipped-names 's/if \[ "\${nskip:-0}" -gt 0 \]; then red_line "\${FL/if false; then red_line "${FL/'
    mut using-line-before-guard 's/^    (cd "\$repo" \&\& {/    fallback_line "using" "$sha"; (cd "$repo" \&\& {/'
    mut promote-exec-fail-second-line 's/else FRC=\$rc FL=using/else inbox_line "RED promote --execute failed"; FRC=$rc FL=using/'
    mut dry-run-using-line-dropped 's/if \[ "\$exe" != 1 \]; then fallback_line "using" "\$sha"; say/if [ "$exe" != 1 ]; then say/'
    mut final-fail-names-dropped 's/^FL="all candidates skipped, last" FS=\$last fail/nskip=0 fail/'
    mut unexpected-exit-net-dropped 's/|| inbox_line "RED promote: exited 1 without a verdict (state \$state)"//'
    mut dispatch-fail-label-dropped 's/FL="clean-room dispatch failed at" //'
    mut exec-rc-dropped 's/FRC=\$rc FL=using/FL=using/'
    mut cleanroom-red-fact-dropped 's/crnote="clean-room run $id ended/crnote="run ended/'
    mut signal-term-trap-dropped "/^trap 'sig=TERM; exit 143' TERM\$/d"
    mut signal-int-trap-dropped "/^trap 'sig=INT; exit 130' INT\$/d"
    mut signal-hup-trap-dropped "/^trap 'sig=HUP; exit 129' HUP\$/d"
    mut killed-line-dropped 's/red_line "killed by SIG\$sig at" "\${cur:-unknown}" ""/:/'
    mut killed-line-needs-no-skip-guard-inverted 's/\[ -z "\$sig" \] || \[ "\${nskip:-0}" -eq 0 \] ||/[ -z "$sig" ] ||/'
    mut breadcrumb-not-written 's/^    \[ "\$nskip" -eq 0 \] || printf .*$/    :/'
    mut stale-breadcrumb-ignored 's/^if \[ -s "\$state\/pending-red" \]; then.*$/:/'
    mut stale-breadcrumb-silent 's/inbox_line "RED previous run died[^;]*;//'
    mut stale-breadcrumb-kept 's/; rm -f -- "\$state\/pending-red"; fi$/; fi/'
    mut breadcrumb-not-removed-on-exit 's/cleanup; rm -f -- "\$state\/pending-red"; /cleanup; /'
    mut dedupe-marker-never-written 's/\[ "\$nskip" -eq 0 \] || : > "\$mk"/:/'
    mut dedupe-ignores-newest-sha 's/fallback-reported-\${cands\[0\]:0:12}/fallback-reported/'
    mut single-candidate-text-changed 's/^\[ "\$nskip" -gt 0 \] || fail "no candidate passed every gate: .*$/:/'
    mut fail-prefixes-notes-again 's/inbox_line "RED promote: \$\* (state \$state)"/inbox_line "RED promote: ${notes:-}$* (state $state)"/'
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
  *"workflow run"*) [ "${FX_DISPATCH_FAIL:-0}" = 1 ] && { echo "HTTP 422 boom"; exit 1; }; exit 0 ;;
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
case "$*" in *--execute*) [ "${FX_PROMOTE_EXEC_RC:-0}" = 0 ] || { echo "promote_nightly: exec failed -- stub" >&2; exit "$FX_PROMOTE_EXEC_RC"; } ;; esac
SH
printf '#!/usr/bin/%s\nexit "${FX_GATE_RC:-0}"\n' env\ bash > "$TMP/gate"
cat > "$TMP/models" <<'SH'
#!/usr/bin/env bash
# stub verifier: state per sha
sha=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = --commit ] && { j=$((i+1)); sha=${!j}; }; done
[ "$sha" != "${FX_KILL_SHA:-none}" ] || kill "-${FX_SIG:-TERM}" "$(cat "$FX_PIDFILE")"
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
case " ${FX_TRAIN_RED_SHA:-none} " in *" $sha "*) echo "NOT RELEASABLE: attempt 2 ${FX_TRAIN_PAD:-}"; exit 1 ;; esac
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
[ -z "\${FX_WIDE:-}" ] || MAX_AGE_DAYS=30
[ -z "\${FX_BOOM:-}" ] || : "\$undefined_boom_var"
[ -z "\${FX_PIDFILE:-}" ] || echo \$\$ > "\$FX_PIDFILE"
EOF
# a harness started under nohup or in the background hands us INT/HUP ignored, and an ignored signal cannot be trapped: reset them
DEFSIG=(); env --default-signal=INT,HUP,TERM true 2> /dev/null && DEFSIG=(env --default-signal=INT,HUP,TERM)
sched() { # name want_rc want_promote want_exec [args...]   (FX_* in the environment)
    local n=$1 wr=$2 wp=$3 we=$4 rc; shift 4
    : > "$TMP/calls"; : > "$INB"; rm -rf "${TMP:?}/pt-flag"; [ -n "${FX_KEEP_STATE:-}" ] || rm -rf "${ST:?}"
    if [ -n "${FX_MARK_PROMOTED:-}" ]; then mkdir -p "$ST/$FX_MARK_PROMOTED"; : > "$ST/$FX_MARK_PROMOTED/already-promoted"; fi
    if [ "${FX_PRESEED:-0}" = 1 ]; then mkdir -p "$ST/${FX_PRESEED_SHA:-$R2}"; echo 999 > "$ST/${FX_PRESEED_SHA:-$R2}/cleanroom-run-id"; fi
    FX_CALLS="$TMP/calls" FX_PT_FLAG="$TMP/pt-flag" FX_PIDFILE="$TMP/sched-pid" FX_R1=$R1 FX_R2=$R2 FX_R0=$R0 \
        "${DEFSIG[@]}" bash "${SCHED_BIN:-$BIN/promote_scheduler.sh}" --repo "$W" --state "$ST" --inbox "$INB" --rate-floor 1000 "$@" > "$TMP/out-$n" 2>&1; rc=$?
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
# C12 (aprender#6677): fallback past a red newer candidate (train-red, not a models-red) must leave a visible RED line naming the skip
inbox_has train-red-fallback-visible "RED fallback: using ${R1:0:9}"
inbox_has train-red-fallback-names-skipped "newer candidate(s) skipped: ${R2:0:9}"
FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 sched train-red-fallback-dry-run 0 1 0
inbox_has train-red-fallback-dry-run-visible "RED fallback"
sched no-fallback-no-red-fallback-line 0 2 1 --execute
grep -q 'RED fallback' "$INB" && bad "all-green run wrote a RED fallback line" || ok
# C12 round 3 (aprender#6677): (1) a cut note keeps its whole "(+N chars cut)" marker inside 300 chars
PAD=$(printf 'p%.0s' $(seq 200))
FX_WIDE=1 FX_TRAIN_PAD=$PAD FX_TRAIN_RED_SHA="$R2 $R1" FX_SHA=$R0 sched fallback-long-notes-cut 0 1 0
inbox_has fallback-long-notes-cut-used "RED fallback: using ${R0:0:9}, 2 newer"
grep 'RED fallback' "$INB" | grep -qE '\(\+[0-9]+ chars cut\)$' && ok || bad "cut marker clipped or missing: $(grep 'RED fallback' "$INB" | tail -c 60)"
expect "fallback line <= 300 chars" 1 "$(grep 'RED fallback' "$INB" | awk 'length($0) <= 300' | wc -l)"
# (2) every path that skips a newer candidate writes the fallback line with the skipped names
g push -q origin "$R2:refs/tags/v1.2.3"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 sched fallback-line-already-promoted 0 0 0 --execute
inbox_has fallback-line-already-promoted "RED fallback: already-promoted at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
g push -q origin ":refs/tags/v1.2.3"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none sched fallback-line-cleanroom-fail 1 0 0 --execute
expect "cleanroom-fail RED count" 1 "$(grep -c "| RED " "$INB")"
inbox_has fallback-line-cleanroom-fail "RED fallback: no clean-room at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none sched fallback-line-cleanroom-dispatched 3 0 0 --execute --dispatch-cleanroom
inbox_has fallback-line-cleanroom-dispatched "RED fallback: clean-room dispatched at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none FX_PRESEED=1 FX_PRESEED_SHA=$R1 FX_RUN_STATE="in_progress " sched fallback-line-cleanroom-pending 3 0 0 --execute --dispatch-cleanroom
inbox_has fallback-line-cleanroom-pending "RED fallback: clean-room pending at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
# (3) a red clean-room run on the skipped sha + fallback to the older one = exactly ONE RED line carrying both facts
FX_PRESEED=1 FX_RUN_STATE="completed failure" FX_SHA=$R1 sched cleanroom-red-fallback-one-line 0 2 1 --execute
expect "cleanroom-red fallback RED line count" 1 "$(grep -c 'RED' "$INB")"
grep 'RED' "$INB" | grep -q "clean-room run 999 ended 'failure'" && grep 'RED' "$INB" | grep -q "RED fallback: using ${R1:0:9}" && ok || bad "single RED line lacks the clean-room fact or the fallback"
# C12 round 4 (aprender#6677): the contract is ONE RED line on EVERY exit path, reported once the outcome is known
red_n() { grep -c '| RED ' "$INB"; }
# (a) an already-promoted marker left by an earlier tick on the OLDER candidate must not silence the fallback past the red newer one
g push -q origin "$R1:refs/tags/v1.2.3"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_MARK_PROMOTED=$R1 sched fallback-line-already-promoted-marker-preseeded 0 0 0 --execute
inbox_has fallback-line-already-promoted-marker-preseeded "RED fallback: already-promoted at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
expect "marker-preseeded RED count" 1 "$(red_n)"
expect "marker-preseeded DONE not repeated" 0 "$(grep -c 'DONE already promoted' "$INB")"
# round 5 (2): deduped, not silenced -- the identical second tick (same promoted sha, same newest red sha) writes NO RED line
FX_KEEP_STATE=1 FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 sched fallback-already-promoted-identical-second-tick 0 0 0 --execute
expect "identical second tick RED count" 0 "$(red_n)"
# ... and a different newest red sha is a different pair: promoted R0, tick A skips R2 and R1 (newest red R2); tick B has R2's bundle gone (newest red R1)
FX_WIDE=1 FX_MODELS_RED_SHA="$R2 $R1" FX_SHA=$R0 sched fallback-pair-a-first-tick 0 0 0 --execute
expect "pair A first tick RED count" 1 "$(red_n)"
FX_WIDE=1 FX_KEEP_STATE=1 FX_MODELS_RED_SHA="$R2 $R1" FX_SHA=$R0 sched fallback-pair-a-second-tick 0 0 0 --execute
expect "pair A second tick RED count" 0 "$(red_n)"
mv -- "$EV/models-crux/$R2" "$TMP/hold-r2"
FX_WIDE=1 FX_KEEP_STATE=1 FX_MODELS_RED_SHA="$R1" FX_SHA=$R0 sched fallback-new-newest-red-sha-writes-again 0 0 0 --execute
mv -- "$TMP/hold-r2" "$EV/models-crux/$R2"
expect "new newest red sha RED count" 1 "$(red_n)"
inbox_has fallback-new-newest-red-sha-writes-again "RED fallback: already-promoted at ${R0:0:9}, 1 newer candidate(s) skipped: ${R1:0:9}"
FX_KEEP_STATE=1 FX_WIDE=1 FX_MODELS_RED_SHA="$R2 $R1" FX_SHA=$R0 sched fallback-pair-a-again-after-b 0 0 0 --execute
expect "pair A after B is already recorded: RED count" 0 "$(red_n)"
g push -q origin ":refs/tags/v1.2.3"
# (c) a clean-room dispatch that fails, or whose run id is unreadable, still names the skipped candidate
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none FX_DISPATCH_FAIL=1 sched cleanroom-dispatch-fails-names-skipped 1 0 0 --execute --dispatch-cleanroom
inbox_has cleanroom-dispatch-fails-names-skipped "RED fallback: clean-room dispatch failed at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
expect "dispatch-fails RED count" 1 "$(red_n)"
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none FX_STALE_RUN=1 sched cleanroom-run-id-unreadable-names-skipped 1 0 0 --execute --dispatch-cleanroom
inbox_has cleanroom-run-id-unreadable-names-skipped "RED fallback: clean-room run id unreadable at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
expect "run-id-unreadable RED count" 1 "$(red_n)"
# (f) every candidate skipped: the final fail line carries the names first
FX_MODELS=red sched all-candidates-skipped-names-first 1 0 0 --execute
inbox_has all-candidates-skipped-names-first "RED fallback: all candidates skipped, last ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9} | why: ${R2:0:9} models not green"
expect "all-skipped RED count" 1 "$(red_n)"
# C12 round 5 (1): a run killed AFTER it skipped a newer candidate still writes ONE RED line naming the skipped sha
for sg in TERM INT HUP; do
    case $sg in TERM) wrc=143 ;; INT) wrc=130 ;; *) wrc=129 ;; esac
    FX_MODELS_RED_SHA=$R2 FX_KILL_SHA=$R1 FX_SIG=$sg FX_SHA=$R1 sched "signal-$sg-after-skip" $wrc 0 0 --execute
    expect "signal $sg after a skip: RED count" 1 "$(red_n)"
    inbox_has "signal-$sg-after-skip" "RED fallback: killed by SIG$sg at ${R1:0:9}, 1 newer candidate(s) skipped: ${R2:0:9}"
    [ ! -e "$ST/pending-red" ] && ok || bad "signal $sg: breadcrumb left behind"
done
# a signal BEFORE any skip owes no fallback line
FX_KILL_SHA=$R2 FX_SIG=TERM FX_SHA=$R2 sched signal-term-before-any-skip 143 0 0 --execute
expect "signal before a skip: RED count" 0 "$(red_n)"
# SIGKILL (OOM) cannot be trapped: the first skip left a breadcrumb; the NEXT tick writes one RED line for it and removes it
mkdir -p "$TMP/leak"
TMPDIR="$TMP/leak" FX_MODELS_RED_SHA=$R2 FX_KILL_SHA=$R1 FX_SIG=KILL FX_SHA=$R1 sched sigkill-after-skip 137 0 0 --execute 2> /dev/null
grep -q "${R2:0:9}" "$ST/pending-red" 2> /dev/null && ok || bad "sigkill: no breadcrumb naming the skipped sha"
rm -rf -- "${TMP:?}/leak"; git -C "$W" worktree prune
FX_KEEP_STATE=1 sched stale-breadcrumb-next-tick 0 1 0
expect "stale breadcrumb: RED count" 1 "$(red_n)"
inbox_has stale-breadcrumb-next-tick "RED previous run died after skipping: ${R2:0:9}"
[ ! -e "$ST/pending-red" ] && ok || bad "stale breadcrumb not removed by the next tick"
FX_KEEP_STATE=1 sched breadcrumb-written-once 0 1 0
expect "breadcrumb is reported once: RED count" 0 "$(red_n)"
# the report-and-remove does not wait for the EXIT trap: a tick that stops before the trap exists (a dirty checkout) must not report the same death twice
mkdir -p "$ST"; printf '%s\n' "${R2:0:9}" > "$ST/pending-red"
printf z >> "$W/f"; FX_KEEP_STATE=1 sched stale-breadcrumb-then-early-stop 1 0 0 --execute; git -C "$W" checkout -q -- f
expect "stale breadcrumb on an early-stopping tick: 'previous run died' lines" 1 "$(grep -c 'previous run died' "$INB")"
FX_KEEP_STATE=1 sched stale-breadcrumb-not-reported-twice 0 1 0
expect "stale breadcrumb already removed: 'previous run died' lines" 0 "$(grep -c 'previous run died' "$INB")"
# a clean run that skipped and fell back leaves no breadcrumb
FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 sched clean-fallback-run-leaves-no-breadcrumb 0 2 1 --execute
expect "clean fallback RED count" 1 "$(red_n)"
[ ! -e "$ST/pending-red" ] && ok || bad "a clean run left its breadcrumb"
# round 5 (3): ONE candidate, nothing skipped: the final line is the pre-diff text from origin/main, byte for byte
OLDMAIN=$(git -C "$HERE" show origin/main:scripts/release/promote_scheduler.sh 2> /dev/null)
if [ -n "$OLDMAIN" ]; then
    grep -qF 'fail "no candidate passed every gate: ${notes:-none}"' <<< "$OLDMAIN" && grep -qF 'inbox_line "RED promote: $* (state $state)"' <<< "$OLDMAIN" && ok || bad "origin/main no longer carries the old single-candidate text"
else printf 'NOTE: origin/main unreadable here; the old-text anchor row is not measured\n' >&2; fi
mv -- "$EV/models-crux/$R1" "$TMP/hold-r1"
FX_MODELS=red sched single-candidate-final-fail-old-text 1 0 0 --execute
mv -- "$TMP/hold-r1" "$EV/models-crux/$R1"
expect "single-candidate final line" "RED promote: no candidate passed every gate: ${R2:0:9} models not green ();  (state $ST)" "$(sed 's/^[^|]*|[^|]*|[^|]*| //' "$INB")"
# the per-path loop: name|env|want_rc|want_promote|want_exec|args -- RED count must be exactly 1 on every one of these exit paths
red_paths=(
    "path-pushtags-guard-fails-after-fallback|FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 FX_PT_OK=0 FX_PT_INSTALL_FAILS=1|1|1|0|--execute"
    "path-pushtags-guard-fails-no-fallback|FX_PT_OK=0 FX_PT_INSTALL_FAILS=1|1|1|0|--execute"
    "path-promote-exec-fails-after-fallback|FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 FX_PROMOTE_EXEC_RC=5|5|2|1|--execute"
    "path-promote-exec-fails-no-fallback|FX_PROMOTE_EXEC_RC=5|5|2|1|--execute"
    "path-dry-run-using-fallback|FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1|0|1|0|"
    "path-execute-using-fallback|FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1|0|2|1|--execute"
    "path-cleanroom-red-then-older|FX_PRESEED=1 FX_RUN_STATE=\"completed failure\" FX_SHA=$R1|0|2|1|--execute"
    "path-cleanroom-missing-after-fallback|FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none|1|0|0|--execute"
    "path-cleanroom-dispatched-after-fallback|FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none|3|0|0|--execute --dispatch-cleanroom"
    "path-cleanroom-pending-after-fallback|FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none FX_PRESEED=1 FX_PRESEED_SHA=$R1 FX_RUN_STATE=\"in_progress \"|3|0|0|--execute --dispatch-cleanroom"
    "path-cleanroom-dispatch-fails-after-fallback|FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 FX_CR=none FX_DISPATCH_FAIL=1|1|0|0|--execute --dispatch-cleanroom"
    "path-cleanroom-run-red-not-pending|FX_CR=none FX_PRESEED=1 FX_RUN_STATE=\"completed failure\"|1|0|0|--execute"
    "path-all-candidates-skipped|FX_MODELS=red|1|0|0|--execute"
    "path-lanes-evidence-missing|FX_LANES=none|1|0|0|--execute"
    "path-unexpected-exit-1|FX_BOOM=1|1|0|0|--execute"
)
for row in "${red_paths[@]}"; do
    IFS='|' read -r pn penv prc pp pe pargs <<< "$row"
    # bashrs disable-next-line=SEC001
    eval "$penv sched $pn $prc $pp $pe $pargs"
    expect "$pn: exactly one RED line" 1 "$(red_n)"
done
inbox_has path-unexpected-exit-1 "RED promote: exited 1 without a verdict"
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
FX_CR=none FX_PRESEED=1 FX_RUN_STATE="completed failure" sched cleanroom-ended-red-not-pending 1 0 0 --execute; inbox_has cleanroom-ended-red "RED fallback: no clean-room at ${R1:0:9}"
inbox_has cleanroom-ended-red-fact "clean-room run 999 ended 'failure'"
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
# the inbox and the floor are required: absent = usage error (2), nothing run
: > "$TMP/calls"; bash "$BIN/promote_scheduler.sh" --repo "$W" --state "$ST" --rate-floor 1000 > /dev/null 2>&1; expect "no --inbox" 2 "$?"
bash "$BIN/promote_scheduler.sh" --repo "$W" --state "$ST" --inbox "$INB" > /dev/null 2>&1; expect "no --rate-floor" 2 "$?"
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
# C12 (aprender#6677) round 2: the fallback is exactly ONE RED line, and a long skip list never loses a name to the 300-char cut
FX_MODELS_RED_SHA=$R2 FX_SHA=$R1 sched newest-models-red-fallback-one-red-line 0 2 1 --execute
expect "newest-models-red-fallback RED line count" 1 "$(grep -c ' RED ' "$INB")"
FX_TRAIN_RED_SHA=$R2 FX_SHA=$R1 sched train-red-fallback-count 0 2 1 --execute
expect "train-red fallback RED line count" 1 "$(grep -c ' RED ' "$INB")"
LONG=""
for k in 1 2 3 4; do
    printf 'q%s' "$k" > "$W/f"; T=$(( $(date +%s) + 600 * k )) # bashrs disable-line=DET002
    GIT_AUTHOR_DATE="@$T" GIT_COMMITTER_DATE="@$T" g commit -q -am "long$k"; L=$(g rev-parse HEAD); LONG+="$L "
    mkdir -p "$EV/models-crux/$L"; printf 'commit=%s\nmeasure=crux\nt1_rc=0\nstate=green\n' "$L" > "$EV/models-crux/$L/verdict"
    printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: ok\n' "${L:0:9}" > "$EV/models-crux/$L/models-t1.log"
done
g push -q origin main
ge() { git -C "$EV" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
ge add -A; ge commit -q -m ev-long; ge push -q origin nightly-evidence
FX_TRAIN_RED_SHA="$LONG" FX_SHA=$R2 sched long-skip-list-fallback 0 2 1 --execute
for L in $LONG; do inbox_has "long-skip-list-names-${L:0:9}" "${L:0:9}"; done
expect "long-skip-list RED line count" 1 "$(grep -c ' RED ' "$INB")"
# lock: a second run while the first holds it does nothing
mkdir -p "$ST"; : > "$TMP/calls"
( exec 8> "$ST/.lock"; flock -n 8; FX_CALLS="$TMP/calls" bash "$BIN/promote_scheduler.sh" --repo "$W" --state "$ST" --inbox "$INB" --rate-floor 1000 --execute > "$TMP/out-lock" 2>&1; echo $? > "$TMP/lock-rc" )
expect "lock rc" 0 "$(cat "$TMP/lock-rc")"; grep -q 'holds the lock' "$TMP/out-lock" && ok || bad "lock message"
expect "lock: nothing ran" 0 "$(wc -l < "$TMP/calls")"
inbox_has lock-held SKIP
printf 'falsify_promote_scheduler: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
