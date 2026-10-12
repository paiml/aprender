#!/usr/bin/env bash
# check_release_t1_lanes_joined.sh -- the T-1 lanes in scripts/release/autopilot.sh. A guard, not a
# release script, so it lives in scripts/ beside check_release_draft_gated.sh.
#
# dogfood and models start together and join before readiness. This guard extracts the join
# block from the autopilot (from "# The join." to its closing "fi"), runs it against stub lanes and
# the autopilot's own say/die shape, and asserts per case:
#   green      both GO            -> rc 0, the pass continues, one GO row per lane in t1-steps.tsv
#   red        dogfood red        -> rc 1, nothing after the join runs; models is never stopped and
#                                    runs to its own verdict (GO)
#   stop       models red         -> rc 1; dogfood, still running, is STOPPED with its child gone,
#                                    well before it would have ended
#   two-reds   dogfood and models -> rc 1, both RED, both named in the STOP line
#   one        models alone, red  -> rc 1, one RED row (from=to runs that lane alone)
#   solo       dogfood alone      -> rc 0, one GO row
#   overlap    two 3 s lanes      -> rc 0 in under 5 s (in series they need 6)
#   interrupt  TERM to the run    -> rc 1 within seconds, nothing after the join, dogfood's child gone
#   self-red   models red; dogfood, signalled, exits 1 by itself -> dogfood RED (not STOPPED), both named
# and under the standing release policy, where the pre-publish dogfood is not run (#5174):
#   policy-both        both selected -> rc 0, models alone runs (one GO row), dogfood never started,
#                                       one NOT-RUN line
#   policy-models-red  models red    -> rc 1, dogfood never started
#   policy-solo        dogfood alone -> rc 0, no lane started, the NOT-RUN line
# and, on the autopilot text itself, that dogfood builds into its own target dir under
# $REPO_ROOT/target while models keeps CARGO_TARGET_DIR (readiness reads the apr models built), and
# that scripts/check_model_parity.sh runs every apr parity under the ladder's GPU lock (dogfood's C14
# parity and the ladder now share the GPU).
# Then it re-runs everything against mutants and requires each to turn it RED.
# MODE. guard_tree runs every scripts/check_*.sh, this one included. REPORT by default (L31: a new
# check blocks only after three green nights; the red, overlap and interrupt cases also read the wall
# clock): a wrong case or a surviving mutant prints a REPORT line and exits 0.
# RELEASE_T1_LANES_ENFORCE=1 makes it exit 1. ENV (rc 2) is rc 2 in both modes.
# Exit 0 = PASS (or report mode), 1 = FAIL under ENFORCE, 2 = could not run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV: not in a git checkout"; exit 2; }
AUTOPILOT=${1:-$ROOT/scripts/release/autopilot.sh}
PARITY=$ROOT/scripts/check_model_parity.sh
[ -f "$AUTOPILOT" ] || { echo "ENV: no autopilot at $AUTOPILOT"; exit 2; }
[ -f "$PARITY" ] || { echo "ENV: no $PARITY"; exit 2; }
TMP=$(mktemp -d) || { echo "ENV: mktemp failed"; exit 2; }
trap 'rm -rf -- "${TMP:?}"' EXIT

# finish BAD: the verdict in the current mode (see MODE above).
finish() {
    [ "$1" = 0 ] && exit 0
    [ "${RELEASE_T1_LANES_ENFORCE:-0}" = 1 ] && exit 1
    echo "REPORT: a case or a mutant landed wrong -- report mode (L31: blocks only after three green nights; RELEASE_T1_LANES_ENFORCE=1 enforces)"
    exit 0
}

# The harness: say/die as the autopilot defines them, run_step from $SEL, and lanes that sleep then
# go red or green. dogfood holds a child (as a lane holds cargo) so a kill that reaches only the lane
# and not its process group leaves the child behind, and the case sees it.
cat > "$TMP/harness.sh" <<'EOF'
set -uo pipefail
AP=$APD; STATUS=$AP/STATUS; LOG=$AP/LOG
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS" >> "$LOG"; }
die() { say "STOP $*"; exit 1; }
run_step() { case " $SEL " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
lane() { sleep "$2"; [ "$3" = 0 ] || die "$1 red"; say "$1 GO"; }
t1_dogfood() {
    : > "$AP/dogfood-ran"
    if [ "${DT:-0}" = 1 ]; then trap 'exit 1' TERM; sleep 5 & wait; exit 1; fi
    sleep "$GD" & echo "$!" > "$AP/dogfood-child"; wait; lane dogfood 0 "$GR"
}
t1_models() { lane models "$MD" "$MR"; }
# shellcheck disable=SC1090
. "$JOIN"
say AFTER-JOIN
EOF

# case_run JOIN NAME SEL GD GR MD MR [TERM-AFTER-S] -> C_RC C_SECS; the run's AP is $TMP/NAME
case_run() {
    local j=$1 n=$2 t0 pid
    mkdir -p "$TMP/$n" || return 2
    t0=$SECONDS
    ( export APD="$TMP/$n" JOIN="$j" SEL="$3" GD=$4 GR=$5 MD=$6 MR=$7 DT="${DT:-0}" AP_POLICY="${AP_POLICY:-0}"
      exec bash "$TMP/harness.sh" ) > "$TMP/$n/out.log" 2>&1 &
    pid=$!
    if [ -n "${8:-}" ]; then sleep "${8}"; kill -TERM "$pid" 2>/dev/null; fi
    wait "$pid"; C_RC=$?; C_SECS=$((SECONDS - t0))
}
verdict() { awk -F'\t' -v s="$2" '$1==s {print $5}' "$TMP/$1/t1-steps.tsv" 2>/dev/null; }
col() { awk -F'\t' -v s="$2" -v c="$3" '$1==s {print $c}' "$TMP/$1/t1-steps.tsv" 2>/dev/null; }
ran() { [ -f "$TMP/$1/dogfood-ran" ]; }
notrun() { [ "$(grep -c 'STEP dogfood NOT-RUN: the standing release policy' "$TMP/$1/STATUS" 2>/dev/null)" = 1 ]; }
rows() { tail -n +2 "$TMP/$1/t1-steps.tsv" 2>/dev/null | wc -l; }
after() { grep -q 'AFTER-JOIN' "$TMP/$1/STATUS" 2>/dev/null; }
child_gone() { local c; c=$(cat "$TMP/$1/dogfood-child" 2>/dev/null); [ -n "$c" ] && ! kill -0 "$c" 2>/dev/null; }

# table JOIN TAG -> 0 when every case holds, else 1 with the first broken case printed
table() {
    local j=$1 t=$2
    case_run "$j" "$t-green" "dogfood models" 1 0 2 0
    [ "$C_RC" = 0 ] && after "$t-green" && [ "$(rows "$t-green")" = 2 ] \
        && [ "$(verdict "$t-green" dogfood)$(verdict "$t-green" models)" = GOGO ] \
        || { echo "green: rc=$C_RC rows=$(rows "$t-green") (want rc 0, AFTER-JOIN, 2 GO rows)"; return 1; }
    case_run "$j" "$t-red" "dogfood models" 1 1 3 0
    [ "$C_RC" = 1 ] && ! after "$t-red" && [ "$C_SECS" -lt 10 ] && [ "$C_SECS" -ge 3 ] \
        && [ "$(verdict "$t-red" dogfood)" = RED ] && [ "$(verdict "$t-red" models)" = GO ] \
        || { echo "red: rc=$C_RC secs=$C_SECS dogfood=$(verdict "$t-red" dogfood) models=$(verdict "$t-red" models) (want rc 1, 3-9 s, dogfood RED, models GO)"; return 1; }
    case_run "$j" "$t-stop" "dogfood models" 20 0 1 1
    [ "$C_RC" = 1 ] && ! after "$t-stop" && [ "$C_SECS" -lt 10 ] \
        && [ "$(verdict "$t-stop" models)" = RED ] && [ "$(verdict "$t-stop" dogfood)" = STOPPED ] && child_gone "$t-stop" \
        || { echo "stop: rc=$C_RC secs=$C_SECS dogfood=$(verdict "$t-stop" dogfood) models=$(verdict "$t-stop" models) (want rc 1, < 10 s, models RED, dogfood STOPPED with its child gone)"; return 1; }
    case_run "$j" "$t-two" "dogfood models" 1 1 3 1
    [ "$C_RC" = 1 ] && ! after "$t-two" && [ "$(verdict "$t-two" models)" = RED ] \
        && grep -q 'STOP .*dogfood models RED' "$TMP/$t-two/STATUS" \
        || { echo "two-reds: rc=$C_RC models=$(verdict "$t-two" models) (want rc 1, models RED, both named)"; return 1; }
    case_run "$j" "$t-one" "models" 0 0 1 1
    [ "$C_RC" = 1 ] && ! after "$t-one" && [ "$(rows "$t-one")" = 1 ] && [ "$(verdict "$t-one" models)" = RED ] \
        || { echo "one: rc=$C_RC rows=$(rows "$t-one") (want rc 1, one RED row)"; return 1; }
    case_run "$j" "$t-solo" "dogfood" 1 0 0 0
    [ "$C_RC" = 0 ] && after "$t-solo" && [ "$(rows "$t-solo")" = 1 ] \
        || { echo "solo: rc=$C_RC rows=$(rows "$t-solo") (want rc 0, one GO row)"; return 1; }
    case_run "$j" "$t-overlap" "dogfood models" 3 0 3 0
    [ "$C_RC" = 0 ] && [ "$C_SECS" -lt 5 ] \
        || { echo "overlap: rc=$C_RC secs=$C_SECS (want rc 0 in < 5 s; in series 6 s)"; return 1; }
    case_run "$j" "$t-int" "dogfood models" 20 0 20 0 1
    [ "$C_RC" != 0 ] && ! after "$t-int" && [ "$C_SECS" -lt 10 ] && child_gone "$t-int" \
        || { echo "interrupt: rc=$C_RC secs=$C_SECS (want non-zero in < 10 s, no AFTER-JOIN, dogfood's child gone)"; return 1; }
    DT=1 case_run "$j" "$t-self" "dogfood models" 0 0 1 1
    [ "$C_RC" = 1 ] && [ "$(verdict "$t-self" dogfood)" = RED ] && grep -q 'STOP .*models dogfood RED' "$TMP/$t-self/STATUS" \
        || { echo "self-red: rc=$C_RC dogfood=$(verdict "$t-self" dogfood) (want dogfood RED: it exited 1 by itself, not by the signal)"; return 1; }
    # #5174: under the standing release policy the pre-publish dogfood is not run -- models runs alone.
    AP_POLICY=1 case_run "$j" "$t-p-both" "dogfood models" 1 0 1 0
    [ "$C_RC" = 0 ] && after "$t-p-both" && ! ran "$t-p-both" && [ "$(rows "$t-p-both")" = 1 ] \
        && [ "$(verdict "$t-p-both" models)" = GO ] && notrun "$t-p-both" \
        || { echo "policy-both: rc=$C_RC rows=$(rows "$t-p-both") models=$(verdict "$t-p-both" models) (want rc 0, models GO alone, dogfood never started, one NOT-RUN line)"; return 1; }
    AP_POLICY=1 case_run "$j" "$t-p-mred" "dogfood models" 1 0 1 1
    [ "$C_RC" = 1 ] && ! after "$t-p-mred" && ! ran "$t-p-mred" && [ "$(verdict "$t-p-mred" models)" = RED ] \
        || { echo "policy-models-red: rc=$C_RC models=$(verdict "$t-p-mred" models) (want rc 1, models RED, dogfood never started)"; return 1; }
    AP_POLICY=1 case_run "$j" "$t-p-solo" "dogfood" 1 0 0 0
    [ "$C_RC" = 0 ] && after "$t-p-solo" && ! ran "$t-p-solo" && [ "$(rows "$t-p-solo")" = 0 ] && notrun "$t-p-solo" \
        || { echo "policy-solo: rc=$C_RC rows=$(rows "$t-p-solo") (want rc 0, no lane started, the NOT-RUN line)"; return 1; }
    return 0
}

# locked PARITY -> 0 when every apr parity in check_model_parity.sh runs under the ladder's GPU lock
locked() {
    local n l
    n=$(grep -c '"\$APR_BIN" parity' "$1")
    l=$(grep -B1 '"\$APR_BIN" parity' "$1" | grep -c 'flock -E 75 -w "\${MODEL_LADDER_LOCK_WAIT:-1800}" "\${MODEL_LADDER_GPU_LOCK:-/tmp/apr-gpu.lock}"')
    [ "$n" -ge 1 ] && [ "$n" = "$l" ] || { echo "check_model_parity.sh: $l of $n apr parity call(s) take the ladder's GPU lock"; return 1; }
}

# dirs AUTOPILOT -> 0 when dogfood exports its own target dir and models exports none
dirs() {
    local g m
    g=$(awk '/^t1_dogfood\(\) \{/{p=1} p&&/export CARGO_TARGET_DIR=/{print; exit} p&&/^\}/{exit}' "$1")
    m=$(awk '/^t1_models\(\) \{/{p=1} p&&/CARGO_TARGET_DIR=/{print; exit} p&&/^\}/{exit}' "$1")
    case "$g" in *'"$REPO_ROOT/target/'?*'"') ;; *) echo "dogfood builds into the shared target dir: '$g'"; return 1 ;; esac
    [ -z "$m" ] || { echo "models moved off CARGO_TARGET_DIR, which readiness reads: $m"; return 1; }
}

# nogo AUTOPILOT -> 0 when every line that can print "dogfood GO" swaps it out under the policy (#5174):
# the dogfood is not run then, so a receipt that names a dogfood GO would be a green nothing measured.
# The post-publish dogfood still runs after the publish, so its GO is struck out before the match.
nogo() {
    local bad
    bad=$(sed 's/post-publish dogfood GO//g' "$1" | grep -n 'dogfood GO' | grep -v '^[0-9]*:[[:space:]]*#' \
        | grep -vF '[ "$AP_POLICY" != 1 ] || { AP_DF_RECEIPT="dogfood not run (policy)"; AP_DF_CLOSE="pre-publish dogfood not run' || true)
    [ -z "$bad" ] || { echo "a line prints dogfood GO with no policy swap: $(printf '%s\n' "$bad" | head -n 1 | cut -c1-100)"; return 1; }
}

# check AUTOPILOT TAG -> 0 when the join and the dirs hold
check() {
    awk '/^# The join\./{p=1} p{print} p&&/^fi$/{exit}' "$1" > "$TMP/$2.join"
    grep -q 'wait -n' "$TMP/$2.join" || { echo "no join block (\"# The join.\" .. fi with wait -n)"; return 1; }
    dirs "$1" && nogo "$1" && locked "${3:-$PARITY}" && table "$TMP/$2.join" "$2"
}

bad=0
if why=$(check "$AUTOPILOT" real); then echo "ok    the join holds (12 cases), the dogfood lane is not started under the policy and no line claims a dogfood GO then, dogfood's target dir is its own, parity takes the GPU lock"
else echo "FAIL  the autopilot's T-1 lanes: $why"; bad=1; fi

# The mutants run whatever the real tree did, so report mode prints every line before its verdict.
fails=0; nm=0
mutant() { # mutant NAME SED-EXPR
    nm=$((nm + 1))
    sed -e "$2" "$AUTOPILOT" > "$TMP/m-$1.sh"
    cmp -s "$AUTOPILOT" "$TMP/m-$1.sh" && { echo "FAIL  mutant $1 did not apply (anchor moved)"; fails=$((fails + 1)); return; }
    if why=$(check "$TMP/m-$1.sh" "m-$1"); then echo "FAIL  mutant $1 survived"; fails=$((fails + 1))
    else echo "ok    mutant $1 killed ($why)" | cut -c1-160; fi
}
mutant no-die      's/^  \[ -z "\$t1_red" \] || die /  [ -z "$t1_red" ] || say /'
mutant no-kill     's/kill -TERM -- "-\$p" 2> \/dev\/null$/: "$p"/'
mutant stop-models '/\[ "\${T1_STEP\[\$p\]}" = models \] && cont[i]nue/d'
mutant no-rows     's/>> "\$AP\/t1-steps.tsv"/> \/dev\/null/'
mutant serial      's/"t1_\$s" & T1_STEP\[\$!\]=\$s;/( "t1_$s" ); : \& T1_STEP[$!]=$s;/'
mutant first-red   's/t1_red="\${t1_red:+\$t1_red }\$s"/t1_red=${t1_red:-$s}/'
mutant no-trap     "/^  trap 'for p in/d"
mutant shared-dogfood '/^  export CARGO_TARGET_DIR="\$REPO_ROOT\/target\/t1-dogfood"$/d'
mutant stopped-any-rc 's/ \&\& \[ "\$rc" -eq "\$t1_term_rc" \]; then v=STOPPED/; then v=STOPPED/'
mutant policy-runs-dogfood 's/ \&\& \[ "\$s:\$AP_POLICY" != dogfood:1 \]//'
mutant policy-drops-every-lane 's/\[ "\$s:\$AP_POLICY" != dogfood:1 \]/[ "$AP_POLICY" != 1 ]/'
mutant no-not-run-line '/^\[ "\$AP_POLICY" != 1 \] || ! run_step dogfood || say /d'
mutant dryrun-claims-go 's/receipts: \$AP_DF_RECEIPT, tag/receipts: dogfood GO, tag/'
mutant close-claims-go 's/Receipts: \$AP_DF_CLOSE, all/Receipts: pre-publish dogfood GO, all/'
mutant no-policy-swap 's/; \[ "\$AP_POLICY" != 1 \] || { AP_DF_RECEIPT="dogfood not run (policy)".*$//'
nm=$((nm + 1))
sed -e 's/flock -E 75 -w "\${MODEL_LADDER_LOCK_WAIT:-1800}" "\${MODEL_LADDER_GPU_LOCK:-\/tmp\/apr-gpu.lock}"/env/' "$PARITY" > "$TMP/m-parity.sh"
if cmp -s "$PARITY" "$TMP/m-parity.sh"; then echo "FAIL  mutant parity-unlocked did not apply (anchor moved)"; fails=$((fails + 1))
elif why=$(check "$AUTOPILOT" m-parity "$TMP/m-parity.sh"); then echo "FAIL  mutant parity-unlocked survived"; fails=$((fails + 1))
else echo "ok    mutant parity-unlocked killed ($why)"; fi
[ "$fails" -eq 0 ] || { echo "FAIL  $fails of $nm mutant(s)"; finish 1; }
[ "$bad" = 0 ] || finish 1
echo "PASS  12 case(s) + target dirs + no dogfood GO under the policy + parity lock, $nm mutant(s) killed: dogfood and models start together (under the policy models runs alone, #5174), join before readiness, and a red in either stops the pass"
finish 0
