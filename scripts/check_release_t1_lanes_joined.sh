#!/usr/bin/env bash
# check_release_t1_lanes_joined.sh -- the T-1 lanes in scripts/release/autopilot.sh. A guard, not a
# release script, so it lives in scripts/ beside check_release_draft_gated.sh.
#
# deep, dogfood and models start together and join before readiness. This guard extracts the join
# block from the autopilot (from "# The join." to its closing "fi"), runs it against stub lanes and
# the autopilot's own say/die shape, and asserts per case:
#   green      all three GO       -> rc 0, the pass continues, one GO row per lane in t1-steps.tsv
#   red        dogfood red        -> rc 1, nothing after the join runs; deep, still running, is
#                                    STOPPED with its child gone, well before it would have ended;
#                                    models is never stopped and runs to its own verdict (GO)
#   two-reds   dogfood and models -> rc 1, both RED, both named in the STOP line
#   one        models alone, red  -> rc 1, one RED row (from=to runs that lane alone)
#   solo       dogfood alone      -> rc 0, one GO row
#   overlap    three 2 s lanes    -> rc 0 in under 5 s (in series they need 6)
#   interrupt  TERM to the run    -> rc 1 within seconds, nothing after the join, deep's child gone
# and, on the autopilot text itself, that deep and dogfood build into their own target dirs under
# $REPO_ROOT/target while models keeps CARGO_TARGET_DIR (readiness reads the apr models built).
# Then it re-runs everything against mutants and requires each to turn it RED.
# Exit 0 = PASS, 1 = FAIL, 2 = could not run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV: not in a git checkout"; exit 2; }
AUTOPILOT=${1:-$ROOT/scripts/release/autopilot.sh}
[ -f "$AUTOPILOT" ] || { echo "ENV: no autopilot at $AUTOPILOT"; exit 2; }
TMP=$(mktemp -d) || { echo "ENV: mktemp failed"; exit 2; }
trap 'rm -rf -- "${TMP:?}"' EXIT

# The harness: say/die as the autopilot defines them, run_step from $SEL, and lanes that sleep then
# go red or green. deep holds a child (as a lane holds cargo) so a kill that reaches only the lane
# and not its process group leaves the child behind, and the case sees it.
cat > "$TMP/harness.sh" <<'EOF'
set -uo pipefail
AP=$APD; STATUS=$AP/STATUS; LOG=$AP/LOG
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS" >> "$LOG"; }
die() { say "STOP $*"; exit 1; }
run_step() { case " $SEL " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
lane() { sleep "$2"; [ "$3" = 0 ] || die "$1 red"; say "$1 GO"; }
t1_deep() { sleep "$DD" & echo "$!" > "$AP/deep-child"; wait; lane deep 0 "$DR"; }
t1_dogfood() { lane dogfood "$GD" "$GR"; }
t1_models() { lane models "$MD" "$MR"; }
# shellcheck disable=SC1090
. "$JOIN"
say AFTER-JOIN
EOF

# case_run JOIN NAME SEL DD DR GD GR MD MR [TERM-AFTER-S] -> C_RC C_SECS; the run's AP is $TMP/NAME
case_run() {
    local j=$1 n=$2 t0 pid
    mkdir -p "$TMP/$n" || return 2
    t0=$(date +%s)
    ( export APD="$TMP/$n" JOIN="$j" SEL="$3" DD=$4 DR=$5 GD=$6 GR=$7 MD=$8 MR=$9
      exec bash "$TMP/harness.sh" ) > "$TMP/$n/out.log" 2>&1 &
    pid=$!
    if [ -n "${10:-}" ]; then sleep "${10}"; kill -TERM "$pid" 2>/dev/null; fi
    wait "$pid"; C_RC=$?; C_SECS=$(( $(date +%s) - t0 ))
}
verdict() { awk -F'\t' -v s="$2" '$1==s {print $5}' "$TMP/$1/t1-steps.tsv" 2>/dev/null; }
rows() { tail -n +2 "$TMP/$1/t1-steps.tsv" 2>/dev/null | wc -l; }
after() { grep -q 'AFTER-JOIN' "$TMP/$1/STATUS" 2>/dev/null; }
child_gone() { local c; c=$(cat "$TMP/$1/deep-child" 2>/dev/null); [ -n "$c" ] && ! kill -0 "$c" 2>/dev/null; }

# table JOIN TAG -> 0 when every case holds, else 1 with the first broken case printed
table() {
    local j=$1 t=$2
    case_run "$j" "$t-green" "deep dogfood models" 1 0 1 0 2 0
    [ "$C_RC" = 0 ] && after "$t-green" && [ "$(rows "$t-green")" = 3 ] \
        && [ "$(verdict "$t-green" deep)$(verdict "$t-green" dogfood)$(verdict "$t-green" models)" = GOGOGO ] \
        || { echo "green: rc=$C_RC rows=$(rows "$t-green") (want rc 0, AFTER-JOIN, 3 GO rows)"; return 1; }
    case_run "$j" "$t-red" "deep dogfood models" 20 0 1 1 3 0
    [ "$C_RC" = 1 ] && ! after "$t-red" && [ "$C_SECS" -lt 10 ] && [ "$C_SECS" -ge 3 ] \
        && [ "$(verdict "$t-red" dogfood)" = RED ] && [ "$(verdict "$t-red" deep)" = STOPPED ] \
        && [ "$(verdict "$t-red" models)" = GO ] && child_gone "$t-red" \
        || { echo "red: rc=$C_RC secs=$C_SECS deep=$(verdict "$t-red" deep) dogfood=$(verdict "$t-red" dogfood) models=$(verdict "$t-red" models) (want rc 1, 3-9 s, dogfood RED, deep STOPPED with its child gone, models GO)"; return 1; }
    case_run "$j" "$t-two" "deep dogfood models" 1 0 1 1 3 1
    [ "$C_RC" = 1 ] && ! after "$t-two" && [ "$(verdict "$t-two" models)" = RED ] \
        && grep -q 'STOP .*dogfood models RED' "$TMP/$t-two/STATUS" \
        || { echo "two-reds: rc=$C_RC models=$(verdict "$t-two" models) (want rc 1, models RED, both named)"; return 1; }
    case_run "$j" "$t-one" "models" 0 0 0 0 1 1
    [ "$C_RC" = 1 ] && ! after "$t-one" && [ "$(rows "$t-one")" = 1 ] && [ "$(verdict "$t-one" models)" = RED ] \
        || { echo "one: rc=$C_RC rows=$(rows "$t-one") (want rc 1, one RED row)"; return 1; }
    case_run "$j" "$t-solo" "dogfood" 0 0 1 0 0 0
    [ "$C_RC" = 0 ] && after "$t-solo" && [ "$(rows "$t-solo")" = 1 ] \
        || { echo "solo: rc=$C_RC rows=$(rows "$t-solo") (want rc 0, one GO row)"; return 1; }
    case_run "$j" "$t-overlap" "deep dogfood models" 2 0 2 0 2 0
    [ "$C_RC" = 0 ] && [ "$C_SECS" -lt 5 ] \
        || { echo "overlap: rc=$C_RC secs=$C_SECS (want rc 0 in < 5 s; in series 6 s)"; return 1; }
    case_run "$j" "$t-int" "deep dogfood models" 20 0 20 0 20 0 1
    [ "$C_RC" != 0 ] && ! after "$t-int" && [ "$C_SECS" -lt 10 ] && child_gone "$t-int" \
        || { echo "interrupt: rc=$C_RC secs=$C_SECS (want non-zero in < 10 s, no AFTER-JOIN, deep's child gone)"; return 1; }
    return 0
}

# dirs AUTOPILOT -> 0 when deep and dogfood export their own target dirs and models exports none
dirs() {
    local d g m
    d=$(awk '/^t1_deep\(\) \{/{p=1} p&&/export CARGO_TARGET_DIR=/{print; exit} p&&/^\}/{exit}' "$1")
    g=$(awk '/^t1_dogfood\(\) \{/{p=1} p&&/export CARGO_TARGET_DIR=/{print; exit} p&&/^\}/{exit}' "$1")
    m=$(awk '/^t1_models\(\) \{/{p=1} p&&/CARGO_TARGET_DIR=/{print; exit} p&&/^\}/{exit}' "$1")
    case "$d" in *'"$REPO_ROOT/target/'?*'"') ;; *) echo "deep builds into the shared target dir: '$d'"; return 1 ;; esac
    case "$g" in *'"$REPO_ROOT/target/'?*'"') ;; *) echo "dogfood builds into the shared target dir: '$g'"; return 1 ;; esac
    [ "$d" != "$g" ] || { echo "deep and dogfood share one target dir: $d"; return 1; }
    [ -z "$m" ] || { echo "models moved off CARGO_TARGET_DIR, which readiness reads: $m"; return 1; }
}

# check AUTOPILOT TAG -> 0 when the join and the dirs hold
check() {
    awk '/^# The join\./{p=1} p{print} p&&/^fi$/{exit}' "$1" > "$TMP/$2.join"
    grep -q 'wait -n' "$TMP/$2.join" || { echo "no join block (\"# The join.\" .. fi with wait -n)"; return 1; }
    dirs "$1" && table "$TMP/$2.join" "$2"
}

if why=$(check "$AUTOPILOT" real); then echo "ok    the join holds (7 cases) and the target dirs are split"
else echo "FAIL  the autopilot's T-1 lanes: $why"; exit 1; fi

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
mutant stop-models '/\[ "\${T1_STEP\[\$p\]}" = models \] && continue/d'
mutant no-rows     's/>> "\$AP\/t1-steps.tsv"/> \/dev\/null/'
mutant serial      's/"t1_\$s" & T1_STEP\[\$!\]=\$s;/( "t1_$s" ); : \& T1_STEP[$!]=$s;/'
mutant first-red   's/t1_red="\${t1_red:+\$t1_red }\$s"/t1_red=${t1_red:-$s}/'
mutant no-trap     "/^  trap 'for p in/d"
mutant shared-deep '/^  export CARGO_TARGET_DIR="\$REPO_ROOT\/target\/t1-deep"$/d'
mutant shared-dogfood 's/target\/t1-dogfood"$/target\/t1-deep"/'
[ "$fails" -eq 0 ] || { echo "FAIL  $fails of $nm mutant(s)"; exit 1; }
echo "PASS  7 case(s) + target dirs, $nm mutant(s) killed: deep, dogfood and models start together, join before readiness, and a red in any stops the pass"
