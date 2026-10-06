#!/usr/bin/env bash
# check_release_t1_lanes_joined.sh -- the T-1 lanes in scripts/release/autopilot.sh. A guard, not a
# release script, so it lives in scripts/ beside check_release_draft_gated.sh.
#
# deep, dogfood and models start together and join before readiness. This guard extracts the join
# block from the autopilot (from "# The join." to its closing "fi"), runs it against stub lanes and
# the autopilot's own say/die shape, and asserts per case:
#   - all green           -> rc 0, the pass continues, one GO row per lane in t1-steps.tsv
#   - one lane red        -> rc 1, nothing after the join runs, that lane RED, a lane still running
#                            is STOPPED (killed with its children), well before it would have ended
#   - a single step       -> from=to runs that lane alone, as before
#   - the lanes overlap   -> three 2 s lanes finish in under 5 s (in series they need 6)
# Then it re-runs the table against mutants of the block (no final die, no kill, no rows, serial
# start) and requires every mutant to turn it RED. Exit 0 = PASS, 1 = FAIL, 2 = could not run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV: not in a git checkout"; exit 2; }
AUTOPILOT=${1:-$ROOT/scripts/release/autopilot.sh}
[ -f "$AUTOPILOT" ] || { echo "ENV: no autopilot at $AUTOPILOT"; exit 2; }
TMP=$(mktemp -d) || { echo "ENV: mktemp failed"; exit 2; }
trap 'rm -rf -- "${TMP:?}"' EXIT

awk '/^# The join\./{p=1} p{print} p&&/^fi$/{exit}' "$AUTOPILOT" > "$TMP/join.sh"
grep -q 'wait -n' "$TMP/join.sh" || { echo "FAIL  no join block (\"# The join.\" .. fi with wait -n) in $AUTOPILOT"; exit 1; }

# The harness: say/die as the autopilot defines them, run_step from $SEL, and lanes that sleep then
# go red or green. The models lane holds a child (as models_t1.sh holds cargo and ssh) so a kill that
# reaches only the lane and not its group leaves the child, and the case sees it.
cat > "$TMP/harness.sh" <<'EOF'
set -uo pipefail
AP=$APD; STATUS=$AP/STATUS; LOG=$AP/LOG
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS" >> "$LOG"; }
die() { say "STOP $*"; exit 1; }
run_step() { case " $SEL " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
lane() { sleep "$2"; [ "$3" = 0 ] || die "$1 red"; say "$1 GO"; }
t1_deep() { lane deep "$DD" "$DR"; }
t1_dogfood() { lane dogfood "$GD" "$GR"; }
t1_models() { sleep "$MD" & echo "$!" > "$AP/models-child"; wait; lane models 0 "$MR"; }
# shellcheck disable=SC1090
. "$JOIN"
say AFTER-JOIN
EOF

# case JOIN NAME SEL DD DR GD GR MD MR -> sets C_RC C_SECS; the run's AP is $TMP/NAME
case_run() {
    local j=$1 n=$2 t0
    mkdir -p "$TMP/$n" || return 2
    t0=$(date +%s)
    ( export APD="$TMP/$n" JOIN="$j" SEL="$3" DD=$4 DR=$5 GD=$6 GR=$7 MD=$8 MR=$9
      bash "$TMP/harness.sh" ) > "$TMP/$n/out.log" 2>&1
    C_RC=$?; C_SECS=$(( $(date +%s) - t0 ))
}
verdict() { awk -F'\t' -v s="$2" '$1==s {print $5}' "$TMP/$1/t1-steps.tsv" 2>/dev/null; }
rows() { tail -n +2 "$TMP/$1/t1-steps.tsv" 2>/dev/null | wc -l; }
after() { grep -q 'AFTER-JOIN' "$TMP/$1/STATUS" 2>/dev/null; }

# table JOIN TAG -> 0 when every case holds, else 1 with the first broken case printed
table() {
    local j=$1 t=$2 c
    case_run "$j" "$t-green" "deep dogfood models" 1 0 1 0 2 0
    [ "$C_RC" = 0 ] && after "$t-green" && [ "$(rows "$t-green")" = 3 ] \
        && [ "$(verdict "$t-green" deep)$(verdict "$t-green" dogfood)$(verdict "$t-green" models)" = GOGOGO ] \
        || { echo "green: rc=$C_RC rows=$(rows "$t-green") (want rc 0, AFTER-JOIN, 3 GO rows)"; return 1; }
    case_run "$j" "$t-red" "deep dogfood models" 1 0 1 1 20 0
    c=$(cat "$TMP/$t-red/models-child" 2>/dev/null)
    [ "$C_RC" = 1 ] && ! after "$t-red" && [ "$C_SECS" -lt 10 ] \
        && [ "$(verdict "$t-red" dogfood)" = RED ] && [ "$(verdict "$t-red" models)" = STOPPED ] \
        && [ "$(verdict "$t-red" deep)" = GO ] && [ -n "$c" ] && ! kill -0 "$c" 2>/dev/null \
        || { echo "red: rc=$C_RC secs=$C_SECS rows=$(rows "$t-red") dogfood=$(verdict "$t-red" dogfood) models=$(verdict "$t-red" models) (want rc 1, no AFTER-JOIN, < 10 s, dogfood RED, models STOPPED with its child gone)"; return 1; }
    case_run "$j" "$t-one" "models" 0 0 0 0 1 1
    [ "$C_RC" = 1 ] && ! after "$t-one" && [ "$(rows "$t-one")" = 1 ] && [ "$(verdict "$t-one" models)" = RED ] \
        || { echo "one-red: rc=$C_RC rows=$(rows "$t-one") (want rc 1, one RED row)"; return 1; }
    case_run "$j" "$t-solo" "dogfood" 0 0 1 0 0 0
    [ "$C_RC" = 0 ] && after "$t-solo" && [ "$(rows "$t-solo")" = 1 ] \
        || { echo "solo: rc=$C_RC rows=$(rows "$t-solo") (want rc 0, one GO row)"; return 1; }
    case_run "$j" "$t-overlap" "deep dogfood models" 2 0 2 0 2 0
    [ "$C_RC" = 0 ] && [ "$C_SECS" -lt 5 ] \
        || { echo "overlap: rc=$C_RC secs=$C_SECS (want rc 0 in < 5 s; in series 6 s)"; return 1; }
    return 0
}

if why=$(table "$TMP/join.sh" real); then echo "ok    the join holds: green, red-stops-siblings, single step, overlap"
else echo "FAIL  the autopilot's join: $why"; exit 1; fi

fails=0
mutant() { # mutant NAME SED-EXPR
    sed -e "$2" "$TMP/join.sh" > "$TMP/m-$1.sh"
    cmp -s "$TMP/join.sh" "$TMP/m-$1.sh" && { echo "FAIL  mutant $1 did not apply (anchor moved)"; fails=$((fails + 1)); return; }
    if why=$(table "$TMP/m-$1.sh" "m-$1"); then echo "FAIL  mutant $1 survived"; fails=$((fails + 1))
    else echo "ok    mutant $1 killed ($why)" | cut -c1-160; fi
}
mutant no-die   's/^  \[ -z "\$t1_red" \] || die /  [ -z "$t1_red" ] || say /'
mutant no-kill  's/kill -TERM -- "-\$p"/: "$p"/'
mutant no-rows  's/>> "\$AP\/t1-steps.tsv"/> \/dev\/null/'
mutant serial   's/"t1_\$s" & T1_STEP\[\$!\]=\$s;/( "t1_$s" ); : \& T1_STEP[$!]=$s;/'
[ "$fails" -eq 0 ] || { echo "FAIL  $fails mutant(s)"; exit 1; }
echo "PASS  5 case(s) and 4 mutant(s): deep, dogfood and models start together, join before readiness, and a red in any stops the pass"
