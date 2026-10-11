#!/usr/bin/env bash
# models_gate_lab_test.sh — the models column is a gate on crux, and no gate needs a LAB job (#5147)
#
# WHAT IT HOLDS. models-nightly.yml has two measures. The `models` job (the column the nightly train reads) gets
#   its state from the job it needs, and that job relays the crux bundle: a red crux is a red `models`. The full
#   ladder is LAB: `models-ladder` relays it on the schedule and files tickets, and nothing needs it, so a red
#   ladder leaves `models` green. Across every producer workflow the train reads, a job that is a verdict lane
#   (a GATE) never needs a LAB job, directly or through another job. LAB jobs: the jobs of the train's lab lanes,
#   and models-ladder.
#
# THE MODEL. The workflow cannot run here, so the rows read it with yq and replay what GitHub would do:
#   models' state = the state of the bundle its needed job relays (--measure crux, else the ladder default).
#   models is red when that state is red; continue-on-error anywhere on the models path makes a red read green.
#
# MUTANTS. --mutants plants each wrong workflow (or train) into a copy and reruns the rows; a mutant is killed
#   only when a row breaks.
#
# USAGE  models_gate_lab_test.sh [--workflow FILE] [--train FILE]     the case table (default: this tree)
#        models_gate_lab_test.sh --mutants                          every planted mutant must break a row
# EXIT   0 every row held, every mutant killed · 1 a row broke or a mutant survived · 2 not_measured (yq missing)
#        · 3 caller error
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
WFREL=.github/workflows/models-nightly.yml
LAB_JOBS="$WFREL:models-ladder"

command -v yq > /dev/null 2>&1 || { echo "NOT_MEASURED: yq is not installed"; exit 2; }

# lanes TRAIN KIND -> 'lane<TAB>workflow<TAB>job regex' for each row of that kind with a producer
lanes() {
    sed -n "/^LANES='/,/'\$/p" "$1" | sed "s/^LANES='//; s/'\$//" \
        | awk -F ';' -v k="$2" -v OFS='\t' '$2 == k && $4 != "-" && $6 != "" { print $1, $4, $6 }'
}
# jobs ROOT WF -> 'id<TAB>name' per job (name defaults to the id)
jobs() { yq -r '.jobs | to_entries | .[] | .key + "\t" + (.value.name // .key)' "$1/$2"; }
# needs ROOT WF ID -> the ids ID needs, transitively
needs() {
    local r=$1 wf=$2 seen=" " todo=("$3") j n
    while [ "${#todo[@]}" -gt 0 ]; do
        j=${todo[0]}; todo=("${todo[@]:1}")
        while IFS= read -r n; do
            [ -n "$n" ] && [ "$n" != null ] || continue
            case "$seen" in *" $n "*) continue ;; esac
            seen="$seen$n "; todo+=("$n"); printf '%s\n' "$n"
        done < <(yq -r "[.jobs.\"$j\".needs] | flatten | .[]" "$r/$wf")
    done
}
# models_state ROOT CRUX LADDER -> what the models column reads when the crux bundle is CRUX and the ladder LADDER
models_state() {
    local r=$1 wf="$1/$WFREL" n run m s
    n=$(yq -r '[.jobs.models.needs] | flatten | .[0]' "$wf")
    [ -n "$n" ] && [ "$n" != null ] || { echo "models needs nothing"; return; }
    run=$(yq -r ".jobs.\"$n\".steps[] | select(.id == \"relay\") | .run" "$wf")
    case "$run" in *"--measure crux"*) m=crux ;; *"--measure ladder"*|*--relay*) m=ladder ;; *) echo "no relay"; return ;; esac
    if [ "$m" = crux ]; then s=$2; else s=$3; fi
    if [ "$s" = red ] && [ "$(yq -r '[.jobs.models."continue-on-error", .jobs.models.steps[]."continue-on-error", .jobs."'"$n"'"."continue-on-error"] | map(select(. == true)) | length' "$wf")" != 0 ]; then
        s=green
    fi
    printf 'models %s (relays %s)\n' "$s" "$m"
}
# gate_needs_lab ROOT TRAIN -> one line per GATE job that needs a LAB job; 'clean N' when none (N gate jobs read)
gate_needs_lab() {
    local r=$1 t=$2 lab="" lane wf re id name n g=0 bad=0
    lab=$(printf '%s\n' "$LAB_JOBS" | tr ' ' '\n')
    while IFS=$'\t' read -r lane wf re; do
        [ -f "$r/$wf" ] || continue
        while IFS=$'\t' read -r id name; do
            [[ $name =~ $re ]] && lab="$lab"$'\n'"$wf:$id"
        done < <(jobs "$r" "$wf")
    done < <(lanes "$t" lab)
    while IFS=$'\t' read -r lane wf re; do
        [ -f "$r/$wf" ] || continue
        while IFS=$'\t' read -r id name; do
            [[ $name =~ $re ]] || continue
            g=$((g + 1))
            if grep -qxF "$wf:$id" <<<"$lab"; then printf 'GATE %s (%s) is a LAB job\n' "$id" "$wf"; bad=1; fi
            while IFS= read -r n; do
                if grep -qxF "$wf:$n" <<<"$lab"; then printf 'GATE %s (%s) needs LAB %s\n' "$id" "$wf" "$n"; bad=1; fi
            done < <(needs "$r" "$wf" "$id")
        done < <(jobs "$r" "$wf")
    done < <(lanes "$t" verdict)
    [ "$bad" = 1 ] || printf 'clean %s\n' "$g"
}
# ladder_shape ROOT TRAIN -> the facts that make models-ladder LAB, one per line
ladder_shape() {
    local wf="$1/$WFREL" name
    name=$(yq -r '.jobs.models-ladder.name // "models-ladder"' "$wf")
    printf 'if=%s\n' "$(yq -r '.jobs.models-ladder.if' "$wf")"
    printf 'needed-by=%s\n' "$(yq -r '[.jobs[] | select(.needs != null) | [.needs] | flatten | .[]] | map(select(. == "models-ladder")) | length' "$wf")"
    printf 'continue-on-error=%s\n' "$(yq -r '[.. | select(tag == "!!map") | select(has("continue-on-error"))] | length' "$wf")"
    printf 'or-true=%s\n' "$(yq -r '.jobs.models-ladder.steps[].run // ""' "$wf" | grep -c '|| true')"
    lanes "$2" verdict | awk -F '\t' -v w="$WFREL" '$2 == w { print $3 }' | while IFS= read -r re; do
        [[ $name =~ $re ]] && printf 'name-is-a-verdict-lane=%s\n' "$name"
    done
    printf 'tickets=%s\n' "$(yq -r '.jobs.models-ladder.steps[].run // ""' "$wf" | grep -c -- '--tickets')"
    printf 'measure=%s\n' "$(yq -r '.jobs.models-ladder.steps[].run // ""' "$wf" | grep -o -- '--measure [a-z]*' | head -n 1)"
}

PASS=0 FAIL=0
# row NAME WANT CMD... -> CMD's output must contain WANT
row() {
    local name=$1 want=$2 o; shift 2
    o=$("$@" 2>&1)
    if [[ $o == *"$want"* ]]; then PASS=$((PASS + 1)); [ -n "${QUIET:-}" ] || printf '  ok    %s\n' "$name"
    else FAIL=$((FAIL + 1)); [ -n "${QUIET:-}" ] || printf '  BROKE %-48s want: %s; got: %s\n' "$name" "$want" "$(printf '%s' "$o" | tr '\n' ' ' | cut -c1-200)"; fi
}
table() {
    local r=$1 t=$2
    row red_crux_is_red_models "models red (relays crux)" models_state "$r" red green
    row green_crux_is_green_models "models green (relays crux)" models_state "$r" green green
    row red_ladder_leaves_models_green "models green (relays crux)" models_state "$r" green red
    row no_gate_needs_a_lab_job "clean " gate_needs_lab "$r" "$t"
    row the_gate_set_is_not_empty "gates read" bash -c '[ "$1" != "clean 0" ] && echo gates read' _ "$(gate_needs_lab "$r" "$t")"
    local s; s=$(ladder_shape "$r" "$t")
    row ladder_runs_on_the_schedule_only "if=github.event_name == 'schedule'" printf '%s' "$s"
    row nothing_needs_the_ladder "needed-by=0" printf '%s' "$s"
    row no_continue_on_error_in_the_workflow "continue-on-error=0" printf '%s' "$s"
    row ladder_has_no_or_true "or-true=0" printf '%s' "$s"
    row ladder_files_tickets "tickets=1" printf '%s' "$s"
    row ladder_relays_the_ladder "measure=--measure ladder" printf '%s' "$s"
    row ladder_name_is_no_verdict_lane "no lane" bash -c '[[ $1 != *name-is-a-verdict-lane* ]] && echo no lane' _ "$s"
}

MUTANTS="w1_models_relays_the_ladder	$WFREL	s/--relay --measure crux --commit/--relay --commit/
w2_models_needs_the_ladder	$WFREL	s/^    needs: models-evidence\$/    needs: [models-evidence, models-ladder]/
w3_ladder_on_every_event	$WFREL	/^    if: github.event_name == 'schedule'\$/d
w4_models_continue_on_error	$WFREL	s/^    timeout-minutes: 15\$/    timeout-minutes: 15\\n    continue-on-error: true/
w5_ladder_named_models	$WFREL	s/^    name: models-ladder-tickets\$/    name: models/
w6_ladder_or_true	$WFREL	s/--out \"\\\$RUNNER_TEMP\\/models-ladder\"\$/--out \"\$RUNNER_TEMP\\/models-ladder\" || true/
w7_ladder_files_nothing	$WFREL	s/--tickets --commit/--relay --commit/
t1_gate_needs_a_lab_lane_job	.github/workflows/release-lanes-nightly.yml	s/^    needs: \\[pick, measure-cleanroom-gpu\\]\$/    needs: [pick, measure-cleanroom-gpu, cleanroom-cpu]/
t2_gate_lane_on_lab_job	scripts/release/nightly_train.sh	s/^(models;verdict;.*;)\\^models\\\$\$/\\1^(models|models-ladder-tickets)\$/"

case "${1:-}" in
    --mutants)
        tmp=$(mktemp -d) || exit 3; trap 'rm -rf -- "${tmp:?}"' EXIT
        killed=0 survived=0
        while IFS=$'\t' read -r name file expr; do
            rm -rf -- "${tmp:?}/r"; mkdir -p "$tmp/r/.github/workflows" "$tmp/r/scripts/release"
            cp -- "$ROOT/$WFREL" "$tmp/r/$WFREL" || exit 3
            cp -- "$ROOT/scripts/release/nightly_train.sh" "$tmp/r/scripts/release/" || exit 3
            for w in $(lanes "$ROOT/scripts/release/nightly_train.sh" verdict | cut -f2; lanes "$ROOT/scripts/release/nightly_train.sh" lab | cut -f2); do
                [ -f "$ROOT/$w" ] && [ ! -f "$tmp/r/$w" ] && cp -- "$ROOT/$w" "$tmp/r/$w"
            done
            cmp -s <(sed -E "$expr" "$tmp/r/$file") "$tmp/r/$file" && { printf '  NOOP     %s (the mutation changed nothing)\n' "$name"; survived=$((survived + 1)); continue; }
            sed -E -i "$expr" "$tmp/r/$file" || exit 3
            PASS=0 FAIL=0; QUIET=1 table "$tmp/r" "$tmp/r/scripts/release/nightly_train.sh"
            if [ "$FAIL" -gt 0 ]; then killed=$((killed + 1)); printf '  killed   %s (%s rows broke)\n' "$name" "$FAIL"
            else survived=$((survived + 1)); printf '  SURVIVED %s\n' "$name"; fi
        done <<< "$MUTANTS"
        printf -- '--- %s/%s mutants killed ---\n' "$killed" "$((killed + survived))"
        [ "$survived" -eq 0 ] ;;
    ""|--workflow|--train)
        R=$ROOT T="$ROOT/scripts/release/nightly_train.sh"
        while [ $# -gt 0 ]; do
            case "$1" in
                --workflow) [ -n "${2:-}" ] && [ -f "$2" ] || { echo "caller error: --workflow FILE"; exit 3; }
                    case "$2" in *..*) echo "caller error: --workflow takes a path without .."; exit 3 ;; esac
                    tmp=$(mktemp -d) || exit 3; trap 'rm -rf -- "${tmp:?}"' EXIT
                    mkdir -p "$tmp/.github/workflows"; cp -r -- "$ROOT/.github/workflows/." "$tmp/.github/workflows/"; cat -- "$2" > "$tmp/$WFREL"; R=$tmp; shift 2 ;;
                --train) [ -n "${2:-}" ] || { echo "caller error: --train FILE"; exit 3; }; T=$2; shift 2 ;;
                *) echo "caller error: $1"; exit 3 ;;
            esac
        done
        table "$R" "$T"
        printf -- '--- %s/%s rows ---\n' "$PASS" "$((PASS + FAIL))"
        [ "$FAIL" -eq 0 ] ;;
    *) echo "caller error: $1 (usage: models_gate_lab_test.sh [--workflow FILE] [--train FILE] | --mutants)"; exit 3 ;;
esac
