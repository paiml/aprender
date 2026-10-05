#!/usr/bin/env bash
# release_night_read_table.sh -- planted rows for scripts/release/release_night_read.sh (T18 #4701, reader side).
# Builds ledgers in nightly_train.sh's bundle.tsv format (header by name, "# C" / "# read" lines), runs the real
# script, and checks "<exit>|<first word of the verdict>". The rule under test: a missing result, or a result from
# another commit, REFUSES (not_measured, exit 2); a red is never outvoted; only a green on H at attempt 1 is GO.
# Then plants mutants and requires a row to catch each, so the table is proven able to see them.
#
# EXIT 0 every row holds and every mutant is caught · 1 otherwise
#
#   bash scripts/release/release_night_read_table.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SUT="$ROOT/scripts/release/release_night_read.sh"
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT

H=$(printf 'a%.0s' {1..40})
O=$(printf 'b%.0s' {1..40})   # another commit
VERDICT="deep-doctests deep-nodefault deep-examples deep-bins-build deep-bins-smoke dogfood models"
HDR=$'lane\tkind\tstate\tproducer\trun_id\trun_head\tconclusion\tattempt\tstarted\tended\treason\thours_lost'

fail=0
row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got '$2', want '$3'"; fail=1; fi; }

fresh() { L=$(mktemp -d "$TD/l.XXXXXX"); }   # unique: table runs in a subshell per mutant

# night <day> <C or -> [read] [override ...] -- a bundle whose verdict lanes are all green on <C> at attempt 1.
# An override "lane=state,run_head,attempt" replaces that lane's row; "lane=" drops it; "lane+=..." adds a row.
night() {
    local d="$1" c="$2" rd="${3:-ok}" l st hd at o b
    shift; shift; [ $# -gt 0 ] && shift
    b="$L/$d/bundle.tsv"; mkdir -p "$L/$d"
    echo "$HDR" > "$b"
    for l in $VERDICT coverage; do
        st=green hd="$c" at=1
        [ "$l" = coverage ] && st=not_measured hd="" at=""
        for o in "$@"; do
            case "$o" in
                "$l=") st=DROP ;;
                "$l="*) IFS=, read -r st hd at <<< "${o#*=}" ;;
            esac
        done
        [ "$st" = DROP ] || printf '%s\tverdict\t%s\tp\t%s\t%s\tc\t%s\t\t\t\t-\n' "$l" "$st" "91" "$hd" "$at" >> "$b"
    done
    for o in "$@"; do
        case "$o" in *"+="*) IFS=, read -r st hd at <<< "${o#*+=}"; printf '%s\tverdict\t%s\tp\t77\t%s\tc\t%s\t\t\t\t-\n' "${o%%+=*}" "$st" "$hd" "$at" >> "$b" ;; esac
    done
    [ "$c" = - ] || printf '# C\t%s\n' "$c" >> "$b"
    printf '# read\t%s\n' "$rd" >> "$b"
}

run() { local rc=0 out; out=$(bash "$1" --ledger "$L" --head "$H" 2>&1) || rc=$?; echo "$rc|$(tail -n 1 <<< "$out" | cut -d' ' -f1)"; }

table() { # table <sut> <label>
    local s="$1" l="$2"
    fresh; night 2026-10-05 "$H"
    row "$l all verdict lanes green on H at attempt 1 -> GO" "$(run "$s")" "0|GO"
    fresh; rmdir "$L"
    row "$l no ledger -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$O"
    row "$l only a night of another commit -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models=not_measured,,"
    row "$l models not_measured -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models=green,$O,1"
    row "$l models green from a run on another commit -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "dogfood=green,,1"
    row "$l dogfood green with no run_head -> refuse (attempt not shifted)" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models=green,$H,2"
    row "$l models green at attempt 2 -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models="
    row "$l models row missing -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "dogfood="
    row "$l dogfood row missing -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "deep-bins-smoke="
    row "$l deep-bins-smoke row missing -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models+=green,$H,1"
    row "$l models row duplicated -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "deep-doctests=red,$H,1"
    row "$l deep-doctests red -> RED" "$(run "$s")" "1|RED"
    fresh; night 2026-10-04 "$H" ok "models=red,$H,1"; night 2026-10-05 "$H"
    row "$l older night of H red, newer green -> RED (never outvoted)" "$(run "$s")" "1|RED"
    fresh; night 2026-10-05 "$H" ok "models=red,$H,1" "models+=green,$H,1"
    row "$l red row hidden behind a duplicate green -> RED" "$(run "$s")" "1|RED"
    fresh; night 2026-10-05 "$H" failed
    row "$l newest night of H read failed -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H"; night 2026-10-06 -
    row "$l a night with no '# C' line (may be H's) -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H"; night 2026-10-05 "$H" ok "models=not_measured,,"
    row "$l newest night of H not measured, older green -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H" ok "models=not_measured,,"; night 2026-10-05 "$H"
    row "$l older night of H not measured, newest green -> GO" "$(run "$s")" "0|GO"
    fresh; night 2026-10-04 "$H"; night 2026-10-05 "$O" ok "models=red,$O,1"
    row "$l another commit's red is not H's -> GO" "$(run "$s")" "0|GO"
    fresh; night 2026-10-05 "$H" ok "coverage=red,$H,1"
    row "$l an info lane red does not gate -> GO" "$(run "$s")" "0|GO"
    fresh; night 2026-10-05 "$H"
    awk -F'\t' 'BEGIN { OFS = "\t" } /^#/ { print; next } { t = $3; $3 = $6; $6 = t; print }' "$L/2026-10-05/bundle.tsv" > "$L/x" && mv "$L/x" "$L/2026-10-05/bundle.tsv"
    row "$l columns reordered: read by header name -> GO" "$(run "$s")" "0|GO"
    fresh; night 2026-10-04 "$H"$'\r' ok "models=red,$H,1"; night 2026-10-05 "$H"
    row "$l older night's '# C' ends in CR (it may be H's) -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H " ok "models=red,$H,1"; night 2026-10-05 "$H"
    row "$l older night's '# C' has a trailing space -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H" ok "models=red,$H,1"; night 2026-10-05 "$H"
    sed -i '1s/\tstate\t/\tstatus\t/' "$L/2026-10-04/bundle.tsv"
    row "$l older night of H has no 'state' column -> refuse (a red could hide)" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H" ok "models=red,$H,1"; night 2026-10-05 "$H"
    sed -i '1s/\trun_id\t/\trun\t/' "$L/2026-10-04/bundle.tsv"
    row "$l older night of H has no 'run_id' column -> refuse, never a guessed run" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H" ok "models=not_measured,$H,1"
    row "$l newest night: lane not_measured though run_head and attempt look right -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-04 "$H" ok "models=RED,$H,1"; night 2026-10-05 "$H"
    row "$l older night of H holds a state the train never writes -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H"; mkdir -p "$L/2026-10-04 x"
    row "$l a dir that is not a UTC day -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H"; mkdir -p "$L/cache"; : > "$L/history.tsv"
    row "$l the train's cache dir and history.tsv are not nights -> GO" "$(run "$s")" "0|GO"
    fresh; night 2026-10-04 "$H"; mkdir -p "$L/2026-10-05"; echo "NOT RELEASABLE" > "$L/2026-10-05/line"
    row "$l a failed night with no bundle holds no row -> passed over, older green GO" "$(run "$s")" "0|GO"
    fresh; mkdir -p "$L/2026-10-05"; echo "NOT RELEASABLE" > "$L/2026-10-05/line"
    row "$l only a failed night with no bundle -> refuse" "$(run "$s")" "2|NOT_MEASURED"
    fresh; night 2026-10-05 "$H"
    row "$l short head -> caller error" "$(rc=0; bash "$s" --ledger "$L" --head "${H:0:9}" > /dev/null 2>&1 || rc=$?; echo "$rc")" 3
}

table "$SUT" real

mutant() { # mutant <label> <sed expr>
    local m="$TD/mutant.sh" c
    sed "$2" "$SUT" > "$m"
    if cmp -s "$SUT" "$m"; then echo "FAIL  mutant $1: the sed did not change the script (pattern drifted)"; fail=1; return; fi
    c=$(table "$m" "  [$1]" 2>&1 | grep -c '^FAIL' || true)
    if [ "$c" -gt 0 ]; then echo "ok    mutant $1 caught by $c row(s)"; else echo "FAIL  mutant $1 survived every row"; fail=1; fi
}
mutant no-run-head-binding  's/^    \[ "\$h" = "\$H" \] || nm/    true || nm/'
mutant retried-green-passes 's/^    \[ "\$a" = 1 \] || nm/    true || nm/'
mutant no-red-scan          's/^        \[ -n "\$r" \] \&\& {/        false \&\& {/'
mutant red-only-last-row    's/if (s == "red" \&\& red == "")/if (0)/'
mutant oldest-night-wins    's/^newest=\${nights%% \*}$/newest=${nights% }; newest=${newest##* }/'
mutant no-read-ok           's/^\[ "\$(meta "\$b" read)" = ok \] || nm/true || nm/'
mutant no-c-line-refusal    's/\[\[ \$c .. ^\[0-9a-f\]{40}\$ \]\] || nm/true || nm/'
mutant no-bad-row-refusal   's/^        \[ -n "\$x" \] \&\& nm/        false \&\& nm/'
mutant no-state-whitelist   's/if (s != "green" \&\& s != "red" \&\& s != "not_measured" \&\& bad == "")/if (0)/'
mutant no-column-check      's/bad = "missing column"/bad = ""/'
mutant any-dir-is-a-night   's/|| nm "night: .\$day. is not a UTC day dir"/|| true/'
mutant state-not-checked    's/^    \[ "\$s" = green \] || nm/    true || nm/'
mutant duplicates-allowed   's/^    \[ "\$n" = 1 \] || nm/    [ "$n" -ge 1 ] || nm/'
mutant models-dropped       's/ dogfood models"$/ dogfood"/'
mutant positional-columns   's/col\[\$i\] = i/col[$i] = 0/'

exit "$fail"
