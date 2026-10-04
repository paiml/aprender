#!/usr/bin/env bash
# models_night_read.sh -- release day reads the night's model ladder for H, and runs none (#4701)
#
#   bash scripts/release/models_night_read.sh --ledger DIR --head SHA --artifact DIR --out DIR
#   bash scripts/release/models_night_read.sh --streak --ledger DIR --today YYYY-MM-DD
#   bash scripts/release/models_night_read.sh --self-test
#   bash scripts/release/models_night_read.sh --mutants
#
# WHY. Release day runs models_t1.sh: the full model ladder on both GPU-host legs, the longest step on
# the critical path. The nightly train's models lane measures the same entry point on main's head every
# night (models-nightly.yml relays the GPU-host run of models_t1.sh at C). So release day can read that
# night's result for the head it ships, H, and run nothing.
#
# WHAT IT READS
#   --ledger DIR    the nightly train's --out: DIR/<UTC day>/bundle.tsv, one per night. Its lane table
#                   (lane, kind, state, producer, run id, run head, conclusion, attempt, ...) and its
#                   "# C" line, the commit that night measured.
#   --artifact DIR  artifact models-t1 of the run the ledger names. models-nightly.yml uploads it only for
#                   a verified, unplanted green or red: verdict, SHA256SUMS, models-t1.log, one <host>.json
#                   receipt per GPU-host leg, judge.log and log tails. It must be the artifact of the run
#                   that H's newest green night names (the GO line repeats it as run=). The verdict carries
#                   no run id, so the reader ties the artifact to H, through commit= and the GO line's
#                   sha9, and not to the run: the caller fetches it by that run id.
#
# THE READ. GO only when all of these hold. Anything else is NO-GO, and release day runs no ladder to
# make up for it:
#   1. some night's C is H;
#   2. every night of H read models green, not_measured, or had no models row. A red, a second models
#      row, or any other state is never outvoted by a later green on the same commit;
#   3. the newest night of H that read models green did so from a run on H, at attempt 1;
#   4. the artifact passes its own SHA256SUMS and holds nothing those sums do not list;
#   5. its verdict says commit=H and state=green;
#   6. its models-t1.log has models_t1.sh's GO line at H's sha9 and no NO-GO line, and it holds the receipt of each host
#      that line names.
# A night whose "# C" line cannot be read may be H's. The train writes that line last, so a cut-short
# bundle has none. Such a night is passed over only when its models row ran on another named commit or
# measured nothing (not_measured). Otherwise it is NO-GO.
# On GO the receipts and judge.log are copied into --out: models_t1.sh's output layout, which
# release_readiness.sh reads as --receipts. stdout gets the night's MODELS lines (the lines autopilot.sh
# keeps from models_t1.sh) and one MODELS-NIGHT line naming the night and the run.
#
# THE STREAK (--streak). The switch to this read needs 3 green nights in a row, then sign-off. The newest
# nights are counted, one per UTC day with no day missing, while each read models green from a run on
# that night's C at attempt 1. The newest night must be --today or the day before, so an old streak
# counts nothing, and each night's directory must be a calendar day. Exit 0 at 3 or more.
#
# NOT WIRED. No release script calls this file. Putting it into autopilot.sh in place of models_t1.sh
# changes what release day accepts; that is the switch, and it waits for the streak and sign-off.
#
# FIXTURES. models_night_cases/<state>/bundle.tsv are nights written by nightly_train.sh itself, replayed
# on a fixture read of one models-nightly run on C (green, red, skipped) and, for no_producer, by the
# train as it stands before its models row.
#
# EXIT 0 GO, or the streak is met · 1 NO-GO, or the streak is not met · 3 caller error.
# Bash, awk, sha256sum and GNU date.
set -uo pipefail
PROG=models_night_read
NEED=3
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
CASES="$HERE/models_night_cases"

nogo() { printf 'MODELS NO-GO: %s\n' "$*"; exit 1; }
caller_error() { printf '%s: caller error: %s\n' "$PROG" "$*" >&2; exit 3; }

# nights LEDGER -> the UTC days under LEDGER that hold a bundle.tsv, newest first
nights() {
    local d
    for d in "$1"/*/; do
        d=${d%/}; d=${d##*/}
        if [[ $d =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [ -f "$1/$d/bundle.tsv" ]; then printf '%s\n' "$d"; fi
    done | LC_ALL=C sort -r
}

# night BUNDLE -> "C state run_id run_head attempt" (tab-separated, "-" for empty) from the night's models
#   row; state "absent" when there is no models row and "ambiguous" when there is more than one
night() {
    awk -F '\t' -v OFS='\t' '
        function f(x) { return x == "" ? "-" : x }
        $1 == "# C" { c = $2 }
        $1 == "models" { s = $3; r = $5; h = $6; a = $8; n++ }
        END {
            if (n != 1) { s = (n ? "ambiguous" : "absent"); r = h = a = "" }
            print f(c), f(s), f(r), f(h), f(a)
        }' "$1"
}

# read_night LEDGER H ARTIFACT OUT -> GO (exit 0) or one NO-GO line (exit 1)
read_night() {
    local ledger=$1 h=$2 art=$3 out=$4 d c s r rh a seen=0 go_d="" go_r="" hist="" listed have go x
    local -a w
    [[ $h =~ ^[0-9a-f]{40}$ ]] || caller_error "--head '$h' is not a full 40-hex sha"
    [ -n "$ledger" ] || caller_error "--ledger is required"
    [ -n "$art" ] || caller_error "--artifact is required"
    [ -n "$out" ] || caller_error "--out is required"
    if [ -e "$out" ] && { [ ! -d "$out" ] || [ -n "$(find "$out" -mindepth 1 -print -quit)" ]; }; then
        caller_error "--out '$out' exists and is not empty: stale receipts would mix with the night's"
    fi
    [ -d "$ledger" ] || nogo "no ledger at '$ledger': release day runs no ladder to make one"
    while IFS= read -r d; do
        IFS=$'\t' read -r c s r rh a < <(night "$ledger/$d/bundle.tsv")
        if ! [[ $c =~ ^[0-9a-f]{40}$ ]]; then  # no readable C: the train writes its "# C" line last
            [ "$rh" != "$h" ] || nogo "night $d has no readable C, and its models run (run $r) was on H ${h:0:10}"
            [ "$s" = not_measured ] || [[ $rh =~ ^[0-9a-f]{40}$ ]] \
                || nogo "night $d has no readable C and models $s names no commit (run $r): it may be H's"
            continue
        fi
        [ "$c" = "$h" ] || continue
        seen=1; printf -v hist '%s %s=%s' "$hist" "$d" "$s"
        case $s in
            green)
                if [ -z "$go_d" ]; then
                    [ "$rh" = "$h" ] || nogo "night $d read models green from a run on ${rh:0:10}, not H ${h:0:10}"
                    [ "$a" = 1 ] || nogo "night $d read models green at attempt $a: a retried green is not a green"
                    go_d=$d; go_r=$r
                fi ;;
            not_measured|absent) ;;
            *) nogo "night $d read models $s on H ${h:0:10} (run $r): a later green never outvotes it" ;;
        esac
    done < <(nights "$ledger")
    [ "$seen" = 1 ] || nogo "no night in the ledger measured H ${h:0:10}: release day runs no ladder to make one"
    [ -n "$go_d" ] || nogo "no night of H ${h:0:10} read models green:$hist"

    [ -d "$art" ] || nogo "no models-t1 artifact at '$art' for run $go_r"
    [ -f "$art/verdict" ] && [ -f "$art/SHA256SUMS" ] || nogo "the artifact of run $go_r has no verdict or no SHA256SUMS"
    (cd -- "$art" && sha256sum --strict --quiet -c SHA256SUMS > /dev/null 2>&1) \
        || nogo "the artifact of run $go_r fails its own SHA256SUMS"
    listed=$(awk '{ print $2 }' "$art/SHA256SUMS" | LC_ALL=C sort)
    have=$(cd -- "$art" && find . -mindepth 1 -maxdepth 1 ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort)
    [ "$listed" = "$have" ] || nogo "the artifact of run $go_r holds an entry its SHA256SUMS does not list"
    [ "$(vget commit "$art/verdict")" = "$h" ] \
        || nogo "the artifact's verdict names commit $(vget commit "$art/verdict" | cut -c1-10), not H ${h:0:10}"
    [ "$(vget state "$art/verdict")" = green ] || nogo "the artifact's verdict is $(vget state "$art/verdict"), not green"
    [ -f "$art/models-t1.log" ] || nogo "the artifact of run $go_r has no models-t1.log"
    if grep -qE '^MODELS ([^ ]+ )?NO-GO' "$art/models-t1.log"; then
        nogo "the artifact's models-t1.log holds a NO-GO line"
    fi
    go=$(grep -E '^MODELS GO on [A-Za-z0-9._-]+ and [A-Za-z0-9._-]+ at [0-9a-f]{9}:' "$art/models-t1.log" | tail -n 1)
    [ -n "$go" ] || nogo "the artifact's models-t1.log has no models_t1.sh GO line"
    read -r -a w <<< "$go"
    [ "${w[7]%:}" = "${h:0:9}" ] || nogo "the GO line in models-t1.log is at ${w[7]%:}, not H ${h:0:9}"
    for x in "${w[3]}" "${w[5]}"; do
        [ -f "$art/$x.json" ] || nogo "the artifact holds no receipt for $x, which the GO line names"
    done

    mkdir -p -- "$out" || caller_error "cannot create --out '$out'"
    cp -- "$art/${w[3]}.json" "$art/${w[5]}.json" "$out/" || nogo "cannot copy the receipts into '$out'"
    if [ -f "$art/judge.log" ]; then cp -- "$art/judge.log" "$out/judge.log" || nogo "cannot copy judge.log into '$out'"; fi
    grep -E '^MODELS ' "$art/models-t1.log"
    printf 'MODELS-NIGHT GO H=%s night=%s run=%s: release day read the night, and ran no ladder\n' "${h:0:10}" "$go_d" "$go_r"
}

# vget KEY VERDICT -> the value of KEY in a key=value verdict file
vget() { awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$2"; }

# day SPEC -> the UTC calendar day GNU date makes of SPEC, empty when it makes none
day() { date -u -d "$1" +%F 2> /dev/null; } # bashrs disable-line=DET002 (calendar arithmetic on an input day, never the clock)

# streak LEDGER TODAY -> one MODELS-NIGHT STREAK line; exit 0 when NEED or more green nights end the
#   ledger and the newest is TODAY or the day before
streak() {
    local ledger=$1 today=$2 d c s r rh a n=0 prev="" want seen="" why=""
    [ -n "$ledger" ] || caller_error "--ledger is required"
    [[ $today =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [ "$(day "$today")" = "$today" ] \
        || caller_error "--streak needs --today YYYY-MM-DD, a calendar day (got '$today')"
    if [ -d "$ledger" ]; then
        while IFS= read -r d; do
            if [ "$(day "$d")" != "$d" ]; then why="$d is not a calendar day"; break; fi
            if [ -z "$prev" ]; then
                want="$today or $(day "$today -1 day")"
                if [ "$d" != "$today" ] && [ "$d" != "$(day "$today -1 day")" ]; then
                    why="the newest night is $d, not $want"; break
                fi
            else
                want=$(day "$prev -1 day")
                if [ "$d" != "$want" ]; then why="no night on $want"; break; fi
            fi
            IFS=$'\t' read -r c s r rh a < <(night "$ledger/$d/bundle.tsv")
            if [ "$s" != green ] || ! [[ $c =~ ^[0-9a-f]{40}$ ]] || [ "$rh" != "$c" ] || [ "$a" != 1 ]; then
                why="$d read models $s from run $r on ${rh:0:10} at attempt $a"
                break
            fi
            n=$((n + 1)); seen="$seen $d"; prev=$d
        done < <(nights "$ledger")
    else
        why="no ledger at $ledger"
    fi
    printf 'MODELS-NIGHT STREAK %s of %s:%s; stopped: %s\n' "$n" "$NEED" "${seen:- none}" "${why:-no older night}"
    [ "$n" -ge "$NEED" ]
}

# ---- the case table -------------------------------------------------------------------------------
H=4701470147014701470147014701470147014701
X=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

# ledger DIR SPEC... -> one night per SPEC "day:case[:mod]" from models_night_cases/<case>. mod x: the
#   night measured X, not H; head: the models row's run is on X; att2: the models row is at attempt 2;
#   att1: the models row is at attempt 1 (a red row carries no attempt; this one does); dup: a second,
#   red, models row; odd: the models row's state is void, which the train never writes; noc: no "# C"
#   line, as when the train's write was cut short; trunc: no "# C" line and no models row; nocx: x, then noc
ledger() {
    local dir=$1 spec day kase mod b; shift
    mkdir -p -- "$dir" || return 1
    for spec in "$@"; do
        day=${spec%%:*}; kase=${spec#*:}; mod=""
        case $kase in *:*) mod=${kase#*:}; kase=${kase%%:*} ;; esac
        [[ $day =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || return 1
        mkdir -p -- "$dir/$day" || return 1; b="$dir/$day/bundle.tsv"
        case ${mod:-} in
            '') cp -- "$CASES/$kase/bundle.tsv" "$b" ;;
            x) sed "s/$H/$X/g" "$CASES/$kase/bundle.tsv" > "$b" ;;
            head) awk -F '\t' -v OFS='\t' -v x="$X" '$1 == "models" { $6 = x } 1' "$CASES/$kase/bundle.tsv" > "$b" ;;
            att2) awk -F '\t' -v OFS='\t' '$1 == "models" { $8 = 2 } 1' "$CASES/$kase/bundle.tsv" > "$b" ;;
            att1) awk -F '\t' -v OFS='\t' '$1 == "models" { $8 = 1 } 1' "$CASES/$kase/bundle.tsv" > "$b" ;;
            dup) awk -F '\t' -v OFS='\t' '1; $1 == "models" { $3 = "red"; print }' "$CASES/$kase/bundle.tsv" > "$b" ;;
            odd) awk -F '\t' -v OFS='\t' '$1 == "models" { $3 = "void" } 1' "$CASES/$kase/bundle.tsv" > "$b" ;;
            noc) awk -F '\t' '$1 != "# C"' "$CASES/$kase/bundle.tsv" > "$b" ;;
            trunc) awk -F '\t' '$1 != "# C" && $1 != "models"' "$CASES/$kase/bundle.tsv" > "$b" ;;
            nocx) sed "s/$H/$X/g" "$CASES/$kase/bundle.tsv" | awk -F '\t' '$1 != "# C"' > "$b" ;;
            *) return 1 ;;
        esac || return 1
    done
}

# seal DIR -> SHA256SUMS over every file in DIR, as models_nightly.sh seals a bundle
seal() { (cd -- "$1" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort | xargs -r sha256sum -- > SHA256SUMS); }

# artifact DIR [MOD] -> a models-t1 artifact for H's green night; MOD breaks one thing
artifact() {
    local a=$1 mod=${2:-} c=$H g=${H:0:9} st=green
    mkdir -p -- "$a" || return 1
    case $mod in commit) c=$X ;; state) st=red ;; sha9) g=${X:0:9} ;; esac
    printf 'commit=%s\nsha9=%s\nversion=0.70.2\nentry=scripts/release/models_t1.sh@%s\nt1_rc=0\nstate=%s\nreason=fixture\n' \
        "$c" "${H:0:9}" "$X" "$st" > "$a/verdict" || return 1
    {
        printf 'MODELS host-a measured by apr 0.70.2 (%s): executed=12 red=0 (model_ladder rc 0)\n' "${H:0:9}"
        printf 'MODELS host-b measured by apr 0.70.2 (%s): executed=12 red=0 (model_ladder rc 0)\n' "${H:0:9}"
        printf 'MODELS GO on host-a and host-b at %s: the judge passed both receipts (apr 0.70.2 (%s))\n' "$g" "${H:0:9}"
    } > "$a/models-t1.log" || return 1
    printf '{"host":"host-a","executed":12,"red":0}\n' > "$a/host-a.json" || return 1
    printf '{"host":"host-b","executed":12,"red":0}\n' > "$a/host-b.json" || return 1
    printf 'judge: PASS\n' > "$a/judge.log" || return 1
    [ "$mod" != nohost ] || rm -f -- "$a/host-b.json" || return 1
    [ "$mod" != nogoline ] || printf 'MODELS host-b NO-GO: no receipt -- fixture\n' >> "$a/models-t1.log" || return 1
    seal "$a" || return 1
    case $mod in
        tamper) printf 'x\n' >> "$a/host-a.json" ;;
        extra) printf 'x\n' > "$a/unlisted" ;;
        subdir) mkdir -p -- "$a/unlisted" ;;
        nosums) rm -f -- "$a/SHA256SUMS" ;;
    esac
}

# files OUT SUBJECT ARGS... -> the subject's output, then the files it left in OUT
files() {
    local out=$1 rc; shift
    bash "$@"; rc=$?
    printf 'files: %s\n' "$(find "$out" -mindepth 1 -maxdepth 1 -printf '%f\n' 2> /dev/null | LC_ALL=C sort | tr '\n' ' ')"
    return "$rc"
}

# cases SUBJECT -> runs the table against SUBJECT; sets BROKE
cases() {
    local s=$1 t
    t=$(mktemp -d) || { printf 'not_measured: no temp dir\n'; BROKE=1; return; }
    BROKE=0
    row() { # row NAME RC NEEDLE FORBID -- CMD...
        local name=$1 want=$2 needle=$3 forbid=$4 o rc; shift 5
        o=$("$@" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && [[ $o == *"$needle"* ]] && { [ -z "$forbid" ] || [[ $o != *"$forbid"* ]]; }; then
            printf 'ok    %s\n' "$name"
        else
            printf 'BROKE %s: rc %s (want %s), output: %s\n' "$name" "$rc" "$want" "$(printf '%s' "$o" | tr '\n' '|' | cut -c1-300)"
            BROKE=$((BROKE + 1))
        fi
    }
    setup() { # setup CMD... -> runs one fixture step; a failed step breaks the table, never a row's verdict
        "$@" || { printf 'BROKE setup: %s\n' "$*"; BROKE=$((BROKE + 1)); }
    }
    setup ledger "$t/l-go" 2026-10-06:green; setup artifact "$t/a-ok"
    row "green night of H: GO, naming the night and the run" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-06 run=601" "" \
        -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-ok" --out "$t/o1"
    row "GO keeps models_t1.sh's MODELS lines" 0 "MODELS GO on host-a and host-b at 470147014" "" \
        -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-ok" --out "$t/o2"
    row "GO leaves both receipts and judge.log in --out" 0 "files: host-a.json host-b.json judge.log" "" \
        -- files "$t/o3" "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-ok" --out "$t/o3"
    setup ledger "$t/l-other" 2026-10-06:green:x
    row "no night measured H" 1 "no night in the ledger measured H 4701470147" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-other" --head "$H" --artifact "$t/a-ok" --out "$t/o4"
    setup mkdir -p -- "$t/l-empty"
    row "empty ledger" 1 "no night in the ledger measured H" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-empty" --head "$H" --artifact "$t/a-ok" --out "$t/o5"
    row "no ledger" 1 "no ledger at" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-none" --head "$H" --artifact "$t/a-ok" --out "$t/o6"
    setup ledger "$t/l-red" 2026-10-06:red
    row "red night of H" 1 "night 2026-10-06 read models red on H 4701470147 (run 601)" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-red" --head "$H" --artifact "$t/a-ok" --out "$t/o7"
    setup ledger "$t/l-redold" 2026-10-06:green 2026-10-05:red
    row "a later green never outvotes a red on H" 1 "night 2026-10-05 read models red" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-redold" --head "$H" --artifact "$t/a-ok" --out "$t/o8"
    setup ledger "$t/l-redx" 2026-10-06:green 2026-10-05:red:x
    row "a red on another commit is not H's" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-06" "" \
        -- bash "$s" --ledger "$t/l-redx" --head "$H" --artifact "$t/a-ok" --out "$t/o9"
    setup ledger "$t/l-nm" 2026-10-06:not_measured
    row "not_measured is not a pass" 1 "no night of H 4701470147 read models green: 2026-10-06=not_measured" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-nm" --head "$H" --artifact "$t/a-ok" --out "$t/o10"
    setup ledger "$t/l-np" 2026-10-06:no_producer
    row "no producer yet (the train before its models row)" 1 "read models green: 2026-10-06=not_measured" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-np" --head "$H" --artifact "$t/a-ok" --out "$t/o11"
    setup ledger "$t/l-nmnew" 2026-10-06:not_measured 2026-10-05:green
    row "an unmeasured night does not hide H's green one" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-05 run=601" "" \
        -- bash "$s" --ledger "$t/l-nmnew" --head "$H" --artifact "$t/a-ok" --out "$t/o12"
    setup ledger "$t/l-head" 2026-10-06:green:head
    row "green from a run on another commit" 1 "from a run on bbbbbbbbbb, not H 4701470147" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-head" --head "$H" --artifact "$t/a-ok" --out "$t/o13"
    setup ledger "$t/l-att" 2026-10-06:green:att2
    row "green at attempt 2" 1 "green at attempt 2: a retried green is not a green" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-att" --head "$H" --artifact "$t/a-ok" --out "$t/o14"
    local m n=15
    for m in tamper:"fails its own SHA256SUMS" extra:"does not list" subdir:"does not list" \
             commit:"names commit bbbbbbbbbb, not H 4701470147" state:"verdict is red, not green" \
             sha9:"is at bbbbbbbbb, not H 470147014" nohost:"no receipt for host-b" \
             nogoline:"models-t1.log holds a NO-GO line" \
             nosums:"no verdict or no SHA256SUMS"; do
        setup artifact "$t/a-${m%%:*}" "${m%%:*}"
        row "artifact ${m%%:*}" 1 "${m#*:}" "MODELS-NIGHT GO" \
            -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-${m%%:*}" --out "$t/o$n"
        n=$((n + 1))
    done
    row "no artifact" 1 "no models-t1 artifact at" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-none" --out "$t/o30"
    setup mkdir -p -- "$t/o31"; setup cp -- "$CASES/green/line" "$t/o31/old.json"
    row "a non-empty --out is refused" 3 "is not empty" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-ok" --out "$t/o31"
    row "H must be a full sha" 3 "not a full 40-hex sha" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-go" --head "${H:0:12}" --artifact "$t/a-ok" --out "$t/o32"
    setup ledger "$t/l-2g" 2026-10-06:green 2026-10-05:green
    row "two green nights of H: GO names the newest" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-06 run=601" "" \
        -- bash "$s" --ledger "$t/l-2g" --head "$H" --artifact "$t/a-ok" --out "$t/o33"
    setup ledger "$t/l-dup" 2026-10-06:green 2026-10-05:green:dup
    row "two models rows on one night of H" 1 "night 2026-10-05 read models ambiguous on H 4701470147" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-dup" --head "$H" --artifact "$t/a-ok" --out "$t/o34"
    setup ledger "$t/l-odd" 2026-10-06:green 2026-10-05:red:odd
    row "a state the train never writes, on H" 1 "night 2026-10-05 read models void on H 4701470147" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-odd" --head "$H" --artifact "$t/a-ok" --out "$t/o35"
    setup ledger "$t/l-noc" 2026-10-06:green 2026-10-05:red:noc
    row "a night with no C whose models run was on H" 1 "night 2026-10-05 has no readable C, and its models run (run 601) was on H" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-noc" --head "$H" --artifact "$t/a-ok" --out "$t/o36"
    setup ledger "$t/l-trunc" 2026-10-06:green 2026-10-05:green:trunc
    row "a cut-short night with no C and no models row" 1 "night 2026-10-05 has no readable C and models absent names no commit" "MODELS-NIGHT GO" \
        -- bash "$s" --ledger "$t/l-trunc" --head "$H" --artifact "$t/a-ok" --out "$t/o37"
    setup ledger "$t/l-nocx" 2026-10-06:green 2026-10-05:red:nocx
    row "a night with no C whose models run was on another commit" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-06" "" \
        -- bash "$s" --ledger "$t/l-nocx" --head "$H" --artifact "$t/a-ok" --out "$t/o38"
    setup ledger "$t/l-nocnm" 2026-10-06:green 2026-10-05:not_measured:noc
    row "a night with no C that measured nothing" 0 "MODELS-NIGHT GO H=4701470147 night=2026-10-06" "" \
        -- bash "$s" --ledger "$t/l-nocnm" --head "$H" --artifact "$t/a-ok" --out "$t/o39"

    setup ledger "$t/s3" 2026-10-06:green 2026-10-05:green 2026-10-04:green
    row "streak: 3 green nights" 0 "MODELS-NIGHT STREAK 3 of 3: 2026-10-06 2026-10-05 2026-10-04" "" -- bash "$s" --streak --ledger "$t/s3" --today 2026-10-06
    setup ledger "$t/s4" 2026-10-06:green 2026-10-05:green 2026-10-04:green 2026-10-03:green
    row "streak: 4 green nights" 0 "MODELS-NIGHT STREAK 4 of 3" "" -- bash "$s" --streak --ledger "$t/s4" --today 2026-10-06
    setup ledger "$t/sx" 2026-10-06:green:x 2026-10-05:green 2026-10-04:green:x
    row "streak: main moved between nights" 0 "MODELS-NIGHT STREAK 3 of 3" "" -- bash "$s" --streak --ledger "$t/sx" --today 2026-10-06
    setup ledger "$t/sred" 2026-10-06:green 2026-10-05:green 2026-10-04:red
    row "streak: a red night ends it" 1 "STREAK 2 of 3: 2026-10-06 2026-10-05; stopped: 2026-10-04 read models red" "" \
        -- bash "$s" --streak --ledger "$t/sred" --today 2026-10-06
    setup ledger "$t/sgap" 2026-10-06:green 2026-10-05:green 2026-10-03:green
    row "streak: a missing night ends it" 1 "STREAK 2 of 3: 2026-10-06 2026-10-05; stopped: no night on 2026-10-04" "" \
        -- bash "$s" --streak --ledger "$t/sgap" --today 2026-10-06
    setup ledger "$t/snm" 2026-10-06:not_measured 2026-10-05:green 2026-10-04:green 2026-10-03:green
    row "streak: the newest night unmeasured" 1 "STREAK 0 of 3: none; stopped: 2026-10-06 read models not_measured" "" \
        -- bash "$s" --streak --ledger "$t/snm" --today 2026-10-06
    setup ledger "$t/satt" 2026-10-06:green 2026-10-05:green 2026-10-04:green:att2
    row "streak: a retried green ends it" 1 "STREAK 2 of 3" "" -- bash "$s" --streak --ledger "$t/satt" --today 2026-10-06
    setup ledger "$t/sred1" 2026-10-06:green 2026-10-05:green 2026-10-04:red:att1
    row "streak: a red night ends it even on its own commit at attempt 1" 1 "stopped: 2026-10-04 read models red from run 601" "" \
        -- bash "$s" --streak --ledger "$t/sred1" --today 2026-10-06
    setup ledger "$t/shead" 2026-10-06:green 2026-10-05:green 2026-10-04:green:head
    row "streak: a green from a run on another commit ends it" 1 "STREAK 2 of 3" "" -- bash "$s" --streak --ledger "$t/shead" --today 2026-10-06
    setup ledger "$t/snp" 2026-10-06:no_producer 2026-10-05:no_producer 2026-10-04:no_producer
    row "streak: no producer yet" 1 "STREAK 0 of 3" "" -- bash "$s" --streak --ledger "$t/snp" --today 2026-10-06
    row "streak: no ledger" 1 "STREAK 0 of 3: none; stopped: no ledger at" "" -- bash "$s" --streak --ledger "$t/s-none" --today 2026-10-06
    row "streak: the newest night may be yesterday" 0 "MODELS-NIGHT STREAK 3 of 3" "" \
        -- bash "$s" --streak --ledger "$t/s3" --today 2026-10-07
    row "streak: an old streak counts nothing" 1 "STREAK 0 of 3: none; stopped: the newest night is 2026-10-06, not 2026-10-09 or 2026-10-08" "" \
        -- bash "$s" --streak --ledger "$t/s3" --today 2026-10-09
    row "streak: --today is required" 3 "--streak needs --today YYYY-MM-DD" "MODELS-NIGHT STREAK" \
        -- bash "$s" --streak --ledger "$t/s3"
    row "streak: --today must be a calendar day" 3 "a calendar day (got '2026-02-30')" "MODELS-NIGHT STREAK" \
        -- bash "$s" --streak --ledger "$t/s3" --today 2026-02-30
    setup ledger "$t/scal" 2026-03-02:green 2026-03-01:green 2026-02-30:green 2026-02-28:green
    row "streak: a night that is not a calendar day ends it" 1 "STREAK 2 of 3: 2026-03-02 2026-03-01; stopped: 2026-02-30 is not a calendar day" "" \
        -- bash "$s" --streak --ledger "$t/scal" --today 2026-03-02
    rm -rf -- "${t:?}"
}

# the planted mutants: each must change the file, still parse, and break at least one row
MUTANTS=(
    's/\*) nogo "night \$d read models/*) : nogo "night $d read models/'
    's/not_measured|absent) ;;/not_measured|absent|ambiguous) ;;/'
    's/^            \*) nogo "night/            red|ambiguous) nogo "night/'
    's/\[ "\$rh" != "\$h" \] || nogo/true || nogo/'
    's/\[ "\$s" = not_measured \] || \[\[/true || [[/'
    's/if \[ -z "\$go_d" \]; then/if true; then/'
    's/if \[ "\$d" != "\$today" \]/if false/'
    's/ && \[ "\$d" != "\$(day "\$today -1 day")" \]//'
    's/if \[ "\$(day "\$d")" != "\$d" \]; then why/if false; then why/'
    's/\[ "\$rh" = "\$h" \] || nogo/true || nogo/'
    's/\[ "\$a" = 1 \] || nogo/true || nogo/'
    's/sha256sum --strict --quiet -c SHA256SUMS/true/'
    's/\[ "\$listed" = "\$have" \] || nogo/true || nogo/'
    's/\[ "\$(vget commit "\$art\/verdict")" = "\$h" \]/true/'
    's/\[ "\$(vget state "\$art\/verdict")" = green \]/true/'
    's/\[ "\${w\[7\]%:}" = "\${h:0:9}" \]/true/'
    's/\[ -f "\$art\/\$x.json" \] || nogo/true || nogo/'
    's/if grep -qE .^MODELS (\[^ \]+ )?NO-GO. /if false /'
    's/\[ "\$c" = "\$h" \] || continue/true || continue/'
    's/if \[ "\$d" != "\$want" \]; then/if false; then/'
    's/^NEED=3$/NEED=2/'
    's/if \[ "\$s" != green \] || /if false || /'
    's/\[ "\$rh" != "\$c" \] || \[ "\$a" != 1 \]/false/'
)

selftest() {
    cases "$SELF"
    printf '%s self-test: %s row(s) broke\n' "$PROG" "$BROKE"
    [ "$BROKE" -eq 0 ]
}

mutants() {
    local t i=0 killed=0 m mt
    cases "$SELF" > /dev/null
    [ "$BROKE" -eq 0 ] || { printf '%s mutants: the unmutated table is not green; run --self-test\n' "$PROG"; return 1; }
    t=$(mktemp -d) || { printf 'not_measured: no temp dir\n'; return 1; }
    for m in "${MUTANTS[@]}"; do
        i=$((i + 1)); mt="$t/m$i/$PROG.sh"; mkdir -p -- "$t/m$i" || return 1
        sed -e "$m" "$SELF" > "$mt" || return 1
        if cmp -s -- "$SELF" "$mt"; then printf 'NOT APPLIED m%s: %s\n' "$i" "$m"; continue; fi
        if ! bash -n "$mt" 2> /dev/null; then printf 'NO PARSE    m%s: %s\n' "$i" "$m"; continue; fi
        cases "$mt" > "$t/m$i.log" 2>&1
        if [ "$BROKE" -gt 0 ]; then killed=$((killed + 1)); printf 'killed   m%s (%s rows): %s\n' "$i" "$BROKE" "$m"
        else printf 'SURVIVED m%s: %s\n' "$i" "$m"; fi
    done
    rm -rf -- "${t:?}"
    printf '%s mutants: %s of %s killed\n' "$PROG" "$killed" "${#MUTANTS[@]}"
    [ "$killed" -eq "${#MUTANTS[@]}" ]
}

MODE=read LEDGER="" HEAD="" ART="" OUT="" TODAY=""
while [ $# -gt 0 ]; do
    case $1 in
        --ledger|--head|--artifact|--out|--today)
            [ $# -ge 2 ] || caller_error "$1 needs a value"
            case $1 in --ledger) LEDGER=$2 ;; --head) HEAD=$2 ;; --artifact) ART=$2 ;; --out) OUT=$2 ;; --today) TODAY=$2 ;; esac
            shift 2 ;;
        --streak) MODE=streak; shift ;;
        --self-test) MODE=selftest; shift ;;
        --mutants) MODE=mutants; shift ;;
        -h|--help) sed -n '2,7p' "$SELF"; exit 0 ;;
        *) caller_error "unknown argument '$1' (see the header of $SELF)" ;;
    esac
done
case $MODE in
    read) read_night "$LEDGER" "$HEAD" "$ART" "$OUT" ;;
    streak) streak "$LEDGER" "$TODAY" ;;
    selftest) selftest ;;
    mutants) mutants ;;
esac
