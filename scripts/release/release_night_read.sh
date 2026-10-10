#!/usr/bin/env bash
# release_night_read.sh -- release day reads the nightly train's verdict lanes for H, and re-runs none (#4701)
#
#   bash scripts/release/release_night_read.sh --ledger DIR --head SHA --measure crux|ladder --artifact DIR --out DIR
#   bash scripts/release/release_night_read.sh --streak --ledger DIR --today YYYY-MM-DD
#   bash scripts/release/release_night_read.sh --self-test
#   bash scripts/release/release_night_read.sh --mutants
#
# WHY. Release day re-runs work the nightly train already measures on main's head each night: the deep
# test lanes, dogfood, and the model measure, models_t1.sh on both GPU hosts. Since 0.71.0 release day's
# model measure is the CRUX smoke (release_policy in contracts/model-capability-ladder-v1.yaml), and
# models_nightly.sh measures that mode too, under its own prefix, with <host>-gpu.json receipts. When the
# train measured the head release day ships, H, release day can read those nights and run nothing.
#
# THE LANES are fixed, never an option: deep-doctests deep-nodefault deep-examples deep-bins-build
# deep-bins-smoke dogfood models. Every other row of a bundle (ci-main, the info lanes such as coverage)
# is not read: this file decides nothing about them.
#
# WHAT IT READS
#   --ledger DIR    the nightly train's --out: DIR/<UTC day>/bundle.tsv, one per night. Its lane table
#                   (lane, kind, state, producer, run id, run head, conclusion, attempt, ...), its
#                   "# C" line, the commit that night measured, and its "# read" line, ok when the
#                   train's read of GitHub succeeded. Columns are found by the header's names, never by
#                   position. Every dir under DIR is a UTC day or the train's fetch cache, "cache"; a
#                   day dir with no bundle.tsv (a night the train did not finish) is passed over.
#   --measure M     the model measure release day ships on: crux (the CRUX smoke: <host>-gpu.json
#                   receipts, the "MODELS GO (CRUX smoke)" line) or ladder (the full ladder: <host>.json,
#                   "MODELS GO"). Required: the reader never guesses which one H is released on.
#   --artifact DIR  artifact models-t1 of the run the models lane's green night names (the models GO
#                   line repeats it as run=). models_nightly.sh seals it for a verified, unplanted green
#                   or red: verdict (commit=, measure=, state=), SHA256SUMS, models-t1.log, one receipt
#                   per GPU host, judge.log and log tails. The verdict carries no run id, so the reader
#                   ties the artifact to H, through commit= and the GO line's sha9, and not to the run:
#                   the caller fetches it by that run id. A verdict with no measure= was sealed before
#                   the key existed, and is a ladder verdict.
#
# THE READ, per lane, over every night whose C is H:
#   RED           some night of H read the lane red: any red row, even beside a green one; a later green
#                 never outvotes it. A night with no readable C whose lane run was red on H is H's red.
#   NOT_MEASURED  no red, and the lane cannot be read as green on H: no night measured H; no night of H
#                 read it green; the newest green came from a run on another commit, or at attempt > 1;
#                 a night of H is unreadable (a header that lacks lane, state, run_id, run_head or
#                 attempt), holds two rows of the lane, or a state the train never writes; or a night
#                 with no readable C may be H's (its lane run was on H, or names no commit). For models,
#                 also any fault of the artifact: SHA256SUMS fails or misses an entry, the verdict names
#                 another commit or the other measure, models-t1.log has a NO-GO line or no GO line of
#                 the measure at H's sha9, or a receipt the GO line names is missing.
#   GO            otherwise: the newest night of H that read the lane green, on a night whose "# read"
#                 is ok, did so from a run on H at attempt 1 (and, for models, the artifact holds).
# A night of H that read the lane not_measured, had no row of it, or was read from a failed read of
# GitHub measured nothing: it neither passes nor fails H, and an older green night of H still counts.
# A night with no readable C (the train writes that line last, so a cut-short bundle has none; a CR or a
# trailing space also leaves none) is passed over only when its lane run was on another named commit or
# measured nothing. A dir under the ledger that is neither a UTC day nor the cache, or a bundle awk cannot
# read, makes every lane NOT_MEASURED: whose night it holds is unknown.
# Only on GO for every lane are the receipts and judge.log copied into --out (models_t1.sh's output
# layout, which release_readiness.sh reads as --receipts) and the night's MODELS lines printed (the lines
# autopilot.sh keeps from models_t1.sh). stdout gets one NIGHT line per lane and one summary NIGHT line.
#
# THE STREAK (--streak). The switch to this read needs 3 green nights in a row, then sign-off. Per lane,
# the newest nights are counted, one per UTC day with no day missing, while each read the lane green from
# a run on that night's C at attempt 1, and its "# read" line is ok. The newest night must be --today or
# the day before, so an old streak counts nothing, and each night's directory must be a calendar day.
# Exit 0 only when every lane counts 3 or more.
#
# NOT WIRED. No release script calls this file. Putting it into autopilot.sh in place of the lanes and
# models_t1.sh changes what release day accepts; that is the switch, and it waits for the streak and
# sign-off. Exit 2 is never a GO. What release day does on a 2 (run the lane itself, as it does today) is
# the switch's decision, not this file's.
#
# FIXTURES. release_night_cases/<state>/bundle.tsv are nights written by nightly_train.sh itself, replayed
# on a fixture read of one models-nightly run on C (green, red, skipped) and, for no_producer, by the
# train as it stands before its models row. The table edits them one column at a time.
#
# EXIT 0 GO, or every lane's streak is met · 1 RED (some lane), or a streak is not met · 2 NOT_MEASURED
# (no lane red, some lane not measured) · 3 caller error. Bash, awk, sha256sum and GNU date.
set -uo pipefail
PROG=release_night_read
NEED=3
LANES=(deep-doctests deep-nodefault deep-examples deep-bins-build deep-bins-smoke dogfood models)
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
CASES="$HERE/release_night_cases"
TABLE="" TAIL="" WHY="" HOST_A="" HOST_B="" RSUF=""

caller_error() { printf '%s: caller error: %s\n' "$PROG" "$*" >&2; exit 3; }
# whole WHY -> every lane NOT_MEASURED: the ledger itself cannot be read
whole() { printf 'NIGHT NOT_MEASURED H=%s: %s\n' "${HEAD:0:10}" "$*"; exit 2; }
# lnm LANE WHY -> LANE's NOT_MEASURED line
lnm() { printf 'NIGHT NOT_MEASURED %s H=%s: %s\n' "$1" "${HEAD:0:10}" "$2"; }

# nights LEDGER -> the UTC days under LEDGER that hold a bundle.tsv, newest first
nights() {
    local d
    for d in "$1"/*/; do
        d=${d%/}; d=${d##*/}
        if [[ $d =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [ -f "$1/$d/bundle.tsv" ]; then printf '%s\n' "$d"; fi
    done | LC_ALL=C sort -r
}

# strays LEDGER -> the first dir under LEDGER that is neither a UTC day nor the train's fetch cache (the
#   train writes only those two). Whose night such a dir holds, the reader cannot tell
strays() {
    local d
    for d in "$1"/*/; do
        [ -d "$d" ] || continue
        d=${d%/}; d=${d##*/}
        [ "$d" = cache ] || [[ $d =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { printf '%s\n' "$d"; return; }
    done
}

# night BUNDLE DAY -> one line per lane: "DAY lane C state run_id run_head attempt read" (tab-separated,
#   "-" for empty) from the night's "# C" and "# read" lines and the lane's row. Columns are found by the
#   header's names, never by position. State "red" when ANY row of the lane is red (that row's run),
#   "absent" when there is no row, "ambiguous" when there is more than one and none is red, and
#   "unreadable: <why>" when the header lacks a column the read needs: a lost lane column would read a
#   red night as one with no row. An empty bundle has no "# C" and no rows: absent, and no C
night() {
    awk -F '\t' -v OFS='\t' -v day="$2" -v lanes="${LANES[*]}" '
        function f(x) { return x == "" ? "-" : x }
        BEGIN { k = split(lanes, L, " ") }
        NR == 1 {
            for (i = 1; i <= NF; i++) col[$i] = i
            m = split("lane state run_id run_head attempt", need, " ")
            for (i = 1; i <= m; i++) if (!(need[i] in col)) { bad = "the header has no " need[i] " column"; break }
            next
        }
        $1 == "# C" { c = $2 }
        $1 == "# read" { rd = $2 }
        /^#/ || bad != "" { next }
        {
            l = $col["lane"]; n[l]++
            if (n[l] == 1) { s[l] = $col["state"]; r[l] = $col["run_id"]; h[l] = $col["run_head"]; a[l] = $col["attempt"] }
            if ($col["state"] == "red" && !(l in red)) { red[l] = $col["run_id"]; redh[l] = $col["run_head"]; reda[l] = $col["attempt"] }
        }
        END {
            for (j = 1; j <= k; j++) {
                l = L[j]
                if (bad != "") { st = "unreadable: " bad; rr = hh = aa = "" }
                else if (l in red) { st = "red"; rr = red[l]; hh = redh[l]; aa = reda[l] }
                else if (n[l] != 1) { st = (n[l] ? "ambiguous" : "absent"); rr = hh = aa = "" }
                else { st = s[l]; rr = r[l]; hh = h[l]; aa = a[l] }
                print day, l, f(c), f(st), f(rr), f(hh), f(aa), f(rd)
            }
        }' "$1"
}

# table LEDGER -> TABLE: night()'s lines for every night, newest first. A bundle awk cannot read ends the
#   table there, with its reason in TAIL, and returns 1
table() {
    local d o
    TABLE="" TAIL=""
    while IFS= read -r d; do
        o=$(night "$1/$d/bundle.tsv" "$d") || { TAIL="the bundle of night $d could not be read"; return 1; }
        TABLE+="$o"$'\n'
    done < <(nights "$1")
}

# vget KEY VERDICT -> the value of KEY in a key=value verdict file
vget() { awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$2"; }

# art_read ART RUN -> 0 when ART is H's green models-t1 artifact of MEASURE, and sets HOST_A, HOST_B;
#   1 (red) or 2 (not measured) with the reason in WHY
art_read() {
    local art=$1 run=$2 listed have go m gre x
    WHY=""
    [ -d "$art" ] || { WHY="no models-t1 artifact at '$art' for run $run"; return 2; }
    [ -f "$art/verdict" ] && [ -f "$art/SHA256SUMS" ] || { WHY="the artifact of run $run has no verdict or no SHA256SUMS"; return 2; }
    (cd -- "$art" && sha256sum --strict --quiet -c SHA256SUMS > /dev/null 2>&1) \
        || { WHY="the artifact of run $run fails its own SHA256SUMS"; return 2; }
    listed=$(awk '{ print $2 }' "$art/SHA256SUMS" | LC_ALL=C sort)
    have=$(cd -- "$art" && find . -mindepth 1 -maxdepth 1 ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort)
    [ "$listed" = "$have" ] || { WHY="the artifact of run $run holds an entry its SHA256SUMS does not list"; return 2; }
    x=$(vget commit "$art/verdict")
    [ "$x" = "$HEAD" ] || { WHY="the artifact of run $run names commit ${x:0:10}, not H"; return 2; }
    m=$(vget measure "$art/verdict")
    [ "${m:-ladder}" = "$MEASURE" ] || { WHY="the artifact of run $run measured ${m:-ladder}, not $MEASURE"; return 2; }
    x=$(vget state "$art/verdict")
    case $x in
        green) ;;
        red) WHY="the artifact's verdict for run $run is red"; return 1 ;;
        *) WHY="the artifact's verdict for run $run is '$x', not green"; return 2 ;;
    esac
    [ -f "$art/models-t1.log" ] || { WHY="the artifact of run $run has no models-t1.log"; return 2; }
    if grep -qE '^MODELS ([^ ]+ )?NO-GO' "$art/models-t1.log"; then
        WHY="the artifact's models-t1.log holds a NO-GO line"; return 2
    fi
    if [ "$MEASURE" = crux ]; then gre='^MODELS GO \(CRUX smoke\) on '; else gre='^MODELS GO on '; fi
    go=$(grep -E "${gre}[A-Za-z0-9._-]+ and [A-Za-z0-9._-]+ at [0-9a-f]{9}:" "$art/models-t1.log" | tail -n 1)
    if [ -z "$go" ]; then
        if [ "$MEASURE" = crux ]; then WHY="the artifact's models-t1.log has no models_t1.sh GO (CRUX smoke) line"
        else WHY="the artifact's models-t1.log has no models_t1.sh GO line"; fi
        return 2
    fi
    gre='^MODELS GO (\(CRUX smoke\) )?on ([A-Za-z0-9._-]+) and ([A-Za-z0-9._-]+) at ([0-9a-f]{9}):'
    [[ $go =~ $gre ]] || { WHY="the GO line in models-t1.log does not parse"; return 2; }
    HOST_A=${BASH_REMATCH[2]} HOST_B=${BASH_REMATCH[3]}
    [ "${BASH_REMATCH[4]}" = "${HEAD:0:9}" ] || { WHY="the GO line in models-t1.log is at ${BASH_REMATCH[4]}, not H's ${HEAD:0:9}"; return 2; }
    for x in "$HOST_A" "$HOST_B"; do
        [ -f "$art/$x$RSUF.json" ] || { WHY="the artifact holds no receipt $x$RSUF.json for $x, which the GO line names"; return 2; }
    done
}

# lane_read LANE ART -> LANE's NIGHT line; returns 0 GO, 1 RED, 2 NOT_MEASURED. For models it also
#   reads ART (art_read), and sets HOST_A and HOST_B
lane_read() {
    local lane=$1 art=$2 d l c s r rh a rd seen=0 go_d="" go_r="" go_h="" go_a="" hist="" red="" why="" k
    while IFS=$'\t' read -r d l c s r rh a rd; do
        [ "$l" = "$lane" ] || continue
        if ! [[ $c =~ ^[0-9a-f]{40}$ ]]; then  # no readable C: the train writes its "# C" line last
            if [ "$rh" = "$HEAD" ]; then
                if [ "$s" = red ]; then
                    [ -n "$red" ] || red="night $d has no readable C, and its $lane run (run $r) was red on H"
                else
                    [ -n "$why" ] || why="night $d has no readable C, and its $lane run (run $r) was on H"
                fi
            elif [ "$s" != not_measured ] && ! [[ $rh =~ ^[0-9a-f]{40}$ ]]; then
                [ -n "$why" ] || why="night $d has no readable C and $lane $s names no commit (run $r): it may be H's"
            fi
            continue
        fi
        [ "$c" = "$HEAD" ] || continue
        [ "$s" != green ] || [ "$rd" = ok ] || s=unread  # a green from a failed read of GitHub is no green
        seen=1; hist+=" $d=$s"
        case $s in
            green) [ -n "$go_d" ] || { go_d=$d; go_r=$r; go_h=$rh; go_a=$a; } ;;
            not_measured|absent|unread) ;;
            red) [ -n "$red" ] || red="night $d read $lane red on H (run $r)" ;;
            ambiguous) [ -n "$why" ] || why="night $d has more than one $lane row on H: which one the train meant is unknown" ;;
            unreadable:*) [ -n "$why" ] || why="night $d of H is unreadable: ${s#unreadable: }; a red could hide in it" ;;
            *) [ -n "$why" ] || why="night $d read $lane $s on H (run $r): a state the train never writes" ;;
        esac
    done <<< "$TABLE"
    if [ -n "$red" ]; then printf 'NIGHT RED %s H=%s: %s\n' "$lane" "${HEAD:0:10}" "$red"; return 1; fi
    [ -z "$why" ] || { lnm "$lane" "$why"; return 2; }
    [ "$seen" = 1 ] || { lnm "$lane" "no night in the ledger measured H"; return 2; }
    [ -n "$go_d" ] || { lnm "$lane" "no night of H read $lane green:$hist"; return 2; }
    if [ "$go_h" != "$HEAD" ]; then
        case $go_h in -) k="a run that names no commit" ;; *) k="a run on ${go_h:0:10}" ;; esac
        lnm "$lane" "night $go_d read $lane green from $k, not H"; return 2
    fi
    [ "$go_a" = 1 ] || { lnm "$lane" "night $go_d read $lane green at attempt $go_a: a retried green is not a green"; return 2; }
    if [ "$lane" != models ]; then
        printf 'NIGHT GO %s H=%s night=%s run=%s\n' "$lane" "${HEAD:0:10}" "$go_d" "$go_r"; return 0
    fi
    art_read "$art" "$go_r"; k=$?
    case $k in
        0) printf 'NIGHT GO models H=%s night=%s run=%s measure=%s\n' "${HEAD:0:10}" "$go_d" "$go_r" "$MEASURE"; return 0 ;;
        1) printf 'NIGHT RED models H=%s: %s\n' "${HEAD:0:10}" "$WHY"; return 1 ;;
        *) lnm models "$WHY"; return 2 ;;
    esac
}

# read_night LEDGER ARTIFACT OUT -> one NIGHT line per lane and a summary; exit 0 GO, 1 RED, 2 NOT_MEASURED
read_night() {
    local ledger=$1 art=$2 out=$3 x lane red="" nm=""
    [[ $HEAD =~ ^[0-9a-f]{40}$ ]] || caller_error "--head '$HEAD' is not a full 40-hex sha"
    case $MEASURE in
        crux) RSUF=-gpu ;;
        ladder) RSUF="" ;;
        *) caller_error "--measure must be crux or ladder (got '$MEASURE'): the model measure H ships on" ;;
    esac
    [ -n "$ledger" ] || caller_error "--ledger is required"
    [ -n "$art" ] || caller_error "--artifact is required"
    [ -n "$out" ] || caller_error "--out is required"
    if [ -e "$out" ] && { [ ! -d "$out" ] || [ -n "$(find "$out" -mindepth 1 -print -quit)" ]; }; then
        caller_error "--out '$out' exists and is not empty: stale receipts would mix with the night's"
    fi
    [ -d "$ledger" ] || whole "no ledger at '$ledger': release day re-runs nothing to make one"
    x=$(strays "$ledger")
    [ -z "$x" ] || whole "'$x' under the ledger is not a UTC day dir, nor the train's cache: whose night it holds is unknown"
    table "$ledger" || whole "$TAIL: a red of H could hide in it"
    for lane in "${LANES[@]}"; do
        lane_read "$lane" "$art"
        case $? in 0) ;; 1) red+=" $lane" ;; *) nm+=" $lane" ;; esac
    done
    if [ -n "$red" ]; then printf 'NIGHT RED H=%s: red:%s\n' "${HEAD:0:10}" "$red"; exit 1; fi
    if [ -n "$nm" ]; then printf 'NIGHT NOT_MEASURED H=%s: not measured:%s; never a GO\n' "${HEAD:0:10}" "$nm"; exit 2; fi
    mkdir -p -- "$out" || caller_error "cannot create --out '$out'"
    cp -- "$art/$HOST_A$RSUF.json" "$art/$HOST_B$RSUF.json" "$out/" || whole "cannot copy the receipts into '$out'"
    if [ -f "$art/judge.log" ]; then cp -- "$art/judge.log" "$out/judge.log" || whole "cannot copy judge.log into '$out'"; fi
    grep -E '^MODELS ' "$art/models-t1.log"
    printf 'NIGHT GO H=%s: every lane read green on a night of H, and release day re-ran none\n' "${HEAD:0:10}"
}

# day SPEC -> the UTC calendar day GNU date makes of SPEC, empty when it makes none
day() { date -u -d "$1" +%F 2> /dev/null; } # bashrs disable-line=DET002 (calendar arithmetic on an input day, never the clock)

# streak LEDGER TODAY -> one NIGHT STREAK line per lane; exit 0 when every lane counts NEED or more green
#   nights at the newest end of the ledger, the newest being TODAY or the day before
streak() {
    local ledger=$1 today=$2 x lane d l c s r rh a rd n prev want seen why stop="" short=0
    [ -n "$ledger" ] || caller_error "--ledger is required"
    [[ $today =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [ "$(day "$today")" = "$today" ] \
        || caller_error "--streak needs --today YYYY-MM-DD, a calendar day (got '$today')"
    if [ ! -d "$ledger" ]; then
        stop="no ledger at $ledger"
    else
        x=$(strays "$ledger")
        if [ -n "$x" ]; then stop="'$x' under the ledger is not a UTC day dir, nor the train's cache"
        else table "$ledger" || :; fi
    fi
    for lane in "${LANES[@]}"; do
        n=0 prev="" seen="" why=$stop
        if [ -z "$why" ]; then
            while IFS=$'\t' read -r d l c s r rh a rd; do
                [ "$l" = "$lane" ] || continue
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
                if [ "$s" != green ] || ! [[ $c =~ ^[0-9a-f]{40}$ ]] || [ "$rh" != "$c" ] || [ "$a" != 1 ]; then
                    why="$d read $lane $s from run $r on ${rh:0:10} at attempt $a"; break
                fi
                if [ "$rd" != ok ]; then why="$d was not read ($rd)"; break; fi
                n=$((n + 1)); seen+=" $d"; prev=$d
            done <<< "$TABLE"
            [ -n "$why" ] || why=${TAIL:-no older night}
        fi
        printf 'NIGHT STREAK %s %s of %s:%s; stopped: %s\n' "$lane" "$n" "$NEED" "${seen:- none}" "$why"
        [ "$n" -ge "$NEED" ] || short=1
    done
    [ "$short" = 0 ]
}

# ---- the case table -------------------------------------------------------------------------------
H=4701470147014701470147014701470147014701
X=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

# ledger DIR SPEC... -> one night per SPEC "day:case[:mod,mod...]" from release_night_cases/<case>, the
#   mods applied in order (see mod)
ledger() {
    local dir=$1 spec day kase mods m b; shift
    local -a ms
    mkdir -p -- "$dir" || return 1
    for spec in "$@"; do
        day=${spec%%:*}; kase=${spec#*:}; mods=""
        case $kase in *:*) mods=${kase#*:}; kase=${kase%%:*} ;; esac
        [[ $day =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || return 1
        mkdir -p -- "$dir/$day" || return 1; b="$dir/$day/bundle.tsv"
        cp -- "$CASES/$kase/bundle.tsv" "$b" || return 1
        IFS=, read -r -a ms <<< "$mods"
        for m in "${ms[@]}"; do mod "$m" "$b" || return 1; done
    done
}

# mod NAME BUNDLE -> BUNDLE with one change. all: every lane but models reads green on H (runs 701..706)
#   at attempt 1; x: the night measured X, not H; head: the models run is on X; att2/att1: the models row
#   is at attempt 2/1 (a red row carries no attempt); dup: a second, red, models row; dupg: a second,
#   green, models row; dupex: a second, red, deep-examples row; exred: deep-examples reads red; covred:
#   the info lane coverage reads red on H; nodoc: no deep-doctests row; odd: the models state is void,
#   which the train never writes; noc: no "# C" line, as when the train's write was cut short; trunc: no
#   "# C" line and no models row; cr/sp: the "# C" value ends in a CR/a space; unread: the "# read" line
#   is a failed read; renamed: the header calls run_head "head"; nolane: the header calls lane "name";
#   nostate/norunid: the header calls state "status"/run_id "run"; norh: the dogfood row has no run head
#   (an empty field, which a tab-split read would collapse into the next column's value);
#   reorder: the state and producer columns swap places, header and rows alike
mod() {
    local m=$1 b=$2
    case $m in
        all) awk -F '\t' -v OFS='\t' -v lanes="${LANES[*]}" -v h="$H" '
                 BEGIN { k = split(lanes, L, " "); for (i = 1; i <= k; i++) if (L[i] != "models") v[L[i]] = 700 + i }
                 $1 in v { $3 = "green"; $5 = v[$1]; $6 = h; $7 = "success"; $8 = 1 } 1' "$b" ;;
        x) sed "s/$H/$X/g" "$b" ;;
        head) awk -F '\t' -v OFS='\t' -v x="$X" '$1 == "models" { $6 = x } 1' "$b" ;;
        att2) awk -F '\t' -v OFS='\t' '$1 == "models" { $8 = 2 } 1' "$b" ;;
        att1) awk -F '\t' -v OFS='\t' '$1 == "models" { $8 = 1 } 1' "$b" ;;
        dup) awk -F '\t' -v OFS='\t' '1; $1 == "models" { $3 = "red"; print }' "$b" ;;
        dupg) awk -F '\t' '1; $1 == "models"' "$b" ;;
        dupex) awk -F '\t' -v OFS='\t' '1; $1 == "deep-examples" { $3 = "red"; $7 = "failure"; print }' "$b" ;;
        exred) awk -F '\t' -v OFS='\t' '$1 == "deep-examples" { $3 = "red"; $7 = "failure"; $8 = "" } 1' "$b" ;;
        covred) awk -F '\t' -v OFS='\t' -v h="$H" '$1 == "coverage" { $3 = "red"; $5 = 900; $6 = h; $7 = "failure"; $8 = 1 } 1' "$b" ;;
        nodoc) awk -F '\t' '$1 != "deep-doctests"' "$b" ;;
        odd) awk -F '\t' -v OFS='\t' '$1 == "models" { $3 = "void" } 1' "$b" ;;
        noc) awk -F '\t' '$1 != "# C"' "$b" ;;
        trunc) awk -F '\t' '$1 != "# C" && $1 != "models"' "$b" ;;
        cr) awk -F '\t' -v OFS='\t' '$1 == "# C" { $2 = $2 "\r" } 1' "$b" ;;
        sp) awk -F '\t' -v OFS='\t' '$1 == "# C" { $2 = $2 " " } 1' "$b" ;;
        unread) awk -F '\t' -v OFS='\t' '$1 == "# read" { $2 = "failed: call budget" } 1' "$b" ;;
        renamed) awk -F '\t' -v OFS='\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "run_head") $i = "head" } 1' "$b" ;;
        nolane) awk -F '\t' -v OFS='\t' 'NR == 1 { $1 = "name" } 1' "$b" ;;
        nostate) awk -F '\t' -v OFS='\t' 'NR == 1 { $3 = "status" } 1' "$b" ;;
        norunid) awk -F '\t' -v OFS='\t' 'NR == 1 { $5 = "run" } 1' "$b" ;;
        norh) awk -F '\t' -v OFS='\t' '$1 == "dogfood" { $6 = "" } 1' "$b" ;;
        reorder) awk -F '\t' -v OFS='\t' '!/^#/ { x = $3; $3 = $4; $4 = x } 1' "$b" ;;
        *) return 1 ;;
    esac > "$b.new" && mv -- "$b.new" "$b"
}

# seal DIR -> SHA256SUMS over every file in DIR, as models_nightly.sh seals a bundle
seal() { (cd -- "$1" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort | xargs -r sha256sum -- > SHA256SUMS); }

# artifact DIR MEASURE [MOD] -> the models-t1 artifact of H's green night, as models_nightly.sh seals it
#   for MEASURE (crux: <host>-gpu.json receipts and the CRUX smoke GO line); MOD breaks one thing.
#   measure: the verdict names the other measure; nomeasure: it names none; othergo: the GO line is the
#   other measure's; rsuf: the receipts carry the other measure's names
artifact() {
    local a=$1 ms=$2 mod=${3:-} c=$H g=${H:0:9} st=green rs="" ors=-gpu vm=$2 other=crux go="MODELS GO on" \
        f="executed=12 red=0 (model_ladder rc 0)" ogo="MODELS GO (CRUX smoke) on" x
    case $ms in
        crux) rs=-gpu ors="" other=ladder go="MODELS GO (CRUX smoke) on" ogo="MODELS GO on" f="cells=40 red=0 (crux_sweep_shards rc 0)" ;;
        ladder) ;;
        *) return 1 ;;
    esac
    case $mod in commit) c=$X ;; state) st=red ;; sha9) g=${X:0:9} ;; measure) vm=$other ;; nomeasure) vm="" ;; othergo) go=$ogo ;; esac
    mkdir -p -- "$a" || return 1
    {
        printf 'commit=%s\nsha9=%s\nversion=0.71.0\nentry=scripts/release/models_t1.sh@%s\n' "$c" "${H:0:9}" "$X"
        [ -z "$vm" ] || printf 'measure=%s\n' "$vm"
        printf 't1_rc=0\nstate=%s\nreason=fixture\n' "$st"
    } > "$a/verdict" || return 1
    {
        printf 'MODELS lambda measured by apr 0.71.0 (%s): %s\n' "${H:0:9}" "$f"
        printf 'MODELS gx10 measured by apr 0.71.0 (%s): %s\n' "${H:0:9}" "$f"
        printf '%s lambda and gx10 at %s: the judge passed both receipts (apr 0.71.0 (%s))\n' "$go" "$g" "${H:0:9}"
    } > "$a/models-t1.log" || return 1
    [ "$mod" = rsuf ] && rs=$ors
    for x in lambda gx10; do printf '{"host":"%s"}\n' "$x" > "$a/$x$rs.json" || return 1; done
    printf 'judge: PASS\n' > "$a/judge.log" || return 1
    [ "$mod" != nohost ] || rm -f -- "${a:?}/gx10$rs.json" || return 1
    [ "$mod" != nogoline ] || printf 'MODELS gx10 NO-GO: no receipt -- fixture\n' >> "$a/models-t1.log" || return 1
    seal "$a" || return 1
    case $mod in
        tamper) printf 'x\n' >> "$a/lambda$rs.json" ;;
        extra) printf 'x\n' > "$a/unlisted" ;;
        subdir) mkdir -p -- "$a/unlisted" ;;
        nosums) rm -f -- "${a:?}/SHA256SUMS" ;;
    esac
}

# noawk DIR -> DIR/awk, an awk that fails on every call: the subject run with DIR first on PATH cannot read
#   a bundle, as on an I/O error or a bundle it may not open
noawk() { mkdir -p -- "$1" && printf '#!/bin/sh\nexit 2\n' > "$1/awk" && chmod +x -- "$1/awk"; }

# files OUT CMD... -> CMD's output, then the files it left in OUT
files() {
    local out=$1 rc; shift
    "$@"; rc=$?
    printf 'files: %s\n' "$(find "$out" -mindepth 1 -maxdepth 1 -printf '%f\n' 2> /dev/null | LC_ALL=C sort | tr '\n' ' ')"
    return "$rc"
}

# cases SUBJECT -> runs the table against SUBJECT; sets BROKE
cases() {
    local s=$1 t m n G="H=4701470147" FG="NIGHT GO H="
    t=$(mktemp -d) || { printf 'not_measured: no temp dir\n'; BROKE=1; return; }
    BROKE=0
    row() { # row NAME RC NEEDLE FORBID -- CMD...
        local name=$1 want=$2 needle=$3 forbid=$4 o rc; shift 5
        o=$("$@" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && [[ $o == *"$needle"* ]] && { [ -z "$forbid" ] || [[ $o != *"$forbid"* ]]; }; then
            printf 'ok    %s\n' "$name"
        else
            printf 'BROKE %s: rc %s (want %s), output: %s\n' "$name" "$rc" "$want" "$(printf '%s' "$o" | tr '\n' '|' | cut -c1-400)"
            BROKE=$((BROKE + 1))
        fi
    }
    setup() { # setup CMD... -> runs one fixture step; a failed step breaks the table, never a row's verdict
        "$@" || { printf 'BROKE setup: %s\n' "$*"; BROKE=$((BROKE + 1)); }
    }
    rd() { # rd LEDGER MEASURE ARTIFACT OUT -> the subject's read of H
        bash "$s" --ledger "$1" --head "$H" --measure "$2" --artifact "$3" --out "$4"
    }
    st() { bash "$s" --streak "$@"; }

    # -- RED first: every way a lane of H can read red
    setup ledger "$t/l-red" 2026-10-06:red:all
    setup artifact "$t/a-cx" crux; setup artifact "$t/a-ld" ladder
    row "red night of H" 1 "NIGHT RED models $G: night 2026-10-06 read models red on H (run 601)" "$FG" -- rd "$t/l-red" crux "$t/a-cx" "$t/o1"
    row "the RED summary names the red lane" 1 "NIGHT RED $G: red: models" "$FG" -- rd "$t/l-red" crux "$t/a-cx" "$t/o2"
    setup ledger "$t/l-redold" 2026-10-06:green:all 2026-10-05:red:all
    row "a later green never outvotes a red on H" 1 "NIGHT RED models $G: night 2026-10-05 read models red" "$FG" -- rd "$t/l-redold" crux "$t/a-cx" "$t/o3"
    setup ledger "$t/l-dup" 2026-10-06:green:all 2026-10-05:green:all,dup
    row "a red row beside a green one on a night of H" 1 "NIGHT RED models $G: night 2026-10-05 read models red on H (run 601)" "$FG" -- rd "$t/l-dup" crux "$t/a-cx" "$t/o4"
    setup ledger "$t/l-dupex" 2026-10-06:green:all,dupex
    row "a red deep-examples row behind its green twin" 1 "NIGHT RED deep-examples $G: night 2026-10-06 read deep-examples red on H (run 703)" "$FG" -- rd "$t/l-dupex" crux "$t/a-cx" "$t/o5"
    setup ledger "$t/l-exred" 2026-10-06:green:all,exred
    row "a red lane besides models is RED" 1 "NIGHT RED $G: red: deep-examples" "$FG" -- rd "$t/l-exred" crux "$t/a-cx" "$t/o6"
    setup ledger "$t/l-mix" 2026-10-06:not_measured:all,exred
    row "RED outranks NOT_MEASURED" 1 "NIGHT RED $G: red: deep-examples" "$FG" -- rd "$t/l-mix" crux "$t/a-cx" "$t/o7"
    setup ledger "$t/l-noc" 2026-10-06:green:all 2026-10-05:red:all,noc
    row "a night with no C whose models run was red on H" 1 "NIGHT RED models $G: night 2026-10-05 has no readable C, and its models run (run 601) was red on H" "$FG" -- rd "$t/l-noc" crux "$t/a-cx" "$t/o8"
    setup artifact "$t/ac-state" crux state
    setup ledger "$t/l-go" 2026-10-06:green:all
    row "an artifact whose verdict is red, on a green night" 1 "NIGHT RED models $G: the artifact's verdict for run 601 is red" "$FG" -- rd "$t/l-go" crux "$t/ac-state" "$t/o10"

    # -- GO
    row "crux: every lane green on a night of H is GO" 0 "NIGHT GO $G: every lane read green on a night of H" "" -- rd "$t/l-go" crux "$t/a-cx" "$t/o11"
    row "crux: the models line names the night, the run and the measure" 0 "NIGHT GO models $G night=2026-10-06 run=601 measure=crux" "" -- rd "$t/l-go" crux "$t/a-cx" "$t/o12"
    row "crux: each lane names its own run" 0 "NIGHT GO dogfood $G night=2026-10-06 run=706" "" -- rd "$t/l-go" crux "$t/a-cx" "$t/o13"
    row "crux: GO keeps models_t1.sh's MODELS lines" 0 "MODELS GO (CRUX smoke) on lambda and gx10 at 470147014" "" -- rd "$t/l-go" crux "$t/a-cx" "$t/o14"
    row "crux: GO leaves the -gpu receipts and judge.log in --out" 0 "files: gx10-gpu.json judge.log lambda-gpu.json" "" -- files "$t/o15" rd "$t/l-go" crux "$t/a-cx" "$t/o15"
    row "ladder: GO" 0 "NIGHT GO models $G night=2026-10-06 run=601 measure=ladder" "" -- rd "$t/l-go" ladder "$t/a-ld" "$t/o16"
    row "ladder: GO leaves the ladder receipts in --out" 0 "files: gx10.json judge.log lambda.json" "" -- files "$t/o17" rd "$t/l-go" ladder "$t/a-ld" "$t/o17"
    setup artifact "$t/al-nomeasure" ladder nomeasure
    row "ladder: a verdict sealed before measure= existed is a ladder one" 0 "measure=ladder" "" -- rd "$t/l-go" ladder "$t/al-nomeasure" "$t/o18"
    setup ledger "$t/l-redx" 2026-10-06:green:all 2026-10-05:red:all,x
    row "a red on another commit is not H's" 0 "NIGHT GO models $G night=2026-10-06" "" -- rd "$t/l-redx" crux "$t/a-cx" "$t/o19"
    setup ledger "$t/l-nmnew" 2026-10-06:not_measured:all 2026-10-05:green:all
    row "an unmeasured night of H does not hide an older green" 0 "NIGHT GO models $G night=2026-10-05 run=601" "" -- rd "$t/l-nmnew" crux "$t/a-cx" "$t/o20"
    setup ledger "$t/l-2g" 2026-10-06:green:all 2026-10-05:green:all
    row "two green nights of H: GO names the newest" 0 "NIGHT GO models $G night=2026-10-06 run=601" "" -- rd "$t/l-2g" crux "$t/a-cx" "$t/o21"
    setup ledger "$t/l-cov" 2026-10-06:green:all,covred
    row "an info lane's red is not read" 0 "NIGHT GO $G: every lane" "" -- rd "$t/l-cov" crux "$t/a-cx" "$t/o22"
    setup ledger "$t/l-nocx" 2026-10-06:green:all 2026-10-05:red:all,x,noc
    row "a night with no C whose run was on another commit" 0 "NIGHT GO models $G night=2026-10-06" "" -- rd "$t/l-nocx" crux "$t/a-cx" "$t/o23"
    setup ledger "$t/l-nocnm" 2026-10-06:green:all 2026-10-05:not_measured:noc
    row "a night with no C that measured nothing" 0 "NIGHT GO $G: every lane" "" -- rd "$t/l-nocnm" crux "$t/a-cx" "$t/o24"
    setup ledger "$t/l-cache" 2026-10-06:green:all; setup mkdir -p -- "$t/l-cache/cache"
    setup cp -- "$CASES/green/line" "$t/l-cache/history.tsv"
    row "the train's cache and a file under the ledger are not nights" 0 "NIGHT GO $G: every lane" "" -- rd "$t/l-cache" crux "$t/a-cx" "$t/o25"
    setup ledger "$t/l-hole" 2026-10-06:green:all; setup mkdir -p -- "$t/l-hole/2026-10-07"
    row "a day dir with no bundle is passed over" 0 "NIGHT GO models $G night=2026-10-06" "" -- rd "$t/l-hole" crux "$t/a-cx" "$t/o26"
    setup ledger "$t/l-unread2" 2026-10-06:green:all,unread 2026-10-05:green:all
    row "an unread night does not hide H's read green one" 0 "NIGHT GO models $G night=2026-10-05" "" -- rd "$t/l-unread2" crux "$t/a-cx" "$t/o27"
    setup ledger "$t/l-reorder" 2026-10-06:green:all,reorder
    row "columns are read by name: a reordered table still reads" 0 "NIGHT GO $G: every lane" "" -- rd "$t/l-reorder" crux "$t/a-cx" "$t/o28"

    # -- NOT_MEASURED: never a GO
    row "--measure is required" 3 "--measure must be crux or ladder (got '')" "$FG" \
        -- bash "$s" --ledger "$t/l-go" --head "$H" --artifact "$t/a-cx" --out "$t/o29"
    row "--measure must name a measure" 3 "--measure must be crux or ladder (got 'smoke')" "$FG" -- rd "$t/l-go" smoke "$t/a-cx" "$t/o30"
    row "a crux artifact is no ladder night" 2 "NIGHT NOT_MEASURED models $G: the artifact of run 601 measured crux, not ladder" "$FG" -- rd "$t/l-go" ladder "$t/a-cx" "$t/o31"
    row "a ladder artifact is no crux night" 2 "NIGHT NOT_MEASURED models $G: the artifact of run 601 measured ladder, not crux" "$FG" -- rd "$t/l-go" crux "$t/a-ld" "$t/o32"
    n=33
    for m in tamper:"fails its own SHA256SUMS" extra:"does not list" subdir:"does not list" \
             commit:"names commit bbbbbbbbbb, not H" sha9:"is at bbbbbbbbb, not H's 470147014" \
             nohost:"no receipt gx10-gpu.json for gx10" rsuf:"no receipt lambda-gpu.json for lambda" \
             nogoline:"models-t1.log holds a NO-GO line" nosums:"no verdict or no SHA256SUMS" \
             measure:"measured ladder, not crux" nomeasure:"measured ladder, not crux" \
             othergo:"has no models_t1.sh GO (CRUX smoke) line"; do
        setup artifact "$t/ac-${m%%:*}" crux "${m%%:*}"
        row "crux artifact ${m%%:*}" 2 "NIGHT NOT_MEASURED models $G: " "$FG" -- rd "$t/l-go" crux "$t/ac-${m%%:*}" "$t/o$n"
        row "crux artifact ${m%%:*}: why" 2 "${m#*:}" "$FG" -- rd "$t/l-go" crux "$t/ac-${m%%:*}" "$t/o$n"
        n=$((n + 1))
    done
    for m in measure:"measured crux, not ladder" othergo:"has no models_t1.sh GO line" rsuf:"no receipt lambda.json for lambda"; do
        setup artifact "$t/al-${m%%:*}" ladder "${m%%:*}"
        row "ladder artifact ${m%%:*}" 2 "${m#*:}" "$FG" -- rd "$t/l-go" ladder "$t/al-${m%%:*}" "$t/o$n"
        n=$((n + 1))
    done
    row "no artifact" 2 "NIGHT NOT_MEASURED models $G: no models-t1 artifact at" "$FG" -- rd "$t/l-go" crux "$t/a-none" "$t/o60"
    setup mkdir -p -- "$t/o61"; setup cp -- "$CASES/green/line" "$t/o61/old.json"
    row "a non-empty --out is refused" 3 "is not empty" "$FG" -- rd "$t/l-go" crux "$t/a-cx" "$t/o61"
    row "H must be a full sha" 3 "not a full 40-hex sha" "$FG" \
        -- bash "$s" --ledger "$t/l-go" --head "${H:0:12}" --measure crux --artifact "$t/a-cx" --out "$t/o62"
    setup ledger "$t/l-other" 2026-10-06:green:all,x
    row "no night measured H" 2 "NIGHT NOT_MEASURED models $G: no night in the ledger measured H" "$FG" -- rd "$t/l-other" crux "$t/a-cx" "$t/o63"
    setup mkdir -p -- "$t/l-empty"
    row "empty ledger" 2 "NIGHT NOT_MEASURED dogfood $G: no night in the ledger measured H" "$FG" -- rd "$t/l-empty" crux "$t/a-cx" "$t/o64"
    row "no ledger" 2 "NIGHT NOT_MEASURED $G: no ledger at" "$FG" -- rd "$t/l-none" crux "$t/a-cx" "$t/o65"
    setup ledger "$t/l-nm" 2026-10-06:not_measured:all
    row "not_measured is never a GO" 2 "no night of H read models green: 2026-10-06=not_measured" "$FG" -- rd "$t/l-nm" crux "$t/a-cx" "$t/o66"
    setup ledger "$t/l-np" 2026-10-06:no_producer:all
    row "no models producer yet" 2 "no night of H read models green: 2026-10-06=not_measured" "$FG" -- rd "$t/l-np" crux "$t/a-cx" "$t/o67"
    setup ledger "$t/l-solo" 2026-10-06:green
    row "models green alone is not a GO" 2 "NIGHT NOT_MEASURED $G: not measured: deep-doctests" "MODELS GO" -- rd "$t/l-solo" crux "$t/a-cx" "$t/o68"
    row "a read that is not a GO writes no receipts" 2 "files: " "gpu.json" -- files "$t/o69" rd "$t/l-solo" crux "$t/a-cx" "$t/o69"
    setup ledger "$t/l-nodoc" 2026-10-06:green:all,nodoc
    row "a lane with no row" 2 "NIGHT NOT_MEASURED deep-doctests $G: no night of H read deep-doctests green: 2026-10-06=absent" "$FG" -- rd "$t/l-nodoc" crux "$t/a-cx" "$t/o70"
    setup ledger "$t/l-head" 2026-10-06:green:all,head
    row "green from a run on another commit" 2 "night 2026-10-06 read models green from a run on bbbbbbbbbb, not H" "$FG" -- rd "$t/l-head" crux "$t/a-cx" "$t/o71"
    setup ledger "$t/l-att" 2026-10-06:green:all,att2
    row "green at attempt 2" 2 "night 2026-10-06 read models green at attempt 2: a retried green is not a green" "$FG" -- rd "$t/l-att" crux "$t/a-cx" "$t/o72"
    setup ledger "$t/l-dupg" 2026-10-06:green:all 2026-10-05:green:all,dupg
    row "two green models rows on one night of H" 2 "night 2026-10-05 has more than one models row on H" "$FG" -- rd "$t/l-dupg" crux "$t/a-cx" "$t/o73"
    setup ledger "$t/l-odd" 2026-10-06:green:all 2026-10-05:green:all,odd
    row "a state the train never writes, on H" 2 "night 2026-10-05 read models void on H (run 601): a state the train never writes" "$FG" -- rd "$t/l-odd" crux "$t/a-cx" "$t/o74"
    setup ledger "$t/l-nocg" 2026-10-06:green:all 2026-10-05:green:all,noc
    row "a night with no C whose models run was green on H" 2 "NIGHT NOT_MEASURED models $G: night 2026-10-05 has no readable C, and its models run (run 601) was on H" "$FG" -- rd "$t/l-nocg" crux "$t/a-cx" "$t/o75"
    setup ledger "$t/l-trunc" 2026-10-06:green:all 2026-10-05:green:all,trunc
    row "a cut-short night with no C and no models row" 2 "night 2026-10-05 has no readable C and models absent names no commit" "$FG" -- rd "$t/l-trunc" crux "$t/a-cx" "$t/o76"
    setup ledger "$t/l-cr" 2026-10-06:green:all,cr
    row "a C line that ends in a CR is no C" 2 "NIGHT NOT_MEASURED models $G: night 2026-10-06 has no readable C, and its models run (run 601) was on H" "$FG" -- rd "$t/l-cr" crux "$t/a-cx" "$t/o77"
    setup ledger "$t/l-sp" 2026-10-06:green:all,sp
    row "a C line that ends in a space is no C" 2 "NIGHT NOT_MEASURED models $G: night 2026-10-06 has no readable C, and its models run (run 601) was on H" "$FG" -- rd "$t/l-sp" crux "$t/a-cx" "$t/o78"
    setup ledger "$t/l-stray" 2026-10-06:green:all; setup mkdir -p -- "$t/l-stray/notes"
    row "a dir under the ledger that is not a UTC day" 2 "NIGHT NOT_MEASURED $G: 'notes' under the ledger is not a UTC day dir" "$FG" -- rd "$t/l-stray" crux "$t/a-cx" "$t/o79"
    setup ledger "$t/l-unread" 2026-10-06:green:all,unread
    row "a green from a failed read of GitHub is not counted" 2 "no night of H read models green: 2026-10-06=unread" "$FG" -- rd "$t/l-unread" crux "$t/a-cx" "$t/o80"
    setup ledger "$t/l-renamed" 2026-10-06:green:all,renamed
    row "a renamed column on a night of H" 2 "night 2026-10-06 of H is unreadable: the header has no run_head column" "$FG" -- rd "$t/l-renamed" crux "$t/a-cx" "$t/o81"
    setup ledger "$t/l-nolane" 2026-10-06:green:all 2026-10-05:red:all,nolane
    row "no lane column never passes a red night of H" 2 "night 2026-10-05 of H is unreadable: the header has no lane column" "$FG" -- rd "$t/l-nolane" crux "$t/a-cx" "$t/o82"
    setup noawk "$t/noawk"
    setup ledger "$t/l-nostate" 2026-10-06:green:all 2026-10-05:red:all,nostate
    row "no state column never passes a red night of H" 2 "night 2026-10-05 of H is unreadable: the header has no state column" "$FG" -- rd "$t/l-nostate" crux "$t/a-cx" "$t/o84"
    setup ledger "$t/l-norunid" 2026-10-06:green:all 2026-10-05:green:all,norunid
    row "no run_id column: never a guessed run" 2 "night 2026-10-05 of H is unreadable: the header has no run_id column" "$FG" -- rd "$t/l-norunid" crux "$t/a-cx" "$t/o85"
    setup ledger "$t/l-norh" 2026-10-06:green:all,norh
    row "a green whose run names no commit is not H's" 2 "NIGHT NOT_MEASURED dogfood $G: night 2026-10-06 read dogfood green from a run that names no commit, not H" "$FG" \
        -- rd "$t/l-norh" crux "$t/a-cx" "$t/o86"
    setup mkdir -p -- "$t/l-onlyhole/2026-10-06"
    row "a ledger holding only a night with no bundle" 2 "NIGHT NOT_MEASURED models $G: no night in the ledger measured H" "$FG" -- rd "$t/l-onlyhole" crux "$t/a-cx" "$t/o88"
    row "a bundle awk cannot read" 2 "NIGHT NOT_MEASURED $G: the bundle of night 2026-10-06 could not be read" "$FG" \
        -- env PATH="$t/noawk:$PATH" bash "$s" --ledger "$t/l-go" --head "$H" --measure crux --artifact "$t/a-cx" --out "$t/o83"

    # -- the streak, per lane
    setup ledger "$t/s3" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:green:all
    row "streak: 3 green nights" 0 "NIGHT STREAK models 3 of 3: 2026-10-06 2026-10-05 2026-10-04" "" -- st --ledger "$t/s3" --today 2026-10-06
    row "streak: every lane is counted" 0 "NIGHT STREAK deep-doctests 3 of 3" "" -- st --ledger "$t/s3" --today 2026-10-06
    setup ledger "$t/s3m" 2026-10-06:green 2026-10-05:green 2026-10-04:green
    row "streak: models alone does not meet it" 1 "NIGHT STREAK deep-doctests 0 of 3: none; stopped: 2026-10-06 read deep-doctests not_measured" "" \
        -- st --ledger "$t/s3m" --today 2026-10-06
    row "streak: models alone still counts its own nights" 1 "NIGHT STREAK models 3 of 3" "" -- st --ledger "$t/s3m" --today 2026-10-06
    setup ledger "$t/s4" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:green:all 2026-10-03:green:all
    row "streak: 4 green nights" 0 "NIGHT STREAK models 4 of 3" "" -- st --ledger "$t/s4" --today 2026-10-06
    setup ledger "$t/sx" 2026-10-06:green:all,x 2026-10-05:green:all 2026-10-04:green:all,x
    row "streak: main moved between nights" 0 "NIGHT STREAK models 3 of 3" "" -- st --ledger "$t/sx" --today 2026-10-06
    setup ledger "$t/sred" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:red:all
    row "streak: a red night ends it" 1 "STREAK models 2 of 3: 2026-10-06 2026-10-05; stopped: 2026-10-04 read models red" "" \
        -- st --ledger "$t/sred" --today 2026-10-06
    setup ledger "$t/sgap" 2026-10-06:green:all 2026-10-05:green:all 2026-10-03:green:all
    row "streak: a missing night ends it" 1 "STREAK models 2 of 3: 2026-10-06 2026-10-05; stopped: no night on 2026-10-04" "" \
        -- st --ledger "$t/sgap" --today 2026-10-06
    setup ledger "$t/snm" 2026-10-06:not_measured:all 2026-10-05:green:all 2026-10-04:green:all 2026-10-03:green:all
    row "streak: the newest night unmeasured" 1 "STREAK models 0 of 3: none; stopped: 2026-10-06 read models not_measured" "" \
        -- st --ledger "$t/snm" --today 2026-10-06
    setup ledger "$t/satt" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:green:all,att2
    row "streak: a retried green ends it" 1 "STREAK models 2 of 3" "" -- st --ledger "$t/satt" --today 2026-10-06
    setup ledger "$t/sred1" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:red:all,att1
    row "streak: a red night ends it even on its own commit at attempt 1" 1 "stopped: 2026-10-04 read models red from run 601" "" \
        -- st --ledger "$t/sred1" --today 2026-10-06
    setup ledger "$t/shead" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:green:all,head
    row "streak: a green from a run on another commit ends it" 1 "STREAK models 2 of 3" "" -- st --ledger "$t/shead" --today 2026-10-06
    setup ledger "$t/snp" 2026-10-06:no_producer:all 2026-10-05:no_producer:all 2026-10-04:no_producer:all
    row "streak: no producer yet" 1 "STREAK models 0 of 3" "" -- st --ledger "$t/snp" --today 2026-10-06
    row "streak: no ledger" 1 "STREAK models 0 of 3: none; stopped: no ledger at" "" -- st --ledger "$t/s-none" --today 2026-10-06
    row "streak: the newest night may be yesterday" 0 "NIGHT STREAK models 3 of 3" "" -- st --ledger "$t/s3" --today 2026-10-07
    row "streak: an old streak counts nothing" 1 "STREAK models 0 of 3: none; stopped: the newest night is 2026-10-06, not 2026-10-09 or 2026-10-08" "" \
        -- st --ledger "$t/s3" --today 2026-10-09
    row "streak: --today is required" 3 "--streak needs --today YYYY-MM-DD" "NIGHT STREAK" -- st --ledger "$t/s3"
    row "streak: --today must be a calendar day" 3 "a calendar day (got '2026-02-30')" "NIGHT STREAK" -- st --ledger "$t/s3" --today 2026-02-30
    setup ledger "$t/scal" 2026-03-02:green:all 2026-03-01:green:all 2026-02-30:green:all 2026-02-28:green:all
    row "streak: a night that is not a calendar day ends it" 1 "STREAK models 2 of 3: 2026-03-02 2026-03-01; stopped: 2026-02-30 is not a calendar day" "" \
        -- st --ledger "$t/scal" --today 2026-03-02
    setup ledger "$t/sunread" 2026-10-06:green:all 2026-10-05:green:all,unread 2026-10-04:green:all
    row "streak: a night the train did not read ends it" 1 "STREAK models 1 of 3: 2026-10-06; stopped: 2026-10-05 was not read (failed: call budget)" "" \
        -- st --ledger "$t/sunread" --today 2026-10-06
    setup ledger "$t/sstray" 2026-10-06:green:all 2026-10-05:green:all 2026-10-04:green:all; setup mkdir -p -- "$t/sstray/notes"
    row "streak: a dir that is not a UTC day counts nothing" 1 "STREAK models 0 of 3: none; stopped: 'notes' under the ledger is not a UTC day dir" "" \
        -- st --ledger "$t/sstray" --today 2026-10-06
    row "streak: a bundle awk cannot read ends it" 1 "STREAK models 0 of 3: none; stopped: the bundle of night 2026-10-06 could not be read" "" \
        -- env PATH="$t/noawk:$PATH" bash "$s" --streak --ledger "$t/s3" --today 2026-10-06
    rm -rf -- "${t:?}"
}

# the planted mutants: each must change the file, still parse, and break at least one row
MUTANTS=(
    's/if (\$col\["state"\] == "red" \&\& !(l in red))/if (0)/'
    's/else if (l in red) { st = "red"/else if (0) { st = "red"/'
    's/i <= m; i++) if/i <= 0; i++) if/'
    's/s\[l\] = \$col\["state"\]/s[l] = $3/'
    's/\[ "\$s" = red \]; then/false; then/'
    's/if \[ "\$rh" = "\$HEAD" \]; then/if false; then/'
    's/elif \[ "\$s" != not_measured \] \&\& /elif false \&\& /'
    's/\[ "\$c" = "\$HEAD" \] || continue/true || continue/'
    's/|| \[ "\$rd" = ok \] || s=unread/|| true/'
    's/green) \[ -n "\$go_d" \] ||/green) false ||/'
    's/not_measured|absent|unread) ;;/not_measured|absent|unread|ambiguous) ;;/'
    's/red) \[ -n "\$red" \] || red=/red) true || red=/'
    's/unreadable:\*) \[ -n "\$why" \] ||/unreadable:*) true ||/'
    's/\*) \[ -n "\$why" \] || why="night \$d read/*) true || why="night $d read/'
    's/\[ -z "\$why" \] || { lnm/true || { lnm/'
    's/\[ "\$seen" = 1 \] || { lnm/true || { lnm/'
    's/if \[ "\$go_h" != "\$HEAD" \]; then/if false; then/'
    's/m = split("lane state run_id run_head attempt"/m = split("lane run_head attempt"/'
    's/function f(x) { return x == "" ? "-" : x }/function f(x) { return x }/'
    's/\[ "\$go_a" = 1 \] || { lnm/true || { lnm/'
    's/if \[ "\$lane" != models \]; then/if true; then/'
    's/sha256sum --strict --quiet -c SHA256SUMS/true/'
    's/\[ "\$listed" = "\$have" \] || { WHY/true || { WHY/'
    's/\[ "\$x" = "\$HEAD" \] || { WHY/true || { WHY/'
    's/\[ "\${m:-ladder}" = "\$MEASURE" \]/true/'
    's/\[ "\${m:-ladder}" = /[ "${m:-crux}" = /'
    's/        red) WHY="the artifact.s verdict for run \$run is red"; return 1 ;;/        red) ;;/'
    's/if grep -qE .^MODELS (\[^ \]+ )?NO-GO. /if false /'
    "s/gre='^MODELS GO \\\\(CRUX smoke\\\\) on '; else/gre='^MODELS GO (\\\\(CRUX smoke\\\\) )?on '; else/"
    's/\[ "\${BASH_REMATCH\[4\]}" = "\${HEAD:0:9}" \]/true/'
    's/\[ -f "\$art\/\$x\$RSUF.json" \] || { WHY/true || { WHY/'
    's/        crux) RSUF=-gpu ;;/        crux) RSUF="" ;;/'
    's/\[ -z "\$x" \] || whole/true || whole/'
    's/table "\$ledger" || whole/table "$ledger" || :/'
    's/\[ "\$d" = cache \] || //'
    's/case \$? in 0) ;; 1) red+=/case $? in 0|2) ;; 1) red+=/'
    's/if \[ -n "\$red" \]; then printf \(.\)NIGHT RED H=/if false; then printf \1NIGHT RED H=/'
    's/if \[ -n "\$nm" \]; then printf/if false; then printf/'
    's/if \[ "\$d" != "\$today" \]/if false/'
    's/ \&\& \[ "\$d" != "\$(day "\$today -1 day")" \]//'
    's/if \[ "\$(day "\$d")" != "\$d" \]; then why/if false; then why/'
    's/if \[ "\$d" != "\$want" \]; then/if false; then/'
    's/^NEED=3$/NEED=2/'
    's/if \[ "\$s" != green \] || /if false || /'
    's/|| \[ "\$rh" != "\$c" \] || \[ "\$a" != 1 \]; then/; then/'
    's/if \[ "\$rd" != ok \]; then why/if false; then why/'
    's/\[ "\$n" -ge "\$NEED" \] || short=1/true || short=1/'
    's/if \[ -n "\$x" \]; then stop=/if false; then stop=/'
    's/\[ -n "\$why" \] || why=\${TAIL:-no older night}/[ -n "$why" ] || why="no older night"/'
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

MODE=read LEDGER="" HEAD="" ART="" OUT="" TODAY="" MEASURE=""
while [ $# -gt 0 ]; do
    case $1 in
        --ledger|--head|--measure|--artifact|--out|--today)
            [ $# -ge 2 ] || caller_error "$1 needs a value"
            case $1 in
                --ledger) LEDGER=$2 ;; --head) HEAD=$2 ;; --measure) MEASURE=$2 ;;
                --artifact) ART=$2 ;; --out) OUT=$2 ;; --today) TODAY=$2 ;;
            esac
            shift 2 ;;
        --streak) MODE=streak; shift ;;
        --self-test) MODE=selftest; shift ;;
        --mutants) MODE=mutants; shift ;;
        -h|--help) sed -n '2,7p' "$SELF"; exit 0 ;;
        *) caller_error "unknown argument '$1' (see the header of $SELF)" ;;
    esac
done
case $MODE in
    read) read_night "$LEDGER" "$ART" "$OUT" ;;
    streak) streak "$LEDGER" "$TODAY" ;;
    selftest) selftest ;;
    mutants) mutants ;;
esac
