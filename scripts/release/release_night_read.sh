#!/usr/bin/env bash
# release_night_read.sh -- T18 #4701, reader side: may release day skip its ladders for head H because the
# nightly train already measured them ON H? Reads the train's ledger only; runs nothing, calls nothing.
#
#   bash scripts/release/release_night_read.sh --ledger DIR --head SHA
#
# WHAT IT READS. --ledger DIR is nightly_train.sh's --out: DIR/<UTC day>/bundle.tsv, one per night. Columns are
#   found by the header's names (lane, state, run_head, attempt), never by position. A night is H's when its
#   "# C" line is H. The "# read" line must be "ok" for a night of H to count as read.
#
# THE LANES are the release-day steps this read lets release day skip: autopilot.sh's deep (doctests,
#   --no-default-features, examples, all-bins build + smoke), dogfood, and the model matrix. The list is a
#   constant in this file, never an option or an environment value: a lane set a caller can shrink is theater.
#
# THE READ, per lane, over every night of H. Columns of one row: state, run_head, attempt.
#   RED           any night of H holds the lane red. A later green on the same commit never outvotes it.
#   GO            the newest read night of H holds it green, from a run on H (run_head == H), at attempt 1.
#   NOT_MEASURED  everything else, and it REFUSES: no night of H at all; H's nights unreadable (a cut-short
#                 bundle whose "# C" is not a 40-hex sha could be H's; a dir that is not a UTC day; a missing
#                 column or a state the train never writes, on any night of H); the lane row missing,
#                 duplicated, not_measured, or any other state; a green from a run on another commit; a green
#                 at attempt 2 or later. not_measured is never a pass, and release day runs no ladder to make
#                 up for it here: the caller falls back to running the steps itself (RQ-2 ruling B).
#
# NOT WIRED. No release script calls this file. Using it in autopilot.sh in place of the steps changes what
#   release day accepts: that is the switch, and it waits for 3 green nights and sign-off (#4701).
#
# stdout, last line: GO H=<sha> night=<day> | RED <lane> night=<day> run=<id> | NOT_MEASURED <lane|night> <why>
# EXIT 0 GO · 1 RED · 2 NOT_MEASURED (refuse) · 3 caller error
#
#   bash scripts/release/release_night_read_table.sh   the case table + mutants
set -uo pipefail

LANES="deep-doctests deep-nodefault deep-examples deep-bins-build deep-bins-smoke dogfood models"

LEDGER="" H=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ledger) LEDGER="${2:-}"; shift 2 ;;
        --head) H="${2:-}"; shift 2 ;;
        *) echo "usage: $0 --ledger DIR --head SHA" >&2; exit 3 ;;
    esac
done
[ -n "$LEDGER" ] && [[ $H =~ ^[0-9a-f]{40}$ ]] || { echo "usage: --ledger DIR and a 40-hex --head SHA are required" >&2; exit 3; }

nm() { echo "NOT_MEASURED $*"; exit 2; }

# meta FILE KEY -> the value of the "# KEY<TAB>value" line, or empty
meta() { awk -F'\t' -v k="# $2" '$1 == k { print $2; exit }' "$1" 2> /dev/null; }

# H's nights, newest first. Each dir under DIR is a UTC day (YYYY-MM-DD), or the train's fetch cache. Any
# other dir refuses: the reader cannot tell whose it is. A day with no bundle.tsv (a failed train night writes
# only its "line") holds no row, so it cannot hide a red and is passed over. A bundle whose "# C" is not a
# 40-hex sha (missing, empty, CRLF, trailing space) may be H's: it refuses.
export LC_ALL=C   # the glob sorts the day dirs; C collation makes ascending explicit
[ -d "$LEDGER" ] || nm "night: no ledger at $LEDGER"
nights=""
for d in "$LEDGER"/*/; do
    [ -d "$d" ] || continue
    day=$(basename "$d")
    [ "$day" = cache ] && continue
    [[ $day =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || nm "night: '$day' is not a UTC day dir"
    b="$LEDGER/$day/bundle.tsv"
    [ -f "$b" ] || continue
    c=$(meta "$b" C)
    [[ $c =~ ^[0-9a-f]{40}$ ]] || nm "night: $day '# C' is '$c', not a 40-hex sha; it may be H's"
    [ "$c" = "$H" ] && nights="$day $nights"
done
[ -n "$nights" ] || nm "night: no night measured ${H:0:10}"
newest=${nights%% *}

# row FILE LANE -> "<count>|<state>|<run_head>|<attempt>|<red run id or empty>|<bad or empty>", columns by the
# header's names. "|", not a tab: read merges runs of a whitespace IFS, so an empty run_head would shift
# attempt into it. The red field is set if ANY row of the lane is red, so a duplicate row cannot hide one.
# bad is set when a required column is missing, or a row's state is not one the train writes: either could
# hide a red, so it refuses.
row() {
    awk -F'\t' -v l="$2" '
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i
                  if (!("lane" in col) || !("state" in col) || !("run_head" in col) || !("attempt" in col) || !("run_id" in col)) bad = "missing column"
                  next }
        /^#/ || bad != "" { next }
        $col["lane"] == l { n++; s = $col["state"]; h = $col["run_head"]; a = $col["attempt"]
                          if (s != "green" && s != "red" && s != "not_measured" && bad == "") bad = "state " s
                          if (s == "red" && red == "") red = ($col["run_id"] == "" ? "none" : $col["run_id"]) }
        END { printf "%d|%s|%s|%s|%s|%s\n", n, s, h, a, red, bad }' "$1"
}

# RED first, over every night of H: a red is never outvoted by a later green on the same commit.
for d in $nights; do
    for l in $LANES; do
        o=$(row "$LEDGER/$d/bundle.tsv" "$l") || nm "night: $d bundle unreadable"
        IFS="|" read -r _ _ _ _ r x <<< "$o"
        [ -n "$x" ] && nm "$l: night $d has a $x; a red could be hiding"
        [ -n "$r" ] && { echo "RED $l night=$d run=$r"; exit 1; }
    done
done

b="$LEDGER/$newest/bundle.tsv"
[ "$(meta "$b" read)" = ok ] || nm "night: $newest read '$(meta "$b" read)', not ok"
for l in $LANES; do
    IFS="|" read -r n s h a _ _ <<< "$(row "$b" "$l")"
    [ "$n" = 1 ] || nm "$l: $n rows in night $newest"
    [ "$s" = green ] || nm "$l: state '$s' in night $newest"
    [ "$h" = "$H" ] || nm "$l: green from a run on '${h:0:10}', not ${H:0:10}"
    [ "$a" = 1 ] || nm "$l: green at attempt '$a', not 1"
done
echo "GO H=$H night=$newest"
exit 0
