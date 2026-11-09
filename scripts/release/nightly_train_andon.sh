#!/usr/bin/env bash
# nightly_train_andon.sh — a red or not_measured Nightly Train opens or updates ONE ticket with ONE owner (#5081, B1 P2)
#
# THE ROW. C324 item 5: "A red opens or updates one ticket with one owner, then stays silent." Before #5081 the train
#   wrote a step summary and an artifact and told nobody: a NOT RELEASABLE night was silent (A3 report 2026-10-10).
#   This runs in its own job of nightly-train.yml after the train job, whatever the train's result, and is the only job
#   there that holds issues: write.
#
# THE DECISION. The night is green only when BOTH the train job succeeded AND its line starts "RELEASABLE ". Anything
#   else writes: a NOT RELEASABLE line (red or not_measured lanes: not_measured is never a pass), the train's trap line,
#   no line at all (the train died, or the artifact is missing), a failed job with a RELEASABLE line, and a successful
#   job with a NOT RELEASABLE line (so a train run without --exit-verdict cannot make a red night silent).
#
# THE TICKET. One open issue labelled LABEL. If one is open the night is a comment on it; otherwise one is created.
#   A failed label or list read exits 1 and never falls through to create: a read we could not make is not "none open".
#   A failed write exits 1, so the andon job goes red rather than silent.
#
# THE OWNER. Fixed table OWNERS below, a constant (never an option or an environment value). The ticket's owner is the
#   owner of reds.tsv's first data row (the lane that lost the most hours); an unmapped lane, a reds.tsv with no data
#   rows, or no reds.tsv falls back to DEFAULT_OWNER, which owns all of B1 (A3 report §owners). Never empty.
#
# DRILLS. Only a scheduled night writes. Any other --event (workflow_dispatch) prints what it would write and calls
#   nothing: a drill never opens or comments on the real ticket.
#
# INPUT. --nt DIR is the train's --out directory (the downloaded artifact): DIR/<UTC day>/line and DIR/<day>/reds.tsv.
#   With more than one day directory the newest is read.
#
# USAGE  nightly_train_andon.sh --nt DIR --result RESULT --event EVENT --run-url URL [--gh CMD]
#        nightly_train_andon.sh --self-test     the case table, against a fake gh
#        nightly_train_andon.sh --mutants       every planted mutant must break a row of the table
# EXIT   0 green, or the ticket was written (or a drill printed it) · 1 a read or write failed, or the table broke · 3 caller error
set -euo pipefail

LABEL="nightly-train-red"
TITLE="nightly train: NOT RELEASABLE (andon, #5081)"
DEFAULT_OWNER="aprender-build-waiter"
OWNERS='dogfood	aprender-78
readiness	aprender-78
assets	aprender-wprodmodels'
MAX_ROWS=20

owner_of() { # LANE -> its owner, or empty
    printf '%s\n' "$OWNERS" | awk -F '\t' -v L="$1" '$1 == L { print $2; exit }'
}

# newest_day DIR -> DIR/<newest day>, or empty. Day-named dirs only: the artifact also carries cache/, which sorts
# after every day, so a bare */ read cache/ and reported a printed line as missing (run 38133969253, #5081).
newest_day() {
    local d last=""
    for d in "$1"/*/; do case "${d%/}" in */[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) last="${d%/}" ;; esac; done
    printf '%s' "$last"
}

andon() {
    local nt="" result="" event="" url="" gh="gh"
    while [ $# -gt 0 ]; do
        case "$1" in
            --nt) nt="${2:-}"; shift 2 ;;
            --result) result="${2:-}"; shift 2 ;;
            --event) event="${2:-}"; shift 2 ;;
            --run-url) url="${2:-}"; shift 2 ;;
            --gh) gh="${2:-}"; shift 2 ;;
            *) echo "andon: unknown argument $1" >&2; return 3 ;;
        esac
    done
    if [ -z "$nt" ] || [ -z "$result" ] || [ -z "$event" ] || [ -z "$url" ]; then
        echo "andon: --nt, --result, --event and --run-url are required" >&2; return 3
    fi
    local day line="" reds=""
    day="$(newest_day "$nt")"
    if [ -n "$day" ] && [ -s "$day/line" ]; then line="$(head -n 1 "$day/line")"; fi
    if [ -n "$day" ] && [ -f "$day/reds.tsv" ]; then reds="$day/reds.tsv"; fi

    local green=""
    if [ "$result" = success ]; then case "$line" in "RELEASABLE "*) green=1 ;; esac; fi
    if [ -n "$green" ]; then echo "andon: green ($line), no ticket"; return 0; fi

    local owner="" first=""
    if [ -n "$reds" ]; then first="$(awk -F '\t' 'NR == 2 { print $2; exit }' "$reds")"; fi
    if [ -n "$first" ]; then owner="$(owner_of "$first")"; fi
    owner="${owner:-$DEFAULT_OWNER}"

    local body
    body="$(
        echo "**Nightly Train is not green.** Run: $url"
        echo
        echo "- train job result: \`$result\`"
        echo "- line: \`${line:-no verdict line (the train died before printing, or the artifact is missing)}\`"
        echo "- owner: $owner"
        echo
        if [ -n "$reds" ] && [ "$(wc -l < "$reds")" -gt 1 ]; then
            echo '| lane | state | hours lost | run | reason | owner |'
            echo '|---|---|---|---|---|---|'
            awk -F '\t' -v M="$MAX_ROWS" 'NR > 1 && NR <= M + 1 { print $2 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$reds" |
                while IFS="$(printf '\t')" read -r lane state hours run reason; do
                    o="$(owner_of "$lane")"
                    printf '| %s | %s | %s | %s | %s | %s |\n' "$lane" "$state" "$hours" "$run" "$reason" "${o:-$DEFAULT_OWNER}"
                done
        else
            echo "no reds.tsv rows: the owner is $DEFAULT_OWNER"
        fi
        echo
        echo "Filed by scripts/release/nightly_train_andon.sh (C324 item 5: one ticket, one owner)."
    )"

    if [ "$event" != schedule ]; then
        echo "andon: drill ($event): no ticket written. Would write, owner $owner:"
        printf '%s\n' "$body"
        return 0
    fi

    "$gh" label create "$LABEL" --description "Nightly Train red or not_measured (#5081)" --color B60205 --force > /dev/null || {
        echo "andon: could not ensure label $LABEL" >&2; return 1; }
    local num
    num="$("$gh" issue list --label "$LABEL" --state open --limit 1 --json number --jq '.[0].number // empty')" || {
        echo "andon: could not list open $LABEL issues; writing nothing rather than a duplicate" >&2; return 1; }
    if [ -n "$num" ]; then
        "$gh" issue comment "$num" --body "$body" > /dev/null || { echo "andon: comment on #$num failed" >&2; return 1; }
        echo "andon: commented on #$num, owner $owner"
    else
        "$gh" issue create --title "$TITLE" --label "$LABEL" --body "$body" > /dev/null || { echo "andon: create failed" >&2; return 1; }
        echo "andon: created a $LABEL ticket, owner $owner"
    fi
}

# ---------------------------------------------------------------------------------------------------------------------
# --self-test: the case table. A fake gh logs every call (one line: its argv) and fails the subcommand FAKE_FAIL names.

fake_gh() { # DIR -> DIR/gh
    cat > "$1/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_LOG"
case "$1 $2" in
    "$FAKE_FAIL") exit 1 ;;
    "issue list") printf '%s\n' "${FAKE_OPEN:-}" ;;
esac
exit 0
EOF
    chmod +x "$1/gh"
}

# plant DIR DAY LINE ROWS... -> DIR/DAY/{line,reds.tsv}; LINE '-' writes no line file, ROWS 'none' no reds.tsv
plant() {
    local dir="$1" day="$2" line="$3"; shift 3
    mkdir -p "$dir/$day"
    [ "$line" = - ] || printf '%s\n' "$line" > "$dir/$day/line"
    [ "${1:-}" = none ] && return 0
    printf 'rank\tlane\tkind\tstate\thours_lost\trun_id\treason\n' > "$dir/$day/reds.tsv"
    local r n=1
    for r in "$@"; do printf '%s\t%s\n' "$n" "$r" >> "$dir/$day/reds.tsv"; n=$((n + 1)); done
}

self_test() {
    local self="$1" tmp fails=0
    tmp="$(mktemp -d)" || return 1
    fake_gh "$tmp"
    local NR="NOT RELEASABLE: dogfood, not_measured (+2 more) [C=0123456789 pin=a.b.c]"
    local OK="RELEASABLE H=0123456789abcdef [C=0123456789 pin=a.b.c]"
    local RD="dogfood	verdict	not_measured	30.00	-	no producer"
    local RM="models	verdict	red	6.00	123	failure"
    local RX="ci-main	verdict	red	2.00	456	failure"

    # check NAME WANT_RC RESULT EVENT FAKE_OPEN FAKE_FAIL GREP... (each GREP: '+re' must match the log+out, '-re' must not)
    check() {
        local name="$1" want="$2" result="$3" event="$4" open="$5" fail="$6"; shift 6
        local rc=0
        : > "$tmp/log"
        FAKE_LOG="$tmp/log" FAKE_OPEN="$open" FAKE_FAIL="$fail" \
            bash "$self" --nt "$tmp/nt" --result "$result" --event "$event" --run-url "https://x/runs/9" --gh "$tmp/gh" \
            > "$tmp/out" 2>&1 || rc=$?
        local ok=1 g
        [ "$rc" = "$want" ] || ok=""
        for g in "$@"; do
            case "$g" in
                +*) grep -qE -- "${g#+}" "$tmp/log" "$tmp/out" || ok="" ;;
                -*) grep -qE -- "${g#-}" "$tmp/log" "$tmp/out" && ok="" ;;
            esac
        done
        if [ -n "$ok" ]; then echo "ok   $name"; else
            echo "FAIL $name (rc $rc, want $want)"; sed 's/^/     | /' "$tmp/log" "$tmp/out" | head -n 12; fails=$((fails + 1)); fi
        rm -rf -- "${tmp:?}/nt"
    }

    plant "$tmp/nt" 2026-10-10 "$NR" "$RD" "$RM"
    check a_red_no_open_creates 0 failure schedule "" "" '+^issue create .*--label nightly-train-red' '+owner: aprender-78' '+runs/9' '-^issue comment'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD" "$RM"
    check b_red_open_comments 0 failure schedule 77 "" '+^issue comment 77 ' '-^issue create'
    plant "$tmp/nt" 2026-10-10 "$OK" none
    check c_green_writes_nothing 0 success schedule "" "" '+andon: green' '-^(label|issue)'
    mkdir -p "$tmp/nt"
    check d_no_artifact_creates 0 failure schedule "" "" '+^issue create ' '+no verdict line' "+owner: $DEFAULT_OWNER"
    plant "$tmp/nt" 2026-10-10 "NOT RELEASABLE: dogfood, not_measured" "$RD"
    check e_not_measured_only_writes 0 failure schedule "" "" '+^issue create '
    plant "$tmp/nt" 2026-10-10 "$NR" "$RX" "$RD"
    check f_unmapped_lane_default_owner 0 failure schedule "" "" '+^issue create ' "+owner: $DEFAULT_OWNER"
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check g_create_fails_exits_1 1 failure schedule "" "issue create" '+create failed'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check g2_comment_fails_exits_1 1 failure schedule 77 "issue comment" '+comment on #77 failed'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check h_list_fails_never_creates 1 failure schedule "" "issue list" '-^issue create' '+could not list'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check h2_label_fails_never_creates 1 failure schedule "" "label create" '-^issue (create|list)' '+could not ensure label'
    plant "$tmp/nt" 2026-10-10 "$NR"
    check i_header_only_reds_default_owner 0 failure schedule "" "" '+^issue create ' "+owner: $DEFAULT_OWNER"
    plant "$tmp/nt" 2026-10-09 "$OK" none
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check j_newest_day_dir_is_read 0 failure schedule "" "" '+^issue create ' '+owner: aprender-78'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    mkdir -p "$tmp/nt/cache"
    check j2_cache_dir_is_not_a_day 0 failure schedule "" "" '+^issue create ' '+line: `NOT RELEASABLE' '+owner: aprender-78' '-no verdict line'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check k_success_job_red_line_writes 0 success schedule "" "" '+^issue create '
    plant "$tmp/nt" 2026-10-10 "$OK" none
    check l_failed_job_green_line_writes 0 failure schedule "" "" '+^issue create '
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check m_drill_writes_nothing 0 failure workflow_dispatch "" "" '+drill' '+owner: aprender-78' '-^(label|issue)'
    plant "$tmp/nt" 2026-10-10 "$NR" "$RD"
    check n_missing_arg_exits_3 3 failure "" "" "" '-^(label|issue)'

    rm -rf -- "${tmp:?}"
    if [ "$fails" -eq 0 ]; then echo "andon self-test: all rows held"; return 0; fi
    echo "andon self-test: $fails row(s) broke"; return 1
}

# ---------------------------------------------------------------------------------------------------------------------
# --mutants: each planted mutant (one sed on a copy of this file) must break at least one row of the table.
MUTANTS='M1_drop_create	s/^        "\$gh" issue create /        true /
M2_red_line_read_as_green	s/case "\$line" in "RELEASABLE "\*)/case "$line" in *RELEASABLE*)/
M3_always_create	s/^    if \[ -n "\$num" \]; then$/    if false; then/
M4_empty_owner	s/^    owner="\${owner:-\$DEFAULT_OWNER}"$/    owner="${owner:-}"/
M5_list_failure_read_as_none	s/^\(    num="\$("\$gh" issue list .*)"\) || {$/\1 || : || {/
M6_job_result_only	s/^    if \[ "\$result" = success \]; then case .*$/    if [ "$result" = success ]; then green=1; fi/
M7_line_only	s/^    if \[ "\$result" = success \]; then case/    if true; then case/
M8_drills_write	s/^    if \[ "\$event" != schedule \]; then$/    if false; then/
M9_label_failure_read_as_success	s/--force > \/dev\/null || {/--force > \/dev\/null || : || {/
M10_any_dir_is_a_day	s/ in \*\/\[0-9\]\[0-9\]\[0-9\]\[0-9\]-\[0-9\]\[0-9\]-\[0-9\]\[0-9\]) last=/ in *) last=/'

mutants() {
    local self="$1" tmp survived=0 name expr
    tmp="$(mktemp -d)" || return 1
    while IFS="$(printf '\t')" read -r name expr; do
        sed "$expr" "$self" > "$tmp/m.sh"
        if cmp -s "$self" "$tmp/m.sh"; then echo "SURVIVED $name (the sed changed nothing: the mutant is stale)"; survived=$((survived + 1)); continue; fi
        if bash "$tmp/m.sh" --self-test > "$tmp/out" 2>&1; then
            echo "SURVIVED $name"; survived=$((survived + 1))
        else
            echo "killed   $name ($(grep -c '^FAIL' "$tmp/out") row(s))"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    if [ "$survived" -eq 0 ]; then echo "andon mutants: all killed"; return 0; fi
    echo "andon mutants: $survived survived"; return 1
}

case "${1:-}" in
    --self-test) self_test "$0" ;;
    --mutants) mutants "$0" ;;
    *) andon "$@" ;;
esac
