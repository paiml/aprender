#!/usr/bin/env bash
# lab_ticket.sh — a red LAB night opens or updates ONE ticket with ONE owner, then stays silent (#5182)
#
# THE ROW. The GATE/LAB rule: "A red opens or updates one ticket with one owner, then stays silent." A LAB check runs off
#   the PR, merge-queue and release paths, so nothing else tells anyone it went red. The caller runs this only on a
#   red night (the workflow step's `if:`); a green night never calls it, and that is the silence.
#
# THE TICKET. One open issue carrying LABEL. If one is open the night is a comment on it; otherwise it is created with
#   TITLE and LABEL. A failed label or list read exits 1 and never falls through to create: a read we could not make
#   is not "none open". A failed write exits 1, so the step goes red rather than silent. The same shape as
#   scripts/release/nightly_train_andon.sh, whose owner table and input layout are the train's own.
#
# THE OWNER. --owner, required and never empty; it is written into the body as "owner: <owner>".
#
# DRILLS. Only a scheduled night writes. Any other --event prints what it would write and calls nothing, so a manual
#   dispatch never opens or comments on the real ticket.
#
# USAGE  lab_ticket.sh --label L --title T --owner O --log FILE --event EVENT --run-url URL [--gh CMD]
#        lab_ticket.sh --self-test     the case table, against a fake gh (a planted red night included)
# EXIT   0 the ticket was written (or a drill printed it) · 1 a read or write failed, or the table broke · 3 caller error
set -euo pipefail

TAIL_LINES=80

die3() { echo "lab_ticket: $*" >&2; exit 3; }

night_body() { # night_body <owner> <run url> <log>
    printf 'owner: %s\n\nRun: %s\n\nLast %s lines of the red steps:\n\n```\n' "$1" "$2" "$TAIL_LINES"
    if [ -s "$3" ]; then tail -n "$TAIL_LINES" "$3"; else echo '(no log: the run failed before the steps wrote one; not measured)'; fi
    printf '```\n'
}

write_ticket() { # write_ticket <gh> <label> <title> <body> <event>
    local gh="$1" label="$2" title="$3" body="$4" event="$5" labels num
    if [ "$event" != schedule ]; then
        printf 'DRILL (event %s): would write to the open "%s" ticket, or create "%s":\n%s\n' "$event" "$label" "$title" "$body"
        return 0
    fi
    labels="$("$gh" label list --limit 500 --json name --jq '.[].name')" || { echo "lab_ticket: label read failed" >&2; return 1; }
    if ! grep -qxF -- "$label" <<< "$labels"; then
        "$gh" label create "$label" --description "LAB night red: one ticket, one owner (GATE/LAB rule)" > /dev/null \
            || { echo "lab_ticket: label create failed" >&2; return 1; }
    fi
    num="$("$gh" issue list --label "$label" --state open --limit 1 --json number --jq '.[0].number // empty')" \
        || { echo "lab_ticket: issue list failed" >&2; return 1; }
    if [ -n "$num" ]; then
        "$gh" issue comment "$num" --body "$body" > /dev/null || { echo "lab_ticket: comment failed" >&2; return 1; }
        echo "lab_ticket: commented on #$num"
    else
        "$gh" issue create --title "$title" --label "$label" --body "$body" > /dev/null \
            || { echo "lab_ticket: create failed" >&2; return 1; }
        echo "lab_ticket: created \"$title\""
    fi
}

self_test() {
    local fails=0 rows=0
    T="$(mktemp -d)"
    trap 'rm -rf "${T:?}"' EXIT
    # The fake gh: logs every call to $T/calls; FAKE_OPEN is the open issue number (empty = none);
    # FAKE_LABELS the existing labels; FAKE_FAIL a subcommand ("label list", "issue list", ...) that fails.
    cat > "$T/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_CALLS"
case "$1 $2" in
  "${FAKE_FAIL:-none}") exit 1 ;;
  "label list") printf '%s\n' ${FAKE_LABELS:-} ;;
  "issue list") [ -n "${FAKE_OPEN:-}" ] && echo "$FAKE_OPEN" ;;
esac
exit 0
EOF
    chmod +x "$T/gh"
    # THE PLANTED RED: a real failing ratchet-shaped step, its log captured as the workflow step does.
    ( echo "FAIL  scripts/planted_baseline.txt grew 3 -> 4 (shrink-only)"; exit 1 ) > "$T/red.log" 2>&1 || true
    : > "$T/empty.log"

    check() { # check <name> <want rc> <event> <open> <labels> <fail> <log> <patterns: +must / -must-not, on calls+stdout>
        local name="$1" want="$2" ev="$3" open="$4" labs="$5" fail="$6" log="$7" got out p; shift 7
        rows=$((rows + 1)); : > "$T/calls"
        out="$(FAKE_CALLS="$T/calls" FAKE_OPEN="$open" FAKE_LABELS="$labs" FAKE_FAIL="$fail" \
            bash "$0" --label lab-x --title "LAB: x red" --owner aprender-cd --log "$log" --event "$ev" \
            --run-url "https://example.invalid/runs/9" --gh "$T/gh" 2>&1)" && got=0 || got=$?
        local all; all="$(cat "$T/calls")"$'\n'"$out"
        local ok=1
        [ "$got" = "$want" ] || ok=0
        for p in "$@"; do
            case "$p" in
                +*) grep -qE -- "${p#+}" <<< "$all" || ok=0 ;;
                -*) grep -qE -- "${p#-}" <<< "$all" && ok=0 ;;
            esac
        done
        if [ "$ok" = 1 ]; then echo "ok    $name"; else echo "BROKE $name (rc $got, want $want)"; echo "$all" | sed 's/^/      /'; fails=$((fails + 1)); fi
    }
    check planted_red_none_open_creates 0 schedule "" lab-x "" "$T/red.log" \
        '+^issue create --title LAB: x red --label lab-x' '+owner: aprender-cd' '+grew 3 -> 4' '+runs/9' '-^issue comment'
    check planted_red_open_comments     0 schedule 42 lab-x "" "$T/red.log" \
        '+^issue comment 42 ' '+grew 3 -> 4' '-^issue create'
    check missing_label_is_created      0 schedule "" "" "" "$T/red.log" '+^label create lab-x' '+^issue create'
    check no_log_still_writes_unmeasured 0 schedule "" lab-x "" "$T/empty.log" '+not measured' '+^issue create'
    check drill_writes_nothing          0 workflow_dispatch "" lab-x "" "$T/red.log" '+DRILL' '-^(issue|label) '
    check failed_list_never_creates     1 schedule "" lab-x "issue list" "$T/red.log" '+issue list failed' '-^issue create'
    check failed_label_read_never_creates 1 schedule "" lab-x "label list" "$T/red.log" '+label read failed' '-^issue create'
    check failed_create_is_red          1 schedule "" lab-x "issue create" "$T/red.log" '+create failed'
    check failed_comment_is_red         1 schedule 42 lab-x "issue comment" "$T/red.log" '+comment failed'
    # caller errors
    rows=$((rows + 1))
    if bash "$0" --label lab-x --title t --owner "" --log "$T/red.log" --event schedule --run-url u --gh "$T/gh" > /dev/null 2>&1; then
        echo "BROKE empty_owner_refused"; fails=$((fails + 1))
    else echo "ok    empty_owner_refused"; fi
    echo "lab_ticket self-test: $((rows - fails)) ok, $fails broke"
    [ "$fails" = 0 ]
}

main() {
    local label="" title="" owner="" log="" event="" url="" gh=gh
    [ "${1:-}" = --self-test ] && { self_test; return; }
    while [ $# -gt 0 ]; do
        case "$1" in
            --label) label="${2:-}"; shift 2 ;;
            --title) title="${2:-}"; shift 2 ;;
            --owner) owner="${2:-}"; shift 2 ;;
            --log) log="${2:-}"; shift 2 ;;
            --event) event="${2:-}"; shift 2 ;;
            --run-url) url="${2:-}"; shift 2 ;;
            --gh) gh="${2:-}"; shift 2 ;;
            *) die3 "unknown argument $1" ;;
        esac
    done
    [ -n "$label" ] && [ -n "$title" ] && [ -n "$owner" ] && [ -n "$event" ] && [ -n "$url" ] && [ -n "$log" ] \
        || die3 "--label, --title, --owner, --log, --event and --run-url are all required and non-empty"
    write_ticket "$gh" "$label" "$title" "$(night_body "$owner" "$url" "$log")" "$event"
}

main "$@"
