# lib_cut_receipt.sh -- the rc cut's receipt ledger: one row BEFORE each step acts, one after (#4327)
#
# Sourced by rc_tag_main.sh, rc_fleet_stage.sh and rc_cut_resumable.sh. A cut is several writes that
# other systems see (a tag, a draft, a dispatch, a binary on each fleet host, the publish). A session
# that died between two of them left no record of how far it got: the next one re-ran the tagger
# (refused: the tag exists) or finished by hand.
#
# Each step writes `intent` before it acts and `done` once it is confirmed. A fresh session reads
# the last row for (step, key):
#   done    -- confirmed; never repeated (a deploy is re-VERIFIED, never re-installed)
#   intent  -- it may or may not have landed: the caller asks the world, not the ledger
#   (none)  -- it never started
#
# Rows: <utc>\t<tag>\t<step>\t<key>\t<intent|done|fail>\t<detail>
# File: $RC_CUT_LEDGER, else ${XDG_STATE_HOME:-$HOME/.local/state}/rc-cut/<tag>.tsv -- on the box that
# runs the cut (lambda), outside any worktree, so a new session in a new worktree finds it.
#
# A SOURCED library: no `set` here, it would change the caller's shell. Failures are return codes.

receipt_file() {  # receipt_file <tag>
    printf '%s\n' "${RC_CUT_LEDGER:-${XDG_STATE_HOME:-$HOME/.local/state}/rc-cut/$1.tsv}"
}

receipt_write() {  # receipt_write <tag> <step> <key> <intent|done|fail> [detail]
    local f
    f=$(receipt_file "$1") || return 2
    mkdir -p -- "$(dirname -- "$f")" || return 2
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" \
        "$(printf '%s' "${5:-}" | tr '\t\n' '  ')" >> "$f" || return 2
}

receipt_field() {  # receipt_field <tag> <step> <key> <awk column> -> that column of the LAST matching row
    local f
    f=$(receipt_file "$1") || return 2
    [ -f "$f" ] || return 0
    awk -F'\t' -v t="$1" -v s="$2" -v k="$3" -v c="$4" '$2==t && $3==s && $4==k {v=$c} END {print v}' "$f"
}
receipt_state() { receipt_field "$1" "$2" "$3" 5; }    # intent | done | fail | empty
receipt_detail() { receipt_field "$1" "$2" "$3" 6; }
receipt_time() { receipt_field "$1" "$2" "$3" 1; }     # utc of the last row
