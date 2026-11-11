#!/usr/bin/env bash
# spec_expiry_warn.sh -- D6 advance warning (aprender#4652): a §12 row that is NOT discharged and whose derived
# expiry is 0..7 days away prints one WARN line, and with --issues opens or updates ONE issue per row.
#
# Why this is a separate script: the D6 andon in scripts/spec_conformance.sh turns the PR path RED at T+1 and stays
# exactly as strict. A warning on the PR path would be an advisory third state (C324 item 6), so this script is LAB:
# it is wired into no PR, merge-queue or release job. It reads evidence/parity/derived_expiries.json, which the
# conformance scan regenerates and the PR gate keeps in sync with §12, so no second parser of §12 exists.
#
# Usage:
#   bash scripts/spec_expiry_warn.sh                  # WARN lines on stdout, exit 0
#   bash scripts/spec_expiry_warn.sh --issues         # also open/update one issue per warned row (gh, REST writes)
#   bash scripts/spec_expiry_warn.sh --self-test      # fixture case table + planted-defect fixtures
# Knobs: SPEC_EXPIRY_JSON (default evidence/parity/derived_expiries.json), SPEC_CONFORMANCE_TODAY (YYYY-MM-DD,
# default today UTC), SPEC_EXPIRY_WARN_DAYS (default 7), SPEC_EXPIRY_REPO (default paiml/aprender).
#
# Output, one line per warned row (the cop report reads these):
#   WARN D6 row=<key> expires=<YYYY-MM-DD> days_left=<n> source=<spec>
# and a final count line, which says how many rows were READ so that "0 warnings" over 0 rows is never a verdict:
#   spec_expiry_warn: rows=<n> undischarged_dated=<n> warned=<n>
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# warn_rows <json> <today> <days>: prints the WARN lines and the count line; exit 2 when the file cannot be read.
warn_rows() {
    local json="$1" today="$2" days="$3" t
    [ -r "$json" ] || { echo "spec_expiry_warn: cannot read $json" >&2; return 2; }
    t="$(date -u -d "$today" +%s 2>/dev/null)" || { echo "spec_expiry_warn: bad today '$today'" >&2; return 2; }
    jq -r --argjson t "$t" --argjson d "$days" '
        def epoch: strptime("%Y-%m-%d") | mktime;
        (.source // "?") as $src
        | (.rows // {}) as $rows
        | [ $rows | to_entries[]
            | select(.value.discharged != true and (.value.expires | type) == "string")
            | . + {left: (((.value.expires | epoch) - $t) / 86400 | floor)} ] as $dated
        | ($dated | map(select(.left >= 0 and .left <= $d))) as $warn
        | ($warn | sort_by(.left, .key)[]
            | "WARN D6 row=\(.key) expires=\(.value.expires) days_left=\(.left) source=\($src)"),
          "spec_expiry_warn: rows=\($rows | length) undischarged_dated=\($dated | length) warned=\($warn | length)"
    ' "$json" || { echo "spec_expiry_warn: $json is not the derived_expiries shape" >&2; return 2; }
}

# open_or_update <repo> <warn-line>: one open issue per row, found by its exact title; a comment when it exists.
open_or_update() {
    local repo="$1" line="$2" key title body n
    key="$(sed -n 's/.* row=\([^ ]*\) .*/\1/p' <<<"$line")"
    title="D6: §12 row $key expiry warning"
    body="$line

Raised by scripts/spec_expiry_warn.sh (aprender#4652). The D6 andon in scripts/spec_conformance.sh turns every PR and main RED the day after this expiry unless the row is discharged or re-dated."
    n="$(gh issue list -R "$repo" --state open --search "\"$title\" in:title" --json number,title \
        --jq "map(select(.title == \"$title\")) | .[0].number // empty")" || return 1
    if [ -n "$n" ]; then
        gh api -X POST "repos/$repo/issues/$n/comments" -f body="$line" --jq .id >/dev/null || return 1
        echo "updated #$n row=$key"
    else
        n="$(gh api -X POST "repos/$repo/issues" -f title="$title" -f body="$body" --jq .number)" || return 1
        [ -n "$n" ] || { echo "spec_expiry_warn: create returned no number for row=$key" >&2; return 1; }
        echo "opened #$n row=$key"
    fi
}

main() {
    local json="${SPEC_EXPIRY_JSON:-$ROOT/evidence/parity/derived_expiries.json}"
    local today="${SPEC_CONFORMANCE_TODAY:-$(date -u +%F)}" days="${SPEC_EXPIRY_WARN_DAYS:-7}" out rc line fail=0
    out="$(warn_rows "$json" "$today" "$days")"; rc=$?
    printf '%s\n' "$out"
    [ "$rc" = 0 ] || return "$rc"
    if [ "${1:-}" = "--issues" ]; then
        while IFS= read -r line; do
            open_or_update "${SPEC_EXPIRY_REPO:-paiml/aprender}" "$line" || fail=1
        done < <(grep '^WARN D6 ' <<<"$out")
    fi
    return "$fail"
}

# ---------------------------------------------------------------------------------------------------------------
self_test() {
    local d fail=0 out self="$0"
    d="$(mktemp -d)"
    # One undischarged row per case, expiry fixed at 2026-10-08; "today" moves around it.
    fx() { # fx <file> <discharged> <expires|null>
        printf '{"source":"FIXTURE","rows":{"r1":{"blocked_by":[],"derived_from":[],"discharged":%s,"expires":%s,"root":true}}}\n' \
            "$2" "$3" > "$1"
    }
    row() { # row <label> <want: warn|none> <today> <json> [script]
        local got
        got="$(SPEC_EXPIRY_JSON="$4" SPEC_CONFORMANCE_TODAY="$3" bash "${5:-$self}" 2>&1)"
        if [ "$2" = warn ] && grep -q '^WARN D6 row=r1 ' <<<"$got" || { [ "$2" = none ] && ! grep -q '^WARN' <<<"$got"; }; then
            printf '  ok    %s\n' "$1"; return 0
        fi
        printf '  FAIL  %s (want %s): %s\n' "$1" "$2" "$(tr '\n' ' ' <<<"$got")"; return 1
    }
    cases() { # cases [script]: the case table; prints one line per row, returns non-zero on any FAIL
        local s="${1:-$self}" f=0
        fx "$d/open.json" false '"2026-10-08"'
        fx "$d/done.json" true '"2026-10-08"'
        fx "$d/nodate.json" false null
        row "T-7 undischarged row warns"         warn 2026-10-01 "$d/open.json" "$s" || f=1
        row "T-0 (expiry day) warns"             warn 2026-10-08 "$d/open.json" "$s" || f=1
        row "T-8 does not warn"                  none 2026-09-30 "$d/open.json" "$s" || f=1
        row "T+1 does not warn (D6 andon owns it)" none 2026-10-09 "$d/open.json" "$s" || f=1
        row "discharged row never warns"         none 2026-10-05 "$d/done.json" "$s" || f=1
        row "undated row never warns"            none 2026-10-05 "$d/nodate.json" "$s" || f=1
        return "$f"
    }
    echo "# case table"
    cases || fail=1

    echo "# count line reads the file (0 warnings over 0 rows is not a verdict)"
    out="$(SPEC_EXPIRY_JSON="$d/open.json" SPEC_CONFORMANCE_TODAY=2026-10-01 bash "$self")"
    grep -qx 'spec_expiry_warn: rows=1 undischarged_dated=1 warned=1' <<<"$out" && echo "  ok    count line" \
        || { echo "  FAIL  count line: $out"; fail=1; }
    SPEC_EXPIRY_JSON="$d/missing.json" bash "$self" >/dev/null 2>&1
    [ "$?" = 2 ] && echo "  ok    unreadable file exits 2" || { echo "  FAIL  unreadable file did not exit 2"; fail=1; }

    echo "# issue writer (stub gh): opens once, then comments on the same issue"
    mkdir -p "$d/bin" && : > "$d/gh.log"
    cat > "$d/bin/gh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$GH_LOG"
case "$1 $2" in
  "issue list") [ -f "$GH_STATE" ] && echo 77; exit 0 ;;
  "api -X") case "$4" in */comments) echo 1; exit 0 ;; */issues) touch "$GH_STATE"; echo 77; exit 0 ;; esac ;;
esac
exit 1
EOF
    chmod +x "$d/bin/gh"
    for _ in 1 2; do
        PATH="$d/bin:$PATH" GH_LOG="$d/gh.log" GH_STATE="$d/gh.state" SPEC_EXPIRY_JSON="$d/open.json" \
            SPEC_CONFORMANCE_TODAY=2026-10-01 bash "$self" --issues >> "$d/issues.out" 2>&1
    done
    [ "$(grep -c '^opened #77 row=r1' "$d/issues.out")" = 1 ] && [ "$(grep -c '^updated #77 row=r1' "$d/issues.out")" = 1 ] \
        && echo "  ok    one issue per row: opened then updated" \
        || { echo "  FAIL  issue writer: $(tr '\n' ' ' < "$d/issues.out")"; fail=1; }

    echo "# defect fixtures (each must turn a case-table row red)"
    defect() { # defect <name> <sed expr>
        sed "$2" "$self" > "$d/defect.sh"
        if cmp -s "$self" "$d/defect.sh"; then echo "  FAIL  defect $1: sed changed nothing"; fail=1; return; fi
        if cases "$d/defect.sh" >/dev/null; then echo "  FAIL  defect $1 survives"; fail=1; else echo "  ok    defect $1 is caught"; fi
    }
    defect drop-warning   's/^        | (\$warn | sort_by/        | (\$warn | map(select(false)) | sort_by/'
    defect window-8-days  's/select(.left >= 0 and .left <= \$d)/select(.left >= 0 and .left <= \$d + 1)/'
    defect warn-expired   's/select(.left >= 0 and/select(.left >= -1 and/'
    defect warn-discharged 's/select(.value.discharged != true and/select(true and/'

    rm -rf "${d:?}"
    [ "$fail" = 0 ] && echo "spec_expiry_warn self-test: ALL PASS" || echo "spec_expiry_warn self-test: RED"
    return "$fail"
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
main "$@"
