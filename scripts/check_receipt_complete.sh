#!/usr/bin/env bash
# check_receipt_complete.sh — an implementation receipt is complete only when
# it says so in a machine-readable terminal marker (PMAT-1056, C0-7, #2984;
# PP-066 driver STEP A6).
#
# WHY THIS EXISTS
# ---------------
# The PP-066 driver resumes from docs/audits/impl-<ticket>-receipt.md after
# any interruption. A receipt that was half-written when a usage limit hit,
# or one whose prose says "done" while its DoD is open, reads the same as a
# finished one unless a marker the writer sets LAST says otherwise. So:
#   * a receipt opens with YAML front matter carrying `status: complete` or
#     `status: partial` (a `partial=false` inside the prose is not a marker);
#   * the writer writes `<path>.tmp` and `mv`s it into place, so a receipt is
#     either wholly present or absent — never truncated (the guard refuses a
#     tracked `*.tmp` receipt as a torn write that was committed);
#   * the DAG (docs/specifications/pp-066-dag.yaml) may mark a row
#     `status: complete` only when that row's receipt carries
#     `status: complete` — the resume logic reads the receipt, not the DAG.
#
#   bash scripts/check_receipt_complete.sh <receipt.md>        # 0 complete · 1 partial or no marker · 2 usage
#   bash scripts/check_receipt_complete.sh --dag [<dag.yaml>]  # every row with status: complete has a complete receipt
#   bash scripts/check_receipt_complete.sh --selftest          # case table, both polarities
#
# A receipt that predates this rule (no front matter) is `partial` for the
# resume logic and RED under `--dag` once its row claims completion: the rule
# arms forward through the DAG, and legacy receipts are not rewritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DAG="docs/specifications/pp-066-dag.yaml"
PROG=check_receipt_complete

usage() {
    printf 'usage: %s <receipt.md> | --dag [<dag.yaml>] | --selftest\n' "$PROG" >&2
    exit 2
}

# receipt_status <file> -> prints complete|partial|none|torn ; the marker is the
# `status:` key inside the leading `---` front-matter block, nothing else.
receipt_status() {
    local f=$1
    [ -f "$f" ] || { printf 'none\n'; return 0; }
    case "$f" in *.tmp) printf 'torn\n'; return 0 ;; esac
    if [ "$(head -n1 "$f")" != "---" ]; then printf 'none\n'; return 0; fi
    local s
    s=$(awk 'NR==1 && $0=="---" {inb=1; next} inb && $0=="---" {exit} inb && /^status:[[:space:]]*/ {sub(/^status:[[:space:]]*/, ""); gsub(/[[:space:]"'"'"']/, ""); print; exit}' "$f")
    case "$s" in
        complete|partial) printf '%s\n' "$s" ;;
        *) printf 'none\n' ;;
    esac
}

check_one() { # exit 0 complete · 1 partial/none/torn
    local f=$1 st
    st=$(receipt_status "$f")
    case "$st" in
        complete) printf 'ok    %s: status: complete\n' "$f"; return 0 ;;
        partial)  printf 'FAIL  %s: status: partial (the DoD is open; not a finished receipt)\n' "$f"; return 1 ;;
        torn)     printf 'FAIL  %s: a *.tmp receipt is a torn write; the writer must mv it into place\n' "$f"; return 1 ;;
        *)        printf 'FAIL  %s: no `status: complete|partial` in the leading front matter (a marker set LAST, never prose)\n' "$f"; return 1 ;;
    esac
}

check_dag() { # every row with status: complete has a receipt with status: complete; no tracked *.tmp receipt
    local dag=$1 rc=0 line rid pid rf st
    [ -f "$dag" ] || { printf '%s: ENV - %s is missing (the box cannot answer)\n' "$PROG" "$dag" >&2; return 2; }
    command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; return 2; }
    while IFS=$'\t' read -r rid pid; do
        rf="$ROOT/docs/audits/impl-${pid}-receipt.md"
        st=$(receipt_status "$rf")
        if [ "$st" = complete ]; then
            printf 'ok    %s (%s): status: complete in %s\n' "$rid" "$pid" "${rf#"$ROOT"/}"
        else
            printf 'FAIL  %s (%s): the DAG says status: complete but %s is %s\n' "$rid" "$pid" "${rf#"$ROOT"/}" "$st"
            rc=1
        fi
    done < <(python3 - "$dag" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for r in d.get("rows", []):
    if r.get("status") == "complete":
        print(f"{r['id']}\t{r.get('pmat_id') or 'UNKNOWN'}")
PY
)
    local torn
    torn=$(cd "$ROOT" && git ls-files 'docs/audits/impl-*-receipt.md.tmp' 2>/dev/null || true)
    if [ -n "$torn" ]; then printf 'FAIL  tracked torn receipt(s): %s\n' "$torn"; rc=1; fi
    if [ "$rc" = 0 ]; then printf 'PASS  every DAG row marked complete has a receipt whose front matter says status: complete; no torn receipt is tracked\n'; fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# --selftest: fixtures under mktemp -d, both polarities
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/rcpt-selftest.XXXXXX")
    safe_rm_scratch() {
        local victim=${1:-} must=${2:-}
        [ -n "$victim" ] || return 0
        [ -n "$must" ]   || return 0
        [ "$victim" != "/" ] || return 0
        case "$victim" in
          *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
          *) return 0 ;;
        esac
    }
    cleanup() { safe_rm_scratch "$TD" 'rcpt-selftest.'; }
    trap cleanup EXIT
    n=0; red=0
    row() { # row <want rc> <label> <cmd...>
        local want=$1 label=$2; shift 2; local rc=0
        n=$((n + 1))
        "$@" >"$TD/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
    }
    printf -- '---\nstatus: complete\nticket: PMAT-1\n---\n# receipt\n' > "$TD/complete.md"
    printf -- '---\nstatus: partial\nticket: PMAT-2\n---\n# receipt\n' > "$TD/partial.md"
    printf -- '# impl receipt — PMAT-3\n\npartial=false\n' > "$TD/legacy.md"
    printf -- '---\nticket: PMAT-4\n---\n# receipt\n' > "$TD/nostatus.md"
    printf -- '---\nstatus: complete\n---\n' > "$TD/torn.md.tmp"
    printf -- '---\nstatus: "complete"\n---\n' > "$TD/quoted.md"
    printf -- '# not front matter\n---\nstatus: complete\n---\n' > "$TD/late.md"
    row 0 "front matter status: complete"                        bash "$0" "$TD/complete.md"
    row 1 "front matter status: partial"                         bash "$0" "$TD/partial.md"
    row 1 "legacy receipt: prose partial=false is not a marker"  bash "$0" "$TD/legacy.md"
    row 1 "front matter with no status key"                      bash "$0" "$TD/nostatus.md"
    row 1 "a *.tmp receipt is a torn write"                      bash "$0" "$TD/torn.md.tmp"
    row 0 "a quoted status value still resolves"                 bash "$0" "$TD/quoted.md"
    row 1 "status in a block that is not the LEADING front matter" bash "$0" "$TD/late.md"
    row 1 "a missing receipt file"                               bash "$0" "$TD/absent.md"
    # --dag rows: a fake repo root with a DAG and receipts
    R="$TD/repo"; mkdir -p "$R/docs/audits" "$R/docs/specifications" "$R/scripts"
    cp "$0" "$R/scripts/check_receipt_complete.sh"
    ( cd "$R" && git init -q . && git config user.email t@t && git config user.name t )
    cat > "$R/docs/specifications/pp-066-dag.yaml" <<'EOF'
rows:
- {id: A-1, pmat_id: PMAT-11, status: complete}
- {id: A-2, pmat_id: PMAT-12, status: open}
EOF
    printf -- '---\nstatus: complete\n---\n' > "$R/docs/audits/impl-PMAT-11-receipt.md"
    row 0 "--dag: the one complete row has a complete receipt; open rows need none" bash "$R/scripts/check_receipt_complete.sh" --dag "$R/docs/specifications/pp-066-dag.yaml"
    printf -- '---\nstatus: partial\n---\n' > "$R/docs/audits/impl-PMAT-11-receipt.md"
    row 1 "--dag: a row marked complete over a partial receipt (the registered mutation)" bash "$R/scripts/check_receipt_complete.sh" --dag "$R/docs/specifications/pp-066-dag.yaml"
    rm -f "$R/docs/audits/impl-PMAT-11-receipt.md"
    row 1 "--dag: a row marked complete with no receipt at all" bash "$R/scripts/check_receipt_complete.sh" --dag "$R/docs/specifications/pp-066-dag.yaml"
    printf -- '---\nstatus: complete\n---\n' > "$R/docs/audits/impl-PMAT-11-receipt.md"
    printf -- '---\nstatus: complete\n---\n' > "$R/docs/audits/impl-PMAT-12-receipt.md.tmp"
    ( cd "$R" && git add -A >/dev/null 2>&1 && git commit -qm x >/dev/null 2>&1 )
    row 1 "--dag: a tracked *.tmp receipt is a torn write that was committed" bash "$R/scripts/check_receipt_complete.sh" --dag "$R/docs/specifications/pp-066-dag.yaml"
    row 2 "--dag: a missing DAG is exit 2, never a pass" bash "$R/scripts/check_receipt_complete.sh" --dag "$R/docs/specifications/nope.yaml"
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

case "${1:-}" in
    --dag) dag="${2:-$ROOT/$DEFAULT_DAG}"; [ -f "$dag" ] || dag="$ROOT/${2:-$DEFAULT_DAG}"; check_dag "$dag" ;;
    ''|--help|-h) usage ;;
    --*) usage ;;
    *) check_one "$1" ;;
esac
