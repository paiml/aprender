#!/usr/bin/env bash
# check_append_only_ledgers.sh -- an append-only ledger is union-merged; a golden is NOT.
#
# WHY
# ---
# docs/audits/impl-estimates.jsonl is appended one row per (ticket, phase). Two
# branches that both append conflict on the last line, so a squash-merge from the
# queue makes EVERY branch carrying it DIRTY. Measured 2026-09-14: #3001 at
# ~07:05Z and #3093 at ~08:35Z, 90 minutes apart, both resolved identically --
# take the union. A resolution that is always the same is a merge driver nobody
# wrote. `.gitattributes` now writes it.
#
# THE SCOPE IS THE WHOLE POINT, which is why this guard exists rather than a
# comment. `*.jsonl merge=union` would be a DEFECT: most .jsonl here are goldens
# and datasets (contrastive-data goldens, datasets/, pr-review receipts), and a
# union merge there silently DUPLICATES rows -- destroying exactly the
# byte-identity those files exist to assert. R3 below is that row.
#
# Usage
#   bash scripts/check_append_only_ledgers.sh              measure this tree
#   bash scripts/check_append_only_ledgers.sh --self-test  the case table
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

LEDGER_GLOB='docs/audits/*.jsonl'

usage() { printf 'check_append_only_ledgers.sh [--self-test|--help]\n'; }

# attr_union PATH -> 0 if git resolves merge=union for it
attr_union() { [ "$(git check-attr merge -- "$1" 2>/dev/null | sed 's/.*: //')" = union ]; }

measure() {
    local rc=0 n_ledger=0 n_other=0 f
    printf '=== append-only ledgers must be union-merged; goldens must NOT be ===\n'

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        n_ledger=$((n_ledger + 1))
        if attr_union "$f"; then
            printf 'ok    union   %s\n' "$f"
        else
            printf 'FAIL  %s is an append-only ledger and is NOT union-merged.\n' "$f"
            printf '      Add it to .gitattributes, or move it out of docs/audits/.\n'
            rc=1
        fi
    done < <(git ls-files "$LEDGER_GLOB")

    # Vacuity: the glob matching nothing would report zero problems and read as a
    # pass. That is the failure mode this whole file is about.
    if [ "$n_ledger" -lt 1 ]; then
        printf 'FAIL (vacuity): no ledger matched %s. The scan is broken, not the tree.\n' "$LEDGER_GLOB"
        return 1
    fi

    # The discrimination half: nothing OUTSIDE docs/audits/ may be union-merged.
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$f" in docs/audits/*) continue ;; esac
        n_other=$((n_other + 1))
        if attr_union "$f"; then
            printf 'FAIL  %s is union-merged and is NOT an append-only ledger.\n' "$f"
            printf '      A union merge DUPLICATES rows. On a golden or a dataset that destroys\n'
            printf '      the byte-identity it exists to assert.\n'
            rc=1
        fi
    done < <(git ls-files '*.jsonl')

    printf '\n%s ledger(s), %s other .jsonl checked\n' "$n_ledger" "$n_other"
    [ "$rc" -eq 0 ] && printf 'PASS\n'
    return "$rc"
}

self_test() {
    local fails=0 rows=0 t
    _row() { rows=$((rows + 1)); if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s (got %s, wanted %s)\n' "$1" "$2" "$3"; fails=1; fi; }

    # R1/R2: the attribute resolves the way the tree claims.
    _row 'R1 docs/audits/impl-estimates.jsonl resolves merge=union' \
         "$(attr_union docs/audits/impl-estimates.jsonl && echo yes || echo no)" yes
    _row 'R2 a golden does NOT resolve merge=union' \
         "$(attr_union crates/aprender-contrastive-data/tests/goldens/golden_corpus_train.jsonl && echo yes || echo no)" no

    # R3 IS THE MECHANISM, not the declaration: a real two-sided append, merged
    # in a throwaway repo, with and without the attribute. An attribute that is
    # declared and does not resolve conflicts is the theater this repo names most.
    t="$(mktemp -d)" || return 1
    (
      cd "$t" || exit 1
      git init -q . && git config user.email t@t && git config user.name t
      printf 'base\n' > l.jsonl && git add -A && git -c core.hooksPath=/dev/null commit -q -m base
      git checkout -q -b side && printf 'base\nSIDE\n' > l.jsonl && git -c core.hooksPath=/dev/null commit -qam side
      git checkout -q - && printf 'base\nMAIN\n' > l.jsonl && git -c core.hooksPath=/dev/null commit -qam main
      git merge side --no-commit >/dev/null 2>&1; printf '%s\n' "$(grep -c '<<<<' l.jsonl)" > without
      git merge --abort 2>/dev/null
      printf '*.jsonl merge=union\n' > .gitattributes
      git add -A && git -c core.hooksPath=/dev/null commit -qm attrs
      git merge side --no-commit >/dev/null 2>&1; printf '%s\n' "$(grep -c '<<<<' l.jsonl)" > with
      grep -c . l.jsonl > lines
    )
    _row 'R3a without the attribute, a two-sided append CONFLICTS' "$(cat "$t/without" 2>/dev/null)" 1
    _row 'R3b with merge=union it does NOT'                        "$(cat "$t/with" 2>/dev/null)"    0
    _row 'R3c and the result is the UNION (base+MAIN+SIDE), not one side' "$(cat "$t/lines" 2>/dev/null)" 3
    rm -rf "${t:?}"

    printf '\n%s row(s), %s\n' "$rows" "$( [ "$fails" -eq 0 ] && echo '0 red / FALSIFIER GREEN' || echo RED )"
    return "$fails"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test|--selftest) self_test; exit $? ;;
    '') measure; exit $? ;;
    *) usage >&2; exit 2 ;;
esac
