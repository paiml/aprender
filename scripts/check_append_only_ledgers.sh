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

# attr_golden PATH -> 0 if .gitattributes declares it `audit-golden`: a
# single-writer file under docs/audits/ that is NOT a ledger (#4354: the review
# corpus, the #4026 oracle captures). Declared, never inferred; never union.
# Only the bare form declares it: `audit-golden=true`, `=false` and
# `-audit-golden` are NOT golden (R5), so such a file must still be union.
attr_golden() { [ "$(git check-attr audit-golden -- "$1" 2>/dev/null | sed 's/.*: //')" = set ]; }

measure() {
    local rc=0 n_ledger=0 n_other=0 f
    printf '=== append-only ledgers must be union-merged; goldens must NOT be ===\n'

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        n_ledger=$((n_ledger + 1))
        if attr_golden "$f"; then
            if attr_union "$f"; then
                printf 'FAIL  %s is declared audit-golden AND union-merged. Pick one.\n' "$f"
                rc=1
            else
                printf 'ok    golden  %s\n' "$f"
            fi
        elif attr_union "$f"; then
            printf 'ok    union   %s\n' "$f"
        else
            printf 'FAIL  %s is an append-only ledger and is NOT union-merged.\n' "$f"
            printf '      Add it to .gitattributes, declare it audit-golden, or move it out of docs/audits/.\n'
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
    # R1b discriminates attr_golden: a predicate that always says golden would turn
    # every ledger into golden+union, and nothing else here would notice.
    _row 'R1b docs/audits/impl-estimates.jsonl (a real ledger) is NOT audit-golden' \
         "$(attr_golden docs/audits/impl-estimates.jsonl && echo yes || echo no)" no

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

    # R4: a declared golden under docs/audits/ resolves golden and NOT union (the
    # real tree), and a golden that is ALSO union is caught (a throwaway tree).
    _row 'R4a docs/audits/review-corpus/corpus-v1.jsonl is audit-golden, not union' \
         "$(attr_golden docs/audits/review-corpus/corpus-v1.jsonl && ! attr_union docs/audits/review-corpus/corpus-v1.jsonl && echo yes || echo no)" yes
    # R4b-R4d run MEASURE itself on a throwaway tree, so the golden+union FAIL is
    # the decision under test and not just the two predicates it reads. R4d is the
    # control: the same tree with the golden declared alone PASSES, so R4c's FAIL
    # comes from the both-declared branch and from nothing else in the scan.
    t="$(mktemp -d)" || return 1
    (
      cd "$t" || exit 1
      git init -q . && mkdir -p docs/audits/g && : > docs/audits/g/x.jsonl
      printf 'docs/audits/**/*.jsonl merge=union\ndocs/audits/g/*.jsonl audit-golden\n' > .gitattributes
      git add -A
      { attr_golden docs/audits/g/x.jsonl && attr_union docs/audits/g/x.jsonl && echo caught || echo missed; } > both
      measure > both.out 2>&1; printf '%s\n' "$?" > both.rc
      grep -c 'declared audit-golden AND union-merged' both.out > both.named
      printf 'docs/audits/g/*.jsonl audit-golden\n' > .gitattributes
      measure > solo.out 2>&1; printf '%s\n' "$?" > solo.rc
    )
    _row 'R4b a golden that is ALSO union-merged is seen as both'         "$(cat "$t/both" 2>/dev/null)"       caught
    _row 'R4c measure FAILs that tree (rc 1)'                             "$(cat "$t/both.rc" 2>/dev/null)"    1
    _row 'R4d ... naming the file as declared audit-golden AND union'     "$(cat "$t/both.named" 2>/dev/null)" 1
    _row 'R4e control: the golden declared alone, measure PASSes (rc 0)'  "$(cat "$t/solo.rc" 2>/dev/null)"    0
    rm -rf "${t:?}"

    # R5: only the bare attribute declares a golden. The value forms are NOT golden,
    # so a file carrying one is still held to union (the stricter reading).
    t="$(mktemp -d)" || return 1
    (
      cd "$t" || exit 1
      git init -q . && mkdir -p docs/audits
      printf 'docs/audits/t.jsonl audit-golden=true\ndocs/audits/f.jsonl audit-golden=false\ndocs/audits/u.jsonl -audit-golden\n' > .gitattributes
      for f in t f u; do attr_golden "docs/audits/$f.jsonl" && echo golden || echo not; done | tr '\n' ' ' > forms
    )
    _row 'R5 audit-golden=true, =false and -audit-golden are NOT golden' "$(cat "$t/forms" 2>/dev/null)" 'not not not '
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
