#!/usr/bin/env bash
# check_pr_review_counts.sh - the counts PR-REVIEW-SKILL-002's own files state as
# MEASURED must equal what the tree derives.
#
# WHY
# ---
# The skill reviewed its own PR and found five counts stated as measured that the very
# commit shipping them falsified:
#
#   * the S3.B trigger "fires on five paths"                 -> 8
#   * the guard's mutation set "119/119", in FOUR places     -> 185 (the PR TITLE said 185)
#   * "this file is 65 tests"                                -> 121
#   * "the 22-row fixture table", twice                      -> 26
#   * "Not yet done: the backtest", in the commit landing results-v3.md at 3 of 3
#
# None of them was a lie anyone told. Each was true when written and went stale when the
# thing it counted grew, and nothing recomputed it. That is precisely the defect class
# this whole skill exists to catch - a stale count stated as measured - reproduced inside
# the artifact that defines it. So the three counts that are DERIVABLE FROM THE TREE are
# derived here, on every run, and the files that state them are checked against the
# derivation.
#
# HOW, AND WHY NOT WITH A CLEVER REGEX
# ------------------------------------
# Each row of the table below is (id, file, occurrences, template). The template is
# rendered with the DERIVED value and counted with `grep -c -F` - a fixed string, no
# regex, no window, no anchor word. The count must match EXACTLY:
#
#   too few  => the number went stale, or the sentence was deleted. Both are defects:
#              a claim that vanishes is not a claim that was checked, and a guard whose
#              universe can be emptied is a guard that passes over nothing.
#   too many => a new site states the number and the table does not know about it. The
#              table is the record of where this repository claims these numbers; a site
#              it has never seen has never been checked.
#
# The first draft of this guard used a keyword window ("any N/N within 200 characters of
# `mutate-guard`") and it was wrong twice before a single line of it ran: SKILL.md breaks
# `reports` and `119/119` across a newline, and ci.yml's "58 cgp tests" is a legitimate
# `[0-9]+ tests` two lines from the one under test. Five of this repository's
# `apr`-invocation patterns were wrong in exactly that way. A fixed string cannot be.
#
# THE FOURTH COUNT IS NOT HERE, ON PURPOSE. "The S3.B trigger fires on N paths" counts a
# DIFF, and a diff moves with every commit, so no derivation of it is stable enough to
# check a written number against. SKILL.md now pins that one to the commit it was measured
# at and prints the command that recomputes it, which is the honest form for a number that
# is a snapshot rather than an invariant.
#
# EVIDENCE FILES ARE DELIBERATELY OUT OF SCOPE. evidence/ is the append-only record of
# what a run measured WHEN IT RAN; "correcting" a past transcript to match today's tree is
# how a measurement becomes folklore.
#
#   bash scripts/check_pr_review_counts.sh              # check
#   bash scripts/check_pr_review_counts.sh --show       # print the derived values
#   bash scripts/check_pr_review_counts.sh --self-test  # both polarities, on copies
#
# EXIT: 0 every row matches; 1 any row does not; 2 the box cannot derive a value.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# The three derivations. Each reads the TREE, never a written-down number.
# ---------------------------------------------------------------------------

# The mutation set is a DERIVATION inside mutate-guard.sh (one `drop` and one `flip` per
# `reject B<n>` site, rescanned from the guard on every run, plus the named `text` edits),
# so its size is whatever that script says today. --list runs no mutants.
derive_mutants() {
    local root=$1 err rows tally rc=0
    err=$(mktemp "${TMPDIR:-/tmp}/prcounts-list.XXXXXX") || return 1
    # TWO derivations of the same number, from the two streams --list writes:
    # the catalogue rows on stdout and its own tally on stderr. They must agree.
    # A single one of them is a number this guard would be trusting rather than
    # deriving -- and the whole point of this file is that a number nothing
    # recomputes goes stale without anybody lying.
    rows=$( (cd "$root" && bash scripts/mutate-guard.sh --list 2>"$err") \
            | awk -F'\t' '$2 == "drop" || $2 == "flip" || $2 == "text" { n += 1 } END { print n + 0 }' )
    rc=${PIPESTATUS[0]}
    tally=$(sed -n 's/^\([0-9][0-9]*\) mutants$/\1/p' "$err" | tail -1)
    rm -f -- "$err"
    [ "$rc" -eq 0 ] || return 1
    [ -n "$tally" ] || return 1
    [ "$rows" -gt 0 ] || return 1
    [ "$rows" = "$tally" ] || {
        echo "$PROG: mutate-guard.sh --list printed $rows catalogue rows but tallied $tally" >&2
        return 1
    }
    printf '%s\n' "$rows"
}

# Fixture rows: directories, not a list. tests/pr-review.bats asserts this number too;
# that assertion and this one are the same fact checked from two sides.
derive_fixture_rows() {
    local root=$1 n
    n=$(find "$root/tests/fixtures/pr-review" -maxdepth 1 -type d -name 'row-*' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# bats tests: `@test` at column 0 in the file the CI step actually runs.
derive_bats_tests() {
    local root=$1 n
    [ -f "$root/tests/pr-review.bats" ] || return 1
    n=$(grep -c '^@test ' "$root/tests/pr-review.bats")
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# S13 (PRREV-015) adds a SECOND table, a SECOND bats file and a SECOND mutation set.
# They are derived here rather than left uncounted, because the universe this guard
# iterates is what decides whether it can fail at all: a table it has never heard of is
# a table whose stated size can go stale exactly the way the first four did. Same shape
# as the three above, one file over.
derive_quorum_mutants() {
    local root=$1 err rows tally rc=0
    err=$(mktemp "${TMPDIR:-/tmp}/prcounts-qlist.XXXXXX") || return 1
    rows=$( (cd "$root" && bash scripts/mutate_quorum_arm.sh --list 2>"$err") \
            | awk -F'\t' '$2 == "drop" || $2 == "flip" || $2 == "text" { n += 1 } END { print n + 0 }' )
    rc=${PIPESTATUS[0]}
    tally=$(sed -n 's/^\([0-9][0-9]*\) mutants$/\1/p' "$err" | tail -1)
    rm -f -- "$err"
    [ "$rc" -eq 0 ] || return 1
    [ -n "$tally" ] || return 1
    [ "$rows" -gt 0 ] || return 1
    [ "$rows" = "$tally" ] || {
        echo "$PROG: mutate_quorum_arm.sh --list printed $rows catalogue rows but tallied $tally" >&2
        return 1
    }
    printf '%s\n' "$rows"
}

derive_quorum_rows() {
    local root=$1 n
    n=$(find "$root/tests/fixtures/pr-review" -maxdepth 1 -type d -name 'q-*' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

derive_quorum_bats_tests() {
    local root=$1 n
    [ -f "$root/tests/pr-review-quorum.bats" ] || return 1
    n=$(grep -c '^@test ' "$root/tests/pr-review-quorum.bats")
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# Falsification tests: the contract's own list, counted from its `- id:` labels rather
# than from the prose that states the total. It went stale unseen - `pass_criteria` said
# 11 over 15 entries at PRREV-013 and over 17 at PRREV-015 - and the only thing that
# noticed was `cargo test -p aprender-contracts --test validate_contracts` printing
# `pr-review-skill-v2: pass_criteria says 11 tests, actual 17` into a corpus-wide failure
# nobody reads per-contract. Counting labels and not lines, the same reason S7's rows
# carry ids B1..B6.
derive_falsification_tests() {
    local root=$1 n
    [ -f "$root/contracts/pr-review-skill-v2.yaml" ] || return 1
    # THE UNIVERSE IS THE BLOCK, NOT A PREFIX. A first draft counted `^- id: F-PRREV-`
    # and the `tree-grew-a-falsifier` row went GREEN when it should have gone RED: an
    # entry added under any other id was invisible to the count that is supposed to
    # notice entries being added. The file holds `- id: KH-PRREV-*` under
    # `kani_harnesses:` too, so a bare `^- id:` over the whole file over-counts. The
    # right universe is the one the key delimits - from `falsification_tests:` to the
    # next top-level key - which is what the parser reading `pass_criteria` sees.
    n=$(awk '/^falsification_tests:/ { inblock = 1; next }
             /^[a-z_]+:/            { inblock = 0 }
             inblock && /^- id: /   { n += 1 }
             END                    { print n + 0 }' \
        "$root/contracts/pr-review-skill-v2.yaml")
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# ---------------------------------------------------------------------------
# The site table. id | file | occurrences | template (@N@ -> derived value)
# ---------------------------------------------------------------------------
SITES='
mutants|.claude/skills/pr-review/SKILL.md|2|@N@/@N@
mutants|contracts/binding.yaml|1|@N@/@N@
mutants|.github/workflows/ci.yml|1|@N@/@N@
fixture_rows|.github/workflows/ci.yml|2|@N@-row
fixture_rows|tests/pr-review.bats|1|@N@ row
fixture_rows|tests/pr-review.bats|1|-eq @N@ ]
bats_tests|.github/workflows/ci.yml|2|@N@ tests
quorum_mutants|docs/specifications/PR-REVIEW-SKILL-002-v2.md|2|@N@/@N@
quorum_mutants|.claude/skills/pr-review/SKILL.md|1|@N@/@N@
quorum_mutants|contracts/pr-review-skill-v2.yaml|1|@N@/@N@
quorum_mutants|.claude/skills/pr-review/SKILL.md|1|@N@-mutant
quorum_mutants|.github/workflows/ci.yml|2|@N@-mutant
quorum_mutants|docs/specifications/PR-REVIEW-SKILL-002-v2.md|2|@N@ mutants
quorum_bats_tests|docs/specifications/PR-REVIEW-SKILL-002-v2.md|1|@N@ rows
quorum_bats_tests|docs/specifications/PR-REVIEW-SKILL-002-v2.md|1|@N@-row
quorum_bats_tests|.claude/skills/pr-review/SKILL.md|2|@N@ rows
quorum_bats_tests|.github/workflows/ci.yml|1|@N@ rows
quorum_rows|tests/pr-review-quorum.bats|1|-eq @N@ ]
quorum_rows|tests/pr-review-quorum.bats|1|expected @N@ q-*
falsification_tests|contracts/pr-review-skill-v2.yaml|1|All @N@ falsification tests
'

# ---------------------------------------------------------------------------
# check <root> - run every row against the tree at <root>. Prints a table.
# ---------------------------------------------------------------------------
check() {
    local root=$1 quiet=${2:-} fails=0
    local -A derived=()
    local v
    v=$(derive_mutants      "$root") || die_env "cannot derive the mutation-set size (mutate-guard.sh --list)"
    derived[mutants]=$v
    v=$(derive_fixture_rows "$root") || die_env "cannot derive the fixture-row count"
    derived[fixture_rows]=$v
    v=$(derive_bats_tests   "$root") || die_env "cannot derive the bats test count"
    derived[bats_tests]=$v
    v=$(derive_quorum_mutants    "$root") || die_env "cannot derive the S13 mutation-set size (mutate_quorum_arm.sh --list)"
    derived[quorum_mutants]=$v
    v=$(derive_quorum_rows       "$root") || die_env "cannot derive the S13 q-* fixture-row count"
    derived[quorum_rows]=$v
    v=$(derive_quorum_bats_tests "$root") || die_env "cannot derive the S13 bats test count"
    derived[quorum_bats_tests]=$v
    v=$(derive_falsification_tests "$root") || die_env "cannot derive the contract's falsification-test count"
    derived[falsification_tests]=$v

    [ -n "$quiet" ] || {
        echo "=== the counts these files state as measured, against the tree ($PROG) ==="
        echo "derived:  mutants=${derived[mutants]}  fixture_rows=${derived[fixture_rows]}  bats_tests=${derived[bats_tests]}"
        echo "          quorum_mutants=${derived[quorum_mutants]}  quorum_rows=${derived[quorum_rows]}  quorum_bats_tests=${derived[quorum_bats_tests]}"
        echo "          falsification_tests=${derived[falsification_tests]}"
    }

    local row id file want tmpl needle got
    while IFS='|' read -r id file want tmpl; do
        [ -n "$id" ] || continue
        if [ ! -f "$root/$file" ]; then
            echo "FAIL  $id  $file is missing" >&2
            fails=$((fails + 1)); continue
        fi
        needle=${tmpl//@N@/${derived[$id]}}
        got=$(grep -c -F -- "$needle" "$root/$file")
        if [ "$got" -eq "$want" ]; then
            [ -n "$quiet" ] || printf 'ok    %-13s %-40s %s x%s\n' "$id" "$file" "$needle" "$got"
        else
            fails=$((fails + 1))
            printf 'FAIL  %-13s %-40s "%s" occurs %s time(s), the table says %s\n' \
                "$id" "$file" "$needle" "$got" "$want" >&2
            if [ "$got" -lt "$want" ]; then
                echo "      The tree derives ${derived[$id]}. Either the file states a" >&2
                echo "      STALE number, or the sentence that stated it was deleted." >&2
                # Show the neighbourhood using the LONGER of the template's two
                # literal halves. `${tmpl%%@N@*}` is EMPTY when the template starts
                # with @N@, and `grep -F ""` matches every line in the file - a
                # diagnostic that prints the top of the file and calls it a hint.
                local pre=${tmpl%%@N@*} post=${tmpl##*@N@} hint
                if [ ${#pre} -ge ${#post} ]; then hint=$pre; else hint=$post; fi
                if [ -n "$hint" ]; then
                    echo "      Lines carrying \"$hint\":" >&2
                    grep -nF -- "$hint" "$root/$file" 2>/dev/null | head -4 | sed 's/^/        /' >&2
                fi
            else
                echo "      A site states this number and the table has never seen it." >&2
                echo "      Add the row rather than widening the count." >&2
            fi
        fi
    done <<< "$(printf '%s\n' "$SITES" | sed '/^[[:space:]]*$/d')"

    if [ "$fails" -ne 0 ]; then
        echo "--- $fails row(s) disagree with the tree ---" >&2
        return 1
    fi
    [ -n "$quiet" ] || echo "PASS  every stated count equals the derived one"
    return 0
}

# ---------------------------------------------------------------------------
# safe_rm_scratch <path> <required-substring> - a recursive delete, guarded.
# SEC011, and not decoration: the two ways `rm -rf -- "$x"` goes wrong are an
# EMPTY x and an x that is not ours. Both are checked before the expansion, and
# a path that fails either check is left alone rather than deleted "carefully".
# ---------------------------------------------------------------------------
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

ST_ROOT=''
cleanup() { safe_rm_scratch "$ST_ROOT" 'prcounts-selftest.'; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# --self-test: MUTATE THE TREE, not the guard's opinion of it. Each row copies
# the repository, makes one edit, and requires the verdict to flip. A guard that
# is only ever run against a passing tree has never been shown to fail.
# ---------------------------------------------------------------------------
self_test() {
    local fails=0
    ST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/prcounts-selftest.XXXXXX") || die_env "mktemp failed"

    local base="$ST_ROOT/tree"
    mkdir -p "$base"
    # Only the files the table names, plus what the derivations read.
    ( cd "$REPO_ROOT" && \
      tar -cf - .claude/skills/pr-review/SKILL.md contracts/binding.yaml \
                .github/workflows/ci.yml tests/pr-review.bats \
                scripts/mutate-guard.sh scripts/check_pr_review_receipt.sh \
                scripts/check_pr_review_counts.sh tests/fixtures/pr-review \
                contracts/pr-review-skill-v2.yaml \
                docs/specifications/PR-REVIEW-SKILL-002-v2.md \
                tests/pr-review-quorum.bats scripts/mutate_quorum_arm.sh \
                scripts/pr_review_quorum_arm.sh ) \
      | ( cd "$base" && tar -xf - ) || die_env "could not stage a copy of the tree"

    local nrows=0
    row() {
        local id=$1 want=$2 desc=$3; shift 3
        nrows=$((nrows + 1))
        local dir="$ST_ROOT/$id" got=0
        safe_rm_scratch "$dir" 'prcounts-selftest.'
        cp -a "$base" "$dir"
        "$@" "$dir" || die_env "row $id could not apply its edit"
        # A MUTATION THAT MATCHED NOTHING IS A BROKEN HARNESS, NOT A DEAD MUTANT.
        # Every edit below is a `sed` against an anchor, and an anchor that has
        # drifted changes nothing while the row still reports a verdict: the
        # unmutated tree passes, the row wanted RED, and the table would read
        # "the guard failed to catch it" when in fact nothing was ever done to
        # it. mutate-guard.sh learned this the same way (its note 1: a probe
        # once reported "34 passed" over a file it had not touched). So the tree
        # must actually differ, and a row that did not change it stops the run
        # with a distinct diagnosis instead of a misattributed failure.
        if [ "$id" != baseline ] && diff -rq "$base" "$dir" >/dev/null 2>&1; then
            die_env "HARNESS-BROKEN: row $id's edit changed nothing (its anchor has drifted)"
        fi
        check "$dir" quiet >/dev/null 2>&1 || got=$?
        if [ "$got" -eq "$want" ]; then
            printf 'ok    %-26s rc=%s  %s\n' "$id" "$got" "$desc"
        else
            printf 'FAIL  %-26s rc=%s (wanted %s)  %s\n' "$id" "$got" "$want" "$desc"
            fails=$((fails + 1))
        fi
    }

    noop()          { :; }
    # DERIVED ANCHORS, and the four rows below are why the paragraph under them exists.
    # They were written with the literal numbers of their day - 185/185, 26-row,
    # 121 tests - and PRREV-015 moved all three to 215/215, 35-row and 149 tests. Every
    # stated count in the tree was updated; the four anchors that MUTATE those counts
    # were not, so each edit matched nothing and `--self-test` exited 2 HARNESS-BROKEN
    # on the very commit that widened the table. The harness caught itself, which is the
    # only reason this is a fix and not a survivor: had the rows silently changed
    # nothing, the table would have read 13/13 green over four mutations never applied.
    # Reading the value from the same derivation the check uses makes each row
    # self-maintaining, and the write-back is by ONE so no row depends on a magnitude.
    stale_mutants() { local n; n=$(derive_mutants "$1")
                      sed -i "s#$n/$n#$((n - 1))/$((n - 1))#" "$1/.claude/skills/pr-review/SKILL.md"; }
    stale_rows()    { local n; n=$(derive_fixture_rows "$1")
                      sed -i "s#$n-row#$((n - 1))-row#"   "$1/.github/workflows/ci.yml"; }
    stale_tests()   { local n; n=$(derive_bats_tests "$1")
                      sed -i "s#$n tests#$((n - 1)) tests#" "$1/.github/workflows/ci.yml"; }
    delete_claim()  { local n; n=$(derive_mutants "$1")
                      sed -i "0,\#$n/$n#{\#$n/$n#d}" "$1/contracts/binding.yaml"; }
    stale_ftests()  { local n; n=$(derive_falsification_tests "$1")
                      sed -i "s#All $n falsification tests#All $((n - 1)) falsification tests#" \
                             "$1/contracts/pr-review-skill-v2.yaml"; }
    # The invented id is deliberately NOT an F-PRREV- one: the row must fire on an entry
    # being ADDED, not on it matching a naming convention, and a prefix-shaped derivation
    # let exactly this edit through green.
    grow_ftests()   { sed -i 's#^- id: F-PRREV-002$#- id: F-INVENTED-999\n  rule: a falsification test the contract does not count\n\n&#' \
                             "$1/contracts/pr-review-skill-v2.yaml"; }
    grow_bats()     { printf '\n@test "a new test the docs do not count" {\n  true\n}\n' >> "$1/tests/pr-review.bats"; }
    grow_rows()     { mkdir -p "$1/tests/fixtures/pr-review/row-27-invented"; }
    grow_mutants()  { sed -i 's#^  \[ -n "$head" \] || reject B1 "predicate.head_sha is absent" || return 1#&\n  [ -n "$head" ] || reject B1 "an invented rule nothing documents" || return 1#' \
                             "$1/scripts/check_pr_review_receipt.sh"; }
    # DERIVED anchors, not literal ones. The four rows above were written with literal
    # numbers and each goes stale the day its table grows: `stale_qrows` anchored on
    # `-eq 55 ]` and stopped matching the moment a 56th fixture landed, at which point
    # the harness correctly refused to report a verdict it had not earned
    # (HARNESS-BROKEN). Reading the value from the same derivation the check uses makes
    # the row self-maintaining while still performing a real edit.
    stale_qmutants(){ local n; n=$(derive_quorum_mutants "$1")
                      sed -i "s#$n/$n#$((n - 18))/$((n - 18))#" "$1/.claude/skills/pr-review/SKILL.md"; }
    stale_qrows()   { local n; n=$(derive_quorum_rows "$1")
                      sed -i "s#-eq $n \]#-eq $((n - 4)) ]#" "$1/tests/pr-review-quorum.bats"; }
    grow_qrows()    { mkdir -p "$1/tests/fixtures/pr-review/q-99-invented"; }
    grow_qbats()    { printf '\n@test "an S13 row the docs do not count" {\n  true\n}\n' >> "$1/tests/pr-review-quorum.bats"; }
    grow_qmutants() { sed -i 's#^  \[ "$verdict" = "PASS" \] \\#  [ -n "$verdict" ] || refuse Q6 "an invented refusal nothing documents" || return 1\n&#' \
                             "$1/scripts/pr_review_quorum_arm.sh"; }

    echo "--- check_pr_review_counts.sh --self-test ---"
    row baseline                0 "the tree as committed"                                   noop
    row stale-mutation-score    1 "the mutation score written back by one"                  stale_mutants
    row stale-fixture-rows      1 "the fixture-row count written back by one"               stale_rows
    row stale-bats-count        1 "the bats test count written back by one"                 stale_tests
    row claim-deleted           1 "the sentence stating the count is deleted"               delete_claim
    row tree-grew-a-test        1 "a bats test is added and no file says so"                grow_bats
    row tree-grew-a-fixture-row 1 "a row-* fixture is added and no file says so"            grow_rows
    row tree-grew-a-mutant      1 "a reject site is added and no file says so"              grow_mutants
    row stale-quorum-mutation   1 "the S13 mutation score written back by 18"               stale_qmutants
    row stale-quorum-rows       1 "the S13 row assertion written back by 4"                  stale_qrows
    row tree-grew-a-quorum-row  1 "a q-* fixture is added and no file says so"              grow_qrows
    row tree-grew-a-quorum-test 1 "an S13 bats test is added and no file says so"           grow_qbats
    row tree-grew-a-refusal     1 "a refuse Q<n> site is added and no file says so"         grow_qmutants
    row stale-falsifier-count   1 "the contract's falsification-test total written back by one" stale_ftests
    row tree-grew-a-falsifier   1 "a falsification test is added and the contract does not count it" grow_ftests

    if [ "$fails" -ne 0 ]; then
        echo "--- $fails row(s) did not produce the required verdict ---" >&2
        return 1
    fi
    # COUNTED, not stated. This line said 13/13 while the table held 13 rows and would
    # have said 13/13 over 15; a guard that hard-codes its own row count is the defect
    # it exists to catch, one level in.
    echo "--- $nrows/$nrows rows, both polarities ---"
    return 0
}

for t in grep find sed tar diff; do
    command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
done

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --show)
      printf 'mutants            %s\n' "$(derive_mutants           "$REPO_ROOT")"
      printf 'fixture_rows       %s\n' "$(derive_fixture_rows      "$REPO_ROOT")"
      printf 'bats_tests         %s\n' "$(derive_bats_tests        "$REPO_ROOT")"
      printf 'quorum_mutants     %s\n' "$(derive_quorum_mutants    "$REPO_ROOT")"
      printf 'quorum_rows        %s\n' "$(derive_quorum_rows       "$REPO_ROOT")"
      printf 'quorum_bats_tests  %s\n' "$(derive_quorum_bats_tests "$REPO_ROOT")"
      printf 'falsification_tests %s\n' "$(derive_falsification_tests "$REPO_ROOT")"
      exit 0 ;;
  -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
  '') check "$REPO_ROOT"; exit $? ;;
  *) echo "$PROG: unknown argument: $1" >&2; exit 1 ;;
esac
