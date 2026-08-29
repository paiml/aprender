#!/usr/bin/env bash
# check_perf_claims_cite_receipts.sh — APR-PERF-GATE-001 v2.2 §3.2 / §5,
# PERF-010. A speed comparison a user reads must point at the receipt that
# produced it.
#
#   bash scripts/check_perf_claims_cite_receipts.sh            # gate
#   bash scripts/check_perf_claims_cite_receipts.sh --selftest # case table
#   bash scripts/check_perf_claims_cite_receipts.sh --update   # re-baseline
#
# THE RULE IT ENFORCES (§3.2, verbatim):
#   "A number in README.md, book/ or docs/ is legal iff it cites an evidence/
#    receipt path."
#
# WHY THIS IS NOT check_no_claim_literals.sh AGAIN. That guard is the NEGATIVE
# half: a comparator ratio and a bare throughput literal are illegal on a
# user-facing surface *regardless of citation*, so asking them to cite anything
# would be pointless — they have to go. This guard is the POSITIVE half, over
# the class that is allowed to stay: a speed comparison that names no
# competitor. "apr ~4.91x faster (ratio 0.203)" is a legal sentence. It is
# legal only if something can prove how it was measured.
#
# The two detectors are deliberately DISJOINT, so a line never carries two
# contradictory remedies (delete it / cite it):
#
#   throughput literal, comparator ratio, [X] figure  -> check_no_claim_literals.sh
#   speed comparison with no named comparator         -> HERE
#
# WHAT "CITES A RECEIPT" MEANS, MECHANICALLY. Within the claim's line or the
# three lines either side of it, a token matching `evidence/<path>` appears,
# AND that path exists in the tree.
#
# Three candidate definitions were weighed:
#
#   a commit SHA    rejected. It records WHEN, not HOW. A SHA can be cited
#                   beside a number nothing measured, and nothing can
#                   dereference it to a measurement. It is provenance theatre.
#   a `receipt:` key rejected. It coins a second dialect for a word this repo
#                   already uses concretely in 33 files — bench_receipt.py,
#                   check_parity_receipt.sh, the receipt.commit staleness arm —
#                   all of which mean a file under evidence/.
#   an evidence/ path  CHOSEN. It is the only one that is DEREFERENCEABLE: the
#                   guard resolves it and asserts the file is there. That turns
#                   "cites" from a syntactic gesture into a checkable claim, and
#                   it is the spelling §3.2 actually writes.
#
# A CITATION THAT RESOLVES TO NOTHING IS WORSE THAN NO CITATION, and it is the
# one failure the baseline may not absorb. An uncited number is inherited debt —
# visible, recorded, ratcheted down. A citation pointing at a path that does not
# exist is an active forgery: it buys the reader's trust with a file nobody can
# open. So a dangling citation FAILS at a baselined location too, and that
# asymmetry is row 11 of the case table.
#
# WHY docs/specifications/ IS OUT OF THE UNIVERSE, same as the sibling guard.
# APR-PERF-GATE-001 is the document that NAMES these defects; its §0.4 table
# literally lists uncited figures in a column headed "no receipt", and §5 names
# the mutation for this file as "uncited number in docs/". Scanning the spec
# would red this guard against its own specification, and the only levers would
# be an exemption or rewording the spec to avoid quoting what it bans. That is
# the self-reference trap that has reddened a sibling guard three times in this
# epic. Residual hole, stated rather than hidden: a claim laundered into
# docs/specifications/ is invisible here. It is not invisible to review, because
# a spec is a design document and a reader does not take its examples for
# results — which is exactly why the exclusion is safe and a book/ chapter's
# would not be.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

BASELINE="scripts/perf_claim_citation_baseline.txt"

# ---------------------------------------------------------------- patterns --
# A SPEED COMPARISON: a ratio bound to a word that means "in less time".
#
# The binding is what makes this tractable. A bare `3x` in prose is a matrix
# dimension, a compression ratio or a data-scale remark — the first draft of
# this pattern also matched `smaller|larger` and flagged "24 bits -> 8 bits =
# 3x smaller" and "salary values are ~6000x larger than age values". Neither is
# a measurement; both are arithmetic. Requiring a speed word drops them and
# costs no real claim, because a speedup that never says it is a speedup is not
# being read as one either.
#
# `(x|×)` rather than `[x×]`: `×` is two bytes in UTF-8, and inside a bracket
# expression under LC_ALL=C that becomes a byte class which matches the trailing
# byte of unrelated multibyte characters. Alternation is locale-safe.
#
# `[0-9][0-9,]*` admits the grouped form — book/src/examples/shell-completion.md
# writes "13,793x faster", and `[0-9]+` alone would have matched only "793x".
SPEEDWORD='faster|slower|speedup|speed-?up'
RATIO_THEN_WORD="[0-9][0-9,]*(\\.[0-9]+)?[[:space:]]*(x|×)[[:space:]]*(\\*\\*)?[[:space:]]*($SPEEDWORD)"
WORD_THEN_RATIO="($SPEEDWORD)[^0-9]{0,20}[0-9][0-9,]*(\\.[0-9]+)?[[:space:]]*(x|×)"
CLAIM_RE="($RATIO_THEN_WORD)|($WORD_THEN_RATIO)"

# A TARGET says what we WANT; a CLAIM says what we GOT. Only the second needs a
# receipt, because only the second is a report about the past. Taken verbatim
# from check_no_claim_literals.sh so the two guards share one dialect for what
# counts as a bar — a second, drifting definition of "target" would be worse
# than either.
TARGET_RE='([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|FAIL:|>=|<=|[><] *[0-9])'

# The citation token, and the window it may appear in.
#
# WINDOW = the claim's line, plus three lines either side. Three is not a round
# number chosen for looking careful: it is the shortest window that covers the
# three shapes citations actually take in this repo's markdown — in-row for a
# table, in the following sentence for prose, and on the line under a fenced
# block. Both edges are pinned by the case table (rows 9 and 10), so widening it
# later requires re-running the table rather than re-reading this comment.
RECEIPT_RE='evidence/[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*'
WINDOW=3

# Resolve every receipt token in a block of text; print each as
# "<path> <exists|missing>". Trailing markdown punctuation is stripped, so a
# citation inside `[...](../evidence/x/y.json)` or backticks resolves.
resolve_citations() {
    local root="$1"
    local text="$2"
    local tok p
    printf '%s\n' "$text" | grep -oE "$RECEIPT_RE" 2>/dev/null | while IFS= read -r tok; do
        p="${tok#"${tok%%evidence/*}"}"
        p=$(printf '%s' "$p" | sed 's/[])`",.;:]*$//')
        if [ -e "$root/$p" ]; then printf '%s exists\n' "$p"; else printf '%s missing\n' "$p"; fi
    done
}

# scan_file <root> <relpath> -> one record per claim:
#     <relpath>:<line>:<uncited|dangling>:<text>
# A cited claim emits nothing. Factored out precisely so --selftest can drive it
# over a fixture tree instead of over the real repo — a case table that can only
# exercise the regex, and never the window or the existence check, would leave
# the two halves that are actually hard completely unproven.
scan_file() {
    # SEPARATE STATEMENTS, NOT `local a=$1 b=$a/x`. bash declares every name in
    # a single `local` before it evaluates any right-hand side, so the second
    # assignment reads an unset variable and `set -u` aborts the function —
    # which the case table caught as five rows reporting `none` (a silent PASS
    # shape) while the loop had never executed. Proven, not remembered:
    #   bash -c 'set -u; f(){ local a="$1" b="$a/x"; echo "$b"; }; f hi'
    #   -> a: unbound variable
    local root="$1"
    local rel="$2"
    local f="$root/$rel"
    [ -f "$f" ] || return 0
    local total; total=$(wc -l < "$f")
    local n text lo hi block cites
    while IFS= read -r n; do
        [ -n "$n" ] || continue
        text=$(sed -n "${n}p" "$f")
        # Kept on separate lines: bashrs reads an arithmetic expansion sharing a
        # line with `[ ]` as unescaped parens in a test (SC1028), and
        # check_shell_lint_ratchet.sh counts error LINES, so a false positive
        # still moves a shrink-only baseline.
        lo=$((n - WINDOW))
        if [ "$lo" -lt 1 ]; then lo=1; fi
        hi=$((n + WINDOW))
        if [ "$hi" -gt "$total" ]; then hi="$total"; fi
        block=$(sed -n "${lo},${hi}p" "$f")
        cites=$(resolve_citations "$root" "$block")
        if printf '%s\n' "$cites" | grep -q ' exists$'; then
            continue                                   # cited, and it resolves
        elif printf '%s\n' "$cites" | grep -q ' missing$'; then
            printf '%s:%s:dangling:%s\n' "$rel" "$n" "$text"
        else
            printf '%s:%s:uncited:%s\n' "$rel" "$n" "$text"
        fi
    done < <(grep -nE "$CLAIM_RE" "$f" 2>/dev/null \
             | grep -vE "$TARGET_RE" \
             | cut -d: -f1)
}

# ---------------------------------------------------------------- selftest --
if [ "${1:-}" = "--selftest" ]; then
    TD=$(mktemp -d) || exit 2
    trap 'rm -rf "${TD:?}"' EXIT
    mkdir -p "$TD/docs" "$TD/evidence/real"
    printf '{}\n' > "$TD/evidence/real/receipt.json"
    t=0; f=0

    row() { # row <name> <want:none|uncited|dangling> <heredoc-file>
        local name="$1" want="$2" file="$3" got
        got=$(scan_file "$TD" "$file" | head -1 | cut -d: -f3)
        [ -n "$got" ] || got=none
        t=$((t + 1))
        if [ "$got" = "$want" ]; then
            printf '  ok    %-9s %s\n' "$want" "$name"
        else
            printf '  FAIL  want %-9s got %-9s %s\n' "$want" "$got" "$name"
            f=$((f + 1))
        fi
    }

    printf -- '--- case table -----------------------------------------------------\n'

    # MUST FLAG — a speed comparison with nothing behind it.
    printf 'apr is ~4.91x faster on this workload.\n'            > "$TD/docs/r1.md"
    row 'bare ratio-then-word claim'          uncited docs/r1.md
    printf 'Algorithm choice: 21x speedup (OLS vs SGD for small p)\n' > "$TD/docs/r2.md"
    row 'word-then-ratio claim'               uncited docs/r2.md
    printf '| Cold start | apr **~1500× faster** (ratio 0.0007) |\n'  > "$TD/docs/r3.md"
    row 'markdown table row, unicode ×'       uncited docs/r3.md

    # MUST NOT FLAG — not this class at all.
    printf 'Target: 2x faster than the current release.\n'       > "$TD/docs/r4.md"
    row 'a target states a bar, not a result' none    docs/r4.md
    printf 'Compression ratio: 24 bits -> 8 bits = 3× smaller\n'  > "$TD/docs/r5.md"
    row 'arithmetic, no speed word'           none    docs/r5.md
    printf 'let slow = mock_candidate(4000, 1.0); // 4000ms latency\n' > "$TD/docs/r6.md"
    row 'a latency literal in sample code'    none    docs/r6.md
    printf 'The input is a 3x3 matrix of f32 values.\n'          > "$TD/docs/r7.md"
    row 'a dimension, not a comparison'       none    docs/r7.md

    # THE CITATION ITSELF — the half a regex-only table cannot reach.
    printf 'apr is ~4.91x faster — see evidence/real/receipt.json\n' > "$TD/docs/r8.md"
    row 'cited on the claim line'             none    docs/r8.md

    { printf 'apr is ~4.91x faster than the reference.\n'
      printf '\n\n'; printf 'Receipt: evidence/real/receipt.json\n'; } > "$TD/docs/r9.md"
    row 'cited 3 lines below (window edge, in)'  none  docs/r9.md

    { printf 'apr is ~4.91x faster than the reference.\n'
      printf '\n\n\n'; printf 'Receipt: evidence/real/receipt.json\n'; } > "$TD/docs/r10.md"
    row 'cited 4 lines below (window edge, out)' uncited docs/r10.md

    printf 'apr is ~4.91x faster — see evidence/real/gone.json\n' > "$TD/docs/r11.md"
    row 'citation resolves to nothing'        dangling docs/r11.md

    # DISCRIMINATION AT THE FILE LEVEL: a documentation file with no speed
    # comparison in it must produce no record at all. Without this row, every
    # row above still passes if scan_file flagged every line it saw.
    { printf '# Installation\n\nRun the installer, then check the version.\n'
      printf 'The binary is self-contained and needs no runtime.\n'; } > "$TD/docs/r12.md"
    n=$(scan_file "$TD" docs/r12.md | grep -c . || true)
    t=$((t + 1))
    if [ "$n" -eq 0 ]; then printf '  ok    %-9s %s\n' none 'prose with no claim is silent'
    else printf '  FAIL  want none      got %s record(s) %s\n' "$n" 'prose with no claim'; f=$((f + 1)); fi

    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    [ "$f" -eq 0 ] || { printf 'SELFTEST FAILED\n'; exit 1; }
    printf 'SELFTEST PASSED\n'
    exit 0
fi

# ---------------------------------------------------------------- universe --
# Same construction as check_no_claim_literals.sh, deliberately: tracked UNION
# working tree (a `git ls-files`-only universe hands an untracked file a free
# pass — the documented shape, four instances in this epic), depth-tolerant
# globs (`a/**/b` requires at least one intervening segment, which once hid 1045
# tracked files from the sibling), minus docs/specifications/ for the reason in
# the header.
mapfile -t SRC < <(
    { git ls-files 'README.md' 'book/**/*.md' 'book/*.md' \
                   'docs/**/*.md' 'docs/*.md' 2>/dev/null
      find README.md book docs -type f -name '*.md' 2>/dev/null
    } | LC_ALL=C sort -u \
    | grep -vE '^docs/specifications/')

printf -- '--- performance claims cite their receipts (PERF-010) ---------------\n'

# VACUITY. A universe that collapsed sweeps clean and reads as a pass — the
# exact failure this epic keeps finding. 511 files at the time of writing; a
# floor of 100 catches a broken glob without breaking on a doc reorganisation.
if [ "${#SRC[@]}" -lt 100 ]; then
    printf 'FAIL  universe collapsed to %s file(s), expected 100+. The scan is\n' "${#SRC[@]}"
    printf '      broken, not the docs. Fix the globs rather than this number.\n'
    exit 1
fi
printf 'universe: %s user-facing markdown file(s)\n' "${#SRC[@]}"

records=""
for rel in "${SRC[@]}"; do
    r=$(scan_file "." "$rel")
    [ -n "$r" ] && records="${records}${r}"$'\n'
done
records=$(printf '%s' "$records" | grep -v '^$' || true)

if [ "${1:-}" = "--update" ]; then
    {
        printf '# Speed comparisons in README.md, book/ and docs/ that cite no\n'
        printf '# receipt, as of the day check_perf_claims_cite_receipts.sh was\n'
        printf '# written. Recorded, not blessed.\n#\n'
        printf '# THE RATCHET: this file may only SHRINK. Remove a line by deleting\n'
        printf '# the claim, or by citing the evidence/ receipt that produced it\n'
        printf '# within three lines of it. A line may only LEAVE this file: it is\n'
        printf '# compared against origin/main by check_perf_claims_cite_receipts.sh\n'
        printf '# and by check_baseline_ratchets.sh, so an append is REFUSED, not\n'
        printf '# merely discouraged.\n#\n'
        printf '# A DANGLING citation is never baselined — see the guard header.\n'
        printf '%s\n' "$records" | grep -v '^$' \
            | awk -F: '$3=="uncited"{print $1":"$2}' | LC_ALL=C sort -u
    } > "$BASELINE"
    printf 'baseline written: %s uncited claim(s)\n' \
        "$(grep -cvE '^\s*(#|$)' "$BASELINE" || true)"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL  %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi

rc=0; known=0; new=0; dang=0
while IFS= read -r rec; do
    [ -n "$rec" ] || continue
    loc="$(printf '%s' "$rec" | cut -d: -f1-2)"
    why="$(printf '%s' "$rec" | cut -d: -f3)"
    txt="$(printf '%s' "$rec" | cut -d: -f4- | cut -c1-110)"
    if [ "$why" = "dangling" ]; then
        printf 'FAIL  %s cites a receipt that does not exist\n      %s\n' "$loc" "$txt"
        dang=$((dang + 1)); rc=1
    elif grep -qxF "$loc" "$BASELINE" 2>/dev/null; then
        known=$((known + 1))
    else
        printf 'FAIL  %s uncited speed comparison\n      %s\n' "$loc" "$txt"
        new=$((new + 1)); rc=1
    fi
done <<< "$records"

printf 'known (baselined, must shrink): %s   new: %s   dangling: %s\n' "$known" "$new" "$dang"

# Stale baseline entries: a location that no longer carries a claim must be
# pruned, or the ratchet silently re-admits a claim at that line later.
stale=0
while IFS= read -r loc; do
    [ -n "$loc" ] || continue
    case "$loc" in '#'*) continue ;; esac
    printf '%s\n' "$records" | grep -q "^${loc}:" || stale=$((stale + 1))
done < "$BASELINE"
if [ "$stale" -gt 0 ]; then
    printf 'REPORT %s stale baseline location/s no longer carry a claim — prune\n' "$stale"
    printf '       with --update, or the ratchet re-admits a claim there later.\n'
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "THE RATCHET: this file may only SHRINK."
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
RATCHET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${RATCHET_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "$RATCHET_ROOT" scripts/perf_claim_citation_baseline.txt set || rc=1

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no NEW uncited speed comparison, and every citation resolves.\n'
else
    printf 'FAIL  a performance number is evidence only if something can prove how\n'
    printf '      it was measured. Cite the evidence/ receipt that produced it\n'
    printf '      within three lines, or delete the number. If it is a TARGET\n'
    printf '      rather than a result, say so — a bar needs no receipt.\n'
fi
exit "$rc"
