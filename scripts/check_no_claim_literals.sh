#!/usr/bin/env bash
# check_no_claim_literals.sh — a CLAIM about measured reality may not be a
# literal on a surface a user reads.
#
# WHY THIS EXISTS AND check_no_fabricated_baselines.sh DOES NOT COVER IT.
# That guard looks for competitor-named VARIABLE ASSIGNMENTS and JSON KEYS —
# `OLLAMA_TPS=163`, `"ollama_baseline": 163`. Both of the fabrications found on
# 2026-08-25 were neither:
#
#   crates/apr-cli/src/commands/serve/server.rs:236
#       "Performance: 800+ tok/s (2.8x Ollama) with batched requests"
#   crates/apr-cli/src/commands/profile_print_hotspot.rs:159
#       "Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)"
#
# The first is a throughput comparison printed by a server that measured
# nothing — and `--batch` in fact HANGS on four concurrent requests. The second
# is a causal diagnosis fired on a threshold by a tool that never inspects
# sampling; it was quoted back as evidence for a root cause. Neither is an
# assignment, so neither was visible to the older guard.
#
# THE DISTINCTION THAT MAKES THIS TRACTABLE. A hardcoded number is not the
# problem; an unearned claim is.
#
#   TARGET      "2x Ollama should be ~1025 tok/s"   in a test    -> allowed
#               a bar to meet. Nobody reads it as a result.
#   CLAIM       "Performance: 800+ tok/s (2.8x Ollama)"  printed -> banned
#               a user reads it as a measurement of what they just got.
#
# So the surface decides, not the literal. Tests, benches, examples and fixtures
# may state targets. Anything a user sees — println!/eprintln! on a shipped
# path, and doc comments on public API — may not.
set -euo pipefail

BASELINE="scripts/claim_literal_baseline.txt"

# ARGUMENT VALIDATION, BEFORE ANY WORK. Every mode below is selected by an
# equality test against "${1:-}", so without this a typo -- `--self-test`,
# `--selftests`, `--updated` -- falls through every one of them and runs the
# DEFAULT GATE, telling a caller who believed they had run the case table that
# it PASSED. A guard that answers a question it was not asked is the same
# defect class as a guard that cannot fail. (The sibling guard grew this case
# for exactly that reason.)
case "${1:-}" in
    ''|--selftest|--update) : ;;
    *)
        printf 'unknown arg: %s\n' "$1" >&2
        printf 'usage: check_no_claim_literals.sh [--selftest|--update]\n' >&2
        exit 2
        ;;
esac

# ONE definition of "this number cites a receipt", shared with the POSITIVE
# guard (check_perf_claims_cite_receipts.sh). See drop_receipted() below and
# the library's own header. Sourced with `|| exit`, never plain: the library is
# option-neutral and reports by return status, so a missing or broken library
# must be a loud failure rather than a guard that quietly stops exempting.
CLAIM_LIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/perf_claim_cite.sh
. "${CLAIM_LIT_ROOT}/scripts/lib/perf_claim_cite.sh" || exit 1

rc=0
printf -- '--- claim literals on user-facing surfaces ------------------------\n'

# A comparison against a named competitor, or a throughput figure, inside a
# string that is PRINTED. `\bx\b` after a number is the ratio idiom.
CLAIM_RE='(println!|eprintln!|write!|writeln!|format!|\.red\(\)|\.green\(\)|\.yellow\(\)|\.cyan\(\))'
# The comparator list also carries the I-12 `[X]` names. §0.1: a `[X]` figure is
# a third-party published claim about a third-party system — 36.9x over
# FasterTransformer, 23x over static batching, 1.8x over vLLM. It may inform a
# design choice and may NEVER appear in README.md, book/ or docs/. Importing
# someone else's number is the same defect as fabricating our own, with a
# better provenance story, so it is banned by the same guard rather than a
# second one that could rot separately.
#
# PERF-049 WIDENED RATIO_RE TWICE, AND THE GUARD COULD NOT SEE ITS OWN SUBJECT
# UNTIL IT DID. The pattern was `[0-9.]+x` followed IMMEDIATELY by a competitor
# name. Two spellings escaped it, and the registry had recorded this row as
# proven using ONE OF THEM:
#
#   (1) U+00D7. `2.93x Ollama` -> rc=1; `2.93× Ollama` -> rc=0. The multiplication
#       sign is the character actually PUBLISHED — the book's "851.8 tok/s =
#       2.93× Ollama", from a harness that never ran Ollama, is written with it.
#       §5's registered mutation for this guard is `add "2.93× Ollama" to book/`,
#       so running the mutation exactly as written left the guard GREEN. The row
#       was evidence for a gate that, on the input the row itself named, could
#       not fail.
#
#   (2) ONE INTERVENING WORD. `36.9x FasterTransformer` was RED and `36.9x over
#       FasterTransformer` was GREEN — and the second is the spelling §0.1 of
#       APR-PERF-GATE-001 itself uses for the `[X]` figures I-12 bans.
#
# THE GAP IS BOUNDED AT FIVE WORDS, AND THE BOUND WAS MEASURED RATHER THAN
# PICKED. Over the 6900-file universe the new-hit set is IDENTICAL at 5 and at
# 6; the last true positive to appear is at 5 ("the 8.2x performance gap between
# realizar and llama.cpp"). Widths 0..6 produce ZERO false positives here, so
# the bound is not what holds the false-positive rate down — the GAP WORD CLASS
# is, and it is the load-bearing half:
#
#   letters, with interior `.` or `-`, plus a <=3-letter abbreviation ending in
#   a dot. NOT digits, NOT `|`, NOT `,`, NOT `(`.
#
# So a markdown table row cannot be crossed (`| 2x | fast | torch |` stays
# green — the pipes are not gap words), a dimension cannot start one (`3x3
# matrix`, `2x2 grid of llama tiles`), and a sentence boundary stops it
# ("4x faster. Ollama uses ggml" stays green, because `faster.` is six letters
# and only an abbreviation may carry a trailing dot, which is what keeps
# `2.9x vs. Ollama` RED). Every one of those is a row in the case table; the
# no-trailing-dot rule was measured to change nothing in this tree, so it is
# free safety rather than a guess.
#
# The separator after the multiplier is `*` and not `+` so that `2.93×Ollama`,
# the tight typographic form, is caught too. It costs nothing: a gap word may
# not begin with a digit, so `1024x1024 torch` cannot be crossed either.
MULT_RE='(x|×)'
COMPETITOR_RE='(Ollama|ollama|llama\.cpp|llama|vLLM|vllm|PyTorch|torch|FasterTransformer|fastertransformer|SGLang|sglang|TensorRT|tensorrt|TGI|tgi|LMDeploy|lmdeploy|TurboMind|turbomind|Orca|static[[:space:]]+batching)'
RATIO_GAP_RE='(([A-Za-z]+([.-][A-Za-z]+)*|[A-Za-z]{1,3}\.)[[:space:]]+){0,5}'
# The LEFT boundary is not decoration. Without it `v1.8x release notes for
# llama` matched — a version string, three ordinary words, and a product name.
# It is the only false positive the case table found, and it found it twice
# over: blocking a preceding LETTER alone still matched at `.8x`, because the
# dot let the regex restart one character in. So a preceding letter, digit or
# DOT all disqualify, and `v1.8x` has no position left to match from.
RATIO_LEFT_RE='(^|[^0-9A-Za-z.])'
RATIO_RE="${RATIO_LEFT_RE}[0-9]+(\.[0-9]+)?${MULT_RE}[[:space:]]*${RATIO_GAP_RE}${COMPETITOR_RE}"
# THE QUANTIFIER WAS COUNTING THE WRONG DIGITS, AND IT MADE THIS PATTERN BLIND
# TO THE COMMONEST SPELLING OF A THROUGHPUT FIGURE.
#
# `[0-9]{2,}` had to match the two digits IMMEDIATELY LEFT of the unit. In
# `X.Y tok/s` -- one fractional digit, which is how essentially every rate in
# this tree is written -- the only digit adjacent to the space is `Y`, and the
# `.` in front of it gives `[0-9]{2,}` nowhere to start. So:
#
#     8.61 tok/s   CAUGHT  by accident, on the `61`
#     57.5 tok/s   MISSED
#     132.3 tok/s  MISSED   <- a three-digit rate, still invisible
#     100  tok/s   CAUGHT
#
# THE MEASURED ESCAPE, #2787 (crates/aprender-serve/src/quantize/batched_matmul.rs):
# the module doc stated `prefill **8.61 tok/s** against decode **7.76 tok/s**`
# on line 10 and an imported llama.cpp figure, `53.8-57.5 tok/s`, on line 12.
# The guard went RED naming line 10 ONLY. An engineer clearing exactly what the
# guard named would have shipped line 12, and did -- it was removed by review,
# not by this gate.
#
# THE WIDENING IS "AT LEAST TWO SIGNIFICANT DIGITS", NOT "ANY DIGITS", AND THE
# DIFFERENCE WAS MEASURED RATHER THAN ARGUED. Over the 6911-file universe:
#
#     [0-9]{2,}                     (before)   0 hits beyond the baseline
#     ([0-9]{2,}|[0-9]+\.[0-9]+)               68 hits
#     [0-9]+(\.[0-9]+)?             (any)      80 hits
#
# The 68 are the decimal class the old quantifier could not read, and they are
# recorded as aperture reveals -- see the ratchet at the tail of this file.
# Twelve of the eighty are BARE SINGLE DIGITS, and that class is deliberately
# left OUT: two of them are `reports `0 tok/s` for a perfectly healthy server`,
# prose ABOUT a number rather than a claim, and a bare `N tok/s` carries the
# worst claim-to-prose ratio of anything measured here. A rate worth stating is
# stated to two figures.
#
# THE LEFT BOUNDARY is the same one RATIO_RE carries, and for the same reason:
# without it the regex can restart one character into a version-like string.
# It was measured to change NOTHING in this tree (the 68-hit set is identical
# with and without it), so it is free safety rather than a guess.
#
# THE UNIT LIST IS DELIBERATELY NOT WIDENED. Adding `tokens/s|tok/sec` was
# measured too: it produces exactly four more hits, all four of them
# `50us/token = 20,000 tokens/sec` -- a unit-conversion identity in a doc
# comment, which is arithmetic, not a claim. Four hits, four false. A
# sub-pattern whose only new hits are false is not evidence of a wider defect
# class.
TPUT_RE="${RATIO_LEFT_RE}([0-9]{2,}|[0-9]+\.[0-9]+)\+?[[:space:]]*tok/s"

# A PLACEHOLDER THAT HAS BEEN GIVEN A UNIT IS READ AS A MEASUREMENT. PERF-010's
# second half, and a class RATIO_RE above has nothing to say about. `Throughput:
# XX tok/s` in a shipped chapter does not read to a user as "we have not
# measured this yet"; it reads as a table someone forgot to finish, and the
# number that eventually fills it inherits whatever trust the surrounding prose
# had already earned.
#
# THE BINDING TO A UNIT IS THE ENTIRE DESIGN, and skipping it would have shipped
# a guard that is pure noise. In THIS repo `[X]` is overwhelmingly a MARKDOWN
# CHECKBOX -- `- [X] APPROVED for Production`, `| Section 9 | [X] PASS / [ ] FAIL
# |` -- and a bare ban on the literal would have been born RED against ten
# checked boxes in docs/qa/ and zero defects. (Worth stating plainly because the
# collision is with the spec's own vocabulary: APR-PERF-GATE-001 uses `[X]` as a
# PROVENANCE TAG meaning "external claim about a third-party system", which is
# the class RATIO_RE handles. Neither of those is a placeholder. Three meanings,
# one spelling.)
#
# So a placeholder counts only where a figure would go: immediately before a
# throughput unit, a memory unit, a latency unit, or a ratio bound to a speed
# word. Every checkbox form fails that test, and the case table pins six of them
# as must-not-match controls.
#
# Zero live instances today, and that is stated rather than hidden: this is a
# RATCHET AGAINST A SHAPE, not a cleanup. The mutation table is therefore the
# only evidence it works, since the tree cannot supply a true positive.
#
# Leading boundary is `(^|[^A-Za-z0-9_])` rather than `\b` -- `\b` is a GNU
# extension this file does not otherwise rely on, and the explicit form is what
# keeps `MAXX tok/s` (a field name) from matching.
PLACEHOLDER_TOK='(\[X+\]|\[TBD\]|\[TODO\]|\[N\]|TBD|TODO|XX+)'
PERF_UNIT_RE='(tok/s|tokens/s|tok/sec|(ms|GB|MB)([^A-Za-z0-9]|$)|(x|×)([[:space:]]+(faster|slower|speedup)|[[:space:]]*$)|%[[:space:]]*(faster|slower|speedup))'
PLACEHOLDER_RE="(^|[^A-Za-z0-9_])${PLACEHOLDER_TOK}[[:space:]]*${PERF_UNIT_RE}"

# A RATIO PUBLISHED AS A TABLE CELL IS INVISIBLE TO RATIO_RE, BY CONSTRUCTION.
#
# RATIO_RE needs a competitor NAME within five gap words of the number, and a
# `|` is deliberately not a gap word (`| 2x | fast | yes | torch |` must stay
# green — see the header). That rule is right for prose and it makes a
# comparison TABLE unreadable, because a table puts the competitor in the
# HEADER and the ratio in a cell three columns away:
#
#   | band | llama agg | apr agg | **agg ratio** | llama dec | apr dec | dec ratio |
#   | c=16 | 1120.8    | 108.4   | **0.097×**    | 71.2      | 110.6   | **1.554×** |
#
# Verbatim from docs/benchmarking-gate-spec.md:49-54. Those are the withdrawn
# ratios PP-LLAMA-001 §2.1 says "appear nowhere else", published in a document
# this guard reported PASS over: the six lines it DID name are the PROSE around
# the table, not the table.
#
# So the header supplies what the row cannot: a decimal ratio inside a `|`
# cell, in a table whose preceding TABLE_HEADER_WINDOW lines carry a competitor
# name in a cell of their own. Both halves are load-bearing:
#
#   THE DECIMAL is what separates the must-fire row above from the must-not-fire
#   `| 2x | fast | yes | torch |` — which carries a competitor in a cell and is
#   still green, because `2x` is a column label, a dimension or a bare multiple,
#   and a measured ratio is written to at least one fractional digit. The case
#   table runs BOTH under the SAME competitor header, so the decimal rule is the
#   only variable between them.
#
#   THE HEADER is what keeps this off every markdown table in the tree. A cell
#   ratio with no competitor named above it is a compression factor, a speedup
#   against ourselves, or a column of scaling efficiencies — none of which is
#   the comparator claim I-12/PP-12 bans.
#
# The left boundary is the one RATIO_RE carries, for the same reason: without it
# `| v1.8x release |` matches at `.8x`. It is spelled as an optional
# "cell prefix ending in a non-alphanumeric, non-dot character" so that a cell
# with no leading space (`|0.097×|`) still matches.
TABLE_HEADER_WINDOW=10
TABLE_CELL_RATIO_RE="\\|([^|]*[^0-9A-Za-z.|])?[0-9]+\\.[0-9]+[[:space:]]*${MULT_RE}[^|]*\\|"
TABLE_HEADER_COMP_RE="\\|[^|]*${COMPETITOR_RE}[^|]*\\|"

# table_ratio_hits_in <root> <md files...> -> `file:line:text` per table hit.
#
# ONE implementation, shared by the sweep and by the case table below, for the
# reason causal_literals_in states: two copies of an extraction rule drift, and
# the drift is invisible precisely because the table keeps passing against its
# own copy.
#
# `grep -H` is not decoration: with a SINGLE file argument grep omits the
# filename, and the case table feeds exactly one file. Without -H every selftest
# row would parse the LINE NUMBER as the path and the header lookup would read
# nothing — a table that passes while checking the wrong thing.
#
# The header block is CAPTURED and then matched, never `sed … | grep -q`: an
# early-exiting reader hands the producer SIGPIPE and pipefail reports 141
# though grep MATCHED. That exact shape has been a live fail-open here.
table_ratio_hits_in() {
    local root="$1"
    shift
    [ "$#" -gt 0 ] || return 0
    local rec f n lo hi block cands
    # The candidate scan runs INSIDE $root so the emitted paths are the ones the
    # caller passed, and the header lookup below can dereference them as
    # "$root/$f". Captured rather than piped: the loop body runs `sed` and
    # `grep`, and a `grep | while` would put those in a subshell whose SIGPIPE
    # status pipefail then reports as a failure.
    cands=$( cd "$root" && grep -HInE "$TABLE_CELL_RATIO_RE" "$@" 2>/dev/null || true )
    [ -n "$cands" ] || return 0
    while IFS= read -r rec; do
        f="${rec%%:*}"
        n=$(printf '%s' "$rec" | cut -d: -f2)
        case "$n" in '' | *[!0-9]*) continue ;; esac
        hi=$((n - 1))
        [ "$hi" -ge 1 ] || continue
        lo=$((n - TABLE_HEADER_WINDOW))
        if [ "$lo" -lt 1 ]; then lo=1; fi
        block=$(sed -n "${lo},${hi}p" "$root/$f" 2>/dev/null)
        if grep -qE "$TABLE_HEADER_COMP_RE" <<< "$block"; then
            printf '%s\n' "$rec"
        fi
    done <<< "$cands"
}

# A TARGET says what we WANT; a CLAIM says what we GOT. Only the second lies.
# Lines that name themselves a target, a threshold or a comparison operator are
# stating a bar, and a bar is allowed to be a constant — that is what a bar IS.
#
# THE MASK WAS A BARE SUBSTRING TEST OVER THE WHOLE LINE, AND THAT IS A
# FAIL-OPEN ON ITS COMPLEMENT.
#
#     ([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|
#      FAIL:|>=|<=|[><] *[0-9])
#
# `[Rr]equire` matches the substring `required`, anywhere on the line, in any
# grammatical role. So this — verbatim from
# docs/specifications/apr-mcp-server-spec.md:278 —
#
#     Q4_K_M is already the format that establishes aprender's 1.43× Ollama
#     parity (on Qwen2.5-1.5B Q4_K_M) — no new kernels required.
#
# is a live comparator claim, matches RATIO_RE, and was MASKED, because the
# sentence happens to end in the word "required". Measured over
# docs/specifications/ at the moment the universe was widened: 140 of 467 raw
# matches were masked this way, by `require`, `expect` or `spec ` appearing
# somewhere in an ordinary English sentence.
#
# THE FIX IS POSITIONAL, NOT LEXICAL. A target word earns the mask when it is
# doing the job of a target: BOUND TO A FIGURE, or LABELLING A TABLE CELL.
#
#   (T1) a whole-cell label       `| Target | 100 tok/s |`
#   (T2) a label before a figure  `Target: 192 tok/s`, `Expected throughput: ~17
#                                 tok/s`, `Target (batched): 50-80 tok/s`
#   (T3) a qualifier after one    `2x Ollama target`, `400 tok/s target`,
#                                 `50 tok/s decode target`
#   (T4) a comparison operator    `>= 10 tok/s`, `> 100`
#   (T5) a verdict prefix         `PASS:`, `FAIL:`
#
# and it does NOT earn it merely by appearing somewhere on the line. `spec ` /
# `SPEC` are dropped outright: a document is not a bar, and "as the spec says"
# masked whatever followed it.
#
# BOTH DIRECTIONS ARE NEEDED, AND THE SECOND WAS FOUND BY MEASURING RATHER THAN
# BY READING. A first draft required the label to come BEFORE the number, which
# is how a bar is usually written -- and it unmasked 22 lines of the form
# `/// PAR-108: Key optimization for 2x Ollama target`, where the word sits
# AFTER the figure it qualifies. Those are bars, and reddening them would teach
# people to stop naming their targets. So the rule is a PROXIMITY rule in both
# directions, not an ordering rule.
#
# THE BOUND IS 20 BYTES AND IT IS NOT A ROUND NUMBER FOR LOOKING CAREFUL. The
# line this tightening exists for is
#
#   ... aprender's 1.43× Ollama parity (on Qwen2.5-1.5B Q4_K_M) — no new
#   kernels required. Hugging Face canonical: `Qwen/Qwen3-Coder-30B-...`
#
# where `required` is 25 bytes from the nearest digit on its left and 31 from
# the nearest on its right. Every real bar measured in this tree sits within 16
# (`50 tok/s decode target`, the longest). 20 separates them with margin at both
# ends, and both ends are rows in the case table, so moving it means re-running
# the table rather than re-reading this comment.
#
# `[^|` in both patterns is load-bearing: a proximity window may not cross a
# markdown table cell, or a `Target` column three cells away would mask an
# unrelated ratio in the same row. The whole-cell form (T1) is what puts that
# legitimate case back.
#
# THE `AFTER` FORM CARRIES A SHORTER VOCABULARY THAN THE OTHERS, AND THE
# DIFFERENCE IS THE RESIDUAL THIS TIGHTENING LEFT BEHIND.
#
# (T3) exists for `2x Ollama target`, `400 tok/s target`, `50 tok/s decode
# target` -- a NOUN naming the bar, sitting after the figure it qualifies.
# `expect` and `require` do not work that way in English. Placed after a number
# they are almost always a VERB in an ordinary sentence, and the sentence is a
# claim:
#
#     apr sustains 851.8 tok/s = 2.93x Ollama, as expected.
#     225 tok/s on the 4090, as required.
#
# Both are comparator/throughput claims. Both were GREEN, because `expected` /
# `required` sat inside the 20-byte window after a digit. The mask was doing the
# opposite of its job: `TARGET_AFTER_RE` was built to keep a NAMED BAR readable,
# and it was reading "and that is what we predicted" as a bar.
#
# Before the figure the two words DO label one -- `the release requires 100
# tok/s`, `expect >= 10 tok/s` -- so they stay in TARGET_BEFORE_RE, and a
# whole-cell `| Expected |` header stays in TARGET_CELL_RE. Only the AFTER form
# narrows, to the three nouns that actually name a bar. `claim_as_expected` /
# `claim_as_required` below are the two lines above, verbatim.
#
# AND THE AFTER WINDOW MAY NOT CROSS SENTENCE PUNCTUATION. `[^|]{0,20}` let the
# window run through `.`, `,`, `;` and `)`, so a bar named in the NEXT clause
# masked a claim in this one. `[^|.,;)]` stops it at the clause boundary, which
# is the same argument `[^|` already made for the table cell: proximity is only
# evidence of a bar while the two are in one phrase.
TARGET_WORD='([Tt]argets?|[Tt]hresholds?|[Gg]oals?|[Ee]xpect(s|ed|ation|ations)?|[Rr]equire(s|d|ment|ments)?)'
TARGET_WORD_AFTER='([Tt]argets?|[Tt]hresholds?|[Gg]oals?)'
TARGET_CELL_RE="\\|[[:space:]]*${TARGET_WORD}[[:space:]]*[|:]"
TARGET_BEFORE_RE="${TARGET_WORD}[^|0-9]{0,20}[0-9]"
TARGET_AFTER_RE="[0-9][^|]{0,20}${TARGET_WORD_AFTER}"
TARGET_OP_RE='(^|[^A-Za-z])(PASS|FAIL):|>=|<=|[><] *[0-9]'
TARGET_RE="${TARGET_CELL_RE}|${TARGET_BEFORE_RE}|${TARGET_AFTER_RE}|${TARGET_OP_RE}"

# THE CAUSAL CLASS KEEPS THE OLD, BROAD MASK, and the split is deliberate
# rather than an oversight left behind.
#
# (T1)-(T5) above are all defined RELATIVE TO A NUMBER -- "labels the figure",
# "qualifies the figure", "operator before a number". A causal claim carries no
# number at all (that is the whole reason DIAGNOSIS_RE exists), so the
# positional forms are undefined for it. The tightening was measured over the
# numeric classes, where the 140 masked matches were; it is not transferred to
# a class where it was neither measured nor meaningful.
#
# RESIDUAL, STATED RATHER THAN LEFT TO BE FOUND: a fabricated diagnosis whose
# sentence contains "expect", "require" or "spec " is still masked. That is the
# behaviour on origin/main, unchanged by this commit, and it is a separate
# ticket rather than an unmeasured widening smuggled in beside this one.
TARGET_PROSE_RE='([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|FAIL:|>=|<=|[><] *[0-9])'

# A CAUSAL claim is a second class, and the ratio/throughput patterns cannot see
# it — it carries no number at all. PERF-014: `apr profile` printed
#   "Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)"
# on a threshold, from a tool that never inspects sampling, D2H, or dispatch. It
# was quoted back in an investigation as evidence for a root cause.
#
# A tool may report a magnitude it measured. It may not name a cause it did not
# determine. The tell is an imperative or an attribution in a printed literal.
# Narrow deliberately. `blame` is a feature name (git blame) and "perturbation
# caused by pruning" in a doc comment is a mathematical fact, not a fabricated
# diagnosis — the first draft flagged 22 lines, nearly all of them prose. What
# is banned is a tool telling the user what to go investigate about a number it
# just printed: an IMPERATIVE aimed at the reader.
#
# PERF-016 WIDENED THIS TWICE OVER, AND THE SECOND HALF IS THE LOAD-BEARING ONE.
#
#   (a) SPELLINGS. `likely using` and `likely due to` were not in the list. The
#       audit found `Critical - likely using wrong backend or naive
#       implementation` shipped in crates/apr-cli/src/commands/profile.rs, so
#       the omission was not hypothetical.
#
#   (b) THE CAUSAL CLASS NO LONGER REQUIRES A PRINT MACRO ON THE SAME LINE.
#       PERF-014's literal was caught only by the accident that a `.red()` sat
#       on it. Move the identical string ONE INDIRECTION AWAY -- a `match` arm
#       in a helper the printer calls -- and CLAIM_RE sees nothing at all. That
#       is exactly where the PERF-016 audit found the next one: a `PerfGrade`
#       description arm, printed verbatim by `print_perf_grade_section`, naming
#       a wrong backend and a naive implementation as the cause of a low
#       efficiency score that nothing on that path can attribute. This guard
#       was GREEN on it, which is why PERF-016 had to be a MANUAL audit.
#
#       So for .rs the causal patterns now apply to every double-quoted literal
#       on a non-comment line, print macro or not. A cause reaching the user
#       through a `const`, a `match` arm or a returned `&'static str` reaches
#       the user identically. Comments stay exempt (the case table's
#       `// investigate sampling sync later` control) and .md prose stays out,
#       because DIAGNOSIS_RE over English flags mathematics, not fabrication.
#
# THE CLASS SPLITS IN TWO, AND THE SPLIT IS WHAT MAKES THE WIDENING SURVIVE THE
# TREE. An ATTRIBUTION says what caused it; an IMPERATIVE says where to go look.
# Only the first asserts an answer, so only the first is a defect wherever it is
# written. Widening the imperative to every literal was tried and MEASURED: over
# 7329 shipped files it produced exactly two hits,
#
#   aprender-data   "Remove or investigate constant columns: {:?}"
#   aprender-gpu    "Performance degraded by {:.1}% under load - investigate bottlenecks"
#
# and both name the thing the tool had just measured. A sub-pattern whose only
# hits in the new scope are false is not evidence of a wider defect class, and
# baselining two correct lines to keep a pattern would be the permission slip
# this file's own ratchet exists to refuse.
ATTRIBUTION_RE='(likely (caused by|due to|using)|probably (caused by|due to|using)|root cause is|suspect(ed)? cause|caused by (the|a|an|its) )'
DIAGNOSIS_RE="(investigate [a-z]|$ATTRIBUTION_RE)"

# Every double-quoted literal on a non-comment line of the given .rs files,
# emitted as `file:line:"literal"`, filtered to the causal class.
#
# ONE implementation, shared by the sweep and by the case table below. Two
# copies of an extraction rule drift, and the drift is invisible precisely
# because the table keeps passing against its own copy.
causal_literals_in() {
    [ "$#" -gt 0 ] || return 0
    #
    # `LIT` is the string-literal regex `"[^"]*"`, built from `\042` rather than
    # written out. Not decoration: that regex contains THREE double-quote
    # characters, and bashrs -- which this repo mandates over shellcheck --
    # counts quotes without parsing awk, so the literal form raises SC1078
    # "did you forget to close this double-quoted string?" against a script that
    # is correct. The `scripts/` lint ratchet is shrink-only, so leaving two
    # false errors behind is leaving them for someone else. Behaviour is
    # identical; the case table below re-proves it either way.
    awk '
        BEGIN { Q = "\042"; LIT = Q "[^" Q "]*" Q }
        /^[[:space:]]*(\/\/|\*|\/\*)/ { next }
        {
            rest = $0
            while (match(rest, LIT)) {
                print FILENAME ":" FNR ":" substr(rest, RSTART, RLENGTH)
                rest = substr(rest, RSTART + RLENGTH)
            }
        }
    ' "$@" 2>/dev/null | grep -E "$ATTRIBUTION_RE" | grep -vE "$TARGET_PROSE_RE"
}

# THE CITATION EXEMPTION, APPLIED AFTER THE MATCH.
#
# PP-LLAMA-001 §6/PP-12 states ONE rule: "a number in README.md / book/ /
# docs/ is legal iff it cites an evidence/ receipt path". That is a
# CONJUNCTION over this guard and check_perf_claims_cite_receipts.sh, and until
# now no single check encoded it. This guard implemented the "delete" half
# UNCONDITIONALLY, so PP-12's own must-not-fire fixture — a figure citing
# `evidence/…/receipt.r1.json` — was RED: citing a receipt beside `225+ tok/s`
# left the line a TPUT_RE hit and there was no spelling of it both guards
# accepted. The spec's rule was therefore unimplementable as written.
#
# So a hit is DROPPED when its own line, or the three lines either side, carry
# a token matching `evidence/<path>` THAT RESOLVES TO A FILE THAT EXISTS. The
# definition is not restated here: it is sourced from
# scripts/lib/perf_claim_cite.sh, the same file the positive guard now uses, so
# the two cannot drift.
#
# APPLIED AFTER THE MATCH, NOT AS PART OF IT, and that ordering is the whole
# design. A pattern that tried to require-or-exclude a citation inside the same
# regex would be unreadable and, worse, unable to check that the path RESOLVES.
# Matching stays a pure line test; the exemption is a second, dereferencing
# pass over the findings.
#
# THE THREE SHAPES A CITATION TAKES ARE ALL ACCEPTED, because §2.1 of the
# master spec writes figures in all three: in the SAME PIPE-TABLE ROW as the
# number, on the line AFTER it, and in the sentence before it.
#
# A DANGLING CITATION NEVER EXEMPTS. An uncited number is inherited debt —
# visible, recorded, ratcheted down. A citation pointing at a file nobody can
# open is an active forgery, and exempting it would make this guard strictly
# easier to satisfy than the rule it implements.
drop_receipted() { # drop_receipted <root> < findings-on-stdin
    local root="$1" rec f n
    while IFS= read -r rec; do
        [ -n "$rec" ] || continue
        f="${rec%%:*}"
        n=$(printf '%s' "$rec" | cut -d: -f2)
        # `claim_citation_exempts`, NOT `claim_line_is_receipted`: the exemption
        # is a CONJUNCTION of "cites a resolving evidence/ FILE" and "is on one
        # of the three surfaces PP-12 names". Applied without the second
        # conjunct it laundered `.rs` doc comments and shipped `println!`s,
        # where the reader never sees the path; applied with `-e` instead of
        # `-f` it laundered two live `2.93x Ollama` lines on the strength of a
        # bare DIRECTORY token. See scripts/lib/perf_claim_cite.sh's header.
        if claim_citation_exempts "$root" "$f" "$n"; then continue; fi
        printf '%s\n' "$rec"
    done
}

if [ "${1:-}" = "--selftest" ]; then
    t=0; f=0
    check() { # check <expect match|nomatch> <line>
        local want="$1" line="$2" got=nomatch
        if grep -qE "$CLAIM_RE" <<< "$line" \
           && grep -qE "$RATIO_RE|$TPUT_RE|$DIAGNOSIS_RE" <<< "$line" \
           && ! grep -qE "$TARGET_RE" <<< "$line" ; then got=match; fi
        t=$((t+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "$(printf '%s' "$line" | cut -c1-64)"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "$(printf '%s' "$line" | cut -c1-52)"; f=$((f+1)); fi
    }
    printf -- '--- case table ---\n'
    # MUST MATCH: claims a user reads as a result
    check match   'println!("Performance: 800+ tok/s (2.8x Ollama) with batched requests");'
    check match   'eprintln!("Achieves Ollama-parity: 100+ tok/s");'
    check match   'println!("{}", "851.8 tok/s = 2.93x Ollama".green());'
    # MUST NOT MATCH: targets, thresholds, and measured values
    check nomatch 'println!("M4 Parity Target: 192 tok/s");'
    check nomatch 'println!("(PASS: >= 10 tok/s)");'
    check nomatch 'println!("Threshold: 100 tok/s");'
    check nomatch 'println!("decode {:.1} tok/s", measured.decode);'   # derived, no literal number
    check nomatch 'let ollama_baseline = 163.0;'                        # an assignment — the OTHER guard owns this
    # PERF-014: a causal claim carries no number, so the ratio patterns are blind to it
    check match   'println!("Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)");'
    check match   'eprintln!("Slow decode — likely caused by KV cache thrashing");'
    check nomatch 'println!("Non-kernel time dominates this pass");'     # a magnitude, no cause
    check nomatch '// investigate sampling sync later'                   # a comment, not printed

    # PERF-016: the causal class, where NO print macro is on the line. Every row
    # here is invisible to check() above -- that is the whole point of the
    # widening, so the table runs the widened extractor instead.
    ct=0; cf=0
    CAUSAL_TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${CAUSAL_TD:?}"' EXIT
    check_causal() { # check_causal <match|nomatch> <one line of rust>
        local want="$1" line="$2" got=nomatch tmp out
        tmp="$CAUSAL_TD/probe.rs"
        printf '%s\n' "$line" > "$tmp"
        # No `| grep -q` here: an early-exiting reader plus pipefail hands the
        # pipeline the PRODUCER's SIGPIPE status and invents a failure. Capture,
        # then test the string.
        out=$(causal_literals_in "$tmp" || true)
        [ -n "$out" ] && got=match
        ct=$((ct+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "$(printf '%s' "$line" | cut -c1-64)"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "$(printf '%s' "$line" | cut -c1-52)"; cf=$((cf+1)); fi
    }
    printf -- '--- causal class (no print macro required) ---\n'
    # MUST FLAG. Row 1 is the live defect PERF-016 removed, verbatim.
    check_causal match   '            Self::F => "Critical — likely using wrong backend or naive implementation",'
    check_causal match   '    let r = "stall — the root cause is the sampling sync";'
    # DELIBERATE nomatch, and the reason is written down: PERF-014's own literal
    # is an IMPERATIVE, and it is caught by the CLAIM_RE sweep above (row 9),
    # which is where imperatives belong because a print site is what makes one a
    # claim. This row exists so that split is asserted rather than assumed.
    check_causal nomatch '        "Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)".red()'
    check_causal match   'const REASON: &str = "slow decode — likely caused by KV cache thrashing";'
    check_causal match   '    let m = if pct > 40.0 { "overhead — probably due to the argmax D2H" } else { "ok" };'
    check_causal match   'fn why() -> &str { "stall — caused by the graph replay path" }'
    # MUST NOT FLAG. These are the half that says the rule discriminates.
    check_causal nomatch '            Self::D => "Poor — significant optimization needed",'      # magnitude only
    check_causal nomatch '        "Non-kernel time dominates this pass — profile the host path to find out why".red()'
    check_causal nomatch '    // investigate sampling sync later'                                # comment
    check_causal nomatch '    /// The residual is likely caused by numerical drift.'             # doc prose
    check_causal nomatch '     * probably due to the accumulation order'                         # block comment body
    check_causal nomatch 'reason: format!("{} takes {:.1}% of total time — check for scalar fallback", h.name, h.percent),'
    check_causal nomatch 'let msg = "expect the root cause is recorded";'                        # TARGET_RE: a spec line
    check_causal nomatch 'let x = "ok"; // likely caused by the drift in run 3'                  # match outside every literal
    # VACUITY: a table that shrank to nothing would sweep clean.
    if [ "$ct" -lt 14 ]; then
        printf '  FAIL  causal table has %s row(s); at least 14 are required\n' "$ct"; cf=$((cf+1))
    fi
    printf '  %s causal case(s), %s failure(s)\n' "$ct" "$cf"

    # PERF-049: THE RATIO TABLE. Separate from check() above on purpose — check()
    # requires a Rust print macro on the line, and the two spellings that
    # escaped this guard were published in MARKDOWN, where there is no macro and
    # the .md sweep applies RATIO_RE directly. A table that could only reach the
    # macro path would have kept passing over the exact hole PERF-049 is about.
    #
    # This repo's claim patterns have been wrong FIVE times; every one was found
    # by a table and none by review. Re-run the table, do not re-read the regex.
    rt=0; rf=0
    check_ratio() { # check_ratio <match|nomatch> <one line of prose>
        local want="$1" line="$2" got=nomatch
        # Herestrings, not `printf | grep -q`: an early-exiting reader hands the
        # producer SIGPIPE and pipefail reports 141 though grep MATCHED.
        if grep -qE "$RATIO_RE|$TPUT_RE" <<< "$line" && ! grep -qE "$TARGET_RE" <<< "$line"; then
            got=match
        fi
        rt=$((rt+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "$(printf '%s' "$line" | cut -c1-66)"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "$(printf '%s' "$line" | cut -c1-54)"; rf=$((rf+1)); fi
    }
    printf -- '--- ratio class (markdown surface: no print macro) ---\n'
    # MUST MATCH -- the two spellings that escaped, and the neighbours that did not.
    check_ratio match   'apr sustains 2.93x Ollama on this workload.'          # ASCII, zero gap
    check_ratio match   'apr sustains 2.93× Ollama on this workload.'          # U+00D7, zero gap
    check_ratio match   '851.8 tok/s = 2.93× Ollama'                           # the PUBLISHED form, verbatim
    check_ratio match   '2.93×Ollama'                                          # tight typographic form
    check_ratio match   '36.9x FasterTransformer'                              # [X] figure, zero gap
    check_ratio match   '36.9x over FasterTransformer'                         # ONE gap word -- the spec's own spelling
    check_ratio match   '36.9× over FasterTransformer'                         # both defects at once
    check_ratio match   '23x over static batching'                             # gap + a two-word competitor
    check_ratio match   '1.8x vLLM'
    check_ratio match   '2.9x faster than Ollama'                              # TWO gap words
    check_ratio match   '2.9x vs. Ollama'                                      # abbreviation may carry its dot
    check_ratio match   '16x convergence gap vs PyTorch'                       # THREE gap words
    check_ratio match   'the 8.2x performance gap between realizar and llama.cpp'   # FIVE, the bound
    check_ratio match   '8x slower than llama.cpp'                             # integer ratio, self-deprecating
    check_ratio match   'Already 2.9x FASTER than Ollama'                      # I-12: illegal in either direction
    # THE TARGET_RE TIGHTENING, ASSERTED. Verbatim from
    # docs/specifications/apr-mcp-server-spec.md:278 as it stood at f21f437c2.
    # It matches RATIO_RE and was MASKED, because the sentence ends in the word
    # "required" and the old mask was a bare substring test over the whole line.
    # 140 of 467 raw matches under docs/specifications/ were masked this way.
    check_ratio match   "Q4_K_M is already the format that establishes aprender's 1.43× Ollama parity (on Qwen2.5-1.5B Q4_K_M) — no new kernels required."
    check_ratio match   "Q4_K_M is already the format that establishes aprender's 1.43x Ollama parity (on Qwen2.5-1.5B Q4_K_M) — no new kernels required."
    # ...and the bar it must NOT swallow with it. A target word LABELLING the
    # line or BOUND TO THE FIGURE still masks; a comparison operator still masks.
    check_ratio nomatch 'Target: 2x Ollama'
    check_ratio nomatch 'expect >= 10 tok/s'
    # THE `AFTER` NARROWING (claim_as_expected / claim_as_required). A verb of
    # confirmation after the figure is not a bar; it is the sentence that makes
    # the figure a claim. Both of these were GREEN.
    check_ratio match   'apr sustains 851.8 tok/s = 2.93x Ollama, as expected.'
    check_ratio match   '225 tok/s on the 4090, as required.'
    # ...and the bars they must not take with them, in BOTH orders, so the
    # narrowing is proved to be positional rather than a vocabulary deletion.
    check_ratio nomatch '2x Ollama target'
    check_ratio nomatch '400 tok/s target for the 1.5B model'
    check_ratio nomatch '50 tok/s decode target'
    check_ratio nomatch '| Expected | 100 tok/s |'
    check_ratio nomatch 'Threshold: 100 tok/s'
    check_ratio nomatch 'M4 Parity Target: 192 tok/s'
    check_ratio nomatch 'the release requires 100 tok/s on this host'
    # MUST NOT MATCH. These matter as much: a false positive here reds every PR
    # in the repository, which is a worse failure than the hole it closes.
    check_ratio nomatch 'a 3x3 matrix'
    check_ratio nomatch 'a 2x2 grid of llama tiles'                            # dimension, not a ratio
    check_ratio nomatch 'reshaped to 1024x1024 before the torch export'        # digits are not gap words
    check_ratio nomatch '| 2x | fast | yes | torch |'                          # a pipe is not a gap word
    check_ratio nomatch 'v1.8x release notes for llama'                        # version string
    check_ratio nomatch 'aprender v0.64.0 ships 2 torch-free crates'           # version + a bare count
    check_ratio nomatch 'Our matmul is 4x faster. Ollama uses ggml.'           # sentence boundary stops the gap
    check_ratio nomatch 'the 2x speedup we would need on six separate kernels before llama'  # SIX words > the bound
    check_ratio nomatch 'Target: 2x Ollama should be ~1025 tok/s'              # TARGET_RE -- a bar, not a claim
    check_ratio nomatch 'the decode ratio must be >= 1.0x llama.cpp'           # a threshold
    check_ratio nomatch 'Ollama runs at 2.9x the batch size'                   # competitor BEFORE the ratio
    check_ratio nomatch 'see section 2 for the llama loader'                   # no ratio at all
    # THE THROUGHPUT HALF (#2787). check_ratio applies RATIO_RE|TPUT_RE, so these
    # rows exercise TPUT_RE directly. The first two are the measured escape,
    # verbatim from crates/aprender-serve/src/quantize/batched_matmul.rs:10 and
    # :12 -- the guard named the first and was blind to the second, which is the
    # whole subject of the widening. Assert both, or a future narrowing back to
    # `[0-9]{2,}` leaves a table that still passes.
    check_ratio match   'prefill **8.61 tok/s** against decode **7.76 tok/s**'  # was CAUGHT (on the 61)
    check_ratio match   '53.8-57.5 tok/s prefill because it batches.'           # was MISSED -- the escape
    check_ratio match   '| GPU | 1.5B | 132.3 tok/s |'                          # 3 digits, still missed before
    check_ratio match   'Achieves Ollama-parity: 100+ tok/s'                    # integer + the `+` idiom
    check_ratio match   'sustained 851.8 tok/s on the 1.5B model'
    # MUST NOT MATCH. The bare single-digit class is OUT by decision, not by
    # accident -- see the header. Two live lines in this tree are prose ABOUT a
    # rate, and a guard that reds them teaches people to stop describing bugs.
    check_ratio nomatch 'reports `0 tok/s` for a perfectly healthy server'
    check_ratio nomatch 'the CPU SIMD path runs at ~5 tok/s'
    check_ratio nomatch 'PASS: >= 10 tok/s'                                     # TARGET_RE -- a bar
    check_ratio nomatch 'the tok/s column is empty'                             # a unit with no figure
    # VACUITY. A table that shrank to nothing would sweep clean, and both halves
    # must stay populated -- an all-positive table proves nothing about noise.
    if [ "$rt" -lt 49 ]; then
        printf '  FAIL  ratio table has %s row(s); at least 49 are required\n' "$rt"; rf=$((rf+1))
    fi
    # THE TIGHTENING MUST BE ASSERTED AGAINST TARGET_RE ALONE. The rows above
    # would still pass if a future edit restored the bare-substring mask and
    # narrowed RATIO_RE to compensate. This says which half moved.
    tr_probe="aprender's 1.43× Ollama parity — no new kernels required."
    if grep -qE "$TARGET_RE" <<< "$tr_probe"; then
        printf '  FAIL  TARGET_RE masks a bare `required` again -- apr-mcp-server-spec.md:278 reopens\n'; rf=$((rf+1))
    fi
    if ! grep -qE "$TARGET_RE" <<< 'M4 Parity Target: 192 tok/s'; then
        printf '  FAIL  TARGET_RE no longer reads a label bound to the figure\n'; rf=$((rf+1))
    fi
    if ! grep -qE "$TARGET_RE" <<< 'PASS: >= 10 tok/s'; then
        printf '  FAIL  TARGET_RE no longer reads a verdict prefix or an operator\n'; rf=$((rf+1))
    fi
    # The AFTER narrowing, asserted against TARGET_RE ALONE. Without these two
    # the rows above would still pass if a future edit put `expect`/`require`
    # back into TARGET_AFTER_RE and narrowed RATIO_RE to compensate.
    ta_probe='851.8 tok/s = 2.93x Ollama, as expected.'
    if grep -qE "$TARGET_RE" <<< "$ta_probe"; then
        printf '  FAIL  TARGET_RE masks `as expected` after a figure again\n'; rf=$((rf+1))
    fi
    if ! grep -qE "$TARGET_RE" <<< '400 tok/s target'; then
        printf '  FAIL  TARGET_RE no longer reads a bar NAMED after the figure (T3)\n'; rf=$((rf+1))
    fi
    if ! grep -qE "$TARGET_RE" <<< 'the release requires 100 tok/s on this host'; then
        printf '  FAIL  TARGET_RE no longer reads `requires` BEFORE the figure\n'; rf=$((rf+1))
    fi
    # The escape must be ASSERTED against TPUT_RE ALONE, not against the union:
    # if a future edit moved the coverage into RATIO_RE the rows above would
    # still pass while the throughput detector had gone back to sleep.
    if ! grep -qE "$TPUT_RE" <<< '53.8-57.5 tok/s prefill because it batches.'; then
        printf '  FAIL  TPUT_RE cannot read a one-fractional-digit rate -- #2787 reopens\n'; rf=$((rf+1))
    fi
    # The probe is a VARIABLE, not an inline herestring. bashrs -- which this
    # repo mandates over shellcheck -- reads the word `for` inside a herestring
    # that shares a line with `then` as a malformed for-loop (SC2135), and
    # scripts/ is gated on a SHRINK-ONLY bashrs error count, so one false error
    # here is one someone else has to triage. Same class as the \042 dance for
    # LIT and the OB/CB dance for markdown checkboxes; identical behaviour.
    fp_probe='reports `0 tok/s` for a perfectly healthy server'
    if grep -qE "$TPUT_RE" <<< "$fp_probe"; then
        printf '  FAIL  TPUT_RE widened past two significant digits -- see the header\n'; rf=$((rf+1))
    fi
    # The two spellings PERF-049 was opened for must be ASSERTED, not merely
    # present: if a future edit narrows RATIO_RE back to ASCII-adjacent, this
    # says so in the guard's own voice rather than in a silently shorter table.
    if ! grep -qE "$RATIO_RE" <<< '2.93× Ollama'; then
        printf '  FAIL  U+00D7 is not covered — the registered mutation cannot bite\n'; rf=$((rf+1))
    fi
    if ! grep -qE "$RATIO_RE" <<< '36.9x over FasterTransformer'; then
        printf '  FAIL  one intervening word still defeats the adjacency\n'; rf=$((rf+1))
    fi
    printf '  %s ratio case(s), %s failure(s)\n' "$rt" "$rf"

    # PERF-010: THE MARKDOWN SWEEP AS IT ACTUALLY RUNS. The ratio table above
    # applies RATIO_RE|TPUT_RE; the .md sweep applies RATIO_RE|TPUT_RE|
    # PLACEHOLDER_RE, and PLACEHOLDER_RE is a detector no row above touches.
    # A table that exercised only the ratio half is how the `[X]`-figure gap
    # survived being "covered by check_no_claim_literals.sh" in the spec's own
    # status table.
    mt=0; mf=0
    check_md() { # check_md <match|nomatch> <one line of markdown>
        local want="$1" line="$2" got=nomatch
        # `<<<` and capture-then-test, never `printf | grep -q`: an early-exiting
        # reader plus pipefail hands the pipeline the PRODUCER's SIGPIPE status
        # (141) even on a successful match. That exact shape was a live fail-open
        # on main.
        if grep -qE "$RATIO_RE|$TPUT_RE|$PLACEHOLDER_RE" <<< "$line" \
           && ! grep -qE "$TARGET_RE" <<< "$line"; then got=match; fi
        mt=$((mt+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "$(printf '%s' "$line" | cut -c1-64)"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "$(printf '%s' "$line" | cut -c1-52)"; mf=$((mf+1)); fi
    }
    printf -- '--- markdown class: [X] figures, comparators, placeholders ---\n'
    # THE SIX SHAPES §0.1 NAMES, through the combined .md pattern.
    check_md match   'aprender achieves 36.9x FasterTransformer throughput.'
    check_md match   'aprender achieves 36.9× over FasterTransformer.'
    check_md match   'Continuous batching gives 23× over static batching.'
    check_md match   'Throughput is 1.8× over vLLM on this workload.'
    check_md match   '| GPU (batched M=16) | Qwen 1.5B | ~850 tok/s | 2.93× Ollama |'
    # THE LIVE ONE. Verbatim from book/src/tools/apr-cli.md:81, which shipped
    # past this guard while it was green because "faster than" sat in the join.
    check_md match   '# Batched GPU mode (2.9x faster than Ollama)'
    check_md match   'realizar is 8.2x slower than llama.cpp on CPU.'
    check_md match   'c=4 TTFT = 256ms (10.7x vs llama.cpp 24ms).'
    # DISCRIMINATION. A ratio is not a claim just because a competitor is named
    # on the same line.
    check_md nomatch 'Ollama users will recognise the 3x3 convolution kernel.'
    check_md nomatch 'Compression ratio: 24 bits -> 8 bits = 3x smaller.'
    check_md nomatch 'Target: 2x Ollama throughput on the 1.5B model.'
    check_md nomatch 'Install Ollama first, then run the comparison harness.'
    # A SIZE RATIO AGAINST A NAMED COMPETITOR IS ALSO A CLAIM, and this row
    # says so deliberately. PERF-010 wrote it as a must-NOT-match on the theory
    # that only SPEED ratios lie, and enforced that with a closed connector
    # list (over|than|vs|versus|compared to). PERF-049 replaced that list with a
    # measured five-gap-word rule -- zero false positives over the 6900-file
    # universe -- and this line matches under it. Re-judged rather than
    # preserved: "3x larger than PyTorch" is an unreceipted comparative claim
    # about a third-party system, which is the class the guard's own failure
    # text describes. The connector list would also have missed "3x the memory
    # of PyTorch". Kept as an assertion so a future narrowing is loud.
    check_md match   'The .apr file is 3x larger than PyTorch pickle output.'
    # README.md's Performance table, VERBATIM as it stood at f21f437c2 (:234).
    # It is a bare throughput figure with no competitor on the line, so only
    # TPUT_RE can read it -- and it is the exact figure PP-LLAMA-001 §2.1
    # contradicts with the tree's own receipt (c=1 decode 103.26 tok/s on the
    # same host class). The ratio table's `100+ tok/s` row does not cover this
    # shape: three significant digits inside a pipe table with a `+` idiom.
    check_md match   '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |'
    check_md match   '| Qwen2.5-Coder 1.5B | Q4_K | 40+ tok/s | CPU (AVX2) |'
    # PLACEHOLDERS BOUND TO A UNIT.
    check_md match   'Throughput: XX tok/s'
    check_md match   '| Decode | [TBD] tok/s | 1.9 GB |'
    check_md match   'apr is [X]x faster than the previous release.'
    check_md match   'Speedup over the baseline: [TBD]×'
    check_md match   'Cold start latency: TODO ms'
    # ...AND THE CHECKBOXES THEY MUST NOT BE CONFUSED WITH. Without these rows a
    # bare `\[X\]` ban looks correct and is born red against docs/qa/.
    # The checkbox rows are built from variables rather than written inline.
    # bashrs -- which this repo mandates over shellcheck -- parses the `[ ]` and
    # `[X]` of a MARKDOWN checkbox as a shell test bracket and raises SC1028 /
    # SC2104 / SC1087 against rows that are correct English. Same class as the
    # \042 dance for LIT above, and the same remedy: keep the token out of the
    # source line. scripts/*.sh is gated on a SHRINK-ONLY bashrs error count, so
    # five false errors here are five someone else has to triage.
    # Even the ASSIGNMENTS are built from parts: bashrs flags a bare `'[ ]'`
    # string as SC2104 "missing space before ]". Brackets by name, then.
    OB='['; CB=']'
    CHECKED="${OB}X${CB}"; UNCHECKED="${OB} ${CB}"
    check_md nomatch "- $CHECKED APPROVED for Production"
    check_md nomatch "| Section 9 Tests/CI/Coverage | $CHECKED PASS / $UNCHECKED FAIL |"
    check_md nomatch "| 9.1.1 | All tests pass | 0 failures | $CHECKED | $UNCHECKED |"
    check_md nomatch '- Other modules: TBD from CI results'
    check_md nomatch 'TODO: add a benchmark harness for the graph module.'
    check_md nomatch 'The MAXX tok/s field is reserved.'
    # VACUITY: a table that shrank to nothing would sweep clean.
    if [ "$mt" -lt 26 ]; then
        printf '  FAIL  markdown table has %s row(s); at least 26 are required\n' "$mt"; mf=$((mf+1))
    fi
    # The placeholder half must be ASSERTED, not merely present: it is the only
    # detector here with ZERO live instances, so a silent narrowing would leave
    # a shorter table and no red.
    if ! grep -qE "$PLACEHOLDER_RE" <<< 'Throughput: XX tok/s'; then
        printf '  FAIL  a unit-bound placeholder is not covered\n'; mf=$((mf+1))
    fi
    if grep -qE "$PLACEHOLDER_RE" <<< '- [X] APPROVED for Production'; then
        printf '  FAIL  PLACEHOLDER_RE reds a markdown checkbox\n'; mf=$((mf+1))
    fi
    printf '  %s markdown case(s), %s failure(s)\n' "$mt" "$mf"

    # THE TABLE CLASS. Every row above is a LINE test, and the withdrawn band
    # table is not readable as one: the competitor is in the header and the
    # ratio is in a cell below it. So this table drives the real extractor
    # (table_ratio_hits_in) over a two-line fixture file, exactly as the causal
    # table drives causal_literals_in -- a table that could only reach a regex
    # would leave the header lookup, the 10-line window and the `-H` filename
    # completely unproven.
    tt=0; tf=0
    TABLE_TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${CAUSAL_TD:?}" "${TABLE_TD:?}"' EXIT
    check_table() { # check_table <match|nomatch> <header line> <row line> [name]
        local want="$1" header="$2" row="$3" name="${4:-}" got=nomatch out
        printf '%s\n%s\n' "$header" "$row" > "$TABLE_TD/probe.md"
        out=$(table_ratio_hits_in "$TABLE_TD" probe.md 2>/dev/null | grep -vE "$TARGET_RE" || true)
        [ -n "$out" ] && got=match
        tt=$((tt+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "${name:-$(printf '%s' "$row" | cut -c1-64)}"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "${name:-$(printf '%s' "$row" | cut -c1-52)}"; tf=$((tf+1)); fi
    }
    printf -- '--- table class: a ratio cell under a competitor header ---\n'
    TBL_HDR='| band | llama agg | subject agg | **agg ratio** | llama dec | subject dec | dec ratio |'
    # MUST FIRE. Verbatim from docs/benchmarking-gate-spec.md:54 -- one of the
    # four withdrawn band rows PP-LLAMA-001 §2.1 says "appear nowhere else",
    # published in a document this guard reported PASS over.
    check_table match   "$TBL_HDR" '| c=16 | 1120.8 | 108.4 | **0.097×** | 71.2 | 110.6 | **1.554×** |'
    check_table match   "$TBL_HDR" '| c=1 | 168.9 | 90.2 | 0.534× | 171.5 | 100.7 | 0.587× |'
    check_table match   "$TBL_HDR" '| c=4 | 484.7 | 111.9 | 0.231x | 123.3 | 113.8 | 0.923x |'
    # MUST NOT FIRE, UNDER THE SAME HEADER, so the DECIMAL is the only variable
    # between this row and the three above. `| 2x |` is a column label, a
    # dimension or a bare multiple; a measured ratio carries a fractional digit.
    check_table nomatch "$TBL_HDR" '| 2x | fast | yes | torch |'
    check_table nomatch "$TBL_HDR" '| c=16 | 1120 | 108 | 1 | 71 | 110 | 2 |'
    check_table nomatch "$TBL_HDR" '| v1.8x release | notes | for | llama |'
    # ...and WITHOUT a competitor header, an identical ratio cell is a
    # compression factor or a scaling efficiency, not a comparator claim.
    check_table nomatch '| band | agg | dec | scaling |' '| c=16 | 1120.8 | 108.4 | **0.097×** |'
    # A cell with no leading space must still match: the left-boundary spelling
    # is an OPTIONAL prefix, not a required one.
    check_table match   "$TBL_HDR" '|c=16|1120.8|0.097×|'
    if [ "$tt" -lt 8 ]; then
        printf '  FAIL  table class has %s row(s); at least 8 are required\n' "$tt"; tf=$((tf+1))
    fi
    printf '  %s table case(s), %s failure(s)\n' "$tt" "$tf"

    # THE CITATION EXEMPTION (PP-12: claim_unreceipted / claim_receipted).
    #
    # These two rows are the ones §6 names, and they are the only rows in this
    # file that exercise drop_receipted() -- the dereferencing pass. A table
    # that stopped at the regexes could not tell an exemption that RESOLVES a
    # path from one that merely sees an `evidence/` token, which is the half
    # that makes "cites" checkable rather than decorative.
    #
    # The claim is always on line 1 of the fixture, so a citation on a LATER
    # line exercises the window's forward edge.
    ct2=0; cf2=0
    CITE_TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${CAUSAL_TD:?}" "${TABLE_TD:?}" "${CITE_TD:?}"' EXIT
    mkdir -p "$CITE_TD/docs" "$CITE_TD/evidence/parity" "$CITE_TD/crates/x/src"
    printf '{}\n' > "$CITE_TD/evidence/parity/receipt.r1.json"
    check_cite_in() { # check_cite_in <relpath> <name> <RED|GREEN> <line...>
        local rel="$1" name="$2" want="$3" got=GREEN n
        shift 3
        mkdir -p "$CITE_TD/$(dirname "$rel")"
        printf '%s\n' "$@" > "$CITE_TD/$rel"
        # The same three detectors the .md sweep applies, then the same
        # exemption the sweep applies after them. Line 1 carries the claim.
        n=1
        if grep -qE "$RATIO_RE|$TPUT_RE|$PLACEHOLDER_RE" <<< "$1" \
           && ! grep -qE "$TARGET_RE" <<< "$1" \
           && ! claim_citation_exempts "$CITE_TD" "$rel" "$n"; then
            got=RED
        fi
        ct2=$((ct2+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-38s %s\n' "$name" "$(printf '%s' "$1" | cut -c1-52)"
        else printf '  FAIL  %-38s want %-5s got %-5s %s\n' "$name" "$want" "$got" "$(printf '%s' "$1" | cut -c1-40)"; cf2=$((cf2+1)); fi
    }
    check_cite() { # check_cite <name> <RED|GREEN> <line...>
        local name="$1"; shift
        check_cite_in docs/probe.md "$name" "$@"
    }
    printf -- '--- citation exemption (PP-12) ---\n'
    # MUST FIRE: the figure with nothing behind it. Verbatim README.md:234.
    check_cite claim_unreceipted RED \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |' \
        'Reproduced from candle-vs-apr.'
    # MUST NOT FIRE: the same figure, citing a receipt that RESOLVES --
    # in the SAME PIPE-TABLE ROW, which is how the master spec's §2.1 table
    # writes its figures.
    check_cite claim_receipted GREEN \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 | evidence/parity/receipt.r1.json |'
    # ...and on a FOLLOWING line, inside the window.
    check_cite claim_receipted_next_line GREEN \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |' \
        '' \
        'Receipt: evidence/parity/receipt.r1.json'
    # THE WINDOW IS A BOUND. Four lines below is out.
    check_cite claim_citation_out_of_window RED \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |' \
        '' '' '' \
        'Receipt: evidence/parity/receipt.r1.json'
    # A DANGLING CITATION IS NOT A CITATION. It buys the reader's trust with a
    # file nobody can open, which is worse than no citation at all.
    check_cite claim_dangling_citation RED \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 | evidence/parity/gone.json |'
    # A COMPARATOR RATIO IS EXEMPTED THE SAME WAY -- PP-12 is one rule, not two.
    check_cite claim_ratio_receipted GREEN \
        'decode 0.650× llama.cpp (evidence/parity/receipt.r1.json)'
    check_cite claim_ratio_unreceipted RED \
        'decode 0.650× llama.cpp on the reference host'
    # A DIRECTORY IS NOT A RECEIPT. The token `evidence/parity/` matches
    # PERF_CLAIM_RECEIPT_RE (as `evidence/parity`) and the old `[ -e ]` test was
    # true of it, so a claim could be legalised by naming a folder nobody can
    # open to find the figure. Two live lines in this tree were exempted by
    # exactly this and by nothing else.
    check_cite claim_dir_citation_does_not_exempt RED \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 | evidence/parity/ |'
    # ...and the SAME line with the FILE named is still green, so the row above
    # is not passing because the fixture directory is missing.
    check_cite claim_file_citation_still_exempts GREEN \
        '| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 | evidence/parity/receipt.r1.json |'
    # THE SURFACE IS A CONJUNCT. PP-12 legalises a cited figure in README.md,
    # book/ and docs/ — the places a reader can follow the path. A `.rs` doc
    # comment is rendered by `cargo doc` far from the repository and a shipped
    # `println!` shows the number and not the path, so the citation buys
    # nothing there. Byte-identical claim and byte-identical citation as
    # `claim_file_citation_still_exempts`; the SURFACE is the only variable.
    check_cite_in crates/x/src/probe.rs claim_rs_citation_does_not_exempt RED \
        '/// sustained 225+ tok/s (evidence/parity/receipt.r1.json)'
    check_cite_in docs/probe.md claim_md_citation_exempts GREEN \
        'sustained 225+ tok/s (evidence/parity/receipt.r1.json)'
    if [ "$ct2" -lt 11 ]; then
        printf '  FAIL  citation table has %s row(s); at least 11 are required\n' "$ct2"; cf2=$((cf2+1))
    fi
    # THE TWO CONJUNCTS, ASSERTED SEPARATELY. The rows above would still pass if
    # a future edit collapsed the exemption back into one test and compensated
    # elsewhere; these say which half moved.
    if claim_citation_surface_ok crates/x/src/probe.rs; then
        printf '  FAIL  the exemption surface admits .rs again — PP-12 names three surfaces\n'; cf2=$((cf2+1))
    fi
    if ! claim_citation_surface_ok README.md || ! claim_citation_surface_ok book/src/a.md \
       || ! claim_citation_surface_ok docs/specifications/a.md; then
        printf '  FAIL  the exemption surface no longer admits README.md / book/ / docs/\n'; cf2=$((cf2+1))
    fi
    printf 'a figure, evidence/parity/\n' > "$CITE_TD/docs/dir-probe.md"
    if claim_line_is_receipted "$CITE_TD" docs/dir-probe.md 1; then
        printf '  FAIL  a bare evidence/ DIRECTORY resolves again — `-e` is back\n'; cf2=$((cf2+1))
    fi
    printf '  %s citation case(s), %s failure(s)\n' "$ct2" "$cf2"

    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    [ "$f" -eq 0 ] && [ "$cf" -eq 0 ] && [ "$rf" -eq 0 ] && [ "$mf" -eq 0 ] \
        && [ "$tf" -eq 0 ] && [ "$cf2" -eq 0 ] || exit 1
    exit 0
fi


# The shipped surface: library and binary sources only. Tests, benches, examples
# and fixtures state targets and are out of scope BY DESIGN, not by oversight.
# TWO HOLES, BOTH MEASURED, BOTH OF THIS REPO'S DOCUMENTED SHAPES.
#
# (a) DEPTH. The glob was `crates/**/src/**/*.rs`, which git expands with a
#     MINIMUM path depth — `**` between two literals must match at least one
#     segment. So `crates/apr-cli/src/dispatch.rs` was NOT in the universe:
#
#       crates/**/src/**/*.rs      -> 6908 files
#       depth-tolerant equivalent  -> 7953 files
#       invisible to the guard     -> 1045 tracked files
#
#     A claim literal in any of those 1045 passed. Verified: `// 2.93x Ollama`
#     in crates/apr-cli/src/ gave rc=0.
#
# (b) TRACKED-ONLY. `git ls-files` gives an untracked file a free pass — the
#     documented tracked-only-universe shape, and the third instance in this
#     epic. Unioned with a working-tree find.
#
# (d) docs/specifications/ USED TO BE EXCLUDED, AND IS NOT ANY MORE (PP-12,
#     PP-LLAMA-001 §12 row 10). The argument for the exclusion was that a
#     specification is where a measured number belongs WITH its provenance, that
#     a spec must be able to QUOTE a banned figure in order to ban it, and that
#     nobody installs apr and reads docs/specifications/. The first two are
#     right and are now served by the CITATION EXEMPTION instead — a figure in a
#     spec is legal exactly when it cites the evidence/ receipt that produced
#     it, which is the rule §3.2 always stated and which no guard could enforce
#     while the whole directory was invisible.
#
#     The third argument is what measurement killed. docs/specifications/ was
#     carrying 327 unreadable ratio and throughput matches, including
#     `apr-mcp-server-spec.md:24` ("It achieves 1.43× Ollama decode perf at 128
#     tokens") and `:276` ("Measured ~196 tok/s on reference hardware") — two
#     figures for which NO receipt exists anywhere under evidence/, on a
#     document that decides what the product ships. A number nobody can
#     dereference is a claim wherever it is written; the exclusion was an
#     aperture, not a policy.
#
#     WHAT REPLACED IT IS NARROWER AND STATED: a spec figure is legal iff it
#     cites evidence/, and the archive of superseded specs is out of scope for
#     the reason in the next paragraph.
# (d) docs/specifications/archive/ IS EXCLUDED FOR THE REASON docs/archive/ IS,
#     and it had to be named separately the moment the parent exclusion went: it
#     is a sibling of docs/archive/, not a child, so removing `^docs/specifications/`
#     would have pulled 125 retired documents into scope in the same stroke. A
#     retired specification is a record, and a guard that punishes the record
#     teaches people to delete it.
# (d) docs/archive/ IS EXCLUDED FOR EXACTLY THE SAME REASON, and it
#     had to be added the moment a spec was archived rather than deleted. Superseding
#     APR-PERF-GATE-001 moved its v2.2 document out of docs/specifications/ (excluded at
#     the time) into docs/archive/perf-2026-09-01/ (not), and that ONE MOVE turned nine
#     lines RED -- every one of them a figure the document QUOTES IN ORDER TO BAN,
#     including `2.93x Ollama` and `36.9x`, the two this guard exists because of.
#
#     The rule is unchanged: a claim a USER READS is the defect. Nobody installs apr and
#     reads an archive of superseded specifications; archiving is how this repository
#     retires a document without deleting the record, and a guard that punishes the
#     archive teaches people to delete instead. The live surfaces -- book/, docs/BEATS.md,
#     README.md and shipped source -- are untouched by this line.
# (c) book/ AND docs/ WERE NOT IN THE UNIVERSE AT ALL, which is the one that
#     mattered most: §9's whole point is that a claim a USER READS is the
#     defect, and book/ is where users read. Five live `2.93x Ollama` claims sat
#     in book/ while this guard reported PASS.
# (d) ROOT-LEVEL *.md WAS NOT IN THE UNIVERSE EITHER, and it is where README.md is.
#     da069a25f published `2.93x Ollama` to book/src/examples/showcase-benchmark.md AND
#     to README.md. F6 brought the first into scope; the second stayed out of BOTH this
#     universe and B4's, and F9 measured the hole rather than widening it, because a
#     scope change that moves only one of the two definitions puts them out of step
#     silently. This commit moves both. README.md is the first page a user reads, which
#     is (c)'s own argument applied to the page above the book.
#
#     `:(glob)` IS LOAD-BEARING. A bare `'*.md'` pathspec matches at every depth -- 3460
#     files here against 6 -- so it would pull in tests/, fixtures/ and evidence/ and
#     re-admit the tests/ and evidence/ prose the greps below exclude by name. The
#     `:(glob)` magic makes `*` stop at `/`, so the pathspec means what it reads as.
#     `root-md-anchor-removed` in scripts/mutate-guard.sh mutates B4's matching anchor.
mapfile -t SRC < <(
    { git ls-files 'crates/*/src/**/*.rs' 'crates/*/src/*.rs' 'src/**/*.rs' 'src/*.rs' \
                   'book/**/*.md' 'book/*.md' 'docs/**/*.md' 'docs/*.md' \
                   ':(glob)*.md' 2>/dev/null
      find crates/*/src src book docs -type f \( -name '*.rs' -o -name '*.md' \) 2>/dev/null
    } | LC_ALL=C sort -u \
    | grep -vE '(^|/)(tests?|benches|examples)/' \
    | grep -vE '^docs/specifications/archive/' \
    | grep -vE '^docs/archive/' \
    | grep -vE '_tests?\.rs$|_test\.rs$|proptests?[_.]|/fixtures?/')


if [ "${#SRC[@]}" -eq 0 ]; then
    printf 'FAIL  the file universe is EMPTY — a guard over no files is vacuous\n'
    exit 1
fi
printf 'universe: %s shipped source file(s)\n' "${#SRC[@]}"

hits=$(grep -InE "$CLAIM_RE" "${SRC[@]}" 2>/dev/null \
       | grep -E "$RATIO_RE|$TPUT_RE|$DIAGNOSIS_RE" | grep -vE "$TARGET_RE" || true)

# Doc comments on shipped code are read by users through `cargo doc`.
dochits=$(grep -InE "^[[:space:]]*(///|//!)" "${SRC[@]}" 2>/dev/null \
          | grep -E "$RATIO_RE|$TPUT_RE" | grep -vE "$TARGET_RE" || true)

# MARKDOWN NEEDS ITS OWN DETECTOR, and this is why adding book/ to the universe
# above was not by itself a fix.
#
# CLAIM_RE requires a Rust print macro — println!, format!, .green(). Markdown
# contains none, so every .md file added to SRC contributed exactly zero hits
# and the guard still reported PASS over them. The universe grew and the
# coverage did not, which is the most convincing kind of false progress: the
# `universe: N file(s)` line goes up and nothing is actually checked.
#
# In prose there is no macro to look for, because ALL of it is the printed
# surface. So for .md the ratio/throughput patterns apply directly. Fenced code
# blocks are deliberately NOT exempt: `book/src/tools/apr-cli.md:1396` is a
# claim inside a sample terminal transcript, and a user reads a transcript as a
# result, not as source.
mdfiles=()
for f in "${SRC[@]}"; do case "$f" in *.md) mdfiles+=("$f") ;; esac; done
mdhits=""
tablehits=""
if [ "${#mdfiles[@]}" -gt 0 ]; then
    # PLACEHOLDER_RE is applied to MARKDOWN ONLY, deliberately. In .rs a bare
    # `TODO` is an ordinary code comment and SATD there is pmat's gate, not
    # this one; adding it to the Rust sweep would import a large, already-owned
    # debt class under a guard that has nothing to say about it. Prose is where
    # an unfilled figure is read as a filled one.
    mdhits=$(grep -InE "$RATIO_RE|$TPUT_RE|$PLACEHOLDER_RE" "${mdfiles[@]}" 2>/dev/null \
             | grep -vE "$TARGET_RE" || true)
    # THE TABLE CLASS. Needs its own pass because it is the only detector here
    # that is not a pure LINE test: the competitor lives in the header row and
    # the ratio in a cell below it. See TABLE_CELL_RATIO_RE.
    tablehits=$(table_ratio_hits_in "." "${mdfiles[@]}" | grep -vE "$TARGET_RE" || true)
fi

# PERF-016: the causal class over .rs, print macro or not. See DIAGNOSIS_RE.
rsfiles=()
for f in "${SRC[@]}"; do case "$f" in *.rs) rsfiles+=("$f") ;; esac; done
causalhits=""
if [ "${#rsfiles[@]}" -gt 0 ]; then
    causalhits=$(causal_literals_in "${rsfiles[@]}" || true)
fi

# Deduped by file:line -- a literal caught by both the CLAIM_RE sweep and the
# causal sweep is ONE finding, and counting it twice would corrupt the ratchet's
# known/new tally.
all=$(printf '%s\n%s\n%s\n%s\n%s\n' "$hits" "$dochits" "$mdhits" "$tablehits" "$causalhits" \
      | grep -v '^$' | awk -F: '!seen[$1":"$2]++' || true)

# THE CITATION EXEMPTION, applied AFTER the match and BEFORE the baseline
# comparison. See drop_receipted(). A hit whose line -- or the three lines
# either side of it -- carries a RESOLVING `evidence/<path>` token is not a
# finding at all: PP-12's rule is "legal iff it cites a receipt", and this is
# the half of that conjunction this guard owns.
exempted=0
before_exempt=$(printf '%s\n' "$all" | grep -c . || true)
all=$(printf '%s\n' "$all" | grep -v '^$' | drop_receipted "." || true)
after_exempt=$(printf '%s\n' "$all" | grep -c . || true)
exempted=$((before_exempt - after_exempt))
printf 'receipted (cite a resolving evidence/ path, exempt): %s\n' "$exempted"

# --update RE-DERIVES THE BASELINE MECHANICALLY, AND IT IS NOT A LOOPHOLE.
#
# The sibling guard has had this since PERF-010 and this one did not, so the
# only way to record an aperture reveal here was to hand-transcribe a few
# hundred `<path>:<line>` coordinates out of a FAIL log -- which is both
# error-prone and, worse, invites transcribing them WRONG in the direction that
# makes the guard quieter.
#
# It cannot launder anything, and the reason is structural rather than
# procedural: --update only writes the file. The RATCHET still judges it on the
# next run, against a ref this branch cannot rewrite, under `set-aperture` --
# so every line this branch WROTE is still refused by name, and every admitted
# line is still printed. What --update removes is transcription, not judgement.
#
# The hand-written header is PRESERVED verbatim. It carries the dated aperture
# paragraphs that say WHY each growth was admitted, and a mode that rewrote
# them from a printf would delete the only record of that reasoning.
if [ "${1:-}" = "--update" ]; then
    header=$(awk 'BEGIN{h=1} h==1 && ($0 ~ /^#/ || $0 ~ /^[[:space:]]*$/) {print; next} {h=0}' "$BASELINE" 2>/dev/null || true)
    {
        [ -n "$header" ] && printf '%s\n' "$header"
        printf '%s\n' "$all" | grep -v '^$' \
            | awk -F: '{print $1":"$2}' | LC_ALL=C sort -u
    } > "$BASELINE.new"
    mv "$BASELINE.new" "$BASELINE"
    printf 'baseline written: %s entr(ies). The ratchet still judges it -- run the\n' \
        "$(grep -cvE '^[[:space:]]*(#|$)' "$BASELINE" || true)"
    printf '                  guard with no argument and read the set-aperture verdict.\n'
    exit 0
fi

known=0
new=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    loc="${line%%:*}:$(printf '%s' "$line" | cut -d: -f2)"
    if [ -f "$BASELINE" ] && grep -qxF "$loc" "$BASELINE"; then
        known=$((known + 1))
    else
        printf 'FAIL  %s\n' "$(printf '%s' "$line" | cut -c1-150)"
        new=$((new + 1)); rc=1
    fi
done <<< "$all"

printf 'known (baselined, must shrink): %s   new: %s\n' "$known" "$new"

# THE RATCHET. A baseline that may grow is a permission slip. Entries must be
# removed as claims are deleted or derived; a new one requires editing this file.
if [ -f "$BASELINE" ]; then
    stale=0
    while IFS= read -r loc; do
        [ -n "$loc" ] || continue
        case "$loc" in '#'*) continue ;; esac
        case " $all " in *"$loc"*) : ;; *) stale=$((stale + 1)) ;; esac
    done < "$BASELINE"
    if [ "$stale" -gt 0 ]; then
        printf 'REPORT %s baseline entry(ies) no longer match — prune them so the\n' "$stale"
        printf '       ratchet cannot silently re-admit a claim at that location.\n'
    fi
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
#     "known (baselined, must shrink)"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
RATCHET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${RATCHET_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "$RATCHET_ROOT" scripts/claim_literal_baseline.txt set-aperture \
    scripts/check_no_claim_literals.sh || rc=1

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no NEW claim literal on a user-facing surface.\n'
else
    printf 'FAIL  a claim about measured reality is a literal on a surface a user\n'
    printf '      reads. Derive it from a measurement, or delete it. If it is a\n'
    printf '      TARGET rather than a claim, it belongs in a test or a spec.\n'
fi
exit "$rc"

# ---------------------------------------------------------------------------
# CASE TABLE. This repo's rule: a guard regex ships a must-match/must-not-match
# table, because the apr-invocation patterns were wrong five times and every one
# was caught by a table, none by review. Run with --selftest.
# ---------------------------------------------------------------------------
