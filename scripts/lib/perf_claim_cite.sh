#!/usr/bin/env bash
# perf_claim_cite.sh — ONE definition of "this number cites a receipt", shared
# by the POSITIVE guard (check_perf_claims_cite_receipts.sh) and the NEGATIVE
# guard (check_no_claim_literals.sh).
#
# WHY THIS IS A LIBRARY AND NOT A SECOND COPY
# -------------------------------------------
# PP-LLAMA-001 §6/PP-12 states one rule:
#
#     "a number in README.md / book/ / docs/ is legal iff it cites an
#      evidence/ receipt path"
#
# and that rule is a CONJUNCTION that no single guard used to encode. The
# positive half asked a comparator-free speed comparison to cite a receipt; the
# negative half deleted a comparator ratio or a bare throughput literal
# regardless of citation. So PP-12's own must-not-fire fixture — "a figure
# citing evidence/…/receipt.r1.json" — was RED: citing a receipt beside
# `225+ tok/s` left the line a TPUT_RE hit in the negative guard, and there was
# no spelling of the line that both guards accepted.
#
# The remedy is an EXEMPTION in the negative guard, applied AFTER the match, and
# it must use the SAME definition of "cites" the positive guard enforces. Two
# copies of that definition drift, and the drift is invisible precisely because
# each guard keeps passing against its own copy — the documented failure shape
# this repository has closed three times (the `apr`-invocation patterns, the
# claim-literal extraction rule, the TARGET vocabulary). So: one file, sourced
# by both.
#
# WHAT "CITES A RECEIPT" MEANS, MECHANICALLY (unchanged from the definition
# check_perf_claims_cite_receipts.sh has enforced since PERF-010):
#
#   Within the claim's line or the CITE_WINDOW lines either side of it, a token
#   matching `evidence/<path>` appears, AND that path exists in the tree.
#
# A commit SHA was rejected (records WHEN, not HOW; nothing can dereference it)
# and a `receipt:` key was rejected (a second dialect for a word 33 files
# already spell as a path under evidence/). An `evidence/` path is the only
# candidate that is DEREFERENCEABLE, which is what turns "cites" from a
# syntactic gesture into a checkable claim.
#
# A DANGLING CITATION NEVER EXEMPTS. `claim_line_is_receipted` requires at least
# one token that RESOLVES; a token pointing at a file nobody can open buys the
# reader's trust with nothing, and treating it as a citation would make the
# exemption strictly easier to satisfy than the rule it implements.
#
# NEITHER DOES A DIRECTORY, and that is a MEASURED hole rather than a
# hypothetical one. The resolver used `[ -e ]`, which is true of a directory, and
# `PERF_CLAIM_RECEIPT_RE` matches `evidence/prrev-021` inside the token
# `evidence/prrev-021/`. Two live lines in this tree --
# `docs/specifications/PR-REVIEW-SKILL-002-v2.md:60` and `:63`, both carrying
# `2.93x Ollama` -- were exempted by that bare directory and by nothing else. A
# directory is not dereferenceable to a number: a reader cannot open it and find
# the figure. `[ -f ]`, then.
#
# AND THE EXEMPTION IS MARKDOWN-ONLY. PP-12 names three surfaces --
# `README.md`, `book/`, `docs/` -- and the exemption was applied to every hit the
# negative guard produced, `.rs` included. On a Rust surface the two halves of
# the rule come apart: `cargo doc` renders the rustdoc line and drops the
# reader nowhere near the repository, so an `evidence/` path beside a figure in
# a doc comment is a string, not a citation anyone can follow. A shipped
# `println!` is worse: the user sees the number and never sees the path at all.
# A number on a `.rs` surface therefore stays a finding whatever sits beside it;
# the remedy is to delete it or move the discussion to `docs/`.
#
# OPTION-NEUTRAL. This file is SOURCED, and `set` in a sourced file mutates the
# CALLER's shell — check_no_claim_literals.sh runs under `set -euo pipefail`
# and check_perf_claims_cite_receipts.sh deliberately runs WITHOUT `-e`. There
# is no `set` at file scope here; every entry point reports by RETURN STATUS:
#
#     . "${REPO_ROOT}/scripts/lib/perf_claim_cite.sh" || exit 1
#
# NOTE ON THIS FILE'S OWN COVERAGE, stated rather than left to be found:
# scripts/check_sourced_libs_option_neutral.sh and
# scripts/check_shell_lint_ratchet.sh both enumerate `scripts/*.sh` at depth 1
# only, so neither reads this path. The option-neutrality above is therefore a
# REVIEWED property here, not a mechanically enforced one, and both guards'
# universes should be widened to `scripts/lib/*.sh` in a follow-up.
#
# Refs: paiml/aprender PP-LLAMA-001 §6/PP-12, PERF-010, #2787.

# The citation token, and the window it may appear in.
#
# WINDOW = the claim's line, plus three lines either side. Three is not a round
# number chosen for looking careful: it is the shortest window that covers the
# three shapes citations actually take in this repository's markdown — in-row
# for a table, in the following sentence for prose, and on the line under a
# fenced block. Both edges are pinned by case-table rows in BOTH guards, so
# widening it later requires re-running two tables rather than re-reading this
# comment.
PERF_CLAIM_RECEIPT_RE='evidence/[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*'
PERF_CLAIM_CITE_WINDOW=3

# Back-compat aliases for the names check_perf_claims_cite_receipts.sh used
# before this file existed. Kept so that guard reads exactly as it did.
RECEIPT_RE="$PERF_CLAIM_RECEIPT_RE"
WINDOW="$PERF_CLAIM_CITE_WINDOW"

# resolve_citations <root> <text>
#   -> one line per receipt token found: "<path> exists" | "<path> missing"
#
# Trailing markdown punctuation is stripped, so a citation inside
# `[...](../evidence/x/y.json)`, inside backticks, or ending a sentence
# resolves. The leading `../` (or any prefix before the first `evidence/`) is
# dropped so a relative link from book/src/** resolves against the repo root.
resolve_citations() {
    local root="$1"
    local text="$2"
    local tok p
    printf '%s\n' "$text" | grep -oE "$PERF_CLAIM_RECEIPT_RE" 2>/dev/null | while IFS= read -r tok; do
        p="${tok#"${tok%%evidence/*}"}"
        p=$(printf '%s' "$p" | sed 's/[])`",.;:]*$//')
        # `-f`, NOT `-e`: a directory is not a receipt. See the header.
        if [ -f "$root/$p" ]; then printf '%s exists\n' "$p"; else printf '%s missing\n' "$p"; fi
    done
}

# claim_citation_surface_ok <relpath>
#   rc 0 = PP-12 allows a citation to legalise a figure on this surface.
#
# The three surfaces PP-12 names, and only those. Everything else -- `.rs`,
# `CHANGELOG.md`, `CLAUDE.md`, anything at the repository root that is not
# `README.md` -- is a surface where the rule is "no figure", full stop.
claim_citation_surface_ok() { # claim_citation_surface_ok <relpath>
    case "${1#./}" in
        README.md | book/* | docs/*) return 0 ;;
        *) return 1 ;;
    esac
}

# claim_line_is_receipted <root> <relpath> <lineno>
#   rc 0 = the line, or one of the CITE_WINDOW lines either side of it, carries
#          an evidence/ token that RESOLVES to a file that exists.
#   rc 1 = it does not (no token at all, or every token dangles).
#
# The window is clamped to the file, so line 1 and the last line work. A file
# that cannot be read is rc 1: "could not check" is never "cited".
claim_line_is_receipted() {
    local root="$1" rel="$2" n="$3"
    local f="$root/$rel"
    local total lo hi block cites
    [ -f "$f" ] || return 1
    case "$n" in '' | *[!0-9]*) return 1 ;; esac
    total=$(wc -l < "$f")
    lo=$((n - PERF_CLAIM_CITE_WINDOW))
    if [ "$lo" -lt 1 ]; then lo=1; fi
    hi=$((n + PERF_CLAIM_CITE_WINDOW))
    if [ "$hi" -gt "$total" ]; then hi="$total"; fi
    block=$(sed -n "${lo},${hi}p" "$f" 2>/dev/null)
    cites=$(resolve_citations "$root" "$block")
    grep -q ' exists$' <<< "$cites"
}

# claim_citation_exempts <root> <relpath> <lineno>
#   rc 0 = PP-12 legalises the figure at this coordinate: the surface is one of
#          the three PP-12 names AND the line (or its window) carries an
#          `evidence/` path that resolves to a FILE.
#
# This, not `claim_line_is_receipted`, is what the negative guard applies. The
# two are kept apart so the citation question and the surface question each stay
# answerable on their own -- and so a future reader can see that the surface
# test is a conjunct of the exemption rather than an accident of where it is
# called from.
claim_citation_exempts() { # claim_citation_exempts <root> <relpath> <lineno>
    claim_citation_surface_ok "$2" || return 1
    claim_line_is_receipted "$1" "$2" "$3"
}
