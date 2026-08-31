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
TPUT_RE='[0-9]{2,}\+?[[:space:]]*tok/s'

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

# A TARGET says what we WANT; a CLAIM says what we GOT. Only the second lies.
# Lines that name themselves a target, a threshold or a comparison operator are
# stating a bar, and a bar is allowed to be a constant — that is what a bar IS.
TARGET_RE='([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|FAIL:|>=|<=|[><] *[0-9])'

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
    ' "$@" 2>/dev/null | grep -E "$ATTRIBUTION_RE" | grep -vE "$TARGET_RE"
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
    # VACUITY. A table that shrank to nothing would sweep clean, and both halves
    # must stay populated -- an all-positive table proves nothing about noise.
    if [ "$rt" -lt 27 ]; then
        printf '  FAIL  ratio table has %s row(s); at least 27 are required\n' "$rt"; rf=$((rf+1))
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
    if [ "$mt" -lt 24 ]; then
        printf '  FAIL  markdown table has %s row(s); at least 24 are required\n' "$mt"; mf=$((mf+1))
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

    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    [ "$f" -eq 0 ] && [ "$cf" -eq 0 ] && [ "$rf" -eq 0 ] && [ "$mf" -eq 0 ] || exit 1
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
# (d) docs/specifications/ IS EXCLUDED, and this guard's own FAIL text is the
#     reason: "If it is a TARGET rather than a claim, it belongs in a test or a
#     spec." A specification is where a measured number belongs WITH its
#     provenance — §1 of APR-PERF-GATE-001 states the four adoption killers as
#     measured figures, and §9 quotes the banned `291` literal in order to ban
#     it. Sweeping docs/** in wholesale (added #2705 r3) caught those four lines
#     and would have forced the spec to describe its own subject matter in
#     euphemism.
#
#     The distinction is NOT "docs are exempt". §9's premise is that a claim a
#     USER READS is the defect. Users read book/ and docs/BEATS.md — both stay in
#     scope, and BEATS.md lines are in the baseline. Nobody installs apr and
#     reads docs/specifications/. If a spec figure ever reaches a user surface it
#     is caught THERE, which is the surface that matters.
# (c) book/ AND docs/ WERE NOT IN THE UNIVERSE AT ALL, which is the one that
#     mattered most: §9's whole point is that a claim a USER READS is the
#     defect, and book/ is where users read. Five live `2.93x Ollama` claims sat
#     in book/ while this guard reported PASS.
mapfile -t SRC < <(
    { git ls-files 'crates/*/src/**/*.rs' 'crates/*/src/*.rs' 'src/**/*.rs' 'src/*.rs' \
                   'book/**/*.md' 'book/*.md' 'docs/**/*.md' 'docs/*.md' 2>/dev/null
      find crates/*/src src book docs -type f \( -name '*.rs' -o -name '*.md' \) 2>/dev/null
    } | LC_ALL=C sort -u \
    | grep -vE '(^|/)(tests?|benches|examples)/' \
    | grep -vE '^docs/specifications/' \
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
if [ "${#mdfiles[@]}" -gt 0 ]; then
    # PLACEHOLDER_RE is applied to MARKDOWN ONLY, deliberately. In .rs a bare
    # `TODO` is an ordinary code comment and SATD there is pmat's gate, not
    # this one; adding it to the Rust sweep would import a large, already-owned
    # debt class under a guard that has nothing to say about it. Prose is where
    # an unfilled figure is read as a filled one.
    mdhits=$(grep -InE "$RATIO_RE|$TPUT_RE|$PLACEHOLDER_RE" "${mdfiles[@]}" 2>/dev/null \
             | grep -vE "$TARGET_RE" || true)
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
all=$(printf '%s\n%s\n%s\n%s\n' "$hits" "$dochits" "$mdhits" "$causalhits" \
      | grep -v '^$' | awk -F: '!seen[$1":"$2]++' || true)

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
