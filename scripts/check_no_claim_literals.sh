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
MODE="${1:-}"
rc=0
# Progress lines go to stderr under --list so stdout is the list and nothing else.
note() {
    case "$MODE" in --list|--explain) printf '%s\n' "$1" >&2 ;; *) printf '%s\n' "$1" ;; esac
}
# AN UNRECOGNISED ARGUMENT IS AN ERROR, NOT A CHECK RUN. `--selftest` mistyped
# in ci.yml would otherwise fall through to the full scan and go green, and the
# job would report that the case tables ran when they never did.
case "$MODE" in
    ''|--selftest|--list|--explain) : ;;
    *) printf 'usage: %s [--selftest | --list | --explain <hash16|path>]\n' "$0" >&2
       exit 2 ;;
esac
note '--- claim literals on user-facing surfaces ------------------------'

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
RATIO_RE='[0-9]+(\.[0-9]+)?x[[:space:]]+(Ollama|ollama|llama\.cpp|llama|vLLM|vllm|PyTorch|torch|FasterTransformer|fastertransformer|SGLang|sglang|TensorRT|tensorrt|TGI|tgi|LMDeploy|lmdeploy|TurboMind|turbomind|static[[:space:]]+batching)'
TPUT_RE='[0-9]{2,}\+?[[:space:]]*tok/s'

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

# ---------------------------------------------------------------------------
# THE BASELINE KEY: file + hash of the claim TEXT. It used to be FILE:LINE.
#
# WHY IT CHANGED. PERF-006 added `pub mod andon;` and a doc comment near the top
# of crates/aprender-serve/src/lib.rs. It added no claim. This guard went
# rc=1 "known: 316 new: 2" against rc=0 "known: 318 new: 0" on base, because two
# baselined claims further down the file MOVED, their FILE:LINE keys stopped
# matching, and the guard reported them as newly written. It was the third
# line-keyed-ledger false red in one day (check_dogfood_coverage.sh G2.1 twice).
#
# A guard that reds for a reason unrelated to its property is a guard people
# learn to route around, which is the failure mode this epic exists to prevent.
# A line NUMBER is not part of the property. The claim text is the property.
#
# WHAT THE KEY DOES AND DOES NOT SURVIVE — the tradeoff, stated:
#
#   survives   inserting/deleting lines anywhere in the file; reindenting the
#              claim; re-aligning a markdown table row (whitespace runs are
#              collapsed before hashing).
#   RE-KEYS    editing the claim TEXT, including changing one digit. This is
#              deliberate and is the ratchet: "1.9x Ollama" edited in place to
#              "2.9x Ollama" is a DIFFERENT claim, hashes differently, and reds
#              as new. It cannot hide behind the entry it replaced.
#              The cost is that a cosmetic edit to a baselined line — fixing a
#              typo, rewrapping prose — also reds and needs the entry re-keyed.
#              That is the price of not letting a number change ride for free,
#              and it is paid on the rarer event.
#   GAMEABLE?  Editing the text to dodge the key makes the guard LOUDER, never
#              quieter: any edited claim leaves the baselined key unmatched
#              (reported as shrinkable) and arrives as a new key (rc=1). There
#              is no text edit that turns a red into a green.
#
# THE SAME CLAIM TWICE IN ONE FILE. Two identical claim lines in one file
# collapse to ONE key, so the entry carries a COUNT and the guard compares
# multiplicity. Without it the second copy would ride in free on the first — a
# genuinely new claim hidden behind an existing key, which is precisely what a
# ratchet may not permit. Occurrences beyond the baselined count are NEW; fewer
# than the baselined count is a shrink, and is REPORTED so the entry gets pruned.
# The tradeoff: within one file, the guard cannot say WHICH copy is new, only
# that there is one more than was recorded. Between two identical copies that
# distinction has no meaning.
# ---------------------------------------------------------------------------
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
    printf 'FAIL  bash >= 4 required (associative arrays, mapfile); found %s\n' \
           "${BASH_VERSION:-unknown}"
    exit 2
fi

HASHER=()
if command -v sha256sum >/dev/null 2>&1; then HASHER=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then HASHER=(shasum -a 256)
fi
if [ "${#HASHER[@]}" -eq 0 ]; then
    # NOT a soft fallback. A weaker key would silently change which claims the
    # ratchet recognises, and every entry would read as new or as stale.
    printf 'FAIL  no sha256 tool (sha256sum / shasum) — the baseline key cannot\n'
    printf '      be computed, so the ratchet cannot be evaluated.\n'
    exit 2
fi

# Tabs to spaces, collapse runs, trim. Whitespace is layout, not claim.
claim_norm() {
    local s="${1//$'\t'/ }"
    while [ "$s" != "${s//  / }" ]; do s="${s//  / }"; done
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}
# claim_key <file> <matched line text>  ->  "<file>\t<16 hex>"
# The file stays in the key: moving a claim into a different file is a real
# change of surface, and should be re-baselined deliberately.
claim_key() {
    local h
    h=$(claim_norm "$2" | "${HASHER[@]}")
    printf '%s\t%s' "$1" "${h:0:16}"
}

if [ "${1:-}" = "--selftest" ]; then
    t=0; f=0
    check() { # check <expect match|nomatch> <line>
        local want="$1" line="$2" got=nomatch
        if printf '%s\n' "$line" | grep -qE "$CLAIM_RE" \
           && printf '%s\n' "$line" | grep -qE "$RATIO_RE|$TPUT_RE|$DIAGNOSIS_RE" \
           && ! printf '%s\n' "$line" | grep -qE "$TARGET_RE"; then got=match; fi
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

    # KEY MECHANICS. The regex table above says what is a claim; this one says
    # what makes two sightings of a claim the SAME baseline entry. The bug this
    # replaced was entirely in the key, not in the patterns, and no case in the
    # table above could have caught it.
    printf -- '--- key table (baseline key is CONTENT, never FILE:LINE) ---\n'
    kcheck() { # kcheck <same|diff> <label> <fileA> <textA> <fileB> <textB>
        local want="$1" label="$2" a b got
        a=$(claim_key "$3" "$4"); b=$(claim_key "$5" "$6")
        if [ "$a" = "$b" ]; then got=same; else got=diff; fi
        t=$((t+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-4s %s\n' "$want" "$label"
        else printf '  FAIL  want %-4s got %-4s %s\n' "$want" "$got" "$label"; f=$((f+1)); fi
    }
    # MUST BE THE SAME KEY: the claim did not change, only where/how it sits.
    kcheck same 'reindented (the line-drift bug)' \
        'a/x.rs' '    println!("851.8 tok/s = 2.93x Ollama");' \
        'a/x.rs' '        println!("851.8 tok/s = 2.93x Ollama");'
    kcheck same 'trailing whitespace' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x Ollama");' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x Ollama");   '
    kcheck same 'markdown table re-aligned' \
        'b/t.md' '| large | 114.5 tok/s | 2.9x FASTER |' \
        'b/t.md' '| large    |   114.5 tok/s | 2.9x FASTER    |'
    # MUST BE A DIFFERENT KEY: a different claim may not inherit an entry.
    kcheck diff 'one digit changed in place' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x Ollama");' \
        'a/x.rs' 'println!("851.8 tok/s = 3.93x Ollama");'
    kcheck diff 'comparator swapped' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x Ollama");' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x vLLM");'
    kcheck diff 'same claim, different file' \
        'a/x.rs' 'println!("851.8 tok/s = 2.93x Ollama");' \
        'a/y.rs' 'println!("851.8 tok/s = 2.93x Ollama");'
    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    [ "$f" -eq 0 ] && [ "$cf" -eq 0 ] || exit 1
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
note "universe: ${#SRC[@]} shipped source file(s)"

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
    mdhits=$(grep -InE "$RATIO_RE|$TPUT_RE" "${mdfiles[@]}" 2>/dev/null \
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

# PASS 1 — key every sighting. ONE SOURCE LINE IS ONE CLAIM: `all` concatenates
# three detectors and a line can match two of them (a fenced `println!` inside a
# .md is seen by both the Rust pass and the markdown pass). Three lines in this
# tree do exactly that. Counting them twice would make the baseline multiplicity
# describe the detector rather than the tree, and would then break whenever a
# detector changed.
declare -A seen_line=()
declare -A obs=()
KEYS=(); TEXTS=()
while IFS= read -r line; do
    [ -n "$line" ] || continue
    file="${line%%:*}"; rest="${line#*:}"
    lineno="${rest%%:*}"; text="${rest#*:}"
    if [ -n "${seen_line["$file:$lineno"]:-}" ]; then continue; fi
    seen_line["$file:$lineno"]=1
    key=$(claim_key "$file" "$text")
    obs["$key"]=$(( ${obs["$key"]:-0} + 1 ))
    KEYS+=("$key"); TEXTS+=("$line")
done <<< "$all"

# --list prints the CURRENT set in baseline format. It exists so the baseline is
# regenerated by the guard that reads it, not by a bash one-off that re-derives
# the key and drifts from it. It writes nothing: piping it over the baseline is
# a blanket amnesty, and has to look like one in a diff.
if [ "$MODE" = "--list" ]; then
    if [ "${#obs[@]}" -gt 0 ]; then
        for k in "${!obs[@]}"; do printf '%s\t%s\n' "$k" "${obs["$k"]}"; done | LC_ALL=C sort
    fi
    exit 0
fi

# --explain <hash|file> resolves a baseline key back to the line(s) it covers.
# A hash is unreadable, and that is the one real cost of content-keying: without
# this, a reviewer cannot tell what an entry protects. Nothing is written.
if [ "$MODE" = "--explain" ]; then
    want="${2:-}"
    if [ -z "$want" ]; then
        printf 'usage: %s --explain <hash16 or path fragment>\n' "$0" >&2; exit 2
    fi
    found=0; i=0
    while [ "$i" -lt "${#KEYS[@]}" ]; do
        case "${KEYS[$i]}" in
            *"$want"*)
                printf '%s\n' "${TEXTS[$i]}"
                printf '  key %s\n' "$(printf '%s' "${KEYS[$i]}" | tr '\t' ' ')"
                found=$((found + 1)) ;;
        esac
        i=$((i + 1))
    done
    if [ "$found" -eq 0 ]; then printf 'no current sighting matches: %s\n' "$want"; fi
    exit 0
fi

# LOAD THE BASELINE. One entry per line: <file> TAB <hash16> TAB <count>.
declare -A base_count=()
if [ -f "$BASELINE" ]; then
    while IFS= read -r bline; do
        [ -n "$bline" ] || continue
        case "$bline" in '#'*) continue ;; esac
        bfile="${bline%%$'\t'*}"; brest="${bline#*$'\t'}"
        bhash="${brest%%$'\t'*}"; bcount="${brest#*$'\t'}"
        # A MALFORMED ENTRY IS LOUD. Skipping it quietly would drop a claim out
        # of the recognised set, and the claim it covers would then read as new
        # — a confusing red pointing at innocent code instead of at this file.
        if [ "$bfile" = "$bline" ] \
           || ! printf '%s' "$bhash" | grep -qE '^[0-9a-f]{16}$' \
           || ! printf '%s' "$bcount" | grep -qE '^[1-9][0-9]*$'; then
            printf 'FAIL  malformed baseline entry (want <file>TAB<hash16>TAB<count>): %s\n' \
                   "$(printf '%s' "$bline" | cut -c1-120)"
            rc=1; continue
        fi
        base_count["$bfile"$'\t'"$bhash"]=$bcount
    done < "$BASELINE"
fi

# PASS 2 — compare multiplicity. The first N sightings of a key are the N that
# were baselined; the N+1th is new.
known=0
new=0
declare -A nth=()
i=0
while [ "$i" -lt "${#KEYS[@]}" ]; do
    key="${KEYS[$i]}"
    nth["$key"]=$(( ${nth["$key"]:-0} + 1 ))
    if [ "${nth["$key"]}" -le "${base_count["$key"]:-0}" ]; then
        known=$((known + 1))
    else
        printf 'FAIL  %s\n' "$(printf '%s' "${TEXTS[$i]}" | cut -c1-150)"
        printf '      key %s\n' "$(printf '%s' "$key" | tr '\t' ' ')"
        new=$((new + 1)); rc=1
    fi
    i=$((i + 1))
done

printf 'known (baselined, must shrink): %s   new: %s\n' "$known" "$new"

# THE RATCHET. A baseline that may grow is a permission slip. Entries must be
# removed as claims are deleted or derived; a new one requires editing this file.
# Shrinkage is REPORTED, not failed — deleting a claim must never red the build
# that deleted it — but it is reported loudly enough that the entry gets pruned,
# because a stale entry is a pre-authorised slot for that exact claim to return.
if [ -f "$BASELINE" ] && [ "${#base_count[@]}" -gt 0 ]; then
    stale=0
    shrunk=$(
        for key in "${!base_count[@]}"; do
            have=${obs["$key"]:-0}
            want=${base_count["$key"]}
            [ "$have" -lt "$want" ] || continue
            printf '       %s  baselined %s, present %s\n' \
                   "$(printf '%s' "$key" | tr '\t' ' ')" "$want" "$have"
        done | LC_ALL=C sort
    )
    if [ -n "$shrunk" ]; then
        stale=$(printf '%s\n' "$shrunk" | grep -c .)
        printf 'REPORT %s baseline entry(ies) now cover fewer occurrences than recorded.\n' "$stale"
        printf '       Prune them: a stale entry is a slot that claim may return into.\n'
        printf '%s\n' "$shrunk"
    fi
fi

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
