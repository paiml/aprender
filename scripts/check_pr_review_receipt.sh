#!/usr/bin/env bash
# check_pr_review_receipt.sh - validate a grounded PR-review receipt.
#
# PR-REVIEW-SKILL-002 v2 S6. Every rejection names exactly one blocking class from
# contracts/pr-review-skill-v2.yaml (B1..B6), because a rejection with no named class
# is how a guard grows a rule nothing governs.
#
#   B1  receipt missing, schema-invalid, unsigned, or internally inconsistent
#   B2  reviewer_actor.id = author_actor.id
#   B3  guard mutation score < 1.00        (PRREV-004; not evaluated here)
#   B4  unverified_comparative_claim
#   B5  breaking API surface with no semver bump   (no consultation emits this)
#   B6  index_is_ancestor = false AND verdict = PASS
#
# S3.E (PRREV-015) adds a FIFTH consultation, `antigravity` - an independent reviewing
# agent from a different vendor and model family, run as its own process. It is ADVISORY:
# S7's admission rule lets a class block only while its measured precision is >= 90% on
# the rolling sample, and S8 requires instrument -> 30 samples -> ratchet. There are zero
# samples, so nothing S3.E emits may carry `precision_class: blocking`, and a receipt that
# says otherwise is internally inconsistent - B1, not a new class. The rules below are
# therefore about the HONESTY of the record, never about the content of agy's findings.
#
# USAGE
#   check_pr_review_receipt.sh <receipt-dir> [<receipt-dir> ...]
#   check_pr_review_receipt.sh --match-path <path>          predicate: S3.B path trigger
#   check_pr_review_receipt.sh --match-message <text>       predicate: S3.B message trigger
#   check_pr_review_receipt.sh --match-comparative <text>   predicate: S3.C.1 comparative claim
#   check_pr_review_receipt.sh --match-shipped-surface <p>  predicate: the user-facing surface
#   check_pr_review_receipt.sh --match-crux-surface <line>  predicate: S3.C surface trigger
#   check_pr_review_receipt.sh --match-mutation-trigger <p> predicate: S3.D scope trigger
#   check_pr_review_receipt.sh --match-target <text>        predicate: a bar, not a claim
#   check_pr_review_receipt.sh --match-rs-published <line>  predicate: printed, or a doc comment
#   check_pr_review_receipt.sh --match-arm-e-same-family <id>  predicate: S3.E, not a second vendor
#
# The nine --match-* forms are pure predicates over one string. They exist so the
# regexes can be driven by a must-match / must-not-match case table
# (tests/fixtures/pr-review/*-cases.tsv) rather than by reading them. This repository's
# guard patterns have been wrong six times; a table caught every one and review caught
# none. They cannot bypass any validation - they evaluate a regex and exit.
#
# ENVIRONMENT
#   PR_REVIEW_REPO     repository the receipt's SHAs are resolved against
#                      (default: the git toplevel of the working directory)
#   PR_REVIEW_PUBKEY   minisign public key (default: .github/pr-review.pub)
#   PR_REVIEW_SCHEMA_DIR  vendored schemas (default: schemas/)
#
# There is no variable that turns a check off. A tool this guard needs and cannot find
# is a REJECTION, not a skip: a gate that cannot run must not read green.
#
# EXIT: 0 every receipt accepted; 1 anything else, including a failed positive control.

set -euo pipefail

PROG=${0##*/}

# ---------------------------------------------------------------------------
# S3.B triggers and S3.C.1 comparative-claim detection.
#
# Each pattern is transcribed from the spec, NOT broadened. Where the spec's literal
# pattern misses something a human would call a CUDA change, that is recorded as a gap
# in tests/fixtures/pr-review/README.md and pinned by a must-not-match case, rather
# than silently widened here. A guard that quietly does more than its spec is as
# unreviewable as one that quietly does less.
# ---------------------------------------------------------------------------

# Paths. S3.B: crates/aprender-gpu/**, crates/aprender-serve/src/cuda/**,
# *cuda*, *ptx*, *cublas*, *fp8*, *nvrtc*. Matched case-INSENSITIVELY: the
# spec writes these as concept globs over a path, and CUDA_NOTES.md is a CUDA file.
CUDA_PATH_RE='(^crates/aprender-gpu/)|(^crates/aprender-serve/src/cuda/)|(cuda)|(ptx)|(cublas)|(fp8)|(nvrtc)'

# Messages. S3.B: sm_\d+, cu[A-Z]\w+, cuda[A-Z]\w+, or a GPU architecture name.
# Matched case-SENSITIVELY - the character classes are explicitly uppercase, so
# case-folding here would silently rewrite the spec's pattern into a different one.
# The architecture list omits Turing and Pascal on purpose: "Turing complete" and
# "PascalCase" are commoner in this repository's commit messages than the 2016/2018
# parts, and a class must hold >=90% precision to stay in the blocking tier (S7
# admission rule). Both omissions are pinned as NO-MATCH rows in
# tests/fixtures/pr-review/cuda-message-cases.tsv rather than left to be rediscovered.
CUDA_MSG_RE='(sm_[0-9]+)|(cu[A-Z][A-Za-z0-9_]+)|(cuda[A-Z][A-Za-z0-9_]+)|(Blackwell)|(Hopper)|(Ada Lovelace)|(Ampere)|(Volta)'

# S3.C.1 "N x competitor". The competitor must be NAMED: a bare "2x speedup" is a
# self-relative claim with no comparator to record, and demanding an artifact hash for
# it would be exactly the over-reach the discrimination rows exist to catch.
#
# THIS PATTERN IS NOT ITS OWN. It is scripts/check_no_claim_literals.sh's RATIO_RE,
# the one #2763/PERF-049 hardened, with the two competitor lists unioned. The first
# draft here was a SECOND, independently drifting implementation of the same rule, and
# it drifted exactly where PERF-049 said it would:
#
#   `36.9x over FasterTransformer` -- the spelling APR-PERF-GATE-001 S0.1 itself uses
#   for the [X] figures, and the literal #2763 hardened its guard to catch -- did NOT
#   match here. The first draft allowed a ZERO-word gap between the ratio and the
#   competitor (a closed connector list: faster than / vs / over / of). #2763 replaced
#   exactly that closed list with a MEASURED five-word bound: over its 6900-file
#   universe the new-hit set is identical at widths 5 and 6, widths 0..6 produce zero
#   false positives, and the last true positive appears at 5 ("the 8.2x performance gap
#   between realizar and llama.cpp"). The bound is reused, not re-derived.
#
# The GAP WORD CLASS is the load-bearing half, not the bound: letters, with interior
# `.` or `-`, plus a <=3-letter abbreviation ending in a dot. NOT digits, NOT `|`, NOT
# `,`. So a markdown table row cannot be crossed, a dimension cannot start one
# (`1024x1024 torch`), and a sentence boundary stops it (`4x faster. Ollama uses ggml`
# stays clean, because `faster.` is six letters and only an abbreviation may carry a
# trailing dot -- which is what keeps `2.9x vs. Ollama` RED).
#
# MEASURED, not argued, over this repository's shipped surface as
# check_no_claim_literals.sh defines it - 6909 files at origin/main 745fa8588, read from
# a pristine worktree of that ref and NOT from a working checkout, which was 66 commits
# behind and gave four different numbers. The pattern at 1e09ca749 hit 61 lines, this one
# hits 73, ZERO of the 61 are lost, and all 12 additions are real comparative claims
# ("8.2x slower than llama.cpp", "16x convergence gap vs PyTorch", "APR is 2-3x slower
# than Ollama"). Zero false positives from the widened competitor list. The transcript is
# evidence/prrev-008/comparative-pattern-measurement.txt.
#
# ONE spelling is wider here than in #2763, and deliberately: `[[:space:]]*` before the
# multiplication sign, so `2.93 × Ollama` matches. That form is a MUST-MATCH row of the
# backtest's corpus table and #2763's RATIO_RE misses it. It cannot cross a digit (a gap
# word may not begin with one), so `1024 x 1024 torch` stays clean.
RATIO_LEFT_RE='(^|[^0-9A-Za-z.])'
MULT_RE='(x|×)'
RATIO_GAP_RE='(([A-Za-z]+([.-][A-Za-z]+)*|[A-Za-z]{1,3}\.)[[:space:]]+){0,5}'
COMPETITOR_RE='(ollama|llama\.cpp|llama-cpp|llamacpp|llama|vllm|pytorch|torch|sklearn|scikit-learn|unsloth|tensorrt|onnxruntime|onnx|transformers|huggingface|candle|burn|ggml|tinygrad|mlx|fastertransformer|sglang|lmdeploy|turbomind|tgi|orca|static[[:space:]]+batching)'
COMPARATIVE_RE="${RATIO_LEFT_RE}[0-9]+(\.[0-9]+)?[[:space:]]*${MULT_RE}[[:space:]]*${RATIO_GAP_RE}${COMPETITOR_RE}"

# S3.E: THE MODELS THAT ARE NOT A SECOND VENDOR.
#
# MEASURED, and it is the finding that changed this arm's design. `agy models` on the
# development box, 2026-08-31, returns fourteen ids - and TWO OF THEM ARE CLAUDE:
#
#   gemini-3.7-flash-{high,medium,low}   gemini-3.6-flash-{high,medium,low}
#   gemini-3.5-flash-{high,medium,low}   gemini-3.1-pro-{high,low}
#   claude-sonnet-4-6                    claude-opus-4-6-thinking
#   gpt-oss-120b-medium
#
# agy is a HARNESS, not a model. Run with no `--model`, or with a Claude one, S3.E is
# THE SAME MODEL FAMILY REVIEWING ITSELF wearing a cross-vendor label - which is exactly
# the self-preference bias S5 cites Huang et al. (ICLR'24) about, and exactly this
# repository's standing rule that a run must never be labelled by INTENT: a receipt
# reading `model_family: cross-vendor` while `--model claude-opus-4-6-thinking` ran is
# `device: GPU` printed by a build with no CUDA in it.
#
# So the model id is RECORDED as an argv element and CHECKED here. This is a
# correctness rule for the arm, not a preference: with a same-family model the arm
# delivers nothing S5 does not already have, while claiming to deliver A5's "first
# configuration that beats single-agent".
#
# THE PATTERN IS DELIBERATELY WIDE, and can be, because its cost is asymmetric. A false
# positive refuses a receipt and the reviewer picks another model id - a second of work.
# A false negative silently voids the arm's entire justification and nothing in the
# artifact says so. `opus`, `sonnet` and `haiku` are listed beside `claude` because a
# harness may expose the model without the vendor prefix, and this is the one rule here
# whose failure is invisible. Both polarities are pinned in
# tests/fixtures/pr-review/arm-e-model-cases.tsv.
ARM_E_SAME_FAMILY_RE='(claude|anthropic|opus|sonnet|haiku)'
match_arm_e_same_family() { grep -Eqi -- "$ARM_E_SAME_FAMILY_RE" <<<"$1"; }

# A TARGET says what we WANT; a CLAIM says what we GOT. Only the second needs a
# comparator recorded, because only the second asserts a measurement. Same rule, same
# spelling, as check_no_claim_literals.sh -- a bar is allowed to be a constant.
TARGET_RE='([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|FAIL:|>=|<=|[><] *[0-9])'

# S3.C surface trigger. Every token below is a DECLARATION spelling measured in this
# tree at 745fa8588, not a guess: #[arg( 2590 hits/86 files, #[command( 199/49,
# .route( 238/35, Router::new 107/37, derive(Parser 111/47, derive(Subcommand 70/47,
# long = " 84/26, short = ' 76/25, ToolDefinition 216 hits.
#
# `Command::new` is DELIBERATELY ABSENT though clap uses it: 752 hits across 293 files,
# overwhelmingly std::process::Command. A blocking-tier class must hold >=90% precision
# (S7 admission rule), and this class only fires on a receipt that claims the surface
# did NOT change -- so a false positive here calls an honest reviewer a liar.
#
# CONFIG KEYS and OUTPUT FORMATS are named by S3.C and are NOT covered. Both spellings
# measured badly: `OutputFormat` is 822 hits/103 files, nearly all internal uses of an
# enum, and an output-format change usually surfaces as an ordinary println!. Recorded
# as must-NOT-match rows in tests/fixtures/pr-review/crux-surface-cases.tsv rather than
# widened here, exactly as the CUDA tables record their two gaps.
CRUX_SURFACE_RE='(#\[arg\()|(#\[command\()|(\.route\()|(Router::new\()|(derive\(Parser)|(derive\(Subcommand)|((^|[^A-Za-z0-9_])long[[:space:]]*=[[:space:]]*")|((^|[^A-Za-z0-9_])short[[:space:]]*=[[:space:]]*.)|(ToolDefinition)'

# S3.D scope trigger. Row 1 of its table (guard-shaped files) and row 2 (Rust source)
# both TRIGGER; row 3 (docs / non-code) does not. The two rows differ in whether the
# RESULT blocks, not in whether the consultation is owed, so one predicate answers
# "is `not-triggered` a lie on this diff".
MUTATION_TRIGGER_RE='(^|/)scripts/check_[^/]*\.sh$|(^|/)scripts/dogfood\.sh$|(^|/)dogfood\.sh$|(^|/)\.github/workflows/ci\.yml$|^contracts/[^/]*\.yaml$|\.rs$'

match_cuda_path()    { grep -Eqi -- "$CUDA_PATH_RE"   <<<"$1"; }
match_cuda_message() { grep -Eq  -- "$CUDA_MSG_RE"    <<<"$1"; }
match_comparative()  { grep -Eqi -- "$COMPARATIVE_RE" <<<"$1"; }
match_target()       { grep -Eq  -- "$TARGET_RE"      <<<"$1"; }
match_crux_surface() { grep -Eq  -- "$CRUX_SURFACE_RE" <<<"$1"; }
match_mutation_trigger() { grep -Eq -- "$MUTATION_TRIGGER_RE" <<<"$1"; }

# The surface a USER READS. Scoped from check_no_claim_literals.sh's, and then MEASURED
# against 300 commits of origin/main rather than assumed -- which changed it twice.
#
# Tests, benches, examples and fixtures state TARGETS and are out of scope BY DESIGN, and
# that is not a convenience: it is what stops this guard from blocking the PR that adds
# its own case table. docs/specifications/ needs no exclusion line of its own -- no part
# of docs/ is on the inclusion list below -- but the reason #2763 excludes it is the same
# reason all of docs/ came off: a specification quoting a banned ratio in order to ban it
# is not publishing it.
#
# docs/**.md IS EXCLUDED TOO, AND THAT IS THE MEASUREMENT. The first draft here included
# it. Over the last 300 commits of origin/main the diff scan fires five times, on two
# commits, and every one is APR-PERF-GATE-001 writing ABOUT claims:
#
#   docs/benchmarking-gate-spec.md  "The book publishes a number no harness produced --
#                                    '851.8 tok/s = 2.93x Ollama'"
#   docs/benchmarking-gate-spec.md  "5.2 Failure #4 -- 2.93x Ollama from a harness that
#                                    never ran Ollama"
#   docs/benchmarking-gate-spec.md  "0.097x llama.cpp at c=16"     <- a real measurement
#   crates/apr-cli/.../server.rs    "// #2696: this printed 'Performance: 800+ tok/s
#                                    (2.8x Ollama)'"
#   crates/apr-cli/src/dispatch.rs  "// 15.7 tok/s decode, 0.099x llama.cpp"  <- real
#
# Two of the five are real comparative claims; three QUOTE a fabricated claim in order to
# ban it, and those three have NO HONEST REMEDY -- S3.C.1's exit is to record a comparator
# command, version and log, and there is no comparator log for a number that was never
# measured. A blocking class whose only exit is to fabricate the evidence it demands is
# worse than the hole it closes, and S7's admission rule already forbids it: 2/5 is 40%
# measured precision against a >=90% bar.
#
# So B4's diff half blocks on book/**.md and on PRINTED literals and doc comments in
# shipped .rs -- the surfaces #2763 measured as user-facing, and where the 2.93x Ollama
# claim was actually published. Over the same 300 commits that scope fires ZERO times,
# which is stated as what it is: no measured false positives AND no measured true
# positives. It is not evidence of precision, it is evidence that this repository has not
# published a competitor ratio to the book in 300 commits.
#
# RESIDUAL, recorded rather than hidden: a comparative claim added to docs/ prose or to a
# plain `//` comment is NOT blocked by B4. Two real ones are named above.
#
# F6 -- THE BOOK IS PROSE, AND ITS DIRECTORY NAMES ARE CHAPTER NAMES, NOT A RUST LAYOUT.
# The exclusion list below was scoped from a Rust project's layout, where `examples/` is
# a cargo target directory. Applied to `book/**` it removed `book/src/examples/` -- and
# that is 153 of the book's 441 published .md pages, 34.7%, every one of them listed in
# book/src/SUMMARY.md on origin/main, so every one a rendered mdBook chapter. It is also
# EXACTLY where `851.8 tok/s = 2.93x Ollama` was published, at da069a25f, from a harness
# that never ran Ollama -- the scar S3.C.1, S9 and S11 are all written about.
#
# Measured, not argued (evidence/pr-review/backtest/results-v3.md):
#   B4 over da069a25f^..da069a25f, book/ excluded  -> 0 fires    (the claim is ACCEPTED)
#   B4 over the same diff, book/ exempted          -> 2 fires    (both the real claim)
#   false positives over all 153 current book/src/examples/ pages          -> 0
#   false positives over every added book/** line in 300 commits of main   -> 0
#
# So the book exemption is checked FIRST, ahead of all four exclusion lines, not merely
# ahead of the benches|examples one: a future `book/src/tests/` chapter is a chapter too
# (none exists today, and this is the difference between fixing the instance and fixing
# the class). The Rust exclusions keep their full force everywhere else, which the case
# table pins with `crates/aprender-core/examples/demo.rs` -> NO-MATCH beside
# `book/src/examples/...` -> MATCH: same directory name, opposite verdict, one variable.
match_shipped_surface() {
  case "$1" in
    book/*.md) return 0 ;;
  esac
  case "$1" in
    tests/*|*/tests/*|test/*|*/test/*)         return 1 ;;
    benches/*|*/benches/*|examples/*|*/examples/*) return 1 ;;
    fixtures/*|*/fixtures/*|fixture/*|*/fixture/*) return 1 ;;
    *_test.rs|*_tests.rs)                      return 1 ;;
  esac
  # docs/** carries no exclusion line because it carries no INCLUSION line: the list
  # below is what B4 scans, and docs/ is not on it. An explicit `docs/*) return 1` was
  # written here first and the mutation sweep reported it SURVIVED - a dead branch no
  # receipt can reach, which is the same shape as a rule nothing tests. The exclusion
  # that matters is the absence below, and `docs-prose-back-in-b4-scope` in
  # scripts/mutate-guard.sh mutates exactly that line; `book-removed-from-b4-scope` and
  # `book-examples-back-out-of-scope` mutate the book exemption above it.
  case "$1" in
    crates/*/src/*.rs|src/*.rs) return 0 ;;
  esac
  return 1
}

# In a .rs file a claim reaches a user through a PRINTED literal or a doc comment
# (`cargo doc` publishes the second). A plain `//` comment does not, and this repository
# writes a lot of them ABOUT claims it has withdrawn. Same split, same spelling, as
# check_no_claim_literals.sh, which applies its ratio pattern to print macros and to
# ///|//! and to nothing else in .rs.
RS_PUBLISHED_RE='(println!|eprintln!|write!|writeln!|format!|\.red\(\)|\.green\(\)|\.yellow\(\)|\.cyan\(\))|(^[[:space:]]*(///|//!))'
match_rs_published() { grep -Eq -- "$RS_PUBLISHED_RE" <<<"$1"; }

# published_claim <path> <line> - S3.C.1's subject: a competitor ratio a USER READS.
# ONE definition, called by B4's diff scan and by the S3.C claim trigger. Two copies of
# a scoping rule drift, and the drift is invisible because each keeps passing against
# its own copy -- which is exactly how B4's pattern came to be a second, weaker
# implementation of #2763's RATIO_RE.
published_claim() {
  match_shipped_surface "$1" || return 1
  case "$1" in *.rs) match_rs_published "$2" || return 1 ;; esac
  match_comparative "$2" || return 1
  if match_target "$2"; then return 1; fi
  return 0
}

case "${1-}" in
  --match-path)        match_cuda_path    "${2?--match-path needs an argument}";        exit $? ;;
  --match-message)     match_cuda_message "${2?--match-message needs an argument}";     exit $? ;;
  --match-comparative) match_comparative  "${2?--match-comparative needs an argument}"; exit $? ;;
  --match-shipped-surface) match_shipped_surface "${2?--match-shipped-surface needs an argument}"; exit $? ;;
  --match-crux-surface)    match_crux_surface    "${2?--match-crux-surface needs an argument}";    exit $? ;;
  --match-mutation-trigger) match_mutation_trigger "${2?--match-mutation-trigger needs an argument}"; exit $? ;;
  --match-target)      match_target       "${2?--match-target needs an argument}";      exit $? ;;
  --match-rs-published) match_rs_published "${2?--match-rs-published needs an argument}"; exit $? ;;
  --match-arm-e-same-family) match_arm_e_same_family "${2?--match-arm-e-same-family needs an argument}"; exit $? ;;
  -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
esac

# ---------------------------------------------------------------------------
# Tools. Absent tool => rejection, never a skip.
# ---------------------------------------------------------------------------
MISSING_TOOLS=""
for t in git jq sha256sum check-jsonschema minisign; do
  command -v "$t" >/dev/null 2>&1 || MISSING_TOOLS="$MISSING_TOOLS $t"
done
if [ -n "$MISSING_TOOLS" ]; then
  echo "$PROG: FAIL - cannot run:$MISSING_TOOLS not on PATH." >&2
  echo "  A gate that cannot execute its own checks must not report green (S6.2)." >&2
  exit 1
fi

SCHEMA_DIR=${PR_REVIEW_SCHEMA_DIR:-schemas}
PUBKEY=${PR_REVIEW_PUBKEY:-.github/pr-review.pub}
REPO=${PR_REVIEW_REPO:-}
if [ -z "$REPO" ]; then
  REPO=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "$PROG: FAIL - not in a git repository and PR_REVIEW_REPO is unset." >&2
    exit 1
  }
fi

VERDICTS='PASS FINDINGS DEGRADED BLOCK'
PREDICATE_TYPE='https://paiml.dev/attestations/pr-review/v2'

# S3.E is owed by receipts written at or after this skill version, and by no other.
#
# A RECEIPT IS A RECORD OF A REVIEW THAT HAPPENED, and this repository already holds one
# - evidence/pr-review/2795/f5fe1479.../ - written at 2.0.0, before the arm existed. The
# alternative to this gate was to back-fill an `antigravity` block into it so it would
# keep validating. That is fabricating the evidence S3.C.1 exists to demand: a
# consultation record for a consultation nobody performed, which is the never-ran-Ollama
# shape with a JSON schema in front of it. So a 2.0.0 receipt is validated by 2.0.0's
# rules and stays honest, and 2.1.0 owes the arm.
#
# THE HOLE THIS LEAVES IS STATED RATHER THAN PAPERED OVER: a reviewer who writes
# `skill_version: 2.0.0` skips S3.E, and this guard cannot tell that from a genuine
# 2.0.0 receipt, because both are exactly the same bytes. Closing it needs a check that
# reads the TREE's current skill version and requires this PR's receipt to match it -
# scripts/check_pr_review_arm4.sh's job, owed as PRREV-018, and deliberately NOT invented
# here: S8 forbids a threshold nobody measured, and the arm it would protect is advisory
# with zero samples. An advisory arm with a stated bypass is worth more than a blocking
# one with an unstated closure.
ARM_E_MIN_VERSION=2.1.0

# version_ge A B - 0 when A >= B under version ordering.
#
# `sed -n 1p`, NOT `head -1`: head exits after the first line, hands sort SIGPIPE, and
# under `set -o pipefail` the command substitution reports 141 for a pipeline that
# produced exactly the right answer. That shape landed four times in this repository in
# one day. sed without `q` reads its input to the end and closes no pipe.
version_ge() {
  lo=$(printf '%s\n%s\n' "$1" "$2" | sort -V | sed -n 1p)
  [ "$lo" = "$2" ] || [ "$1" = "$2" ]
}

REJECT_CLASS=''
REJECT_REASON=''
reject() { REJECT_CLASS=$1; REJECT_REASON=$2; return 1; }

# sha256 of stdin, bare hex.
sha256_stdin() { sha256sum | cut -d' ' -f1; }

# ---------------------------------------------------------------------------
# THE DIFF, READ BY THE GUARD ITSELF.
#
# Every recomputation below reads `git diff`, never the receipt's own account of it.
# A blocking class that decides whether to block by consulting the report of the party
# it may be about to block is circular, and B4 was exactly that: `match_comparative`
# had ONE call site, inside a loop over findings THE REVIEWER WROTE. A diff publishing
# `2.93x Ollama` was ACCEPTED when the receipt was silent about it, and REJECTED when
# the reviewer chose to mention it -- same diff, same empty comparative_claims[], the
# verdict turning only on the reviewer's candour. That is "no claim" and "did not look"
# being the same artifact, which is the distinction S3.0 exists to make impossible.
# ---------------------------------------------------------------------------

# changed_lines <base> <head> [+|+-]
# Emits  <path><TAB><line text>  for added (`+`) or added-and-removed (`+-`) lines.
# Removed lines matter for a SURFACE change: deleting a flag changes the CLI too.
changed_lines() {
  local b=$1 h=$2 want=${3:-+}
  git -C "$REPO" diff --unified=0 "$b" "$h" 2>/dev/null \
  | awk -v want="$want" '
      # The four header forms are matched EXACTLY. `/^--- /` would also swallow a
      # deleted line whose own text begins with two dashes, and `/^\+\+\+ /` an added
      # line beginning with two pluses: the diff prefix makes them indistinguishable
      # from a header unless the a/ b/ /dev/null shape is required.
      /^\+\+\+ b\//          { file = substr($0, 7); next }
      /^\+\+\+ \/dev\/null/  { file = "?"; next }
      /^--- a\//             { next }
      /^--- \/dev\/null/     { next }
      /^\+/ { print file "\t" substr($0, 2); next }
      /^-/  { if (want == "+-") print file "\t" substr($0, 2) }
    '
}

# -------------------------------------------------------------------------
# validate_receipt <dir>
# 0 = accept. 1 = reject, with REJECT_CLASS / REJECT_REASON set.
# Ordering is load-bearing and follows the contract's preconditions: the receipt
# parses against the vendored schema before any mark is read, and the signature
# verifies before any blocking class is evaluated.
# -------------------------------------------------------------------------
validate_receipt() {
  local dir=$1
  local rcpt="$dir/receipt.intoto.jsonl"
  local sarif="$dir/findings.sarif"
  local sig="$rcpt.minisig"

  # --- B1: presence. A missing receipt is RED, not skipped (S6.3). -----------
  [ -d "$dir" ]   || reject B1 "no such receipt directory: $dir" || return 1
  [ -f "$rcpt" ]  || reject B1 "receipt.intoto.jsonl is missing - a missing receipt is RED, not skipped, per S6.3" || return 1
  [ -f "$sarif" ] || reject B1 "findings.sarif is missing" || return 1

  # --- B1: it is JSON Lines: exactly one JSON document, on one line. ---------
  # RECORDS, not newlines. `wc -l` counts line TERMINATORS, so a perfectly valid
  # single-record file with no trailing newline reports 0 and an earlier draft of this
  # block rejected it. bashrs surfaced the dead `lines=$((lines))` that hid the bug.
  local newlines lastbyte records
  newlines=$(wc -l < "$rcpt")
  lastbyte=$(tail -c1 "$rcpt")
  records=$newlines
  if [ -n "$lastbyte" ]; then
    records=$((newlines + 1))
  fi
  if [ "$records" -ne 1 ]; then
    reject B1 "receipt.intoto.jsonl holds $records JSON record/s; the in-toto Statement must be exactly one" || return 1
  fi
  jq -e . "$rcpt"  >/dev/null 2>&1 || reject B1 "receipt.intoto.jsonl is not parseable JSON" || return 1
  jq -e . "$sarif" >/dev/null 2>&1 || reject B1 "findings.sarif is not parseable JSON" || return 1

  # --- B1: schema gate, offline, against the vendored copies (S6.2). ---------
  local sout
  if ! sout=$(check-jsonschema --schemafile "$SCHEMA_DIR/in-toto-statement-v1.json" "$rcpt" 2>&1); then
    reject B1 "receipt fails schemas/in-toto-statement-v1.json: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-220)" || return 1
  fi
  if ! sout=$(check-jsonschema --schemafile "$SCHEMA_DIR/sarif-2.1.0.json" "$sarif" 2>&1); then
    reject B1 "findings.sarif fails schemas/sarif-2.1.0.json: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-220)" || return 1
  fi

  # --- B1: signature, before any blocking class is evaluated (S4.3). --------
  # The signature proves the receipt came from the signing environment. It does NOT
  # prove the review was honest, and the PR comment says so (S4.3, R1: L1-self).
  [ -f "$PUBKEY" ] || reject B1 "public key $PUBKEY is absent; an unverifiable signature is not a verified one" || return 1
  [ -f "$sig" ]    || reject B1 "receipt is unsigned - no $sig" || return 1
  if ! sout=$(minisign -V -m "$rcpt" -p "$PUBKEY" 2>&1); then
    reject B1 "signature does not verify against $PUBKEY: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-160)" || return 1
  fi

  # --- read the predicate ---------------------------------------------------
  local ptype alevel head base verdict author reviewer subj_sha skill_ver
  ptype=$(jq -r '.predicateType // ""' "$rcpt")
  [ "$ptype" = "$PREDICATE_TYPE" ] || reject B1 "predicateType is '$ptype', expected '$PREDICATE_TYPE'" || return 1

  alevel=$(jq -r '.predicate.attestation_level // ""' "$rcpt")
  [ "$alevel" = "L1-self" ] || reject B1 "attestation_level is '$alevel'; a skill invoked by the authoring agent is self-attestation, and R1 requires it to say so" || return 1

  # skill_version decides which rules this receipt is judged by (ARM_E_MIN_VERSION
  # above), so an absent one is not a cosmetic omission: it is a receipt that does not
  # say which contract it was written against.
  skill_ver=$(jq -r '.predicate.skill_version // ""' "$rcpt")
  [ -n "$skill_ver" ] \
    || reject B1 "predicate.skill_version is absent; the version is what selects the rule set this receipt is judged by, so a receipt that omits it cannot be judged against any" || return 1

  head=$(jq -r '.predicate.head_sha // ""' "$rcpt")
  base=$(jq -r '.predicate.base_sha // ""' "$rcpt")
  verdict=$(jq -r '.predicate.verdict // ""' "$rcpt")
  author=$(jq -r '.predicate.author_actor.id // ""' "$rcpt")
  reviewer=$(jq -r '.predicate.reviewer_actor.id // ""' "$rcpt")
  subj_sha=$(jq -r '.subject[0].digest.sha1 // ""' "$rcpt")

  [ -n "$head" ] || reject B1 "predicate.head_sha is absent" || return 1
  [ -n "$base" ] || reject B1 "predicate.base_sha is absent" || return 1
  [ "$subj_sha" = "$head" ] || reject B1 "the sha1 digest of subject 0 is $subj_sha but the predicate reviews head_sha $head" || return 1
  grep -qx -- "$verdict" <<<"$(tr ' ' '\n' <<<"$VERDICTS")" \
    || reject B1 "verdict '$verdict' is outside { PASS FINDINGS DEGRADED BLOCK }" || return 1

  # --- B1: the findings the receipt points at are the findings on disk. -----
  local fref_path fref_sha actual_sha
  fref_path=$(jq -r '.predicate.findings_ref.path // ""' "$rcpt")
  fref_sha=$(jq -r '.predicate.findings_ref.sha256 // ""' "$rcpt")
  [ "$fref_path" = "findings.sarif" ] || reject B1 "findings_ref.path is '$fref_path', expected 'findings.sarif'" || return 1
  actual_sha=$(sha256_stdin < "$sarif")
  [ "$fref_sha" = "$actual_sha" ] \
    || reject B1 "findings_ref.sha256 ($fref_sha) is not sha256(findings.sarif) ($actual_sha)" || return 1

  # --- B1: the cost block. Record-only is not unenforced (contract S8). ------
  jq -e '.predicate.cost | (.input_tokens|type=="number") and (.output_tokens|type=="number") and (.wall_seconds|type=="number")' \
      "$rcpt" >/dev/null 2>&1 \
    || reject B1 "predicate.cost must carry numeric input_tokens, output_tokens and wall_seconds; record-only is not unenforced" || return 1

  # --- B2: author / reviewer separation (S5). -------------------------------
  [ -n "$author" ]   || reject B1 "author_actor.id is absent"   || return 1
  [ -n "$reviewer" ] || reject B1 "reviewer_actor.id is absent" || return 1
  [ "$reviewer" != "$author" ] \
    || reject B2 "reviewer_actor.id = author_actor.id = '$author'; a self-review is not a review (S5)" || return 1

  # --- B1: the diff boundary really is the merge base (S2, row 10). ---------
  # This is the fix for diff-scope pollution by other agents' merges in the
  # parallel-worktree workflow: a floating base pulls their commits into the review.
  git -C "$REPO" cat-file -e "${head}^{commit}" 2>/dev/null \
    || reject B1 "head_sha $head does not resolve to a commit in $REPO" || return 1
  local computed_base
  computed_base=$(git -C "$REPO" merge-base refs/remotes/origin/main "$head" 2>/dev/null) \
    || reject B1 "cannot compute merge-base(origin/main, $head) in $REPO" || return 1
  [ "$base" = "$computed_base" ] \
    || reject B1 "base_sha $base is not git merge-base origin/main $head (= $computed_base); the diff scope of this review is not the merge base (S2)" || return 1

  # --- consultation statuses -----------------------------------------------
  local pmat_st cuda_st crux_st mut_st ag_st
  pmat_st=$(jq -r '.predicate.consultations.pmat.status // ""' "$rcpt")
  cuda_st=$(jq -r '.predicate.consultations.cuda.status // ""' "$rcpt")
  crux_st=$(jq -r '.predicate.consultations.crux.status // ""' "$rcpt")
  mut_st=$(jq  -r '.predicate.consultations.mutation.status // ""' "$rcpt")
  ag_st=$(jq   -r '.predicate.consultations.antigravity.status // ""' "$rcpt")

  # --- S3.E: is the second-vendor arm owed, and is it here? ------------------
  # TWO independent facts, because they fail differently. `arm_e_required` comes from
  # the receipt's declared version; `arm_e_present` from whether the block exists. A
  # 2.0.0 receipt that carries the block anyway is still checked in full - a block
  # nothing validates is worse than no block, and the version gate exists to spare the
  # honest historical receipt, not to open an unchecked field.
  local arm_e_required=0 arm_e_present=0
  if version_ge "$skill_ver" "$ARM_E_MIN_VERSION"; then arm_e_required=1; fi
  if jq -e '.predicate.consultations | has("antigravity")' "$rcpt" >/dev/null 2>&1; then
    arm_e_present=1
  fi
  if [ "$arm_e_required" -eq 1 ] && [ "$arm_e_present" -eq 0 ]; then
    reject B1 "skill_version $skill_ver owes the S3.E antigravity consultation and consultations.antigravity is absent; S3.E's trigger is unconditional, exactly as S3.A's is, and an absent consultation is indistinguishable from one that found nothing (S3.0)" || return 1
  fi

  # The vocabulary and the unreachable rule are applied over a LIST, and antigravity
  # joins it rather than getting a private copy of either rule. Two implementations of
  # one rule drift, and each stays green against its own copy - F4 and D8, in the guard
  # that implements F4.
  local -a CONSULT=(pmat cuda crux mutation)
  if [ "$arm_e_present" -eq 1 ]; then CONSULT+=(antigravity); fi

  local k st
  for k in "${CONSULT[@]}"; do
    st=$(jq -r --arg k "$k" '.predicate.consultations[$k].status // ""' "$rcpt")
    case "$st" in
      consulted|not-triggered|unreachable) ;;
      "") reject B1 "consultations.$k.status is absent; an omitted consultation is indistinguishable from one that found nothing (S3.0)" || return 1 ;;
      *)  reject B1 "consultations.$k.status is '$st', outside { consulted, not-triggered, unreachable }" || return 1 ;;
    esac
  done

  # --- B1: an unreachable source must not read clean (S3.0, rows 5 and 6). --
  # This is also S3.E's `unavailable` state. agy can fail SLOWLY - `--print-timeout`
  # defaults to 5m and a repository-scale review needs more - so a timeout is recorded
  # as `unreachable`, never as a run that found nothing. The two are the same artifact
  # otherwise, which is the distinction S3.0 exists to make impossible.
  for k in "${CONSULT[@]}"; do
    st=$(jq -r --arg k "$k" '.predicate.consultations[$k].status // ""' "$rcpt")
    if [ "$st" = "unreachable" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "consultations.$k is unreachable but the verdict is PASS; an unreachable source must be DEGRADED, not clean (S3.0)" || return 1
    fi
  done

  # =========================================================================
  # EVERY CONSULTATION GETS BOTH HALVES.
  #
  # Audited before this block was written: only cuda's trigger was recomputed from the
  # diff, only mutation's emptiness was checked, and NO consultation had both. So
  # `cuda: consulted, queries: []` was ACCEPTED while the analogous
  # `mutation.attempted: 0` was rejected, and `pmat: not-triggered` was ACCEPTED on a
  # code PR though S3.A calls pmat unconditional. Half a rule per consultation is not
  # three-quarters of a gate; it is four different gates, three of which cannot fail.
  #
  #                       trigger recomputed        emptiness checked
  #   pmat                unconditional (S3.A)      the four S3.A arrays
  #   cuda                path + message (S3.B)     queries[] non-empty, well-formed
  #   crux                surface + claim (S3.C)    surfaces[] or claims[] non-empty
  #   mutation            file shape (S3.D)         attempted > 0, counts coherent
  # =========================================================================

  # --- the diff, read once. ------------------------------------------------
  local changed_files commit_msgs
  changed_files=$(git -C "$REPO" diff --name-only "$base" "$head" 2>/dev/null || true)
  commit_msgs=$(git -C "$REPO" log --format=%B "$base..$head" 2>/dev/null || true)

  # --- B1: pmat is UNCONDITIONAL, so not-triggered is never true of it. -----
  # S3.A: "Trigger: unconditional", and S8.4 repeats it -- "pmat always (cheap,
  # deterministic); CUDA/CRUX/mutation trigger on shape". S6.3's row 7 reads "all
  # consultations not-triggered", which contradicts both; the two normative statements
  # win over the illustrative row, and row 7 now carries `pmat: consulted` with four
  # empty arrays. That fixture previously blessed this exact defect with a
  # trigger_reason reading "pmat is unconditional; not-triggered is never correct for
  # it" -- a fixture stating the rule it exempted.
  [ "$pmat_st" != "not-triggered" ] \
    || reject B1 "consultations.pmat is not-triggered, but S3.A makes pmat unconditional on every PR; an unmeasured CB-200 is Skip, and Skip is not a pass" || return 1

  # --- B1: the CUDA consultation was skipped while its trigger fired (row 1).
  if [ "$cuda_st" = "not-triggered" ]; then
    local f fired=''
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      if match_cuda_path "$f"; then fired="path $f"; break; fi
    done <<<"$changed_files"
    if [ -z "$fired" ] && [ -n "$commit_msgs" ] && match_cuda_message "$commit_msgs"; then
      fired="commit message"
    fi
    [ -z "$fired" ] \
      || reject B1 "consultations.cuda is not-triggered but its S3.B trigger fires on this diff ($fired); 'the docs said nothing' and 'I did not ask' must not be the same artifact" || return 1
  fi

  # --- B1: the CRUX consultation was skipped while its trigger fired. -------
  # Two routes, because S3.C has two: a changed surface DECLARATION, and S3.C.1's
  # comparative claim. Both are recomputed from the diff.
  if [ "$crux_st" = "not-triggered" ]; then
    local cf cl crux_fired=''
    while IFS=$'\t' read -r cf cl; do
      [ -n "$cf" ] || continue
      case "$cf" in
        tests/*|*/tests/*|test/*|*/test/*|benches/*|*/benches/*|examples/*|*/examples/*|fixtures/*|*/fixtures/*) continue ;;
      esac
      if match_crux_surface "$cl"; then crux_fired="surface declaration in $cf"; break; fi
    done < <(changed_lines "$base" "$head" '+-')
    if [ -z "$crux_fired" ]; then
      while IFS=$'\t' read -r cf cl; do
        [ -n "$cf" ] || continue
        published_claim "$cf" "$cl" || continue
        crux_fired="comparative claim in $cf"; break
      done < <(changed_lines "$base" "$head" '+')
    fi
    [ -z "$crux_fired" ] \
      || reject B1 "consultations.crux is not-triggered but its S3.C trigger fires on this diff ($crux_fired); a surface nobody looked at is not a surface with no gap" || return 1
  fi

  # --- B1: the MUTATION consultation was skipped while its trigger fired. ---
  # S3.D triggers on the FILE SHAPE. Rows 1 and 2 of its table differ in whether the
  # result BLOCKS, not in whether the consultation is owed; only row 3 (docs/non-code)
  # is untriggered. `unreachable` stays open as the honest path -- it is DEGRADED, and
  # DEGRADED proceeds on a feature branch (S7).
  if [ "$mut_st" = "not-triggered" ]; then
    local mf mut_fired=''
    while IFS= read -r mf; do
      [ -n "$mf" ] || continue
      if match_mutation_trigger "$mf"; then mut_fired="$mf"; break; fi
    done <<<"$changed_files"
    [ -z "$mut_fired" ] \
      || reject B1 "consultations.mutation is not-triggered but S3.D triggers on this diff ($mut_fired); only docs and non-code are untriggered" || return 1
  fi

  # --- B1: pmat consulted must SHOW the four things S3.A requires. ---------
  # An ABSENT key and an EMPTY array are the difference between "did not look" and
  # "looked and found nothing" -- S3.0's whole subject, one level up from the SARIF.
  # duplication_hits is the one S3.A calls "the highest-EV field in the receipt".
  if [ "$pmat_st" = "consulted" ]; then
    local pmat_missing
    # TWO jq traps in four lines, both caught by a probe rather than by reading.
    # (a) `$k` must be bound BEFORE has() is applied: in `$p|has(.)` the argument `.`
    #     is evaluated against has()'s own INPUT, which is $p, so it asks whether the
    #     object has a key named after itself - always false, and the check passed over
    #     everything.
    # (b) `getpath([$k])`, not `$p[$k]`: bashrs parses the jq program as shell and
    #     raises SC1087 (array expansion) on the second form. scripts/ is gated on a
    #     SHRINK-ONLY bashrs error count, so two false errors here are two someone else
    #     has to triage. Same class as the \042 dance in check_no_claim_literals.sh.
    pmat_missing=$(jq -r '.predicate.consultations.pmat as $p
        | ["complexity_delta","tdg_delta","satd_introduced","duplication_hits"]
        | map(. as $k | select((($p | has($k)) | not) or (($p | getpath([$k]) | type) != "array")))
        | join(", ")' "$rcpt")
    [ -z "$pmat_missing" ] \
      || reject B1 "pmat.status is consulted but these S3.A outputs are absent or are not arrays: $pmat_missing; an absent field is 'did not look', which S3.0 requires to be distinguishable from an empty one" || return 1
  fi

  # --- B1: cuda consulted must have ASKED something. -----------------------
  # The analogue of mutation.attempted = 0, and the one S8 counts as a
  # `vacuous_consultation`. A `no-authority-found` entry IS a query, so the honest
  # path -- "I asked and the corpus had nothing" -- stays open; what closes is
  # recording a consultation that asked nothing at all.
  if [ "$cuda_st" = "consulted" ]; then
    local n_queries bad_query
    n_queries=$(jq -r '[.predicate.consultations.cuda.queries[]?] | length' "$rcpt")
    [ "$n_queries" -gt 0 ] \
      || reject B1 "cuda.status is consulted with queries: []; a consultation that asked nothing is DEGRADED, not clean, exactly as mutation.attempted=0 is (S3.B, S8 vacuous_consultations)" || return 1
    # `first`, NOT `| head -1`. An early-exiting reader hands the producer SIGPIPE and
    # `set -o pipefail` then reports 141 for a command substitution that produced exactly
    # the right answer. Four instances of that shape landed in this repository in one
    # day, one of them a check that PASSED on the error. jq can take the first element
    # itself, so there is no second process to close the pipe.
    bad_query=$(jq -r '[
        [ .predicate.consultations.cuda.queries[]? ] | to_entries[]
        | .key as $i | .value as $q
        | if (($q.q // "") | tostring | length) == 0 then "#\($i) has an empty q"
          elif ((["found","no-authority-found"]) | index($q.result | tostring)) == null
            then "#\($i) records result \"\($q.result)\", outside { found, no-authority-found }"
          elif ($q.result == "found") and ((($q.excerpt_sha256 // "") | tostring | length) == 0)
            then "#\($i) records result found with no excerpt_sha256"
          else empty end ] | first // ""' "$rcpt")
    [ -z "$bad_query" ] \
      || reject B1 "cuda query $bad_query; S3.B requires every device-behaviour claim to carry either a citation or a NAMED query that returned nothing" || return 1
  fi

  # --- B1: crux consulted must have LOOKED at something. -------------------
  if [ "$crux_st" = "consulted" ]; then
    local crux_shape n_surfaces n_claims coverage gap
    crux_shape=$(jq -r '.predicate.consultations.crux as $c
        | ["surfaces","comparative_claims"]
        | map(. as $k | select((($c | has($k)) | not) or (($c | getpath([$k]) | type) != "array")))
        | join(", ")' "$rcpt")
    [ -z "$crux_shape" ] \
      || reject B1 "crux.status is consulted but these S3.C outputs are absent or are not arrays: $crux_shape" || return 1
    n_surfaces=$(jq -r '[.predicate.consultations.crux.surfaces[]?] | length' "$rcpt")
    n_claims=$(jq -r   '[.predicate.consultations.crux.comparative_claims[]?] | length' "$rcpt")
    [ "$n_surfaces" -gt 0 ] || [ "$n_claims" -gt 0 ] \
      || reject B1 "crux.status is consulted with no surfaces and no comparative claims; a consultation over nothing passes vacuously, which is the shape S3.D calls DEGRADED" || return 1
    coverage=$(jq -r '.predicate.consultations.crux.crux_coverage // ""' "$rcpt")
    gap=$(jq -r      '.predicate.consultations.crux.gap_effect    // ""' "$rcpt")
    case "$coverage" in covered|none) ;;
      *) reject B1 "crux.crux_coverage is '$coverage', outside { covered, none }; S3.C makes 'no contract covers this surface' a FINDING, not a blank" || return 1 ;;
    esac
    case "$gap" in closes|widens|none) ;;
      *) reject B1 "crux.gap_effect is '$gap', outside { closes, widens, none }" || return 1 ;;
    esac
  fi

  # --- B1: a consulted-but-vacuous mutation run (row 2). -------------------
  # A mutation set that matches nothing passes vacuously - the same shape as
  # `pv lint <FILE>` returning PASS over zero contracts.
  if [ "$mut_st" = "consulted" ]; then
    local attempted killed n_survivors
    attempted=$(jq -r '.predicate.consultations.mutation.attempted // "null"' "$rcpt")
    killed=$(jq -r    '.predicate.consultations.mutation.killed    // "null"' "$rcpt")
    case "$attempted" in
      ''|null|*[!0-9]*) reject B1 "mutation.status is consulted but attempted is '$attempted'" || return 1 ;;
    esac
    [ "$attempted" -gt 0 ] \
      || reject B1 "mutation.status is consulted with attempted=0; a run that attempts nothing is DEGRADED, not clean (S3.D)" || return 1
    case "$killed" in ''|null|*[!0-9]*) reject B1 "mutation.killed is '$killed', not a count" || return 1 ;; esac
    [ "$killed" -le "$attempted" ] \
      || reject B1 "mutation records killed=$killed of attempted=$attempted; a score above one is a miscount, and guard_mutation_score is read from these two numbers" || return 1
    # S3.D: "Surviving mutants are recorded with mutant, file, line, killed: false."
    # A survivor count that does not match the arithmetic makes the survivors list -
    # the only part a reader can act on - unfalsifiable.
    # `?` SUPPRESSES THE TYPE ERROR, NOT THE VALUE. `[ "12 survived"[]? ] | length` is
    # 0 - not an error and not 12 - so a survivors field holding a STRING, a null, or
    # nothing at all made this arithmetic agree with attempted == killed, and the clause
    # below could not fire. An adversarial verifier shipped exactly that receipt through
    # this guard and then through S13's arm script, which reads the same field with the
    # same idiom. The type is established before the length is believed.
    case "$(jq -r 'if (.predicate.consultations.mutation | has("survivors")) then (.predicate.consultations.mutation.survivors | type) else "absent" end' "$rcpt")" in
      array) ;;
      *) reject B1 "mutation.status is consulted and survivors is $(jq -r 'if (.predicate.consultations.mutation | has("survivors")) then "a " + (.predicate.consultations.mutation.survivors | type) else "absent" end' "$rcpt"), not a list; jq counts a non-list survivors field as EMPTY, so 'not recorded' and 'none survived' become the same artifact (S3.0, S3.D)" || return 1 ;;
    esac
    n_survivors=$(jq -r '[.predicate.consultations.mutation.survivors[]?] | length' "$rcpt")
    [ "$n_survivors" -eq $((attempted - killed)) ] \
      || reject B1 "mutation records attempted=$attempted killed=$killed, so $((attempted - killed)) mutant/s survived, but survivors[] holds $n_survivors; S3.D requires every survivor to be named" || return 1
  fi

  # =========================================================================
  # S3.E - THE FOURTH-VENDOR ARM.
  #
  # S3.A..S3.D consult SOURCES: an index, a documentation corpus, a contract set, a
  # mutation run. S3.E consults a different REVIEWING AGENT, from a different vendor
  # and a different model family, in its own process with its own tools. That is what
  # makes it a stronger form of S5's separation than a second prompt of this model:
  # S5 cites Huang et al. (ICLR'24) on self-preference bias and on intrinsic
  # self-correction degrading reasoning, and neither result is escaped by asking the
  # same family twice. A5 calls a separate grounded critic the first configuration
  # that beats single-agent; a same-family critic is not one.
  #
  # EVERY RULE BELOW IS ABOUT THE RECORD, NOT ABOUT THE FINDINGS. The arm is advisory
  # (zero samples, S7's admission rule), so the guard may not act on what agy SAID. It
  # may only refuse a receipt that misdescribes what agy DID.
  # =========================================================================
  if [ "$arm_e_present" -eq 1 ]; then
    # S3.E's trigger is unconditional - every PR - for the same reason S3.A's is, and
    # the reason is not cost. A shape-based trigger exempts exactly the diffs where an
    # independent reader is worth most: the small ones that look obvious, which is what
    # every PR in S9's spine looked like to its author. All four carry `reviews=0,
    # comments=0`, and S9.3 records that S5's separation "has still never been exercised
    # on this epic". Cost is instrumented instead (`usage`, below), so a threshold can be
    # DERIVED from 30 samples later rather than guessed now - S10 row 8.4's argument,
    # reused because it is the same argument.
    [ "$ag_st" != "not-triggered" ] \
      || reject B1 "consultations.antigravity is not-triggered, but S3.E's trigger is unconditional on every PR exactly as S3.A's is; a shape trigger would exempt the small diffs that look obvious, which is every PR in S9's spine" || return 1

    if [ "$ag_st" = "consulted" ]; then
      local ag_attempted ag_ident ag_usage ag_div ag_nfind ag_sum

      # --- vacuity. The rule S8 fixes at zero, applied to the fifth arm. ----
      # `attempted` is the number of agy INVOCATIONS this consultation made. Same
      # shape, same zero, as mutation.attempted and cuda.queries[]: a consultation
      # recorded as performed that performed nothing passes vacuously.
      ag_attempted=$(jq -r '.predicate.consultations.antigravity.attempted // "null"' "$rcpt")
      case "$ag_attempted" in
        ''|null|*[!0-9]*) reject B1 "antigravity.status is consulted but attempted is '$ag_attempted', which is not a count of agy invocations" || return 1 ;;
      esac
      [ "$ag_attempted" -gt 0 ] \
        || reject B1 "antigravity.status is consulted with attempted=0; a consultation that invoked nothing is DEGRADED, not clean, exactly as mutation.attempted=0 and cuda.queries=[] are (S8 vacuous_consultations = 0)" || return 1

      # --- WHICH BINARY, AND WHOSE MODEL. ----------------------------------
      # This repository has had four `apr` binaries coexist and a bare invocation
      # resolve to a 26-day-old one; the standing rule is to resolve explicitly and
      # record what was resolved. `binary_path` is the OUTPUT of that resolution, which
      # is the opposite of a hardcoded path - it is provenance, recorded per run.
      #
      # `model_id` is what makes "cross-vendor" a CHECKABLE claim rather than an
      # asserted one - it is the argv value, and the rule below reads it.
      # `model_family` is the human-readable label beside it and is deliberately NOT
      # what the rule reads: a label is what a receipt can get wrong for free, and
      # row 35 is exactly a receipt whose model_family says google/gemini while its
      # model_id says claude-opus-4-6-thinking. Without model_id the arm's whole
      # justification - that it is not the same family reviewing itself, S5 - rests on
      # nothing in the artifact an automated check can reach.
      #
      # The version is RECORDED, NOT PINNED. A pinned version makes the arm fail closed
      # on an upgrade, and S7 has no measured reason to block anything here; an
      # unrecorded one makes every later precision sample unattributable to a build.
      ag_ident=$(jq -r '.predicate.consultations.antigravity as $a
          | ["agy_version","binary_path","model_id","model_family"]
          | map(. as $k | select((($a | has($k)) | not) or ((($a | getpath([$k])) // "" | tostring | length) == 0)))
          | join(", ")' "$rcpt")
      [ -z "$ag_ident" ] \
        || reject B1 "antigravity.status is consulted but these S3.E identity fields are absent or empty: $ag_ident; a review by an unrecorded binary of an unrecorded model cannot be shown to be cross-vendor, which is the arm's whole justification (S5, A5)" || return 1

      # --- THE ARM MUST ACTUALLY BE A SECOND VENDOR. -----------------------
      # agy is a harness that can route to Claude: `agy models` lists
      # claude-sonnet-4-6 and claude-opus-4-6-thinking beside the Gemini ids. With a
      # Claude model - or with `--model` omitted and the default landing there - S3.E
      # is the same family reviewing itself, S5's self-preference case, and the receipt
      # would still read `antigravity` in every field a reader checks. The mechanism is
      # proven ENGAGED here rather than asserted by the arm's name.
      local ag_model
      ag_model=$(jq -r '.predicate.consultations.antigravity.model_id // ""' "$rcpt")
      if match_arm_e_same_family "$ag_model"; then
        reject B1 "antigravity.model_id is '$ag_model', which is the reviewing agent's OWN model family; S3.E exists to be a different vendor and family than the primary reviewer (S5, A5, Huang et al. ICLR'24), and agy is a harness that can route to Claude - so a same-family model makes this arm self-review wearing a cross-vendor label" || return 1
      fi

      # --- COST, wired into S8's cost_per_actionable. -----------------------
      # agy's own `usage` block is real token accounting, so S8's continuous metric has
      # a numerator that was measured rather than estimated. Record-only is not
      # unenforced - the same rule the receipt's own `cost` block already carries.
      ag_usage=$(jq -r '.predicate.consultations.antigravity.usage as $u
          | if ($u | type) != "object" then "usage is not an object"
            else (["input_tokens","output_tokens","total_tokens"]
                  | map(. as $k | select((($u | has($k)) | not) or ((($u | getpath([$k])) | type) != "number")))
                  | join(", ")) end' "$rcpt")
      [ -z "$ag_usage" ] \
        || reject B1 "antigravity.usage must carry numeric input_tokens, output_tokens and total_tokens (missing or non-numeric: $ag_usage); S8's cost_per_actionable is fed from agy's own usage block, and record-only is not unenforced" || return 1

      # --- WHOSE MEASUREMENT IS IT. ----------------------------------------
      # agy runs as a separate process with its own tools, so a finding it marks
      # `measured` was measured by IT, not by this reviewer. S1 says a `measured` claim
      # is "produced by a command this run executed" - and for an agy finding, "this
      # run" is agy's run.
      #
      # THE RULING, so the receipt does not have to be read twice to find it: the
      # PRIMARY REVIEWER DOES NOT RE-RUN THEM, and the receipt says so in this field.
      # Re-running and adjudicating would dissolve the independence the arm exists to
      # create - the disagreement would disappear into the primary's judgement, which
      # is the outcome `divergence` below exists to prevent. A run that DID re-verify
      # records `true`, and then the re-run commands are the primary's own `measured`
      # marks in the SARIF. What is forbidden is leaving it unsaid.
      jq -e '.predicate.consultations.antigravity
             | has("reverified_by_primary") and (.reverified_by_primary | type == "boolean")' \
         "$rcpt" >/dev/null 2>&1 \
        || reject B1 "antigravity.reverified_by_primary must be present and boolean; agy's 'measured' claims were produced by agy's process, and a receipt that does not say whether the primary reviewer re-ran them leaves the reader unable to tell whose measurement it is (S1)" || return 1

      # --- DISAGREEMENT IS SIGNAL. ------------------------------------------
      # Recorded, never resolved silently in the primary's favour. S5's audit_divergence
      # is the same instrument one level up (a second invocation of the SAME reviewer);
      # this is its cross-vendor sibling, and S8 records it as arm_e_agreement_rate.
      #
      # `contradicted` is the row that matters and the row a lazy implementation drops:
      # agy and the primary reached OPPOSITE conclusions on one subject. A receipt that
      # cannot represent that is a receipt in which the primary always wins.
      ag_div=$(jq -r '.predicate.consultations.antigravity.divergence as $d
          | if ($d | type) != "object" then "divergence is not an object"
            else (["agreed","agy_only","primary_only","contradicted"]
                  | map(. as $k | select((($d | has($k)) | not)
                        or ((($d | getpath([$k])) | type) != "number")
                        or ((($d | getpath([$k])) | floor) != ($d | getpath([$k])))
                        or (($d | getpath([$k])) < 0)))
                  | join(", ")) end' "$rcpt")
      [ -z "$ag_div" ] \
        || reject B1 "antigravity.divergence must carry whole non-negative agreed, agy_only, primary_only and contradicted (bad or missing: $ag_div); a disagreement with no place to be written down is a disagreement resolved in the primary's favour, which is the failure S5 names (Huang et al., ICLR'24)" || return 1

      # Every agy finding is accounted for by exactly one of the three columns that
      # describe an agy finding. `primary_only` is deliberately OUTSIDE the identity -
      # it counts the primary's findings agy did not raise, which are not in this array.
      # Without the identity, `divergence` is four numbers nothing constrains, and
      # `{0,0,0,0}` beside twelve agy findings would read as perfect agreement.
      ag_nfind=$(jq -r '[.predicate.consultations.antigravity.findings[]?] | length' "$rcpt")
      ag_sum=$(jq -r '.predicate.consultations.antigravity.divergence
                      | (.agreed + .agy_only + .contradicted)' "$rcpt")
      [ "$ag_nfind" -eq "$ag_sum" ] \
        || reject B1 "antigravity records $ag_nfind finding/s but divergence accounts for $ag_sum of them (agreed + agy_only + contradicted); every agy finding is exactly one of agreed, agy-only or contradicted, and an unbalanced ledger is one in which disagreement can go unrecorded" || return 1
    fi
  fi

  # --- B6: a stale index must not read PASS (S3.A, row 9). ------------------
  if [ "$pmat_st" = "consulted" ]; then
    local idx idx_rec idx_computed
    idx=$(jq -r '.predicate.consultations.pmat.index_commit // ""' "$rcpt")
    # NOT `// "null"`: jq's alternative operator treats `false` as absent, so
    # `false // "null"` yields the string "null" and a correctly-recorded stale index
    # would be rejected as a misreport. Fixture row 9 caught this.
    idx_rec=$(jq -r 'if (.predicate.consultations.pmat | has("index_is_ancestor"))
                     then (.predicate.consultations.pmat.index_is_ancestor | tostring)
                     else "absent" end' "$rcpt")
    [ -n "$idx" ] || reject B1 "pmat.status is consulted but index_commit is absent; an unrecorded index cannot be shown to describe this PR" || return 1
    git -C "$REPO" cat-file -e "${idx}^{commit}" 2>/dev/null \
      || reject B1 "pmat.index_commit $idx does not resolve to a commit in $REPO" || return 1
    if git -C "$REPO" merge-base --is-ancestor "$idx" "$head" 2>/dev/null; then
      idx_computed=true
    else
      idx_computed=false
    fi
    # A receipt that MISREPORTS the ancestry is invalid whatever the verdict: the
    # 66-commit drift scar is an index answering about code that is not in the PR.
    [ "$idx_rec" = "$idx_computed" ] \
      || reject B1 "index_is_ancestor is recorded as $idx_rec but merge-base --is-ancestor $idx $head says $idx_computed" || return 1
    if [ "$idx_computed" = false ] && [ "$verdict" = "PASS" ]; then
      reject B6 "index_commit $idx is not an ancestor of head $head and the verdict is PASS; a stale index answers about code that is not in this PR (S3.A)" || return 1
    fi
  fi

  # --- B1: S3.A duplication coverage is RECORDED, never merely absent. -------
  #
  # PRREV-007 F4 measured two holes in the field S3.A calls "the highest-EV field in the
  # receipt", and both have the same shape:
  #
  #   (a) pmat's semantic index is Rust-only. On #2742 - 46 files, 7,244 insertions -
  #       3,533 of those insertions (48.8%) are sh, py and yaml, and a `pmat query`
  #       cannot return any of them. `duplication_hits: []` therefore meant "the half I
  #       can see is clean" and READ as "nothing like this exists".
  #   (b) prior art on an UNMERGED SIBLING BRANCH is invisible by construction. #2781
  #       found #2742's only because #2742 had merged 17 hours earlier. Luck, not
  #       mechanism.
  #   (c) F7: prior art that landed on origin/main AFTER the merge base is in NEITHER
  #       region, and the receipt did not even NAME that region. #2781's blind region is
  #       exactly #2742 - 1 commit, 46 files, 11 of them the prior art, including
  #       crates/apr-cli/src/commands/test_llm_band.rs. Measured, not estimated: one
  #       `git grep` over it costs 1 s against 20 s for the 774-branch sibling sweep.
  #       So it is swept, `merge_base_to_main` is a required coverage key, and the
  #       horizon must name all three regions whether or not each was reached.
  #
  # S3.0 applied to this field: it must not be possible to read "searched and found
  # nothing" as identical to "could not search". So the coverage the run ACHIEVED is
  # recorded per surface, an unrecorded coverage claim is rejected, and a surface the run
  # could not reach cannot sit under a PASS - exactly the rule rows 5 and 6 already
  # apply to an unreachable consultation.
  #
  # scripts/pr_review_duplication_scan.sh produces this block. The guard does NOT re-run
  # it: a gate that recomputes a 19-second sweep on every receipt is a gate that gets
  # routed around, and the attestation is L1-self for exactly this class of field.
  if [ "$pmat_st" = "consulted" ]; then
    local cov_missing cov_bad cov_none horizon_missing dup_sib dup_scanned dup_total
    # duplication_hits is NOT re-checked here. PRREV-009 wrote a `type == "array"` test at
    # this point and PRREV-008 independently wrote a stronger one 129 lines up, over all
    # four S3.A outputs at once; merging the two lanes left this one UNREACHABLE, because
    # every receipt that could trip it has already been rejected by the earlier branch.
    #
    # The fixture table did not notice - 112 tests, 0 failures, straight over a dead rule.
    # scripts/mutate-guard.sh did: `reject-50-drop` SURVIVED, which is the definition of a
    # rule the guard states and nothing tests. Deleting it is the fix; a permanently
    # unkillable mutant would put a hole in a score S8 fixes at one.
    #
    # It is also, exactly, the defect F4 exists to detect: two implementations of one rule,
    # each green against its own copy, in the guard that implements F4. Recorded here
    # rather than quietly removed, so the next person to add a per-field check looks 129
    # lines up first.

    jq -e '.predicate.consultations.pmat.duplication_coverage | (type == "object") and (length > 0)' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "pmat.status is consulted but duplication_coverage is absent; an unrecorded coverage claim cannot be told apart from a searched-and-empty one, and S3.0 forbids exactly that (F4)" || return 1

    # `merge_base_to_main` joins this list rather than getting a rule of its own, and that
    # is deliberate: the two rules below - a method outside the vocabulary is REJECTED,
    # and `none` may not sit under a PASS - then apply to the third region for free. A new
    # rule would need a new mutant and a new fixture; an entry in this list is covered by
    # the ones already here. Degrade the METHOD, never the count.
    cov_missing=$(jq -r '(["rust","shell","python","config","docs","other","sibling_branches","merge_base_to_main"]
                          - (.predicate.consultations.pmat.duplication_coverage | keys)) | join(", ")' "$rcpt")
    [ -z "$cov_missing" ] \
      || reject B1 "duplication_coverage records no verdict for: $cov_missing; a surface with no entry is the silently-absent coverage F4 exists to stop" || return 1

    # `.value` is bound to $v FIRST. Inside `["..."] | index(f)`, jq evaluates f with the
    # ARRAY as input, so a bare `.value` there reads the array's non-existent .value,
    # errors, and the check silently passes over everything. The dup-cov-bad-method probe
    # caught exactly that: with the unbound form the guard ACCEPTED "shell": "yes".
    cov_bad=$(jq -r '.predicate.consultations.pmat.duplication_coverage | to_entries
                     | map(select(.value as $v | (["semantic","lexical","none"] | index($v | tostring)) == null))
                     | map(.key + "=" + (.value | tostring)) | join(", ")' "$rcpt")
    [ -z "$cov_bad" ] \
      || reject B1 "duplication_coverage holds a method outside { semantic, lexical, none }: $cov_bad" || return 1

    cov_none=$(jq -r '.predicate.consultations.pmat.duplication_coverage | to_entries
                      | map(select(.value == "none")) | map(.key) | join(", ")' "$rcpt")
    if [ -n "$cov_none" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "duplication_coverage could not search [$cov_none] and the verdict is PASS; a surface that was not searched must read DEGRADED, the same rule rows 5 and 6 apply to an unreachable consultation (S3.0)" || return 1
    fi

    jq -e '.predicate.consultations.pmat.duplication_horizon
           | (type == "array") and (length > 0) and (map(type == "string") | all)' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "duplication_horizon is absent or is not a non-empty array of refspecs; an unstated horizon makes 'nothing found' and 'did not look off this branch' the same artifact (F4)" || return 1

    # F7: THE HORIZON NAMES ALL THREE REGIONS, WHETHER OR NOT EACH WAS REACHED.
    #
    # The scan used to build this array from the METHOD, so a region it did not sweep was
    # simply ABSENT - and the pre-F7 receipt read
    # `["HEAD","refs/remotes/origin/* unmerged into origin/main"]` under a PASS while a
    # 46-file region sat outside both. An absent region is unfalsifiable: nothing in the
    # artifact distinguishes "there is no such region" from "I never looked there". S3.0
    # forbids exactly that, and F4's rows 23/24 already forbid it for an unsearched
    # LANGUAGE surface; this is the same rule for an unsearched REF region.
    #
    # Each entry is `<component>=<refspec>`, so a region cannot be dropped by deleting a
    # line, and the coverage map above says which of the three was actually searched.
    horizon_missing=$(jq -r '(["head","siblings","merge_base_to_main"]
                              - [ .predicate.consultations.pmat.duplication_horizon[]?
                                  | (capture("^(?<k>[a-z_]+)=") | .k)? ]) | join(", ")' "$rcpt")
    [ -z "$horizon_missing" ] \
      || reject B1 "duplication_horizon names no region for: $horizon_missing; the horizon has three components (head, siblings, merge_base_to_main) and an unnamed one makes 'nothing found there' and 'never looked there' the same artifact (F7)" || return 1

    # WHOLE numbers, and the `floor` clauses are not decoration. The two rules below
    # compare these with `[ -lt ]`, which cannot read "2.0": it errors, the `if` reads
    # false, and BOTH horizon rules are skipped - so a receipt writing
    # `scanned: 2.0, total: 40` would sail past the capped-horizon check under a PASS.
    # A guard that fails OPEN on a value its own type check admitted is the class this
    # whole file exists to remove, so the type check admits integers only.
    jq -e '.predicate.consultations.pmat
           | (.horizon_branches_total | type == "number") and (.horizon_branches_scanned | type == "number")
             and ((.horizon_branches_total | floor) == .horizon_branches_total)
             and ((.horizon_branches_scanned | floor) == .horizon_branches_scanned)
             and (.horizon_branches_scanned >= 0) and (.horizon_branches_scanned <= .horizon_branches_total)' \
       "$rcpt" >/dev/null 2>&1 \
      || reject B1 "horizon_branches_total and horizon_branches_scanned must both be whole numbers with 0 <= scanned <= total; the horizon's denominator is what makes a partial sweep visible, and a fractional count would make the shell comparisons below error and skip" || return 1

    dup_sib=$(jq -r '.predicate.consultations.pmat.duplication_coverage.sibling_branches // ""' "$rcpt")
    dup_scanned=$(jq -r '.predicate.consultations.pmat.horizon_branches_scanned' "$rcpt")
    dup_total=$(jq -r '.predicate.consultations.pmat.horizon_branches_total' "$rcpt")

    if [ "$dup_sib" != "none" ] && [ "$dup_scanned" -eq 0 ] && [ "$dup_total" -gt 0 ]; then
      reject B1 "duplication_coverage claims the sibling branches were searched ($dup_sib) with horizon_branches_scanned=0 of $dup_total; a sweep that attempted nothing passes vacuously, which S3.D already calls DEGRADED for mutation" || return 1
    fi

    if [ "$dup_scanned" -lt "$dup_total" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "the horizon sweep covered $dup_scanned of $dup_total sibling branches and the verdict is PASS; the branches it skipped were not searched, and unsearched is DEGRADED (F4)" || return 1
    fi

    jq -e '.predicate.consultations.pmat.symbols_searched | type == "number"' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "pmat.symbols_searched must be a number; a scan whose needle count is unrecorded cannot have its precision judged, and record-only is not unenforced" || return 1
  fi

  # --- S1 / S4.2: every claim carries a mark, and a cited mark is verified. --
  local v
  while IFS= read -r v; do
    [ -n "$v" ] || continue
    case "$v" in
      no-grounding\|*)      reject B1 "a finding carries no properties.grounding (${v#*|}); an unmarked claim is a defect in the review (S1)" || return 1 ;;
      bad-grounding\|*)     reject B1 "the grounding of a finding is outside { cited, measured, asserted } (${v#*|})" || return 1 ;;
      no-failure-scenario\|*) reject B1 "a finding has an empty failure_scenario (${v#*|}); a finding that cannot name the failure it permits is a comment (S4.2)" || return 1 ;;
      cited-no-source\|*)   reject B1 "a cited finding has an empty source (${v#*|}); a citation with no source is an assertion wearing the mark of a citation (S1.1)" || return 1 ;;
      cited-no-excerpt\|*)  reject B1 "a cited finding has an empty excerpt (${v#*|}); a recorded excerpt that is not there is the same failure as no excerpt (S1.1)" || return 1 ;;
      cited-no-digest\|*)   reject B1 "a cited finding has no excerpt_sha256 (${v#*|}); cited is verified, not just labelled (S1.1)" || return 1 ;;
      asserted-blocking\|*) reject B1 "a finding marked asserted is classed blocking (${v#*|}); an asserted claim never blocks (S1)" || return 1 ;;
    esac
  done < <(jq -r '
      [ .runs[]? | .results[]? ]
      | map((.ruleId // "<no ruleId>") as $id | .properties as $p
            | if ($p == null) or (($p.grounding // "") | tostring | length) == 0
                then "no-grounding|" + $id
              elif (["cited","measured","asserted"] | index($p.grounding|tostring)) == null
                then "bad-grounding|" + $id + " = " + ($p.grounding|tostring)
              elif (($p.failure_scenario // "") | tostring | length) == 0
                then "no-failure-scenario|" + $id
              elif ($p.grounding == "asserted") and ($p.precision_class == "blocking")
                then "asserted-blocking|" + $id
              elif $p.grounding == "cited" and (($p.source // "") | tostring | length) == 0
                then "cited-no-source|" + $id
              elif $p.grounding == "cited" and (($p.excerpt // "") | tostring | length) == 0
                then "cited-no-excerpt|" + $id
              elif $p.grounding == "cited" and (($p.excerpt_sha256 // "") | tostring | length) == 0
                then "cited-no-digest|" + $id
              else empty end)
      | .[]' "$sarif")

  # --- S1.1: excerpt_sha256 = sha256(excerpt), over the bytes AS STORED. ----
  local line id want got
  while IFS=$'\t' read -r id want got; do
    [ -n "$id" ] || continue
    got=$(printf '%s' "$got" | base64 -d | sha256_stdin)
    [ "$want" = "$got" ] \
      || reject B1 "cited finding '$id' records excerpt_sha256 $want but sha256(excerpt) is $got; a citation whose digest does not match its excerpt is not verified (S1.1)" || return 1
  done < <(jq -r '
      .runs[]? | .results[]? | select(.properties.grounding == "cited")
      | [(.ruleId // "<no ruleId>"), .properties.excerpt_sha256,
          (.properties.excerpt | @base64) ] | @tsv' "$sarif")

  # --- B1: S3.E IS ADVISORY, SO NOTHING IT EMITS MAY BLOCK. -----------------
  #
  # S7's admission rule: a class may block only while its measured precision on the
  # rolling sample is >= 90%, and S8 requires instrument -> 30 samples -> ratchet. S3.E
  # has ZERO samples, so it cannot be admitted to the blocking tier - not because anyone
  # doubts agy, but because the rule that governs the tier has nothing to apply.
  #
  # This is S7.1's argument in the other direction. There it was B5, a class whose sample
  # could never accrue and which therefore could not be DEMOTED. Here the sample WILL
  # accrue - `arm_e_actionable_rate` and `arm_e_agreement_rate` are recorded from the
  # first PR - so the arm can be PROMOTED later, by the same edit to
  # contracts/pr-review-skill-v2.yaml a demotion would use, once 30 samples exist.
  # Promotion is a ticket, not a silent config change, for exactly the reason demotion is.
  #
  # A finding is refused on its `precision_class`, never on its content or its severity:
  # agy may report anything it likes at `advisory`, and level `error` is still allowed.
  # What it may not do is claim an authority the instrumentation has not yet earned it.
  local ag_blocking
  while IFS= read -r ag_blocking; do
    [ -n "$ag_blocking" ] || continue
    reject B1 "the antigravity run states finding '$ag_blocking' with precision_class blocking; S3.E is advisory until 30 samples exist (S7's admission rule, S8's ratchet), so a blocking class from it is a receipt claiming an authority no measurement supports" || return 1
  done < <(jq -r '
      .runs[]? | select((.tool.driver.name // "") == "antigravity")
      | .results[]? | select((.properties.precision_class // "") == "blocking")
      | (.ruleId // "<no ruleId>")' "$sarif")

  # --- B4: a comparative claim carries its comparator (S3.C.1). -------------
  # The 2.93x Ollama rule: the book published that ratio from a harness that never
  # ran Ollama. Every field below is what makes the claim reproducible by a reader.
  local claim missing
  while IFS= read -r claim; do
    [ -n "$claim" ] || continue
    reject B4 "$claim" || return 1
  done < <(jq -r '
      def need($c; $n):
        [ "command","version","env_sha256","artifact_sha256","log_path" ]
        | map(. as $f | $c | getpath([$f]) as $v
              | select(($v == null) or (($v | tostring) == "") or ($v == [])) | $f)
        | if length == 0 then empty
          else "comparative claim " + $n + " is missing comparator field(s): "
               + (join(", ")) + "; absent any field the claim is unverified and blocks (S3.C.1)"
          end;
      [ .predicate.consultations.crux.comparative_claims[]? ]
      | to_entries[]
      | need(.value.comparator // {}; "#" + (.key|tostring)
             + " (" + ((.value.claim // "<unnamed>")|tostring) + ")")
      ' "$rcpt")

  # A comparative claim asserted in a FINDING must also be backed. A claim the
  # reviewer wrote into a result but never recorded in comparative_claims[] is the
  # never-ran-Ollama shape with an extra step.
  local n_recorded
  n_recorded=$(jq -r '[ .predicate.consultations.crux.comparative_claims[]? ] | length' "$rcpt")
  while IFS=$'\t' read -r id got; do
    [ -n "$id" ] || continue
    got=$(printf '%s' "$got" | base64 -d)
    if match_comparative "$got"; then
      [ "$n_recorded" -gt 0 ] || reject B4 "finding $id states a comparative claim -- $(printf '%s' "$got" | cut -c1-70) -- but consultations.crux.comparative_claims is empty; a competitor ratio with no recorded comparator is unverified, per S3.C.1" || return 1
    fi
  done < <(jq -r '.runs[]? | .results[]?
      | [(.ruleId // "<no ruleId>"), ((.message.text // "") | @base64)] | @tsv' "$sarif")

  # --- B4, THE HALF THAT DOES NOT ASK THE REVIEWER. ------------------------
  # The check above reads the SARIF, so it can only see a ratio the reviewer chose to
  # write down. This one reads the DIFF, exactly as the S3.B trigger is recomputed
  # thirty lines up. Measured with a signed discrimination pair before it existed: one
  # head adding `apr sustains 2.93x Ollama on 1.5B Q4_K decode.` to book/ was ACCEPTED
  # with comparative_claims: [] and a silent SARIF, and REJECTED [B4] when the same
  # ratio appeared in a finding. Identical diff; the verdict turned on candour.
  #
  # SCOPED TO THE USER-FACING SURFACE, and that is load-bearing twice over: it is the
  # surface S9's scar is about (the book published the ratio), and it is what keeps
  # this rule from red-ing the PR that adds its own case table -- fixtures state
  # targets, and a spec quoting a banned literal in order to ban it is not publishing
  # a claim. TARGET_RE drops the remaining bars ("Target: 2x Ollama") for the same
  # reason check_no_claim_literals.sh drops them.
  #
  # THREE GAPS, RECORDED RATHER THAN WIDENED.
  #
  # (a) The PR BODY is not read: this guard has no GitHub client. Commit messages are
  #     not a substitute -- this repository's own commit messages QUOTE the banned
  #     ratios in order to ban them, so scanning them would red the very commits that
  #     fix the defect. S3.C.1 lists four surfaces; two are covered here.
  # (b) docs/ prose is not read, and (c) neither is a plain `//` comment. Both were in
  #     scope in the first draft and both were MEASURED OUT over 300 commits of
  #     origin/main -- see match_shipped_surface, where the five hits that decided it
  #     are listed one by one, three of them quotations with no honest remedy.
  if [ "$n_recorded" -eq 0 ]; then
    local dfile dline
    while IFS=$'\t' read -r dfile dline; do
      [ -n "$dfile" ] || continue
      published_claim "$dfile" "$dline" || continue
      reject B4 "the diff publishes a comparative claim on a user-facing surface -- $dfile: $(printf '%s' "$dline" | cut -c1-70) -- while consultations.crux.comparative_claims is empty; a competitor ratio the review never recorded is unverified and blocks (S3.C.1)" || return 1
    done < <(changed_lines "$base" "$head" '+')
  fi

  return 0
}

# -------------------------------------------------------------------------
# S6.1 POSITIVE CONTROL, FIRST.
#
# Before validating anything real, the guard validates deliberately malformed receipts
# and REQUIRES a non-zero exit from each. If a malformed receipt passes, this guard's
# GREEN is a count of files rather than a verdict, and the run fails. Same idiom as
# dogfood.sh's `pv validate` positive control.
#
# TWO controls at DIFFERENT DEPTHS, and each asserts WHICH class fired.
#
# The depth matters: a schema-only control stays green even if every semantic branch
# below the schema gate is deleted, which is precisely the mutation PRREV-004 will try.
# So the second control is signature-valid and schema-valid, and its only defect is
# semantic (B2) - it can only fire by reaching the actor check.
#
# Asserting the CLASS matters just as much. The first draft of this guard had a
# self-review control whose signature was a stub, so it "fired" at the signature branch
# while reporting itself as the self-review control: a control that passes for a reason
# other than the one it names is mislabeled evidence, not a control. Requiring the
# expected class turns that silent mislabel into a loud failure.
# -------------------------------------------------------------------------
PC_TMP=$(mktemp -d "${TMPDIR:-/tmp}/pr-review-positive-control.XXXXXX")
case "$PC_TMP" in
  */pr-review-positive-control.*) ;;
  *) echo "$PROG: FAIL - refusing to use scratch dir $PC_TMP" >&2; exit 1 ;;
esac
cleanup() {
  # This expands into rm -rf, so it is gated twice: the path must still look like the
  # scratch directory this script created, and must be neither empty nor the root.
  cleanup_dir=${PC_TMP:-}
  case "$cleanup_dir" in
    */pr-review-positive-control.*) ;;
    *) return 0 ;;
  esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then
    return 0
  fi
  rm -rf -- "$cleanup_dir"
}
trap cleanup EXIT

GUARD_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PC_SEED_DIR=${PR_REVIEW_POSITIVE_CONTROL_DIR:-$GUARD_DIR/../tests/fixtures/pr-review/positive-control}

# run_control <name> <expected-class> <expected-reason-substring> <dir>
#
# The REASON is asserted, not just the class. B1 covers a dozen branches, so a control
# that fires on a different one still reports the class it was asked for. Measured: with
# only the class asserted, deleting the in-toto schema gate left the schema control
# firing (on the signature branch) and the mutation SURVIVED. Pinning the reason kills it.
run_control() {
  local name=$1 want=$2 want_reason=$3 dir=$4
  REJECT_CLASS=''; REJECT_REASON=''
  if validate_receipt "$dir"; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL FAILED ($name)
  A deliberately malformed receipt was ACCEPTED (exit 0). This guard's verdicts cannot
  be trusted this run: a green result would be a count of files, not a verdict (S6.1).
  Refusing to validate anything.
EOF
    return 1
  fi
  if [ "$REJECT_CLASS" != "$want" ]; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Expected the receipt to be rejected under $want; it was rejected under $REJECT_CLASS:
    $REJECT_REASON
  The control fired for a reason other than the one it exists to test, so it is no
  longer evidence that the $want branch works. Refusing to validate anything.
EOF
    return 1
  fi
  case "$REJECT_REASON" in
    *"$want_reason"*) ;;
    *)
      cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Rejected under the expected class $want, but on the wrong branch. Expected a reason
  containing:
    $want_reason
  got:
    $REJECT_REASON
  The control no longer proves the branch it names is wired. Refusing to validate.
EOF
      return 1 ;;
  esac
  printf 'positive-control  %-15s fired (%s: %s)\n' "$name" "$REJECT_CLASS" \
    "$(printf '%s' "$REJECT_REASON" | cut -c1-64)"
  return 0
}

# Control 1: schema depth. Synthesized inline - it must not depend on any committed
# artifact, so that a deleted fixture tree cannot silently take the control with it.
PC1="$PC_TMP/schema-invalid"; mkdir -p "$PC1"
printf '{"_type":"https://in-toto.io/Statement/v0.9","subject":[],"predicateType":"nope"}\n' \
  > "$PC1/receipt.intoto.jsonl"
printf '{"version":"2.1.0","runs":[]}\n' > "$PC1/findings.sarif"
: > "$PC1/receipt.intoto.jsonl.minisig"
run_control schema-invalid B1 "fails schemas/in-toto-statement-v1.json" "$PC1" || exit 1

# Control 2: semantic depth. A committed receipt that is schema-valid AND correctly
# signed, whose only defect is that the reviewer is the author. It carries its own
# public key, so it verifies whatever PR_REVIEW_PUBKEY is set to for the real run.
# seeded_control <name> <seed-subdir> <class> <reason-substring>
seeded_control() {
  local name=$1 seed="$PC_SEED_DIR/$2" class=$3 reason=$4 d="$PC_TMP/$1"
  if [ ! -f "$seed/receipt.intoto.jsonl" ]; then
    echo "$PROG: FAIL - positive-control fixture missing at $seed" >&2
    echo "  Without it, deleting the check it pins would leave this guard green (S6.1)." >&2
    echo "  Refusing to validate anything." >&2
    return 1
  fi
  mkdir -p "$d"
  cp -- "$seed/receipt.intoto.jsonl" "$seed/findings.sarif" \
        "$seed/receipt.intoto.jsonl.minisig" "$d/" || {
    echo "$PROG: FAIL - positive-control fixture at $seed is incomplete" >&2; return 1; }
  (
    PUBKEY="$PC_SEED_DIR/positive-control.pub"
    run_control "$name" "$class" "$reason" "$d"
  )
}

seeded_control self-review     self-review     B2 "reviewer_actor.id ="        || exit 1
seeded_control findings-digest findings-digest B1 "findings_ref.sha256"        || exit 1
seeded_control cost-missing    cost-missing    B1 "predicate.cost must carry"  || exit 1

# -------------------------------------------------------------------------
# The real receipts.
# -------------------------------------------------------------------------
if [ "$#" -eq 0 ]; then
  echo "$PROG: FAIL - no receipt directory given." >&2
  echo "  usage: $PROG <receipt-dir> [<receipt-dir> ...]" >&2
  echo "  A run over zero receipts is not a pass (S6.3: a missing receipt is RED)." >&2
  exit 1
fi

RC=0
for d in "$@"; do
  REJECT_CLASS=''; REJECT_REASON=''
  if validate_receipt "$d"; then
    printf 'ACCEPT  %s\n' "$d"
  else
    printf 'REJECT  %s  [%s]\n    %s\n' "$d" "$REJECT_CLASS" "$REJECT_REASON"
    RC=1
  fi
done
exit "$RC"
