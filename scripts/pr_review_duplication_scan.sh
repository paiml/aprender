#!/usr/bin/env bash
# pr_review_duplication_scan.sh - the non-Rust and off-branch half of S3.A duplication_hits.
#
# WHY THIS EXISTS
#
# S3.A calls duplication_hits "the highest-EV field in the receipt" and prescribes
# `pmat query` as its mechanism. PRREV-007's backtest measured what that mechanism can
# see, and the answer is: less than half of the diff it was designed for.
#
#   (a) pmat's semantic index is Rust-only. 10,247 tracked .rs files; 10,317 files
#       indexed. A semantic query aimed squarely at scripts/perf_gate.sh's job returned
#       10 results, all .rs, and not it. #2742 - the prior art PERF-055 nearly rewrote -
#       is 46 files / 7,244 insertions, of which 3,533 (48.8%) are sh, py and yaml.
#       Those are exactly the guards, gates and harnesses this epic keeps re-writing.
#
#   (b) Prior art on an UNMERGED SIBLING BRANCH is invisible by construction. B6 requires
#       index_commit to be an ancestor of HEAD, which is correct for staleness and also
#       guarantees the index holds only this branch's history. #2781 found #2742's prior
#       art only because #2742 had merged 17 hours earlier. Reverse the order and S3.A
#       returns []. That is luck, not mechanism.
#
#   (c) F7: prior art that landed on origin/main AFTER this branch's merge base is in
#       NEITHER region. Not on HEAD - the branch predates it. Not an unmerged sibling -
#       it merged. #2781's blind region is exactly #2742: 1 commit, 46 files, 11 of them
#       the prior art. Measured: one git grep over that region costs 1 s against 20 s for
#       the 774-branch sweep, and it returns test_llm_band.rs.
#
# So this script does the crude thing pmat cannot: a LEXICAL sweep for the diff's new
# symbol and file names, over every text file regardless of language, over the unmerged
# sibling branches as well as HEAD, and over merge-base..origin/main.
#
# THE HORIZON HAS THREE COMPONENTS AND ALL THREE ARE ALWAYS NAMED - `head`, `siblings`,
# `merge_base_to_main`. Whether each was SEARCHED is a separate field
# (duplication_coverage), because a region that is absent from the horizon is a region
# whose silence cannot be read.
#
# WHAT IT IS AND IS NOT - state this honestly or it becomes the next piece of theater
#
#   IS      an exact, word-boundary, case-sensitive name match. It finds a second
#           definition that kept the same name, and a file added twice under the same
#           basename. That is the shape of every duplication this epic actually shipped:
#           check_pr_review_wiring.sh, run_bands, perf_gate.
#   IS NOT  semantic. A re-implementation under a DIFFERENT name is INVISIBLE to it, and
#           no amount of grep fixes that. Its recall is unknown and this script does not
#           claim one; what it claims is that 0% coverage of 48.8% of the diff is worse
#           than lexical coverage of all of it.
#   PRECISION is bounded by the needle filter (NEEDLE_MIN_LEN, the stop list) and is
#           REPORTED, not asserted: symbols_searched, hits_total and hits_recorded go in
#           the receipt so a reader can judge the ratio instead of trusting an adjective.
#
# The receipt records the method per surface - semantic | lexical | none - because S3.0
# applied to this field means it must not be possible to read "searched and found
# nothing" as "could not search". scripts/check_pr_review_receipt.sh rejects a receipt
# whose duplication_coverage is absent, incomplete, or records a surface it could not
# reach while the verdict still reads PASS.
#
# USAGE
#   pr_review_duplication_scan.sh --base <sha> --head <sha> [options]
#
#     --repo DIR            repository to scan          (default: git toplevel here)
#     --horizon MODE        all | since | none          (default: all)
#     --horizon-since DAYS  with --horizon since        (default: 30)
#     --max-branches N      cap the sweep, 0 = no cap   (default: 0)
#     --rust-semantic       record rust coverage as `semantic`; pass this ONLY when the
#                           caller actually ran the S3.A `pmat query` pass. Without it
#                           rust is recorded `lexical`, which is the truth of this
#                           script on its own.
#     --json OUT            write the JSON block here   (default: stdout)
#
#   pr_review_duplication_scan.sh --extract-symbol '<one added diff line>'
#     Pure predicate: print the definition names that line introduces, one per line.
#     Exit 0 if any, 1 if none. It exists so the extraction patterns are driven by a
#     must-match / must-not-match case table
#     (tests/fixtures/pr-review/duplication-symbol-cases.tsv) rather than by reading
#     them. This repository's guard patterns have been wrong six times; a table caught
#     every one and review caught none.
#
# EXIT: 0 the scan completed (with or without hits); 1 it could not run. A scan that
# could not run must not print a coverage map claiming it did.

set -euo pipefail

PROG=${0##*/}

# ---------------------------------------------------------------------------
# The extraction patterns.
#
# One sed -E per definition form, each run as its own pass over the added lines. NOT one
# sed with several `s///p` commands: after the first substitution the pattern space IS
# the symbol, and the next expression would match against that instead of the source
# line. Separate passes cost nothing and cannot interfere.
# ---------------------------------------------------------------------------
RE_RUST_FN='^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(default[[:space:]]+)?(const[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?(extern[[:space:]]+"[^"]*"[[:space:]]+)?fn[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*$'
RE_RUST_TY='^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(struct|enum|trait|union)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*$'
# The `{` is REQUIRED, not optional, and it need NOT end the line. Both halves of that
# are measured, not stylistic:
#   - required, because with it optional a Rust tail expression `foo()` alone on a line
#     reads as a shell function definition, and that was the largest source of junk
#     needles on this repository's own diffs.
#   - not end-anchored, because `name() { :; }` on one line is a shell function too, and
#     end-anchoring it returned symbol: 0 on the counterfactual in evidence/prrev-009.
# Both polarities are pinned in tests/fixtures/pr-review/duplication-symbol-cases.tsv.
RE_SH_FN='^[[:space:]]*(function[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\(\)[[:space:]]*\{.*$'
RE_SH_KW='^[[:space:]]*function[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*(\(\))?[[:space:]]*\{?[[:space:]]*$'
RE_PY_DEF='^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\(.*$'
RE_PY_CLS='^[[:space:]]*class[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*[:(].*$'

# A needle shorter than this is a name collision waiting to happen, not a duplication
# signal. Measured on this repository: at 4 the sweep drowns in `new`, `run`, `path`; at
# 6 the hits are names a human recognises as the same thing. The number is a precision
# control, and it is RECORDED in the output so a later run can ratchet it against
# measured precision rather than re-guess it.
NEEDLE_MIN_LEN=${PR_REVIEW_NEEDLE_MIN_LEN:-6}

# Names so common in this tree that a match carries no information. Kept SHORT on
# purpose: every entry is recall deliberately given up, so each must earn its place.
NEEDLE_STOPLIST='cleanup
default
deserialize
serialize
teardown
to_string
usage
main_loop
parse_args
setup_file
from_str'

# extract_symbols <source-line>  -> the definition names on stdout, one per line.
extract_symbols() {
  local line=$1
  {
    printf '%s\n' "$line" | sed -nE "s/$RE_RUST_FN/\8/p"
    printf '%s\n' "$line" | sed -nE "s/$RE_RUST_TY/\4/p"
    printf '%s\n' "$line" | sed -nE "s/$RE_SH_FN/\2/p"
    printf '%s\n' "$line" | sed -nE "s/$RE_SH_KW/\1/p"
    printf '%s\n' "$line" | sed -nE "s/$RE_PY_DEF/\2/p"
    printf '%s\n' "$line" | sed -nE "s/$RE_PY_CLS/\1/p"
  } | awk 'NF && !seen[$0]++'
}

if [ "${1-}" = --extract-symbol ]; then
  SUBJ=${2?--extract-symbol needs an argument}
  # A diff line is accepted with or without its leading '+', so the case table can be
  # written in either form.
  OUT=$(extract_symbols "${SUBJ#+}")
  [ -n "$OUT" ] || exit 1
  printf '%s\n' "$OUT"
  exit 0
fi
if [ "${1-}" = -h ] || [ "${1-}" = --help ]; then
  sed -n '2,70p' "$0"
  exit 0
fi

# ---------------------------------------------------------------------------
# Arguments.
# ---------------------------------------------------------------------------
BASE=''
HEAD_REF=''
REPO=''
HORIZON=all
HORIZON_SINCE=30
MAX_BRANCHES=0
RUST_METHOD=lexical
JSON_OUT=''

while [ "$#" -gt 0 ]; do
  case $1 in
    --base)           BASE=${2:?--base needs a sha};             shift 2 ;;
    --head)           HEAD_REF=${2:?--head needs a sha};         shift 2 ;;
    --repo)           REPO=${2:?--repo needs a directory};       shift 2 ;;
    --horizon)        HORIZON=${2:?--horizon needs a mode};      shift 2 ;;
    --horizon-since)  HORIZON_SINCE=${2:?--horizon-since days};  shift 2 ;;
    --max-branches)   MAX_BRANCHES=${2:?--max-branches needs N}; shift 2 ;;
    --rust-semantic)  RUST_METHOD=semantic;                      shift ;;
    --json)           JSON_OUT=${2:?--json needs a path};        shift 2 ;;
    *) echo "$PROG: unknown argument '$1'" >&2; exit 1 ;;
  esac
done

for t in git jq sed awk; do
  command -v "$t" >/dev/null 2>&1 || {
    echo "$PROG: FAIL - $t is not on PATH." >&2
    echo "  A scan that could not run must not print a coverage map saying it did." >&2
    exit 1; }
done

case "$HORIZON" in all|since|none) ;; *)
  echo "$PROG: FAIL - --horizon must be all, since or none (got '$HORIZON')" >&2; exit 1 ;;
esac
[ -n "$BASE" ]     || { echo "$PROG: FAIL - --base is required" >&2; exit 1; }
[ -n "$HEAD_REF" ] || { echo "$PROG: FAIL - --head is required" >&2; exit 1; }

if [ -z "$REPO" ]; then
  REPO=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "$PROG: FAIL - not in a git repository and --repo was not given" >&2; exit 1; }
fi
g() { git -C "$REPO" "$@"; }
g cat-file -e "${BASE}^{commit}" 2>/dev/null \
  || { echo "$PROG: FAIL - base $BASE does not resolve in $REPO" >&2; exit 1; }
g cat-file -e "${HEAD_REF}^{commit}" 2>/dev/null \
  || { echo "$PROG: FAIL - head $HEAD_REF does not resolve in $REPO" >&2; exit 1; }

# EPOCHSECONDS (bash builtin) is an integer-seconds drop-in for `date +%s`
# that spawns no subprocess; only its DIFFERENCE (WALL, below) is recorded,
# so this is genuinely a wall-clock duration measurement, not a build-artifact
# timestamp in need of a reproducible source.
START=$EPOCHSECONDS

TMP=$(mktemp -d "${TMPDIR:-/tmp}/prrev-dupscan.XXXXXX")
case "$TMP" in
  */prrev-dupscan.*) ;;
  *) echo "$PROG: FAIL - refusing to use scratch dir $TMP" >&2; exit 1 ;;
esac
cleanup_scratch() {
  # Expands into rm -rf, so it is gated twice: the path must still look like the scratch
  # directory this run created, and must be neither empty nor the root.
  cleanup_dir=${TMP:-}
  case "$cleanup_dir" in */prrev-dupscan.*) ;; *) return 0 ;; esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then return 0; fi
  rm -rf -- "$cleanup_dir"
  return 0
}
trap cleanup_scratch EXIT

# ---------------------------------------------------------------------------
# 1. The needles: what this diff introduces.
# ---------------------------------------------------------------------------
g diff --name-only "$BASE" "$HEAD_REF"                    > "$TMP/changed.txt"
g diff --name-only --diff-filter=A "$BASE" "$HEAD_REF"    > "$TMP/added-files.txt"
g diff --unified=0 "$BASE" "$HEAD_REF"                    > "$TMP/diff.txt"

# added source lines; the '+++' file headers are not source
awk '/^\+/ && !/^\+\+\+/ { print substr($0, 2) }' "$TMP/diff.txt" > "$TMP/added-lines.txt"

: > "$TMP/needles-symbol.txt"
sed -nE "s/$RE_RUST_FN/\8/p" "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"
sed -nE "s/$RE_RUST_TY/\4/p" "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"
sed -nE "s/$RE_SH_FN/\2/p"   "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"
sed -nE "s/$RE_SH_KW/\1/p"   "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"
sed -nE "s/$RE_PY_DEF/\2/p"  "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"
sed -nE "s/$RE_PY_CLS/\1/p"  "$TMP/added-lines.txt" >> "$TMP/needles-symbol.txt"

# basenames of newly ADDED files, with and without their extension
awk -F/ '{ print $NF }' "$TMP/added-files.txt" > "$TMP/needles-file-raw.txt"
{
  cat "$TMP/needles-file-raw.txt"
  sed -E 's/\.[A-Za-z0-9_]+$//' "$TMP/needles-file-raw.txt"
} > "$TMP/needles-file.txt"

printf '%s\n' "$NEEDLE_STOPLIST" > "$TMP/stop.txt"
filter_needles() {  # <in> <out>
  awk -v minlen="$NEEDLE_MIN_LEN" '
    NR == FNR { stop[$0] = 1; next }
    { n = $0
      sub(/^[[:space:]]+/, "", n); sub(/[[:space:]]+$/, "", n)
      if (n == "" || length(n) < minlen) next
      if (n in stop) next
      if (!seen[n]++) print n }' "$TMP/stop.txt" "$1" > "$2"
}
filter_needles "$TMP/needles-symbol.txt" "$TMP/n-symbol.txt"
filter_needles "$TMP/needles-file.txt"   "$TMP/n-file.txt"
awk '!seen[$0]++' "$TMP/n-symbol.txt" "$TMP/n-file.txt" > "$TMP/needles.txt"

N_SYMBOL=$(awk 'END { print NR + 0 }' "$TMP/n-symbol.txt")
N_FILE=$(awk   'END { print NR + 0 }' "$TMP/n-file.txt")
N_NEEDLE=$(awk 'END { print NR + 0 }' "$TMP/needles.txt")

# ---------------------------------------------------------------------------
# 2. The HEAD sweep: every text file, every language, minus this diff's own files.
#
# This is the half pmat cannot do. `git grep -I -n -w -F` is exact-name, word-boundary
# and language-blind, which is the whole point: shell, python, yaml and markdown are as
# searchable as Rust.
#
# git grep, NOT `git grep | head`. A producer piped into a truncating filter returns the
# FILTER's status and takes SIGPIPE, which has read as "no matches" in this repository
# four times. The status below is read from git grep itself.
# ---------------------------------------------------------------------------
: > "$TMP/head-raw.txt"
HEAD_METHOD=lexical
if [ "$N_NEEDLE" -gt 0 ]; then
  NEEDLE_ARGS=()
  while IFS= read -r n; do
    [ -n "$n" ] || continue
    NEEDLE_ARGS+=(-e "$n")
  done < "$TMP/needles.txt"
  set +e
  g grep -I -n -w -F --no-color "${NEEDLE_ARGS[@]}" "$HEAD_REF" \
     > "$TMP/head-raw.txt" 2> "$TMP/head.err"
  grep_rc=$?
  set -e
  # 0 = matches, 1 = no matches, >1 = the search itself failed. A failed search is
  # `none`: it did not look, and must not be recorded as having looked and found nothing.
  if [ "$grep_rc" -gt 1 ]; then
    HEAD_METHOD=none
    : > "$TMP/head-raw.txt"
  fi
fi

# git grep over a REV prefixes every line with "<rev>:". Strip exactly that one field.
sed 's/^[^:]*://' "$TMP/head-raw.txt" > "$TMP/head-hits.txt"

# Attribute each surviving line to the needle that matched it, excluding the diff's own
# files - or every needle would match the line that defined it. Word matching is done
# by hand rather than with a GNU-only \y, so this runs the same under mawk.
#
# ONE definition, called twice: once for the HEAD sweep and once for the
# merge-base..origin/main sweep below. Two copies of an attribution rule drift, and the
# drift is invisible because each keeps passing against its own copy - which is F4's own
# finding, and F8 was that defect inside the guard that implements F4.
#
# attribute_hits <grep-output> <out.tsv> <region-file|"">
#   region-file empty  -> every path is in scope (the HEAD sweep)
#   region-file given  -> only paths listed there are in scope (a bounded region)
attribute_hits() {
  awk -v nf="$TMP/needles.txt" -v cf="$TMP/changed.txt" -v rf="$3" '
    function isword(c) { return (c ~ /[A-Za-z0-9_]/) }
    function wordfind(hay, nee,   p, off, before, after) {
      off = 0
      while (1) {
        p = index(substr(hay, off + 1), nee)
        if (p == 0) return 0
        p += off
        before = (p == 1) ? "" : substr(hay, p - 1, 1)
        after  = substr(hay, p + length(nee), 1)
        if (!isword(before) && !isword(after)) return 1
        off = p
      }
    }
    BEGIN {
      while ((getline l < nf) > 0) { if (l != "") needles[++nn] = l }
      while ((getline l < cf) > 0) { if (l != "") changed[l] = 1 }
      if (rf != "") { use_region = 1
        while ((getline l < rf) > 0) { if (l != "") region[l] = 1 } }
    }
    {
      i = index($0, ":");            if (i == 0) next
      path = substr($0, 1, i - 1);   rest = substr($0, i + 1)
      j = index(rest, ":");          if (j == 0) next
      line = substr(rest, 1, j - 1); content = substr(rest, j + 1)
      if (path in changed) next
      if (use_region && !(path in region)) next
      for (k = 1; k <= nn; k++) {
        if (wordfind(content, needles[k])) { print needles[k] "\t" path "\t" line; break }
      }
    }' "$1" > "$2"
}
attribute_hits "$TMP/head-hits.txt" "$TMP/hits-head.tsv" ""

# ---------------------------------------------------------------------------
# 3. The sibling-branch sweep: prior art that has not merged yet.
#
# `git for-each-ref --no-merged origin/main` is the horizon. Measured on this repository
# (evidence/prrev-009/coverage-measurements.txt): 786 remote heads, 771 of them unmerged,
# and a tree-only added-file sweep over all 771 costs 17.3 s. That is cheap enough that
# --horizon all is the default and a cap is a deliberate, RECORDED reduction rather than
# an unavoidable one.
#
# Tree-only on purpose: --name-only --diff-filter=A reads trees, never blobs, which is
# what makes the whole-horizon sweep affordable. The consequence is stated rather than
# hidden - off-branch matching is by FILENAME, not by symbol. `check_pr_review_wiring.sh`
# is exactly that shape, and so is every other case F4 names.
# ---------------------------------------------------------------------------
SIB_METHOD=none

# THE LOCAL CLONE IS NOT THE REMOTE, and conflating them is how this sweep would come to
# report complete coverage of nothing. `git for-each-ref refs/remotes/origin` enumerates
# what THIS clone has FETCHED; on a shallow or CI checkout that is main and nothing else,
# so a horizon of zero branches "swept in full" would satisfy every rule in the guard.
#
# So the remote is asked directly and BOTH numbers are recorded. ls-remote is used as a
# CROSS-CHECK, not as the horizon: it returns names, and diffing a branch needs its
# objects, which only a fetch provides. That is why the sweep runs over remote-tracking
# refs and not over ls-remote output - a deliberate departure from the obvious reading,
# stated here rather than left to be re-derived.
#
# Measured: 1.89 s over 786 heads. Bounded by `timeout` because an unreachable remote
# must degrade the record, never hang the review.
g for-each-ref --format='%(refname:short)' refs/remotes/origin > "$TMP/local-refs.txt" 2>/dev/null \
  || : > "$TMP/local-refs.txt"
LOCAL_ORIGIN_REFS=$(awk '$0 != "origin/HEAD"' "$TMP/local-refs.txt" | awk 'END { print NR + 0 }')
REMOTE_HEADS=-1                      # -1 => could not ask; serialised as null
if [ "$HORIZON" != none ]; then
  if timeout 30 git -C "$REPO" ls-remote --heads origin > "$TMP/ls-remote.txt" 2>/dev/null; then
    REMOTE_HEADS=$(awk 'END { print NR + 0 }' "$TMP/ls-remote.txt")
  fi
fi

: > "$TMP/branches-all.txt"
if g rev-parse --verify --quiet refs/remotes/origin/main >/dev/null 2>&1; then
  # THE DENOMINATOR IS THE CANDIDATE SET, NOT EVERY REF. Three exclusions, each with a
  # reason, because horizon_branches_total is what horizon_branches_scanned is judged
  # against and an inflated denominator would make a complete sweep read as partial:
  #   --no-merged origin/main   already-merged work is on HEAD and the HEAD sweep has it
  #   --no-merged <head>        an ANCESTOR of this branch is already in the HEAD sweep
  #   --no-contains <head>      a DESCENDANT of this branch is our own work coming back
  # Measured at 0.11 s over 786 refs, so this is a filter, not a budget.
  g for-each-ref --no-merged refs/remotes/origin/main \
    --no-merged "$HEAD_REF" --no-contains "$HEAD_REF" \
    --format='%(committerdate:unix) %(refname:short)' refs/remotes/origin \
    > "$TMP/branches-all.txt" 2>/dev/null || : > "$TMP/branches-all.txt"
fi
BR_TOTAL=$(awk 'END { print NR + 0 }' "$TMP/branches-all.txt")

: > "$TMP/branches.txt"
if [ "$HORIZON" != none ]; then
  CUT=0
  if [ "$HORIZON" = since ]; then
    # $EPOCHSECONDS (bash builtin) avoids a `date -d` subprocess; guarded so
    # a non-numeric --horizon-since falls back to CUT=0 exactly as the old
    # `date ... || echo 0` did, instead of an arithmetic error under set -e.
    # The `10#` prefix forces BASE-10 interpretation of $HORIZON_SINCE: bare
    # arithmetic expansion treats a leading-zero numeral as octal (bash
    # builtin, unlike `date -d` which is a plain string parser), so "08"/"09"
    # would fatally error ("value too great for base") and "010" would
    # silently compute from octal 8 instead of decimal 10. `10#$HORIZON_SINCE`
    # reads the same digits `date -d "$HORIZON_SINCE days ago"` always did.
    case "$HORIZON_SINCE" in
      ''|*[!0-9]*) CUT=0 ;;
      *) CUT=$((EPOCHSECONDS - 10#$HORIZON_SINCE * 86400)) ;;
    esac
  fi
  awk -v cut="$CUT" '$1 >= cut { print $2 }' "$TMP/branches-all.txt" > "$TMP/branches.txt"
  if [ "$MAX_BRANCHES" -gt 0 ]; then
    awk -v m="$MAX_BRANCHES" 'NR <= m' "$TMP/branches.txt" > "$TMP/branches.cap"
    mv -- "$TMP/branches.cap" "$TMP/branches.txt"
  fi
  # The horizon was enumerated. Whether it held anything is a separate fact, recorded as
  # horizon_branches_total / _scanned rather than folded into the method.
  SIB_METHOD=lexical
fi
BR_SCANNED=$(awk 'END { print NR + 0 }' "$TMP/branches.txt")

# A clone holding no branch but main cannot look off-branch AT ALL. Recording `lexical`
# over an empty horizon would be a sweep that attempted nothing reading exactly like one
# that searched everything and found nothing - the vacuity S3.D already names for
# mutation. Degrade the METHOD, not the count: the guard's existing rule (`none` may not
# read PASS) then does the rest, and no new rule is added for the mutation set to cover.
if [ "$SIB_METHOD" = lexical ] && [ "$LOCAL_ORIGIN_REFS" -le 1 ]; then
  SIB_METHOD=none
fi

: > "$TMP/branch-added.tsv"
if [ "$SIB_METHOD" = lexical ] && [ "$N_NEEDLE" -gt 0 ] && [ "$BR_SCANNED" -gt 0 ]; then
  while IFS= read -r br; do
    [ -n "$br" ] || continue
    g diff --name-only --diff-filter=A "refs/remotes/origin/main...$br" 2>/dev/null \
      > "$TMP/one-branch.txt" || : > "$TMP/one-branch.txt"
    # printf in the shell rather than a sed per branch: 766 branches is 766 processes,
    # and the sweep is already the dominant cost in this script.
    while IFS= read -r bp; do
      [ -n "$bp" ] || continue
      printf '%s\t%s\n' "$br" "$bp" >> "$TMP/branch-added.tsv"
    done < "$TMP/one-branch.txt"
  done < "$TMP/branches.txt"
fi

# NO "skip paths this diff also adds" rule here, and the omission is deliberate. An
# earlier draft had one, to keep a branch that already carried our work from matching
# itself - and it silently discarded the single strongest signal there is: TWO BRANCHES
# ADDING THE SAME PATH. It made the counterfactual in evidence/prrev-009 return zero
# hits, which is how it was caught. Our own lineage is excluded by ref, at the
# for-each-ref above, which is the right place for it.
awk -F'\t' -v nf="$TMP/needles.txt" '
  BEGIN {
    while ((getline l < nf) > 0) { if (l != "") needle[l] = 1 }
  }
  {
    br = $1; p = $2
    n = p; sub(/^.*\//, "", n)
    stem = n; sub(/\.[A-Za-z0-9_]+$/, "", stem)
    if (n in needle)         { print n    "\t" br "\t" p; next }
    if (stem in needle)      { print stem "\t" br "\t" p }
  }' "$TMP/branch-added.tsv" > "$TMP/hits-branch.tsv"

# ---------------------------------------------------------------------------
# 3b. THE THIRD REGION: merge-base..origin/main. (F7)
#
# The horizon had TWO regions - HEAD, and the unmerged siblings - and prior art that
# landed on origin/main AFTER this branch's merge base is in NEITHER. It is not on HEAD
# (the branch predates it), it is not an unmerged sibling (it merged), and B6 forbids an
# index newer than HEAD from supplying it. The receipt did not even NAME the region, so
# `duplication_hits: []` and "did not look there" were the same artifact - which is
# precisely the defect F4 was raised to fix, one region over.
#
# This is the most ordinary shape there is: your branch is a day behind and someone
# merged the thing you are about to write. Measured on #2781, whose blind region is
# EXACTLY #2742 - 1 commit, 46 files, 11 of them the prior art the review missed:
#
#   git grep over origin/main with #2781's 22 needles   rc=0, 1 s, 1636 raw lines
#   the 774-branch sibling sweep on the same PR         20 s
#   of #2742's 46 files, 5 are hit, including crates/apr-cli/src/commands/test_llm_band.rs
#
# One second, against 20 for the region that found less. So it is SWEPT, not merely
# recorded - and it is still recorded, because a region that could not be searched must
# read `none` and `none` may not sit under a PASS.
#
# SCOPED TO THE REGION, not to all of origin/main. The grep runs over the whole tree
# (git grep takes a rev, not a diff) and the attribution then keeps only paths in
# `git diff --name-only BASE origin/main`. Without that filter every hit already visible
# on HEAD would be counted a second time under a different `where`, and the ratio the
# scan reports about itself would stop being judgeable.
# ---------------------------------------------------------------------------
MB_METHOD=none
MB_FILES=0
: > "$TMP/hits-main.tsv"
: > "$TMP/main-changed.txt"
MB_REFSPEC="$BASE..refs/remotes/origin/main"
if [ "$HORIZON" != none ] && g rev-parse --verify --quiet refs/remotes/origin/main >/dev/null 2>&1; then
  if g diff --name-only "$BASE" refs/remotes/origin/main > "$TMP/main-changed.txt" 2>/dev/null; then
    MB_FILES=$(awk 'END { print NR + 0 }' "$TMP/main-changed.txt")
    MB_METHOD=lexical
    # An EMPTY region is searched-in-full, not unsearched: base == origin/main means main
    # has not moved since the fork, and there is nothing there to find. Skipping the grep
    # is an optimisation over zero files, not a coverage claim - MB_FILES records which.
    if [ "$MB_FILES" -gt 0 ] && [ "$N_NEEDLE" -gt 0 ]; then
      set +e
      g grep -I -n -w -F --no-color "${NEEDLE_ARGS[@]}" refs/remotes/origin/main \
         > "$TMP/main-raw.txt" 2> "$TMP/main.err"
      mb_rc=$?
      set -e
      # Same three-way read as the HEAD sweep: 0 matched, 1 no matches, >1 the search
      # itself failed - and a failed search is `none`, never "looked and found nothing".
      if [ "$mb_rc" -gt 1 ]; then
        MB_METHOD=none
        : > "$TMP/main-raw.txt"
      fi
      sed 's/^[^:]*://' "$TMP/main-raw.txt" > "$TMP/main-hits.txt"
      attribute_hits "$TMP/main-hits.txt" "$TMP/hits-main.tsv" "$TMP/main-changed.txt"
    fi
  else
    : > "$TMP/main-changed.txt"
  fi
fi

# ---------------------------------------------------------------------------
# 4. AMBIENT NEEDLES ARE DROPPED, AND THE DROP IS RECORDED.
#
# Duplication is a RARE name collision. A needle that matches half the tree is a common
# word in this repository, not a second implementation: measured on the PRREV-001..005
# range, `receipt`, `findings` and `README` alone produced 6,292 of the run's hits and
# every one of them was noise. So a needle over the threshold is dropped whole.
#
# This is a precision control and it is therefore REPORTED, never silent:
# needles_dropped_ambient is in the output. A dropped needle is recall given up, and
# recall given up without a number beside it is how a scan comes to look clean.
# ---------------------------------------------------------------------------
AMBIENT_MAX=${PR_REVIEW_AMBIENT_MAX:-8}
cut -f1 "$TMP/hits-head.tsv" "$TMP/hits-branch.tsv" "$TMP/hits-main.tsv" | awk 'NF' | sort | uniq -c \
  | awk -v m="$AMBIENT_MAX" '$1 > m { $1 = ""; sub(/^ +/, ""); print }' > "$TMP/ambient.txt"
N_AMBIENT=$(awk 'END { print NR + 0 }' "$TMP/ambient.txt")
drop_ambient() {  # <in.tsv> <out.tsv>
  awk -F'\t' -v af="$TMP/ambient.txt" '
    BEGIN { while ((getline l < af) > 0) { if (l != "") ambient[l] = 1 } }
    !($1 in ambient)' "$1" > "$2"
}
drop_ambient "$TMP/hits-head.tsv"   "$TMP/hits-head-f.tsv"
drop_ambient "$TMP/hits-branch.tsv" "$TMP/hits-branch-f.tsv"
drop_ambient "$TMP/hits-main.tsv"   "$TMP/hits-main-f.tsv"

# ---------------------------------------------------------------------------
# 5. The coverage map, the horizon, and the cost. All three RECORDED, none inferred.
# ---------------------------------------------------------------------------
{
  awk -F'\t' '{ printf "{\"needle\":\"%s\",\"kind\":\"name\",\"where\":\"HEAD\",\"ref\":\"HEAD\",\"path\":\"%s\",\"line\":%d,\"method\":\"lexical\"}\n", $1, $2, $3 }' "$TMP/hits-head-f.tsv"
  awk -F'\t' '{ printf "{\"needle\":\"%s\",\"kind\":\"filename\",\"where\":\"branch\",\"ref\":\"%s\",\"path\":\"%s\",\"line\":0,\"method\":\"lexical\"}\n", $1, $2, $3 }' "$TMP/hits-branch-f.tsv"
  awk -F'\t' '{ printf "{\"needle\":\"%s\",\"kind\":\"name\",\"where\":\"main\",\"ref\":\"origin/main\",\"path\":\"%s\",\"line\":%d,\"method\":\"lexical\"}\n", $1, $2, $3 }' "$TMP/hits-main-f.tsv"
} | awk '!seen[$0]++' > "$TMP/hits.jsonl"

HITS_TOTAL=$(awk 'END { print NR + 0 }' "$TMP/hits.jsonl")
HITS_CAP=${PR_REVIEW_HITS_CAP:-100}
awk -v m="$HITS_CAP" 'NR <= m' "$TMP/hits.jsonl" > "$TMP/hits-cap.jsonl"
HITS_RECORDED=$(awk 'END { print NR + 0 }' "$TMP/hits-cap.jsonl")

# THE HORIZON NAMES ALL THREE REGIONS, ALWAYS. (F7)
#
# It used to be built from the METHOD - the sibling entry appeared only when the sweep
# ran - so a region that was not searched was simply ABSENT from the horizon, and absent
# is the one thing S3.0 forbids: it made "found nothing there" and "never looked there"
# the same artifact. The horizon now states WHICH REGIONS EXIST; duplication_coverage
# states whether each was SEARCHED. Two questions, two fields, neither inferable from
# the other.
#
# Each entry is `<component>=<refspec>` and the guard requires all three components by
# name, so a future region cannot be dropped by deleting a line.
HORIZON_LIST=$(jq -n --arg head "head=$HEAD_REF" \
  --arg sib "siblings=refs/remotes/origin/* unmerged into origin/main" \
  --arg mb "merge_base_to_main=$MB_REFSPEC" \
  '[$head, $sib, $mb]')

END=$EPOCHSECONDS
WALL=$((END - START))

jq -n \
  --argjson hits "$(jq -cs . "$TMP/hits-cap.jsonl")" \
  --arg rust "$RUST_METHOD" \
  --arg head_method "$HEAD_METHOD" \
  --arg sib "$SIB_METHOD" \
  --arg mb "$MB_METHOD" \
  --argjson mb_files "$MB_FILES" \
  --argjson horizon "$HORIZON_LIST" \
  --argjson br_total "$BR_TOTAL" \
  --argjson br_scanned "$BR_SCANNED" \
  --argjson local_refs "$LOCAL_ORIGIN_REFS" \
  --argjson remote_heads "$REMOTE_HEADS" \
  --argjson n_symbol "$N_SYMBOL" \
  --argjson n_file "$N_FILE" \
  --argjson n_needle "$N_NEEDLE" \
  --argjson hits_total "$HITS_TOTAL" \
  --argjson hits_recorded "$HITS_RECORDED" \
  --argjson minlen "$NEEDLE_MIN_LEN" \
  --argjson ambient "$N_AMBIENT" \
  --argjson ambient_max "$AMBIENT_MAX" \
  --argjson wall "$WALL" \
  '{
     duplication_hits: $hits,
     duplication_coverage: {
       rust:   (if $head_method == "none" then "none" else $rust end),
       shell:  $head_method,
       python: $head_method,
       config: $head_method,
       docs:   $head_method,
       other:  $head_method,
       sibling_branches: $sib,
       merge_base_to_main: $mb
     },
     duplication_horizon: $horizon,
     horizon_branches_total: $br_total,
     horizon_branches_scanned: $br_scanned,
     horizon_refs: { local_origin_refs: $local_refs,
                     remote_heads: (if $remote_heads < 0 then null else $remote_heads end) },
     merge_base_to_main_files: $mb_files,
     symbols_searched: $n_needle,
     needles: { symbol: $n_symbol, filename: $n_file, min_length: $minlen,
                dropped_ambient: $ambient, ambient_max_hits: $ambient_max },
     hits_total: $hits_total,
     hits_recorded: $hits_recorded,
     wall_seconds: $wall
   }' > "$TMP/out.json"

if [ -n "$JSON_OUT" ]; then
  cp -- "$TMP/out.json" "$JSON_OUT"
  printf '%s: wrote %s (%s needles, %s hits, %s/%s sibling branches, %ss)\n' \
    "$PROG" "$JSON_OUT" "$N_NEEDLE" "$HITS_TOTAL" "$BR_SCANNED" "$BR_TOTAL" "$WALL" >&2
else
  cat -- "$TMP/out.json"
fi
exit 0
