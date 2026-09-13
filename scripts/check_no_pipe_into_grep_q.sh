#!/usr/bin/env bash
# check_no_pipe_into_grep_q.sh — `producer | grep -q PATTERN` under `pipefail`
# reports a verdict the producer's death chose, not one grep reached.
#
# THE DEFECT. `grep -q` exits the instant it matches. The producer is still
# writing, gets SIGPIPE, and exits 141. With `pipefail` the PIPELINE is 141, so:
#
#   * a positive row  `... && cmd | grep -q WANTED`   goes RED though it MATCHED
#   * a negative row  `... && ! cmd | grep -q BANNED` goes GREEN though it MATCHED
#
# Measured 2026-09-13: `scripts/tests/ratchet_semantics_test.sh` false-REDded
# `guard-cargo` twice in one afternoon — row C2 on main (run 34759351727) and
# row S3 in a merge group (run 34765202728). Both printed the wanted string in
# their own failure output, and both ejected work from the merge queue.
# Reproduced at 2/200 with a 100 KB payload; the small-payload instances are
# consistent with the same mechanism, since nothing else in those `&&` chains
# can be false.
#
# THE FIX is not to drop `pipefail`. Feed grep without a pipe:
#
#   grep -q PATTERN <<<"$out"                 # a variable (here-string: no pipe)
#   grep -q PATTERN file                      # a file
#   cmd > "$t" 2>&1; grep -q PATTERN "$t"     # a command
#
# THE RATCHET. Sites predating this guard are carried as a baseline COUNT that
# may only fall; a new one is refused. Bare: judge the tree. --self-test: the
# case table. --update: restamp, downward only.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE="$ROOT/scripts/pipe_grep_q_baseline.txt"
SELF="scripts/check_no_pipe_into_grep_q.sh"

# The hazard: a pipe whose right-hand side is `grep -q`. `-qF`, `-qE`, `-qi`
# and every other cluster containing q exits early too, so the cluster is
# [A-Za-z]*q[A-Za-z]*. ERE throughout.
HAZARD='\|[[:space:]]*grep[[:space:]]+(-[A-Za-z]*q[A-Za-z]*|--quiet|--silent)([[:space:]]|$)'

# Does this file's own text turn pipefail on? Any spelling that names it.
PIPEFAIL='^[[:space:]]*set[[:space:]]+-[a-zA-Z]*o?[[:space:]]*[^#]*pipefail'

enables_pipefail() { grep -qE "$PIPEFAIL" "$1"; }

# Universe: tracked files UNION the working tree — a brand-new untracked script
# must not get a free pass (feedback_tracked_only_universe_gives_a_free_pass).
shell_files() {
  {
    git -C "$ROOT" ls-files '*.sh' 2>/dev/null
    ( cd "$ROOT" && find . -name '*.sh' -type f \
        -not -path './target/*' -not -path './.git/*' \
        -not -path './.claude/*' -not -path './node_modules/*' 2>/dev/null | sed 's|^\./||' )
  } | sort -u
}

# Findings to $1 (may be /dev/null); the COUNT on stdout. The guard's own file
# is excluded: it spells the pattern in prose and in its case table, and a scan
# that includes its own definition can never pass.
scan() {
  local findings=$1 f n total=0 files=0
  : > "$findings"
  while IFS= read -r f; do
    [ "$f" = "$SELF" ] && continue
    [ -r "$ROOT/$f" ] || continue
    files=$((files + 1))
    enables_pipefail "$ROOT/$f" || continue
    n=$(grep -cE "$HAZARD" "$ROOT/$f" 2>/dev/null) || n=0
    [ "${n:-0}" -gt 0 ] || continue
    grep -nE "$HAZARD" "$ROOT/$f" 2>/dev/null | sed "s|^|  $f:|" >> "$findings"
    total=$((total + n))
  done < <(shell_files)
  # VACUITY FLOOR: a scan that walked no files measured nothing, and "0 sites"
  # over an empty universe is the shape this guard exists to refuse elsewhere.
  if [ "$files" -lt 20 ]; then
    echo "VACUOUS: the scan reached $files shell file(s); this tree has far more" >&2
    return 2
  fi
  printf '%s\n' "$total"
}

read_baseline() {
  [ -f "$BASELINE" ] || { echo "FAIL: no baseline at $BASELINE" >&2; return 2; }
  local v
  v=$(grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | head -1 | tr -d '[:space:]')
  case "$v" in ''|*[!0-9]*) echo "FAIL: baseline is not a count: '${v}'" >&2; return 2 ;; esac
  printf '%s\n' "$v"
}

# ---------------------------------------------------------------- case table
# Every row is a string put to the SHIPPED regex, both polarities. The patterns
# were wrong five times in this repo's history and every one was caught by a
# table like this, none by reading the regex (CLAUDE.md, Verification
# Discipline #7).
CHECKS=0; FAILED=0
ok()  { CHECKS=$((CHECKS+1)); printf 'ok    %s\n' "$1"; }
bad() { CHECKS=$((CHECKS+1)); FAILED=$((FAILED+1)); printf 'FAIL  %s\n' "$1"; }

row_match()    { if grep -qE "$HAZARD" <<<"$2"; then ok "$1"; else bad "$1 -- wanted a MATCH on: $2"; fi; }
row_no_match() { if grep -qE "$HAZARD" <<<"$2"; then bad "$1 -- wanted NO match on: $2"; else ok "$1"; fi; }

self_test() {
  echo "=== the hazard regex: what it must and must not call a site ==="
  row_match    'H1  a plain pipe into grep -q'            '  cmd | grep -q WANTED'
  row_match    'H2  the -qF cluster'                      '  printf "%s" "$o" | grep -qF X'
  row_match    'H3  -qE with no space before the pipe'    '  x |grep -qE "^ok"'
  row_match    'H4  the NEGATED form (the false-GREEN half)' '  && ! cmd | grep -q BANNED'
  row_match    'H5  --quiet spelled long'                 '  cmd | grep --quiet X'
  row_match    'H6  the flag at end of line (pattern continues)' '  cmd | grep -q'
  row_no_match 'H7  the here-string fix'                  '  grep -q X <<<"$out"'
  row_no_match 'H8  grep reading a FILE'                  '  grep -q X "$tmp"'
  row_no_match 'H9  grep -c through a pipe counts, it does not exit early' '  cmd | grep -c X'
  row_no_match 'H10 a q in the PATTERN, not in a flag'    '  cmd | grep -v q'
  row_no_match 'H11 grep -n through a pipe reads to EOF'  '  cmd | grep -nE "^x"'

  echo "=== the pipefail predicate: only files that arm it are judged ==="
  local t; t=$(mktemp -d) || return 2
  case "$t" in ""|"/"|"/*") echo "refusing to use scratch dir '$t'" >&2; return 2 ;; esac
  printf 'set -euo pipefail\n' > "$t/a"; if enables_pipefail "$t/a"; then ok 'P1  set -euo pipefail'; else bad 'P1  set -euo pipefail not recognised'; fi
  printf 'set -o pipefail\n'   > "$t/b"; if enables_pipefail "$t/b"; then ok 'P2  set -o pipefail'; else bad 'P2  set -o pipefail not recognised'; fi
  printf 'set -uo pipefail\n'  > "$t/c"; if enables_pipefail "$t/c"; then ok 'P3  set -uo pipefail'; else bad 'P3  set -uo pipefail not recognised'; fi
  printf 'set -eu\n'           > "$t/d"; if enables_pipefail "$t/d"; then bad 'P4  a file WITHOUT pipefail must not be judged'; else ok 'P4  no pipefail: not judged'; fi
  printf '# set -o pipefail is what we do NOT do here\n' > "$t/e"
  if enables_pipefail "$t/e"; then bad 'P5  a COMMENT about pipefail must not arm the file'; else ok 'P5  a comment does not arm the file'; fi
  [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"

  echo "=== the mutation: drop the q requirement and H9/H10/H11 must go RED ==="
  local mutant='\|[[:space:]]*grep[[:space:]]+-' m=0
  grep -qE "$mutant" <<<'  cmd | grep -c X'   && m=$((m+1))
  grep -qE "$mutant" <<<'  cmd | grep -v q'   && m=$((m+1))
  grep -qE "$mutant" <<<'  cmd | grep -nE "^x"' && m=$((m+1))
  if [ "$m" -eq 3 ]; then
    ok 'M1  a regex that forgets the q flags all three non-hazards -- so the q in the shipped regex is load bearing'
  else
    bad "M1  the mutant flagged $m of 3; the q requirement is not what makes H9/H10/H11 pass"
  fi

  printf '\n%s checks, %s failed\n' "$CHECKS" "$FAILED"
  [ "$FAILED" -eq 0 ]
}

# -------------------------------------------------------------------- driver
main() {
  local findings count base
  findings=$(mktemp) || return 2
  trap 'rm -f "$findings"' RETURN
  count=$(scan "$findings") || { rm -f "$findings"; return 2; }
  base=$(read_baseline) || { rm -f "$findings"; return 2; }

  case "${1:-}" in
    --update)
      if [ "$count" -gt "$base" ]; then
        echo "REFUSED: --update may only lower the baseline ($base -> $count is a RISE)"
        cat "$findings"; rm -f "$findings"; return 1
      fi
      printf '# `producer | grep -q` sites under pipefail. This number may only FALL.\n# Re-stamp with: bash %s --update\n%s\n' "$SELF" "$count" > "$BASELINE"
      echo "baseline restamped: $base -> $count"; rm -f "$findings"; return 0 ;;
  esac

  echo "=== pipe-into-grep-q sites under pipefail (measured, not remembered) ==="
  echo "  baseline $base   measured $count"
  if [ "$count" -gt "$base" ]; then
    echo
    echo "FAIL: $count sites, ceiling $base. A pipe into 'grep -q' under pipefail"
    echo "      reports the PRODUCER's SIGPIPE, not grep's verdict. Use a here-string"
    echo "      (grep -q P <<<\"\$var\"), a file, or a temp file. Do not raise the ceiling."
    echo
    cat "$findings"; rm -f "$findings"; return 1
  fi
  if [ "$count" -lt "$base" ]; then
    echo "  PASS -- and $((base - count)) fewer than the ceiling. Lower it:"
    echo "          bash $SELF --update"
  else
    echo "  PASS -- at the ceiling, which is not over it."
  fi
  rm -f "$findings"; return 0
}

usage() {
  cat <<'USAGE'
check_no_pipe_into_grep_q.sh -- refuse `producer | grep -q` under `pipefail`.

  (no argument)  judge the tree against scripts/pipe_grep_q_baseline.txt
  --self-test    run the case table (hazard regex both polarities, the
                 pipefail predicate, and the mutation that proves the q
                 requirement is load bearing)
  --update       restamp the baseline; a RISE is refused
  --help         this text

guard_tree.sh discovers the self-test by finding the string "self-test" in
this output, and then runs BOTH rows. Removing it from here silently drops
the case table out of CI.
USAGE
}

case "${1:-}" in
  --self-test)   self_test; exit $? ;;
  --help|-h)     usage; exit 0 ;;
  *)             main "${1:-}"; exit $? ;;
esac
