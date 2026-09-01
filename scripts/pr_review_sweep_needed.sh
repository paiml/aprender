#!/usr/bin/env bash
# pr_review_sweep_needed.sh - does this diff require the guard's mutation sweep?
#
# S3.D's own trigger table says the guard mutation set is required when the diff
# "touches scripts/check_*.sh, dogfood.sh, ci.yml gate logic, or a contracts/*.yaml
# falsifier", and that a docs / non-code diff is "not triggered". `ci.yml` ran
# `scripts/mutate-guard.sh` with NO `if:` at all -- the full 219-mutant sweep on every
# pull request, docs-only included. CI contradicted the specification it enforces.
#
# THAT IS NOT A TIDINESS POINT; IT IS THE QUEUE. The sweep runs the whole bats suite
# once per mutant. On an idle 48-core box Arm 3 measured 3091s (51.5 min) over 185
# mutants; with several pull requests sharing the runner host it exceeds the job's
# 150-minute cap and the job is CANCELLED -- which reads as a failure nobody can
# distinguish from a real one. Measured on 2026-09-01: three receipt jobs started
# within nine minutes of each other, intel's load average went 48 -> 133, and #2836's
# was cancelled at exactly 150 minutes having proved nothing.
#
# WHAT MAKES THE SCORE CHANGE, AND WHY THE LIST IS DERIVED RATHER THAN CHOSEN.
# The sweep's result is a function of exactly what it reads. `build_tree` in
# scripts/mutate-guard.sh copies six paths into every mutant tree, and the harness
# itself defines the mutants, so the answer cannot change unless one of those seven
# moves. The list below is that list. A fixture edit CAN turn a killed mutant into a
# survivor without the guard changing at all, which is why this is not simply
# "did the guard change".
#
# FAIL-CLOSED, IN EVERY DIRECTION. A bad base, an unreadable diff, a missing git, an
# argument that is not a commit -- every one of them prints `yes`. "Could not decide"
# is never "not needed": the whole defect class this repository keeps finding is a gate
# that skips its own subject and reports green.
#
#   usage:  pr_review_sweep_needed.sh <base> <head>      prints `needed=yes|no`
#           pr_review_sweep_needed.sh --self-test        runs the case table
#
# Always exits 0 in the normal path; the ANSWER is on stdout, never in the status, so a
# caller cannot mistake a crash for a `no`.
set -uo pipefail

# The seven inputs, derived from build_tree() in scripts/mutate-guard.sh plus the
# harness. `scripts/check_mutation_inputs_match.sh`-style drift is guarded by the
# self-test row `harness-list-drifts`, which fails when build_tree copies a path that
# is not named here.
SWEEP_INPUTS=(
  'schemas/'
  '.claude/skills/pr-review/SKILL.md'
  'tests/fixtures/pr-review/'
  'tests/pr-review.bats'
  'scripts/check_pr_review_receipt.sh'
  'scripts/pr_review_duplication_scan.sh'
  'scripts/mutate-guard.sh'
)

answer() { printf 'needed=%s\n' "$1"; [ -n "${2-}" ] && printf '  reason: %s\n' "$2" >&2; return 0; }

decide() {  # decide <base> <head>
  local base=$1 head=$2 files rc

  command -v git >/dev/null 2>&1 || { answer yes "no git on PATH"; return 0; }
  git rev-parse --verify "$base^{commit}" >/dev/null 2>&1 \
    || { answer yes "base '$base' is not a commit"; return 0; }
  git rev-parse --verify "$head^{commit}" >/dev/null 2>&1 \
    || { answer yes "head '$head' is not a commit"; return 0; }

  # Status read from the command, never from a pipeline tail.
  files=$(git diff --name-only "$base" "$head" 2>/dev/null); rc=$?
  [ "$rc" -eq 0 ] || { answer yes "git diff exited $rc"; return 0; }

  local f i
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    for i in "${SWEEP_INPUTS[@]}"; do
      case "$i" in
        */) case "$f" in "$i"*) answer yes "$f is under $i"; return 0 ;; esac ;;
        *)  [ "$f" = "$i" ] && { answer yes "$f is a sweep input"; return 0; } ;;
      esac
    done
  done <<< "$files"

  answer no "no sweep input changed between $base and $head"
  return 0
}

# --- the case table ---------------------------------------------------------
# Each row builds a real commit in a throwaway repository and asserts the answer.
# A rule with no row is a rule nothing tests.
self_test() {
  local tmp rc fails=0 rows=0
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/sweep-needed-selftest.XXXXXX") || return 2
  # SEC011: the path is PROVED to be the one mktemp just made before anything
  # recursive touches it. An empty or unexpected $tmp aborts instead of deleting.
  case "$tmp" in
    */sweep-needed-selftest.??????) ;;
    *) echo "self-test: refusing to use scratch dir '$tmp'" >&2; return 2 ;;
  esac
  [ -d "$tmp" ] || { echo "self-test: '$tmp' is not a directory" >&2; return 2; }
  # shellcheck disable=SC2064  # $tmp is validated above and must expand NOW, not at RETURN
  trap "rm -rf -- '$tmp'" RETURN

  ( cd "$tmp" && git init -q . && git config user.email t@t && git config user.name t \
    && mkdir -p schemas .claude/skills/pr-review tests/fixtures/pr-review scripts docs \
    && echo x > schemas/s.json \
    && echo x > .claude/skills/pr-review/SKILL.md \
    && echo x > tests/fixtures/pr-review/f.json \
    && echo x > tests/pr-review.bats \
    && echo x > scripts/check_pr_review_receipt.sh \
    && echo x > scripts/pr_review_duplication_scan.sh \
    && echo x > scripts/mutate-guard.sh \
    && echo x > docs/readme.md \
    && echo x > src.rs \
    && git add -A && git commit -qm base ) || { echo "self-test: fixture repo failed" >&2; return 2; }

  local base; base=$(git -C "$tmp" rev-parse HEAD)

  row() {  # row <label> <want> <path-to-touch>
    rows=$((rows + 1))
    ( cd "$tmp" && echo "changed-$rows" >> "$3" && git add -A && git commit -qm "$1" ) || return 1
    local head got
    head=$(git -C "$tmp" rev-parse HEAD)
    got=$( cd "$tmp" && decide "$base" "$head" 2>/dev/null )
    got=${got#needed=}
    if [ "$got" = "$2" ]; then
      printf 'ok    %-28s want=%-3s got=%s\n' "$1" "$2" "$got"
    else
      printf 'FAIL  %-28s want=%-3s got=%s\n' "$1" "$2" "$got"; fails=$((fails + 1))
    fi
    ( cd "$tmp" && git reset -q --hard "$base" )
  }

  # every one of the seven inputs must trigger
  row guard-changed        yes scripts/check_pr_review_receipt.sh
  row bats-changed         yes tests/pr-review.bats
  row fixture-changed      yes tests/fixtures/pr-review/f.json
  row schema-changed       yes schemas/s.json
  row skill-changed        yes .claude/skills/pr-review/SKILL.md
  row scanner-changed      yes scripts/pr_review_duplication_scan.sh
  row harness-changed      yes scripts/mutate-guard.sh
  # and the discrimination cases: without these, "always yes" reads green
  row docs-only            no  docs/readme.md
  row unrelated-source     no  src.rs

  # fail-closed rows: a question it cannot answer is `yes`
  local got
  got=$( cd "$tmp" && decide "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef" "$base" 2>/dev/null )
  rows=$((rows + 1))
  if [ "${got#needed=}" = yes ]; then printf 'ok    %-28s want=yes got=yes\n' bad-base
  else printf 'FAIL  %-28s want=yes got=%s\n' bad-base "${got#needed=}"; fails=$((fails+1)); fi

  # the list must not drift from build_tree's
  local here missing=""
  here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
  if [ -f "$here/mutate-guard.sh" ]; then
    rows=$((rows + 1))
    local p
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      case " ${SWEEP_INPUTS[*]} " in *" $p "*) ;; *) missing="$missing $p" ;; esac
    done < <(sed -n '/^build_tree()/,/^}/p' "$here/mutate-guard.sh" \
             | grep -oE '\$(ROOT|SNAP)/[A-Za-z0-9_./-]+' | sed -E 's|^\$(ROOT\|SNAP)/||' \
             | sed -E 's|^(schemas)$|\1/|; s|^(tests/fixtures/pr-review)$|\1/|' | sort -u)
    if [ -z "$missing" ]; then printf 'ok    %-28s build_tree adds no path this list lacks\n' harness-list-drifts
    else printf 'FAIL  %-28s build_tree copies:%s\n' harness-list-drifts "$missing"; fails=$((fails+1)); fi
  fi

  printf -- '--- %d rows, %d failure(s) ---\n' "$rows" "$fails"
  [ "$fails" -eq 0 ]
}

case "${1-}" in
  --self-test) self_test; exit $? ;;
  '')          answer yes "called with no arguments"; exit 0 ;;
  *)           decide "$1" "${2-HEAD}"; exit 0 ;;
esac
