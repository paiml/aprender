#!/usr/bin/env bash
#
# check_workflow_path_filters.sh - a path-filtered workflow must not be able to
# go dark, and must gate a PR exactly as strictly as it gates main.
#
# WHY THIS EXISTS
# ---------------
# `.github/workflows/book.yml` filtered on `paths: book/**`. FALSIFY-BOOK-CLI-PARITY-001
# asserts that every `apr` subcommand has a book chapter - but watching only the
# BOOK meant that adding a subcommand never ran the gate that checks subcommands.
# `apr beat-run` shipped in #1995 with no chapter and the gate stayed green for
# three months (mdBook CI last ran on main 2026-05-14). Behind it sat a lib-parity
# gate failing identically, a `pv` step that exited 127 on a binary the runner
# never had, and two chapter examples training to NaN. Each was invisible until
# the one before it was fixed.
#
# A path filter is a claim that nothing outside those paths can break this gate.
# This script checks the two ways that claim silently becomes false.
#
# RULE 1 - push/pull_request symmetry.
#   If `push` watches a path that `pull_request` does not, a PR touching only that
#   path is GREEN (the workflow never runs), and then main goes RED the moment it
#   merges. That is a main-red generator, and it is invisible in review because
#   both lists look reasonable on their own. Found live in book-contracts.yml:
#   push watched `contracts/apr-book-schema-*` and
#   `crates/aprender-core/tests/book_contracts.rs`; pull_request did not.
#
# RULE 2 - a gate that runs code must watch the code it runs.
#   book-contracts.yml executes `crates/aprender-core/examples/ch*`, which train
#   models using aprender-core's optimizers, losses and layers. It watched the
#   EXAMPLES but not the library they exercise, so a regression in SGD could break
#   all 27 chapter examples without ever triggering the workflow that runs them.
#   Declared per workflow in REQUIRED_COVERAGE below.
#
# Both rules are mechanical. Neither is a judgement call, which is the point -
# five wrong guard regexes in this repo were caught by tables, none by review.
#
#   bash scripts/check_workflow_path_filters.sh            # check
#   bash scripts/check_workflow_path_filters.sh --self-test # 4-case table

set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WF_DIR="${REPO_ROOT}/.github/workflows"
FILTER_DUMP="${REPO_ROOT}/scripts/lib/workflow_path_filters.py"

# Workflows that RUN code from a crate must watch that crate's source.
# Format: <workflow basename>|<path prefix that must appear in both filters>
REQUIRED_COVERAGE=("book-contracts.yml|crates/aprender-core/src/**")

# Print `push_paths` and `pr_paths` for one workflow, one per line, prefixed.
# Uses python3 + PyYAML: the `on:` key parses as the boolean True in YAML 1.1,
# which is exactly the sort of detail a hand-rolled grep gets wrong.
dump_filters() {
  python3 "$FILTER_DUMP" "$1"

}

check_workflow() {
  local wf="$1" name out rc
  name="$(basename "$wf")"
  out="$(dump_filters "$wf" 2>/tmp/wfpf_err.$$)"; rc=$?
  if [ "$rc" -eq 3 ]; then
    printf 'FAIL %s: could not be parsed as YAML:\n' "$name"
    sed 's|^|       |' /tmp/wfpf_err.$$; rm -f /tmp/wfpf_err.$$
    return 1
  fi
  rm -f /tmp/wfpf_err.$$
  [ -z "$out" ] && return 0

  local push pr
  push="$(printf '%s\n' "$out" | awk -F'\t' '$1=="PUSH"{print $2}' | sort -u)"
  pr="$(printf '%s\n' "$out" | awk -F'\t' '$1=="PR"{print $2}' | sort -u)"

  # An unfiltered event cannot go dark; only compare when BOTH are filtered.
  case "$push" in *'<unfiltered>'*) return 0 ;; esac
  case "$pr"   in *'<unfiltered>'*) return 0 ;; esac
  [ -z "$push" ] && return 0
  [ -z "$pr" ] && return 0

  local fail=0 only_push only_pr
  only_push="$(comm -23 <(printf '%s\n' "$push") <(printf '%s\n' "$pr"))"
  only_pr="$(comm -13 <(printf '%s\n' "$push") <(printf '%s\n' "$pr"))"

  if [ -n "$only_push" ]; then
    printf '\nFAIL %s: push watches path(s) that pull_request does not.\n' "$name"
    printf '%s\n' "$only_push" | sed 's|^|       + |'
    printf '     A PR touching only these is GREEN because the workflow never runs,\n'
    printf '     and main goes RED when it merges.\n'
    fail=1
  fi
  if [ -n "$only_pr" ]; then
    printf '\nFAIL %s: pull_request watches path(s) that push does not.\n' "$name"
    printf '%s\n' "$only_pr" | sed 's|^|       + |'
    printf '     main is then gated more weakly than the PR that changed it.\n'
    fail=1
  fi

  # Rule 2: declared coverage of the code this workflow executes.
  local line req
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in "${name}|"*) ;; *) continue ;; esac
    req="${line#*|}"
    if ! grep -Fxq "$req" <<< "$push" || ! grep -Fxq "$req" <<< "$pr"; then
      printf '\nFAIL %s: runs code from `%s` but does not watch it in BOTH filters.\n' "$name" "$req"
      printf '     A regression in that source breaks this gate without triggering it.\n'
      fail=1
    fi
  done < <(printf '%s\n' "${REQUIRED_COVERAGE[@]}")

  return "$fail"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"
  if [ -z "${TD:-}" ] || [ ! -d "$TD" ]; then
    printf 'FAIL: could not create a temp dir for the case table.\n' >&2; exit 1
  fi
  trap 'rm -rf "${TD:?}"' EXIT

  # 1 - asymmetric (the live book-contracts.yml bug). MUST fail.
  cat > "$TD/wf1.yml" <<'YML'
on:
  push:
    paths: ["book/**", "crates/aprender-core/tests/book_contracts.rs"]
  pull_request:
    paths: ["book/**"]
YML
  # 2 - symmetric. MUST pass.
  cat > "$TD/wf2.yml" <<'YML'
on:
  push:
    paths: ["book/**"]
  pull_request:
    paths: ["book/**"]
YML
  # 3 - no path filter at all: cannot go dark. MUST pass.
  cat > "$TD/wf3.yml" <<'YML'
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
YML
  # 4 - PR stricter than push. MUST fail (main gated more weakly).
  cat > "$TD/wf4.yml" <<'YML'
on:
  push:
    paths: ["book/**"]
  pull_request:
    paths: ["book/**", "scripts/book-gate.sh"]
YML

  fails=0
  for c in 1 4; do
    if check_workflow "$TD/wf${c}.yml" >/dev/null 2>&1; then
      printf 'FAIL  row %s NOT flagged - the guard is blind to a real defect shape\n' "$c"
      fails=$((fails + 1))
    else
      printf 'ok    row %s flagged (must turn RED)\n' "$c"
    fi
  done
  for c in 2 3; do
    if check_workflow "$TD/wf${c}.yml" >/dev/null 2>&1; then
      printf 'ok    row %s clean (must stay GREEN)\n' "$c"
    else
      printf 'FAIL  row %s flagged - false positive\n' "$c"
      fails=$((fails + 1))
    fi
  done

  if [ "$fails" -ne 0 ]; then
    printf '\nSELF-TEST FAILED (%s/4 wrong)\n' "$fails"; exit 1
  fi
  printf '\nSELF-TEST PASSED (4/4)\n'
  exit 0
fi

# ---------------------------------------------------------------------------
printf '=== path-filtered workflows must not go dark (check_workflow_path_filters.sh) ===\n'

if [ ! -d "$WF_DIR" ]; then
  printf 'FAIL: %s does not exist.\n' "$WF_DIR"; exit 1
fi

scanned=0
violations=0
filtered=0
for wf in "$WF_DIR"/*.yml "$WF_DIR"/*.yaml; do
  [ -e "$wf" ] || continue
  scanned=$((scanned + 1))
  if grep -q '^[[:space:]]*paths:' "$wf" 2>/dev/null; then
    filtered=$((filtered + 1))
  fi
  check_workflow "$wf" || violations=$((violations + 1))
done

# Vacuity: a scan that looked at nothing must not report clean.
if [ "$scanned" -lt 5 ]; then
  printf '\nFAIL (vacuity): scanned only %s workflow file(s). Fix the glob, not this number.\n' "$scanned"
  exit 1
fi

printf '\nscanned %s workflow(s), %s of them path-filtered\n' "$scanned" "$filtered"

if [ "$violations" -ne 0 ]; then
  printf '\n%s workflow(s) can go dark. A path filter is a claim that nothing outside\n' "$violations"
  printf 'those paths can invalidate the gate. Make that true, or drop the filter.\n'
  exit 1
fi

printf 'PASS: every path-filtered workflow gates PRs and main identically.\n'
exit 0
