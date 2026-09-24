#!/usr/bin/env bash
# scripts/check_coverage_report_scoped.sh -- every `llvm-cov report` must be SCOPED (#4023).
#
# This repo's root Cargo.toml is also a package (the `apr` facade). An unscoped `llvm-cov report`
# covers only that facade and writes an EMPTY lcov: it blanked coverage once before (the
# Makefile's single-phase note) and again in coverage-nightly run 35892421393. The older
# cargo-llvm-cov rejected `report --workspace` and 0.9.0 rejects `report --exclude`, so the
# accepted scope is the derived list from scripts/coverage_report_scope.py, or an explicit
# `-p`/`--package`. This refuses any `llvm-cov report` invocation line that has none.
#
# Usage:  check_coverage_report_scoped.sh [file ...]   default: Makefile scripts/*.sh .github/workflows/*.yml
#         check_coverage_report_scoped.sh --self-test   the case table below
# Executed, never sourced.
set -euo pipefail

# A line invoking `llvm-cov report` (not a comment) is scoped if it carries one of these.
SCOPED_RE='coverage_report_scope\.py|(^|[[:space:]])(-p|--package)[[:space:]=]'
INVOKE_RE='llvm-cov[[:space:]]+report([[:space:]]|$)'
COMMENT_RE='^[[:space:]]*@?#'

scan() {
  local bad=0 scanned=0 f n line
  for f in "$@"; do
    [ -f "$f" ] || continue
    scanned=$((scanned + 1))
    n=0
    while IFS= read -r line || [ -n "$line" ]; do
      n=$((n + 1))
      [[ "$line" =~ $COMMENT_RE ]] && continue
      [[ "$line" =~ $INVOKE_RE ]] || continue
      if ! [[ "$line" =~ $SCOPED_RE ]]; then
        echo "UNSCOPED llvm-cov report: $f:$n: ${line#"${line%%[![:space:]]*}"}"
        bad=$((bad + 1))
      fi
    done < "$f"
  done
  # Anti-vacuity: a scan that read nothing proves nothing (run from the wrong
  # directory, or every path missing). It must fail, not pass silently.
  if [ "$scanned" -eq 0 ]; then
    echo "check_coverage_report_scoped: scanned 0 files (none of: $*) -- refusing to pass vacuously"
    return 2
  fi
  if [ "$bad" -gt 0 ]; then
    echo "check_coverage_report_scoped: $bad unscoped \`llvm-cov report\` invocation(s). Scope each with"
    echo "  \$(python3 scripts/coverage_report_scope.py [--exclude NAME]) or -p/--package: an unscoped"
    echo "  report covers only the root facade and writes an empty lcov (#4023)."
    return 1
  fi
  return 0
}

self_test() {
  local dir rc fails=0
  dir=$(mktemp -d)
  trap 'rm -rf "${dir:?}"' RETURN
  # must_flag: the two shapes that blanked coverage
  printf '\t@cargo-llvm-cov llvm-cov report --lcov --output-path x.info\n' > "$dir/unscoped.mk"
  printf '\t@cargo-llvm-cov llvm-cov report --workspace --lcov --output-path x.info\n' > "$dir/workspace.mk"
  # must_pass: derived scope, explicit -p, --package=, a comment, and a non-report llvm-cov call
  printf '\t@cargo-llvm-cov llvm-cov report $$(python3 scripts/coverage_report_scope.py) --lcov\n' > "$dir/derived.mk"
  printf '\t@cargo-llvm-cov llvm-cov report -p a -p b --summary-only\n' > "$dir/explicit.mk"
  printf '\t@cargo-llvm-cov llvm-cov report --package=a --lcov\n' > "$dir/package.mk"
  printf '\t@# an unscoped `llvm-cov report` is the defect\n' > "$dir/comment.mk"
  printf '\t@cargo-llvm-cov llvm-cov test --no-report --workspace --lib\n' > "$dir/test.mk"
  for c in unscoped workspace; do
    rc=0; scan "$dir/$c.mk" > /dev/null || rc=$?
    [ "$rc" -eq 1 ] || { echo "SELF-TEST FAIL: $c must be flagged (rc=$rc)"; fails=$((fails + 1)); }
  done
  rc=0; scan "$dir/does-not-exist.mk" > /dev/null || rc=$?
  [ "$rc" -eq 2 ] || { echo "SELF-TEST FAIL: scanning 0 files must refuse (rc=$rc)"; fails=$((fails + 1)); }
  for c in derived explicit package comment test; do
    rc=0; scan "$dir/$c.mk" > /dev/null || rc=$?
    [ "$rc" -eq 0 ] || { echo "SELF-TEST FAIL: $c must pass (rc=$rc)"; fails=$((fails + 1)); }
  done
  if [ "$fails" -eq 0 ]; then echo "check_coverage_report_scoped self-test: 8/8 cases OK"; return 0; fi
  return 1
}

case "${1:-}" in
  --help|-h) echo "usage: $0 [file ...] | --self-test   (no args: scan Makefile scripts/*.sh .github/workflows/*.yml)" ;;
  --self-test) self_test ;;
  "")
    shopt -s nullglob
    files=()
    for f in Makefile scripts/*.sh .github/workflows/*.yml; do
      # this guard's own case table holds unscoped lines by design
      [ "$f" = "scripts/check_coverage_report_scoped.sh" ] || files+=("$f")
    done
    scan "${files[@]}"
    ;;
  *) scan "$@" ;;
esac
