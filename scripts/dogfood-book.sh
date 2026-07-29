#!/usr/bin/env bash
# Book end-to-end dogfood per BOOK-CLOSEOUT-001.
#
# Runs every structural + behavioral gate and reports a single GO/WARN/FAIL
# verdict. Used as the final close-out check before declaring the book
# "complete and provable".
#
# Per CLAUDE.md:
# - Use bashrs (not shellcheck) for shell quality
# - Use pv (not bash) for contract validation
# - Use pmat comply for cross-contract gating
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

step() {
  echo ""
  echo "-- $1 --"
}

pass() {
  echo "PASS  $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
  echo "WARN  $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
  echo "FAIL  $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

# ---------------------------------------------------------------------------
# Phase 1: Linkcheck
# ---------------------------------------------------------------------------
step "Phase 1: mdbook-linkcheck"
if bash scripts/check_book_linkcheck.sh > /tmp/dogfood-linkcheck.log 2>&1; then
  pass "FALSIFY-BOOK-LINKCHECK-001: 0 broken file links"
else
  N=$(grep -c "File not found:" /tmp/dogfood-linkcheck.log || echo 0)
  fail "FALSIFY-BOOK-LINKCHECK-001: $N broken targets"
fi

# ---------------------------------------------------------------------------
# Phase 2 + 4: CLI parity + example block
# ---------------------------------------------------------------------------
step "Phase 2+4: CLI coverage"
if bash scripts/check_book_cli_parity.sh > /tmp/dogfood-cli-parity.log 2>&1; then
  pass "FALSIFY-BOOK-CLI-PARITY-001: all CLI subcommands have a chapter"
else
  N=$(grep -c FAIL /tmp/dogfood-cli-parity.log || echo 0)
  fail "FALSIFY-BOOK-CLI-PARITY-001: $N commands without chapters"
fi

if bash scripts/check_book_example_block.sh > /tmp/dogfood-cli-example.log 2>&1; then
  pass "FALSIFY-BOOK-EXAMPLE-001: all CLI chapters have a fenced bash example"
else
  N=$(grep -c FAIL /tmp/dogfood-cli-example.log || echo 0)
  fail "FALSIFY-BOOK-EXAMPLE-001: $N CLI chapters missing bash example"
fi

# ---------------------------------------------------------------------------
# Phase 3: Library parity + example block
# ---------------------------------------------------------------------------
step "Phase 3: Library coverage"
if bash scripts/check_book_lib_parity.sh > /tmp/dogfood-lib-parity.log 2>&1; then
  pass "FALSIFY-BOOK-LIB-PARITY-001: all aprender-core pub modules have chapters"
else
  N=$(grep -c FAIL /tmp/dogfood-lib-parity.log || echo 0)
  fail "FALSIFY-BOOK-LIB-PARITY-001: $N modules without chapters"
fi

if bash scripts/check_book_lib_example_block.sh > /tmp/dogfood-lib-example.log 2>&1; then
  pass "FALSIFY-BOOK-LIB-EXAMPLE-001: all lib chapters have a fenced rust example"
else
  N=$(grep -c FAIL /tmp/dogfood-lib-example.log || echo 0)
  fail "FALSIFY-BOOK-LIB-EXAMPLE-001: $N lib chapters missing rust example"
fi

# ---------------------------------------------------------------------------
# Phase 4: pv validate completeness contract
# ---------------------------------------------------------------------------
step "Phase 4: contract validity"
if pv validate contracts/apr-book-completeness-v1.yaml > /tmp/dogfood-pv.log 2>&1; then
  if grep -qE '(^|[^0-9])0 error\(s\), 0 warning\(s\)' /tmp/dogfood-pv.log; then
    pass "pv validate apr-book-completeness-v1.yaml: clean"
  else
    warn "pv validate has warnings (non-blocking)"
  fi
else
  fail "pv validate apr-book-completeness-v1.yaml: validation errors"
fi

if pv validate contracts/apr-page-cli-run-v1.yaml > /dev/null 2>&1; then
  pass "pv validate apr-page-cli-run-v1.yaml: clean (sample)"
else
  warn "pv validate apr-page-cli-run-v1.yaml had issues (sample)"
fi

# ---------------------------------------------------------------------------
# Phase 5: README contract claims
# ---------------------------------------------------------------------------
step "Phase 5: README claim consistency"
if bash scripts/check_readme_claims.sh > /tmp/dogfood-readme.log 2>&1; then
  pass "FALSIFY-README-001..006: README claims match repo state"
else
  N=$(grep -c FAIL /tmp/dogfood-readme.log || echo 0)
  fail "README claim drift: $N mismatches"
fi

# ---------------------------------------------------------------------------
# Phase 6: Execution validation (advisory)
# ---------------------------------------------------------------------------
step "Phase 6: example execution validation (advisory)"
if [ -x scripts/check_book_examples_executable.sh ]; then
  if bash scripts/check_book_examples_executable.sh > /tmp/dogfood-exec.log 2>&1; then
    pass "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: bash examples execute cleanly"
  else
    warn "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: some examples failed"
  fi
else
  warn "Phase 6 execution harness not yet shipped (shape-only today)"
fi

if [ -x scripts/check_book_examples_compile.sh ]; then
  if bash scripts/check_book_examples_compile.sh > /tmp/dogfood-compile.log 2>&1; then
    pass "FALSIFY-BOOK-EXAMPLE-COMPILES-001: rust examples compile"
  else
    warn "FALSIFY-BOOK-EXAMPLE-COMPILES-001: some rust examples do not compile"
  fi
else
  warn "Phase 6 rust compile harness not yet shipped (shape-only today)"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
step "Final: mdbook build"
if (cd book && mdbook build > /tmp/dogfood-build.log 2>&1); then
  pass "mdbook build: clean"
else
  fail "mdbook build: failed (see /tmp/dogfood-build.log)"
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo ""
echo "==============================================="
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "VERDICT: FAIL  ($FAIL_COUNT failed, $WARN_COUNT warned, $PASS_COUNT passed)"
  exit 1
fi
if [ "$WARN_COUNT" -gt 0 ]; then
  echo "VERDICT: WARN  ($WARN_COUNT warned, $PASS_COUNT passed; Phase 6 deferred is expected)"
  exit 0
fi
echo "VERDICT: GO  ($PASS_COUNT gates passed; book is complete and provable)"
exit 0
