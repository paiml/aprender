#!/bin/bash
set -euo pipefail

# Local equivalent of .github/workflows/book-contracts.yml
# Run this before pushing to catch all provable contract failures.
# Usage: scripts/book-ci-local.sh

FAIL=0
TOTAL_GATES=0
PASSED_GATES=0

gate() {
    local name="$1"
    TOTAL_GATES=$((TOTAL_GATES+1))
    echo -n "  $name: "
}

pass() {
    PASSED_GATES=$((PASSED_GATES+1))
    echo "PASS"
}

fail() {
    FAIL=1
    echo "FAIL — $1"
}

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Book Contract CI (local)                             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# === JOB 1: Chapter examples compile ===
echo "Job 1: Chapter Examples Compile"
for ch in ch01_hello_apr ch02_tensors ch03_apr_format ch04_supervised \
  ch05_unsupervised ch06_ensembles ch07_model_selection ch08_transformer \
  ch09_inference ch10_training ch11_formats ch12_serving ch13_profiling \
  ch14_contracts ch15_orchestrate ch16_timeseries ch17_bayesian \
  ch18_graphs ch19_text ch20_rag ch21_vs_candle ch22_vs_llamacpp \
  ch23_training_bench ch24_switch_pytorch ch25_switch_ollama \
  ch26_switch_ndarray ch27_switch_unsloth; do
  gate "$ch compile"
  cargo build -p aprender-core --example "$ch" 2>/dev/null && pass || fail "compilation error"
done
echo ""

# === JOB 2: Chapter examples run ===
echo "Job 2: Chapter Examples Run"
for ch in ch01_hello_apr ch02_tensors ch03_apr_format ch04_supervised \
  ch05_unsupervised ch06_ensembles ch07_model_selection ch08_transformer \
  ch09_inference ch10_training ch11_formats ch12_serving ch13_profiling \
  ch14_contracts ch15_orchestrate ch16_timeseries ch17_bayesian \
  ch18_graphs ch19_text ch20_rag ch21_vs_candle ch22_vs_llamacpp \
  ch23_training_bench ch24_switch_pytorch ch25_switch_ollama \
  ch26_switch_ndarray ch27_switch_unsloth; do
  gate "$ch run"
  cargo run -p aprender-core --example "$ch" >/dev/null 2>&1 && pass || fail "exit non-zero"
done
echo ""

# === JOB 3: Integration tests ===
echo "Job 3: Integration Tests"
gate "book_contracts"
cargo test -p aprender-core --test book_contracts 2>/dev/null && pass || fail "test failure"
echo ""

# === JOB 4: Contract enforcement ===
echo "Job 4: Contract Enforcement"
gate "27 chapter contracts exist"
ALL_CONTRACTS=true
for n in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27; do
  [ -f "contracts/apr-book-ch${n}-v1.yaml" ] || { ALL_CONTRACTS=false; break; }
done
$ALL_CONTRACTS && pass || fail "missing contract"

gate "27 chapter pages have PCU frontmatter"
ALL_PCU=true
for f in book/src/chapters/ch*.md; do
  grep -q "PCU:" <<< "$(head -1 "$f")" || { ALL_PCU=false; break; }
done
$ALL_PCU && pass || fail "missing PCU"

gate "every contract has 5 falsification conditions"
ALL_CONDITIONS=true
for c in contracts/apr-book-ch*-v1.yaml; do
  [ "$(grep -c 'condition:' "$c")" -ge 5 ] || { ALL_CONDITIONS=false; break; }
done
$ALL_CONDITIONS && pass || fail "contract with <5 conditions"
echo ""

# === JOB 5: Namespace discipline ===
echo "Job 5: Namespace Discipline"
gate "no legacy imports in examples"
LEGACY=$(grep -rlE 'use (trueno|realizar|entrenar|batuta|presentar|renacer)::' crates/aprender-core/examples/ch*.rs 2>/dev/null | wc -l) || LEGACY=0
[ "$LEGACY" -eq 0 ] && pass || fail "$LEGACY files with legacy imports"

gate "no legacy imports in book pages"
LEGACY_BOOK=$(grep -rlE 'use (trueno|realizar|entrenar|batuta|presentar|renacer)::' book/src/ 2>/dev/null | wc -l) || LEGACY_BOOK=0
[ "$LEGACY_BOOK" -eq 0 ] && pass || fail "$LEGACY_BOOK files"

gate "no placeholder text"
PLACEHOLDERS=$(grep -rliE '\bTODO\b|\bTBD\b|\bWIP\b|\bcoming soon\b|\bunder construction\b' book/src/ 2>/dev/null | grep -v SUMMARY | grep -v zero-tolerance | grep -v jidoka | wc -l) || PLACEHOLDERS=0
[ "$PLACEHOLDERS" -eq 0 ] && pass || fail "$PLACEHOLDERS files"
echo ""

# === JOB 6: SUMMARY.md integrity ===
echo "Job 6: SUMMARY.md Integrity"
gate "no dead links"
DEAD=0
while IFS= read -r line; do
  p=$(echo "$line" | grep -oP '\./[^)]+' || true)
  [ -n "$p" ] && [ ! -f "book/src/${p#./}" ] && DEAD=$((DEAD+1))
done < book/src/SUMMARY.md
[ "$DEAD" -eq 0 ] && pass || fail "$DEAD dead links"

gate "all 27 chapters listed"
ALL_LISTED=true
for n in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27; do
  grep -q "ch${n}-" book/src/SUMMARY.md || { ALL_LISTED=false; break; }
done
$ALL_LISTED && pass || fail "chapter missing from SUMMARY.md"
echo ""

# === RESULT ===
echo "═══════════════════════════════════════════════════════"
echo "  Gates: $PASSED_GATES/$TOTAL_GATES passed"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: ALL GATES PASS — safe to push"
else
    echo "  RESULT: FAILED — fix before pushing"
    exit 1
fi
