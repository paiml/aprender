#!/bin/bash

cd "$(dirname "$0")/src" || exit 1

# Create placeholder files
files=(
  "architecture/design-principles.md"
  "architecture/out-of-core-execution.md"
  "architecture/heterogeneous-computing.md"
  "components/storage/arrow-backend.md"
  "components/storage/parquet-integration.md"
  "components/storage/morsel-driven.md"
  "components/storage/gpu-transfer-queue.md"
  "components/dispatcher/selection-algorithm.md"
  "components/dispatcher/cost-model.md"
  "components/dispatcher/5x-rule.md"
  "components/dispatcher/performance.md"
  "components/query/jit-compiler.md"
  "components/query/kernel-fusion.md"
  "components/query/operator-variants.md"
  "components/gpu/parallel-reduction.md"
  "components/gpu/aggregations.md"
  "components/gpu/hash-join.md"
  "components/gpu/memory-management.md"
  "components/simd/trueno-integration.md"
  "components/simd/simd-primitives.md"
  "components/simd/cpu-optimization.md"
  "tdd/red-green-refactor.md"
  "tdd/test-first.md"
  "tdd/property-based-testing.md"
  "tdd/integration-testing.md"
  "tdd/backend-equivalence.md"
  "toyota/poka-yoke.md"
  "toyota/genchi-genbutsu.md"
  "toyota/muda.md"
  "toyota/jidoka.md"
  "toyota/heijunka.md"
  "toyota/kaizen.md"
  "quality/tdg-score.md"
  "quality/code-coverage.md"
  "quality/mutation-testing.md"
  "quality/clippy.md"
  "quality/ci.md"
  "academic/research-papers.md"
  "academic/leis-2014.md"
  "academic/funke-2018.md"
  "academic/gregg-2011.md"
  "academic/bress-2014.md"
  "academic/neumann-2011.md"
  "academic/wu-2012.md"
  "dev/getting-started.md"
  "dev/building.md"
  "dev/running-tests.md"
  "dev/contributing.md"
  "dev/roadmap.md"
  "case-studies/core-002.md"
  "case-studies/proptest-morsels.md"
  "case-studies/integration-pipeline.md"
  "performance/benchmarking.md"
  "performance/backend-comparison.md"
  "performance/scalability.md"
  "performance/optimization.md"
  "troubleshooting/common-issues.md"
  "troubleshooting/gpu-setup.md"
  "troubleshooting/debugging.md"
  "troubleshooting/performance-debugging.md"
  "appendix/glossary.md"
  "appendix/references.md"
  "appendix/api-docs.md"
  "appendix/license.md"
)

for file in "${files[@]}"; do
  if [ ! -f "$file" ]; then
    # Extract title from filename
    title=$(basename "$file" .md | sed 's/-/ /g' | sed 's/\b./\U&/g')

    # Create file with placeholder
    cat > "$file" << EOF
# $title

TODO: Content coming soon.
EOF
    echo "Created: $file"
  fi
done

echo "Done! Created $(find . -name '*.md' | wc -l) markdown files total"
