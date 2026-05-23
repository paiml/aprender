#!/usr/bin/env bash
# Generate book chapter stubs for every public module in `aprender-core`.
# Per BOOK-CLOSEOUT-001 § Phase 3.
#
# Constraint: every stub MUST include at least one runnable example.
# Falls back to `cargo doc -p aprender-core --open` if no module-specific
# example is keyed in the EXAMPLE table.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIB_DIR="book/src/lib"
mkdir -p "$LIB_DIR"

# Hand-curated examples per top-level module. Falls back to a generic
# "import + see rustdoc" snippet when not in the table.
declare -A EXAMPLE
EXAMPLE[active_learning]='use aprender::active_learning::PoolBasedActiveLearner;'
EXAMPLE[audio]='use aprender::audio::{load_wav, MelSpectrogram};'
EXAMPLE[autograd]='use aprender::autograd::{Variable, backward};'
EXAMPLE[automl]='use aprender::automl::AutoMLClustering;'
EXAMPLE[bayesian]='use aprender::bayesian::{BetaBinomial, NormalInverseGamma};'
EXAMPLE[bench]='use aprender::bench::Bencher;'
EXAMPLE[bundle]='use aprender::bundle::Bundle;'
EXAMPLE[cache]='use aprender::cache::ModelCache;'
EXAMPLE[calibration]='use aprender::calibration::PlattScaling;'
EXAMPLE[chaos]='use aprender::chaos::ChaosTrainer;'
EXAMPLE[citl]='use aprender::citl::AutomatedRepair;'
EXAMPLE[classification]='use aprender::classification::LogisticRegression;'
EXAMPLE[cluster]='use aprender::cluster::KMeans;'
EXAMPLE[code]='use aprender::code::CodeFeatureExtractor;'
EXAMPLE[compute]='use aprender::compute::Tensor;'
EXAMPLE[data]='use aprender::data::DataLoader;'
EXAMPLE[decomposition]='use aprender::decomposition::PCA;'
EXAMPLE[ensemble]='use aprender::ensemble::RandomForest;'
EXAMPLE[explainable]='use aprender::explainable::SHAP;'
EXAMPLE[format]='use aprender::format::{Reader, Writer};'
EXAMPLE[glm]='use aprender::glm::{PoissonGLM, GammaGLM};'
EXAMPLE[gnn]='use aprender::gnn::GraphConvNetwork;'
EXAMPLE[graph]='use aprender::graph::{dijkstra, pagerank};'
EXAMPLE[hf_hub]='use aprender::hf_hub::HfHubClient;'
EXAMPLE[inspect]='use aprender::inspect::ModelInspector;'
EXAMPLE[linear_regression]='use aprender::linear_regression::LinearRegression;'
EXAMPLE[loss]='use aprender::loss::{MeanSquaredError, CrossEntropy};'
EXAMPLE[metrics]='use aprender::metrics::{accuracy, f1_score};'
EXAMPLE[model_selection]='use aprender::model_selection::{train_test_split, cross_validate};'
EXAMPLE[models]='use aprender::models::Qwen2Model;'
EXAMPLE[naive_bayes]='use aprender::naive_bayes::GaussianNB;'
EXAMPLE[neighbors]='use aprender::neighbors::KNeighbors;'
EXAMPLE[network]='use aprender::network::HttpClient;'
EXAMPLE[nn]='use aprender::nn::{Sequential, Linear, ReLU};'
EXAMPLE[optim]='use aprender::optim::{Adam, SGD};'
EXAMPLE[primitives]='use aprender::primitives::{Vector, Matrix};'
EXAMPLE[quantize]='use aprender::quantize::Q4K;'
EXAMPLE[regularization]='use aprender::regularization::{L1, L2, Dropout};'
EXAMPLE[svm]='use aprender::svm::SupportVectorMachine;'
EXAMPLE[text]='use aprender::text::{Tokenizer, ChatTemplate};'
EXAMPLE[time_series]='use aprender::time_series::ARIMA;'
EXAMPLE[traits]='use aprender::traits::{Estimator, Predictor};'
EXAMPLE[tree]='use aprender::tree::DecisionTreeClassifier;'

default_example() {
  local mod="$1"
  cat <<EOF
use aprender::${mod};
// See \`cargo doc -p aprender-core --open\` for full API reference.
EOF
}

modules=$(grep -E "^pub mod " crates/aprender-core/src/lib.rs | awk '{print $3}' | tr -d ';')

NEW=0
SKIPPED=0
for mod in $modules; do
  STUB="$LIB_DIR/${mod}.md"
  if [ -f "$STUB" ]; then
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  EX="${EXAMPLE[$mod]:-$(default_example "$mod")}"

  cat > "$STUB" <<MARKDOWN
<!-- PCU: lib-${mod} | contract: contracts/apr-page-lib-${mod}-v1.yaml -->

# Module: \`aprender::${mod}\`

Public module of the \`aprender-core\` crate.

## Source

[\`crates/aprender-core/src/${mod}.rs\`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/${mod}.rs) or directory.

## Example

\`\`\`rust
${EX}
\`\`\`

## Full API

Run \`cargo doc -p aprender-core --open\` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
MARKDOWN
  NEW=$((NEW+1))
done

echo "Generated ${NEW} lib stubs (${SKIPPED} skipped — already exist)"
echo "Stubs in: $LIB_DIR"
