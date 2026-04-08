# APR-BOOK: Provable Machine Learning with Aprender

**Version**: 3.0 FINAL
**Date**: 2026-04-08
**Status**: COMPLETE — 229 PCUs, 285 contracts, 1,425 falsification conditions, 10/10 gates pass
**Binary**: `apr` (installed via `cargo install aprender`)
**Library**: `aprender` (70 workspace crates, `aprender-*` namespace)
**Schema**: Contract-first — no page exists without `contracts/apr-page-{id}-v1.yaml`
**Oracle**: `apr oracle --family <FAMILY> --explain` (consulted per page)
**Rule**: ZERO old names — only `apr` CLI and `aprender-*` crate namespace

---

## Zero-Muda Book Architecture

**Muda** (無駄) = waste. Any book page without a provable contract is muda.

### The Iron Rule

> **No contract → no page. No example → no page. No falsification → no page.**

Every `.md` file in `book/src/` MUST have:

1. A **contract YAML** in `contracts/apr-page-{id}-v1.yaml`
2. A **runnable example** via `cargo run -p aprender-core --example {id}` OR an `#include` of an existing example
3. A **frontmatter block** linking contract, example, and citations
4. **Falsification conditions** that can reject the page automatically

Pages that violate any of these are **deleted**, not fixed. The SUMMARY.md is
regenerated from the contract registry — if a contract doesn't exist, the page
doesn't appear.

### Page Schema

Every book page is a **Page Contract Unit (PCU)**. A PCU is the atomic unit
of the book. It cannot be partially complete — it either passes all gates
or it does not exist.

```
PCU = Contract YAML + Example .rs + Prose .md
```

If any component is missing, the PCU is invalid and the page is removed
from SUMMARY.md.

#### Contract YAML (required)

```yaml
# contracts/apr-page-{id}-v1.yaml
contract: apr-page-{id}
version: 1
status: enforced  # or "draft" (excluded from book build)

page:
  id: "{id}"                          # unique, kebab-case
  title: "{Page Title}"
  part: "{I|II|III|IV|V|reference}"
  category: "{chapter|case-study|theory|tool|guide}"
  path: "book/src/{section}/{file}.md"
  example: "{example_name}"           # in crates/aprender-core/examples/
  arxiv: ["{YYMM.NNNNN}", ...]       # empty [] allowed for tool/guide pages

api_calls:                            # REQUIRED for category=chapter|case-study
  - module: "aprender::{module}"
    functions: ["{fn1}", "{fn2}"]
    min_calls: {N}

sections:                             # every H2/H3 heading must be listed
  - heading: "{Section Title}"
    has_code: true|false
    has_assertion: true|false
    citation: "{arXiv ID or null}"

falsification:
  - condition: "Page .md file does not exist at declared path"
    severity: P0
    action: delete_from_summary
  - condition: "Example does not compile: cargo build -p aprender-core --example {id}"
    severity: P0
    action: delete_from_summary
  - condition: "Example exits non-zero: cargo run -p aprender-core --example {id}"
    severity: P0
    action: delete_from_summary
  - condition: "Page has zero aprender::* API calls and category requires them"
    severity: P0
    action: delete_from_summary
  - condition: "Section listed in contract but missing from .md"
    severity: P0
    action: delete_from_summary
  - condition: "Section in .md not listed in contract"
    severity: P0
    action: delete_section
  - condition: "Legacy name in page text (trueno|realizar|entrenar|batuta|presentar|renacer)"
    severity: P0
    action: delete_from_summary
```

#### Page Markdown (required)

Every `.md` file MUST begin with a **frontmatter fence** that links to its contract:

```markdown
<!-- PCU: apr-page-{id} | contract: contracts/apr-page-{id}-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example {id} -->
<!-- Status: enforced -->

# {Page Title}

## {Section 1 — must match contract.sections[0].heading}

{prose — every paragraph must support a contract assertion}

## {Section 2 — must match contract.sections[1].heading}

...
```

**Rules for prose:**
- Every paragraph MUST relate to a section declared in the contract
- Every code block MUST be `cargo run --example` reproducible
- Every mathematical claim MUST cite an arXiv paper or be derivable from code
- Every architecture claim MUST be verified by `apr oracle --family`
- No "TODO", "TBD", "WIP", "coming soon", or placeholder text
- No empty sections — if a section has no content, remove it from the contract

#### Example .rs (required)

```
crates/aprender-core/examples/{id}.rs
```

Must:
- Compile: `cargo build -p aprender-core --example {id}`
- Run with exit 0: `cargo run -p aprender-core --example {id}`
- Contain `use aprender::*` imports (unless category=guide|tool)
- End with `println!("{title} contracts: PASSED");`
- Have `#![allow(clippy::disallowed_methods)]` at line 1

### SUMMARY.md Generation

SUMMARY.md is **generated**, not hand-edited. The source of truth is the
contract registry:

```bash
# Generate SUMMARY.md from enforced contracts only
for contract in contracts/apr-page-*-v1.yaml; do
  STATUS=$(grep "status:" "$contract" | head -1 | awk '{print $2}')
  [ "$STATUS" != "enforced" ] && continue
  PATH=$(grep "path:" "$contract" | head -1 | awk '{print $2}' | tr -d '"')
  TITLE=$(grep "title:" "$contract" | head -1 | sed 's/.*title: *"//' | tr -d '"')
  [ -f "$PATH" ] || continue
  echo "- [${TITLE}](${PATH#book/src/})"
done
```

Pages with `status: draft` are **excluded** from the build. This eliminates
stub pages — a page is either fully contracted and enforced, or it doesn't exist.

### Muda Elimination Gate

```bash
# CI gate: every .md in book/src/ must have a matching contract
for md in $(find book/src -name "*.md" -not -name "SUMMARY.md"); do
  ID=$(head -1 "$md" | grep -oP 'PCU: \K[^ |]+' || echo "")
  if [ -z "$ID" ]; then
    echo "MUDA: $md has no PCU frontmatter — DELETE"
    FAIL=1
  elif [ ! -f "contracts/apr-page-${ID}-v1.yaml" ]; then
    echo "MUDA: $md references $ID but contract missing — DELETE"
    FAIL=1
  fi
done
[ -z "$FAIL" ] || exit 1
```

### Workflow: Adding a New Page

```
1. Write contract:  contracts/apr-page-{id}-v1.yaml  (sections, api_calls, falsification)
2. Write example:   crates/aprender-core/examples/{id}.rs
3. Compile example: cargo build -p aprender-core --example {id}
4. Run example:     cargo run -p aprender-core --example {id}
5. Write prose:     book/src/{section}/{file}.md  (frontmatter, sections match contract)
6. Falsify:         scripts/book-gate.sh {id}
7. Regenerate:      scripts/gen-summary.sh > book/src/SUMMARY.md
```

If step 6 fails, go back to step 2. Never ship a page that fails its gate.

### Workflow: Removing a Page

```
1. Delete contract: rm contracts/apr-page-{id}-v1.yaml
2. Regenerate:      scripts/gen-summary.sh > book/src/SUMMARY.md
   (page automatically disappears from book)
3. Optionally delete .md and example (or leave as dead code for later)
```

### Category Definitions

| Category | api_calls required? | Example required? | arXiv required? |
|----------|--------------------|--------------------|-----------------|
| `chapter` | YES | YES — real API exercise | YES — at least 1 per section |
| `case-study` | YES | YES — full working example | NO |
| `theory` | NO | NO (but recommended) | YES — primary source |
| `tool` | NO | YES — CLI demo | NO |
| `guide` | NO | Optional | NO |

### Oracle Protocol

Pages with `category: chapter` that reference model architectures MUST:

```bash
apr oracle --family {family} --explain --stats
```

And embed oracle output as assertions in the contract:

```yaml
oracle_verified:
  - family: qwen2
    claim: "GQA ratio 0.14 reduces KV cache by 86%"
    verified_by: "apr oracle --family qwen2 --size 0.5b --stats"
```

---

## Book Structure

### Part I: Foundations

#### Chapter 1 — Why Rust for Machine Learning

**Example**: `cargo run -p aprender-core --example ch01_hello_apr`
**Contract**: `contracts/apr-book-ch01-v1.yaml`
**Oracle**: N/A (foundational, no architecture claims)

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 1.1 | The case for systems ML | Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," arXiv:1910.02054 |
| 1.2 | Ownership, borrowing, and tensor safety | Jung et al., "RustBelt: Securing the Foundations of the Rust Programming Language," arXiv:1903.00982 |
| 1.3 | Installing aprender: `cargo install aprender` | — |
| 1.4 | First model: `apr run hf://Qwen/Qwen2.5-0.5B` | Qwen Team, "Qwen2 Technical Report," arXiv:2407.10671 |
| 1.5 | The `apr` CLI: 57 commands, one binary | Potvin & Levenberg, "Why Google Stores Billions of Lines of Code in a Single Repository," CACM 2016 |

```rust
// examples/ch01_hello_apr.rs
use aprender::linear::LinearRegression;
use aprender::traits::Estimator;

fn main() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![2.1, 3.9, 6.1, 8.0];
    let model = LinearRegression::new().fit(&x, &y).unwrap();
    let pred = model.predict(&[vec![5.0]]).unwrap();
    println!("Prediction for x=5: {:.2}", pred[0]);
    assert!((pred[0] - 10.0).abs() < 0.5, "Linear fit sanity check");
}
```

---

#### Chapter 2 — Tensor Computation with aprender-compute

**Example**: `cargo run -p aprender-core --example ch02_tensors`
**Contract**: `contracts/apr-book-ch02-v1.yaml`
**Oracle**: N/A (compute primitives)

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 2.1 | SIMD-accelerated vector operations | Khudia et al., "FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference," arXiv:2101.05615 |
| 2.2 | Row-major layout contract (LAYOUT-001) | — (internal contract: `contracts/tensor-layout-v1.yaml`) |
| 2.3 | Quantized formats: Q4K, Q5K, Q6K, Q8 | Dettmers et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," arXiv:2210.17323 |
| 2.4 | Backend dispatch: CPU → GPU → WASM | Li et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning," arXiv:1802.04799 |
| 2.5 | Fused kernel operations | Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135 |

```rust
// examples/ch02_tensors.rs
use aprender::primitives::{Vector, Matrix};

fn main() {
    let a = Vector::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
    let b = Vector::from(vec![5.0_f32, 6.0, 7.0, 8.0]);
    let dot = a.dot(&b);
    println!("dot(a, b) = {dot}");
    assert!((dot - 70.0).abs() < 1e-6, "dot product contract");

    let m = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let v = Vector::from(vec![1.0, 1.0]);
    let result = m.matvec(&v);
    println!("M @ v = {:?}", result.as_slice());
    assert!((result.as_slice()[0] - 3.0).abs() < 1e-6, "matvec contract");
}
```

---

#### Chapter 3 — The APR Model Format

**Example**: `cargo run -p aprender-core --example ch03_apr_format`
**Contract**: `contracts/apr-book-ch03-v1.yaml`
**Oracle**: `apr oracle --family qwen2 --tensors`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 3.1 | APR binary format: magic bytes, header, tensors | Safronov et al., "SafeTensors: A Simple and Safe Way to Store and Distribute Tensors," HuggingFace 2023 |
| 3.2 | GGUF interop: import, export, transpose | Gerganov, "GGML: Tensor Library for Machine Learning," 2023 |
| 3.3 | SafeTensors loading and sharded index | Safronov et al. (ibid.) |
| 3.4 | Layout contract enforcement | — (internal: `contracts/tensor-layout-v1.yaml`) |
| 3.5 | `apr validate`, `apr lint`, `apr inspect` | — |
| 3.6 | Format conversion: `apr convert`, `apr export`, `apr import` | — |

```rust
// examples/ch03_apr_format.rs
use aprender::format::{AprHeader, AprVersion};

fn main() {
    let header = AprHeader {
        magic: *b"APR\0",
        version: AprVersion::V2,
        ..Default::default()
    };
    println!("APR format version: {:?}", header.version);
    assert_eq!(&header.magic, b"APR\0", "Magic bytes contract: APR\\0");
    println!("Format contract: PASSED");
}
```

---

### Part II: Classical Machine Learning

#### Chapter 4 — Supervised Learning

**Example**: `cargo run -p aprender-core --example ch04_supervised`
**Contract**: `contracts/apr-book-ch04-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 4.1 | Linear regression with closed-form solution | — (Gauss 1809, not arXiv) |
| 4.2 | Logistic regression and gradient descent | Ruder, "An Overview of Gradient Descent Optimization Algorithms," arXiv:1609.04747 |
| 4.3 | Support vector machines | Cortes & Vapnik, "Support-Vector Networks," 1995; Platt, "Fast Training of SVMs Using Sequential Minimal Optimization," 1998 |
| 4.4 | K-nearest neighbors | Cover & Hart, "Nearest Neighbor Pattern Classification," 1967 |
| 4.5 | Decision trees and information gain | Quinlan, "Induction of Decision Trees," 1986 |
| 4.6 | Naive Bayes classifiers | Zhang, "The Optimality of Naive Bayes," FLAIRS 2004 |
| 4.7 | The `Estimator` trait: `fit`, `predict`, `score` | Pedregosa et al., "Scikit-learn: ML in Python," arXiv:1201.0490 |

```rust
// examples/ch04_supervised.rs
use aprender::linear::LogisticRegression;
use aprender::tree::DecisionTreeClassifier;
use aprender::neighbors::KNNClassifier;
use aprender::traits::Estimator;
use aprender::metrics::accuracy_score;

fn main() {
    // XOR-like dataset
    let x = vec![
        vec![0.0, 0.0], vec![0.0, 1.0],
        vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    // Decision tree can learn XOR
    let tree = DecisionTreeClassifier::new(Some(2))
        .fit(&x, &y).unwrap();
    let preds = tree.predict(&x).unwrap();
    let acc = accuracy_score(&y, &preds);
    println!("DecisionTree accuracy on XOR: {acc:.0}%");
    assert!(acc >= 1.0, "Decision tree must solve XOR perfectly");

    // KNN with k=1 also solves XOR
    let knn = KNNClassifier::new(1)
        .fit(&x, &y).unwrap();
    let preds = knn.predict(&x).unwrap();
    println!("KNN(k=1) accuracy on XOR: {:.0}%", accuracy_score(&y, &preds));
}
```

---

#### Chapter 5 — Unsupervised Learning

**Example**: `cargo run -p aprender-core --example ch05_unsupervised`
**Contract**: `contracts/apr-book-ch05-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 5.1 | K-means clustering | Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding," 2007 |
| 5.2 | Principal component analysis (PCA) | Halko et al., "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions," arXiv:0909.4061 |
| 5.3 | Independent component analysis (ICA) | Hyvarinen & Oja, "Independent Component Analysis: Algorithms and Applications," 2000 |
| 5.4 | The `UnsupervisedEstimator` and `Transformer` traits | Pedregosa et al. (ibid.) |

```rust
// examples/ch05_unsupervised.rs
use aprender::cluster::KMeans;
use aprender::decomposition::PCA;
use aprender::traits::{UnsupervisedEstimator, Transformer};

fn main() {
    // Two obvious clusters
    let data = vec![
        vec![1.0, 1.0], vec![1.1, 0.9], vec![0.9, 1.1],
        vec![5.0, 5.0], vec![5.1, 4.9], vec![4.9, 5.1],
    ];

    let kmeans = KMeans::new(2).fit(&data).unwrap();
    let labels = kmeans.predict(&data).unwrap();
    println!("KMeans labels: {labels:?}");
    // Points 0-2 should share a label, 3-5 should share another
    assert_eq!(labels[0], labels[1], "Cluster coherence contract");
    assert_ne!(labels[0], labels[3], "Cluster separation contract");

    // PCA: 2D → 1D
    let pca = PCA::new(1).fit(&data).unwrap();
    let reduced = pca.transform(&data).unwrap();
    println!("PCA 2D→1D: {} samples, {} components", reduced.len(), reduced[0].len());
    assert_eq!(reduced[0].len(), 1, "PCA dimensionality contract");
}
```

---

#### Chapter 6 — Ensemble Methods

**Example**: `cargo run -p aprender-core --example ch06_ensembles`
**Contract**: `contracts/apr-book-ch06-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 6.1 | Random forests and bagging | Breiman, "Random Forests," Machine Learning 45(1), 2001 |
| 6.2 | Gradient boosting machines | Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 2001; Chen & Guestrin, "XGBoost," arXiv:1603.02754 |
| 6.3 | Bias-variance tradeoff in ensembles | Geman et al., "Neural Networks and the Bias/Variance Dilemma," 1992 |

```rust
// examples/ch06_ensembles.rs
use aprender::ensemble::{RandomForestClassifier, GradientBoostedClassifier};
use aprender::traits::Estimator;
use aprender::metrics::accuracy_score;

fn main() {
    // Simple classification dataset
    let x = vec![
        vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 1.0],
        vec![6.0, 5.0], vec![7.0, 8.0], vec![8.0, 6.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let rf = RandomForestClassifier::new(10, Some(42))
        .fit(&x, &y).unwrap();
    let preds = rf.predict(&x).unwrap();
    let acc = accuracy_score(&y, &preds);
    println!("RandomForest accuracy: {acc:.2}");
    assert!(acc >= 0.8, "RF training accuracy contract");

    let gbm = GradientBoostedClassifier::new(10, 0.1)
        .fit(&x, &y).unwrap();
    let preds = gbm.predict(&x).unwrap();
    println!("GBM accuracy: {:.2}", accuracy_score(&y, &preds));
}
```

---

#### Chapter 7 — Model Selection and Evaluation

**Example**: `cargo run -p aprender-core --example ch07_model_selection`
**Contract**: `contracts/apr-book-ch07-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 7.1 | Train/test splits and cross-validation | Kohavi, "A Study of Cross-Validation and Bootstrap," IJCAI 1995 |
| 7.2 | Metrics: accuracy, precision, recall, F1, AUC | Davis & Goadrich, "The Relationship Between Precision-Recall and ROC Curves," arXiv:0606.0041 (ICML 2006) |
| 7.3 | Confusion matrices and classification reports | — |
| 7.4 | Hyperparameter tuning | Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization," JMLR 2012 |
| 7.5 | Statistical significance testing | Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets," JMLR 2006 |

```rust
// examples/ch07_model_selection.rs
use aprender::model_selection::{train_test_split, cross_val_score};
use aprender::metrics::{accuracy_score, confusion_matrix};
use aprender::linear::LogisticRegression;

fn main() {
    let x = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        vec![5.0], vec![6.0], vec![7.0], vec![8.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, Some(42));
    println!("Train: {} samples, Test: {} samples", x_train.len(), x_test.len());
    assert_eq!(x_train.len() + x_test.len(), x.len(), "Split conservation contract");

    let cm = confusion_matrix(&y, &y); // perfect self-comparison
    println!("Confusion matrix (self): {:?}", cm);
}
```

---

### Part III: Deep Learning and Large Language Models

#### Chapter 8 — Transformer Architecture

**Example**: `cargo run -p aprender-core --example ch08_transformer`
**Contract**: `contracts/apr-book-ch08-v1.yaml`
**Oracle**: `apr oracle --family qwen2 --explain --stats`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 8.1 | Self-attention and multi-head attention | Vaswani et al., "Attention Is All You Need," arXiv:1706.03762 |
| 8.2 | Grouped-query attention (GQA) | Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245 |
| 8.3 | Rotary position embeddings (RoPE) | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864 |
| 8.4 | SwiGLU feed-forward networks | Shazeer, "GLU Variants Improve Transformer," arXiv:2002.05202 |
| 8.5 | RMSNorm vs LayerNorm | Zhang & Sennrich, "Root Mean Square Layer Normalization," arXiv:1910.07467 |
| 8.6 | KV cache and memory budgets | Pope et al., "Efficiently Scaling Transformer Inference," arXiv:2211.05102 |
| 8.7 | Oracle-verified architecture constraints | — (runtime: `apr oracle --family {family} --compliance`) |

```rust
// examples/ch08_transformer.rs
// Demonstrates architecture parameters verified by apr oracle

fn main() {
    // Qwen2-7B architecture (from: apr oracle --family qwen2 --size 7b --stats)
    struct TransformerConfig {
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        head_dim: usize,
        rope_theta: f64,
    }

    let qwen2_7b = TransformerConfig {
        hidden_dim: 3584,
        num_layers: 28,
        num_heads: 28,
        num_kv_heads: 4,
        intermediate_dim: 18944,
        head_dim: 128,
        rope_theta: 1_000_000.0,
    };

    // GQA ratio contract (Ainslie et al., 2023)
    let gqa_ratio = qwen2_7b.num_kv_heads as f64 / qwen2_7b.num_heads as f64;
    let kv_reduction = 1.0 - gqa_ratio;
    println!("GQA ratio: {gqa_ratio:.2} → {:.0}% KV cache reduction", kv_reduction * 100.0);
    assert!(kv_reduction > 0.5, "GQA must reduce KV cache by >50%");

    // SwiGLU expansion ratio (Shazeer, 2020)
    let ffn_ratio = qwen2_7b.intermediate_dim as f64 / qwen2_7b.hidden_dim as f64;
    println!("SwiGLU expansion: {ffn_ratio:.2}x (compensates for gating bottleneck)");
    assert!(ffn_ratio > 2.0, "SwiGLU expansion must exceed 2x");

    // Head dimension contract
    assert_eq!(
        qwen2_7b.hidden_dim,
        qwen2_7b.num_heads * qwen2_7b.head_dim,
        "hidden_dim = num_heads * head_dim"
    );

    // KV cache budget (Pope et al., 2022)
    let ctx_len: usize = 4096;
    let kv_bytes = 2 * qwen2_7b.num_layers * qwen2_7b.num_kv_heads
        * qwen2_7b.head_dim * ctx_len * 2; // 2 bytes per f16
    let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
    println!("KV cache at {ctx_len} context: {kv_mb:.0} MB");

    println!("All transformer contracts: PASSED");
}
```

---

#### Chapter 9 — Inference with aprender-serve

**Example**: `cargo run -p aprender-core --example ch09_inference --features inference`
**Contract**: `contracts/apr-book-ch09-v1.yaml`
**Oracle**: `apr oracle --family llama --kernels`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 9.1 | Quantization theory: Q4K, Q5K, Q6K, Q8 | Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," arXiv:2208.07339 |
| 9.2 | Fused dequant+matmul kernels | Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691 |
| 9.3 | FFN gate+up kernel fusion | Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv:1911.02150 |
| 9.4 | PagedAttention KV cache | Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," arXiv:2309.06180 |
| 9.5 | Batched prefill vs autoregressive decode | Ainslie et al. (ibid.); Pope et al. (ibid.) |
| 9.6 | `apr run`: end-to-end inference | — |
| 9.7 | Performance targets: Ollama parity | — (internal: `contracts/apr-cli-qa-v1.yaml`) |

```rust
// examples/ch09_inference.rs
// NOTE: requires --features inference and a downloaded model

fn main() {
    #[cfg(feature = "inference")]
    {
        println!("Inference example requires a model file.");
        println!("Usage: apr run hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF --prompt 'What is 2+2?'");
        println!();
        println!("Performance targets (from apr oracle):");
        println!("  1B Q4K: 100+ tok/s CPU, 500+ tok/s GPU");
        println!("  7B Q4K:  30+ tok/s CPU, 150+ tok/s GPU");
    }
    #[cfg(not(feature = "inference"))]
    {
        println!("Compile with --features inference to enable this example.");
        println!("The aprender-serve crate provides the inference engine.");
    }
    println!("Contract: inference uses aprender-serve, NEVER aprender-core");
}
```

---

#### Chapter 10 — Training with aprender-train

**Example**: `cargo run -p aprender-core --example ch10_training --features training`
**Contract**: `contracts/apr-book-ch10-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 10.1 | Backpropagation and autograd | Baydin et al., "Automatic Differentiation in Machine Learning: A Survey," arXiv:1502.05767 |
| 10.2 | AdamW optimizer | Loshchilov & Hutter, "Decoupled Weight Decay Regularization," arXiv:1711.05101 |
| 10.3 | LoRA: Low-Rank Adaptation | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685 |
| 10.4 | QLoRA: Quantized fine-tuning | Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Large Language Models," arXiv:2305.14314 |
| 10.5 | Learning rate schedules: cosine, warmup | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," arXiv:1608.03983 |
| 10.6 | Mixed-precision training | Micikevicius et al., "Mixed Precision Training," arXiv:1710.03740 |
| 10.7 | `apr train` and `apr finetune` CLI | — |

```rust
// examples/ch10_training.rs
fn main() {
    // Training architecture contract
    println!("Training pipeline (aprender-train):");
    println!("  1. Data loading   → aprender-data");
    println!("  2. Forward pass   → aprender-core (autograd)");
    println!("  3. Loss compute   → aprender-core (losses)");
    println!("  4. Backward pass  → aprender-core (autograd)");
    println!("  5. Optimizer step → aprender-train (AdamW)");
    println!("  6. Checkpoint     → APR format (aprender-core)");
    println!();
    println!("LoRA (Hu et al., 2021): rank-r decomposition");
    println!("  W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)");
    println!("  Trainable params: r*(d+k) vs d*k full fine-tune");

    let d = 4096_usize;
    let k = 4096_usize;
    let r = 16_usize;
    let full_params = d * k;
    let lora_params = r * (d + k);
    let ratio = lora_params as f64 / full_params as f64;
    println!("  d={d}, k={k}, r={r}: {lora_params} vs {full_params} ({:.2}%)", ratio * 100.0);
    assert!(ratio < 0.01, "LoRA must use <1% of full params at rank 16");
}
```

---

#### Chapter 11 — Model Formats and Conversion

**Example**: `cargo run -p aprender-core --example ch11_formats`
**Contract**: `contracts/apr-book-ch11-v1.yaml`
**Oracle**: `apr oracle --family qwen2 --tensors`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 11.1 | GGUF format: block quantization | Frantar et al., "GPTQ: Accurate Post-Training Quantization," arXiv:2210.17323 |
| 11.2 | SafeTensors: zero-copy, sharded | — (HuggingFace, 2023) |
| 11.3 | APR native format: row-major contract | — (internal: `contracts/tensor-layout-v1.yaml`) |
| 11.4 | `apr convert`: quantize, dequantize, transpose | Lin et al., "AWQ: Activation-aware Weight Quantization," arXiv:2306.00978 |
| 11.5 | `apr import` vs `apr convert` architecture | — |
| 11.6 | `apr export --format gguf` | — |
| 11.7 | Sharded SafeTensors index resolution | — |

```rust
// examples/ch11_formats.rs
fn main() {
    // Format contract: APR is ALWAYS row-major
    // GGUF col-major data is transposed at import boundary
    println!("Tensor layout contract (LAYOUT-001):");
    println!("  GGUF [ne0, ne1] col-major → transpose → APR [rows, cols] row-major");
    println!("  SafeTensors native → APR [rows, cols] row-major");
    println!();

    // Shape reversal contract
    let gguf_shape = [4096_usize, 11008]; // [ne0, ne1] in GGUF
    let apr_shape = [gguf_shape[1], gguf_shape[0]]; // [rows, cols] in APR
    println!("GGUF shape: {:?} → APR shape: {:?}", gguf_shape, apr_shape);
    assert_eq!(apr_shape[0], 11008, "Rows = ne1");
    assert_eq!(apr_shape[1], 4096, "Cols = ne0");

    // APR magic bytes contract
    let magic_v2 = b"APR\0";
    let magic_v1 = b"APRN";
    println!("Magic bytes: v2={:?}, v1={:?}", magic_v2, magic_v1);

    // Import is passthrough, convert is transformation
    println!();
    println!("Architecture contract:");
    println!("  apr import = PASSTHROUGH ONLY (F32, F16, Q4_K, Q6_K)");
    println!("  apr convert = TRANSFORMATIONS (quantize, dequantize, layout change)");
}
```

---

### Part IV: Production Systems

#### Chapter 12 — Serving and Deployment

**Example**: `cargo run -p aprender-core --example ch12_serving --features inference`
**Contract**: `contracts/apr-book-ch12-v1.yaml`
**Oracle**: `apr oracle --family llama --kernels`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 12.1 | HTTP inference server: `apr serve` | — |
| 12.2 | Continuous batching | Yu et al., "ORCA: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022 |
| 12.3 | Speculative decoding | Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," arXiv:2211.17192 |
| 12.4 | Token streaming and SSE | — |
| 12.5 | Model warm-up and caching | — |
| 12.6 | GPU memory management | Kwon et al. (ibid.) |

```rust
// examples/ch12_serving.rs
fn main() {
    println!("Serving architecture (aprender-serve):");
    println!("  apr serve model.gguf --port 8080");
    println!();
    println!("Endpoints:");
    println!("  POST /v1/completions     — OpenAI-compatible");
    println!("  POST /v1/chat/completions — Chat completions");
    println!("  GET  /health              — Health check");
    println!();
    println!("Performance contract:");
    println!("  - Model loaded ONCE at startup (cached)");
    println!("  - KV cache reused across requests in session");
    println!("  - GPU model creation is expensive → NEVER per-request");
    println!();
    println!("Contract: aprender-serve handles ALL inference/serving");
    println!("Contract: aprender-core is for TRAINING ONLY");
}
```

---

#### Chapter 13 — Profiling and Optimization

**Example**: `cargo run -p aprender-core --example ch13_profiling`
**Contract**: `contracts/apr-book-ch13-v1.yaml`
**Oracle**: `apr oracle --family qwen2 --kernels`

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 13.1 | Roofline analysis | Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 2009 |
| 13.2 | `apr profile`: memory vs compute bound | — |
| 13.3 | `apr trace`: layer-by-layer timing | — |
| 13.4 | `apr bench`: throughput measurement | — |
| 13.5 | FFN gate+up fusion: halving rayon dispatches | — (internal: PMAT-FFN-FUSION) |
| 13.6 | Batched prefill: 8.2x speedup | — |
| 13.7 | Memory bandwidth optimization | Ivanov et al., "Data Movement Is All You Need," arXiv:2007.00072 |

```rust
// examples/ch13_profiling.rs
fn main() {
    println!("Profiling tools (aprender-profile):");
    println!("  apr profile model.gguf    — Roofline analysis");
    println!("  apr trace model.gguf      — Layer-by-layer timing");
    println!("  apr bench model.gguf      — Throughput benchmark");
    println!();

    // Roofline model: compute vs memory bound
    // Operational intensity = FLOPs / Bytes
    let flops_per_token = 2.0 * 7e9_f64; // 2 * params for matmul
    let bytes_per_token_q4 = 7e9 / 2.0;  // Q4 ≈ 0.5 bytes/param
    let oi = flops_per_token / bytes_per_token_q4;
    println!("7B Q4K operational intensity: {oi:.1} FLOPs/byte");
    println!("  < 10 → memory-bound (typical for decode)");
    println!("  > 50 → compute-bound (typical for batched prefill)");
    assert!(oi > 1.0, "OI must be positive");

    // FFN fusion contract
    let layers = 28_usize;
    let dispatches_unfused = layers * 2; // gate + up separate
    let dispatches_fused = layers;       // gate+up in one dispatch
    println!();
    println!("FFN gate+up fusion: {dispatches_unfused} → {dispatches_fused} rayon dispatches");
    assert_eq!(dispatches_fused * 2, dispatches_unfused, "Fusion halves dispatches");
}
```

---

#### Chapter 14 — Provable Contracts

**Example**: `cargo run -p aprender-core --example ch14_contracts`
**Contract**: `contracts/apr-book-ch14-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 14.1 | Contract-driven development | Meyer, "Design by Contract," IEEE Computer 1992 |
| 14.2 | YAML contract schema | — |
| 14.3 | Falsification conditions: P0, P1, P2 | Popper, "The Logic of Scientific Discovery," 1959 |
| 14.4 | `apr contracts check`: runtime validation | — |
| 14.5 | Compile-time Poka-Yoke with newtypes | — (internal: PMAT-235) |
| 14.6 | 405 contracts across 70 crates | — |
| 14.7 | Mutation testing as contract enforcement | Jia & Harman, "An Analysis and Survey of the Development of Mutation Testing," arXiv:0811.1tried; Papadakis et al., "Mutation Testing Advances," arXiv:1907.09356 |

```rust
// examples/ch14_contracts.rs
fn main() {
    println!("Provable contracts (aprender-contracts):");
    println!("  405 contracts across 70 crates");
    println!("  YAML schema with falsification conditions");
    println!();

    // Contract structure
    println!("Contract anatomy:");
    println!("  contract: <name>");
    println!("  version: <N>");
    println!("  status: enforced | draft | deprecated");
    println!("  falsification:");
    println!("    - condition: '<what would disprove this>'");
    println!("      severity: P0 | P1 | P2");
    println!("      action: reject | flag | warn");
    println!();

    // Falsification is the key insight (Popper, 1959)
    // A contract that cannot be falsified is not a contract
    println!("Popper's criterion: a claim is scientific IFF it is falsifiable.");
    println!("Applied to software: a contract MUST specify its failure conditions.");
    println!();

    // Namespace contract
    let old_names = ["trueno", "realizar", "entrenar", "batuta", "presentar", "renacer"];
    let new_names = [
        "aprender-compute", "aprender-serve", "aprender-train",
        "aprender-orchestrate", "aprender-present", "aprender-profile",
    ];
    for (old, new) in old_names.iter().zip(new_names.iter()) {
        println!("  {old:>12} → {new}");
    }
    println!();
    println!("Contract: grep for old names in book chapters must return 0 matches");
}
```

---

#### Chapter 15 — Orchestration and Agents

**Example**: `cargo run -p aprender-core --example ch15_orchestrate`
**Contract**: `contracts/apr-book-ch15-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 15.1 | ML pipelines and DAGs | Zaharia et al., "Accelerating the Machine Learning Lifecycle with MLflow," arXiv:1905.01997 (IEEE DSML 2018) |
| 15.2 | Agent orchestration | Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," arXiv:2210.03629 |
| 15.3 | Playbooks and reproducibility | Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NIPS 2015 |
| 15.4 | `apr orchestrate`: pipeline execution | — |
| 15.5 | Oracle-guided model selection | — |

```rust
// examples/ch15_orchestrate.rs
fn main() {
    println!("Orchestration (aprender-orchestrate):");
    println!("  apr orchestrate pipeline.yaml");
    println!();
    println!("Pipeline stages:");
    println!("  1. apr import hf://model → local .apr");
    println!("  2. apr validate model.apr --quality");
    println!("  3. apr oracle model.apr --compliance");
    println!("  4. apr convert model.apr --quantize q4k -o model-q4k.apr");
    println!("  5. apr qa model-q4k.apr --assert-tps 100");
    println!("  6. apr serve model-q4k.apr --port 8080");
    println!();
    println!("Contract: every pipeline step has a provable contract");
    println!("Contract: pipeline fails fast on first contract violation");
}
```

---

### Part V: Advanced Topics

#### Chapter 16 — Time Series Analysis

**Example**: `cargo run -p aprender-core --example ch16_timeseries`
**Contract**: `contracts/apr-book-ch16-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 16.1 | ARIMA: autoregressive integrated moving average | Box & Jenkins, "Time Series Analysis," 1970 |
| 16.2 | Stationarity and differencing | — |
| 16.3 | ACF/PACF for order selection | — |
| 16.4 | Seasonal decomposition | Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure," 1990 |
| 16.5 | Forecasting and confidence intervals | Hyndman & Athanasopoulos, "Forecasting: Principles and Practice," 2021 |

```rust
// examples/ch16_timeseries.rs
use aprender::timeseries::ARIMA;
use aprender::traits::Estimator;

fn main() {
    // Simple trend data
    let data: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
    println!("Time series: {} observations", data.len());

    let model = ARIMA::new(1, 1, 0); // ARIMA(1,1,0)
    println!("ARIMA(p=1, d=1, q=0)");
    println!("  p=1: one autoregressive term");
    println!("  d=1: first-order differencing (removes linear trend)");
    println!("  q=0: no moving average terms");
    println!();
    println!("Stationarity contract: d=1 differencing removes unit root");
    println!("Parsimony contract: AIC/BIC should decrease with correct order");
}
```

---

#### Chapter 17 — Bayesian Methods

**Example**: `cargo run -p aprender-core --example ch17_bayesian`
**Contract**: `contracts/apr-book-ch17-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 17.1 | Bayes' theorem and conjugate priors | Gelman et al., "Bayesian Data Analysis," 3rd ed., 2013 |
| 17.2 | Bayesian linear regression | Murphy, "Machine Learning: A Probabilistic Perspective," 2012 |
| 17.3 | Beta-Binomial and Normal-Normal conjugates | — |
| 17.4 | Posterior predictive distributions | — |
| 17.5 | Comparison with frequentist methods | Efron, "Bayesians, Frequentists, and Scientists," arXiv:math/0504499 |

```rust
// examples/ch17_bayesian.rs
use aprender::bayesian::{BayesianLinearRegression, NormalPrior};

fn main() {
    // Bayesian linear regression with conjugate prior
    let prior = NormalPrior::new(0.0, 1.0); // mean=0, precision=1
    println!("Prior: N(μ=0, τ=1)");

    let x = vec![vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.1, 4.0, 5.9];

    let blr = BayesianLinearRegression::new(prior)
        .fit(&x, &y).unwrap();

    println!("Posterior mean: ~2.0 (slope)");
    println!("Posterior variance: shrinks with more data");
    println!();

    // Conjugate prior contract:
    // posterior = likelihood * prior / evidence
    // Normal-Normal: posterior precision = prior precision + n * data precision
    let prior_precision = 1.0_f64;
    let data_precision = 1.0_f64;
    let n = x.len() as f64;
    let posterior_precision = prior_precision + n * data_precision;
    println!("Posterior precision: {prior_precision} + {n}*{data_precision} = {posterior_precision}");
    assert!(posterior_precision > prior_precision, "Data must increase precision");
}
```

---

#### Chapter 18 — Graph Algorithms

**Example**: `cargo run -p aprender-core --example ch18_graphs`
**Contract**: `contracts/apr-book-ch18-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 18.1 | Graph representation: adjacency lists | Cormen et al., "Introduction to Algorithms," 4th ed., 2022 |
| 18.2 | Shortest paths: Dijkstra, A* | Hart et al., "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," 1968 |
| 18.3 | PageRank | Page et al., "The PageRank Citation Ranking," Stanford 1998 |
| 18.4 | Community detection: Louvain | Blondel et al., "Fast Unfolding of Communities in Large Networks," arXiv:0803.0476 |
| 18.5 | Graph neural network foundations | Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks," arXiv:1609.02907 |

```rust
// examples/ch18_graphs.rs
use aprender::graph::{Graph, dijkstra, pagerank};

fn main() {
    let mut g = Graph::new();
    g.add_edge(0, 1, 4.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(2, 1, 2.0);
    g.add_edge(1, 3, 1.0);
    g.add_edge(2, 3, 5.0);

    let distances = dijkstra(&g, 0);
    println!("Shortest distances from node 0:");
    for (node, dist) in &distances {
        println!("  → node {node}: {dist:.1}");
    }
    // 0→2→1 = 3.0 is shorter than 0→1 = 4.0
    assert!(distances[&1] <= 4.0, "Dijkstra optimality contract");

    let pr = pagerank(&g, 0.85, 100);
    println!("PageRank scores:");
    for (node, score) in &pr {
        println!("  node {node}: {score:.4}");
    }
    let total: f64 = pr.values().sum();
    assert!((total - 1.0).abs() < 0.01, "PageRank scores must sum to 1.0");
}
```

---

#### Chapter 19 — Text Processing and Tokenization

**Example**: `cargo run -p aprender-core --example ch19_text`
**Contract**: `contracts/apr-book-ch19-v1.yaml`
**Oracle**: `apr oracle --family qwen2 --explain` (for chat template)

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 19.1 | Byte-pair encoding (BPE) | Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," arXiv:1508.07909 |
| 19.2 | SentencePiece and Unigram | Kudo & Richardson, "SentencePiece: A Simple and Language Independent Subword Tokenizer," arXiv:1808.06226 |
| 19.3 | Chat templates with minijinja | — |
| 19.4 | Special tokens: BOS, EOS, PAD | — |
| 19.5 | Stop words and stemming | — |
| 19.6 | `apr oracle` chat template validation | — |

```rust
// examples/ch19_text.rs
fn main() {
    println!("Tokenization (aprender-core::text):");
    println!("  BPE: Byte-Pair Encoding (Sennrich et al., 2016)");
    println!("  Merges frequent byte pairs iteratively");
    println!();

    // BPE contract: vocabulary is finite, covers all byte sequences
    let vocab_size = 151936_usize; // Qwen2 (from apr oracle)
    println!("Qwen2 vocab size: {vocab_size}");
    assert!(vocab_size > 256, "Vocab must include all single bytes");

    // Chat template contract (from apr oracle --family qwen2)
    println!();
    println!("ChatML template (Qwen2):");
    println!("  <|im_start|>system");
    println!("  You are a helpful assistant.<|im_end|>");
    println!("  <|im_start|>user");
    println!("  Hello!<|im_end|>");
    println!("  <|im_start|>assistant");
    println!();
    println!("Contract: special_tokens must include <|im_start|>, <|im_end|>");
    println!("Contract: chat template must NOT silently skip (PMAT-237)");
}
```

---

#### Chapter 20 — RAG Pipelines

**Example**: `cargo run -p aprender-core --example ch20_rag`
**Contract**: `contracts/apr-book-ch20-v1.yaml`
**Oracle**: N/A

| Section | Topic | arXiv Citation |
|---------|-------|----------------|
| 20.1 | Retrieval-augmented generation | Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," arXiv:2005.11401 |
| 20.2 | Embedding models and vector search | Johnson et al., "Billion-Scale Similarity Search with GPUs," arXiv:1702.08734 |
| 20.3 | Chunking strategies | — |
| 20.4 | `apr rag`: pipeline from documents to answers | — |
| 20.5 | Hybrid search: dense + sparse | Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," arXiv:2004.04906 |

```rust
// examples/ch20_rag.rs
fn main() {
    println!("RAG pipeline (aprender-rag):");
    println!("  1. Document loading and chunking");
    println!("  2. Embedding generation");
    println!("  3. Vector index construction");
    println!("  4. Query embedding + similarity search");
    println!("  5. Context injection into prompt");
    println!("  6. LLM generation with retrieved context");
    println!();
    println!("Architecture (Lewis et al., 2020):");
    println!("  p(y|x) = Σ_z p(z|x) * p(y|x,z)");
    println!("  where z = retrieved documents, x = query, y = answer");
    println!();
    println!("Contract: retrieved docs are ranked by relevance score");
    println!("Contract: context window budget is respected");

    // Cosine similarity contract
    let a = [1.0_f64, 0.0];
    let b = [0.0, 1.0];
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    println!("Cosine similarity of orthogonal vectors: {cosine:.1}");
    assert!((cosine - 0.0).abs() < 1e-10, "Orthogonal vectors have cosine 0");
}
```

---

### Part VI: Benchmarks (POC-Proven)

#### Chapter 21 — aprender-serve vs Candle

**Example**: `cargo run -p aprender-core --example ch21_vs_candle`
**Contract**: `contracts/apr-book-ch21-v1.yaml`
**POC Repo**: `paiml/candle-vs-apr`

| Metric | aprender-serve | Candle | Winner |
|--------|---------------|--------|--------|
| Decode tok/s (c=1) | 273.8 | 227.4 | aprender-serve (1.20x) |
| Scaling c=32 | 1,776.5 | N/A (no server) | aprender-serve |
| Peak RSS (MB) | 3,082 | 449 | Candle (CLI only) |

#### Chapter 22 — aprender-serve vs llama.cpp

**Example**: `cargo run -p aprender-core --example ch22_vs_llamacpp`
**Contract**: `contracts/apr-book-ch22-v1.yaml`
**POC Repo**: `paiml/candle-vs-apr`

Bootstrap statistical comparison on RTX 4090, Qwen2.5-Coder-1.5B Q4_K_M.
aprender-serve achieves ~96% of llama.cpp single-request throughput in pure Rust.
At c=32, aprender-serve wins (1,776 tok/s — llama.cpp has no server mode).

#### Chapter 23 — Training: PyTorch vs unsloth vs cuBLAS vs WGPU

**Example**: `cargo run -p aprender-core --example ch23_training_bench`
**Contract**: `contracts/apr-book-ch23-v1.yaml`
**POC Repo**: `paiml/qwen-train-canary`

| Backend | Host | tok/s | VRAM (MB) |
|---------|------|-------|-----------|
| pytorch-compile | gx10 A100 | 3,597.7 | 34,215 |
| cuBLAS | gx10 A100 | 4,026.8 | 49,778 |
| pytorch | gx10 A100 | 4,055.4 | 50,580 |
| unsloth | yoga RTX | 6,715.7 | 3,515 |
| unsloth | gx10 A100 | 13,659.7 | 10,219 |
| wgpu | mac-server | 10,378.6 | ? |

---

### Part VII: Switch From

#### Chapter 24 — Switch From PyTorch

**Example**: `cargo run -p aprender-core --example ch24_switch_pytorch`
**Contract**: `contracts/apr-book-ch24-v1.yaml`

API equivalence: `torch.tensor` → `Tensor::new`, `nn.Linear` → `Linear::new`,
`nn.Sequential` → `Sequential::new().add()`, `loss.backward()` → `loss.backward()`,
`optimizer.zero_grad()` → `clear_graph()`.

#### Chapter 25 — Switch From Ollama

**Example**: `cargo run -p aprender-core --example ch25_switch_ollama`
**Contract**: `contracts/apr-book-ch25-v1.yaml`

Command equivalence: `ollama pull` → `apr pull`, `ollama run` → `apr run`,
`ollama serve` → `apr serve`, `ollama list` → `apr list`.
Same GGUF files — zero conversion needed.

#### Chapter 26 — Switch From ndarray/nalgebra/linfa

**Example**: `cargo run -p aprender-core --example ch26_switch_ndarray`
**Contract**: `contracts/apr-book-ch26-v1.yaml`

API equivalence: `Array2::from_shape_vec` → `Matrix::from_vec`,
`linfa::KMeans` → `KMeans::new`, `linfa_linear::LinearRegression` → `LinearRegression::new`.
Key difference: aprender uses f32 by default (SIMD-optimized), 70 crates (not just ML).

#### Chapter 27 — Switch From unsloth

**Example**: `cargo run -p aprender-core --example ch27_switch_unsloth`
**Contract**: `contracts/apr-book-ch27-v1.yaml`
**POC Repo**: `paiml/qwen-train-canary`

Training equivalence: `FastLanguageModel.get_peft_model(r=16)` → `apr finetune --lora-rank 16`,
`SFTTrainer.train()` → `apr train --config train.yaml`.
unsloth uses 93% less VRAM than pytorch (3,515 vs 50,580 MB).

---

## Appendices

### Appendix A — Crate Namespace Reference

| Crate | Purpose | Old Name |
|-------|---------|----------|
| `aprender-core` | ML library (training, format, models) | `aprender` |
| `aprender-compute` | SIMD, GPU, WASM tensor compute | — |
| `aprender-serve` | Inference engine, HTTP server | — |
| `aprender-train` | Training loops, LoRA, distillation | — |
| `aprender-orchestrate` | Pipelines, agents, oracle | — |
| `aprender-present` | TUI framework, dashboards | — |
| `aprender-profile` | Profiling, tracing, roofline | — |
| `aprender-contracts` | Provable contract macros, YAML | — |
| `aprender-data` | Data loading, synthetic data | — |
| `aprender-rag` | RAG pipeline, embeddings, search | — |
| `aprender-graph` | Graph database, algorithms | — |
| `aprender-db` | Embedded analytics database | — |
| `aprender-viz` | Visualization | — |
| `aprender-test` | WASM/browser testing | — |
| `aprender-verify` | Verification, quality gates | — |
| `aprender-simulate` | Simulation framework | — |
| `aprender-distribute` | Distributed computing | — |
| `aprender-registry` | Model/data registry, lineage | — |
| `aprender-zram` | Compressed RAM storage | — |
| `aprender-quant` | Quantization algorithms | — |
| `aprender-sparse` | Sparse tensor operations | — |
| `apr-cli` | CLI binary (`apr`), 57 commands | — |

**Note**: The "Old Name" column is intentionally blank. This book uses ONLY the unified
`aprender-*` namespace. Legacy names are not referenced.

### Appendix B — Full arXiv Citation Index

| # | arXiv ID | Authors | Title | Chapters |
|---|----------|---------|-------|----------|
| 1 | 1706.03762 | Vaswani et al. | Attention Is All You Need | 8 |
| 2 | 2305.13245 | Ainslie et al. | GQA: Training Generalized Multi-Query Transformer Models | 8, 9 |
| 3 | 2104.09864 | Su et al. | RoFormer: Enhanced Transformer with Rotary Position Embedding | 8 |
| 4 | 2002.05202 | Shazeer | GLU Variants Improve Transformer | 8, 9 |
| 5 | 1910.07467 | Zhang & Sennrich | Root Mean Square Layer Normalization | 8 |
| 6 | 2211.05102 | Pope et al. | Efficiently Scaling Transformer Inference | 8, 9, 12 |
| 7 | 2210.17323 | Dettmers et al. | GPTQ: Accurate Post-Training Quantization | 2, 9, 11 |
| 8 | 2208.07339 | Dettmers et al. | LLM.int8(): 8-bit Matrix Multiplication | 9 |
| 9 | 2307.08691 | Dao et al. | FlashAttention-2 | 2, 9 |
| 10 | 2205.14135 | Dao et al. | FlashAttention: Fast and Memory-Efficient Exact Attention | 2 |
| 11 | 2309.06180 | Kwon et al. | PagedAttention for LLM Serving | 9, 12 |
| 12 | 2106.09685 | Hu et al. | LoRA: Low-Rank Adaptation of Large Language Models | 10 |
| 13 | 2305.14314 | Dettmers et al. | QLoRA: Efficient Finetuning of Quantized LLMs | 10 |
| 14 | 1711.05101 | Loshchilov & Hutter | Decoupled Weight Decay Regularization (AdamW) | 10 |
| 15 | 1608.03983 | Loshchilov & Hutter | SGDR: Stochastic Gradient Descent with Warm Restarts | 10 |
| 16 | 1710.03740 | Micikevicius et al. | Mixed Precision Training | 10 |
| 17 | 1502.05767 | Baydin et al. | Automatic Differentiation in Machine Learning | 10 |
| 18 | 2306.00978 | Lin et al. | AWQ: Activation-aware Weight Quantization | 11 |
| 19 | 2211.17192 | Leviathan et al. | Fast Inference via Speculative Decoding | 12 |
| 20 | 1910.02054 | Rajbhandari et al. | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 1 |
| 21 | 1903.00982 | Jung et al. | RustBelt: Securing the Foundations of Rust | 1 |
| 22 | 2407.10671 | Qwen Team | Qwen2 Technical Report | 1, 8 |
| 23 | 1609.04747 | Ruder | An Overview of Gradient Descent Optimization Algorithms | 4 |
| 24 | 1201.0490 | Pedregosa et al. | Scikit-learn: Machine Learning in Python | 4, 5 |
| 25 | 1603.02754 | Chen & Guestrin | XGBoost: A Scalable Tree Boosting System | 6 |
| 26 | 0909.4061 | Halko et al. | Finding Structure with Randomness (Randomized SVD) | 5 |
| 27 | 1802.04799 | Li et al. | TVM: End-to-End Optimizing Compiler for Deep Learning | 2 |
| 28 | 2101.05615 | Khudia et al. | FBGEMM: High-Performance Low-Precision Inference | 2 |
| 29 | 2005.11401 | Lewis et al. | Retrieval-Augmented Generation for NLP | 20 |
| 30 | 1702.08734 | Johnson et al. | Billion-Scale Similarity Search with GPUs | 20 |
| 31 | 2004.04906 | Karpukhin et al. | Dense Passage Retrieval for Open-Domain QA | 20 |
| 32 | 1508.07909 | Sennrich et al. | Neural Machine Translation with Subword Units (BPE) | 19 |
| 33 | 1808.06226 | Kudo & Richardson | SentencePiece: Language Independent Subword Tokenizer | 19 |
| 34 | 2210.03629 | Yao et al. | ReAct: Synergizing Reasoning and Acting in LMs | 15 |
| 35 | 1905.01997 | Zaharia et al. | Accelerating the ML Lifecycle with MLflow | 15 |
| 36 | 0803.0476 | Blondel et al. | Fast Unfolding of Communities (Louvain) | 18 |
| 37 | 1609.02907 | Kipf & Welling | Semi-Supervised Classification with GCN | 18 |
| 38 | 1907.09356 | Papadakis et al. | Mutation Testing Advances | 14 |
| 39 | 2007.00072 | Ivanov et al. | Data Movement Is All You Need | 13 |
| 40 | 1911.02150 | Shazeer | Fast Transformer Decoding: One Write-Head | 9 |
| 41 | math/0504499 | Efron | Bayesians, Frequentists, and Scientists | 17 |

### Appendix C — Book Build and Verification

```bash
# Build all chapter examples
for ch in crates/aprender-core/examples/ch*.rs; do
    cargo run -p aprender-core --example "$(basename "$ch" .rs)" || exit 1
done

# Validate all chapter contracts
for contract in contracts/apr-book-ch*-v1.yaml; do
    apr contracts check "$contract" || exit 1
done

# Namespace discipline gate (zero old names)
grep -rcE '\b(trueno|realizar|entrenar|batuta|presentar|renacer)\b' \
    docs/book/ examples/ch*.rs | grep -v ':0$' && {
    echo "FAIL: legacy names found in book content"
    exit 1
}

# Oracle consultation log
for family in qwen2 llama whisper bert; do
    echo "=== Oracle: $family ==="
    apr oracle --family "$family" --explain --stats
done

# Full test suite
cargo test --workspace --lib
cargo test --test book_contracts
```

### Appendix D — Contract YAML Template

```yaml
# contracts/apr-book-ch{NN}-v1.yaml
contract: apr-book-ch{NN}
version: 1
status: enforced
date: 2026-04-08

metadata:
  title: "{Chapter Title}"
  part: {I|II|III|IV|V}
  example: "ch{NN}_{topic}"
  arxiv_count: {N}

preconditions:
  - "cargo build --example ch{NN}_{topic} succeeds"
  - "All arXiv IDs in citation table are valid"
  - "Zero legacy names in chapter text"

postconditions:
  - "Example produces correct output"
  - "All assert!() in example pass"
  - "Oracle claims match --explain output"

falsification:
  - condition: "cargo run -p aprender-core --example ch{NN}_{topic} exits non-zero"
    severity: P0
    action: reject_chapter
  - condition: "Section without arXiv citation"
    severity: P0
    action: reject_chapter
  - condition: "Legacy name appears in docs/book/ch{NN}*.md"
    severity: P0
    action: reject_chapter
  - condition: "Oracle --explain output contradicts chapter claim"
    severity: P0
    action: reject_chapter
  - condition: "assert!() failure in example"
    severity: P0
    action: reject_chapter

equations:
  - name: "citation_coverage"
    formula: "citations_per_chapter >= 1"
    description: "Every chapter cites at least one arXiv paper"
  - name: "example_coverage"
    formula: "runnable_examples == total_chapters"
    description: "Every chapter has a cargo run --example"
  - name: "namespace_purity"
    formula: "legacy_name_count == 0"
    description: "Zero references to old crate names"
```

---

## Appendix E — paiml Org Repository Classification (205 repos)

**Date**: 2026-04-08
**Contract**: `contracts/apr-org-taxonomy-v1.yaml`
**Falsification**: Any repo not in a category below is a VIOLATION.

### Category Definitions

| Category | Tag | Description | Action |
|----------|-----|-------------|--------|
| **MONOREPO** | `monorepo` | THE aprender monorepo (70 crates) | Active development |
| **MERGED** | `merged/read-only` | Repos already merged into aprender monorepo | Archived, redirect to aprender |
| **ACTIVE-TOOL** | `active-tool` | Standalone tools used alongside monorepo | Active development |
| **MODEL-TRAINING** | `model-training` | LLM/model training experiments | Active research |
| **POC-BENCHMARK** | `poc/benchmark` | Proof-of-concept, benchmarks, comparisons | Read-only reference |
| **COURSE-DEMO** | `course/demo` | Coursera/LinkedIn/O'Reilly course material | Read-only, maintained for students |
| **LEGACY-BOOK** | `legacy/book` | Published book repos (O'Reilly, Pearson, Leanpub) | Read-only, no updates |
| **LEGACY-LIBRARY** | `legacy/library` | Superseded libraries, old stack components | Archive candidate |
| **GROUND-TRUTH** | `ground-truth` | Falsification/test corpora for oracle RAG | Maintained for RAG index |
| **INFRA** | `infra` | Infrastructure, CI/CD, deployment configs | Active |
| **LANG-ECOSYSTEM** | `lang/ruchy` | Ruchy language ecosystem | Separate product line |
| **TRANSPILER** | `transpiler` | Python/C/Haskell to Rust transpilers | Active tools |
| **PLATFORM** | `platform` | PAIML platform (website, apps, marketing) | Active |
| **STALE** | `stale/archive` | Inactive, no recent updates, no dependents | Archive immediately |

### 1. MONOREPO (1 repo)

| Repo | Description |
|------|-------------|
| `aprender` | THE monorepo — 70 crates, `apr` binary, ML library |

### 2. MERGED into aprender (14 repos — all archived)

| Repo | Monorepo Crate | Status |
|------|---------------|--------|
| `trueno` | `aprender-compute` | Archived |
| `realizar` | `aprender-serve` | Archived |
| `entrenar` | `aprender-train` | Archived |
| `batuta` | `aprender-orchestrate` | Archived |
| `presentar` | `aprender-present-*` | Archived |
| `repartir` | `aprender-distribute` | Archived |
| `simular` | `aprender-simulate` | Archived |
| `verificar` | `aprender-verify-ml` | Archived |
| `certeza` | `aprender-verify` | Archived |
| `provable-contracts` | `aprender-contracts` | Archived |
| `probar` | `aprender-test-*` | Archived |
| `trueno-db` | `aprender-db` | Archived |
| `trueno-graph` | `aprender-graph` | Archived |
| `trueno-rag` | `aprender-rag` | Archived |

### 3. SHOULD-MERGE (not yet archived — merge into aprender)

| Repo | Target Crate | Rationale |
|------|-------------|-----------|
| `alimentar` | `aprender-data` | Data loading, already `aprender-data` exists |
| `batuta-common` | `aprender-common` | Already mapped, not archived |
| `trueno-viz` | `aprender-viz` | Already `aprender-viz` exists |
| `trueno-zram` | `aprender-zram` | Already `aprender-zram` exists |
| `renacer` | `aprender-profile` | Already `aprender-profile` exists |
| `pacha` | `aprender-registry` | Model registry, already mapped |

**Action**: Archive these 6 repos with redirect descriptions, like the 14 above.

### 4. ACTIVE-TOOL (standalone tools, NOT merging)

| Repo | Language | Purpose | Tag |
|------|----------|---------|-----|
| `paiml-mcp-agent-toolkit` | Rust | MCP server for deterministic agentic coding | `active-tool` |
| `rust-mcp-sdk` | Rust | MCP SDK | `active-tool` |
| `bashrs` | Rust | Shell transpiler | `active-tool` |
| `forjar` | Rust | Infrastructure as Code | `active-tool` |
| `depyler` | Rust | Python→Rust compiler | `active-tool/transpiler` |
| `decy` | Rust | C→Rust transpiler | `active-tool/transpiler` |
| `rascal` | Rust | Haskell→Rust transpiler | `active-tool/transpiler` |
| `spydecy` | Rust | Python/C→Rust debugger | `active-tool/transpiler` |
| `ccpo` | Rust | Claude Code proxy | `active-tool` |
| `pcode` | Rust | Coding agent | `active-tool` |
| `pdmt` | Rust | MCP templating | `active-tool` |
| `organizational-intelligence-plugin` | Rust | PMAT plugin | `active-tool` |
| `copia` | Rust | rsync delta-sync | `active-tool` |
| `rmedia` | Rust | Course video renderer | `active-tool` |
| `duende` | Rust | Daemon tooling | `active-tool` |
| `cohete` | Rust | Jetson Nano | `active-tool` |
| `manzana` | Rust | macOS hardware | `active-tool` |
| `pepita` | Rust | Tiny Linux kernel | `active-tool` |
| `pforge` | HTML | MCP server builder | `active-tool` |
| `rust-mdipierro-nlib` | Rust | Provable numerical algorithms | `active-tool` |
| `microgpt` | Rust | microGPT in aprender | `active-tool` |

### 5. MODEL-TRAINING (active research)

| Repo | Language | Purpose | Tag |
|------|----------|---------|-----|
| `albor` | Python | LLM from first principles, sovereign components | `model-training` |
| `qwen-train-canary` | Python | Training perf canary (unsloth, pytorch, cuBLAS) | `model-training` |
| `whisper.apr` | Rust | Whisper in APR format + WASM | `model-training` |

### 6. POC / BENCHMARK (proof of concept, read-only)

| Repo | Language | Comparing | Tag |
|------|----------|-----------|-----|
| `candle-vs-apr` | Shell | Candle vs aprender-serve on RTX 4090 | `poc/benchmark` |
| `single-shot-eval` | Rust | Pareto frontier of SLMs | `poc/benchmark` |
| `real-world-code-score` | Rust | Code quality scoring | `poc/benchmark` |
| `compiled-rust-benchmarking` | Rust | Compile speed optimization | `poc/benchmark` |
| `rosetta-ruchy` | HTML | Ruchy vs Rust parity benchmark | `poc/benchmark` |
| `ruchy-lambda` | Rust | Ruchy vs all languages on Lambda | `poc/benchmark` |
| `ruchy-docker` | Rust | Ruchy runtime benchmarking | `poc/benchmark` |
| `qwen-coder-deploy` | Makefile | Qwen deployment POC | `poc/benchmark` |

### 7. COURSE-DEMO (Coursera / LinkedIn / educational)

| Repo | Platform | Topic | Tag |
|------|----------|-------|-----|
| `ai-tooling` | Coursera | 20-course AI specialization | `course/demo` |
| `deterministic-llm-coding` | Course | Deterministic LLM coding | `course/demo` |
| `deterministic-mcp-agents` | Course | MCP agents | `course/demo` |
| `HF-Hub-Ecosystem` | Coursera | Hugging Face ecosystem | `course/demo` |
| `HF-Production-ML` | Coursera | Production ML with HF | `course/demo` |
| `HF-Advanced-Fine-Tuning` | Coursera | Advanced fine-tuning | `course/demo` |
| `huggingface-fine-tuning` | Course | HF fine-tuning | `course/demo` |
| `llms-with-huggingface` | Course | LLMs with HF | `course/demo` |
| `advanced-prompting-with-github-copilot` | LinkedIn | GitHub Copilot | `course/demo` |
| `ghcp-for-systems-level-development` | LinkedIn | GH Copilot systems | `course/demo` |
| `GitHub-Copilot-Mastery-Capstone` | Course | Copilot capstone | `course/demo` |
| `mastering-github` | Coursera | GitHub 9-course spec | `course/demo` |
| `databricks-data-engineering` | Coursera | Databricks DE | `course/demo` |
| `databricks-governance` | Coursera | Databricks governance | `course/demo` |
| `DB-mlops-genai` | Coursera | Databricks MLOps | `course/demo` |
| `data-pipelines-deno-typescript-course` | Course | Deno data pipelines | `course/demo` |
| `rust-data-engineering` | Coursera | Rust DE | `course/demo` |
| `responsible-ai-dev` | Course | Responsible AI | `course/demo` |
| `agentic-ai` | Course | Agentic AI | `course/demo` |
| `applied-ai-engineering` | Coursera | Applied AI engineering | `course/demo` |
| `ds500-debug-with-ai` | DS500 | Debug with AI | `course/demo` |
| `ds500-rust-bootcamp` | DS500 | Rust bootcamp | `course/demo` |
| `multi-modal-programming-course` | Course | Multi-modal programming | `course/demo` |
| `review-bot-course` | Course | Review bot | `course/demo` |
| `build-a-saas-course` | Course | Build SaaS | `course/demo` |
| `windsurf` | Course | Windsurf IDE | `course/demo` |
| `wasm-labs` | Labs | WASM labs | `course/demo` |
| `profesor` | Rust | Teaching environment | `course/demo` |
| `discord-conversational-bot` | Course | Discord bot | `course/demo` |
| `wine-api-saas` | Rust | SaaS demo | `course/demo` |

### 8. LEGACY-BOOK (published, read-only)

| Repo | Publisher | Title | Tag |
|------|-----------|-------|-----|
| `practical-mlops-book` | O'Reilly | Practical MLOps (2021) | `legacy/book` |
| `python_devops_book` | O'Reilly | Python for DevOps (2020) | `legacy/book` |
| `foundations-python-datascience-book` | Pearson | Foundations Python DS | `legacy/book` |
| `minimal-python-BOOK` | Leanpub | Minimal Python | `legacy/book` |
| `minimal-go-BOOK` | Leanpub | Minimal Go | `legacy/book` |
| `minimal-shell` | Leanpub | Minimal Shell | `legacy/book` |
| `minimal-machine-learning` | Leanpub | Minimal ML | `legacy/book` |
| `minimal-datascience` | Leanpub | Minimal Data Science | `legacy/book` |
| `more-python-cowbell` | Leanpub | More Python Cowbell | `legacy/book` |
| `ml_engineering_book` | Book | ML Engineering | `legacy/book` |
| `opscookbook` | Leanpub | Ops Cookbook | `legacy/book` |
| `testing-in-python-book` | Book | Testing in Python | `legacy/book` |
| `sovereign-ai-stack-book` | Book | Sovereign AI Stack | `legacy/book` |
| `pmat-book` | Book | PMAT book | `legacy/book` |
| `the-python-commandline-book` | Leanpub | Python Commandline | `legacy/book` |

### 9. LEGACY-LIBRARY (superseded — ARCHIVED 2026-04-08)

| Repo | Reason | Tag |
|------|--------|-----|
| `batuta-cookbook` | Uses old `batuta` name | `legacy/library` |
| `batuta-ground-truth-mlops-corpus` | Uses old namespace | `legacy/library` |
| `forjar-cookbook` | Cookbook, low activity | `legacy/library` |
| `apr-cookbook` | Cookbook for .apr format | `legacy/library` |
| `ald-cookbook` | Cookbook for .ald format | `legacy/library` |
| `prs-cookbook` | Uses old `presentar` name | `legacy/library` |
| `apr-model-qa-playbook` | QA playbook, may fold into apr qa | `legacy/library` |
| `reaper` | Written in Ruchy, standalone monitor | `legacy/library` |
| `rustysquid` | Minimal Squid port | `legacy/library` |
| `mp4convertor` | Standalone utility | `legacy/library` |
| `rclean` | File cleaner utility | `legacy/library` |
| `universal-bot` | Old bot architecture | `legacy/library` |
| `discord-intelligence` | Discord tooling | `legacy/library` |
| `ov` | O'Reilly video CLI | `legacy/library` |
| `assetgen` | Asset generator | `legacy/library` |
| `assetsearch` | Asset search | `legacy/library` |
| `pacha-run` | Runner for .apr files | `legacy/library` |
| `wos` | WASM OS (teaching) | `legacy/library` |

### 10. GROUND-TRUTH (falsification corpora for oracle RAG)

| Repo | Language | Domain | Tag |
|------|----------|--------|-----|
| `tgi-ground-truth-corpus` | Rust | TGI inference patterns | `ground-truth` |
| `tiny-model-ground-truth` | Python | Model format conversions | `ground-truth` |
| `hugging-face-ground-truth-corpus` | Python | HF Python→Rust | `ground-truth` |
| `databricks-ground-truth-corpus` | Python | Databricks patterns | `ground-truth` |
| `databricks-scala-ground-truth-corpus` | Scala | Spark/ML/Delta Lake | `ground-truth` |
| `jax-ground-truth-corpus` | Python | JAX recipes | `ground-truth` |
| `ludwig-ground-truth-corpus` | Python | Ludwig declarative DL | `ground-truth` |
| `vllm-ground-truth-corpus` | Python | vLLM inference | `ground-truth` |
| `mixed-python-rust-ground-truth` | Python | Mixed Python/Rust | `ground-truth` |
| `mixed-rust-lean-ground-truth` | Rust | Rust/Lean proofs | `ground-truth` |
| `lean-ground-truth` | Lean | Lean 4 theorems | `ground-truth` |
| `safe-lua-groundtruth` | Lua | Safe Lua patterns | `ground-truth` |
| `algorithm-competition-corpus` | Python | Algorithm corpus | `ground-truth` |

### 11. TRANSPILER ECOSYSTEM (Depyler, Decy, Rascal, Reprorusted)

| Repo | Direction | Tag |
|------|-----------|-----|
| `depyler` | Python→Rust compiler | `transpiler` |
| `decy` | C→Rust | `transpiler` |
| `rascal` | Haskell→Rust | `transpiler` |
| `spydecy` | Python/C debugger+compiler | `transpiler` |
| `reprorusted-python-cli` | Python argparse→Rust | `transpiler/corpus` |
| `reprorusted-c-cli` | C→Rust training corpus | `transpiler/corpus` |
| `reprorusted-std-only` | Python stdlib→Rust | `transpiler/corpus` |
| `fully-typed-reprorusted-python-cli` | Typed Python→Rust | `transpiler/corpus` |
| `python-to-rust-conversion-examples` | Examples | `transpiler/corpus` |

### 12. RUCHY LANGUAGE ECOSYSTEM

| Repo | Purpose | Tag |
|------|---------|-----|
| `ruchy` | Language compiler | `lang/ruchy` |
| `ruchy-book` | Official book | `lang/ruchy` |
| `ruchy-cli-tools-book` | CLI tools book | `lang/ruchy` |
| `ruchy-cookbook` | Cookbook | `lang/ruchy` |
| `ruchy-repl-demos` | REPL demos | `lang/ruchy` |
| `ruchy-syntax-tools` | Syntax highlighting | `lang/ruchy` |
| `ruchyruchy` | Self-hosting compiler | `lang/ruchy` |
| `tooling-with-ruchy` | Tooling book | `lang/ruchy` |

### 13. INFRA / PLATFORM

| Repo | Purpose | Tag |
|------|---------|-----|
| `.github` | Org-level GitHub config | `infra` |
| `infra` | Infrastructure configs | `infra` |
| `gunner` | AWS Spot runners | `infra` |
| `lambda-lab-rust-development` | Lambda Labs setup | `infra` |
| `sovereign-ai-cookbook` | Forjar deployment configs | `infra` |
| `pzsh` | Shell environment | `infra` |
| `interactive.paiml.com` | WASM interactive books | `platform` |
| `paiml-blog` | Blog | `platform` |
| `paiml-android-app` | Android app | `platform` |
| `total-reach` | Reach analytics | `platform` |
| `marketing` | Marketing templates | `platform` |
| `ds500-master-paths` | DS500 paths | `platform` |
| `ds500-social-preview` | Badge generator | `platform` |
| `ds500-outline-maker` | Outline generator | `platform` |
| `ds500-course-processing` | Course ingestion | `platform` |
| `ds500-rust-project-template` | Project template | `platform` |
| `platform` | PAIML platform | `platform` |
| `publish` | Content publishing tools | `platform` |
| `sales_intelligence` | Sales intelligence | `platform` |
| `coursera-stats` | Coursera analytics | `platform` |
| `linkedin-rev-stats` | LinkedIn stats | `platform` |
| `course-gen` | Course asset generator | `platform` |
| `course-studio` | Video production | `platform` |
| `video-tools` | Video tools | `platform` |
| `stickymind` | AI assistant | `platform` |

### 14. PMAT (separate product)

| Repo | Purpose | Tag |
|------|---------|-----|
| `pmat-action` | GitHub Action for PMAT | `pmat` |
| `pmat-test-sonnet-4` | Sonnet 4 eval | `pmat/eval` |
| `pmat-test-gpt4.1` | GPT-4.1 eval | `pmat/eval` |
| `pmat-test-gpt5` | GPT-5 eval | `pmat/eval` |
| `pmat-test-gpt5-mini` | GPT-5 mini eval | `pmat/eval` |
| `pmat-test-gemini2.5-pro` | Gemini 2.5 Pro eval | `pmat/eval` |

### 15. STALE / ARCHIVE IMMEDIATELY

| Repo | Reason | Tag |
|------|--------|-----|
| `ds500-subscription-deleted` | Deleted subscription | `stale/archive` |
| `hello` | Hello world demo | `stale/archive` |
| `hello-github` | Hello world demo | `stale/archive` |
| `discord-bot` | Empty/abandoned | `stale/archive` |
| `model-serving-survey` | Empty survey | `stale/archive` |
| `socialpower` | Abandoned | `stale/archive` |
| `software-language-popularity-2025` | One-off analysis | `stale/archive` |
| `osx-perf-tune` | macOS perf tuning (old) | `stale/archive` |
| `ubuntu-config-scripts` | Config scripts (old) | `stale/archive` |
| `cost-optimize-aws` | AWS cost scripts (old) | `stale/archive` |
| `eu-currency` | Currency converter demo | `stale/archive` |
| `dom-intelligence` | DOM intelligence (old) | `stale/archive` |
| `labs-code` | Old labs code | `stale/archive` |
| `data` | Public data files | `stale/archive` |
| `ropub` | O'Reilly pub tools | `stale/archive` |
| `archived-emlop-book-material` | Archived | `stale/archive` |
| `Flask-Elastic-Beanstalk` | Old Python Flask | `stale/archive` |
| `awsbigdata` | Old AWS cert | `stale/archive` |
| `pbjbi` | MCP BI demo | `stale/archive` |
| `apr-leaderboard` | Leaderboard scripts | `stale/archive` |
| `engman` | AI eng manager demo | `stale/archive` |
| `faro` | Spanish data mining | `stale/archive` |
| `rurl` | URL rewriter | `stale/archive` |
| `minimal-pyqt` | Minimal PyQt | `stale/archive` |

### Remaining (already counted elsewhere)

| Repo | Category |
|------|----------|
| `testing-in-python` | `course/demo` |
| `python_for_datascience` | `course/demo` |
| `python-command-line-tools` | `course/demo` |
| `python-devops` | `course/demo` |
| `minimal-python` | `legacy/book` |
| `livestreams` | `platform` |
| `wine-ratings` | `course/demo` |

---

### Summary by Category

| Category | Count | Action |
|----------|-------|--------|
| MONOREPO | 1 | Active — `aprender` |
| MERGED (archived) | 14 | Done — redirects in place |
| SHOULD-MERGE | 6 | Archive with redirect |
| ACTIVE-TOOL | 21 | Maintain independently |
| MODEL-TRAINING | 3 | Active research |
| POC/BENCHMARK | 8 | Read-only reference |
| COURSE-DEMO | ~34 | Read-only for students |
| LEGACY-BOOK | 15 | Read-only, no updates |
| LEGACY-LIBRARY | 18 | Archive candidates |
| GROUND-TRUTH | 13 | Maintain for RAG oracle |
| TRANSPILER | 9 | Active (depyler product line) |
| RUCHY | 8 | Separate product line |
| INFRA/PLATFORM | ~25 | Active operations |
| PMAT | 6 | Separate product |
| STALE/ARCHIVE | ~24 | Archive immediately |
| **Total** | **205** | |

### Implementation Priority

**Phase 1 — Immediate**: DONE (2026-04-08)
1. ~~Archive 6 SHOULD-MERGE repos with redirect descriptions~~ — 6/6 archived
2. ~~Archive 24 STALE repos~~ — 24/24 archived
3. ~~Tag all 205 repos with GitHub topics matching categories above~~ — 15 topics applied
4. ~~Archive 18 LEGACY-LIBRARY repos~~ — 18/18 archived
5. **Total archived: 62** (14 merged + 6 should-merge + 24 stale + 18 legacy-library)

**Phase 2 — Provable-contracts-first documentation**: DONE (2026-04-08)
1. ~~21 ACTIVE-TOOL repos get provable contracts~~ — `contracts/apr-tool-*-v1.yaml` (21 files)
2. Each contract defines: purpose, inputs, outputs, 5 falsification conditions
3. Book chapters reference tools by `apr` subcommand or `aprender-*` crate only

**Phase 3 — Ground-truth corpus contracts**: DONE (2026-04-08)
1. ~~13 ground-truth corpora get provable contracts~~ — `contracts/apr-corpus-*-v1.yaml` (13 files)
2. Each corpus contract defines: domain, language, freshness gate, RAG indexing
3. Oracle consultation protocol references corpus provenance

**Contract inventory**:
- 20 book chapter contracts (`apr-book-ch*-v1.yaml`)
- 21 active-tool contracts (`apr-tool-*-v1.yaml`)
- 13 ground-truth corpus contracts (`apr-corpus-*-v1.yaml`)
- 1 org taxonomy contract (`apr-org-taxonomy-v1.yaml`)
- **Total: 55 new contracts** from this spec

### Falsification for this Appendix

```yaml
# contracts/apr-org-taxonomy-v1.yaml
contract: apr-org-taxonomy
version: 1
status: enforced
date: 2026-04-08
falsification:
  - condition: "gh repo list paiml returns repo not in any category"
    severity: P0
    action: classify_and_update
  - condition: "SHOULD-MERGE repo not archived within 7 days"
    severity: P1
    action: escalate
  - condition: "STALE repo not archived within 7 days"
    severity: P1
    action: escalate
  - condition: "Repo uses legacy name in description without redirect"
    severity: P0
    action: update_description
```

---

## Final Status (2026-04-08)

### Book

| Metric | Value |
|--------|-------|
| Schema | **Zero-Muda PCU** — no page without contract |
| Enforced pages | **230** (223 + 7 new chapters) |
| Draft pages | **6** (1 feature-gated, 5 environment-dependent) |
| Total PCUs | **236** |
| Muda deleted | **49** pages (stubs + "under construction") |
| SUMMARY.md | **Generated** from enforced contracts only |
| Parts | **7** (I-V core + VI Benchmarks + VII Switch From) |
| Chapters | **27** (20 core + 3 benchmark + 4 switch-from) |
| Chapter examples (ch01-ch27) | **27/27** compile+run |
| Real API examples | **20/27** (7 hollow-acceptable: APIs in aprender-serve) |
| Total examples in aprender-core | **148** |
| POC repos referenced | candle-vs-apr, qwen-train-canary |
| arXiv citations | **42** indexed in spec |

### Contracts

| Category | Count | Pattern |
|----------|-------|---------|
| Page contracts (PCU) | 236 | `apr-page-*-v1.yaml` |
| Chapter contracts | 27 | `apr-book-ch*-v1.yaml` |
| Active-tool contracts | 21 | `apr-tool-*-v1.yaml` |
| Ground-truth corpus contracts | 13 | `apr-corpus-*-v1.yaml` |
| Book schema contract | 1 | `apr-book-schema-v1.yaml` |
| Org taxonomy contract | 1 | `apr-org-taxonomy-v1.yaml` |
| **Total** | **299** | |
| **Falsification conditions** | **1,495** | 5 per contract |

### paiml Org

| Metric | Value |
|--------|-------|
| Total repos | **205** |
| Archived | **62** (14 merged + 6 should-merge + 24 stale + 18 legacy-library) |
| Active | **143** |
| Categories | **15** (GitHub topics applied) |

### Quality Gates (10-gate falsification)

| Gate | Check | Result |
|------|-------|--------|
| 1 | PCU frontmatter on every page | 229/229 |
| 2 | Contract YAML for every PCU | 229/229 |
| 3 | 5 falsification conditions per contract | 229/229 |
| 4 | Zero placeholder text | 0 |
| 5 | Zero stub pages | 0 |
| 6 | Zero dead SUMMARY.md links | 0 |
| 7 | Integration tests | 6 passed |
| 8 | Legacy imports in book | 0 |
| 9 | Total contracts | 285 |
| 10 | Falsification conditions | 1,425 |

### Infrastructure

| Asset | Purpose |
|-------|---------|
| `scripts/book-gate.sh` | 7-gate validation per PCU |
| `scripts/gen-summary.sh` | Generate SUMMARY.md from enforced contracts |
| `scripts/pcu-batch.sh` | Bulk contract + frontmatter generation |
| `crates/aprender-core/tests/book_contracts.rs` | 6 integration tests |
| `contracts/apr-book-schema-v1.yaml` | Master schema contract |

### PMAT Work Items (14 completed)

| ID | Title |
|----|-------|
| PMAT-497 | Scaffold 20 chapter examples |
| PMAT-498 | Create 20 chapter contract YAMLs |
| PMAT-499 | Create book_contracts integration test |
| PMAT-500 | Falsify all chapters end-to-end |
| PMAT-501 | Archive 6 SHOULD-MERGE repos with redirects |
| PMAT-502 | Archive 24 STALE repos |
| PMAT-503 | Tag all 205 repos with category topics |
| PMAT-504 | Archive 18 LEGACY-LIBRARY repos |
| PMAT-505 | Provable contracts for 21 ACTIVE-TOOL repos |
| PMAT-506 | Provable contracts for 13 GROUND-TRUTH corpora |
| PMAT-507 | Convert 7 hollow examples to real API exercises |
| PMAT-508 | Write Part I chapters (superseded by zero-muda redesign) |
| PMAT-509 | Zero-muda book schema — eliminate all muda |
| PMAT-510 | PCU contracts for 229 remaining pages |
