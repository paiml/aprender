# CODEX: Multi-Tech Python-to-Rust Training Data Specification

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Draft
**Project:** verificar + aprender integration

---

## Executive Summary

**CODEX** (COde Data EXtraction) is a unified pipeline combining **aprender** (AutoML) with **verificar** (synthetic data factory) to generate high-quality training data for Python-to-Rust transpilation. The pipeline uses ML-driven filtering, adaptive generation, and active learning to maximize training data utility while minimizing verification oracle costs.

---

## Problem Statement

Current transpiler training approaches face key challenges:

1. **Verification Bottleneck**: Oracle execution (sandbox Python + Rust) is expensive (~100ms/sample)
2. **Low Signal-to-Noise**: Most generated programs are trivial or redundant
3. **Distribution Mismatch**: Uniform random sampling doesn't match real-world bug distribution
4. **Sparse Feedback**: Binary correctness labels waste gradient information

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CODEX PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  verificar  │    │   aprender  │    │  verificar  │    │   aprender  │  │
│  │  Generator  │───▶│  Filter     │───▶│  Oracle     │───▶│  Labeler    │  │
│  │             │    │  (Quality)  │    │             │    │  (Rich)     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│       │                   │                   │                   │         │
│       │                   │                   │                   │         │
│       ▼                   ▼                   ▼                   ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Candidate  │    │  High-Value │    │  (source,   │    │  Training   │  │
│  │  Programs   │    │  Subset     │    │  target,    │    │  Dataset    │  │
│  │             │    │             │    │  verdict)   │    │  .parquet   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    FEEDBACK LOOP (Active Learning)                    │   │
│  │  aprender::GradientBoosting predicts informative samples → Generator │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Integration

### 1. Quality-Gated Generation (Pre-Oracle Filter)

**Goal:** Reduce oracle calls by 10x via ML-based quality prediction.

```rust
use aprender::tree::RandomForestClassifier;
use verificar::generator::{Generator, SamplingStrategy};
use verificar::data::CodeFeatures;

/// Quality gate using aprender RandomForest
pub struct QualityGate {
    model: RandomForestClassifier,
    threshold: f32,
}

impl QualityGate {
    /// Train on historical (features, oracle_passed) pairs
    pub fn train(examples: &[(CodeFeatures, bool)]) -> Self {
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(100)
            .with_max_depth(Some(10));

        let (features, labels) = examples_to_matrix(examples);
        model.fit(&features, &labels).unwrap();

        Self { model, threshold: 0.7 }
    }

    /// Predict if sample is worth verifying
    pub fn should_verify(&self, features: &CodeFeatures) -> bool {
        let x = features_to_row(features);
        self.model.predict_proba(&x)[1] > self.threshold
    }
}
```

**Features for quality prediction:**
- `ast_depth`: Deeper = more interesting
- `num_operators`: Mathematical complexity
- `num_control_flow`: Branching logic
- `cyclomatic_complexity`: Path diversity
- `uses_edge_values`: Boundary conditions (0, -1, empty)
- `type_coercion_count`: Python→Rust type mapping challenges

### 2. Bug Prediction Model (Defect Likelihood)

**Goal:** Prioritize samples likely to reveal transpiler bugs.

```rust
use aprender::tree::GradientBoostingClassifier;
use verificar::ml::CommitFeatures;

/// Bug predictor trained on historical defect-fix commits
pub struct BugPredictor {
    model: GradientBoostingClassifier,
}

impl BugPredictor {
    /// Train on defect-fix commit features from 1,296 PAIML commits
    pub fn train(commits: &[CommitFeatures], labels: &[bool]) -> Self {
        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(200)
            .with_learning_rate(0.1)
            .with_max_depth(5);

        let features = commits_to_matrix(commits);
        model.fit(&features, &labels).unwrap();

        Self { model }
    }

    /// Predict probability of triggering a bug
    pub fn predict_bug_prob(&self, code: &str, features: &CodeFeatures) -> f32 {
        let x = extract_features(code, features);
        self.model.predict_proba(&x)[1]
    }
}
```

**Defect category allocation (from PAIML intelligence):**

| Priority | Category | Allocation | Training Weight |
|----------|----------|------------|-----------------|
| P0 | ASTTransform | 50% | 2.0x |
| P1 | OwnershipBorrow | 20% | 1.5x |
| P2 | StdlibMapping | 15% | 1.2x |
| P3 | LanguageSpecific | 15% | 1.0x |

### 3. Adaptive Generation (Active Learning)

**Goal:** Dynamically adjust sampling strategy based on oracle feedback.

```rust
use aprender::cluster::KMeans;
use verificar::generator::SamplingStrategy;

/// Active learner that prioritizes unexplored regions
pub struct ActiveSampler {
    /// Embedding model for code representation
    embedder: CodeEmbedder,
    /// K-means clusters of verified samples
    clusters: KMeans,
    /// Per-cluster success rates
    cluster_stats: Vec<ClusterStats>,
}

impl ActiveSampler {
    /// Thompson Sampling: explore uncertain clusters
    pub fn sample_strategy(&self) -> SamplingStrategy {
        let ucb_scores: Vec<f32> = self.cluster_stats
            .iter()
            .map(|s| s.mean_reward + 2.0 * s.uncertainty())
            .collect();

        let target_cluster = argmax(&ucb_scores);
        let centroid = self.clusters.centroids()[target_cluster].clone();

        SamplingStrategy::Targeted {
            feature_bias: centroid_to_features(&centroid),
            exploration_epsilon: 0.1,
        }
    }

    /// Update with oracle feedback
    pub fn update(&mut self, sample: &CodeSample, passed: bool) {
        let embedding = self.embedder.embed(&sample.source);
        let cluster_id = self.clusters.predict(&embedding);
        self.cluster_stats[cluster_id].update(passed);
    }
}
```

### 4. Rich Label Generation (Beyond Binary)

**Goal:** Extract maximum signal from each oracle invocation.

```rust
use aprender::linear_model::LinearRegression;
use verificar::oracle::ExecutionResult;

/// Multi-task labeler extracting rich supervision
pub struct RichLabeler;

impl RichLabeler {
    /// Generate rich labels from oracle execution
    pub fn label(
        source: &str,
        target: &str,
        source_result: &ExecutionResult,
        target_result: &ExecutionResult,
    ) -> RichLabels {
        RichLabels {
            // Binary correctness
            correct: source_result.output == target_result.output,

            // Semantic similarity (for soft labels)
            output_similarity: jaccard_similarity(
                &source_result.output,
                &target_result.output,
            ),

            // Performance ratio (for distillation)
            runtime_ratio: target_result.duration.as_secs_f32()
                / source_result.duration.as_secs_f32().max(0.001),

            // Error category (for multi-class)
            error_category: classify_error(source_result, target_result),

            // AST diff features (for localization)
            ast_diff: compute_ast_diff(source, target),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RichLabels {
    pub correct: bool,
    pub output_similarity: f32,
    pub runtime_ratio: f32,
    pub error_category: ErrorCategory,
    pub ast_diff: AstDiff,
}
```

### 5. Data Quality Scoring Pipeline

**Goal:** Rank training examples by informativeness.

```rust
use aprender::decomposition::PCA;
use aprender::metrics::silhouette_score;

/// Score training examples by quality
pub struct DataQualityScorer {
    pca: PCA,
    reference_embeddings: Matrix,
}

impl DataQualityScorer {
    /// Score a training example
    pub fn score(&self, example: &TrainingExample) -> QualityScore {
        let embedding = self.embed(example);

        QualityScore {
            // Novelty: distance from existing examples
            novelty: self.min_distance_to_reference(&embedding),

            // Diversity: contribution to overall variance
            diversity: self.variance_contribution(&embedding),

            // Difficulty: model uncertainty
            difficulty: self.predict_difficulty(example),

            // Coverage: AST node types covered
            coverage: self.ast_coverage(example),
        }
    }

    /// Filter to top-k most informative examples
    pub fn select_top_k(&self, examples: &[TrainingExample], k: usize) -> Vec<TrainingExample> {
        let mut scored: Vec<_> = examples
            .iter()
            .map(|e| (e.clone(), self.score(e).composite()))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).map(|(e, _)| e).collect()
    }
}
```

---

## Pipeline Execution

### Full Pipeline (batch mode)

```rust
use verificar::generator::Generator;
use verificar::oracle::SandboxOracle;
use aprender::preprocessing::StandardScaler;

pub async fn run_codex_pipeline(config: CodexConfig) -> Dataset {
    // 1. Initialize components
    let generator = Generator::new(Language::Python);
    let quality_gate = QualityGate::load(&config.quality_model_path)?;
    let bug_predictor = BugPredictor::load(&config.bug_model_path)?;
    let oracle = SandboxOracle::new(config.timeout);
    let labeler = RichLabeler;
    let active_sampler = ActiveSampler::new(config.n_clusters);

    let mut dataset = Vec::new();
    let mut verified_count = 0;

    // 2. Generate with adaptive sampling
    for batch in 0..config.n_batches {
        let strategy = active_sampler.sample_strategy();
        let candidates = generator.generate(strategy, config.batch_size);

        // 3. Quality gate (10x speedup)
        let high_quality: Vec<_> = candidates
            .into_iter()
            .filter(|c| quality_gate.should_verify(&c.features))
            .collect();

        // 4. Bug-priority sorting
        let mut prioritized = high_quality;
        prioritized.sort_by(|a, b| {
            bug_predictor.predict_bug_prob(&b.source, &b.features)
                .partial_cmp(&bug_predictor.predict_bug_prob(&a.source, &a.features))
                .unwrap()
        });

        // 5. Oracle verification (expensive)
        for candidate in prioritized.iter().take(config.oracle_budget) {
            let (source_result, target_result) = oracle.execute_pair(
                &candidate.source,
                &candidate.target,
            ).await?;

            // 6. Rich labeling
            let labels = labeler.label(
                &candidate.source,
                &candidate.target,
                &source_result,
                &target_result,
            );

            // 7. Update active learner
            active_sampler.update(&candidate, labels.correct);

            dataset.push(TrainingExample {
                source: candidate.source.clone(),
                target: candidate.target.clone(),
                labels,
            });

            verified_count += 1;
        }

        log::info!(
            "Batch {}: {} verified, {} total examples",
            batch, verified_count, dataset.len()
        );
    }

    // 8. Quality-based selection
    let scorer = DataQualityScorer::train(&dataset);
    let final_dataset = scorer.select_top_k(&dataset, config.final_size);

    // 9. Export to Parquet
    export_parquet(&final_dataset, &config.output_path)?;

    Ok(final_dataset)
}
```

### Configuration

```yaml
# codex-config.yaml
pipeline:
  n_batches: 1000
  batch_size: 100
  oracle_budget: 20  # 20% pass quality gate
  final_size: 50000

generator:
  language: python
  max_depth: 5
  strategies:
    - exhaustive: 0.2
    - coverage_guided: 0.5
    - boundary: 0.3

quality_gate:
  model_path: models/quality_rf.apr
  threshold: 0.7

bug_predictor:
  model_path: models/bug_gb.apr
  defect_weights:
    ast_transform: 2.0
    ownership_borrow: 1.5
    stdlib_mapping: 1.2
    language_specific: 1.0

active_learning:
  n_clusters: 50
  exploration_epsilon: 0.1
  ucb_coefficient: 2.0

labeling:
  rich_labels: true
  error_categories:
    - type_mismatch
    - ownership_violation
    - lifetime_error
    - panic_divergence
    - output_mismatch

output:
  format: parquet
  path: data/codex_python_rust.parquet
  columns:
    - source
    - target
    - correct
    - output_similarity
    - runtime_ratio
    - error_category
    - features
```

---

## Expected Outcomes

| Metric | Baseline | CODEX Target |
|--------|----------|--------------|
| Oracle calls per 1K examples | 1000 | 100 |
| Bug-revealing rate | 2% | 15% |
| Dataset diversity (silhouette) | 0.3 | 0.6 |
| Training convergence (epochs) | 50 | 20 |
| Final model accuracy | 85% | 92% |

---

## Implementation Roadmap

### Phase 1: Quality Gate (VER-050)
- [ ] Feature extraction pipeline
- [ ] RandomForest training on historical data
- [ ] Integration with verificar generator

### Phase 2: Bug Predictor (VER-051)
- [ ] Commit feature extraction from PAIML repos
- [ ] GradientBoosting model training
- [ ] Defect category weighting

### Phase 3: Active Learning (VER-052)
- [ ] Code embedding via TF-IDF + SVD
- [ ] K-means clustering
- [ ] Thompson Sampling integration

### Phase 4: Rich Labeling (VER-053)
- [ ] Error categorization taxonomy
- [ ] AST diff computation
- [ ] Soft label generation

### Phase 5: Integration (VER-054)
- [ ] End-to-end pipeline
- [ ] Parquet export with schema
- [ ] Benchmarks and validation

---

## Dependencies

```toml
[dependencies]
aprender = "0.9"      # ML algorithms
verificar = "0.1"     # Synthetic data generation
trueno = "0.7"        # SIMD tensor ops
serde = "1"
serde_yaml = "0.9"
parquet = "54"
tokio = { version = "1", features = ["full"] }
```

---

## References

1. **PAIML Vision Sync**: `/docs/specifications/paiml-sai-vision-sync.md`
2. **Aprender Documentation**: `../aprender/README.md`
3. **Verificar Architecture**: `../verificar/CLAUDE.md`
4. **Defect Analysis**: 1,296 defect-fix commits across PAIML transpilers
