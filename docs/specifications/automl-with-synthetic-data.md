# AutoML with Synthetic Data Specification v1.0

## aprender Automated Synthetic Data Generation for AutoML

**Status**: Draft
**Version**: 1.1.0
**Target Release**: v0.14.0
**Depends On**: AutoML Specification v1.0
**Review Status**: Approved with Comments (2025-11-26)

---

## 1. Executive Summary

This specification extends aprender's AutoML module with automatic synthetic data generation capabilities. The system creates augmented training data to improve model performance, particularly for low-resource domains like shell autocomplete and code translation (Python-to-Rust oracle). Synthetic data is generated, validated, and integrated into the AutoML optimization loop as a first-class hyperparameter.

## 2. Motivation

### 2.1 Problem Statement

Many machine learning tasks suffer from insufficient training data:

1. **Shell Autocomplete SLMs**: Limited corpus of shell command sequences with proper context
2. **Python-to-Rust Oracle**: Sparse parallel corpus for code translation
3. **Domain-Specific NLP**: Specialized vocabularies with minimal labeled examples
4. **Long-Tail Classification**: Rare classes with few positive examples

### 2.2 Solution Approach

Integrate synthetic data generation as a tunable component within the AutoML loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AutoML + Synthetic Data Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Synthetic│→ │  Search  │→ │  Trial   │→ │  Eval    │            │
│  │ Generator│  │  Space   │  │  Runner  │  │  Engine  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│       ↑              ↑             ↓             ↓                  │
│       │         ┌────────────────────────────────────┐              │
│       │         │    Surrogate Model (TPE/GP)        │              │
│       │         └────────────────────────────────────┘              │
│       │                        ↓                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Synthetic Data Hyperparameters:                              │  │
│  │  - augmentation_ratio: 0.0..2.0                               │  │
│  │  - generation_strategy: [BackTranslation, EDA, Template, ...]│  │
│  │  - quality_threshold: 0.0..1.0                                │  │
│  │  - diversity_weight: 0.0..1.0                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Architecture

### 3.1 Core Components

```rust
/// Synthetic data generator trait
pub trait SyntheticGenerator: Send + Sync {
    type Input;
    type Output;

    /// Generate synthetic examples from seed data
    fn generate(&self, seed: &[Self::Input], config: &SyntheticConfig)
        -> Result<Vec<Self::Output>>;

    /// Estimate quality of generated samples
    fn quality_score(&self, generated: &Self::Output, seed: &Self::Input) -> f32;

    /// Measure diversity of generated batch
    fn diversity_score(&self, batch: &[Self::Output]) -> f32;
}

/// Configuration for synthetic data generation
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Ratio of synthetic to original data (0.0 = none, 2.0 = 2x original)
    pub augmentation_ratio: f32,
    /// Minimum quality threshold for accepting generated samples
    pub quality_threshold: f32,
    /// Weight given to diversity vs quality in sample selection
    pub diversity_weight: f32,
    /// Maximum generation attempts per sample
    pub max_attempts: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

/// Andon alert handler for production monitoring (Toyota Jidoka)
pub trait AndonHandler: Send + Sync {
    /// Called when rejection rate exceeds threshold - HALT pipeline
    fn on_high_rejection(&self, rejection_rate: f32, threshold: f32);
    /// Called when quality score drifts below historical baseline
    fn on_quality_drift(&self, current: f32, baseline: f32);
}

/// Default Andon: log and halt
pub struct DefaultAndon;

impl AndonHandler for DefaultAndon {
    fn on_high_rejection(&self, rejection_rate: f32, threshold: f32) {
        panic!("ANDON HALT: Rejection rate {rejection_rate:.1}% exceeds {threshold:.1}% - check quality_score drift");
    }
    fn on_quality_drift(&self, current: f32, baseline: f32) {
        eprintln!("ANDON WARNING: Quality {current:.3} below baseline {baseline:.3}");
    }
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            augmentation_ratio: 0.5,
            quality_threshold: 0.7,
            diversity_weight: 0.3,
            max_attempts: 10,
            seed: 42,
            // Andon thresholds (Toyota Jidoka)
            andon_rejection_threshold: 0.90,  // Halt if >90% rejected
            andon_enabled: true,
        }
    }
}
```

### 3.2 Generation Strategies

```rust
/// Available synthetic data generation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationStrategy {
    /// Template-based generation with slot filling [1]
    Template,
    /// Easy Data Augmentation (synonym replacement, etc.) [2]
    EDA,
    /// Back-translation through pivot language [3, 10]
    BackTranslation,
    /// Mixup interpolation in embedding space [4]
    MixUp,
    /// Grammar-based recombination from rules [5]
    GrammarBased,
    /// Self-training with pseudo-labels [6]
    SelfTraining,
    /// Programmatic weak supervision [7]
    WeakSupervision,
}

/// Strategy selector for hybrid generation
pub struct HybridGenerator {
    strategies: Vec<(GenerationStrategy, f32)>,  // (strategy, weight)
    quality_estimator: Box<dyn QualityEstimator>,
}
```

### 3.3 Integration with AutoML Search Space

```rust
use aprender::automl::{SearchSpace, AutoTuner, TPE};
use aprender::synthetic::{SyntheticConfig, GenerationStrategy};

/// Type-safe synthetic data hyperparameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticParam {
    AugmentationRatio,
    QualityThreshold,
    DiversityWeight,
    Strategy,
}

impl ParamKey for SyntheticParam {
    fn name(&self) -> &'static str {
        match self {
            Self::AugmentationRatio => "augmentation_ratio",
            Self::QualityThreshold => "quality_threshold",
            Self::DiversityWeight => "diversity_weight",
            Self::Strategy => "strategy",
        }
    }
}

/// Create search space including synthetic data hyperparameters
fn create_search_space() -> SearchSpace<SyntheticParam> {
    SearchSpace::new()
        .add(SyntheticParam::AugmentationRatio, 0.0..2.0)
        .add(SyntheticParam::QualityThreshold, 0.5..0.95)
        .add(SyntheticParam::DiversityWeight, 0.0..0.5)
        .add_categorical(SyntheticParam::Strategy, &[
            "template", "eda", "back_translation", "mixup"
        ])
}
```

## 4. Use Case: Shell Autocomplete SLM

### 4.1 Problem Definition

Train a Small Language Model (SLM) for shell command autocomplete with limited training data:

```rust
/// Shell command with context
#[derive(Debug, Clone)]
pub struct ShellSample {
    /// Previous commands in session
    pub history: Vec<String>,
    /// Current partial input
    pub prefix: String,
    /// Completed command (label)
    pub completion: String,
    /// Working directory context
    pub cwd: PathBuf,
}

/// Generator for shell autocomplete training data
pub struct ShellSyntheticGenerator {
    /// Command grammar rules (e.g., git subcommands, cargo commands)
    grammar: ShellGrammar,
    /// Common command patterns from man pages
    templates: Vec<CommandTemplate>,
    /// Embedding model for semantic similarity
    embedder: CommandEmbedder,
}
```

### 4.2 Generation Pipeline

```rust
impl SyntheticGenerator for ShellSyntheticGenerator {
    type Input = ShellSample;
    type Output = ShellSample;

    fn generate(&self, seeds: &[ShellSample], config: &SyntheticConfig)
        -> Result<Vec<ShellSample>>
    {
        let mut synthetic = Vec::new();
        let target_count = (seeds.len() as f32 * config.augmentation_ratio) as usize;

        for seed in seeds {
            // Strategy 1: Template substitution
            // "git commit -m '{msg}'" → "git commit -m 'fix: update deps'"
            let template_samples = self.generate_from_template(seed)?;

            // Strategy 2: Argument permutation
            // "cargo build --release" → "cargo build --release --target wasm32"
            let permuted = self.permute_arguments(seed)?;

            // Strategy 3: Context variation
            // Change cwd, history to create new contexts
            let context_varied = self.vary_context(seed)?;

            // Filter by quality and add to synthetic set
            for sample in template_samples.chain(permuted).chain(context_varied) {
                if self.quality_score(&sample, seed) >= config.quality_threshold {
                    synthetic.push(sample);
                }
                if synthetic.len() >= target_count {
                    return Ok(synthetic);
                }
            }
        }

        Ok(synthetic)
    }

    fn quality_score(&self, generated: &ShellSample, seed: &ShellSample) -> f32 {
        let semantic_sim = self.embedder.similarity(&generated.completion, &seed.completion);
        let grammar_valid = self.grammar.is_valid(&generated.completion) as u8 as f32;
        let context_coherent = self.context_coherence(&generated);

        0.4 * semantic_sim + 0.4 * grammar_valid + 0.2 * context_coherent
    }

    fn diversity_score(&self, batch: &[ShellSample]) -> f32 {
        // Measure command diversity using embedding variance
        let embeddings: Vec<_> = batch.iter()
            .map(|s| self.embedder.embed(&s.completion))
            .collect();
        self.compute_batch_diversity(&embeddings)
    }
}
```

### 4.3 AutoML Integration for Shell SLM

```rust
use aprender::automl::{AutoTuner, TPE};
use aprender::synthetic::ShellSyntheticGenerator;

fn train_shell_autocomplete() -> Result<ShellModel> {
    let original_data = load_shell_history()?;
    let generator = ShellSyntheticGenerator::new()?;

    // Search space includes model AND synthetic data hyperparameters
    let space = SearchSpace::new()
        // Model hyperparameters
        .add(ModelParam::HiddenSize, 64..512)
        .add(ModelParam::NumLayers, 1..4)
        .add(ModelParam::Dropout, 0.0..0.5)
        // Synthetic data hyperparameters (jointly optimized)
        .add(SyntheticParam::AugmentationRatio, 0.0..2.0)
        .add(SyntheticParam::QualityThreshold, 0.5..0.95)
        .add(SyntheticParam::Strategy, &["template", "eda", "grammar"]);

    let objective = |params: &ParamSet| {
        // Generate synthetic data with current hyperparameters
        let config = SyntheticConfig::from_params(params);
        let synthetic = generator.generate(&original_data, &config)?;

        // Combine original + synthetic
        let augmented_data = original_data.iter()
            .chain(synthetic.iter())
            .collect();

        // Train model with current architecture hyperparameters
        let model = ShellModel::from_params(params);
        model.fit(&augmented_data)?;

        // Evaluate on held-out validation set (original data only)
        model.score(&validation_data)
    };

    let tpe = TPE::new(100).with_gamma(0.25).with_startup_trials(10);
    let result = AutoTuner::new(tpe)
        .early_stopping(20)
        .maximize(&space, objective)?;

    // Train final model with best hyperparameters
    let best_config = SyntheticConfig::from_params(&result.best_params);
    let best_synthetic = generator.generate(&original_data, &best_config)?;
    let final_data: Vec<_> = original_data.iter()
        .chain(best_synthetic.iter())
        .collect();

    let model = ShellModel::from_params(&result.best_params);
    model.fit(&final_data)?;

    Ok(model)
}
```

## 5. Use Case: Python-to-Rust Oracle

### 5.1 Problem Definition

Train a code translation model with limited parallel corpus:

```rust
/// Parallel code sample
#[derive(Debug, Clone)]
pub struct CodePair {
    /// Source Python code
    pub python: String,
    /// Target Rust code
    pub rust: String,
    /// Semantic tags (e.g., "loop", "error_handling", "io")
    pub tags: Vec<String>,
    /// AST node types present
    pub ast_features: AstFeatures,
}

/// Generator for code translation training data
///
/// **EXPERIMENTAL** (v0.14.0): Code translation is AI-Complete.
/// Ship Shell SLM first; defer this to v0.15.0 for production use.
/// See review: docs/specifications/reviews/automl-with-synthetic-data-review.md
pub struct CodeTranslationGenerator {
    /// Python parser for AST manipulation
    python_parser: PythonParser,
    /// Rust parser for validation
    rust_parser: RustParser,
    /// Code embeddings for semantic similarity
    code_embedder: CodeEmbedder,
    /// Type inference engine
    type_inferencer: TypeInferencer,
    /// **[NASA V&V]** Sandbox executor for functional correctness
    /// Codex-class models hallucinate bugs that compile but fail tests
    sandbox: SandboxExecutor,
}
```

### 5.2 Generation Strategies

```rust
impl SyntheticGenerator for CodeTranslationGenerator {
    type Input = CodePair;
    type Output = CodePair;

    fn generate(&self, seeds: &[CodePair], config: &SyntheticConfig)
        -> Result<Vec<CodePair>>
    {
        let mut synthetic = Vec::new();

        for seed in seeds {
            // Strategy 1: Variable renaming (preserves semantics)
            // def foo(x): → def process(value):
            let renamed = self.rename_variables(seed)?;

            // Strategy 2: Code pattern composition
            // Combine loop patterns, error handling from different seeds
            let composed = self.compose_patterns(seed, seeds)?;

            // Strategy 3: Type-guided mutation
            // Change types while maintaining correctness
            let type_mutated = self.mutate_types(seed)?;

            // Strategy 4: Back-translation via intermediate representation
            // Python → AST → mutate → Python' → Rust'
            let back_translated = self.back_translate(seed)?;

            for sample in renamed.chain(composed).chain(type_mutated).chain(back_translated) {
                // Validate Rust code compiles
                if self.rust_parser.is_valid(&sample.rust) {
                    if self.quality_score(&sample, seed) >= config.quality_threshold {
                        synthetic.push(sample);
                    }
                }
            }
        }

        // Deduplicate by semantic hash
        self.deduplicate(&mut synthetic);

        Ok(synthetic)
    }

    fn quality_score(&self, generated: &CodePair, seed: &CodePair) -> f32 {
        // Semantic similarity of code embeddings
        let python_sim = self.code_embedder.similarity(&generated.python, &seed.python);
        let rust_sim = self.code_embedder.similarity(&generated.rust, &seed.rust);

        // Type consistency check
        let type_consistent = self.type_inferencer
            .check_consistency(&generated.python, &generated.rust) as u8 as f32;

        // Rust compilation success (binary)
        let compiles = self.rust_parser.compiles(&generated.rust) as u8 as f32;

        // **[NASA V&V]** Functional correctness via sandbox execution
        // Addresses Chen et al. Codex hallucination issue - compiles != correct
        let functional_correct = if compiles > 0.5 {
            // Generate unit tests from Python behavior, run against Rust
            let tests = self.sandbox.generate_tests_from_python(&seed.python);
            let pass_rate = self.sandbox.run_rust_tests(&generated.rust, &tests);
            pass_rate  // 0.0 - 1.0
        } else {
            0.0
        };

        // Weights adjusted: functional correctness now primary signal
        0.15 * python_sim + 0.15 * rust_sim + 0.1 * type_consistent
            + 0.2 * compiles + 0.4 * functional_correct
    }

    fn diversity_score(&self, batch: &[CodePair]) -> f32 {
        // Measure AST pattern diversity
        let patterns: HashSet<_> = batch.iter()
            .flat_map(|p| &p.tags)
            .collect();
        patterns.len() as f32 / batch.len() as f32
    }
}
```

### 5.3 AutoML Integration for Code Translation

```rust
fn train_python_rust_oracle() -> Result<TranslationModel> {
    let parallel_corpus = load_parallel_corpus()?;  // ~1000 samples
    let generator = CodeTranslationGenerator::new()?;

    let space = SearchSpace::new()
        // Transformer hyperparameters
        .add(ModelParam::NumHeads, 4..16)
        .add(ModelParam::HiddenDim, 256..1024)
        .add(ModelParam::NumLayers, 2..8)
        // Synthetic data hyperparameters
        .add(SyntheticParam::AugmentationRatio, 0.5..3.0)  // More aggressive
        .add(SyntheticParam::QualityThreshold, 0.6..0.9)
        .add(SyntheticParam::Strategy, &[
            "variable_rename", "pattern_compose", "type_mutate", "back_translate"
        ]);

    // Joint optimization of model architecture + data augmentation
    let result = AutoTuner::new(TPE::new(200))
        .time_budget(Duration::from_hours(4))
        .maximize(&space, |params| train_and_evaluate(params, &parallel_corpus, &generator))?;

    Ok(build_final_model(&result))
}
```

## 6. Quality Assurance

### 6.1 Validation Pipeline

```rust
/// Validates generated synthetic samples before inclusion
pub struct SyntheticValidator {
    /// Minimum semantic similarity to seed
    min_similarity: f32,
    /// Maximum allowed overlap with existing data
    max_overlap: f32,
    /// Domain-specific validators
    validators: Vec<Box<dyn Validator>>,
}

impl SyntheticValidator {
    pub fn validate(&self, sample: &impl SyntheticSample,
                    seed: &impl SyntheticSample,
                    existing: &[impl SyntheticSample]) -> ValidationResult
    {
        // Check semantic similarity bounds
        let similarity = sample.similarity(seed);
        if similarity < self.min_similarity {
            return ValidationResult::Rejected("Too dissimilar from seed");
        }
        if similarity > self.max_overlap {
            return ValidationResult::Rejected("Too similar (near-duplicate)");
        }

        // Run domain validators
        for validator in &self.validators {
            if let Err(e) = validator.validate(sample) {
                return ValidationResult::Rejected(e);
            }
        }

        ValidationResult::Accepted
    }
}
```

### 6.2 Diversity Metrics

```rust
/// Ensures generated data doesn't collapse to mode
pub struct DiversityMonitor {
    /// Embedding-based diversity (variance in latent space)
    embedding_diversity: f32,
    /// Token-level n-gram diversity
    ngram_diversity: f32,
    /// Label distribution entropy (for classification)
    label_entropy: f32,
}

impl DiversityMonitor {
    pub fn compute(&self, batch: &[impl Embeddable]) -> DiversityScore {
        let embeddings: Vec<_> = batch.iter()
            .map(|x| x.embed())
            .collect();

        // Compute pairwise distances
        let mut distances = Vec::new();
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                distances.push(cosine_distance(&embeddings[i], &embeddings[j]));
            }
        }

        DiversityScore {
            mean_distance: mean(&distances),
            min_distance: min(&distances),
            coverage: self.estimate_coverage(&embeddings),
        }
    }
}
```

## 7. API Design

### 7.1 High-Level API

```rust
use aprender::automl::{AutoML, SyntheticConfig};

// Simple integration: just enable synthetic data
let automl = AutoML::default()
    .synthetic_data(SyntheticConfig::default())
    .time_limit(Duration::from_minutes(60))
    .metric(Metric::Accuracy);

let result = automl.fit(&X_train, &y_train)?;
```

### 7.2 Custom Generator Integration

```rust
use aprender::automl::AutoML;
use aprender::synthetic::{SyntheticGenerator, register_generator};

// Register custom generator for specific domain
register_generator::<ShellSample>("shell", ShellSyntheticGenerator::new());
register_generator::<CodePair>("code_translation", CodeTranslationGenerator::new());

// AutoML automatically selects registered generator based on data type
let automl = AutoML::default()
    .synthetic_data(SyntheticConfig {
        augmentation_ratio: 1.0,
        quality_threshold: 0.8,
        ..Default::default()
    });
```

### 7.3 Callback Integration

```rust
/// Monitor synthetic data generation during AutoML
pub trait SyntheticCallback: Send + Sync {
    /// Called after each batch of synthetic samples is generated
    fn on_batch_generated(&mut self, count: usize, config: &SyntheticConfig);
    /// Called when quality falls below threshold
    fn on_quality_below_threshold(&mut self, actual: f32, threshold: f32);
    /// Called when diversity metrics indicate collapse
    fn on_diversity_collapse(&mut self, score: &DiversityScore);
}

/// Default logging callback
pub struct LoggingCallback;

impl SyntheticCallback for LoggingCallback {
    fn on_batch_generated(&mut self, count: usize, config: &SyntheticConfig) {
        println!("Generated {count} samples with ratio {}", config.augmentation_ratio);
    }

    fn on_quality_below_threshold(&mut self, actual: f32, threshold: f32) {
        eprintln!("Warning: quality {actual:.2} < threshold {threshold:.2}");
    }

    fn on_diversity_collapse(&mut self, score: &DiversityScore) {
        eprintln!("Warning: diversity collapsed, mean_distance={:.3}", score.mean_distance);
    }
}

let automl = AutoML::default()
    .synthetic_data(SyntheticConfig::default())
    .add_callback(LoggingCallback);
```

## 8. Performance Considerations

### 8.1 Caching Strategy

```rust
/// Cache generated samples to avoid regeneration
pub struct SyntheticCache {
    /// Seed hash → generated samples
    cache: HashMap<u64, Vec<SyntheticSample>>,
    /// Maximum cache size in bytes
    max_size: usize,
    /// LRU eviction order
    lru: LinkedList<u64>,
}

impl SyntheticCache {
    pub fn get_or_generate<G: SyntheticGenerator>(
        &mut self,
        seeds: &[G::Input],
        config: &SyntheticConfig,
        generator: &G,
    ) -> Result<&[G::Output]> {
        let key = self.compute_key(seeds, config);

        if !self.cache.contains_key(&key) {
            let generated = generator.generate(seeds, config)?;
            self.insert(key, generated);
        }

        Ok(&self.cache[&key])
    }
}
```

### 8.2 Batched Generation

```rust
/// Generate synthetic data in batches to manage memory
/// Note: aprender avoids rayon dependency; use trueno's SIMD for parallelism
pub fn generate_batched<G: SyntheticGenerator>(
    generator: &G,
    seeds: &[G::Input],
    config: &SyntheticConfig,
    batch_size: usize,
) -> Result<Vec<G::Output>> {
    let mut all_synthetic = Vec::new();

    for chunk in seeds.chunks(batch_size) {
        let batch = generator.generate(chunk, config)?;
        all_synthetic.extend(batch);
    }

    Ok(all_synthetic)
}

/// Streaming generation for memory-constrained environments
pub struct SyntheticStream<'a, G: SyntheticGenerator> {
    generator: &'a G,
    seeds: &'a [G::Input],
    config: &'a SyntheticConfig,
    current_idx: usize,
    batch_size: usize,
}

impl<'a, G: SyntheticGenerator> Iterator for SyntheticStream<'a, G> {
    type Item = Result<Vec<G::Output>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.seeds.len() {
            return None;
        }
        let end = (self.current_idx + self.batch_size).min(self.seeds.len());
        let chunk = &self.seeds[self.current_idx..end];
        self.current_idx = end;
        Some(self.generator.generate(chunk, self.config))
    }
}
```

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Mode Collapse** | Generated samples converge to limited patterns | Diversity monitoring with automatic threshold; hybrid strategy mixing |
| **Label Noise Propagation** | Incorrect synthetic labels degrade model | Quality threshold filtering; validation on held-out original data only |
| **Distribution Shift** | Synthetic data doesn't match real distribution | Semantic similarity bounds; domain-specific validators |
| **Overfitting to Artifacts** | Model learns generation artifacts, not patterns | Regularization during training; artifact detection in quality scoring |
| **Computational Cost** | Generation overhead exceeds training savings | Caching; batched generation; early stopping when quality plateaus |
| **Evaluation Contamination** | Synthetic data leaks into validation set | Strict train/val/test separation; evaluate only on original data |
| **Silent Quality Drift** [Toyota] | Embedder bias causes garbage generation | **Andon mechanism**: Halt if rejection rate >90%; alert on quality drift |
| **Code Hallucination** [NASA] | Rust compiles but has subtle logic bugs | **Sandbox V&V**: Generate tests from Python, execute against Rust |
| **AI-Complete Overreach** [Startup] | Code Oracle delays Shell SLM MVP | **Decoupling**: Ship Shell SLM v0.14.0; defer Code Oracle to v0.15.0 |

### 9.1 Quality Degradation Detection

```rust
/// Detect when synthetic data is hurting rather than helping
pub struct QualityDegradationDetector {
    /// Baseline score without synthetic data
    baseline_score: f32,
    /// Minimum improvement required to justify synthetic data
    min_improvement: f32,
    /// Rolling window of recent scores
    recent_scores: VecDeque<f32>,
}

impl QualityDegradationDetector {
    pub fn should_disable_synthetic(&self) -> bool {
        let recent_mean = self.recent_scores.iter().sum::<f32>()
            / self.recent_scores.len() as f32;
        recent_mean < self.baseline_score - self.min_improvement
    }
}
```

## 10. Implementation Roadmap

**Updated per review feedback (2025-11-26):** Decoupled Code Oracle to v0.15.0.

| Phase | Features | Target Release | Status |
|-------|----------|----------------|--------|
| 1 | Core `SyntheticGenerator` trait, `SyntheticConfig`, basic validators | v0.13.0 | Planned |
| 2 | EDA, Template strategies, AutoML integration, **Andon mechanism** | v0.13.1 | Planned |
| 3 | Shell autocomplete generator, quality metrics, diversity monitoring | v0.14.0 | **MVP** |
| 4 | Advanced strategies (MixUp, weak supervision), caching | v0.14.1 | Planned |
| 5 | **[EXPERIMENTAL]** Code translation generator, sandbox V&V, back-translation | v0.15.0 | Deferred |

**Rationale:** Shell Autocomplete is a "Structured Prediction" problem (tractable). Code Translation
is "AI-Complete" (Chen et al. Codex shows high hallucination rate). Ship proven value first.

## 11. References (Peer-Reviewed Publications)

[1] **Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019).** AutoAugment: Learning Augmentation Strategies from Data. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 113-123.
> Introduces automated search for optimal data augmentation policies using reinforcement learning. Demonstrates that learned augmentation strategies transfer across datasets. Foundation for our strategy selection mechanism—AutoAugment shows augmentation is a searchable hyperparameter.

[2] **Wei, J., & Zou, K. (2019).** EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 6382-6388.
> Proposes four simple text augmentation operations: synonym replacement, random insertion, random swap, random deletion. Shows 0.5-3% accuracy improvement on small datasets. Basis for our `GenerationStrategy::EDA` implementation for text data.

[3] **Sennrich, R., Haddow, B., & Birch, A. (2016).** Improving Neural Machine Translation Models with Monolingual Data. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 86-96.
> Introduces back-translation for generating synthetic parallel data from monolingual corpora. State-of-the-art for low-resource translation. Core technique for our `GenerationStrategy::BackTranslation` in the Python-to-Rust oracle.

[4] **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).** mixup: Beyond Empirical Risk Minimization. *International Conference on Learning Representations (ICLR)*.
> Proposes training on convex combinations of examples and labels. Improves generalization and calibration. Implemented as `GenerationStrategy::MixUp` for embedding-space interpolation of code and command samples.

[5] **Jia, R., & Liang, P. (2016).** Data Recombination for Neural Semantic Parsing. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 12-22.
> Introduces grammar-based recombination for generating synthetic training data in semantic parsing. Achieves 3-5% accuracy improvements on benchmark datasets. Directly motivates our `GenerationStrategy::GrammarBased` for shell command generation using command grammar rules.

[6] **Xie, Q., Luong, M. T., Hovy, E., & Le, Q. V. (2020).** Self-Training With Noisy Student Improves ImageNet Classification. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 10687-10698.
> Shows self-training with noise injection achieves state-of-the-art on ImageNet. Pseudo-labels on unlabeled data provide significant gains. Basis for our `GenerationStrategy::SelfTraining` using model predictions as synthetic labels.

[7] **Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017).** Snorkel: Rapid Training Data Creation with Weak Supervision. *Proceedings of the VLDB Endowment*, 11(3), 269-282.
> Introduces programmatic labeling functions for weak supervision. Enables creation of large training sets from heuristic rules. Foundation for `GenerationStrategy::WeakSupervision` using domain rules (e.g., shell command syntax) to label synthetic data.

[8] **Anaby-Tavor, A., Carmeli, B., Goldbraich, E., Kanber, A., Kour, G., Shlomov, S., Tepper, N., & Zwerdling, N. (2020).** Do Not Have Enough Data? Deep Learning to the Rescue! *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05), 7383-7390.
> Proposes LAMBDA for language model-based data augmentation. Fine-tuned language models generate class-conditional synthetic examples. Validates our approach of using domain-specific generation for low-resource classification.

[9] **Feng, S. Y., Gangal, V., Wei, J., Chandar, S., Vosoughi, S., Mitamura, T., & Hovy, E. (2021).** A Survey of Data Augmentation Approaches for NLP. *Findings of the Association for Computational Linguistics (ACL-IJCNLP)*, 968-988.
> Comprehensive survey of 100+ NLP augmentation techniques organized by linguistic level (character, word, sentence, document). Provides taxonomy informing our strategy selection. Reference for comparing augmentation effectiveness across domains.

[10] **Roziere, B., Lachaux, M. A., Chanussot, L., & Lample, G. (2020).** Unsupervised Translation of Programming Languages. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 20601-20611.
> Presents TransCoder for unsupervised code translation between programming languages using back-translation. Achieves state-of-the-art on Java↔Python and C++↔Python translation. Directly informs our Python-to-Rust oracle architecture and back-translation strategy for code synthesis.

---

## Appendix A: Quality Metrics Reference

| Metric | Description | Target Range |
|--------|-------------|--------------|
| Semantic Similarity | Cosine similarity in embedding space | 0.6 - 0.95 |
| Syntactic Validity | Parse success rate | > 0.99 |
| Label Preservation | Classification agreement with seed | > 0.95 |
| Diversity Score | Mean pairwise distance | > 0.3 |
| Novelty Score | Distance from nearest seed | > 0.1 |

## Appendix B: Shell Command Grammar (Excerpt)

```ebnf
command     ::= simple_cmd | pipeline | compound
simple_cmd  ::= cmd_name (option | argument)*
cmd_name    ::= "git" | "cargo" | "npm" | "docker" | ...
option      ::= "-" letter+ | "--" word ("=" value)?
argument    ::= word | quoted_string | substitution
pipeline    ::= command ("|" command)+
compound    ::= "if" | "for" | "while" | ...
```
