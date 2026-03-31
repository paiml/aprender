# Advanced Neural Network Pruning: A Comprehensive Specification for Aprender

**Authors:** PAIML Research Team
**Date:** 2026-01-05
**Version:** 1.0.0
**Status:** DRAFT - AWAITING REVIEW
**Type:** Implementation Specification

---

## Abstract

This specification defines the implementation of state-of-the-art neural network pruning techniques for the Aprender machine learning library. Drawing from peer-reviewed research in magnitude pruning, activation-weighted pruning (Wanda), second-order methods (SparseGPT), and structured pruning (Minitron, LLM-Pruner), we present a comprehensive Rust-native implementation that adheres to Toyota Way principles of quality-at-source (Jidoka), continuous improvement (Kaizen), and waste elimination (Muda). The specification includes a 100-point Popperian Falsification QA checklist, PMAT compliance requirements, and integration with the probador testing framework and renacer tracing infrastructure.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Abstractions](#4-core-abstractions)
5. [Pruning Algorithms](#5-pruning-algorithms)
6. [Sparsity Patterns](#6-sparsity-patterns)
7. [Integration with Entrenar](#7-integration-with-entrenar)
8. [Toyota Way Compliance](#8-toyota-way-compliance)
9. [Peer-Reviewed Citations](#9-peer-reviewed-citations)
10. [100-Point Popperian Falsification QA Checklist](#10-100-point-popperian-falsification-qa-checklist)
11. [PMAT Compliance](#11-pmat-compliance)
12. [Enhanced Testing with Probador](#12-enhanced-testing-with-probador)
13. [Tracing with Renacer](#13-tracing-with-renacer)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [References](#15-references)

---

## 1. Introduction

### 1.1 Motivation

Large Language Models (LLMs) have achieved remarkable performance across diverse tasks, yet their computational requirements present significant deployment challenges. A 7B parameter model requires approximately 14GB of memory in FP16, making edge deployment impractical. Neural network pruning offers a principled approach to model compression by removing redundant parameters while preserving task performance.

> "The key insight of modern pruning research is that neural networks are significantly over-parameterized, and substantial compression can be achieved with minimal accuracy degradation."
> — Frankle & Carlin (2019), The Lottery Ticket Hypothesis

### 1.2 Scope

This specification covers:

1. **Core Pruning Primitives** (Aprender) - Importance scoring, sparsity masks, sparse tensors
2. **Pruning Algorithms** (Aprender) - Magnitude, Wanda, SparseGPT, Minitron depth/width
3. **Training Integration** (Entrenar) - Schedules, callbacks, calibration pipelines
4. **Quality Assurance** - Falsification testing, PMAT gates, tracing

### 1.3 Design Philosophy

Following Toyota Way principles:

| Principle | Application to Pruning |
|-----------|----------------------|
| **Jidoka** (Quality at Source) | Importance scores computed with verified numerical stability |
| **Kaizen** (Continuous Improvement) | Iterative pruning with feedback loops |
| **Muda** (Waste Elimination) | Remove redundant parameters without accuracy loss |
| **Genchi Genbutsu** (Go and See) | Calibration data reveals actual activation patterns |
| **Heijunka** (Level Loading) | Balanced sparsity across layers |

---

## 2. Theoretical Foundation

### 2.1 The Pruning Problem

Given a neural network $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ with parameters $\theta \in \mathbb{R}^n$, pruning seeks a sparse parameter vector $\theta_s$ with $\|\theta_s\|_0 \leq k$ such that:

$$\mathcal{L}(f_{\theta_s}) \approx \mathcal{L}(f_\theta)$$

where $\mathcal{L}$ is the task loss and $k \ll n$ is the sparsity budget.

### 2.2 Importance Estimation Taxonomy

The fundamental question in pruning is: **which parameters are important?** We categorize methods based on their information source and computational complexity.

| Method | Importance Metric | Computational Cost | Data Requirement | Key Reference |
|--------|------------------|-------------------|------------------|---------------|
| **Magnitude** | $\|w_i\|$ | $O(n)$ | None | Han et al. (2015) |
| **Gradient** | $\|w_i \cdot \nabla_{w_i} \mathcal{L}\|$ | $O(n)$ | Training data | Molchanov et al. (2016) |
| **Taylor** | $\|w_i \cdot \nabla_{w_i} \mathcal{L} + \frac{1}{2} w_i^2 H_{ii}\|$ | $O(n^2)$ | Training data | Molchanov et al. (2019) |
| **Activation-Weighted** | $\|w_i\| \cdot \|\mathbf{a}_i\|$ | $O(n)$ | Calibration data | Sun et al. (2023) |
| **Hessian-Based** | $(H^{-1})_{ii}^{-1} w_i^2$ | $O(n^3)$ | Calibration data | LeCun et al. (1989); Frantar & Alistarh (2023) |

### 2.3 Sparsity Patterns

**Unstructured Sparsity:** Any individual weight can be pruned.
- Pros: Maximum flexibility, highest compression
- Cons: Requires sparse hardware (e.g., NVIDIA Ampere 2:4)

**Structured N:M Sparsity:** In every M consecutive elements, at most N are non-zero.
- 2:4 sparsity: 50% sparsity with 2x speedup on Ampere GPUs
- 4:8 sparsity: 50% sparsity with different granularity

**Block Sparsity:** Entire rows, columns, or attention heads removed.
- Pros: Direct wall-clock speedup on all hardware
- Cons: Coarser granularity may impact accuracy

---

## 3. Architecture Overview

### 3.1 Module Placement

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENTRENAR                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  prune/                                                  │   │
│  │  ├── schedule.rs      # GradualPruning, OneShotPruning  │   │
│  │  ├── callback.rs      # PruningCallback for train loop  │   │
│  │  ├── calibrate.rs     # Activation collection pipeline  │   │
│  │  └── pipeline.rs      # Prune→Finetune→Export workflows │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ uses                              │
│                              ▼                                   │
├─────────────────────────────────────────────────────────────────┤
│                         APRENDER                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  pruning/                                                │   │
│  │  ├── mod.rs           # Public API, trait definitions   │   │
│  │  ├── importance.rs    # Importance scoring algorithms   │   │
│  │  ├── mask.rs          # SparsityMask, N:M patterns      │   │
│  │  ├── sparse.rs        # Sparse tensor representations   │   │
│  │  ├── magnitude.rs     # L1/L2 magnitude pruning         │   │
│  │  ├── wanda.rs         # Weight-Activation pruning       │   │
│  │  ├── sparsegpt.rs     # Hessian-based OBS pruning       │   │
│  │  ├── depth.rs         # Layer removal (Minitron)        │   │
│  │  ├── width.rs         # Channel pruning (Minitron)      │   │
│  │  └── graph.rs         # Dependency tracking (LLM-Pruner)│   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ uses                              │
│                              ▼                                   │
├─────────────────────────────────────────────────────────────────┤
│                          TRUENO                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SIMD-accelerated matrix operations                      │   │
│  │  - Sparse matrix-vector multiplication                   │   │
│  │  - Blocked Cholesky decomposition                        │   │
│  │  - Efficient top-k selection                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calibration     │     │  Importance      │     │  Sparsity        │
│  Data (C4, etc.) │────▶│  Computation     │────▶│  Mask            │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                │                          │
                                │ activation stats         │ mask
                                ▼                          ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Model           │────▶│  WrappedModule   │────▶│  Pruned Model    │
│  (Dense)         │     │  (with hooks)    │     │  (Sparse)        │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           │ optional
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Fine-tuning     │
                                                  │  (Entrenar)      │
                                                  └──────────────────┘
```

---

## 4. Core Abstractions

### 4.1 Importance Trait

```rust
/// Core trait for importance estimation algorithms.
///
/// # Toyota Way: Jidoka (Quality at Source)
/// All implementations must validate numerical stability before returning scores.
/// NaN or Inf values trigger immediate failure (Andon cord).
pub trait Importance: Send + Sync {
    /// Compute importance scores for parameters in a module.
    ///
    /// # Arguments
    /// * `module` - The neural network module to analyze
    /// * `context` - Optional calibration context with activation statistics
    ///
    /// # Returns
    /// * `ImportanceScores` - Per-parameter importance values
    ///
    /// # Errors
    /// * `PruningError::NumericalInstability` - If scores contain NaN/Inf
    fn compute(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError>;

    /// Returns the name of this importance method for logging.
    fn name(&self) -> &'static str;

    /// Whether this method requires calibration data.
    fn requires_calibration(&self) -> bool;
}

/// Importance scores with metadata for analysis.
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    /// Raw importance values, same shape as parameter tensor.
    pub values: Tensor,
    /// Statistical summary for logging.
    pub stats: ImportanceStats,
    /// Method that produced these scores.
    pub method: String,
}

#[derive(Debug, Clone)]
pub struct ImportanceStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub sparsity_at_threshold: Vec<(f32, f32)>, // (threshold, sparsity_ratio)
}
```

### 4.2 Pruner Trait

```rust
/// High-level pruning interface.
///
/// # Toyota Way: Genchi Genbutsu (Go and See)
/// Pruners must operate on actual model weights, not abstractions.
pub trait Pruner: Send + Sync {
    /// Generate a sparsity mask based on importance scores.
    ///
    /// # Arguments
    /// * `scores` - Pre-computed importance scores
    /// * `target_sparsity` - Desired fraction of weights to prune (0.0 to 1.0)
    /// * `pattern` - Sparsity pattern constraint (unstructured, N:M, block)
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError>;

    /// Apply a sparsity mask to a module, zeroing pruned weights.
    ///
    /// # Safety
    /// This operation modifies weights in-place. The mask must match
    /// the module's parameter shapes exactly.
    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError>;

    /// Combined operation: compute importance, generate mask, apply.
    fn prune(
        &self,
        module: &mut dyn Module,
        target_sparsity: f32,
        pattern: SparsityPattern,
        context: Option<&CalibrationContext>,
    ) -> Result<PruningResult, PruningError>;
}

/// Result of a pruning operation with diagnostics.
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// Actual achieved sparsity (may differ from target for structured pruning).
    pub achieved_sparsity: f32,
    /// Number of parameters pruned.
    pub parameters_pruned: usize,
    /// Total parameters in module.
    pub total_parameters: usize,
    /// Per-layer sparsity breakdown.
    pub layer_sparsity: HashMap<String, f32>,
    /// Estimated memory savings in bytes.
    pub memory_savings_bytes: usize,
}
```

### 4.3 Sparsity Mask

```rust
/// Represents which parameters to keep (1) or prune (0).
///
/// # Toyota Way: Poka-Yoke (Mistake-Proofing)
/// Masks are immutable after creation and validate shape compatibility.
#[derive(Debug, Clone)]
pub struct SparsityMask {
    /// Binary mask tensor (1 = keep, 0 = prune).
    mask: Tensor,
    /// Pattern used to generate this mask.
    pattern: SparsityPattern,
    /// Actual sparsity ratio.
    sparsity: f32,
}

impl SparsityMask {
    /// Create a new mask with validation.
    ///
    /// # Errors
    /// * `PruningError::InvalidMask` - If mask contains non-binary values
    /// * `PruningError::ShapeMismatch` - If mask shape is invalid
    pub fn new(mask: Tensor, pattern: SparsityPattern) -> Result<Self, PruningError> {
        // Validate binary values
        let non_binary = mask.iter()
            .any(|&v| v != 0.0 && v != 1.0);
        if non_binary {
            return Err(PruningError::InvalidMask("Mask must be binary".into()));
        }

        // Validate pattern constraints
        pattern.validate(&mask)?;

        let sparsity = 1.0 - mask.mean();
        Ok(Self { mask, pattern, sparsity })
    }

    /// Apply mask to weights: w_masked = w * mask
    pub fn apply(&self, weights: &mut Tensor) -> Result<(), PruningError> {
        if weights.shape() != self.mask.shape() {
            return Err(PruningError::ShapeMismatch {
                expected: self.mask.shape().to_vec(),
                got: weights.shape().to_vec(),
            });
        }
        *weights = weights.mul(&self.mask);
        Ok(())
    }
}

/// Sparsity pattern constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityPattern {
    /// No structural constraint - any element can be pruned.
    Unstructured,
    /// N:M sparsity - in every M elements, at most N are non-zero.
    NM { n: usize, m: usize },
    /// Block sparsity - entire blocks of size (h, w) are pruned together.
    Block { height: usize, width: usize },
    /// Row sparsity - entire rows (output channels) pruned.
    Row,
    /// Column sparsity - entire columns (input channels) pruned.
    Column,
}
```

### 4.4 Calibration Context

```rust
/// Holds activation statistics collected during calibration forward passes.
///
/// # Toyota Way: Genchi Genbutsu (Go and See)
/// Real activation patterns from calibration data, not synthetic estimates.
#[derive(Debug, Clone)]
pub struct CalibrationContext {
    /// Per-layer input activation statistics.
    pub activation_stats: HashMap<String, ActivationStats>,
    /// Number of calibration samples processed.
    pub num_samples: usize,
    /// Calibration dataset identifier.
    pub dataset: String,
}

#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// L2 norm of input activations per input channel.
    /// Shape: [input_features]
    pub input_norms: Tensor,
    /// Running mean of squared activations.
    /// Shape: [input_features]
    pub squared_mean: Tensor,
    /// Sample count for this layer.
    pub count: usize,
}

impl ActivationStats {
    /// Online update with new batch of activations.
    ///
    /// Uses Welford's algorithm for numerical stability.
    pub fn update(&mut self, activations: &Tensor) {
        let batch_size = activations.shape()[0];
        let new_count = self.count + batch_size;

        // Compute batch statistics
        let batch_norms = activations.pow(2.0).sum_dim(0).sqrt();
        let batch_sq_mean = activations.pow(2.0).mean_dim(0);

        // Welford's online update
        let delta = &batch_norms - &self.input_norms;
        self.input_norms = &self.input_norms + &delta * (batch_size as f32 / new_count as f32);

        let delta_sq = &batch_sq_mean - &self.squared_mean;
        self.squared_mean = &self.squared_mean + &delta_sq * (batch_size as f32 / new_count as f32);

        self.count = new_count;
    }
}
```

---

## 5. Pruning Algorithms

### 5.1 Magnitude Pruning

**Reference:** Han et al. (2015), "Learning both Weights and Connections for Efficient Neural Networks"

The simplest and most widely-used pruning method. Parameters with smallest absolute values are removed.

```rust
/// Magnitude-based importance estimation.
///
/// # Algorithm
/// For each weight w_ij:
///   importance(w_ij) = |w_ij|^p
///
/// where p is the norm order (1 for L1, 2 for L2).
///
/// # Citation
/// Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights
/// and connections for efficient neural networks. NeurIPS.
pub struct MagnitudeImportance {
    /// Norm order: 1 for L1, 2 for L2.
    pub norm: usize,
    /// Whether to normalize by layer (local) or globally.
    pub scope: ImportanceScope,
}

impl Importance for MagnitudeImportance {
    fn compute(
        &self,
        module: &dyn Module,
        _context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        let weights = module.weight()?;

        let values = match self.norm {
            1 => weights.abs(),
            2 => weights.pow(2.0),
            p => weights.abs().pow(p as f32),
        };

        // Validate numerical stability (Jidoka)
        if values.has_nan() || values.has_inf() {
            return Err(PruningError::NumericalInstability {
                method: "MagnitudeImportance",
                details: "NaN or Inf in importance scores".into(),
            });
        }

        Ok(ImportanceScores {
            values,
            stats: compute_stats(&values),
            method: format!("magnitude_l{}", self.norm),
        })
    }

    fn name(&self) -> &'static str { "magnitude" }
    fn requires_calibration(&self) -> bool { false }
}
```

### 5.2 Wanda (Weights and Activations)

**Reference:** Sun et al. (2023), "A Simple and Effective Pruning Approach for Large Language Models"

Wanda combines weight magnitude with input activation norms, achieving state-of-the-art results without fine-tuning.

```rust
/// Wanda: Weight and Activation pruning.
///
/// # Algorithm
/// For each weight w_ij connecting input j to output i:
///   importance(w_ij) = |w_ij| * ||X_j||_2
///
/// where X_j is the j-th column of input activations across calibration samples.
///
/// # Key Insight
/// Weights connected to frequently-activated inputs are more important,
/// even if the weight magnitude itself is small.
///
/// # Citation
/// Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective
/// pruning approach for large language models. arXiv:2306.11695.
pub struct WandaImportance {
    /// Sparsity pattern (unstructured or N:M).
    pub pattern: SparsityPattern,
}

impl Importance for WandaImportance {
    fn compute(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        let context = context.ok_or(PruningError::CalibrationRequired {
            method: "Wanda",
        })?;

        let weights = module.weight()?;
        let layer_name = module.name();

        let stats = context.activation_stats.get(&layer_name)
            .ok_or(PruningError::MissingActivationStats { layer: layer_name.clone() })?;

        // Wanda metric: |W| * sqrt(activation_norm)
        // The sqrt comes from treating activation_norm as sum of squared inputs
        let scaler = stats.input_norms.sqrt();

        // Broadcast scaler across output dimension
        // weights: [out_features, in_features]
        // scaler: [in_features]
        let values = weights.abs() * scaler.unsqueeze(0);

        // Validate (Jidoka)
        if values.has_nan() || values.has_inf() {
            return Err(PruningError::NumericalInstability {
                method: "Wanda",
                details: format!(
                    "NaN/Inf in Wanda scores. Weight range: [{:.4}, {:.4}], Scaler range: [{:.4}, {:.4}]",
                    weights.min(), weights.max(),
                    scaler.min(), scaler.max()
                ),
            });
        }

        Ok(ImportanceScores {
            values,
            stats: compute_stats(&values),
            method: "wanda".into(),
        })
    }

    fn name(&self) -> &'static str { "wanda" }
    fn requires_calibration(&self) -> bool { true }
}
```

### 5.3 SparseGPT (Hessian-Based)

**Reference:** Frantar & Alistarh (2023), "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"

SparseGPT uses second-order information (Hessian) to compensate for pruning error, achieving better accuracy than Wanda at higher sparsity.

```rust
/// SparseGPT: Optimal Brain Surgeon for LLMs.
///
/// # Algorithm
/// 1. Compute Hessian H = X^T X (input correlations)
/// 2. For each column j in order:
///    a. Find optimal weight to prune (lowest saliency)
///    b. Update remaining weights to compensate: W -= (w_j / H_jj) * H_j
///    c. Update Hessian inverse using rank-1 update
///
/// # Key Insight
/// Second-order information allows weight updates that minimize
/// the output perturbation caused by pruning.
///
/// # Citation
/// Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive language models
/// can be accurately pruned in one-shot. ICML.
pub struct SparseGPTImportance {
    /// Block size for block-wise processing (memory efficiency).
    pub block_size: usize,
    /// Damping factor for Hessian stability.
    pub damp: f32,
    /// Whether to use activation order (recommended).
    pub act_order: bool,
}

impl Default for SparseGPTImportance {
    fn default() -> Self {
        Self {
            block_size: 128,
            damp: 0.01,
            act_order: true,
        }
    }
}

impl SparseGPTImportance {
    /// Compute Hessian from calibration activations.
    ///
    /// H = (1/n) * X^T * X + damp * I
    fn compute_hessian(&self, activations: &Tensor) -> Tensor {
        let n = activations.shape()[0] as f32;
        let xtx = activations.t().matmul(activations) / n;

        // Add damping for numerical stability
        let d = xtx.shape()[0];
        let identity = Tensor::eye(d);
        xtx + identity * self.damp
    }

    /// Block-wise OBS update for memory efficiency.
    fn prune_block(
        &self,
        weights: &mut Tensor,
        hessian_inv: &mut Tensor,
        mask: &mut Tensor,
        block_start: usize,
        block_end: usize,
        target_sparsity: f32,
    ) -> Result<(), PruningError> {
        let block_size = block_end - block_start;
        let num_prune = (block_size as f32 * target_sparsity) as usize;

        for _ in 0..num_prune {
            // Compute saliency: w^2 / H^{-1}_{jj}
            let w_block = weights.slice(1, block_start, block_end);
            let h_diag = hessian_inv.diagonal().slice(0, block_start, block_end);

            let saliency = w_block.pow(2.0) / h_diag;

            // Find minimum saliency (considering existing mask)
            let masked_saliency = saliency + (1.0 - mask.slice(1, block_start, block_end)) * f32::MAX;
            let (min_row, min_col) = masked_saliency.argmin_2d();
            let global_col = block_start + min_col;

            // Get values for update
            let w_j = weights[[min_row, global_col]];
            let h_jj = hessian_inv[[global_col, global_col]];

            // Update weights to compensate (OBS formula)
            // W[:, :] -= (w_j / H_jj) * H[:, j]
            let h_col = hessian_inv.select(1, global_col);
            let update = h_col * (w_j / h_jj);
            *weights = weights.clone() - update.unsqueeze(0);

            // Zero the pruned weight
            weights[[min_row, global_col]] = 0.0;
            mask[[min_row, global_col]] = 0.0;

            // Rank-1 Hessian inverse update (Woodbury)
            let h_row = hessian_inv.select(0, global_col);
            let outer = h_col.outer(&h_row);
            *hessian_inv = hessian_inv.clone() - outer / h_jj;
        }

        Ok(())
    }
}

impl Importance for SparseGPTImportance {
    fn compute(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        // For SparseGPT, importance and pruning are interleaved.
        // This returns the initial saliency (w^2 / H^{-1}_{jj}) for visualization.
        let context = context.ok_or(PruningError::CalibrationRequired {
            method: "SparseGPT",
        })?;

        let weights = module.weight()?;
        let layer_name = module.name();

        let activations = context.get_raw_activations(&layer_name)?;
        let hessian = self.compute_hessian(&activations);

        // Cholesky decomposition for stable inversion
        let hessian_inv = hessian.cholesky()?.inverse()?;

        // Initial saliency
        let h_diag = hessian_inv.diagonal();
        let saliency = weights.pow(2.0) / h_diag.unsqueeze(0);

        Ok(ImportanceScores {
            values: saliency,
            stats: compute_stats(&saliency),
            method: "sparsegpt_saliency".into(),
        })
    }

    fn name(&self) -> &'static str { "sparsegpt" }
    fn requires_calibration(&self) -> bool { true }
}
```

### 5.4 Minitron Depth Pruning

**Reference:** Muralidharan et al. (2024), "Compact Language Models via Pruning and Knowledge Distillation"

Removes entire transformer layers based on block importance scores.

```rust
/// Minitron Depth Pruning: Layer removal based on Block Importance (BI).
///
/// # Algorithm
/// For each transformer layer l:
///   BI(l) = 1 - cosine_similarity(input_l, output_l)
///
/// Layers with lowest BI contribute least to output transformation.
///
/// # Key Insight
/// If a layer's output is very similar to its input, the layer
/// is performing minimal transformation and can be removed.
///
/// # Citation
/// Muralidharan, S., et al. (2024). Compact language models via pruning
/// and knowledge distillation. arXiv:2407.14679.
pub struct DepthPruner {
    /// Number of layers to remove.
    pub num_layers_to_remove: usize,
    /// Whether to use iterative removal (recommended).
    pub iterative: bool,
}

impl DepthPruner {
    /// Compute Block Importance scores for all layers.
    pub fn compute_block_importance(
        &self,
        model: &dyn Module,
        calibration_inputs: &[Tensor],
    ) -> Result<Vec<(usize, f32)>, PruningError> {
        let mut scores = Vec::new();

        for (idx, layer) in model.layers().enumerate() {
            let mut total_bi = 0.0;

            for input in calibration_inputs {
                let layer_input = layer.input_activations(input)?;
                let layer_output = layer.forward(&layer_input);

                // Block Importance = 1 - cosine_similarity
                let cos_sim = cosine_similarity(&layer_input, &layer_output);
                let bi = 1.0 - cos_sim;
                total_bi += bi;
            }

            let avg_bi = total_bi / calibration_inputs.len() as f32;
            scores.push((idx, avg_bi));
        }

        // Sort by importance (ascending - lowest BI first)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(scores)
    }

    /// Remove the least important layers.
    pub fn prune_layers(
        &self,
        model: &mut dyn Module,
        calibration_inputs: &[Tensor],
    ) -> Result<DepthPruningResult, PruningError> {
        let mut removed_layers = Vec::new();

        if self.iterative {
            // Remove one layer at a time, recomputing scores
            for _ in 0..self.num_layers_to_remove {
                let scores = self.compute_block_importance(model, calibration_inputs)?;
                let (layer_idx, bi_score) = scores[0];

                model.remove_layer(layer_idx)?;
                removed_layers.push((layer_idx, bi_score));
            }
        } else {
            // Compute scores once and remove all at once
            let scores = self.compute_block_importance(model, calibration_inputs)?;
            let to_remove: Vec<_> = scores.iter()
                .take(self.num_layers_to_remove)
                .map(|&(idx, score)| (idx, score))
                .collect();

            // Remove in reverse order to maintain indices
            for (idx, score) in to_remove.into_iter().rev() {
                model.remove_layer(idx)?;
                removed_layers.push((idx, score));
            }
        }

        Ok(DepthPruningResult {
            removed_layers,
            original_depth: model.num_layers() + self.num_layers_to_remove,
            final_depth: model.num_layers(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct DepthPruningResult {
    /// List of (layer_index, block_importance_score) for removed layers.
    pub removed_layers: Vec<(usize, f32)>,
    /// Original number of layers.
    pub original_depth: usize,
    /// Final number of layers after pruning.
    pub final_depth: usize,
}
```

### 5.5 Minitron Width Pruning

**Reference:** Muralidharan et al. (2024)

Removes channels (hidden dimensions) based on activation importance.

```rust
/// Minitron Width Pruning: Channel removal based on activation importance.
///
/// # Algorithm
/// For each hidden dimension d:
///   importance(d) = mean(|activations[:, d]|^2)
///
/// Channels with lowest activation magnitude are pruned across all layers.
///
/// # Constraint
/// Attention heads must be pruned as complete units to maintain validity.
pub struct WidthPruner {
    /// Target hidden dimension (must be divisible by num_heads).
    pub target_hidden_dim: usize,
    /// Target intermediate dimension for FFN.
    pub target_intermediate_dim: usize,
}

impl WidthPruner {
    /// Compute channel importance from activations.
    pub fn compute_channel_importance(
        &self,
        model: &dyn Module,
        calibration_inputs: &[Tensor],
    ) -> Result<ChannelImportance, PruningError> {
        let mut hidden_importance = Tensor::zeros(&[model.hidden_size()]);
        let mut intermediate_importance = Tensor::zeros(&[model.intermediate_size()]);

        for input in calibration_inputs {
            // Track hidden state activations
            let hidden_acts = model.get_hidden_activations(input)?;
            hidden_importance = hidden_importance + hidden_acts.pow(2.0).mean_dim(0);

            // Track FFN intermediate activations
            let ffn_acts = model.get_intermediate_activations(input)?;
            intermediate_importance = intermediate_importance + ffn_acts.pow(2.0).mean_dim(0);
        }

        let n = calibration_inputs.len() as f32;
        hidden_importance = hidden_importance / n;
        intermediate_importance = intermediate_importance / n;

        Ok(ChannelImportance {
            hidden: hidden_importance,
            intermediate: intermediate_importance,
        })
    }

    /// Prune channels while maintaining attention head consistency.
    pub fn prune_width(
        &self,
        model: &mut dyn Module,
        calibration_inputs: &[Tensor],
    ) -> Result<WidthPruningResult, PruningError> {
        let importance = self.compute_channel_importance(model, calibration_inputs)?;

        // Hidden dimension must be divisible by number of heads
        let num_heads = model.num_attention_heads();
        let head_dim = self.target_hidden_dim / num_heads;

        if self.target_hidden_dim % num_heads != 0 {
            return Err(PruningError::InvalidConfiguration {
                message: format!(
                    "target_hidden_dim ({}) must be divisible by num_heads ({})",
                    self.target_hidden_dim, num_heads
                ),
            });
        }

        // Find top-k channels to keep
        let hidden_keep = importance.hidden.topk(self.target_hidden_dim).indices;
        let intermediate_keep = importance.intermediate.topk(self.target_intermediate_dim).indices;

        // Prune all linear layers consistently
        model.prune_hidden_dim(&hidden_keep)?;
        model.prune_intermediate_dim(&intermediate_keep)?;

        Ok(WidthPruningResult {
            original_hidden_dim: model.hidden_size(),
            final_hidden_dim: self.target_hidden_dim,
            original_intermediate_dim: model.intermediate_size(),
            final_intermediate_dim: self.target_intermediate_dim,
            hidden_channels_kept: hidden_keep.to_vec(),
            intermediate_channels_kept: intermediate_keep.to_vec(),
        })
    }
}
```

---

## 6. Sparsity Patterns

### 6.1 N:M Sparsity Implementation

```rust
/// Generate N:M sparsity mask from importance scores.
///
/// # Algorithm
/// For each group of M consecutive elements, keep the N with highest importance.
///
/// # Citations
/// * Zhou, A., et al. (2021). "Learning N:M fine-grained structured sparse neural networks from scratch." ICLR.
/// * Mishra, A., et al. (2021). "Accelerating sparse deep neural networks." arXiv:2104.08378.
///
/// # Hardware Support
/// - 2:4 sparsity: NVIDIA Ampere (A100, RTX 30xx) - 2x speedup
/// - 4:8 sparsity: Future hardware
pub fn generate_nm_mask(
    scores: &Tensor,
    n: usize,
    m: usize,
) -> Result<SparsityMask, PruningError> {
    if n >= m {
        return Err(PruningError::InvalidPattern {
            message: format!("N ({}) must be less than M ({})", n, m),
        });
    }

    let shape = scores.shape();
    let total_elements = shape.iter().product::<usize>();

    if total_elements % m != 0 {
        return Err(PruningError::InvalidPattern {
            message: format!(
                "Total elements ({}) must be divisible by M ({})",
                total_elements, m
            ),
        });
    }

    let flat_scores = scores.flatten();
    let mut mask = Tensor::zeros(&[total_elements]);

    // Process each group of M elements
    for group_start in (0..total_elements).step_by(m) {
        let group_end = group_start + m;
        let group_scores = flat_scores.slice(0, group_start, group_end);

        // Find indices of top N elements in this group
        let top_indices = group_scores.topk(n).indices;

        // Set mask to 1 for top N elements
        for &idx in top_indices.iter() {
            mask[group_start + idx] = 1.0;
        }
    }

    let mask = mask.reshape(shape);
    SparsityMask::new(mask, SparsityPattern::NM { n, m })
}
```

### 6.2 Block Sparsity Implementation

```rust
/// Generate block sparsity mask.
///
/// # Algorithm
/// Aggregate importance scores within each block, prune lowest blocks.
pub fn generate_block_mask(
    scores: &Tensor,
    block_height: usize,
    block_width: usize,
    target_sparsity: f32,
) -> Result<SparsityMask, PruningError> {
    let [rows, cols] = scores.shape()[..] else {
        return Err(PruningError::InvalidShape {
            expected: "2D tensor".into(),
            got: format!("{}D tensor", scores.ndim()),
        });
    };

    if rows % block_height != 0 || cols % block_width != 0 {
        return Err(PruningError::InvalidPattern {
            message: format!(
                "Shape [{}, {}] not divisible by block size [{}, {}]",
                rows, cols, block_height, block_width
            ),
        });
    }

    let num_block_rows = rows / block_height;
    let num_block_cols = cols / block_width;
    let num_blocks = num_block_rows * num_block_cols;
    let num_prune = (num_blocks as f32 * target_sparsity) as usize;

    // Compute block importance (sum of element importance within block)
    let mut block_scores = Vec::with_capacity(num_blocks);
    for br in 0..num_block_rows {
        for bc in 0..num_block_cols {
            let r_start = br * block_height;
            let r_end = r_start + block_height;
            let c_start = bc * block_width;
            let c_end = c_start + block_width;

            let block = scores.slice_2d(r_start, r_end, c_start, c_end);
            let importance = block.sum();
            block_scores.push((br, bc, importance));
        }
    }

    // Sort by importance (ascending)
    block_scores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Create mask (all ones initially)
    let mut mask = Tensor::ones(&[rows, cols]);

    // Zero out the lowest-importance blocks
    for (br, bc, _) in block_scores.iter().take(num_prune) {
        let r_start = br * block_height;
        let r_end = r_start + block_height;
        let c_start = bc * block_width;
        let c_end = c_start + block_width;

        mask.slice_2d_mut(r_start, r_end, c_start, c_end).fill(0.0);
    }

    SparsityMask::new(mask, SparsityPattern::Block {
        height: block_height,
        width: block_width
    })
}
```

---

## 7. Integration with Entrenar

### 7.1 Pruning Callback

```rust
// In entrenar/src/prune/callback.rs

use aprender::pruning::{Pruner, Importance, SparsityPattern, CalibrationContext};

/// Callback for pruning during training.
///
/// # Toyota Way: Kaizen (Continuous Improvement)
/// Gradual pruning allows the model to adapt incrementally.
pub struct PruningCallback<I: Importance, P: Pruner> {
    importance: I,
    pruner: P,
    schedule: PruningSchedule,
    pattern: SparsityPattern,
    calibration_loader: DataLoader,
    current_sparsity: f32,
}

/// Pruning schedule definitions.
#[derive(Debug, Clone)]
pub enum PruningSchedule {
    /// Prune once at specified step.
    OneShot { step: usize },
    /// Gradually increase sparsity over steps.
    Gradual {
        start_step: usize,
        end_step: usize,
        initial_sparsity: f32,
        final_sparsity: f32,
        frequency: usize, // prune every N steps
    },
    /// Cubic sparsity schedule (Zhu & Gupta, 2017).
    Cubic {
        start_step: usize,
        end_step: usize,
        final_sparsity: f32,
    },
}

impl<I: Importance, P: Pruner> Callback for PruningCallback<I, P> {
    fn on_step_end(&mut self, trainer: &mut Trainer, step: usize) -> Result<(), TrainError> {
        let target_sparsity = self.schedule.sparsity_at_step(step);

        if target_sparsity > self.current_sparsity {
            // Collect calibration data
            let context = self.collect_calibration(trainer.model())?;

            // Compute importance and prune
            let result = self.pruner.prune(
                trainer.model_mut(),
                target_sparsity,
                self.pattern,
                Some(&context),
            )?;

            self.current_sparsity = result.achieved_sparsity;

            // Log metrics
            trainer.log_metric("pruning/sparsity", self.current_sparsity);
            trainer.log_metric("pruning/params_pruned", result.parameters_pruned as f64);
        }

        Ok(())
    }
}

impl PruningSchedule {
    pub fn sparsity_at_step(&self, step: usize) -> f32 {
        match self {
            PruningSchedule::OneShot { step: prune_step } => {
                if step >= *prune_step { 1.0 } else { 0.0 }
            }
            PruningSchedule::Gradual {
                start_step, end_step, initial_sparsity, final_sparsity, ..
            } => {
                if step < *start_step {
                    *initial_sparsity
                } else if step >= *end_step {
                    *final_sparsity
                } else {
                    let progress = (step - start_step) as f32 / (end_step - start_step) as f32;
                    initial_sparsity + progress * (final_sparsity - initial_sparsity)
                }
            }
            PruningSchedule::Cubic { start_step, end_step, final_sparsity } => {
                // s_t = s_f * (1 - (1 - t/T)^3)
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    *final_sparsity
                } else {
                    let t = (step - start_step) as f32;
                    let T = (end_step - start_step) as f32;
                    final_sparsity * (1.0 - (1.0 - t / T).powi(3))
                }
            }
        }
    }
}
```

### 7.2 YAML Configuration

```yaml
# entrenar pruning configuration

pruning:
  enabled: true
  method: wanda  # magnitude | wanda | sparsegpt | minitron_depth | minitron_width

  # Target sparsity
  target_sparsity: 0.5  # 50% weights pruned

  # Sparsity pattern
  pattern:
    type: nm  # unstructured | nm | block | row | column
    n: 2
    m: 4

  # Schedule
  schedule:
    type: gradual  # oneshot | gradual | cubic
    start_step: 1000
    end_step: 10000
    initial_sparsity: 0.0
    final_sparsity: 0.5
    frequency: 100  # prune every 100 steps

  # Calibration
  calibration:
    dataset: c4
    num_samples: 128
    sequence_length: 2048

  # Method-specific options
  wanda:
    # No additional options

  sparsegpt:
    block_size: 128
    damp: 0.01
    act_order: true

  minitron_depth:
    num_layers_to_remove: 4
    iterative: true

  minitron_width:
    target_hidden_dim: 2048
    target_intermediate_dim: 5632

# Post-pruning fine-tuning
fine_tune_after_pruning: true
fine_tune_steps: 1000
fine_tune_lr: 1e-5
```

---

## 8. Toyota Way Compliance

### 8.1 Principle Mapping

| Toyota Principle | Pruning Implementation |
|-----------------|----------------------|
| **Jidoka** (Quality at Source) | All importance computations validate for NaN/Inf. Sparsity masks validate binary values and shape compatibility. |
| **Kaizen** (Continuous Improvement) | Gradual pruning schedules. Iterative importance recomputation. Metrics logging for analysis. |
| **Muda** (Waste Elimination) | Pruning itself eliminates redundant parameters. Calibration uses minimal samples (128 typical). |
| **Genchi Genbutsu** (Go and See) | Calibration data reveals actual activation patterns, not theoretical estimates. |
| **Heijunka** (Level Loading) | Global pruning balances sparsity across layers. Per-layer metrics ensure no single layer is over-pruned. |
| **Andon** (Stop the Line) | Numerical instability triggers immediate error. Invalid configurations fail fast. |
| **Poka-Yoke** (Mistake-Proofing) | Type system prevents invalid sparsity patterns. Shape validation prevents mismatched masks. |
| **Hansei** (Reflection) | Post-pruning evaluation compares perplexity/accuracy. Pruning results logged for analysis. |

### 8.2 Quality Gates

```rust
/// Quality gate validation for pruning operations.
///
/// # Toyota Way: Andon
/// Fails fast if quality thresholds are violated.
pub struct PruningQualityGate {
    /// Maximum allowed perplexity increase after pruning.
    pub max_perplexity_increase: f32,
    /// Maximum layer sparsity imbalance (std / mean).
    pub max_sparsity_imbalance: f32,
    /// Minimum calibration samples required.
    pub min_calibration_samples: usize,
}

impl PruningQualityGate {
    pub fn validate(&self, result: &PruningResult, eval: &EvalMetrics) -> Result<(), QualityError> {
        // Check perplexity increase
        let ppl_increase = (eval.post_prune_ppl - eval.pre_prune_ppl) / eval.pre_prune_ppl;
        if ppl_increase > self.max_perplexity_increase {
            return Err(QualityError::PerplexityIncrease {
                expected_max: self.max_perplexity_increase,
                actual: ppl_increase,
            });
        }

        // Check sparsity balance across layers
        let sparsities: Vec<f32> = result.layer_sparsity.values().copied().collect();
        let mean = sparsities.iter().sum::<f32>() / sparsities.len() as f32;
        let variance = sparsities.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / sparsities.len() as f32;
        let imbalance = variance.sqrt() / mean;

        if imbalance > self.max_sparsity_imbalance {
            return Err(QualityError::SparsityImbalance {
                expected_max: self.max_sparsity_imbalance,
                actual: imbalance,
            });
        }

        Ok(())
    }
}
```

---

## 9. Peer-Reviewed Citations

### 9.1 Foundational Works

| # | Citation | Key Contribution | Venue |
|---|----------|-----------------|-------|
| 1 | Han, S., Pool, J., Tran, J., & Dally, W. (2015). *Learning both weights and connections for efficient neural networks.* | Established magnitude pruning as baseline; demonstrated 10x compression on AlexNet. | NeurIPS |
| 2 | Frankle, J., & Carlin, M. (2019). *The lottery ticket hypothesis: Finding sparse, trainable neural networks.* | Proved existence of sparse subnetworks that train to full accuracy. | ICLR (Best Paper) |
| 3 | LeCun, Y., Denker, J., & Solla, S. (1989). *Optimal brain damage.* | Introduced Hessian-based importance for principled pruning. | NeurIPS |
| 4 | Hassibi, B., & Stork, D. (1993). *Second order derivatives for network pruning: Optimal brain surgeon.* | Extended OBD with weight compensation for improved accuracy. | NeurIPS |

### 9.2 Modern LLM Pruning

| # | Citation | Key Contribution | Venue |
|---|----------|-----------------|-------|
| 5 | Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). *A simple and effective pruning approach for large language models.* | Wanda: activation-weighted pruning without fine-tuning. | arXiv:2306.11695 |
| 6 | Frantar, E., & Alistarh, D. (2023). *SparseGPT: Massive language models can be accurately pruned in one-shot.* | One-shot 50% sparsity with <1 perplexity increase on OPT-175B. | ICML |
| 7 | Frantar, E., & Alistarh, D. (2022). *Optimal brain compression: A framework for accurate post-training quantization and pruning.* | Unified framework for quantization and pruning via OBS. | NeurIPS |
| 8 | Muralidharan, S., et al. (2024). *Compact language models via pruning and knowledge distillation.* | Minitron: depth/width pruning with distillation. | arXiv:2407.14679 |

### 9.3 Structured Pruning

| # | Citation | Key Contribution | Venue |
|---|----------|-----------------|-------|
| 9 | Ma, X., et al. (2023). *LLM-Pruner: On the structural pruning of large language models.* | Graph-based dependency tracking for consistent structured pruning. | NeurIPS |
| 10 | Xia, M., et al. (2023). *Sheared LLaMA: Accelerating language model pre-training via structured pruning.* | Combined structured pruning with continued pretraining. | arXiv:2310.06694 |
| 11 | Men, X., et al. (2024). *ShortGPT: Layers in large language models are more redundant than you expect.* | Block Importance metric for layer removal. | arXiv:2403.03853 |

### 9.4 Sparsity Patterns and Hardware

| # | Citation | Key Contribution | Venue |
|---|----------|-----------------|-------|
| 12 | Zhou, A., et al. (2021). *Learning N:M fine-grained structured sparse neural networks from scratch.* | Training-time N:M sparsity with SR-STE. | ICLR |
| 13 | Mishra, A., et al. (2021). *Accelerating sparse deep neural networks.* | NVIDIA Ampere 2:4 structured sparsity hardware support. | arXiv:2104.08378 |
| 14 | Pool, J., & Yu, C. (2021). *Channel permutations for N:M sparsity.* | Channel reordering to improve N:M accuracy. | NeurIPS |

### 9.5 Pruning Schedules and Theory

| # | Citation | Key Contribution | Venue |
|---|----------|-----------------|-------|
| 15 | Zhu, M., & Gupta, S. (2017). *To prune, or not to prune: Exploring the efficacy of pruning for model compression.* | Cubic sparsity schedule outperforms one-shot. | ICLR Workshop |
| 16 | Gale, T., Elsen, E., & Hooker, S. (2019). *The state of sparsity in deep neural networks.* | Comprehensive benchmark of pruning methods. | arXiv:1902.09574 |
| 17 | Blalock, D., et al. (2020). *What is the state of neural network pruning?* | Meta-analysis of 81 pruning papers; standardized evaluation. | MLSys |

---

## 10. 100-Point Popperian Falsification QA Checklist

> "The criterion of the scientific status of a theory is its falsifiability."
> — Karl Popper, *The Logic of Scientific Discovery*

This checklist is designed to **falsify** claims about the pruning implementation. Each item attempts to find a counterexample or failure case. A robust implementation should survive all falsification attempts.

### 10.1 Numerical Stability (Items 1-15)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 1 | Apply magnitude pruning to a layer with all-zero weights. | Should return uniform importance (all zeros) without NaN. | [ ] |
| 2 | Apply Wanda to a layer where calibration activations are all zeros. | Should return error or handle gracefully (not NaN). | [ ] |
| 3 | Compute SparseGPT Hessian with perfectly collinear inputs. | Should detect singular matrix and apply damping. | [ ] |
| 4 | Prune a layer to 99.9% sparsity. | Should maintain numerical stability in remaining weights. | [ ] |
| 5 | Apply importance scoring to FP16 weights near subnormal range. | Should not overflow/underflow. | [ ] |
| 6 | Compute block importance with very large activation values (1e30). | Should normalize or handle overflow. | [ ] |
| 7 | Apply N:M mask generation with scores containing -Inf. | Should error before propagating invalid values. | [ ] |
| 8 | Run calibration with a single sample (n=1). | Should warn about statistical instability. | [ ] |
| 9 | Apply Wanda with activation norms of exactly zero for some channels. | Should handle division by zero. | [ ] |
| 10 | Compute SparseGPT with Hessian condition number > 1e10. | Should apply adaptive damping. | [ ] |
| 11 | Apply pruning to a tensor with shape [1, 1]. | Should handle degenerate dimensions. | [ ] |
| 12 | Run iterative pruning with target sparsity of exactly 0.0. | Should be no-op, not error. | [ ] |
| 13 | Run iterative pruning with target sparsity of exactly 1.0. | Should zero all weights cleanly. | [ ] |
| 14 | Apply magnitude importance to weights with values {-1e-38, 1e-38}. | Should distinguish magnitudes correctly. | [ ] |
| 15 | Compute Welford update with count overflow (>2^63 samples). | Should handle or error gracefully. | [ ] |

### 10.2 Shape and Dimension Handling (Items 16-30)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 16 | Apply 2:4 mask to a layer with 7 elements (not divisible by 4). | Should return clear error message. | [ ] |
| 17 | Apply block sparsity with block size larger than layer dimensions. | Should return clear error message. | [ ] |
| 18 | Apply column pruning to a 1D tensor. | Should return shape error. | [ ] |
| 19 | Apply a mask with wrong shape to a layer. | Should fail at application time with shape mismatch. | [ ] |
| 20 | Compute importance for an empty module (no parameters). | Should return empty scores, not panic. | [ ] |
| 21 | Apply depth pruning to a model with only 1 layer. | Should warn/error if num_layers_to_remove >= 1. | [ ] |
| 22 | Apply width pruning targeting more hidden dims than exist. | Should clamp or error clearly. | [ ] |
| 23 | Generate N:M mask with N=0 (prune everything). | Should be valid (all zeros). | [ ] |
| 24 | Generate N:M mask with N=M (prune nothing). | Should be valid (all ones). | [ ] |
| 25 | Apply mask to a view/slice of a tensor. | Should handle non-contiguous memory. | [ ] |
| 26 | Compute importance for a layer with shape [0, 512]. | Should handle zero batch dimension. | [ ] |
| 27 | Apply pruning to a quantized (INT8) tensor. | Should dequantize, prune, requantize or error. | [ ] |
| 28 | Generate block mask with height=1, width=1 (equivalent to unstructured). | Should produce same result as unstructured. | [ ] |
| 29 | Apply structured pruning removing more heads than exist. | Should error with clear message. | [ ] |
| 30 | Compute activation stats for 3D/4D tensors (conv layers). | Should reduce to per-channel stats correctly. | [ ] |

### 10.3 Algorithm Correctness (Items 31-50)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 31 | Verify magnitude L1 importance equals `|w|`. | Exact equality check. | [ ] |
| 32 | Verify magnitude L2 importance equals `w^2`. | Exact equality check. | [ ] |
| 33 | Verify Wanda importance equals `|w| * sqrt(act_norm)`. | Exact equality check (within FP tolerance). | [ ] |
| 34 | Verify 2:4 mask has exactly 2 non-zeros per 4 elements. | Structural verification for all groups. | [ ] |
| 35 | Verify pruning at 50% sparsity removes ~50% of weights. | Within 1% tolerance for unstructured. | [ ] |
| 36 | Verify SparseGPT weight update formula matches paper. | Compare with reference implementation. | [ ] |
| 37 | Verify block importance equals 1 - cosine_similarity. | Exact formula verification. | [ ] |
| 38 | Verify global pruning threshold selects correct quantile. | Statistical test across random inputs. | [ ] |
| 39 | Verify activation stats are unbiased estimators. | Compare with batch computation. | [ ] |
| 40 | Verify Welford online algorithm matches offline computation. | Exact match within FP precision. | [ ] |
| 41 | Verify pruning preserves zero weights (doesn't introduce non-zeros). | Mask should only remove, not add. | [ ] |
| 42 | Verify iterative and one-shot depth pruning remove same layers given same scores. | Deterministic result. | [ ] |
| 43 | Verify channel pruning maintains head-divisibility constraint. | Post-prune dim divisible by num_heads. | [ ] |
| 44 | Verify SparseGPT Hessian inverse update is numerically stable. | Compare with direct inversion. | [ ] |
| 45 | Verify cubic schedule formula: `s_f * (1 - (1 - t/T)^3)`. | Exact match at 0, T/2, T. | [ ] |
| 46 | Verify gradual schedule is monotonically increasing. | No sparsity decrease over time. | [ ] |
| 47 | Verify importance scores are non-negative for magnitude methods. | No negative importance. | [ ] |
| 48 | Verify mask application is idempotent: `apply(apply(w, m), m) == apply(w, m)`. | Exact equality. | [ ] |
| 49 | Verify pruning a pre-pruned layer doesn't double-count zeros. | Effective sparsity is union. | [ ] |
| 50 | Verify N:M constraint holds after applying mask to arbitrary input. | Structural check. | [ ] |

### 10.4 Edge Cases and Boundaries (Items 51-65)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 51 | Prune at exactly the step boundary in gradual schedule. | Correct sparsity at boundary. | [ ] |
| 52 | Call importance computation with no calibration for Wanda. | Clear error: calibration required. | [ ] |
| 53 | Prune with empty calibration dataset. | Error with helpful message. | [ ] |
| 54 | Apply two different masks sequentially to same layer. | Masks should compose correctly. | [ ] |
| 55 | Interrupt calibration mid-batch and resume. | Partial stats should be valid. | [ ] |
| 56 | Prune a frozen (requires_grad=false) layer. | Should still prune weights. | [ ] |
| 57 | Prune a shared weight tensor (used in multiple places). | Pruning should affect all uses. | [ ] |
| 58 | Apply pruning during training (not just inference). | Gradients should flow through mask. | [ ] |
| 59 | Serialize and deserialize a pruned model. | Sparsity should be preserved. | [ ] |
| 60 | Load a pruned model into unpruned architecture. | Shape mismatch error for structured. | [ ] |
| 61 | Compute importance on CPU, apply mask on GPU. | Should handle device transfer. | [ ] |
| 62 | Prune with different dtypes (FP32 importance, FP16 weights). | Should cast correctly. | [ ] |
| 63 | Apply pruning concurrently from multiple threads. | Should be thread-safe or error. | [ ] |
| 64 | Prune a model with no attention layers (pure MLP). | Depth pruning should still work. | [ ] |
| 65 | Prune with calibration data longer than model max length. | Should truncate or error. | [ ] |

### 10.5 Performance and Resource Bounds (Items 66-80)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 66 | Prune a 7B model in <10GB memory. | Memory stays within budget. | [ ] |
| 67 | Compute SparseGPT Hessian for 4096-dim layer in <1min. | Time bound verified. | [ ] |
| 68 | Generate N:M mask for 1B elements in <1s. | Time bound verified. | [ ] |
| 69 | Calibration with 128 samples completes in <5min for 7B model. | Time bound verified. | [ ] |
| 70 | Iterative depth pruning (10 iterations) completes in <30min. | Time bound verified. | [ ] |
| 71 | Memory usage during pruning doesn't exceed 2x model size. | Peak memory tracked. | [ ] |
| 72 | Importance computation is O(n) for magnitude methods. | Benchmark verified. | [ ] |
| 73 | SparseGPT block-wise processing reduces memory vs full matrix. | Memory comparison. | [ ] |
| 74 | Pruning does not cause GPU OOM on 24GB GPU for 13B model. | With gradient checkpointing. | [ ] |
| 75 | Calibration data loading doesn't bottleneck GPU. | Prefetch pipeline verified. | [ ] |
| 76 | Mask generation is parallelizable across layers. | Parallel speedup measured. | [ ] |
| 77 | SparseGPT Cholesky is faster with SIMD/trueno. | Benchmark comparison. | [ ] |
| 78 | Structured pruning produces smaller model file. | File size reduction verified. | [ ] |
| 79 | Inference speedup with 2:4 sparsity on Ampere. | >1.5x speedup measured. | [ ] |
| 80 | Inference speedup with block pruning on CPU. | >1.2x speedup measured. | [ ] |

### 10.6 Integration and Compatibility (Items 81-90)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 81 | Export pruned model to GGUF format. | Valid GGUF file produced. | [ ] |
| 82 | Export pruned model to SafeTensors format. | Valid SafeTensors file produced. | [ ] |
| 83 | Load pruned model in llama.cpp. | Inference runs correctly. | [ ] |
| 84 | Fine-tune pruned model with entrenar. | Training converges. | [ ] |
| 85 | Apply quantization after pruning. | Combined compression works. | [ ] |
| 86 | Apply LoRA to pruned model. | LoRA adapters attach correctly. | [ ] |
| 87 | Prune a model loaded from HuggingFace Hub. | HF model compatibility. | [ ] |
| 88 | Prune with distributed training (multi-GPU). | Masks synchronized correctly. | [ ] |
| 89 | Use pruning callback with entrenar's existing callbacks. | Callback composition works. | [ ] |
| 90 | Generate pruning config from YAML. | Config parsing validates correctly. | [ ] |

### 10.7 Documentation and Error Messages (Items 91-100)

| # | Falsification Attempt | Expected Outcome | Status |
|---|----------------------|------------------|--------|
| 91 | Every public function has doc comments. | `cargo doc` succeeds without warnings. | [ ] |
| 92 | Error messages include actionable information. | User can diagnose from message alone. | [ ] |
| 93 | Examples in docs compile and run. | `cargo test --doc` passes. | [ ] |
| 94 | README includes quickstart example. | Copy-paste example works. | [ ] |
| 95 | Changelog documents breaking changes. | Version migration path clear. | [ ] |
| 96 | Type signatures are documented for public traits. | Trait docs include all methods. | [ ] |
| 97 | Configuration options are documented with defaults. | All config fields documented. | [ ] |
| 98 | Performance characteristics documented. | Complexity noted for each algorithm. | [ ] |
| 99 | Citation information included for each algorithm. | Academic attribution complete. | [ ] |
| 100 | Security considerations documented. | No known vulnerabilities. | [ ] |

---

## 11. PMAT Compliance

### 11.1 Configuration

```toml
# pmat.toml additions for pruning module

[quality]
# Existing thresholds apply
max_cyclomatic_complexity = 10
max_cognitive_complexity = 15
min_test_coverage = 95
max_satd_comments = 0
min_tdg_score = 95

[analysis]
include = [
    "src/**/*.rs",
    "src/pruning/**/*.rs",  # New pruning module
]
exclude = ["target/**", "benches/**"]

[pruning_specific]
# Custom rules for pruning code
max_numerical_operations_per_function = 50  # Limit complexity of math
require_stability_validation = true          # All math must check NaN/Inf
require_shape_validation = true              # All tensor ops must validate shapes
```

### 11.2 Quality Gates for Pruning

```rust
/// PMAT quality gate for pruning module.
#[derive(Debug)]
pub struct PruningPmatGate {
    /// Minimum test coverage for pruning module.
    pub min_coverage: f32,
    /// Maximum cyclomatic complexity per function.
    pub max_complexity: usize,
    /// Require numerical stability checks.
    pub require_stability_checks: bool,
}

impl Default for PruningPmatGate {
    fn default() -> Self {
        Self {
            min_coverage: 0.95,     // 95% coverage
            max_complexity: 10,     // Max 10 cyclomatic complexity
            require_stability_checks: true,
        }
    }
}

impl PruningPmatGate {
    pub fn validate(&self) -> Result<(), PmatError> {
        // Run coverage check
        let coverage = run_coverage("src/pruning")?;
        if coverage < self.min_coverage {
            return Err(PmatError::CoverageBelowThreshold {
                expected: self.min_coverage,
                actual: coverage,
            });
        }

        // Run complexity analysis
        let functions = analyze_complexity("src/pruning")?;
        for func in &functions {
            if func.cyclomatic > self.max_complexity {
                return Err(PmatError::ComplexityExceeded {
                    function: func.name.clone(),
                    expected: self.max_complexity,
                    actual: func.cyclomatic,
                });
            }
        }

        // Check for stability validations
        if self.require_stability_checks {
            let missing = find_unvalidated_math("src/pruning")?;
            if !missing.is_empty() {
                return Err(PmatError::MissingStabilityChecks {
                    locations: missing,
                });
            }
        }

        Ok(())
    }
}
```

### 11.3 Technical Debt Gradient (TDG) Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Test Coverage | ≥95% | `cargo llvm-cov` |
| Mutation Kill Rate | ≥80% | `cargo mutants` |
| Cyclomatic Complexity | ≤10 | `pmat analyze` |
| Cognitive Complexity | ≤15 | `pmat analyze` |
| SATD Comments | 0 | Grep for TODO/FIXME/HACK |
| Documentation Coverage | 100% | `cargo doc --document-private-items` |
| Clippy Warnings | 0 | `cargo clippy -- -D warnings` |

---

## 12. Enhanced Testing with Probador

### 12.1 Integration with jugar-probar

```rust
// tests/pruning_visual_tests.rs

use jugar_probar::snapshot::{expect_snapshot, SnapshotConfig};
use aprender::pruning::{MagnitudeImportance, Importance, SparsityMask};
use aprender::nn::Linear;

/// Visual snapshot tests for importance score distributions.
#[test]
fn test_magnitude_importance_distribution() {
    let layer = Linear::new(512, 256);
    let importance = MagnitudeImportance { norm: 2, scope: ImportanceScope::Global };

    let scores = importance.compute(&layer, None).unwrap();

    // Snapshot the distribution statistics
    let snapshot = serde_json::json!({
        "method": scores.method,
        "min": scores.stats.min,
        "max": scores.stats.max,
        "mean": scores.stats.mean,
        "std": scores.stats.std,
        "shape": scores.values.shape(),
    });

    expect_snapshot(&snapshot, SnapshotConfig::default())
        .with_name("magnitude_l2_512x256")
        .assert_matches();
}

/// Visual test for mask patterns.
#[test]
fn test_nm_mask_pattern_visual() {
    let scores = Tensor::rand(&[8, 8]);
    let mask = generate_nm_mask(&scores, 2, 4).unwrap();

    // Visualize mask as ASCII art for snapshot
    let visual = mask_to_ascii(&mask.mask);

    expect_snapshot(&visual, SnapshotConfig::default())
        .with_name("nm_2_4_mask_8x8")
        .assert_matches();
}

fn mask_to_ascii(mask: &Tensor) -> String {
    let [rows, cols] = mask.shape()[..] else { panic!() };
    let mut result = String::new();

    for r in 0..rows {
        for c in 0..cols {
            result.push(if mask[[r, c]] > 0.5 { '█' } else { '░' });
        }
        result.push('\n');
    }

    result
}
```

### 12.2 Property-Based Testing

```rust
// tests/pruning_properties.rs

use proptest::prelude::*;
use aprender::pruning::*;

proptest! {
    /// Property: Magnitude importance is always non-negative.
    #[test]
    fn magnitude_importance_non_negative(
        weights in prop::collection::vec(-1000.0f32..1000.0, 1..1000)
    ) {
        let tensor = Tensor::from_vec(weights.clone(), &[weights.len()]);
        let layer = MockLayer::with_weights(tensor);
        let importance = MagnitudeImportance { norm: 2, scope: ImportanceScope::Global };

        let scores = importance.compute(&layer, None).unwrap();

        prop_assert!(scores.values.iter().all(|&v| v >= 0.0));
    }

    /// Property: N:M mask has exactly N non-zeros per M elements.
    #[test]
    fn nm_mask_structure_valid(
        n in 1usize..4,
        m in 2usize..8,
        size in 1usize..100,
    ) {
        prop_assume!(n < m);
        let total = size * m;

        let scores = Tensor::rand(&[total]);
        let mask = generate_nm_mask(&scores, n, m).unwrap();

        // Check each group of M elements
        for group_start in (0..total).step_by(m) {
            let group_sum: f32 = (0..m)
                .map(|i| mask.mask[[group_start + i]])
                .sum();
            prop_assert_eq!(group_sum as usize, n, "Group at {} has wrong count", group_start);
        }
    }

    /// Property: Pruning reduces or maintains parameter count.
    #[test]
    fn pruning_reduces_parameters(
        target_sparsity in 0.0f32..1.0,
        size in 10usize..1000,
    ) {
        let weights = Tensor::rand(&[size]);
        let mut layer = MockLayer::with_weights(weights.clone());
        let initial_nonzero = weights.iter().filter(|&&v| v != 0.0).count();

        let pruner = MagnitudePruner::new(target_sparsity);
        pruner.prune(&mut layer, target_sparsity, SparsityPattern::Unstructured, None).unwrap();

        let final_nonzero = layer.weight().unwrap().iter().filter(|&&v| v != 0.0).count();

        prop_assert!(final_nonzero <= initial_nonzero);
    }

    /// Property: Mask application is idempotent.
    #[test]
    fn mask_application_idempotent(
        weights in prop::collection::vec(-100.0f32..100.0, 10..100),
        mask_vals in prop::collection::vec(prop::bool::ANY, 10..100),
    ) {
        prop_assume!(weights.len() == mask_vals.len());

        let mut tensor1 = Tensor::from_vec(weights.clone(), &[weights.len()]);
        let mut tensor2 = tensor1.clone();
        let mask_tensor = Tensor::from_vec(
            mask_vals.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            &[mask_vals.len()]
        );
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        mask.apply(&mut tensor1).unwrap();
        mask.apply(&mut tensor1).unwrap();  // Apply twice
        mask.apply(&mut tensor2).unwrap();  // Apply once

        prop_assert_eq!(tensor1.to_vec(), tensor2.to_vec());
    }
}
```

### 12.3 Mutation Testing Configuration

```toml
# .cargo-mutants.toml additions

[[mutants]]
path = "src/pruning/**/*.rs"
timeout_multiplier = 3.0  # Numerical code may be slow

# Mutations to apply
mutations = [
    "arithmetic",      # +, -, *, /
    "comparison",      # <, >, <=, >=, ==, !=
    "boundary",        # off-by-one errors
    "negation",        # !condition
    "return_value",    # return different values
]

# Functions to skip (known to be trivial)
skip_functions = [
    "name",            # Trait method returning static str
    "requires_calibration",  # Simple bool return
]

# Minimum kill rate
min_kill_rate = 0.80
```

---

## 13. Tracing with Renacer

### 13.1 Performance Assertions

```toml
# renacer.toml additions for pruning

[[assertion]]
name = "pruning_importance_latency"
type = "critical_path"
max_duration_ms = 5000  # Importance computation <5s per layer
fail_on_violation = true
enabled = true

[[assertion]]
name = "pruning_mask_generation_latency"
type = "critical_path"
max_duration_ms = 1000  # Mask generation <1s
fail_on_violation = true
enabled = true

[[assertion]]
name = "pruning_memory_budget"
type = "memory_usage"
max_bytes = 2147483648  # 2GB maximum for pruning operations
tracking_mode = "peak"
fail_on_violation = true
enabled = true

[[assertion]]
name = "calibration_syscall_budget"
type = "span_count"
max_spans = 5000  # Calibration shouldn't be I/O bound
fail_on_violation = false  # Warning only
enabled = true

# Pruning-specific anti-patterns
[[assertion]]
name = "detect_redundant_computation"
type = "anti_pattern"
pattern = "RedundantComputation"
threshold = 0.7  # Detect recomputing same importance twice
fail_on_violation = false
enabled = true

[[assertion]]
name = "detect_memory_thrashing"
type = "anti_pattern"
pattern = "MemoryThrashing"
threshold = 0.8  # Detect excessive allocations during pruning
fail_on_violation = false
enabled = true
```

### 13.2 Instrumentation

```rust
// src/pruning/instrumentation.rs

use renacer::{span, event, Level};

impl<I: Importance> InstrumentedImportance<I> {
    #[span(level = Level::INFO, name = "importance_computation")]
    pub fn compute_instrumented(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        event!(Level::DEBUG,
            method = self.inner.name(),
            module = module.name(),
            has_context = context.is_some(),
        );

        let scores = self.inner.compute(module, context)?;

        event!(Level::INFO,
            importance_min = scores.stats.min,
            importance_max = scores.stats.max,
            importance_mean = scores.stats.mean,
        );

        Ok(scores)
    }
}

impl<P: Pruner> InstrumentedPruner<P> {
    #[span(level = Level::INFO, name = "pruning_operation")]
    pub fn prune_instrumented(
        &self,
        module: &mut dyn Module,
        target_sparsity: f32,
        pattern: SparsityPattern,
        context: Option<&CalibrationContext>,
    ) -> Result<PruningResult, PruningError> {
        event!(Level::INFO,
            target_sparsity = target_sparsity,
            pattern = ?pattern,
            module = module.name(),
        );

        let result = self.inner.prune(module, target_sparsity, pattern, context)?;

        event!(Level::INFO,
            achieved_sparsity = result.achieved_sparsity,
            parameters_pruned = result.parameters_pruned,
            memory_savings_bytes = result.memory_savings_bytes,
        );

        Ok(result)
    }
}
```

### 13.3 Golden Trace Validation

```rust
// tests/pruning_golden_traces.rs

use renacer::golden::{GoldenTrace, compare_traces};

#[test]
fn test_wanda_pruning_trace_stability() {
    let trace = GoldenTrace::capture(|| {
        let model = load_test_model();
        let calibration = load_calibration_data();

        let pruner = WandaPruner::new(0.5, SparsityPattern::Unstructured);
        pruner.prune(&mut model, &calibration).unwrap();
    });

    compare_traces(
        &trace,
        "golden_traces/wanda_pruning_baseline.json",
        TraceCompareConfig {
            allow_timing_variance: 0.2,  // 20% timing variance allowed
            require_same_calls: true,
            require_same_order: true,
        }
    ).expect("Trace should match golden baseline");
}
```

---

## 14. Implementation Roadmap

### Phase 1: Core Primitives (Aprender)

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| `ImportanceScores` struct | Low | None |
| `SparsityMask` struct with validation | Medium | None |
| `Importance` trait definition | Low | None |
| `Pruner` trait definition | Low | Importance trait |
| `MagnitudeImportance` implementation | Low | Importance trait |
| Unit tests for Phase 1 | Medium | All above |

### Phase 2: Activation-Based Methods (Aprender)

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| `CalibrationContext` struct | Medium | None |
| `ActivationStats` with Welford update | Medium | None |
| `WandaImportance` implementation | Medium | CalibrationContext |
| Forward hook infrastructure for activation capture | High | Module trait |
| Unit tests for Phase 2 | Medium | All above |

### Phase 3: Second-Order Methods (Aprender)

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| Hessian computation from activations | High | CalibrationContext |
| Cholesky decomposition (trueno integration) | Medium | trueno |
| `SparseGPTImportance` implementation | High | Hessian, Cholesky |
| Block-wise OBS update | High | SparseGPTImportance |
| Unit tests for Phase 3 | High | All above |

### Phase 4: Structured Pruning (Aprender)

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| N:M mask generation | Medium | SparsityMask |
| Block mask generation | Medium | SparsityMask |
| `DepthPruner` (Minitron) | Medium | Block importance |
| `WidthPruner` (Minitron) | High | Channel importance |
| Dependency graph for consistent pruning | High | Module introspection |
| Unit tests for Phase 4 | High | All above |

### Phase 5: Training Integration (Entrenar)

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| `PruningSchedule` enum | Low | None |
| `PruningCallback` implementation | Medium | Callback trait |
| Calibration data loader integration | Medium | DataLoader |
| YAML configuration parsing | Medium | Config system |
| Fine-tuning pipeline integration | Medium | Trainer |
| Integration tests for Phase 5 | High | All above |

### Phase 6: Quality Assurance

| Task | Complexity | Dependencies |
|------|-----------|--------------|
| 100-point falsification checklist completion | High | All phases |
| PMAT gate implementation | Medium | PMAT infrastructure |
| Probador snapshot tests | Medium | jugar-probar |
| Property-based tests | High | proptest |
| Mutation testing validation | High | cargo-mutants |
| Renacer performance assertions | Medium | renacer |
| Golden trace baselines | Medium | renacer |
| Documentation completion | Medium | All phases |

---

## 15. References

### Primary Sources

1. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. *Advances in Neural Information Processing Systems*, 28.
2. Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2016). Pruning convolutional neural networks for resource efficient inference. *ICLR*.
3. Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance estimation for neural network pruning. *CVPR*.
4. Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*.
5. Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive language models can be accurately pruned in one-shot. *International Conference on Machine Learning*.
6. Muralidharan, S., et al. (2024). Compact language models via pruning and knowledge distillation. *arXiv preprint arXiv:2407.14679*.
7. Ma, X., et al. (2023). LLM-Pruner: On the structural pruning of large language models. *Advances in Neural Information Processing Systems*.
8. Xia, M., Zhong, Z., & Chen, D. (2023). Sheared LLaMA: Accelerating language model pre-training via structured pruning. *arXiv preprint arXiv:2310.06694*.
9. Men, X., et al. (2024). ShortGPT: Layers in large language models are more redundant than you expect. *arXiv preprint arXiv:2403.03853*.
10. Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *International Conference on Learning Representations*.
11. LeCun, Y., Denker, J., & Solla, S. (1989). Optimal brain damage. *Advances in Neural Information Processing Systems*, 2.
12. Hassibi, B., & Stork, D. (1993). Second order derivatives for network pruning: Optimal brain surgeon. *Advances in Neural Information Processing Systems*, 5.

### Structured Sparsity & Hardware

13. Zhou, A., et al. (2021). Learning N:M fine-grained structured sparse neural networks from scratch. *ICLR*.
14. Mishra, A., et al. (2021). Accelerating sparse deep neural networks. *arXiv preprint arXiv:2104.08378*.
15. Pool, J., & Yu, C. (2021). Channel permutations for N:M sparsity. *Advances in Neural Information Processing Systems*.

### Methodology & Analysis

16. Zhu, M., & Gupta, S. (2017). To prune, or not to prune: Exploring the efficacy of pruning for model compression. *ICLR Workshop*.
17. Gale, T., Elsen, E., & Hooker, S. (2019). The state of sparsity in deep neural networks. *arXiv preprint arXiv:1902.09574*.
18. Blalock, D., et al. (2020). What is the state of neural network pruning? *MLSys*.

### Toyota Way & QA References

19. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.
20. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.
21. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
22. Beck, K. (2002). *Test-Driven Development: By Example*. Addison-Wesley.
23. Claessen, K., & Hughes, J. (2000). QuickCheck: A lightweight tool for random testing of Haskell programs. *ACM SIGPLAN Notices*, 35(9), 268-279.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-05 | PAIML Research | Initial specification |

---

**Status:** DRAFT - AWAITING REVIEW

**Next Steps:**
1. Review by stakeholders
2. Prototype Phase 1 implementation
3. Validate falsification checklist items
4. Iterate based on feedback
