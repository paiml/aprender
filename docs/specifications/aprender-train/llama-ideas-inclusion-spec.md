# LLaMA Integration & Quality Enhancement Specification

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Proposed
**Authors:** Pragmatic AI Labs

## Executive Summary

This specification outlines the integration of LLaMA transformer architecture into entrenar's training infrastructure,
enhanced with quality methodologies from certeza, PMAT roadmap workflows, and renacer tracing capabilities. The goal is
to provide a complete, production-ready example of training and fine-tuning transformer models using entrenar's
autograd, optimizers, and LoRA/QLoRA features.

## 1. LLaMA Architecture Integration

### 1.1 Reference Implementation (from llama2.rs)

**Objective:** Create `entrenar/examples/llama2/` demonstrating complete LLaMA 2 architecture using entrenar primitives.

**Components:**

```rust
// examples/llama2/architecture.rs
use entrenar::{Tensor, autograd::ops::{matmul, layer_norm, gelu}};

pub struct LLaMAConfig {
    pub vocab_size: usize,      // 32000 (default)
    pub hidden_size: usize,     // 4096 (7B model)
    pub num_layers: usize,      // 32 (7B model)
    pub num_heads: usize,       // 32
    pub intermediate_size: usize, // 11008
    pub max_seq_len: usize,     // 2048
}

pub struct LLaMALayer {
    // Self-attention
    pub q_proj: Tensor,  // Linear(hidden_size, hidden_size)
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,

    // Feed-forward
    pub gate_proj: Tensor,  // Linear(hidden_size, intermediate_size)
    pub up_proj: Tensor,
    pub down_proj: Tensor,

    // Layer norms
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
}

impl LLaMALayer {
    /// Forward pass using entrenar ops
    pub fn forward(&self, x: &Tensor, config: &LLaMAConfig) -> Tensor {
        // Pre-norm architecture (RMS norm in actual LLaMA)
        let normed = layer_norm(x);

        // Multi-head attention
        let q = matmul(&self.q_proj, &normed, config.hidden_size, config.hidden_size, 1);
        let k = matmul(&self.k_proj, &normed, config.hidden_size, config.hidden_size, 1);
        let v = matmul(&self.v_proj, &normed, config.hidden_size, config.hidden_size, 1);

        let attn_output = scaled_dot_product_attention(&q, &k, &v, config.num_heads);
        let attn_output = matmul(&self.o_proj, &attn_output, config.hidden_size, config.hidden_size, 1);

        // Residual connection
        let hidden = x + &attn_output;

        // Feed-forward with SwiGLU activation
        let normed = layer_norm(&hidden);
        let gate = matmul(&self.gate_proj, &normed, config.intermediate_size, config.hidden_size, 1);
        let up = matmul(&self.up_proj, &normed, config.intermediate_size, config.hidden_size, 1);
        let ffn_output = &gelu(&gate) * &up;  // SwiGLU approximation
        let ffn_output = matmul(&self.down_proj, &ffn_output, config.hidden_size, config.intermediate_size, 1);

        // Residual connection
        hidden + &ffn_output
    }
}
```

**Deliverables:**
1. `examples/llama2/architecture.rs` - Complete model definition
2. `examples/llama2/train.rs` - Training from scratch
3. `examples/llama2/finetune_lora.rs` - LoRA fine-tuning
4. `examples/llama2/finetune_qlora.rs` - Memory-efficient QLoRA fine-tuning
5. `examples/llama2/README.md` - Documentation with memory benchmarks

**Success Criteria:**
- Train toy LLaMA (124M params) on 1 GPU
- Fine-tune with LoRA: 99.9% parameter reduction
- Fine-tune with QLoRA: 75% memory reduction
- All examples tested in CI/CD

---

## 2. Certeza Quality Annotations (10 Peer-Reviewed Ideas)

### 2.1 Three-Tiered Testing for Transformer Training

**Annotation 1: Tiered Feedback Loops**

```rust
// Makefile targets adapted from certeza
tier1: ## ON-SAVE: Fast gradient checks (<5s)
    cargo test --lib gradient_
    cargo clippy --all-targets -- -D warnings

tier2: ## ON-COMMIT: Full property tests (1-5min)
    cargo test --all
    cargo llvm-cov --all-features --workspace

tier3: ## ON-MERGE: Mutation testing (hours)
    cargo mutants --file src/autograd/ops.rs
    cargo mutants --file src/lora/layer.rs
```

**Rationale:** Different verification techniques at different time scales prevent context-switching waste. LLaMA
examples should have fast gradient checks (tier1) for inner loop development, full integration tests (tier2) for
commits, and mutation testing (tier3) for merge gates.

**Implementation:** Add `examples/llama2/Makefile` with tiered targets.

---

### 2.2 Property-Based Testing for Transformer Ops

**Annotation 2: Attention Mechanism Properties**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn attention_output_shape_invariant(
        batch_size in 1usize..=8,
        seq_len in 1usize..=128,
        hidden_size in 64usize..=512,
    ) {
        let q = Tensor::randn(vec![batch_size * seq_len * hidden_size], true);
        let k = Tensor::randn(vec![batch_size * seq_len * hidden_size], true);
        let v = Tensor::randn(vec![batch_size * seq_len * hidden_size], true);

        let output = scaled_dot_product_attention(&q, &k, &v, 8);

        // Shape invariant: output.shape == input.shape
        assert_eq!(output.data().len(), q.data().len());
    }

    #[test]
    fn attention_is_permutation_equivariant(seq_len in 2usize..=16) {
        // Property: Permuting input sequence permutes output identically
        let q = Tensor::randn(vec![seq_len * 64], true);
        let k = q.clone();
        let v = q.clone();

        let output1 = scaled_dot_product_attention(&q, &k, &v, 8);

        // Permute inputs
        let (q_perm, k_perm, v_perm) = permute_sequence(&q, &k, &v, &[1, 0, 2, 3]);
        let output2 = scaled_dot_product_attention(&q_perm, &k_perm, &v_perm, 8);

        // Outputs should be identically permuted
        assert_permutation_equal(&output1, &output2, &[1, 0, 2, 3]);
    }
}
```

**Rationale:** Transformers have deep mathematical properties (equivariance, shape preservation) that property-based
tests can verify across wide input ranges. Catches edge cases that unit tests miss.

**Implementation:** Add `tests/property_llama.rs` with 20+ properties for LLaMA components.

---

### 2.3 Risk-Based Verification Allocation

**Annotation 3: Focus on High-Risk Code**

| Component | Risk Level | Verification Strategy |
|-----------|------------|----------------------|
| **Attention gradients** | Very High | Property tests + Coverage (100%) + Mutation (90%) + Gradient checking |
| **LoRA forward/backward** | High | Property tests + Coverage (95%) + Mutation (85%) |
| **Feed-forward layers** | Medium | Unit tests + Coverage (90%) + Mutation (80%) |
| **Embeddings** | Low | Unit tests + Coverage (85%) |

**Rationale:** From certeza line 436-441: "Spend 40% of verification time on the 5-10% highest-risk code." Attention
mechanism bugs are catastrophic (silent gradient errors); embedding bugs are obvious (immediate NaN).

**Implementation:**
- Run `cargo mutants --file src/autograd/attention.rs` nightly (tier3)
- Gradient checking on every attention backward pass test
- 100% coverage enforcement for attention.rs

---

### 2.4 Mutation Testing for Gradient Correctness

**Annotation 4: Mutation-Resistant Test Cases**

```rust
#[test]
fn test_attention_backward_mutation_resistant() {
    // Test case designed to catch common mutations:
    // 1. Off-by-one errors in matrix dimensions
    // 2. Sign flips in gradient computation
    // 3. Missing scaling factors

    let q = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let k = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);
    let v = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], true);

    let output = scaled_dot_product_attention(&q, &k, &v, num_heads=2);
    backward(&output);

    // Mutation: scale_factor = 1.0 (instead of 1/sqrt(d_k))
    // This test catches it because output changes significantly
    let expected_q_grad = vec![0.123, 0.456, ...];  // Pre-computed
    assert_approx_eq!(q.grad(), &expected_q_grad, epsilon=0.001);

    // Mutation: grad_k = grad_output @ Q (missing transpose)
    // This test catches it because shapes mismatch or values wrong
    let expected_k_grad = vec![0.789, 0.234, ...];
    assert_approx_eq!(k.grad(), &expected_k_grad, epsilon=0.001);
}
```

**Rationale:** Certeza achieves 97.7% mutation score by designing tests that kill common mutations. Attention backward
passes have subtle bugs (scaling, transpose) that standard tests miss.

**Implementation:** Add mutation-resistant tests for all autograd ops in `tests/mutation_resistant_attention.rs`.

---

### 2.5 Formal Verification for Invariants

**Annotation 5: Kani Proofs for Critical Properties**

```rust
#[cfg(kani)]
#[kani::proof]
fn verify_attention_shape_preservation() {
    let seq_len: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len > 0 && seq_len <= 16);  // Bounded model checking
    kani::assume(hidden_size > 0 && hidden_size <= 64);

    let q_data = vec![0.0; seq_len * hidden_size];
    let q = Tensor::from_vec(q_data, false);

    let output = self_attention(&q, &q, &q, num_heads=4);

    // Invariant: output.len() == input.len()
    kani::assert(output.data().len() == q.data().len(), "Shape must be preserved");
}
```

**Rationale:** From certeza line 451-458: Formal verification for 1-5% of highest-risk code. Attention shape bugs cause
silent errors; formally prove they can't happen.

**Implementation:** Add `src/autograd/attention_proofs.rs` with Kani proofs (tier3, nightly CI).

---

### 2.6 Chaos Engineering for Training Robustness

**Annotation 6: Training Under Adversarial Conditions**

```rust
use certeza::chaos::ChaosConfig;

#[test]
fn test_llama_training_under_memory_pressure() {
    // Simulate OOM conditions during training
    let chaos = ChaosConfig::new()
        .with_memory_limit(128 * 1024 * 1024)  // 128MB limit
        .with_timeout(Duration::from_secs(10))
        .build();

    chaos.run(|| {
        let model = LLaMAModel::new(config);
        let optimizer = AdamW::new(lr=0.001, weight_decay=0.01);

        // Training should gracefully handle OOM
        // Either: succeed with reduced batch size
        // Or: fail with clear error (not silent corruption)
        let result = train_step(&model, &batch, &optimizer);

        match result {
            Ok(_) => {},  // Success
            Err(e) if e.is_oom() => {},  // Expected failure
            Err(e) => panic!("Unexpected error: {}", e),
        }
    });
}
```

**Rationale:** Certeza line 276-305 shows chaos testing prevents silent failures. LLaMA training on GPUs hits OOM
frequently; chaos tests ensure graceful degradation.

**Implementation:** Add `tests/chaos_llama.rs` with memory/CPU limit tests (tier3).

---

### 2.7 Fuzz Testing for Tokenization

**Annotation 7: Discover Edge Cases in Input Processing**

```rust
// fuzz/fuzz_targets/tokenizer.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use entrenar_llama::tokenizer::Tokenizer;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let tokenizer = Tokenizer::new("tokenizer.json");

        // Should never panic, even on malicious input
        let tokens = tokenizer.encode(text);

        // Invariant: decode(encode(text)) should round-trip
        let decoded = tokenizer.decode(&tokens);

        // May differ in whitespace, but shouldn't crash
        let _ = decoded;
    }
});
```

**Rationale:** Certeza line 318-334 shows fuzz testing discovers edge cases. Tokenizers have complex state machines;
fuzzing finds Unicode/boundary bugs.

**Implementation:** Add `fuzz/fuzz_targets/` directory with tokenizer, model input fuzzing (tier3).

---

### 2.8 SIMD-Accelerated Test Data Generation

**Annotation 8: Fast Property Test Generation with Trueno**

```rust
use trueno::Vector;

proptest! {
    #[test]
    fn attention_softmax_normalization(seq_len in 4usize..=128) {
        // Use Trueno for fast test data generation
        let scores = Vector::randn(seq_len);

        let attention_weights = softmax(&scores);

        // Property: softmax outputs sum to 1.0
        let sum = trueno::sum(&attention_weights.data());
        assert!((sum - 1.0).abs() < 1e-5, "Softmax must normalize");

        // Property: all weights in [0, 1]
        for &w in attention_weights.data() {
            assert!(w >= 0.0 && w <= 1.0, "Softmax outputs in [0,1]");
        }
    }
}
```

**Rationale:** Certeza line 183 shows trueno integration for fast SIMD stats. Property tests generate millions of test
cases; SIMD speeds up test execution 3-10x.

**Implementation:** Use trueno::Vector in `tests/property_llama.rs` for test data generation.

---

### 2.9 Coverage-Driven Development

**Annotation 9: Zero Uncovered Branches in Critical Paths**

```bash
# Makefile target
coverage-critical:
    cargo llvm-cov --lcov --output-path lcov.info
    lcov --rc lcov_branch_coverage=1 \
         --extract lcov.info '*/src/autograd/attention.rs' \
         --extract lcov.info '*/src/lora/layer.rs' \
         -o critical.info
    genhtml critical.info -o coverage-report
    @echo "Critical path coverage:"
    @lcov --summary critical.info | grep -E 'lines|branches'
    @echo "Target: 100% lines, 95%+ branches"
```

**Rationale:** Certeza line 420-431 targets 95%+ coverage. LLaMA attention has critical conditional branches
(padding, causal masking); uncovered branches = production bugs.

**Implementation:** Add `make coverage-critical` target, fail CI if attention.rs coverage < 95%.

---

### 2.10 Toyota Way: Andon Cord for Training

**Annotation 10: Halt Training on Quality Regression**

```rust
pub struct TrainingMonitor {
    baseline_tdg_score: f32,
    current_tdg_score: f32,
}

impl TrainingMonitor {
    pub fn check_quality_regression(&self) -> Result<(), TrainingError> {
        if self.current_tdg_score < self.baseline_tdg_score - 5.0 {
            // Pull Andon Cord: Stop training immediately
            return Err(TrainingError::QualityRegression {
                baseline: self.baseline_tdg_score,
                current: self.current_tdg_score,
                message: "🚨 ANDON CORD: Training halted due to quality regression. \
                         Code quality dropped by >5 points. \
                         Fix quality issues before resuming training."
            });
        }
        Ok(())
    }
}
```

**Rationale:** Certeza line 222-245 applies Toyota Way principles. Long-running training jobs can mask quality
regressions; Andon Cord stops the line when defects detected.

**Implementation:** Add pre-training quality gate in `examples/llama2/train.rs` checking TDG baseline.

---

## 3. PMAT Roadmap Implementation Ideas

### 3.1 Automated Roadmap Generation

**Idea 1: PMAT-Driven Sprint Planning**

```yaml
# Generated by: pmat generate-roadmap --project entrenar-llama
roadmap_version: '1.0'
github_enabled: true
github_repo: paiml/entrenar
roadmap:
  - id: ENT-LLAMA-001
    item_type: epic
    title: "LLaMA Architecture Integration"
    status: planned
    priority: critical
    phases:
      - phase: "Phase 1: Core Architecture"
        subtasks:
          - "Implement LLaMALayer with entrenar ops"
          - "Add RoPE positional embeddings"
          - "Implement SwiGLU activation"
          - "Add causal attention masking"
        acceptance_criteria:
          - "All unit tests pass"
          - "Gradient checking passes (epsilon=1e-3)"
          - "Property tests verify equivariance"
      - phase: "Phase 2: Training Loop"
        subtasks:
          - "Implement data loading pipeline"
          - "Add training loop with AdamW"
          - "Implement learning rate scheduling"
          - "Add checkpointing"
        acceptance_criteria:
          - "Train 124M model to loss < 3.0"
          - "TDG score ≥ 85 (B+ grade)"
      - phase: "Phase 3: LoRA Fine-Tuning"
        subtasks:
          - "Apply LoRA to attention layers"
          - "Verify gradient flow isolation"
          - "Benchmark memory usage"
          - "Add adapter save/load"
        acceptance_criteria:
          - "99.9% parameter reduction"
          - "Fine-tuning converges"
          - "Memory benchmark in README"
```

**Rationale:** PMAT README line 89 shows workflow prompts enforce EXTREME TDD. Auto-generate roadmap from specification
ensures systematic implementation.

**Implementation:** Run `pmat generate-roadmap` on this spec, create `docs/roadmaps/llama-integration.yaml`.

---

### 3.2 TDG Enforcement for Example Code

**Idea 2: Quality Gates for Examples**

```bash
# Pre-commit hook for examples/llama2/
pmat tdg check-quality \
  --path examples/llama2/ \
  --min-grade A \
  --fail-on-violation

# Baseline tracking
pmat tdg baseline create \
  --output .pmat/llama-baseline.json \
  --path examples/llama2/

# Regression detection
pmat tdg check-regression \
  --baseline .pmat/llama-baseline.json \
  --path examples/llama2/ \
  --max-score-drop 5.0 \
  --fail-on-regression
```

**Rationale:** PMAT line 231-256 shows TDG enforcement system. LLaMA examples are high-visibility code; enforce A grade
minimum to prevent technical debt.

**Implementation:** Add `.pmat/tdg-rules.toml` with `llama_min_grade = "A"`, install git hooks.

---

### 3.3 Mutation Testing Integration

**Idea 3: PMAT Mutation Analysis**

```bash
# Run mutation testing on LLaMA autograd ops
pmat mutate \
  --target src/autograd/attention.rs \
  --threshold 85 \
  --output-format json > mutation-report.json

# Quality gate in CI
pmat mutate \
  --target examples/llama2/ \
  --threshold 80 \
  --failures-only \
  --fail-on-threshold
```

**Rationale:** PMAT line 190-228 shows mutation testing evaluates test suite quality. LLaMA attention backward pass
needs 85%+ mutation score to catch gradient bugs.

**Implementation:** Add `make mutation-llama` target, run in tier3 CI.

---

### 3.4 Git-Commit Correlation for Training History

**Idea 4: Track Quality at Training Checkpoints**

```bash
# Analyze quality at specific training checkpoint commits
pmat tdg history \
  --commit checkpoint-epoch-100 \
  --path examples/llama2/train.rs

# Compare quality between training runs
pmat tdg history \
  --range baseline-v1..optimized-v2 \
  --format json | jq '.history[] | {commit, score: .score.grade}'

# Find when quality dropped
pmat tdg history --since HEAD~50 --format json | \
  jq '.history[] | select(.score.grade | test("C|D|F"))'
```

**Rationale:** PMAT line 317-367 shows git-commit correlation for "quality archaeology." LLaMA training spans weeks;
track which commit introduced bugs.

**Implementation:** Run `pmat tdg --with-git-context` after each training checkpoint.

---

### 3.5 Workflow Prompts for LLaMA Development

**Idea 5: Pre-Configured AI Prompts**

```bash
# Get prompt for code coverage enforcement
pmat prompt code-coverage \
  --set TEST_CMD="cargo test --package entrenar-llama" \
  --set COVERAGE_CMD="cargo llvm-cov --package entrenar-llama" \
  --format text | pbcopy

# Debug training issues with Five Whys
pmat prompt debug \
  --set ISSUE="Attention gradients exploding during fine-tuning" \
  --format text

# Refactor attention hotspots
pmat prompt refactor-hotspots \
  --set TARGET_FILE="src/autograd/attention.rs"
```

**Rationale:** PMAT line 133-187 shows 11 workflow prompts enforce EXTREME TDD and Toyota Way. LLaMA development
benefits from standardized prompts.

**Implementation:** Create `prompts/llama-development.yaml` with LLaMA-specific prompts.

---

## 4. Renacer Tracing Integration

### 4.1 Training Performance Profiling

**Idea 6: Syscall-Level Training Bottlenecks**

```bash
# Profile LLaMA training with renacer
renacer --function-time --source -- \
  cargo run --release --example llama2-train

# Output identifies I/O bottlenecks:
# Function Profiling Summary:
# ========================
# Total functions profiled: 234
# Total syscalls: 4,231
#
# Top 10 Hot Paths (by total time):
#   1. llama2::load_checkpoint  - 45.2% (1.2s, 67 syscalls) ⚠️ SLOW I/O
#   2. entrenar::backward       - 32.1% (850ms, 2345 syscalls)
#   3. trueno::matmul_simd      - 12.4% (330ms, 123 syscalls)
```

**Rationale:** Renacer line 343-363 shows function profiling detects I/O bottlenecks. LLaMA training bottlenecked by
checkpoint loading; renacer identifies it.

**Implementation:** Add `make profile-llama` target running renacer on training example.

---

### 4.2 OpenTelemetry Distributed Tracing

**Idea 7: End-to-End Training Observability**

```bash
# Start Jaeger backend
docker-compose -f docker-compose-jaeger.yml up -d

# Export LLaMA training traces to Jaeger
renacer --otlp-endpoint http://localhost:4317 \
        --otlp-service-name llama-training \
        --trace-compute \
        --trace-compute-threshold 100 \
        -- cargo run --release --example llama2-train

# View in Jaeger UI (http://localhost:16686):
# - Service: "llama-training"
# - Root span: "process: llama2-train"
#   - Child span: "forward_pass" (234ms)
#     - Child span: "attention_layer_0" (45ms)
#       - Child span: "syscall: read" (12ms)
#       - Child span: "compute: trueno_matmul" (28ms)
#     - Child span: "attention_layer_1" (43ms)
#   - Child span: "backward_pass" (421ms)
#   - Child span: "optimizer_step" (89ms)
```

**Rationale:** Renacer line 421-447 shows OTLP export to Jaeger/Tempo. LLaMA training is complex; distributed tracing
visualizes forward/backward pass timing.

**Implementation:** Add OTLP tracing to `examples/llama2/train.rs`, document in README.

---

### 4.3 Real-Time Anomaly Detection

**Idea 8: Detect Training Anomalies**

```bash
# Monitor LLaMA training with real-time anomaly detection
renacer --anomaly-realtime \
        --anomaly-threshold 3.0 \
        --stats-extended \
        -- cargo run --release --example llama2-train

# Detects anomalies like:
# ⚠️ ANOMALY: read took 5234 μs (4.2σ from baseline 102.3 μs) - 🟡 Medium
# ⚠️ ANOMALY: matmul took 12345 μs (6.3σ from baseline 1234.5 μs) - 🔴 High
#
# === Real-Time Anomaly Detection Report ===
# Total anomalies detected: 23
# Severity Distribution:
#   🔴 High (>5.0σ):   3 anomalies (likely hardware issues)
#   🟡 Medium (4-5σ):  8 anomalies (investigate)
#   🟢 Low (3-4σ):    12 anomalies (noise)
```

**Rationale:** Renacer line 394-419 shows real-time anomaly detection with severity classification. LLaMA training hits
hardware issues (GPU throttling, disk contention); catch them early.

**Implementation:** Run `renacer --anomaly-realtime` during long training runs, alert on High severity.

---

### 4.4 ML-Based Anomaly Detection

**Idea 9: Cluster-Based Outlier Detection**

```bash
# Use KMeans clustering to detect abnormal training patterns
renacer --ml-anomaly \
        --ml-clusters 5 \
        --ml-compare \
        --format json \
        -- cargo run --release --example llama2-train > training-profile.json

# Analysis:
# {
#   "ml_analysis": {
#     "clusters": 5,
#     "silhouette_score": 0.72,  // Good separation
#     "high_latency_cluster": {
#       "cluster_id": 4,
#       "avg_latency_us": 12345,
#       "syscalls": ["read", "fsync", "write"],
#       "count": 23
#     },
#     "outliers": [
#       {"syscall": "read", "latency_us": 45678, "z_score": 8.2}
#     ]
#   }
# }
```

**Rationale:** Renacer line 58-65 shows ML anomaly detection using aprender KMeans. Traditional z-score fails with
multi-modal latency distributions; ML clustering handles it.

**Implementation:** Add post-training analysis script using renacer ML output.

---

### 4.5 Transpiler Source Mapping for Fine-Tuning

**Idea 10: Trace Fine-Tuning Back to Original Code**

```bash
# Scenario: User fine-tunes LLaMA using Python→Rust transpiled code (via Depyler)

# 1. Transpile Python fine-tuning script to Rust
depyler finetune.py --output finetune_rs.rs --source-map finetune.sourcemap.json

# 2. Run with source mapping
renacer --transpiler-map finetune.sourcemap.json \
        --function-time \
        --source \
        -- ./finetune_rs

# Output shows original Python source locations:
# read(3, buf, 1024) = 1024  [finetune.py:42 in load_dataset]
# write(1, "epoch 1", 7) = 7  [finetune.py:89 in train_loop]
#
# Function Profiling:
#   1. load_dataset (Python:finetune.py:35)  - 45.2% (1.2s)
#   2. train_loop (Python:finetune.py:78)    - 32.1% (850ms)
```

**Rationale:** Renacer line 67-79 shows transpiler source mapping for Python→Rust
and C→Rust. Users fine-tuning LLaMA with transpiled code need to debug
in original language.

**Implementation:** Add source mapping example in `examples/llama2/transpiled_finetuning.md`.

---

## 5. Implementation Roadmap

### Phase 1: Core LLaMA Architecture (Weeks 1-4)

**Deliverables:**
- `examples/llama2/architecture.rs` - Complete LLaMA model
- `examples/llama2/train.rs` - Training from scratch
- `tests/property_llama.rs` - 20+ property tests
- `tests/mutation_resistant_attention.rs` - Mutation-resistant tests

**Quality Gates:**
- TDG Score ≥ 90 (A- grade)
- Test Coverage ≥ 95%
- Mutation Score ≥ 85%
- All gradient checks pass (epsilon=1e-3, threshold=0.2)

---

### Phase 2: LoRA/QLoRA Fine-Tuning (Weeks 5-6)

**Deliverables:**
- `examples/llama2/finetune_lora.rs` - LoRA fine-tuning
- `examples/llama2/finetune_qlora.rs` - QLoRA fine-tuning
- `examples/llama2/memory_benchmarks.rs` - Memory profiling
- `book/src/examples/llama2-finetuning.md` - Documentation

**Quality Gates:**
- Memory benchmarks: 99.9% param reduction (LoRA), 75% memory reduction (QLoRA)
- Convergence test: Fine-tuning loss decreases monotonically
- Integration test: Fine-tuned model improves on target task

---

### Phase 3: Quality Infrastructure (Weeks 7-8)

**Deliverables:**
- `Makefile` with tier1/tier2/tier3 targets
- `.pmat/tdg-baseline.json` - Quality baseline
- `fuzz/fuzz_targets/tokenizer.rs` - Fuzz testing
- `tests/chaos_llama.rs` - Chaos engineering tests

**Quality Gates:**
- Pre-commit hooks pass (<10s)
- Tier3 mutation testing passes (>85% mutation score)
- Chaos tests pass under memory pressure
- Fuzz testing runs 1M iterations without crashes

---

### Phase 4: Tracing & Observability (Weeks 9-10)

**Deliverables:**
- `make profile-llama` - Renacer profiling target
- `docker-compose-jaeger.yml` - OTLP tracing setup
- `scripts/analyze_training.sh` - Post-training analysis
- `book/src/advanced/llama-tracing.md` - Tracing guide

**Quality Gates:**
- Renacer profiling identifies top 3 bottlenecks
- OTLP traces viewable in Jaeger UI
- Anomaly detection catches hardware issues
- Documentation includes example traces

---

## 6. Success Metrics

### Quantitative Metrics

1. **Code Quality**
   - TDG Score: ≥ 90 (A- grade) for all LLaMA code
   - Test Coverage: ≥ 95% (line), ≥ 90% (branch)
   - Mutation Score: ≥ 85% for critical paths
   - Zero clippy warnings (strict mode)

2. **Training Performance**
   - Train 124M LLaMA: < 4 hours on 1x A100 GPU
   - LoRA fine-tuning: 10x faster than full fine-tuning
   - QLoRA memory: 75% reduction vs LoRA

3. **Example Quality**
   - All examples run in CI/CD
   - Zero hallucinations in documentation
   - Property tests verify mathematical properties
   - Chaos tests pass under resource limits

### Qualitative Metrics

1. **Developer Experience**
   - Tier1 feedback < 5 seconds (flow state maintained)
   - Clear error messages from quality gates
   - Documentation has working code examples
   - Examples demonstrate best practices

2. **Production Readiness**
   - OTLP tracing for debugging
   - Anomaly detection catches hardware issues
   - Quality gates prevent regressions
   - Source mapping for transpiled code

---

## 7. References

1. **llama2.rs** - Karpathy's LLaMA 2 reference implementation
2. **Certeza** - Asymptotic test effectiveness framework (TDG 93.3/100)
3. **PMAT** - Multi-language agent toolkit with EXTREME TDD workflows
4. **Renacer** - System call tracer with OTLP and ML anomaly detection
5. **Entrenar** - Training infrastructure (this project, TDG 100/100)

---

## 8. Appendix: Example Output

### A. LLaMA Training with Full Observability Stack

```bash
# Terminal 1: Start Jaeger
docker-compose -f docker-compose-jaeger.yml up -d

# Terminal 2: Run training with full stack
renacer --otlp-endpoint http://localhost:4317 \
        --otlp-service-name llama-training \
        --trace-compute \
        --trace-transpiler-decisions \
        --anomaly-realtime \
        --ml-anomaly \
        --stats-extended \
        -- cargo run --release --example llama2-train \
           --config configs/124m.json \
           --epochs 10 \
           --batch-size 32

# Output:
# [renacer: OTLP export enabled]
# [entrenar: Loading checkpoint...]
# ⚠️ ANOMALY: read took 5234 μs (4.2σ) - 🟡 Medium
# [entrenar: Epoch 1/10, Loss: 3.245]
# [entrenar: Backward pass: 421ms]
# [entrenar: Optimizer step: 89ms]
# ...
# [entrenar: Training complete!]
#
# === Real-Time Anomaly Detection Report ===
# Total anomalies: 12 (3 High, 5 Medium, 4 Low)
#
# === ML Anomaly Analysis ===
# Silhouette Score: 0.72 (good separation)
# High-latency cluster: 23 syscalls (read, fsync, write)
# Outliers: 3 extreme syscalls (>8σ)
#
# View traces at: http://localhost:16686
```

### B. Property Test Output

```bash
$ cargo test property_attention_equivariance

running 1 test
test property_attention_equivariance ... ok
    [proptest] passed 10000 cases

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

### C. Mutation Testing Report

```bash
$ cargo mutants --file src/autograd/attention.rs

Mutating src/autograd/attention.rs
  ✅ src/autograd/attention.rs:123: KILLED by test_attention_backward_gradient_check
  ✅ src/autograd/attention.rs:145: KILLED by property_attention_shape_invariant
  ⏭️ src/autograd/attention.rs:167: SKIPPED (unreachable)
  ❌ src/autograd/attention.rs:189: SURVIVED (add missing test!)

Mutation Score: 87.5% (7 killed / 8 total)
Target: 85% - PASS ✅
```

---

**End of Specification**