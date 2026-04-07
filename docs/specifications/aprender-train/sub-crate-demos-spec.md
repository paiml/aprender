# Entrenar Sub-Crate Demonstrations Specification

**Version**: 1.0.0
**Status**: Draft
**Authors**: PAIML Engineering
**Date**: 2025-01-28

## Executive Summary

This specification defines five demonstration sub-crates for the Entrenar training library, each showcasing distinct
capabilities of the HuggingFace distillation pipeline. The design follows Toyota Production System (TPS) principles and
PMAT Extreme TDD methodology to ensure zero-defect quality.

---

## Toyota Way Integration

### Guiding Principles

| TPS Principle | Application to Sub-Crates |
|---------------|---------------------------|
| **Jidoka** (Built-in Quality) | Each sub-crate validates inputs at boundaries; errors surface immediately |
| **Kaizen** (Continuous Improvement) | Incremental feature addition with full regression coverage |
| **Heijunka** (Level Scheduling) | Sub-crates share common infrastructure; work parallelizable |
| **Muda** (Waste Elimination) | No redundant code between crates; shared `entrenar-common` |
| **Andon** (Problem Visualization) | Rich error messages with actionable diagnostics |
| **Kanban** (Pull System) | Features implemented on-demand per user workflow |
| **Genchi Genbutsu** (Go and See) | Real model testing, not just mocks |
| **Nemawashi** (Consensus Building) | Spec review before implementation |

### Quality Gates (PMAT Extreme TDD)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PMAT Quality Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1 (<5s)  │ Format + Clippy + Unit Tests                  │
│  Tier 2 (<30s) │ Tier 1 + Integration Tests                    │
│  Tier 3 (<5m)  │ Tier 1+2 + Property Tests (200K iterations)   │
│  Full CI       │ Tier 3 + Coverage (>90%) + Mutants (>80%)     │
└─────────────────────────────────────────────────────────────────┘
```

**Zero Tolerance Metrics**:
- Test Coverage: >90% (cargo llvm-cov)
- Mutation Kill Rate: >80% (cargo-mutants)
- TDG Score: >90 (A grade)
- Cyclomatic Complexity: ≤10
- Cognitive Complexity: ≤15

---

## Sub-Crate 1: `entrenar-distill`

### Purpose

End-to-end CLI for knowledge distillation workflows, implementing the complete teacher-student training pipeline from
model fetching through export.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     entrenar-distill                           │
├────────────────────────────────────────────────────────────────┤
│  CLI Layer (clap)                                              │
│    ├── run      → Execute distillation pipeline                │
│    ├── estimate → Memory/compute requirements                  │
│    ├── validate → Check config before training                 │
│    └── export   → Convert trained model to target format       │
├────────────────────────────────────────────────────────────────┤
│  Pipeline Orchestration                                        │
│    ├── FetchStage    → HfModelFetcher                         │
│    ├── LoadStage     → SafeTensorsTeacher                     │
│    ├── TrainStage    → DistillationTrainer                    │
│    └── ExportStage   → Exporter (SafeTensors/GGUF/APR)        │
├────────────────────────────────────────────────────────────────┤
│  entrenar::hf_pipeline (library)                               │
└────────────────────────────────────────────────────────────────┘
```

### Commands

```bash
# Full distillation run
entrenar-distill run \
  --teacher meta-llama/Llama-2-7b \
  --student TinyLlama/TinyLlama-1.1B \
  --config distill.yaml \
  --output ./distilled-model

# Pre-flight memory estimation (Jidoka: validate before commit)
entrenar-distill estimate \
  --teacher microsoft/codebert-base \
  --batch-size 32 \
  --sequence-length 512

# Config validation (Andon: surface problems early)
entrenar-distill validate --config distill.yaml

# Export to inference format
entrenar-distill export \
  --input ./checkpoint \
  --format gguf \
  --quantize q4_0 \
  --output ./model.gguf
```

### YAML Configuration Schema

```yaml
# distill.yaml
teacher:
  model_id: "meta-llama/Llama-2-7b"
  revision: "main"
  format: safetensors

student:
  model_id: "TinyLlama/TinyLlama-1.1B"

  lora:
    enabled: true
    rank: 64
    alpha: 16
    target_modules: [q_proj, k_proj, v_proj, o_proj]
    dropout: 0.1

distillation:
  temperature: 4.0
  alpha: 0.7

  progressive:
    enabled: true
    layer_mapping: [[0, 3], [1, 7], [2, 11]]
    weight: 0.3

  attention:
    enabled: true
    weight: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 100
  gradient_accumulation: 4
  mixed_precision: bf16

dataset:
  source: "huggingface"
  name: "wikitext"
  split: "train"
  max_length: 512
  streaming: true

output:
  dir: "./distilled-model"
  checkpoint_every: 1000
  format: safetensors
```

### Academic Foundation

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *NIPS Deep Learning Workshop*. | Foundation of temperature-scaled KL divergence loss |
| 2 | Romero, A., et al. (2015). "FitNets: Hints for Thin Deep Nets." *ICLR 2015*. | Intermediate layer matching (progressive distillation) |
| 3 | Zagoruyko, S., & Komodakis, N. (2017). "Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer." *ICLR 2017*. | Attention map distillation methodology |
| 4 | Sun, S., et al. (2019). "Patient Knowledge Distillation for BERT Model Compression." *EMNLP 2019*. | Layer-wise distillation for transformers |
| 5 | Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *NeurIPS 2019 EMC² Workshop*. | Practical BERT distillation at scale |
| 6 | Jiao, X., et al. (2020). "TinyBERT: Distilling BERT for Natural Language Understanding." *EMNLP 2020*. | Two-stage distillation with data augmentation |
| 7 | Wang, W., et al. (2020). "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." *NeurIPS 2020*. | Self-attention relation distillation |
| 8 | Touvron, H., et al. (2021). "Training data-efficient image transformers & distillation through attention." *ICML 2021*. | DeiT attention-based distillation |
| 9 | Park, W., et al. (2019). "Relational Knowledge Distillation." *CVPR 2019*. | Distance-wise and angle-wise distillation |
| 10 | Tung, F., & Mori, G. (2019). "Similarity-Preserving Knowledge Distillation." *ICCV 2019*. | Pairwise activation similarity preservation |

### Test Strategy

```rust
// Property test: Distillation loss decreases over training
proptest! {
    #[test]
    fn distillation_loss_monotonic_decrease(
        teacher_logits in prop::collection::vec(-10.0f32..10.0, 100..1000),
        epochs in 1usize..10
    ) {
        let trainer = DistillationTrainer::new(config, teacher);
        let mut prev_loss = f32::MAX;

        for _ in 0..epochs {
            let loss = trainer.train_epoch(&dataset);
            prop_assert!(loss < prev_loss * 1.1); // Allow 10% variance
            prev_loss = loss;
        }
    }
}

// Integration test: Full pipeline smoke test
#[test]
fn full_pipeline_produces_valid_output() {
    let config = DistillationYamlConfig::from_str(MINIMAL_CONFIG)?;
    let result = Pipeline::run(&config)?;

    assert!(result.output_path.exists());
    assert!(result.metrics.final_loss < result.metrics.initial_loss);
}
```

---

## Sub-Crate 2: `entrenar-shell`

### Purpose

Interactive REPL (Read-Eval-Print Loop) for exploratory model analysis, distillation experiments, and rapid prototyping.
Implements the Genchi Genbutsu principle through hands-on model interaction.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      entrenar-shell                            │
├────────────────────────────────────────────────────────────────┤
│  REPL Engine (rustyline)                                       │
│    ├── Lexer      → Tokenize commands                         │
│    ├── Parser     → Build AST                                 │
│    ├── Evaluator  → Execute commands                          │
│    └── Printer    → Format output (trueno-viz)                │
├────────────────────────────────────────────────────────────────┤
│  Command Registry                                              │
│    ├── ModelCommands    → fetch, load, inspect, compare       │
│    ├── MemoryCommands   → estimate, profile, optimize         │
│    ├── TrainCommands    → distill, finetune, evaluate         │
│    ├── ExportCommands   → save, convert, quantize             │
│    └── SystemCommands   → help, history, config, quit         │
├────────────────────────────────────────────────────────────────┤
│  State Machine                                                 │
│    ├── ModelState       → Currently loaded models             │
│    ├── SessionState     → Training history, metrics           │
│    └── ConfigState      → User preferences                    │
└────────────────────────────────────────────────────────────────┘
```

### Interactive Session Example

```
$ entrenar-shell
Entrenar Shell v0.2.0 - Interactive Distillation Environment
Type 'help' for commands, 'quit' to exit.

entrenar> fetch microsoft/codebert-base
✓ Fetched microsoft/codebert-base (438 MB, SafeTensors)
  Architecture: RoBERTa, Layers: 12, Hidden: 768

entrenar> memory --batch 32 --seq 512
┌─────────────────────────────────────────┐
│ Memory Estimate: microsoft/codebert-base│
├─────────────────────────────────────────┤
│ Precision │ Model    │ Activations │ Total │
├───────────┼──────────┼─────────────┼───────┤
│ FP32      │ 498 MB   │ 2.1 GB      │ 2.6 GB│
│ FP16      │ 249 MB   │ 1.0 GB      │ 1.3 GB│
│ INT8      │ 125 MB   │ 512 MB      │ 637 MB│
└─────────────────────────────────────────┘

entrenar> inspect layers
Layer 0: embeddings (23.4M params)
Layer 1-12: encoder.layer.* (7.1M params each)
Layer 13: pooler (590K params)
Total: 124.6M parameters

entrenar> set student TinyLlama/TinyLlama-1.1B
✓ Student model configured

entrenar> distill --temperature 4.0 --alpha 0.7 --epochs 3 --dry-run
Dry run configuration:
  Teacher: microsoft/codebert-base (124.6M)
  Student: TinyLlama-1.1B (1.1B)
  Loss: KL(T=4.0) * 0.7 + CE * 0.3
  Estimated time: 2h 34m (on current hardware)

Proceed with training? [y/N]: y

entrenar> history
┌────────────────────────────────────────────────────┐
│ Session History                                    │
├──────┬─────────────────────────┬──────────┬────────┤
│ #    │ Command                 │ Duration │ Status │
├──────┼─────────────────────────┼──────────┼────────┤
│ 1    │ fetch codebert-base     │ 12.3s    │ ✓      │
│ 2    │ memory --batch 32       │ 0.1s     │ ✓      │
│ 3    │ inspect layers          │ 0.2s     │ ✓      │
│ 4    │ distill --epochs 3      │ 2h 31m   │ ✓      │
└──────┴─────────────────────────┴──────────┴────────┘

entrenar> export safetensors ./my-model --include-metadata
✓ Exported to ./my-model/model.safetensors (892 MB)

entrenar> quit
Session saved to ~/.entrenar/sessions/2025-01-28-001.json
```

### Academic Foundation

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Perez, F., & Granger, B. E. (2007). "IPython: A System for Interactive Scientific Computing." *Computing in Science & Engineering*. | REPL design for scientific computing |
| 2 | Kluyver, T., et al. (2016). "Jupyter Notebooks – a publishing format for reproducible computational workflows." *ELPUB 2016*. | Interactive notebook paradigm |
| 3 | Rule, A., et al. (2018). "Exploration and Explanation in Computational Notebooks." *CHI 2018*. | User patterns in exploratory analysis |
| 4 | Kery, M. B., et al. (2018). "The Story in the Notebook: Exploratory Data Science using a Literate Programming Tool." *CHI 2018*. | Literate programming for ML |
| 5 | Head, A., et al. (2019). "Managing Messes in Computational Notebooks." *CHI 2019*. | State management in interactive sessions |
| 6 | Chattopadhyay, S., et al. (2020). "What's Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities." *CHI 2020*. | UX improvements for ML tools |
| 7 | Wang, A. Y., et al. (2022). "Documentation Practices in Computational Notebooks." *CSCW 2022*. | Documentation patterns in ML workflows |
| 8 | Amershi, S., et al. (2019). "Software Engineering for Machine Learning: A Case Study." *ICSE-SEIP 2019*. | ML engineering best practices |
| 9 | Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*. | Managing ML system complexity |
| 10 | Patel, K., et al. (2008). "Investigating Statistical Machine Learning as a Tool for Software Development." *CHI 2008*. | Interactive ML tool design |

### Test Strategy

```rust
// Command parsing property test
proptest! {
    #[test]
    fn valid_commands_parse_without_panic(
        cmd in "(fetch|inspect|memory|distill|export) [a-z0-9/-]{1,50}"
    ) {
        let result = CommandParser::parse(&cmd);
        // Should either succeed or return structured error
        assert!(result.is_ok() || result.unwrap_err().is_user_error());
    }
}

// State consistency test
#[test]
fn session_state_survives_serialization_roundtrip() {
    let state = SessionState::with_model("test-model");
    let json = serde_json::to_string(&state)?;
    let restored: SessionState = serde_json::from_str(&json)?;
    assert_eq!(state, restored);
}
```

---

## Sub-Crate 3: `entrenar-lora`

### Purpose

Specialized tool for LoRA/QLoRA configuration optimization, memory planning, and adapter management. Implements Kaizen
through iterative configuration refinement.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       entrenar-lora                            │
├────────────────────────────────────────────────────────────────┤
│  CLI Layer                                                     │
│    ├── plan     → Optimal config for constraints               │
│    ├── compare  → Side-by-side method comparison               │
│    ├── merge    → Combine adapter with base model              │
│    ├── inspect  → Analyze adapter weights                      │
│    └── convert  → Format conversion (safetensors↔pth)          │
├────────────────────────────────────────────────────────────────┤
│  Optimization Engine                                           │
│    ├── MemoryPlanner     → VRAM/RAM constraint solver          │
│    ├── RankOptimizer     → Optimal rank selection              │
│    ├── ModuleSelector    → Target module recommendation        │
│    └── QuantizationAdvisor → 4-bit vs 8-bit guidance          │
├────────────────────────────────────────────────────────────────┤
│  Adapter Operations                                            │
│    ├── MergeEngine       → SVD-based weight merging            │
│    ├── ScaleAdjuster     → Alpha/rank ratio optimization       │
│    └── SparsityAnalyzer  → Effective rank analysis             │
└────────────────────────────────────────────────────────────────┘
```

### Commands

```bash
# Plan optimal LoRA configuration for hardware constraints
entrenar-lora plan \
  --model meta-llama/Llama-2-7b \
  --vram 16GB \
  --method auto  # Chooses LoRA vs QLoRA based on fit

# Output:
# ┌─────────────────────────────────────────────────────────┐
# │ Recommended Configuration for 16GB VRAM                 │
# ├─────────────────────────────────────────────────────────┤
# │ Method: QLoRA (4-bit)                                   │
# │ Rank: 64                                                │
# │ Alpha: 16                                               │
# │ Target Modules: q_proj, k_proj, v_proj, o_proj          │
# │ Trainable Parameters: 33.5M (0.48% of total)            │
# │ Estimated Memory: 14.2 GB (88.8% utilization)           │
# │ Training Speedup: 3.2x vs full fine-tuning              │
# └─────────────────────────────────────────────────────────┘

# Compare different fine-tuning approaches
entrenar-lora compare \
  --model 7B \
  --methods full,lora-16,lora-64,qlora-4bit

# Output:
# ┌──────────────┬─────────┬───────────┬──────────┬─────────┐
# │ Method       │ Memory  │ Params    │ Speed    │ Quality │
# ├──────────────┼─────────┼───────────┼──────────┼─────────┤
# │ Full         │ 56 GB   │ 7.0B      │ 1.0x     │ ★★★★★   │
# │ LoRA r=16    │ 16 GB   │ 8.4M      │ 2.8x     │ ★★★★☆   │
# │ LoRA r=64    │ 18 GB   │ 33.5M     │ 2.5x     │ ★★★★★   │
# │ QLoRA 4-bit  │ 6 GB    │ 33.5M     │ 1.8x     │ ★★★★☆   │
# └──────────────┴─────────┴───────────┴──────────┴─────────┘

# Merge adapter with base model
entrenar-lora merge \
  --base model.safetensors \
  --adapter lora-adapter.safetensors \
  --output merged.safetensors \
  --scale 1.0

# Inspect adapter structure
entrenar-lora inspect adapter.safetensors
# Output:
# Adapter Analysis:
#   Rank: 64
#   Alpha: 16
#   Scale: 0.25 (alpha/rank)
#   Target Modules: 32 (q,k,v,o × 8 layers)
#   Total Parameters: 33,554,432
#   Effective Rank (SVD): 58.3 (91.1% utilization)
#   Sparsity: 2.3%
```

### Academic Foundation

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. | Foundation of low-rank adaptation |
| 2 | Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*. | 4-bit quantization with LoRA |
| 3 | Aghajanyan, A., et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL 2021*. | Theoretical basis for low-rank effectiveness |
| 4 | Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *ACL 2021*. | Alternative PEFT method comparison |
| 5 | Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." *ICML 2019*. | Adapter modules for transfer learning |
| 6 | Liu, H., et al. (2022). "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning." *NeurIPS 2022*. | PEFT efficiency analysis |
| 7 | Lester, B., et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." *EMNLP 2021*. | Soft prompt tuning at scale |
| 8 | He, J., et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." *ICLR 2022*. | Unified PEFT framework |
| 9 | Zaken, E. B., et al. (2022). "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models." *ACL 2022*. | Bias-only fine-tuning |
| 10 | Karimi Mahabadi, R., et al. (2021). "Compacter: Efficient Low-Rank Hypercomplex Adapter Layers." *NeurIPS 2021*. | Hypercomplex adapter layers |

### Test Strategy

```rust
// Memory estimation accuracy test
proptest! {
    #[test]
    fn memory_estimate_within_10_percent(
        model_params in 1_000_000u64..70_000_000_000,
        rank in 1u32..256,
        bits in prop::sample::select(vec![4, 8, 16, 32])
    ) {
        let estimate = MemoryPlanner::estimate(model_params, rank, bits);
        let actual = simulate_allocation(model_params, rank, bits);

        let error = (estimate as f64 - actual as f64).abs() / actual as f64;
        prop_assert!(error < 0.10, "Memory estimate error: {:.1}%", error * 100.0);
    }
}

// Merge correctness test
#[test]
fn lora_merge_produces_mathematically_correct_weights() {
    let base = Tensor::randn([1024, 1024]);
    let lora_a = Tensor::randn([1024, 64]);
    let lora_b = Tensor::randn([64, 1024]);
    let alpha = 16.0;
    let rank = 64;

    let merged = MergeEngine::merge(&base, &lora_a, &lora_b, alpha, rank);
    let expected = &base + &(&lora_a.matmul(&lora_b) * (alpha / rank as f32));

    assert_tensors_close(&merged, &expected, 1e-5);
}
```

---

## Sub-Crate 4: `entrenar-inspect`

### Purpose

Deep inspection and analysis of SafeTensors models with architecture detection, memory profiling, and format conversion.
Implements Andon through comprehensive model visualization.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      entrenar-inspect                          │
├────────────────────────────────────────────────────────────────┤
│  CLI Layer                                                     │
│    ├── info     → Quick model summary                          │
│    ├── layers   → Detailed layer breakdown                     │
│    ├── memory   → Training memory profile                      │
│    ├── compare  → Diff two models                              │
│    ├── validate → Check model integrity                        │
│    └── convert  → Format conversion                            │
├────────────────────────────────────────────────────────────────┤
│  Analysis Engine                                               │
│    ├── ArchitectureDetector  → Identify model family           │
│    ├── TensorAnalyzer        → Weight statistics               │
│    ├── MemoryProfiler        → Per-layer memory breakdown      │
│    └── IntegrityChecker      → Validate tensor shapes/types    │
├────────────────────────────────────────────────────────────────┤
│  Format Handlers                                               │
│    ├── SafeTensorsReader     → Parse .safetensors              │
│    ├── GGUFWriter            → Export to llama.cpp format      │
│    └── APRWriter             → Export to JSON-based format     │
└────────────────────────────────────────────────────────────────┘
```

### Commands

```bash
# Quick model info
entrenar-inspect info model.safetensors
# Output:
# ┌─────────────────────────────────────────────────────────┐
# │ Model: model.safetensors                                │
# ├─────────────────────────────────────────────────────────┤
# │ Architecture: LLaMA                                     │
# │ Parameters: 6,738,415,616 (6.7B)                       │
# │ Layers: 32                                              │
# │ Hidden Size: 4096                                       │
# │ Vocabulary: 32000                                       │
# │ Precision: FP16                                         │
# │ File Size: 12.5 GB                                      │
# └─────────────────────────────────────────────────────────┘

# Detailed layer breakdown
entrenar-inspect layers model.safetensors --verbose
# Output:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Layer Analysis                                                  │
# ├─────────────────────────┬─────────────┬───────────┬─────────────┤
# │ Layer                   │ Shape       │ Params    │ Memory      │
# ├─────────────────────────┼─────────────┼───────────┼─────────────┤
# │ model.embed_tokens      │ [32000,4096]│ 131.1M    │ 262 MB      │
# │ model.layers.0.q_proj   │ [4096,4096] │ 16.8M     │ 33.6 MB     │
# │ model.layers.0.k_proj   │ [4096,4096] │ 16.8M     │ 33.6 MB     │
# │ ...                     │             │           │             │
# │ model.lm_head           │ [4096,32000]│ 131.1M    │ 262 MB      │
# └─────────────────────────┴─────────────┴───────────┴─────────────┘

# Training memory estimate
entrenar-inspect memory model.safetensors \
  --batch-size 32 \
  --sequence-length 2048 \
  --optimizer adamw \
  --precision bf16

# Compare two models
entrenar-inspect compare model-v1.safetensors model-v2.safetensors
# Output:
# Differences:
#   + model.layers.32.* (new layer added)
#   ~ model.embed_tokens: shape changed [32000,4096] → [50000,4096]
#   - model.pooler.* (removed)

# Validate model integrity
entrenar-inspect validate model.safetensors --strict
# Output:
# ✓ All tensor shapes consistent
# ✓ No NaN/Inf values detected
# ✓ Dtype consistency verified
# ✓ Architecture constraints satisfied

# Convert format
entrenar-inspect convert model.safetensors \
  --to gguf \
  --quantize q4_0 \
  --output model.gguf
```

### Academic Foundation

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS 2019*. | Tensor storage and serialization patterns |
| 2 | Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC 2020*. | Memory profiling methodology |
| 3 | Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv*. | Large model architecture patterns |
| 4 | Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*. | GPT architecture analysis |
| 5 | Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." *arXiv*. | LLaMA architecture specification |
| 6 | Jiang, A. Q., et al. (2023). "Mistral 7B." *arXiv*. | Sliding window attention architecture |
| 7 | Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." *arXiv*. | Scaling analysis methodology |
| 8 | Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." *NeurIPS 2022*. | Model size vs compute tradeoffs |
| 9 | Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. | Quantization format analysis |
| 10 | Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*. | 8-bit quantization methodology |

### Test Strategy

```rust
// Architecture detection property test
proptest! {
    #[test]
    fn architecture_detection_is_deterministic(
        seed in 0u64..1000
    ) {
        let model = generate_random_model(seed);
        let arch1 = ArchitectureDetector::detect(&model);
        let arch2 = ArchitectureDetector::detect(&model);

        prop_assert_eq!(arch1, arch2);
    }
}

// Format conversion roundtrip test
#[test]
fn safetensors_to_gguf_preserves_weights() {
    let original = SafeTensorsReader::read("test.safetensors")?;

    GGUFWriter::write(&original, "test.gguf")?;
    let converted = GGUFReader::read("test.gguf")?;

    for (name, original_tensor) in original.tensors() {
        let converted_tensor = converted.get(name)?;
        // Allow quantization error for q4_0
        assert_tensors_close(original_tensor, converted_tensor, 0.05);
    }
}
```

---

## Sub-Crate 5: `entrenar-bench`

### Purpose

Comprehensive benchmarking suite for distillation strategies, measuring loss convergence, throughput, and quality
metrics. Implements Kaizen through data-driven optimization.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       entrenar-bench                           │
├────────────────────────────────────────────────────────────────┤
│  CLI Layer                                                     │
│    ├── temperature → Sweep temperature hyperparameter          │
│    ├── compare     → Compare distillation strategies           │
│    ├── profile     → Full training profile with metrics        │
│    ├── ablation    → Systematic ablation study                 │
│    └── report      → Generate analysis report                  │
├────────────────────────────────────────────────────────────────┤
│  Benchmark Engine                                              │
│    ├── Sweeper           → Hyperparameter sweep executor       │
│    ├── MetricsCollector  → Loss, throughput, memory tracking   │
│    ├── StatisticalAnalyzer → Significance testing              │
│    └── Visualizer        → Chart generation (trueno-viz)       │
├────────────────────────────────────────────────────────────────┤
│  Experiment Framework                                          │
│    ├── ConfigGenerator   → Systematic config generation        │
│    ├── ResultsAggregator → Cross-run statistics                │
│    └── ReportWriter      → Markdown/HTML output                │
└────────────────────────────────────────────────────────────────┘
```

### Commands

```bash
# Temperature sweep
entrenar-bench temperature \
  --range 1.0..8.0 \
  --step 0.5 \
  --teacher microsoft/codebert-base \
  --student distilbert-base-uncased \
  --epochs 3

# Output:
# Temperature Sweep Results
# ┌─────────────┬────────────┬────────────┬────────────┐
# │ Temperature │ Final Loss │ Accuracy   │ Throughput │
# ├─────────────┼────────────┼────────────┼────────────┤
# │ 1.0         │ 0.8234     │ 78.2%      │ 1240 ex/s  │
# │ 1.5         │ 0.7891     │ 79.1%      │ 1235 ex/s  │
# │ 2.0         │ 0.7456     │ 80.3%      │ 1232 ex/s  │
# │ 2.5         │ 0.7123     │ 81.2%      │ 1228 ex/s  │
# │ 3.0         │ 0.6834     │ 82.1%      │ 1225 ex/s  │
# │ 3.5         │ 0.6712     │ 82.8%      │ 1220 ex/s  │
# │ 4.0         │ 0.6589     │ 83.4%      │ 1215 ex/s  │ ← optimal
# │ 4.5         │ 0.6623     │ 83.2%      │ 1210 ex/s  │
# │ 5.0         │ 0.6701     │ 82.9%      │ 1205 ex/s  │
# │ ...         │            │            │            │
# └─────────────┴────────────┴────────────┴────────────┘
#
# Recommendation: T=4.0 achieves optimal accuracy/loss tradeoff
# Statistical significance: p < 0.001 vs baseline (T=1.0)

# Compare distillation strategies
entrenar-bench compare \
  --strategies kd-only,progressive,attention,combined \
  --teacher meta-llama/Llama-2-7b \
  --student TinyLlama-1.1B

# Output:
# Strategy Comparison (5 runs each, mean ± std)
# ┌──────────────┬─────────────────┬─────────────────┬────────────┐
# │ Strategy     │ Final Loss      │ BLEU Score      │ Time       │
# ├──────────────┼─────────────────┼─────────────────┼────────────┤
# │ KD-only      │ 0.823 ± 0.012   │ 34.2 ± 0.4      │ 2h 12m     │
# │ Progressive  │ 0.756 ± 0.008   │ 36.8 ± 0.3      │ 2h 45m     │
# │ Attention    │ 0.789 ± 0.015   │ 35.6 ± 0.5      │ 2h 28m     │
# │ Combined     │ 0.712 ± 0.006   │ 38.1 ± 0.2      │ 3h 15m     │ ★
# └──────────────┴─────────────────┴─────────────────┴────────────┘
#
# Statistical Analysis:
#   Combined vs KD-only: p < 0.001 (significant improvement)
#   Combined vs Progressive: p = 0.023 (significant)
#   Combined vs Attention: p < 0.001 (significant)

# Generate comprehensive profile
entrenar-bench profile \
  --config distill.yaml \
  --output report.html \
  --include-loss-curves \
  --include-memory-timeline \
  --include-throughput-histogram

# Ablation study
entrenar-bench ablation \
  --base-config distill.yaml \
  --ablate "progressive.enabled,attention.enabled,temperature,alpha" \
  --output ablation-results.csv
```

### Visualization Output (trueno-viz)

```
Loss Curve (T=4.0, α=0.7)
│
│ 1.2 ┤ ●
│     │  ╲
│ 1.0 ┤   ╲●
│     │     ╲
│ 0.8 ┤      ╲●
│     │        ╲●
│ 0.6 ┤          ╲●──●──●──●
│     │
│ 0.4 ┤
│     │
│ 0.2 ┤
│     └──────────────────────────
│       0   2   4   6   8   10
│                Epoch
```

### Academic Foundation

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Mattson, P., et al. (2020). "MLPerf Training Benchmark." *MLSys 2020*. | ML benchmarking methodology |
| 2 | Bouthillier, X., et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." *MLSys 2021*. | Statistical rigor in ML benchmarks |
| 3 | Henderson, P., et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI 2018*. | Reproducibility in ML experiments |
| 4 | Lucic, M., et al. (2018). "Are GANs Created Equal? A Large-Scale Study." *NeurIPS 2018*. | Large-scale ablation methodology |
| 5 | Melis, G., et al. (2018). "On the State of the Art of Evaluation in Neural Language Models." *ICLR 2018*. | Evaluation methodology critique |
| 6 | Dodge, J., et al. (2019). "Show Your Work: Improved Reporting of Experimental Results." *EMNLP 2019*. | Reporting standards for ML |
| 7 | Bhojanapalli, S., et al. (2021). "Understanding Robustness of Transformers for Image Classification." *ICCV 2021*. | Ablation study methodology |
| 8 | Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv*. | Large-scale training benchmarks |
| 9 | Lipton, Z. C., & Steinhardt, J. (2019). "Troubling Trends in Machine Learning Scholarship." *Queue*. | Scientific rigor in ML |
| 10 | Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*. | ML reproducibility checklist |

### Test Strategy

```rust
// Statistical significance test
proptest! {
    #[test]
    fn statistical_tests_reject_null_when_effect_exists(
        effect_size in 0.5f64..2.0,
        n_samples in 10usize..100
    ) {
        let control = generate_samples(0.0, 1.0, n_samples);
        let treatment = generate_samples(effect_size, 1.0, n_samples);

        let p_value = StatisticalAnalyzer::welch_t_test(&control, &treatment);

        // With effect size >= 0.5 and n >= 10, should detect difference
        prop_assert!(p_value < 0.05, "Failed to detect effect: p={}", p_value);
    }
}

// Benchmark reproducibility test
#[test]
fn benchmark_results_are_reproducible_with_seed() {
    let config = BenchConfig::default().seed(42);

    let result1 = Sweeper::run(&config);
    let result2 = Sweeper::run(&config);

    assert_eq!(result1.metrics, result2.metrics);
}

// Report generation test
#[test]
fn report_contains_all_required_sections() {
    let results = run_minimal_benchmark();
    let report = ReportWriter::generate_html(&results);

    assert!(report.contains("<h2>Summary</h2>"));
    assert!(report.contains("<h2>Methodology</h2>"));
    assert!(report.contains("<h2>Results</h2>"));
    assert!(report.contains("<h2>Statistical Analysis</h2>"));
    assert!(report.contains("<h2>Recommendations</h2>"));
}
```

---

## Shared Infrastructure: `entrenar-common`

All sub-crates share common infrastructure to eliminate Muda (waste):

```rust
// entrenar-common/src/lib.rs
pub mod cli {
    pub use clap::{Parser, Subcommand, Args};
    pub mod styles;  // Consistent terminal styling
    pub mod progress; // Progress bars (indicatif)
}

pub mod config {
    pub use crate::yaml::*;
    pub use crate::validation::*;
}

pub mod output {
    pub use trueno_viz::*;  // Visualization
    pub mod table;          // ASCII tables
    pub mod json;           // JSON output mode
}

pub mod error {
    pub use thiserror::Error;
    pub type Result<T> = std::result::Result<T, EntrenarError>;
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
| Ticket | Description | Hours |
|--------|-------------|-------|
| ENT-084 | Create entrenar-common shared infrastructure | 16 |
| ENT-085 | Implement CLI framework with clap | 8 |
| ENT-086 | Add trueno-viz integration for tables/charts | 12 |

### Phase 2: Core Sub-Crates (Week 3-6)
| Ticket | Description | Hours |
|--------|-------------|-------|
| ENT-087 | entrenar-distill: Pipeline orchestration | 24 |
| ENT-088 | entrenar-distill: YAML config validation | 16 |
| ENT-089 | entrenar-shell: REPL engine | 32 |
| ENT-090 | entrenar-shell: Command registry | 20 |
| ENT-091 | entrenar-lora: Memory planner | 20 |
| ENT-092 | entrenar-lora: Merge engine | 16 |
| ENT-093 | entrenar-inspect: Architecture detector | 24 |
| ENT-094 | entrenar-inspect: Format converters | 20 |
| ENT-095 | entrenar-bench: Sweep executor | 24 |
| ENT-096 | entrenar-bench: Statistical analyzer | 16 |

### Phase 3: Integration & Polish (Week 7-8)
| Ticket | Description | Hours |
|--------|-------------|-------|
| ENT-097 | Cross-crate integration tests | 24 |
| ENT-098 | Documentation and examples | 16 |
| ENT-099 | Performance optimization | 16 |
| ENT-100 | Release preparation | 8 |

**Total Estimated Hours: 312**

---

## Dependency Graph

```
                    ┌─────────────────┐
                    │ entrenar (lib)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────┐ ┌─────────────┐
    │ entrenar-common │ │ trueno  │ │ trueno-viz  │
    └────────┬────────┘ └────┬────┘ └──────┬──────┘
             │               │             │
    ┌────────┴────────┬──────┴──────┬──────┴──────┐
    │                 │             │             │
    ▼                 ▼             ▼             ▼
┌──────────┐  ┌───────────┐ ┌────────────┐ ┌────────────┐
│ distill  │  │   shell   │ │    lora    │ │  inspect   │
└──────────┘  └───────────┘ └────────────┘ └────────────┘
                                                │
                                          ┌─────┴─────┐
                                          ▼           ▼
                                    ┌──────────┐ ┌─────────┐
                                    │  bench   │ │  (all)  │
                                    └──────────┘ └─────────┘
```

---

## References (Complete Bibliography)

### Knowledge Distillation (Sub-Crate 1)
1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Deep Learning
   Workshop*.
2. Romero, A., et al. (2015). FitNets: Hints for Thin Deep Nets. *ICLR 2015*.
3. Zagoruyko, S., & Komodakis, N. (2017). Paying More Attention to Attention. *ICLR 2017*.
4. Sun, S., et al. (2019). Patient Knowledge Distillation for BERT Model Compression. *EMNLP 2019*.
5. Sanh, V., et al. (2019). DistilBERT. *NeurIPS 2019 EMC² Workshop*.
6. Jiao, X., et al. (2020). TinyBERT. *EMNLP 2020*.
7. Wang, W., et al. (2020). MiniLM. *NeurIPS 2020*.
8. Touvron, H., et al. (2021). DeiT. *ICML 2021*.
9. Park, W., et al. (2019). Relational Knowledge Distillation. *CVPR 2019*.
10. Tung, F., & Mori, G. (2019). Similarity-Preserving Knowledge Distillation. *ICCV 2019*.

### Interactive Computing (Sub-Crate 2)
11. Perez, F., & Granger, B. E. (2007). IPython. *Computing in Science & Engineering*.
12. Kluyver, T., et al. (2016). Jupyter Notebooks. *ELPUB 2016*.
13. Rule, A., et al. (2018). Exploration and Explanation in Computational Notebooks. *CHI 2018*.
14. Kery, M. B., et al. (2018). The Story in the Notebook. *CHI 2018*.
15. Head, A., et al. (2019). Managing Messes in Computational Notebooks. *CHI 2019*.
16. Chattopadhyay, S., et al. (2020). What's Wrong with Computational Notebooks? *CHI 2020*.
17. Wang, A. Y., et al. (2022). Documentation Practices in Computational Notebooks. *CSCW 2022*.
18. Amershi, S., et al. (2019). Software Engineering for Machine Learning. *ICSE-SEIP 2019*.
19. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS 2015*.
20. Patel, K., et al. (2008). Investigating Statistical Machine Learning as a Tool. *CHI 2008*.

### Parameter-Efficient Fine-Tuning (Sub-Crate 3)
21. Hu, E. J., et al. (2021). LoRA. *ICLR 2022*.
22. Dettmers, T., et al. (2023). QLoRA. *NeurIPS 2023*.
23. Aghajanyan, A., et al. (2021). Intrinsic Dimensionality. *ACL 2021*.
24. Li, X. L., & Liang, P. (2021). Prefix-Tuning. *ACL 2021*.
25. Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML 2019*.
26. Liu, H., et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning. *NeurIPS 2022*.
27. Lester, B., et al. (2021). The Power of Scale for Prompt Tuning. *EMNLP 2021*.
28. He, J., et al. (2022). Unified View of Parameter-Efficient Transfer Learning. *ICLR 2022*.
29. Zaken, E. B., et al. (2022). BitFit. *ACL 2022*.
30. Karimi Mahabadi, R., et al. (2021). Compacter. *NeurIPS 2021*.

### Model Analysis (Sub-Crate 4)
31. Paszke, A., et al. (2019). PyTorch. *NeurIPS 2019*.
32. Rajbhandari, S., et al. (2020). ZeRO. *SC 2020*.
33. Shoeybi, M., et al. (2019). Megatron-LM. *arXiv*.
34. Brown, T., et al. (2020). GPT-3. *NeurIPS 2020*.
35. Touvron, H., et al. (2023). LLaMA. *arXiv*.
36. Jiang, A. Q., et al. (2023). Mistral 7B. *arXiv*.
37. Chowdhery, A., et al. (2022). PaLM. *arXiv*.
38. Hoffmann, J., et al. (2022). Chinchilla. *NeurIPS 2022*.
39. Frantar, E., et al. (2023). GPTQ. *ICLR 2023*.
40. Dettmers, T., et al. (2022). LLM.int8(). *NeurIPS 2022*.

### Benchmarking Methodology (Sub-Crate 5)
41. Mattson, P., et al. (2020). MLPerf Training Benchmark. *MLSys 2020*.
42. Bouthillier, X., et al. (2021). Accounting for Variance in ML Benchmarks. *MLSys 2021*.
43. Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters. *AAAI 2018*.
44. Lucic, M., et al. (2018). Are GANs Created Equal? *NeurIPS 2018*.
45. Melis, G., et al. (2018). On the State of the Art of Evaluation. *ICLR 2018*.
46. Dodge, J., et al. (2019). Show Your Work. *EMNLP 2019*.
47. Bhojanapalli, S., et al. (2021). Understanding Robustness of Transformers. *ICCV 2021*.
48. Goyal, P., et al. (2017). Accurate, Large Minibatch SGD. *arXiv*.
49. Lipton, Z. C., & Steinhardt, J. (2019). Troubling Trends in ML Scholarship. *Queue*.
50. Pineau, J., et al. (2021). Improving Reproducibility in ML Research. *JMLR*.

---

## Approval

- [ ] Technical Review
- [ ] Architecture Review
- [ ] Security Review
- [ ] Documentation Review

**Approved By**: _________________ **Date**: _________________

---

## Peer Review Annotations (Toyota Way Analysis)

The following annotations review the specification through the lens of the Toyota Production System (TPS), highlighting
alignment with lean principles and supporting academic literature.

### 1. Jidoka & Poka-Yoke (Mistake Proofing)
**Ref:** `entrenar-distill validate` & `entrenar-inspect validate`
*   **Annotation:** The explicit inclusion of pre-flight validation commands implements software *Poka-Yoke*. By
    catching configuration errors (e.g., tensor shape mismatches, NaN values) before the expensive training process
    begins, we adhere to the *Jidoka* principle of stopping the line to fix problems immediately.
*   **Support:** Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-yoke System*. In a software
    context, **Pugh, K. (2011)** in *Lean-Agile Acceptance Test-Driven-Development* argues that executable
    specifications (like these validation commands) are the ultimate form of defect prevention.

### 2. Heijunka (Leveling) via Resource Planning
**Ref:** `entrenar-lora plan` (MemoryPlanner)
*   **Annotation:** The `MemoryPlanner` component is a digital implementation of *Heijunka*. It levels the production
    load by predicting memory demands against hardware constraints, preventing the *Muri* (overburden) of the GPU which
    typically results in OOM crashes.
*   **Support:** **Rajbhandari, S., et al. (2020)** ("ZeRO: Memory Optimizations...") demonstrate that static analysis
    of model architecture versus device state is critical for efficient training of large models, effectively leveling
    the schedule for hardware utilization.

### 3. Genchi Genbutsu (Go and See)
**Ref:** `entrenar-shell` (REPL)
*   **Annotation:** The interactive REPL facilitates *Genchi Genbutsu*. Rather than relying on documentation or
    assumptions ("reports"), the engineer interacts directly with the "gemba" (the actual model tensors and weights) to
    verify their state.
*   **Support:** **Kery, M. B., et al. (2018)** ("The Story in the Notebook") found that immediate, interactive feedback
    loops in exploratory programming significantly reduce cognitive load and error rates compared to batch-process
    debugging.

### 4. Kaizen (Continuous Improvement) via Sweep
**Ref:** `entrenar-bench` (Sweeper)
*   **Annotation:** The hyperparameter sweep functionality automates the data collection required for *Kaizen*. It moves
    optimization from "gut feel" to scientific method, reducing the *Muda* of ineffective training runs.
*   **Support:** **Bergstra, J., & Bengio, Y. (2012)** ("Random Search for Hyper-Parameter Optimization") empirically
    prove that systematic (even random) exploration of the hyperparameter space yields better results with less compute
    resource waste than manual tuning.

### 5. Andon (Visualizing Problems)
**Ref:** `entrenar-inspect` (IntegrityChecker)
*   **Annotation:** The integrity checker functions as an automated *Andon* cord. By detecting silent data corruptions
    (like architecture incompatibility or precision loss) and halting the process with "rich error messages," it
    prevents defects from passing downstream.
*   **Support:** **Sculley, D., et al. (2015)** ("Hidden Technical Debt in Machine Learning Systems") identify "Data
    Dependencies Cost" and the lack of automated verification as a primary source of system accumulation. This tool
    directly pays down that debt.

### 6. Muda (Waste) Reduction via Modularity
**Ref:** `entrenar-common`
*   **Annotation:** The extraction of shared logic into `entrenar-common` eliminates the *Muda* of defect correction in
    duplicate code. This follows the "Single Source of Truth" pattern, essential for maintainability.
*   **Support:** **Parnas, D. L. (1972)** ("On the criteria to be used in decomposing systems into modules") established
    that modularity based on information hiding (encapsulating the "secret" of implementation) is the key to reducing
    the cost of software change.

### 7. Standardization (Standardized Work)
**Ref:** `entrenar-common::cli::styles`
*   **Annotation:** Standardizing terminal output and interaction patterns reduces the cognitive *Muri* for users. In
    TPS, Standardized Work is the baseline for improvement; here, a consistent CLI UX allows users to focus on the ML
    task, not the tool syntax.
*   **Support:** **Norman, D. A. (2013)** (*The Design of Everyday Things*) emphasizes that consistency in signifiers
    and feedback mechanisms is critical for reducing operator error in complex systems.

### 8. SMED (Single Minute Exchange of Die)
**Ref:** `entrenar-lora convert` & `entrenar-distill export`
*   **Annotation:** The ability to rapidly convert between training formats (SafeTensors/PyTorch) and inference formats
    (GGUF) mimics *SMED*. It reduces the "changeover time" between the training phase and the deployment phase,
    accelerating the feedback loop.
*   **Support:** **Fowler, M. (2018)** (*Refactoring*) describes "Data Mappers" and "Gateway" patterns as essential for
    decoupling systems, allowing independent evolution and rapid interchange of parts (formats).

### 9. Visual Control
**Ref:** `trueno-viz` (Loss Curves)
*   **Annotation:** Embedding visualizations directly in the terminal provides *Visual Control*. Abnormalities in loss
    convergence (e.g., divergence, stalling) become immediately apparent without requiring context switching to a web
    browser (Wait Time/Muda).
*   **Support:** **Tufte, E. R. (2001)** (*The Visual Display of Quantitative Information*) argues that "graphical
    excellence" consists of complex ideas communicated with clarity, precision, and efficiency, which is critical for
    rapid decision-making in engineering.

### 10. Respect for People (Human-Centric Design)
**Ref:** Error Handling Strategy ("Actionable Diagnostics")
*   **Annotation:** The specification's insistence on "rich error messages with actionable diagnostics" embodies the TPS
    principle of *Respect for People*. It treats the developer as an intelligent operator who needs clear information to
    succeed, not cryptic codes.
*   **Support:** **Myers, B. A., et al. (2016)** ("Programmers are Users too") highlight that usability flaws in API and
    tool design are a major contributor to software defects. Actionable error messages are a primary intervention for
    improving developer productivity and satisfaction.
