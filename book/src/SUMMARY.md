# Entrenar - Training & Optimization Library

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First Training Loop](./getting-started/first-training-loop.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Design Philosophy](./architecture/design-philosophy.md)
- [Module Organization](./architecture/module-organization.md)
- [Type System](./architecture/type-system.md)
- [Memory Management](./architecture/memory-management.md)

# Autograd Engine

- [What is Automatic Differentiation?](./autograd/what-is-autograd.md)
- [Tape-Based Computation Graphs](./autograd/tape-based-graphs.md)
- [Tensor Operations](./autograd/tensor-operations.md)
  - [Matrix Multiplication](./autograd/matmul.md)
  - [Activations (ReLU, GELU, Swish)](./autograd/activations.md)
  - [Layer Normalization](./autograd/layer-norm.md)
  - [Attention Mechanism](./autograd/attention.md)
- [Backward Pass](./autograd/backward-pass.md)
- [Gradient Computation](./autograd/gradient-computation.md)
- [Finite Difference Validation](./autograd/finite-difference.md)

# Optimizers

- [Overview](./optimizers/overview.md)
- [Stochastic Gradient Descent (SGD)](./optimizers/sgd.md)
- [Adam Optimizer](./optimizers/adam.md)
- [AdamW (Decoupled Weight Decay)](./optimizers/adamw.md)
- [Learning Rate Schedulers](./optimizers/schedulers.md)
  - [Cosine Annealing](./optimizers/cosine-scheduler.md)
  - [Step Decay](./optimizers/step-decay.md)
  - [Exponential Decay](./optimizers/exponential-decay.md)
- [Gradient Clipping](./optimizers/gradient-clipping.md)
- [SIMD-Accelerated Updates](./optimizers/simd-acceleration.md)
- [Optimizer Theory](./optimizers/theory.md)

# LoRA (Low-Rank Adaptation)

- [What is LoRA?](./lora/what-is-lora.md)
- [Parameter-Efficient Fine-Tuning](./lora/parameter-efficient-finetuning.md)
- [LoRA Layer Architecture](./lora/layer-architecture.md)
  - [Low-Rank Matrices A and B](./lora/low-rank-matrices.md)
  - [Scaling Factor (alpha/rank)](./lora/scaling-factor.md)
  - [Merge and Unmerge](./lora/merge-unmerge.md)
- [Target Module Selection](./lora/target-modules.md)
- [Gradient Flow Isolation](./lora/gradient-flow.md)
- [Adapter Persistence](./lora/adapter-persistence.md)
  - [Saving Adapters](./lora/saving-adapters.md)
  - [Loading Adapters](./lora/loading-adapters.md)
  - [Sharing Adapters](./lora/sharing-adapters.md)

# QLoRA (Quantized LoRA)

- [Memory-Efficient Fine-Tuning](./quantization/qlora-overview.md)
- [4-bit Quantization](./quantization/4bit-quantization.md)
  - [Block-Wise Quantization](./quantization/block-wise-quantization.md)
  - [Scale Factors](./quantization/scale-factors.md)
  - [Quantization/Dequantization](./quantization/quant-dequant.md)
- [QLoRA Layer](./quantization/qlora-layer.md)
- [On-the-Fly Dequantization](./quantization/on-the-fly-dequant.md)
- [Memory Benchmarks](./quantization/memory-benchmarks.md)
  - [LoRA vs QLoRA Comparison](./quantization/lora-vs-qlora.md)
  - [Transformer Model Benchmarks](./quantization/transformer-benchmarks.md)
  - [Compression Ratios](./quantization/compression-ratios.md)
- [Trade-offs and Best Practices](./quantization/tradeoffs.md)

# Model Merging

- [Overview](./merging/overview.md)
- [TIES Algorithm](./merging/ties.md)
- [DARE Algorithm](./merging/dare.md)
- [SLERP Algorithm](./merging/slerp.md)
- [Multi-Model Ensembles](./merging/multi-model.md)
- [Merge Best Practices](./merging/best-practices.md)

# Pruning

- [Overview](./pruning/overview.md)
- [Pruning Schedules](./pruning/schedules.md)
- [Calibration](./pruning/calibration.md)
- [Pipeline Stages](./pruning/pipeline.md)

# Knowledge Distillation

- [What is Distillation?](./distillation/what-is-distillation.md)
- [Temperature-Scaled KL Divergence](./distillation/temperature-kl.md)
- [Multi-Teacher Ensemble](./distillation/multi-teacher.md)
- [Progressive Layer-Wise](./distillation/progressive.md)
- [Distillation Loss Functions](./distillation/loss-functions.md)
- [Student-Teacher Architecture](./distillation/student-teacher.md)

# Training Loops

- [Trainer API](./training/trainer-api.md)
- [Callback System](./training/callback-system.md)
- [Train Config](./training/train-config.md)
- [Basic Training Loop](./training/basic-loop.md)
- [Batching and Data Loading](./training/batching.md)
- [Loss Functions](./training/loss-functions.md)
- [Validation and Testing](./training/validation.md)
- [Checkpointing](./training/checkpointing.md)
- [Early Stopping](./training/early-stopping.md)
- [Curriculum Learning](./training/curriculum-learning.md)
- [Explainability](./training/explainability.md)

# Real-Time Monitoring

- [Overview](./monitor/overview.md)
- [Experiment Tracking](./monitor/experiment-tracking.md)
- [Quality Gates (Jidoka)](./monitor/quality-gates.md)
- [Metrics Collection](./monitor/metrics-collection.md)
- [Terminal Dashboard](./monitor/dashboard.md)
- [Drift Detection](./monitor/drift-detection.md)
- [Andon Alerting (Jidoka)](./monitor/andon.md)
- [Inference Monitoring](./monitor/inference-monitoring.md)
- [Model Lineage](./monitor/lineage.md)
- [Export Formats](./monitor/export.md)
- [Hansei Reports](./monitor/hansei.md)

# Dashboard

- [Overview](./dashboard/overview.md)
- [DashboardSource Trait](./dashboard/dashboard-source.md)
- [WASM Bindings](./dashboard/wasm.md)

# Ecosystem Integration

- [Overview](./ecosystem/overview.md)
- [Batuta Integration](./ecosystem/batuta.md)
- [Realizar GGUF Export](./ecosystem/realizar.md)
- [Ruchy Session Bridge](./ecosystem/ruchy.md)

# MLOps (Toyota Way)

- [Overview](./mlops/overview.md)
- [Experiment Tracking](./mlops/experiment-tracking.md)
- [Preflight Validation (Jidoka)](./mlops/preflight.md)
- [Model Registry (Kanban)](./mlops/model-registry.md)
- [Hyperparameter Optimization (Kaizen)](./mlops/hpo.md)
- [Differential Privacy](./mlops/differential-privacy.md)
- [GPU Monitoring (Andon)](./mlops/gpu-monitoring.md)
- [REST API Server](./mlops/rest-api.md)
- [Cloud Storage](./mlops/cloud-storage.md)
- [LLM Evaluation](./mlops/llm-evaluation.md)

# Model I/O

- [Overview](./io/overview.md)
- [Save Models](./io/save-models.md)
- [Load Models](./io/load-models.md)
- [Model Metadata](./io/metadata.md)
- [Supported Formats](./io/formats.md)
  - [SafeTensors Format](./io/safetensors-format.md)
  - [JSON Format](./io/json-format.md)
  - [YAML Format](./io/yaml-format.md)
  - [GGUF Format](./io/gguf-format.md)

# Command-Line Interface

- [CLI Overview](./cli/overview.md)
- [Research Commands](./cli/research.md)
- [Benchmark Commands](./cli/benchmark.md)
- [Inspect Command](./cli/inspect.md)
- [Audit Command](./cli/audit.md)
- [Monitor Command](./cli/monitor.md)
- [Completion Command](./cli/completion.md)

# Declarative Training

- [Overview](./declarative/overview.md)
- [YAML Mode Training (v1.0)](./declarative/yaml-mode.md)
- [YAML Examples Catalog](./declarative/yaml-examples.md)
- [Toyota Way QA Process](./declarative/qa-process.md)
- [YAML Configuration](./declarative/yaml-config.md)
- [train_from_yaml Function](./declarative/train-from-yaml.md)
- [Configuration Schema](./declarative/schema.md)
- [Optimizer Builders](./declarative/optimizer-builders.md)
- [Model Builders](./declarative/model-builders.md)

# API Reference

- [Tensor API](./api-reference/tensor-api.md)
- [Autograd Operations](./api-reference/autograd-ops.md)
- [Optimizer API](./api-reference/optimizer-api.md)
- [LoRA API](./api-reference/lora-api.md)
- [QLoRA API](./api-reference/qlora-api.md)
- [Configuration System](./api-reference/configuration.md)
- [Error Handling](./api-reference/error-handling.md)

# Model Evaluation (APR-073)

- [Overview](./eval/overview.md)
- [Classification Metrics](./eval/classification-metrics.md)
- [ModelEvaluator & Leaderboard](./eval/model-evaluator.md)
- [Cross-Validation](./eval/cross-validation.md)
- [Drift Detection](./eval/drift-detection.md)
  - [KS Test](./eval/ks-test.md)
  - [Chi-Square Test](./eval/chi-square-test.md)
  - [PSI (Population Stability Index)](./eval/psi.md)
- [Auto-Retraining (Andon Loop)](./eval/auto-retraining.md)
- [Generative AI Metrics](./eval/generative-metrics.md)

# HuggingFace Pipeline

- [Leaderboard Integration](./hf-pipeline/leaderboard.md)
- [Publishing to HF Hub](./hf-pipeline/publishing.md)

# Examples

## Getting Started
- [Basic Training Loop](./examples/training-loop.md)
- [Linear Regression with Autograd](./examples/linear-regression.md)
- [Training a Simple MLP](./examples/simple-mlp.md)
- [Model I/O (Save/Load)](./examples/model-io.md)
- [Declarative YAML Training](./examples/train-from-yaml.md)

## Neural Network Training
- [MNIST Training (CPU)](./examples/mnist-train.md)
- [MNIST Training (GPU)](./examples/mnist-train-gpu.md)
- [Transformer Fine-Tuning](./examples/transformer-finetuning.md)
- [Custom Loss Functions](./examples/custom-loss.md)
- [Learning Rate Scheduling](./examples/lr-scheduling.md)
- [Gradient Clipping](./examples/gradient-clipping-example.md)

## LLaMA 2 Examples
- [LLaMA 2 Training from Scratch](./examples/llama2-train.md)
- [LLaMA 2 LoRA Fine-Tuning](./examples/llama2-lora.md)
- [LLaMA 2 QLoRA Fine-Tuning](./examples/llama2-qlora.md)
- [LLaMA 2 Memory Benchmarks](./examples/llama2-memory.md)

## LoRA & Quantization
- [Fine-Tuning with LoRA](./examples/lora-finetuning.md)
- [Memory-Efficient QLoRA](./examples/qlora-example.md)
- [Adapter Sharing](./examples/adapter-sharing.md)
- [CUDA Backend Configuration](./examples/cuda-backend.md)

## Knowledge Distillation
- [Basic Distillation](./examples/distillation.md)
- [HuggingFace Model Distillation](./examples/hf-distillation.md)

## Model Operations
- [Model Merging (TIES/DARE/SLERP)](./examples/merge-models.md)
- [Pruning Pipeline](./examples/pruning-pipeline.md)

## Monitoring & Analysis
- [Training Monitoring](./examples/monitoring.md)
- [Inference Monitoring](./examples/inference-monitor.md)
- [Explainability (Feature Attribution)](./examples/explainability.md)
- [Drift Detection Simulation](./examples/drift-simulation.md)
- [P-Value Calibration Check](./examples/calibration-check.md)

## CLI Examples
- [CLI Benchmark](./examples/cli-bench.md)
- [CLI Audit (Bias Detection)](./examples/cli-audit.md)
- [CLI Monitor (Drift)](./examples/cli-monitor.md)
- [CLI Inspect (Model Analysis)](./examples/cli-inspect.md)

## Fine-tuning
- [Test Generation (Qwen2-0.5B)](./examples/finetune-test-gen.md)
- [Real Fine-Tuning (Qwen2.5-Coder-0.5B)](./examples/finetune-real.md)
- [Classification HP Tuning](./examples/classify-tune.md)

## Evaluation & Publishing
- [OpenASR Leaderboard (Whisper)](./examples/open-asr-leaderboard.md)

## Advanced
- [Compiler-in-the-Loop (CITL)](./examples/citl.md)
- [Research Artifact Management](./examples/research.md)
- [Sovereign Deployment](./examples/sovereign.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [EXTREME TDD Methodology](./development/extreme-tdd.md)
- [Testing Strategy](./development/testing-strategy.md)
  - [Unit Tests](./development/unit-tests.md)
  - [Property-Based Tests](./development/property-based-tests.md)
  - [Gradient Checking Tests](./development/gradient-checking.md)
  - [Mutation Testing](./development/mutation-testing.md)
- [Quality Gates](./development/quality-gates.md)
  - [Pre-Commit Hooks](./development/pre-commit-hooks.md)
  - [Continuous Integration](./development/continuous-integration.md)
  - [Code Coverage](./development/code-coverage.md)
  - [Clippy Linting](./development/clippy-linting.md)
- [Benchmarking](./development/benchmarking.md)
- [PMAT Toyota Workflow](./development/pmat-workflow.md)
- [Provable Contracts](./development/contracts.md)

# Best Practices

- [Optimizer Selection](./best-practices/optimizer-selection.md)
- [Learning Rate Tuning](./best-practices/lr-tuning.md)
- [LoRA Configuration](./best-practices/lora-configuration.md)
- [Memory Optimization](./best-practices/memory-optimization.md)
- [Gradient Stability](./best-practices/gradient-stability.md)
- [Debugging Training Issues](./best-practices/debugging-training.md)
- [Performance Profiling](./best-practices/profiling.md)

# Search Algorithms

- [Overview](./search/overview.md)
- [Monte Carlo Tree Search (MCTS)](./search/mcts.md)
  - [State and Action Spaces](./search/state-action.md)
  - [UCB1/PUCT Selection](./search/selection.md)
  - [Expansion and Simulation](./search/expansion.md)
  - [Backpropagation](./search/backpropagation.md)
- [Policy Network Integration](./search/policy-network.md)

# Generative Models

- [Overview](./generative/overview.md)
- [Code Generation GANs](./generative/code-gan.md)
  - [Generator Architecture](./generative/generator.md)
  - [Discriminator Architecture](./generative/discriminator.md)
  - [Latent Space Interpolation](./generative/latent-space.md)
  - [Training Loop](./generative/training.md)
- [Mode Collapse Detection](./generative/mode-collapse.md)

# Advanced Topics

- [Custom Backward Passes](./advanced/custom-backward.md)
- [Implementing New Optimizers](./advanced/new-optimizers.md)
- [Custom LoRA Variants](./advanced/custom-lora.md)
- [Advanced Quantization](./advanced/advanced-quantization.md)
- [Distributed Training](./advanced/distributed-training.md)
- [Model Parallelism](./advanced/model-parallelism.md)
- [Compiler-in-the-Loop (CITL)](./advanced/citl.md)

# Sovereign Deployment

- [Overview](./sovereign/overview.md)

# Academic Research

- [Overview](./research/overview.md)

# Specifications

- [Autograd Specification](./specifications/autograd-spec.md)
- [Optimizer Specification](./specifications/optimizer-spec.md)
- [LoRA Specification](./specifications/lora-spec.md)
- [Quantization Specification](./specifications/quantization-spec.md)
- [Academic Foundations](./specifications/academic-foundations.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [Mathematical Notation](./appendix/mathematical-notation.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
- [Benchmarking Results](./appendix/benchmark-results.md)
