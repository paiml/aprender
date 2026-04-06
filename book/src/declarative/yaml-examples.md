# YAML Examples Catalog

This page catalogs all 30 YAML configuration examples, organized by category. Each example demonstrates a specific
training scenario.

## Overview

| Section | Examples | Focus Area |
|---------|----------|------------|
| A | 6 | Basic Training & Data |
| B | 2 | Compiler-in-the-Loop |
| C | 4 | Model Architecture |
| D | 4 | Optimization & Schedulers |
| E | 4 | Monitoring & Alerts |
| F | 4 | Reliability & Checkpoints |
| G | 2 | Inference & Output |
| H | 2 | Research & Privacy |
| I | 1 | Ecosystem Integration |
| J | 1 | Edge Cases |

## Section A: Basic Training & Data

### mnist_cpu.yaml

MNIST baseline training on CPU.

```bash
entrenar train examples/yaml/mnist_cpu.yaml
```

**QA Focus**: Verify alimentar downloads and caches correctly.

### csv_data.yaml

Training on local CSV tabular data.

```bash
entrenar train examples/yaml/csv_data.yaml
```

**QA Focus**: CSV parsing robustness (headers, types).

### parquet_data.yaml

High-throughput columnar data loading.

```bash
entrenar train examples/yaml/parquet_data.yaml
```

**QA Focus**: Parquet read performance.

### multiworker.yaml

Multi-worker data loading.

```bash
entrenar train examples/yaml/multiworker.yaml
```

**QA Focus**: No data corruption with parallel workers.

### dropout.yaml

Regularization with dropout layers.

```bash
entrenar train examples/yaml/dropout.yaml
```

**QA Focus**: Dropout disabled during validation.

### deterministic.yaml

Bit-exact reproducible training.

```bash
entrenar train examples/yaml/deterministic.yaml
```

**QA Focus**: Same seed produces identical results.

## Section B: Compiler-in-the-Loop (CITL)

### citl_suggest.yaml

CITL optimization suggestions.

```bash
entrenar train examples/yaml/citl_suggest.yaml
```

**QA Focus**: Suggestions are actionable.

### citl_workspace.yaml

CITL workspace management.

```bash
entrenar train examples/yaml/citl_workspace.yaml
```

**QA Focus**: Workspace isolation.

## Section C: Model Architecture

### custom_arch.yaml

Custom model architecture definition.

```bash
entrenar train examples/yaml/custom_arch.yaml
```

**QA Focus**: Layer connections validated.

### llama2_mock.yaml

LLaMA-2 mock model for testing.

```bash
entrenar train examples/yaml/llama2_mock.yaml
```

**QA Focus**: Architecture matches real LLaMA.

### lora.yaml

LoRA fine-tuning configuration.

```bash
entrenar train examples/yaml/lora.yaml
```

**QA Focus**: Only adapter weights updated.

### qlora.yaml

QLoRA 4-bit fine-tuning.

```bash
entrenar train examples/yaml/qlora.yaml
```

**QA Focus**: VRAM usage < 50% of full fine-tune.

## Section D: Optimization & Schedulers

### grad_clip.yaml

Gradient clipping for stability.

```bash
entrenar train examples/yaml/grad_clip.yaml
```

**QA Focus**: Gradient norms bounded.

### grad_accum.yaml

Gradient accumulation for large effective batch.

```bash
entrenar train examples/yaml/grad_accum.yaml
```

**QA Focus**: Accumulation count matches config.

### lr_schedule.yaml

Learning rate scheduling (cosine).

```bash
entrenar train examples/yaml/lr_schedule.yaml
```

**QA Focus**: LR follows expected curve.

### distillation.yaml

Knowledge distillation from teacher.

```bash
entrenar train examples/yaml/distillation.yaml
```

**QA Focus**: Student approaches teacher quality.

## Section E: Monitoring & Alerts

### andon.yaml

Andon alerting system (Jidoka).

```bash
entrenar train examples/yaml/andon.yaml
```

**QA Focus**: Alerts trigger on anomalies.

### outlier.yaml

Outlier detection during training.

```bash
entrenar train examples/yaml/outlier.yaml
```

**QA Focus**: Outliers flagged, not silently ignored.

### bias.yaml

Bias detection and mitigation.

```bash
entrenar train examples/yaml/bias.yaml
```

**QA Focus**: Demographic parity metrics tracked.

### drift.yaml

Data/model drift detection.

```bash
entrenar train examples/yaml/drift.yaml
```

**QA Focus**: Drift alerts when distribution shifts.

## Section F: Reliability & Checkpoints

### checkpoint.yaml

Checkpoint saving and resumption.

```bash
entrenar train examples/yaml/checkpoint.yaml
```

**QA Focus**: Resume from checkpoint is exact.

### config_validate.yaml

Strict configuration validation.

```bash
entrenar validate examples/yaml/config_validate.yaml
```

**QA Focus**: Invalid configs rejected early.

### long_run.yaml

Extended training duration test.

```bash
entrenar train examples/yaml/long_run.yaml
```

**QA Focus**: No memory leaks over hours.

### locked.yaml

Lockfile for reproducibility.

```bash
entrenar train examples/yaml/locked.yaml
```

**QA Focus**: Lockfile pins all dependencies.

## Section G: Inference & Output

### latency.yaml

Inference latency benchmarking.

```bash
entrenar bench examples/yaml/latency.yaml
```

**QA Focus**: Latency meets SLA.

### json_output.yaml

JSON format output generation.

```bash
entrenar train examples/yaml/json_output.yaml
```

**QA Focus**: JSON is valid and complete.

## Section H: Research & Privacy

### dp.yaml

Differential privacy training.

```bash
entrenar train examples/yaml/dp.yaml
```

**QA Focus**: Privacy budget (epsilon) tracked.

### release.yaml

Production release configuration.

```bash
entrenar train examples/yaml/release.yaml
```

**QA Focus**: All 25 QA points pass.

## Section I: Ecosystem Integration

### session.yaml

Session management with Ruchy.

```bash
entrenar train examples/yaml/session.yaml
```

**QA Focus**: Session state persists correctly.

## Section J: Edge Cases

### soak.yaml

Soak test for extended stability.

```bash
entrenar train examples/yaml/soak.yaml
```

**QA Focus**: System stable over extended period.

## Running All Examples

### Validation Only

```bash
# Validate all YAML configs
for f in examples/yaml/*.yaml; do
  echo "Validating $f..."
  entrenar validate "$f"
done
```

### Integration Tests

```bash
# Run all integration tests
cargo test --test yaml_mode_integration
```

### Quick Reference Table

| File | Scenario | Key Config |
|------|----------|------------|
| `mnist_cpu.yaml` | MNIST CPU baseline | `device: cpu` |
| `csv_data.yaml` | CSV data source | `format: csv` |
| `parquet_data.yaml` | Parquet data | `format: parquet` |
| `multiworker.yaml` | Parallel loading | `num_workers: 4` |
| `dropout.yaml` | Regularization | `dropout: 0.5` |
| `deterministic.yaml` | Reproducibility | `seed: 42, deterministic: true` |
| `citl_suggest.yaml` | CITL suggestions | `citl.mode: suggest` |
| `citl_workspace.yaml` | CITL workspace | `citl.workspace: ...` |
| `custom_arch.yaml` | Custom layers | `architecture.layers: [...]` |
| `llama2_mock.yaml` | LLaMA mock | `source: builtin://llama2-mock` |
| `lora.yaml` | LoRA adapters | `lora.enabled: true` |
| `qlora.yaml` | 4-bit QLoRA | `lora.quantize_bits: 4` |
| `grad_clip.yaml` | Gradient clipping | `gradient.clip_norm: 1.0` |
| `grad_accum.yaml` | Accumulation | `gradient.accumulation_steps: 8` |
| `lr_schedule.yaml` | LR scheduler | `scheduler.name: cosine` |
| `distillation.yaml` | Distillation | `distillation.teacher: ...` |
| `andon.yaml` | Alerts | `monitoring.alerts: [...]` |
| `outlier.yaml` | Outlier detection | `inspect.outliers: true` |
| `bias.yaml` | Bias metrics | `inspect.bias_columns: [...]` |
| `drift.yaml` | Drift detection | `monitoring.drift_detection.enabled: true` |
| `checkpoint.yaml` | Checkpointing | `checkpoint.save_every: 500` |
| `config_validate.yaml` | Validation | `strict_validation: true` |
| `long_run.yaml` | Long training | `epochs: 100` |
| `locked.yaml` | Lockfile | `lockfile: entrenar.lock` |
| `latency.yaml` | Latency bench | `benchmark.target_latency_ms: 50` |
| `json_output.yaml` | JSON output | `report.format: json` |
| `dp.yaml` | Differential privacy | `privacy.dp.enabled: true` |
| `release.yaml` | Production release | `require_peer_review: true` |
| `session.yaml` | Session mgmt | `session.enabled: true` |
| `soak.yaml` | Soak test | `stress.duration_hours: 8` |

## Next Steps

- [QA Process](./qa-process.md) - 25-point checklist
- [YAML Mode Training](./yaml-mode.md) - Complete schema reference
- [Schema Reference](./schema.md) - All configuration options
