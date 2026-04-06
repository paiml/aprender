# Toyota Way QA Process

This document defines the **25-point QA checklist** used for validating all YAML Mode training scenarios. Based on
Toyota Way manufacturing principles.

## Philosophy

Every training run is treated as a **manufacturing process** where quality is built in, not inspected out:

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Stop on defect (NaN/Inf halts training) |
| **Poka-yoke** | Schema validation prevents configuration errors |
| **Genchi Genbutsu** | Go and see - observe actual training metrics |
| **Andon** | Visual alerts for anomalies |
| **Kaizen** | Continuous improvement via experiment tracking |

## The 25-Point QA Checklist

For *every* training scenario, validate these 25 points:

### Category A: Safety & Ethics (5 points)

| # | Check | Description |
|---|-------|-------------|
| A1 | Human Oversight | Operator is present and `andon` system is active |
| A2 | Stop-Mechanism | Process halts immediately on critical failure (NaN/Inf) |
| A3 | Data Privacy | Input data scanned for PII/PHI before ingestion |
| A4 | Bias Check | Training data distribution verified for demographic parity |
| A5 | Impact Analysis | Potential downstream harm of model failure assessed |

### Category B: Data & Inputs (5 points)

| # | Check | Description |
|---|-------|-------------|
| B1 | Source Integrity | Input SHA256 hashes match manifest |
| B2 | Normalization | Input features scaled (0-1 or -1 to 1) correctly |
| B3 | Splitting | Train/Val/Test split is stratified and leak-free |
| B4 | Augmentation | Augmentations are deterministic (fixed seed) |
| B5 | Format | Data types (f32/f16) match hardware capabilities |

### Category C: Compute & Resources (5 points)

| # | Check | Description |
|---|-------|-------------|
| C1 | Resource Cap | Memory usage < 90% of available RAM/VRAM |
| C2 | Compute Affinity | Process pinned to correct CPU cores/GPU device |
| C3 | Thermal Safety | System temperatures monitored during run |
| C4 | Energy Budget | Estimated energy cost < approved budget |
| C5 | Concurrency | No race conditions in multi-thread/multi-GPU dataloading |

### Category D: Process & Training (5 points)

| # | Check | Description |
|---|-------|-------------|
| D1 | Convergence | Loss curve shows monotonic decrease (smoothed) |
| D2 | Generalization | Validation loss tracks training loss (no divergence) |
| D3 | Precision | No underflow/overflow in mixed-precision ops |
| D4 | Determinism | Global seed produces bit-exact reproduction |
| D5 | Checkpointing | Atomic writes for model states; no corruption on crash |

### Category E: Output & Artifacts (5 points)

| # | Check | Description |
|---|-------|-------------|
| E1 | Format Validity | Output `.apr` or `.safetensors` passes validator |
| E2 | Explainability | Saliency maps/attribution generated if required |
| E3 | Versioning | Artifact tagged with git commit and config hash |
| E4 | Performance | Inference latency meets SLA (< 100ms etc.) |
| E5 | Documentation | Run logs and observations archived |

## QA Workflow

### 1. Pre-Training

```bash
# Validate configuration (Poka-yoke)
entrenar validate config.yaml

# Check data integrity
entrenar check-data config.yaml --verify-hashes
```

### 2. During Training

Monitor the terminal dashboard for:
- Loss explosion (Andon alert)
- GPU memory pressure
- Learning rate schedule
- Gradient norms

### 3. Post-Training

```bash
# Validate output artifacts
entrenar verify-output ./experiments/my-run/

# Generate QA report
entrenar qa-report config.yaml --output qa-report.md
```

## Using the Checklist

### For Each Scenario

1. **Run training** with the YAML config
2. **Complete checklist** - mark each of the 25 points
3. **Document exceptions** - note any deviations
4. **Archive results** - store logs and checklist

### Example Checklist (YAML-001: MNIST CPU)

```markdown
## QA Checklist: YAML-001 MNIST Baseline CPU

**Date**: 2025-11-30
**Operator**: @engineer
**Config**: examples/yaml/mnist_cpu.yaml

### Safety & Ethics
- [x] A1: Human oversight - operator present
- [x] A2: Stop mechanism - NaN detection enabled
- [x] A3: Data privacy - MNIST is public domain
- [x] A4: Bias check - balanced digit distribution
- [x] A5: Impact analysis - demo only, no production use

### Data & Inputs
- [x] B1: Source integrity - alimentar verified
- [x] B2: Normalization - 0-1 scaling applied
- [x] B3: Splitting - stratified 80/10/10
- [x] B4: Augmentation - none used
- [x] B5: Format - float32 on CPU

### Compute & Resources
- [x] C1: Resource cap - 2GB RAM used (<8GB available)
- [x] C2: Compute affinity - CPU only
- [x] C3: Thermal safety - N/A for CPU demo
- [x] C4: Energy budget - minimal
- [x] C5: Concurrency - single-threaded

### Process & Training
- [x] D1: Convergence - loss decreased monotonically
- [x] D2: Generalization - val_loss tracked train_loss
- [x] D3: Precision - float32, no mixed precision
- [x] D4: Determinism - seed=42 reproducible
- [x] D5: Checkpointing - epoch checkpoints saved

### Output & Artifacts
- [x] E1: Format validity - safetensors validated
- [x] E2: Explainability - N/A for demo
- [x] E3: Versioning - git commit tagged
- [x] E4: Performance - 50ms inference
- [x] E5: Documentation - logs archived

**Result**: PASS (25/25)
```

## Integration with CI/CD

Add QA gates to your pipeline:

```yaml
# .github/workflows/training-qa.yml
jobs:
  qa-validation:
    steps:
      - name: Validate configs
        run: |
          for f in examples/yaml/*.yaml; do
            entrenar validate "$f"
          done

      - name: Run QA suite
        run: cargo test --test yaml_mode_integration

      - name: Generate QA report
        run: entrenar qa-report --all --output qa-report.md
```

## References

1. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill.
2. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-yoke System*.
3. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*.
4. Poppendieck, M. & T. (2003). *Lean Software Development*.

## Next Steps

- [YAML Examples Catalog](./yaml-examples.md) - All 30 example configurations
- [YAML Mode Training](./yaml-mode.md) - Complete schema reference
- [Quality Gates](../development/quality-gates.md) - Development quality standards
