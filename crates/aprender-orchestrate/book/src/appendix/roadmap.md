# Appendix I: Roadmap

Current status of Sovereign AI Stack components, planned features, and community contribution areas.

## Stack Component Status

| Component | Version | Maturity | Notes |
|-----------|---------|----------|-------|
| trueno | 0.14.x | Stable | SIMD/GPU primitives |
| trueno-db | 0.3.x | Beta | GPU-first analytics DB |
| trueno-zram-core | 0.3.x | Beta | SIMD compression |
| repartir | 2.0.x | Stable | Distributed compute |
| aprender | 0.24.x | Stable | ML algorithms, APR v2 |
| entrenar | 0.5.x | Beta | Training, LoRA/QLoRA |
| realizar | 0.5.x | Beta | Inference engine |
| whisper-apr | 0.1.x | Alpha | Pure Rust Whisper ASR |
| simular | 0.1.x | Alpha | Simulation engine |
| jugar | 0.1.x | Alpha | Game engine |
| alimentar | 0.2.x | Beta | Parquet/Arrow loading |
| pacha | 0.2.x | Beta | Model registry |
| renacer | 0.9.x | Stable | Syscall tracing |
| batuta | 0.6.x | Beta | Orchestration |

## Planned Features

### Near-Term

| Feature | Component | Description |
|---------|-----------|-------------|
| Plugin API | batuta | Custom transpiler plugins |
| ONNX import | realizar | Direct ONNX model loading |
| WebGPU compute | trueno | Browser GPU acceleration |

### Medium-Term (3-6 Months)

| Feature | Component | Description |
|---------|-----------|-------------|
| Go transpiler | batuta | Go to Rust transpilation |
| Model merge | entrenar | TIES/DARE/SLERP strategies |
| Speculative decoding | realizar | Draft model acceleration |

### Long-Term (6-12 Months)

| Feature | Component | Description |
|---------|-----------|-------------|
| Self-hosted training | entrenar | Full training without Python |
| Federated learning | entrenar + repartir | Privacy-preserving distributed training |

## Community Contribution Areas

| Level | Areas |
|-------|-------|
| Beginner | Docs, Oracle recipes, test coverage, clippy fixes |
| Intermediate | Dependency mappings, benchmarks, ARM SIMD, WASM compat |
| Advanced | Transpiler plugins, GPU kernels, distributed strategies |

## Version Policy

Components follow semver. Targeting 1.0 requires: 95%+ coverage, stable API, complete docs.

```bash
batuta stack versions          # Check current versions
make stack-outdated            # Find outdated deps
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
