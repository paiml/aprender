# Rosetta Format Conversion (Archived from Section 10)

> Archived from `docs/specifications/qwen2.5-coder-showcase-demo.md`, Section 10 (lines 918-982).

## 10. Rosetta Format Conversion (Simplified Provenance)

### Canonical Import Path: SafeTensors (NOT GGUF)

**SafeTensors is the ONLY canonical import source for APR files.**

GGUF files are pre-quantized with mixed quant formats (Q4_K, Q5_0, Q6_K, Q8_0) that APR
cannot always represent exactly. `apr import` enforces exact passthrough — it REJECTS
quant formats it cannot preserve (Q4_0, Q4_1, Q5_0, Q5_K, Q8_0). This is by design:
import must be lossless.

SafeTensors files contain F16/BF16/F32 weights with no quantization decisions baked in.
Quantization is applied during import via `--quantize`, giving full control over the
output format. This is the correct provenance chain:

```
SafeTensors (F16/BF16) ──apr import──► APR (native) ──apr export──► GGUF (for ollama)
                           ▲                                           │
                           │                                           ▼
                      Ground truth                              ollama parity target
```

GGUF import exists only for diagnostic comparison (`apr diff`, `apr validate`), NOT as a
production import path.

### The Three Primary Paths

| # | Conversion | Command | Status |
|---|-----------|---------|--------|
| 1 | SafeTensors -> APR (canonical) | `apr import model.safetensors -o model.apr` | **Pass** (4 shards, 339 tensors, Q4_K, 4.0 GB, correct inference on CPU) |
| 2 | APR -> GGUF (export for ollama) | `apr export model.apr --format gguf -o model.gguf` | **Pass** (functional, but dequantizes Q4K→F32: 4GB→28GB. Quant-preserving export needed.) |
| 3 | Full chain: ST -> APR -> GGUF | `apr rosetta chain "st -> apr -> gguf" --input model.safetensors` | **Pass** (steps 1+2 both work; GGUF output is F32, needs quant-preserving export) |

### Conversion Verification Tools

```bash
# Round-trip verification
apr rosetta verify original.apr roundtrip.apr --tolerance 1e-6

# Per-tensor fingerprint for corruption detection
apr rosetta fingerprint model.apr > fingerprint.json

# Validate tensor statistics against reference
apr rosetta validate-stats model.apr --reference fingerprint.json

# Compare inference output
apr rosetta compare-inference model.apr model.gguf --prompt "2+2="
```

### Jidoka Stop Conditions

Conversion halts immediately on: NaN, Inf, dimension mismatch, tensor count mismatch, checksum failure, vocab size mismatch, architecture mismatch.

### Rosetta Falsification Gates (F-ROSETTA-*)

| ID | Prediction | Test | Expected | Status |
|----|-----------|------|----------|--------|
| F-ROSETTA-001 | ST->APR preserves tensor count | `apr tensors` on both, compare count | Identical tensor count | **Pass** (both APR and GGUF: 339 tensors; APR 3.99 GB all Q4_K, GGUF 4.34 GB mixed Q4_K/Q6_K) |
| F-ROSETTA-002 | SafeTensors->APR->GGUF roundtrip produces valid output | `apr import` ST->APR, `apr export` APR->GGUF, `apr run` GGUF | Correct inference output | **Pass** (PMAT-252 raw passthrough: ST→APR (4.0GB Q4K) → GGUF (4.0GB Q4K) zero-loss, inference outputs "2+2 is 4" — correct math, minor tokenizer rendering artifacts) |
| F-ROSETTA-003 | Chain command produces valid GGUF | `apr export model.apr --format gguf` then `apr run` on output | Correct inference output | **Pass** (PMAT-252: raw Q4K block passthrough, 339 tensors, 4.0 GiB, weights bit-identical to APR source, inference correct) |
| F-ROSETTA-004 | Fingerprint detects tensor corruption | Flip 1 byte in APR file, re-fingerprint | Different fingerprint hash | **Pass** (3 tests: single-byte corruption via sign-bit flip, stability for identical data, small perturbation detection; all use `compute_tensor_stats` checksums) |
| F-ROSETTA-005 | NaN in source halts conversion | Inject NaN into SafeTensors tensor | Jidoka stop, exit != 0 | **Pass** (compute_tensor_validation NaN detection verified in rosetta) |
| F-ROSETTA-006 | Vocab size mismatch halts conversion | Modify vocab_size in config.json | Jidoka stop, "vocab size mismatch" | **Pass** (import.rs vocabulary validation verified, PMAT-232) |

---
