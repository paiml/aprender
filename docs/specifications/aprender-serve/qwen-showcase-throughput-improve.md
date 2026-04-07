---
title: "Qwen2.5-Coder-0.5B Showcase Throughput Improvement"
document_id: REALIZAR-QWEN-PERF-001
version: "1.4.0"
status: IMPLEMENTATION
date: 2026-02-01
authors:
  - Claude Code
  - Noah Gift
reviewer: Dr. Karl Popper (AI Agent)
issue_refs:
  - "QWEN-PERF-001"
  - "THROUGHPUT-MVP"
acceptance_criteria:
  - "AC1: GGUF CPU throughput >= 10 tok/s (from 3.0)"
  - "AC2: APR GPU throughput >= 100 tok/s (from 0.9)"
  - "AC3: SafeTensors warm cache >= 50 tok/s (from 0.0)"
  - "AC4: GQA integer division implemented (vLLM pattern)"
  - "AC5: Tokenizer property caching implemented (151K vocab)"
  - "AC6: All existing tests pass (make test-fast)"
  - "AC7: Test coverage maintained >= 80%"
  - "AC8: Zero clippy warnings (make lint)"
  - "AC9: TDG score >= 93.0"
  - "AC10: MQS score >= 500/1000 (from 270)"
citations:
  - "Popper, K. (1959). The Logic of Scientific Discovery. Hutchinson & Co."
  - "Kwon, W. et al. (2023). PagedAttention. SOSP 2023. DOI:10.1145/3600006.3613165"
  - "Shazeer, N. (2019). Fast Transformer Decoding. arXiv:1911.02150"
  - "Dao, T. et al. (2022). FlashAttention. NeurIPS 2022. arXiv:2205.14135"
  - "Williams, S. et al. (2009). Roofline Model. CACM 52(4). DOI:10.1145/1498765.1498785"
---

# Qwen2.5-Coder-0.5B Showcase Throughput Improvement Specification

**Document ID:** REALIZAR-QWEN-PERF-001
**Version:** 1.4.0
**Status:** IMPLEMENTATION
**Date:** 2026-02-01
**Reviewer:** Dr. Karl Popper (AI Agent)
**Classification:** Engineering Specification with Strict Falsification Framework

> "The game of science is, in principle, without end. He who decides one day that scientific statements do not call for any further test, and that they can be regarded as finally verified, retires from the game." — Karl Popper, *The Logic of Scientific Discovery*

---

## Executive Summary: A Refutable Proposal

This document proposes a set of hypotheses explaining the catastrophic throughput failure (MQS Score 270/1000) of Qwen2.5-Coder-0.5B-Instruct. We do not seek to prove these hypotheses correct; rather, we define the exact experimental conditions under which they must be rejected.

**Critical Bottleneck Assertion:** The system is currently failing to saturate *any* hardware resource (Compute, Memory Bandwidth, or PCIe), indicating software logic locks (mutex contention, serialization, or algorithmic complexity) rather than physical limits.

**Primary Prediction:** If logical blocking is removed, throughput MUST scale linearly with memory bandwidth until the Roofline is hit. Any deviation from this falsifies our understanding of the system's architecture.

---

## 1. Problem Statement & Falsifiable Symptoms

### 1.1 Observable State vs. Physical Limits

| Format      | Current (tok/s) | Theoretical Max (tok/s)* | Efficiency | Status |
|-------------|-----------------|--------------------------|------------|--------|
| GGUF (CPU)  | 3.0             | ~35.0                    | 8.5%       | **ANOMALOUS** |
| APR (GPU)   | 0.9             | ~1800.0                  | 0.05%      | **CATASTROPHIC** |
| SafeTensors | 0.0             | ~1800.0                  | 0.0%       | **NON-EXISTENT** |

*\*Theoretical Max calculated based on DDR5-4800 (CPU) and RTX 4090 (GPU) bandwidth for Q4_0/FP16 models. 0.5B model size ~400MB. RTX 4090 BW ~1000GB/s. Max theoretical ~2500 req/s. 1800 conservative.*

### 1.2 The "Impossible" Observation
The fact that `GGUF (CPU)` outperforms `APR (GPU)` (3.0 vs 0.9) falsifies the hypothesis that "GPUs are inherently faster." It proves the existence of a software overhead so severe it negates 1000x raw hardware superiority.

---

## 2. Root Cause Analysis (Deep Falsification)

### 2.1 APR Format: The "Corrupted Tensor" Hypothesis

**Observation:** Throughput is 0.9 tok/s.
**Naïve Cause:** "The code is slow."
**Popperian Analysis:**
1.  **Hypothesis:** The conversion corrupts tensor names/shapes, leading to a fallback "slow path" (CPU emulation or unoptimized kernel).
2.  **Attack:** If tensors were merely named wrong, the graph shouldn't run at all (Crash).
3.  **Refined Root Cause:** The APR converter *force-validates* incorrect tensor mappings (e.g., mapping GQA heads to MHA slots), causing the inference engine to perform excessive broadcasting or memory copies *during* `forward()`, destroying locality.

### 2.2 GGUF CPU: The "Memory Waste" Hypothesis

**Observation:** 3.0 tok/s.
**Hypothesis:** GQA implementation is naively broadcasting KV pairs.
**Mechanism:** Qwen2.5 uses GQA (Grouped Query Attention). If the engine expands KV heads to match Query heads *physically* in memory (x7 duplication), it consumes 7x the necessary bandwidth.
**Falsification:** If we observe memory bandwidth utilization > 80% at 3.0 tok/s, this hypothesis is **CONFIRMED** (we are bandwidth bound by waste). If utilization is < 10%, this hypothesis is **FALSIFIED** (the bottleneck is compute/latency, not bandwidth).

---

## 3. Strict Falsification Protocols (The "F-Tests")

We define "F-Tests" (Falsification Tests). Passing an F-Test means we **failed to disprove** the hypothesis (a success for the theory).

### H1: The APR "Silent Corruption" Hypothesis

**Claim:** APR throughput is < 1 tok/s because the `convert` tool incorrectly maps Qwen2 GQA tensors, forcing the runtime into a "repair loop" or inefficient kernel fallback.

**Prediction (P1):** Correcting the `q_proj` / `k_proj` / `v_proj` mapping logic will result in an immediate throughput jump to > 50 tok/s.

**F-Test 1 (F-H1):**
1.  Patch `src/convert/mod.rs` to correctly handle Qwen2 GQA tensor names.
2.  Convert model.
3.  Run inference.
4.  **FALSIFICATION CONDITION:** Throughput remains < 10 tok/s.
    *   *Interpretation:* If throughput stays low, the mapping was NOT the primary bottleneck. The hypothesis is false.

### H2: The SafeTensors "Cold Cache" Myth

**Claim:** SafeTensors inference is slow/impossible only because of conversion overhead.

**Prediction (P2):** A "Warm" inference run (where conversion artifacts are cached) will be statistically indistinguishable from native APR inference.

**F-Test 2 (F-H2):**
1.  Run SafeTensors inference twice.
2.  Measure Latency(Run 2).
3.  Measure Latency(Native APR).
4.  **FALSIFICATION CONDITION:** `|Latency(Run 2) - Latency(APR)| > 5%`.
    *   *Interpretation:* If cached SafeTensors is still slower, there is runtime overhead *distinct* from conversion (e.g., inefficient memory layout in the cached structure).

### H3: The "Memory Wall" Delusion (GGUF CPU)

**Claim:** We are NOT compute bound; we are bound by redundant memory operations due to poor GQA handling.

**Prediction (P3):** Reducing KV head duplication (via integer division logic) will linearly increase throughput relative to the reduction in memory traffic.

**F-Test 3 (F-H3):**
1.  Measure baseline memory bandwidth (via `perf` or `intel_pcm`) at 3.0 tok/s.
2.  Implement "Virtual Broadcasting" (index remapping).
3.  Measure new throughput.
4.  **FALSIFICATION CONDITION:** Throughput improves < 50% despite removing 85% (6/7ths) of KV memory traffic.
    *   *Interpretation:* If we cut memory traffic but speed doesn't increase, we were never memory bound. The bottleneck lies in instruction dispatch or latency.

---

## 4. Implementation Plan: The "Minimum Viable Fix"

We will implement *only* enough to test the hypotheses.

### Phase 1: The GQA Falsification (Day 1)
*Goal: Prove H3 (GGUF CPU).*
1.  **Refactor:** `attention.rs` to use index remapping for KV heads.
2.  **Measure:** `cargo run --release --example bench_toks -- --model qwen2.5.gguf`
3.  **Result:** If < 10 tok/s, trigger **PIVOT PROTOCOL A** (Investigation of Tokenizer Latency).

### Phase 2: The APR Repair (Day 2)
*Goal: Prove H1 (APR GPU).*
1.  **Refactor:** `src/convert/mod.rs` to handle `model.layers.{i}.self_attn.q_proj` mapping to APR standard.
2.  **Validate:** `cargo run --release --example check_layer_structure`
3.  **Measure:** `realizar run model.apr`
4.  **Result:** If < 50 tok/s, trigger **PIVOT PROTOCOL B** (Kernel Profiling via Nsight).

---

## 5. Peer-Reviewed Optimization Strategies (The "Why")

We adopt strategies only if they have withstood scrutiny in comparable systems (vLLM, TGI).

| Strategy | Source | Justification (Popperian) |
|----------|--------|---------------------------|
| **Integer Div GQA** | vLLM [15] | Removing memory writes is the only way to beat bandwidth limits. |
| **Radix Caching** | TGI [14] | Amortization is required when Setup Cost >> Compute Cost. |
| **Paged Attention** | [11] | Fragmentation is inevitable in long-running processes; contiguous alloc is a falsified strategy for serving. |

---

## 6. The "Catastrophic Failure" Protocol

If all hypotheses are falsified (i.e., we fix mappings and memory, but throughput remains < 5 tok/s):

1.  **Stop Engineering.** Do not "tweak" parameters.
2.  **Audit the Clock.** Verify `std::time::Instant` precision and overhead.
3.  **Audit the Bus.** Verify PCIe link width/speed (`lspci -vv`).
4.  **Audit the Kernels.** Are we launching 1 thread per block? (Common CUDA mistake).

## 7. QA Checklist & Verification Matrix

### Section A: Pre-Flight Controls (Must Pass to Start)
| # | Control | Threshold | Verified? |
|---|---------|-----------|-----------|
| A1 | CPU Frequency | > 3.0 GHz (No Powersave) | [ ] |
| A2 | GPU Link Speed | PCIe Gen4 x16 | [ ] |
| A3 | Background Load | < 1.0 Load Avg | [ ] |

### Section B: Falsification Execution
| # | Test | Hypothesis | Pass/Fail Criteria | Result |
|---|------|------------|--------------------|--------|
| B1 | `test_h1_apr_mapping` | H1 (Tensor Names) | **FAIL** if Throughput < 10 | [ ] |
| B2 | `test_h2_safetensors_cache` | H2 (Cache Warmth) | **FAIL** if Delta > 5% | [ ] |
| B3 | `test_h4_gqa_broadcast` | H4 (Mem BW) | **FAIL** if Speedup < 2x | [ ] |

### Section C: Final Review
- [ ] **Independent Reviewer:** Confirm F-Tests were run honestly.
- [ ] **Code Owner:** Confirm no "hacks" (e.g., hardcoded skipping of layers) were used to achieve numbers.

---

## 8. PMAT Compliance Requirements

All implementation work MUST maintain PMAT quality gates. These are non-negotiable constraints.

### 8.1 Quality Gate Thresholds

| Metric | Threshold | Current | Command |
|--------|-----------|---------|---------|
| **TDG Score** | ≥ 93.0 (A Grade) | 93.9 | `pmat analyze tdg` |
| **Dead Code** | ≤ 15% | 34.5% ⚠️ | `pmat quality-gate --checks dead-code` |
| **Complexity** | Cognitive ≤ 25, Cyclomatic ≤ 30 | 147 violations | `pmat analyze complexity` |
| **SATD** | 0 critical, ≤ 10 total | 34 violations | `pmat analyze satd` |
| **Provability** | ≥ 0.70 | 0.65 ⚠️ | `pmat quality-gate --checks provability` |
| **Test Coverage** | ≥ 80% lines | 80.08% | `make coverage` |

### 8.2 Pre-Commit Protocol

Before any commit related to this spec, run:

```bash
# Tier 1: Sub-second check (ON-SAVE)
make tier1

# Tier 2: Pre-commit gate (30s target)
make tier2

# Full quality gate
pmat quality-gate --fail-on-violation
```

### 8.3 SATD Elimination (Self-Admitted Technical Debt)

Any new code MUST NOT introduce SATD comments. Existing SATD in modified files MUST be resolved:

```bash
# Check for SATD in modified files
pmat analyze satd --format detailed

# SATD markers to eliminate:
# - TODO, FIXME, HACK, XXX, OPTIMIZE
# - Any "Performance:" or "Security:" flagged comments
```

### 8.4 Complexity Control

New functions MUST comply with:

| Type | Max Allowed | Tool |
|------|-------------|------|
| Cognitive Complexity | 20 (recommended), 25 (max) | `pmat analyze complexity` |
| Cyclomatic Complexity | 25 (recommended), 30 (max) | `pmat analyze complexity` |
| Function Length | 100 lines | `.clippy.toml` |

**If refactoring increases complexity beyond thresholds, the change is REJECTED.**

### 8.5 Dead Code Prevention

```bash
# Run dead code analysis
pmat quality-gate --checks dead-code --max-dead-code 15.0

# Current violations requiring cleanup:
# - src/quantize/simd.rs (94.7% dead)
# - src/quantize/fused_k.rs (80.0% dead)
# - src/quantize/activation.rs (84.2% dead)
# - src/quantize/parallel_dequant.rs (73.7% dead)
```

**Rule:** Any file touched by this spec work MUST reduce dead code, not increase it.

### 8.6 Provability Enhancement

Target: Increase provability score from 0.65 to ≥ 0.70

```bash
# Check provability
pmat quality-gate --checks provability

# Add provability annotations:
# - #[invariant(...)] for state invariants
# - #[requires(...)] for preconditions
# - #[ensures(...)] for postconditions
# - assert!() for runtime checks
```

### 8.7 F-Test: PMAT Compliance

**Hypothesis (H-PMAT):** All optimization work can be completed while maintaining quality gates.

**Falsification Condition:** If any of the following occur, STOP and refactor first:
1. TDG Score drops below 90.0
2. Dead code exceeds 20%
3. New SATD comments introduced
4. Test coverage drops below 75%

**F-Test Command:**
```bash
# Must pass before and after each phase
pmat quality-gate --format summary --fail-on-violation

# If this fails, the implementation is REJECTED regardless of throughput gains.
```

---

## 9. Makefile Integration

Ensure the following targets are used during development:

```bash
# Fast feedback loop
make tier1          # Sub-second: check + clippy
make tier2          # 30s: fmt + clippy + test --lib
make test-fast      # 5 min: excludes heavy tests

# Quality gates
make lint           # Auto-format + clippy --fix + zero warnings
make coverage       # llvm-cov with 95% target

# Full validation before merge
make quality-gates  # All gates must pass
```

---

**References:**
[4] Popper, K. (1959). *The Logic of Scientific Discovery*. (We assert that all engineering metrics must be testable).
[11] Kwon, W., et al. (2023). *PagedAttention*. SOSP 2023.
[14] HuggingFace. (2024). *Text Generation Inference*. GitHub.
[15] vLLM Team. (2024). *vLLM*. GitHub.
[16] PMAT. (2025). *Professional Multi-language Analysis Toolkit*. Batuta Stack.

**Signed:**
*Dr. Karl Popper (Agent Proxy)*
*Date: 2026-02-01*

---

**Document Revision History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-01 | Initial specification with 6 hypotheses |
| 1.1.0 | 2026-02-01 | Added vLLM/TGI optimization strategies |
| 1.2.0-POPPER | 2026-02-01 | Peer review with strict falsification framework |
| 1.3.0-PMAT | 2026-02-01 | Added PMAT compliance requirements (Section 8-9) |