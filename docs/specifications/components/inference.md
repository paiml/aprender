# Inference Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §15
**Status**: Active
**CLI**: `apr run`, `apr chat`
**Implementation**: `crates/apr-cli/src/commands/run.rs`, `chat.rs`
**Library**: `realizar`

---

## 1. Overview

Direct model execution via `apr run` (single prompt) and `apr chat` (interactive).
Supports streaming, GPU/CPU selection, chat templates, inference tracing, and
Roofline profiling.

## 2. CLI Interface

```
apr run <SOURCE> [PROMPT] \
  [--stream] [--max-tokens <N>] [--chat] \
  [--gpu|--no-gpu] [--offline] [--trace] [--profile]

apr chat <FILE> \
  [--temperature <F>] [--top-p <F>] [--max-tokens <N>] \
  [--system <PROMPT>] [--inspect]
```

## 3. Source Types

- Local APR/GGUF/SafeTensors files
- `hf://org/repo` — auto-download and cache
- URLs — download to cache

## 4. Features

- **Streaming**: Token-by-token output
- **Chat template**: ChatML wrapping for Instruct models
- **Offline mode**: Block all network access (sovereign compliance)
- **Tracing**: Layer-by-layer inference trace (APR-TRACE-001)
- **Profiling**: Inline Roofline analysis
- **Benchmark mode**: Output tok/s, latency

## 5. Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | 0=greedy, higher=more random |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `max_tokens` | 32/512 | Token generation limit |

---

## Provable Contracts

### Contract: `inference-v1.yaml`

```yaml
metadata:
  description: "Model inference — run, chat, streaming, sampling"
  depends_on:
    - "softmax-kernel-v1"
    - "sampling-algorithms-v1"

equations:
  greedy_decode:
    formula: "token = argmax(logits)"
    invariants:
      - "Deterministic: same input → same output"
      - "Selected token has highest logit value"

  nucleus_sampling:
    formula: "sample from {t : Σ_{t' with p(t')>=p(t)} p(t') <= top_p}"
    invariants:
      - "Only tokens within top_p cumulative probability considered"
      - "Temperature applied before softmax"
      - "top_p = 1.0 → full distribution"

  temperature_scaling:
    formula: "p_i = softmax(logits_i / T)"
    invariants:
      - "T = 0 → greedy (argmax)"
      - "T → ∞ → uniform distribution"
      - "T > 0 required (validated at construction)"

  kv_cache:
    formula: "cache[layer][pos] = (K, V) for all generated positions"
    invariants:
      - "Cache grows by 1 entry per generated token"
      - "Cached K/V reused, not recomputed"
      - "Cache eviction preserves most recent entries"

proof_obligations:
  - type: invariant
    property: "Greedy determinism"
    formal: "greedy(logits) is deterministic"
  - type: invariant
    property: "Temperature positivity"
    formal: "T > 0 enforced at construction"
  - type: invariant
    property: "Nucleus subset"
    formal: "sampled token always within top_p cumulative mass"
  - type: invariant
    property: "KV cache consistency"
    formal: "cache.len() == num_generated_tokens after generation"

falsification_tests:
  - id: FALSIFY-INF-001
    rule: "Greedy determinism"
    prediction: "same prompt + temp=0 → same output tokens"
    if_fails: "Non-deterministic code path in inference"
  - id: FALSIFY-INF-002
    rule: "Temperature zero"
    prediction: "temp=0 selects argmax token"
    if_fails: "Temperature=0 handling (division by zero or wrong branch)"
  - id: FALSIFY-INF-003
    rule: "Nucleus bound"
    prediction: "sampled token within top_p cumulative probability"
    if_fails: "Nucleus sampling sort or cumsum incorrect"
  - id: FALSIFY-INF-004
    rule: "KV cache growth"
    prediction: "cache size increments by 1 per token"
    if_fails: "Cache append or position tracking bug"
  - id: FALSIFY-INF-005
    rule: "Offline blocks network"
    prediction: "hf:// source + --offline → error, not download"
    if_fails: "Offline flag not checked before network access"

kani_harnesses:
  - id: KANI-INF-001
    obligation: "Greedy determinism"
    property: "argmax is unique for distinct logits"
    bound: 8
    strategy: stub_float
```

### Binding Requirements

```yaml
  - contract: inference-v1.yaml
    equation: greedy_decode
    module_path: "realizar::inference"
    function: sample_greedy
    status: implemented

  - contract: inference-v1.yaml
    equation: nucleus_sampling
    module_path: "realizar::inference"
    function: sample_nucleus
    status: implemented

  - contract: inference-v1.yaml
    equation: kv_cache
    module_path: "realizar::gguf::inference"
    function: forward_single_with_cache
    status: implemented
```
