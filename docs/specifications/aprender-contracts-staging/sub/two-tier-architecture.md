# 26. Two-Tier Architecture and Compositional Contracts

Version: 1.0
Status: proposed
Date: 2026-04-09

## Two-Tier Contract Layout

Contracts are organized in two tiers:

```
contracts/
  # Tier 1: Generic kernel contracts (algorithm-level)
  softmax-kernel-v1.yaml          "How softmax works"
  matmul-kernel-v1.yaml           "How matmul works"
  attention-kernel-v1.yaml        "How attention works" (depends_on: softmax)
  inference-pipeline-v1.yaml      "How inference works" (depends_on: attention, rmsnorm, ...)
  roofline-model-v1.yaml          "Performance bound model"
  mqs-scoring-v1.yaml             "Model quality scoring"
  ...

  # Tier 2: Per-library contracts + bindings
  aprender/
    binding.yaml                  Maps generic contracts → aprender functions
    tokenizer-loading-v1.yaml     Library-specific contract
    training-loop-v1.yaml         Library-specific contract
  trueno/
    binding.yaml                  Maps generic contracts → trueno SIMD functions
    tiled-matmul-shader-v1.yaml   Library-specific contract
  entrenar/
    binding.yaml                  Maps generic contracts → entrenar GPU functions
    cuda-classify-training-v1.yaml
  realizar/
    binding.yaml                  Maps generic contracts → realizar orchestration
```

**Tier 1** contracts define the math — equations, invariants, proof obligations,
Kani harnesses. They are algorithm-specific, not library-specific. The same
`softmax-kernel-v1.yaml` governs every library that implements softmax.

**Tier 2** contracts are per-library. Each subdirectory contains:
1. `binding.yaml` — maps Tier 1 equations to the library's actual functions
2. Library-specific contracts — contracts that only apply to that library

## How Bindings Connect the Tiers

One generic contract serves multiple libraries through per-library bindings:

```
softmax-kernel-v1.yaml (the algorithm)
  ├── aprender/binding.yaml:  softmax → aprender::nn::functional::softmax
  ├── trueno/binding.yaml:    softmax → trueno::blis::softmax::softmax_avx2
  ├── entrenar/binding.yaml:  softmax → entrenar::kernels::softmax_forward
  └── realizar/binding.yaml:  softmax → realizar::gpu::softmax_wgsl
```

Each binding entry maps `(contract, equation)` to `(function, module_path, status)`.
The `bindings_for(stem)` method resolves this at runtime.

## The Composition Problem

Current contracts verify individual kernels in isolation. But the sovereign
stack is a **pipeline**: tokens flow through trueno's kernels, composed by
realizar's orchestrator, served by aprender's CLI. The question:

> If trueno's softmax is correct AND trueno's matmul is correct, is
> realizar's attention layer correct?

This is compositional verification. Three levels of composition exist:

### Level 1: Intra-Contract Composition (SOLVED)

Contracts already use `depends_on` to declare dependencies:

```yaml
# attention-kernel-v1.yaml
metadata:
  depends_on: [softmax-kernel-v1]
equations:
  attention:
    formula: "Attention(Q,K,V) = softmax(QK^T/√d_k) · V"
```

The `pv graph` command visualizes this DAG. Kani harnesses use
`strategy: compositional` to stub verified sub-components.

### Level 2: Cross-Contract Pipeline Composition (PARTIALLY SOLVED)

`inference-pipeline-v1.yaml` composes multiple kernels into a pipeline:

```yaml
metadata:
  depends_on:
    - softmax-kernel-v1
    - attention-kernel-v1
    - rmsnorm-kernel-v1
    - embedding-algebra-v1
equations:
  prefill_phase:
    formula: "H_L = layer_L(... layer_1(embed(tokens)))"
  decode_step:
    formula: "h_t = layer_L(... layer_1(embed(token_t), kv_cache))"
  layer_composition:
    formula: "h_{l+1} = h_l + sublayer(norm(h_l))"
```

This verifies the composition of algorithms but **not** the composition
of implementations across repos.

### Level 3: Cross-Repo Pipeline Contracts (NOT YET IMPLEMENTED)

**This is what's missing.** When the call chain spans repos:

```
User request
  → aprender::serve::handler (apr-cli)
    → realizar::pipeline::forward_pass
      → trueno::blis::rmsnorm
      → trueno::blis::attention (→ trueno::blis::softmax + trueno::blis::matmul)
      → trueno::blis::swiglu
    → realizar::pipeline::sample
  → response
```

Each repo binds the same kernel contracts independently, but nobody
verifies that **trueno's softmax output format matches realizar's
attention input expectation**. The type system catches shape mismatches,
but invariant composition (e.g., "softmax output sums to 1.0, which
attention depends on for valid weight normalization") is not checked.

## Design: Cross-Repo Pipeline Bindings

To solve Level 3, we need **pipeline binding files** that declare
cross-repo data flow:

```yaml
# contracts/pipelines/inference-forward-v1.yaml
metadata:
  version: "1.0.0"
  description: "Cross-repo inference pipeline: trueno → realizar → aprender"
  pipeline: true

stages:
  - name: tokenize
    repo: aprender
    binding: aprender/binding.yaml
    contract: bpe-tokenization-v1
    equation: encode
    output_invariant: "token_ids ∈ [0, vocab_size)"

  - name: embed
    repo: aprender
    binding: aprender/binding.yaml
    contract: embedding-lookup-v1
    equation: embedding_lookup
    input_requires: "token_ids ∈ [0, vocab_size)"
    output_invariant: "shape = [seq_len, d_model], all finite"

  - name: transformer_block
    repo: trueno
    binding: trueno/binding.yaml
    repeat: num_layers
    stages:
      - contract: rmsnorm-kernel-v1
        equation: rmsnorm
        input_requires: "shape = [seq_len, d_model], all finite"
        output_invariant: "shape preserved, unit variance"
      - contract: attention-kernel-v1
        equation: attention
        input_requires: "normalized hidden states"
        output_invariant: "shape = [seq_len, d_model], all finite"
      - contract: swiglu-kernel-v1
        equation: swiglu
        output_invariant: "shape = [seq_len, d_model]"

  - name: decode
    repo: realizar
    binding: realizar/binding.yaml
    contract: sampling-algorithms-v1
    equation: sample
    input_requires: "logits shape = [vocab_size], all finite"
    output_invariant: "token_id ∈ [0, vocab_size)"

cross_boundary_obligations:
  - property: "Tokenizer output valid for embedder"
    from_stage: tokenize
    to_stage: embed
    formal: "∀t ∈ encode(text): 0 ≤ t < vocab_size"

  - property: "Embedding output valid for transformer"
    from_stage: embed
    to_stage: transformer_block
    formal: "shape(embed(tokens)) = [len(tokens), d_model] ∧ all_finite"

  - property: "Transformer output valid for sampler"
    from_stage: transformer_block
    to_stage: decode
    formal: "shape(H_L) = [seq_len, d_model] ∧ all_finite"
```

## Verification Strategy for Pipelines

```
                        [Compositional Kani]
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        [trueno stubs]  [realizar stubs]  [aprender stubs]
              │               │               │
        softmax_verified rmsnorm_verified tokenize_verified
        matmul_verified  attention_verified embed_verified
```

Each repo's Kani harnesses verify individual kernels. Pipeline
verification uses `strategy: compositional` — stub the verified
sub-components and verify only the composition glue:

1. **Input/output type compatibility** — output invariant of stage N
   implies input precondition of stage N+1
2. **Shape flow** — tensor dimensions are compatible across boundaries
3. **Numeric stability** — finite inputs produce finite outputs at
   every stage (no NaN propagation)

## Implementation Plan

| Phase | What | Tool |
|-------|------|------|
| P1 | `pv pipeline` CLI command | Parse pipeline YAML, validate cross-boundary obligations |
| P2 | Pipeline bindings | New YAML schema with `stages` + `cross_boundary_obligations` |
| P3 | Pipeline scoring | D6 dimension: fraction of pipeline stages with verified boundaries |
| P4 | Pipeline Kani | Compositional harnesses that stub verified stages |

## Sovereign Stack Pipeline Map (25 crates)

```
batuta (orchestrator, 196K LOC)
  ├── Analysis:    depyler, decy, bashrs, ruchy
  ├── Inference:   aprender → realizar → trueno
  ├── Training:    entrenar → trueno
  ├── Serving:     alimentar, renacer, pacha
  ├── Quality:     certeza, pmat, probar
  ├── Distributed: repartir, pepita
  ├── Viz:         presentar, trueno-viz
  └── Storage:     trueno-db, trueno-graph, trueno-rag, trueno-zram
```

The critical pipeline for contract verification:

```
tokens → aprender(embed) → trueno(rmsnorm,attn,ffn)×L → realizar(sample) → token
         ├── bpe-tokenization-v1    ├── rmsnorm-kernel-v1     ├── sampling-algorithms-v1
         ├── embedding-lookup-v1    ├── attention-kernel-v1
         └── special-tokens-v1     ├── swiglu-kernel-v1
                                   └── roofline-model-v1
```

## Sovereign Stack Enforcement Status (25 crates, measured 2026-03-27)

| Level | Crates | Description |
|-------|--------|-------------|
| **Full L3** | aprender, entrenar, realizar, ruchy (4/25) | build.rs + binding.yaml + trait tests |
| **L2** | trueno, bashrs (2/25) | Partial (build.rs or traits, not both) |
| **Paper only** | depyler, decy, presentar (3/25) | binding.yaml exists, no compile-time enforcement |
| **None** | 16/25 crates | No contracts at all (~502K LOC uncontracted) |

## Batuta Oracle: Sovereign Stack Component Map

```
batuta oracle "transformer inference pipeline"
  → entrenar (training, 85%)
  → realizar (serving, 85%)
  → trueno (SIMD backend, 80%)
  Integration pattern: training_to_inference
```

The oracle confirms the critical three-repo pipeline:
**trueno** (SIMD kernels) → **realizar** (orchestration) → **aprender** (serving).
This is the pipeline that needs cross-repo compositional contracts first.

## Theoretical Foundations

The compositional contract design draws from established formal methods:

**Assume-Guarantee Contracts.** Dardik & Kang (2025) show that
decomposing a system into components with assume-guarantee contracts
allows inferring local inductive invariants per component, whose
conjunction forms a global system invariant. This directly maps to our
pipeline model: each stage's `output_invariant` is the next stage's
`input_requires` — the assume-guarantee pair.

> "The conjunction of all local invariants becomes an inductive
> invariant for the entire system." — arXiv:2509.06250

**Kani Function Contracts.** Kani's `#[kani::requires]` /
`#[kani::ensures]` / `#[kani::modifies]` with `stub_verified`
attribute (RFC 0009, stable since Kani 0.33.0) enables modular
verification: prove a function satisfies its contract, then replace
calls with contract stubs in downstream harnesses. This is exactly
the compositional strategy for cross-repo pipeline verification.

> "Contracts enable divide-and-conquer verification — prove a method
> satisfies its contract, then replace calls by permitted behaviors."
> — Kani Function Contracts RFC (2024)

**Roofline Performance Bounds.** Yuan et al. (2024) apply the
roofline model to LLM inference, showing decode is memory-bound and
prefill is compute-bound. Our `roofline-model-v1.yaml` + `pv roofline`
CLI implement these equations as contract-derived performance ceilings.

> "During decode, all computations are memory-bound, resulting in
> performance significantly below computational capacity."
> — arXiv:2402.16363

**Rust Verification Landscape.** Le Blanc & Lam (2024) survey
Rust verification tools including Kani (bounded model checking),
Creusot (deductive verification with prophecies), and Flux
(refinement types). Our stack uses Kani for L4 and Lean 4 for L5.

> "Bounded model checking is a good choice for Rust verification."
> — arXiv:2410.01981

**Compositional Neural Network Verification.** Duong et al. (2025)
apply assume-guarantee reasoning to neural network verification,
decomposing networks into sub-components verified independently.
The same principle applies to our transformer block pipeline: verify
each kernel (softmax, matmul, rmsnorm) independently, then compose.

## References (Section 26)

- Dardik & Kang (2025). "Compositional Inductive Invariant Inference
  via Assume-Guarantee Reasoning." arXiv:2509.06250
- Incer et al. (2023). "Pacti: Scaling Assume-Guarantee Reasoning
  for System Analysis and Design." arXiv:2303.17751
- Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline
  Model Insights." arXiv:2402.16363
- Le Blanc & Lam (2024). "Surveying the Rust Verification Landscape."
  arXiv:2410.01981
- Matsushita et al. (2024). "Lessons Learned from Verifying the Rust
  Standard Library." arXiv:2510.01072
- Kani Team (2024). "Function Contracts for Kani." RFC 0009.
  model-checking.github.io/kani
- Denis, Jourdan & Marché (2022). "Creusot: A Foundry for the
  Deductive Verification of Rust Programs." ICFEM 2022.
- Duong et al. (2025). "Compositional Neural Network Verification
  via Assume-Guarantee Reasoning."
- Williams et al. (2009). "Roofline: An Insightful Visual Performance
  Model for Multicore Architectures." CACM 52(4).
