---
name: find-contracts
description: >
  Find which provable-contracts YAML contracts a Hugging Face model needs,
  create missing contracts, generate Rust artifacts, and implement all stubs.
  Triggers on: "find contracts", "contract coverage", "what contracts does X need",
  "which contracts for", HF model analysis, model contract gap analysis.
argument-hint: "[huggingface-org/model-name]"
---

# Find Contracts for a Hugging Face Model

Full pipeline: analyze a Hugging Face model's `config.json`, determine required
contracts, create missing YAML contracts, generate Rust artifacts, and implement
all stubs to completion (zero-tolerance: no `unimplemented!()` or SATD markers).

## Procedure

### Step 1: Fetch the model config

Fetch the config using curl:

```bash
curl -sL "https://huggingface.co/$ARGUMENTS/resolve/main/config.json"
```

**Error handling:**
- **404**: Report "Model not found: $ARGUMENTS — check the org/model-name spelling"
- **401/403**: Report "Model is gated or private — you may need `huggingface-cli login` or request access at https://huggingface.co/$ARGUMENTS"
- **Non-JSON response**: Report "Unexpected response — this may not be a valid HF model repo"

Parse the JSON and proceed to field extraction.

### Step 2: Extract architecture fields

Extract these fields from `config.json` (use `null` for missing fields):

| Field | Purpose |
|-------|---------|
| `model_type` | Architecture family |
| `architectures` | Model class list |
| `hidden_act` / `activation_function` | Activation function |
| `hidden_size` | Model dimension |
| `intermediate_size` | FFN inner dimension |
| `num_attention_heads` | Query head count |
| `num_key_value_heads` | KV head count (GQA if != num_attention_heads) |
| `num_hidden_layers` | Layer count |
| `vocab_size` | Vocabulary size |
| `rope_theta` | RoPE base frequency |
| `rope_scaling` | RoPE scaling config |
| `partial_rotary_factor` | Partial RoPE (Phi-style) |
| `rms_norm_eps` | RMSNorm epsilon (implies RMSNorm) |
| `layer_norm_eps` / `layer_norm_epsilon` | LayerNorm epsilon (implies LayerNorm) |
| `sliding_window` | Sliding window attention size |
| `tie_word_embeddings` | Tied input/output embeddings |
| `qk_layernorm` / `qk_norm` | QK normalization |
| `attn_logit_softcapping` | Attention logit capping (Gemma2) |
| `multi_query` | Multi-query attention (Falcon) |
| `parallel_attn` | Parallel attention (Falcon) |
| `use_alibi` / `alibi` | ALiBi position encoding |
| `ssm_cfg` / `d_state` / `d_conv` | SSM/Mamba fields |

### Step 3: Map fields to required contracts

Use the mapping in [config-to-contract-mapping.md](references/config-to-contract-mapping.md).

Three categories:

**A) Universal contracts** — every transformer LLM needs these 13:
- `model-config-algebra-v1`
- `softmax-kernel-v1`
- `matmul-kernel-v1`
- `linear-projection-v1`
- `embedding-lookup-v1`
- `embedding-algebra-v1`
- `tensor-shape-flow-v1`
- `cross-entropy-kernel-v1`
- `inference-pipeline-v1`
- `attention-kernel-v1`
- `attention-scaling-v1`
- `kv-cache-sizing-v1`
- `kv-cache-equivalence-v1`

**B) Conditional contracts** — triggered by specific config values. See the
full mapping table in [config-to-contract-mapping.md](references/config-to-contract-mapping.md).

Key rules:
- `hidden_act=silu` → `silu-kernel-v1` + `swiglu-kernel-v1` (most SiLU models use SwiGLU FFN)
- `hidden_act` contains `gelu` → `gelu-kernel-v1`
- `rms_norm_eps` present → `rmsnorm-kernel-v1`
- `layer_norm_eps` present → `layernorm-kernel-v1`
- `rope_theta` present → `rope-kernel-v1`
- `rope_scaling` not null → `rope-extrapolation-v1`
- `partial_rotary_factor` present → `rope-kernel-v1` + `absolute-position-v1`
- `num_key_value_heads != num_attention_heads` → `gqa-kernel-v1`
- `sliding_window` set → `sliding-window-attention-v1`
- `qk_layernorm` or `qk_norm` true → `qk-norm-v1`
- `attn_logit_softcapping` set → `attention-scaling-v1` (already universal, but note capping variant)
- `tie_word_embeddings` true → `tied-embeddings-v1`
- `use_alibi` or `alibi` true → `alibi-kernel-v1`
- SSM fields present → `ssm-kernel-v1`
- GatedDeltaNet architecture → `gated-delta-net-v1` + `hybrid-layer-dispatch-v1` + `conv1d-kernel-v1`

See [known-architectures.md](references/known-architectures.md) for edge cases
(Falcon multi_query, Phi partial_rotary_factor, Gemma2 softcapping, DeepSeek MLA, etc.).

**C) Model-specific contracts** — always check for these:
- `<model_type>-shapes-v1.yaml` (e.g., `qwen35-shapes-v1.yaml`)
- `<model_type>-e2e-verification-v1.yaml` (e.g., `qwen35-e2e-verification-v1.yaml`)

For the model name normalization: use `model_type` from config, lowercase, replace
hyphens with nothing. Examples: `llama`, `qwen2`, `mistral`, `phi`, `gemma2`, `falcon`.

### Step 4: Check existing contracts

Glob `contracts/*.yaml` in the project root to get the list of all existing contracts.
Compare against the required set from Step 3.

### Step 5: Output gap analysis

Format the output in four sections:

#### A) Model Config Summary

```
## Model: $ARGUMENTS
| Field                  | Value            |
|------------------------|------------------|
| model_type             | ...              |
| hidden_act             | ...              |
| hidden_size            | ...              |
| num_attention_heads    | ...              |
| num_key_value_heads    | ...              |
| num_hidden_layers      | ...              |
| rope_theta             | ...              |
| rope_scaling           | ...              |
| rms_norm_eps           | ...              |
| sliding_window         | ...              |
| tie_word_embeddings    | ...              |
```

Only include fields that are present (non-null).

#### B) Contract Coverage Matrix

```
## Contract Coverage

| Contract                    | Status   | Trigger                    |
|-----------------------------|----------|----------------------------|
| model-config-algebra-v1     | EXISTS   | universal                  |
| silu-kernel-v1              | EXISTS   | hidden_act=silu            |
| llama-shapes-v1             | MISSING  | model-specific             |
| ...                         | ...      | ...                        |
```

Status values: `EXISTS`, `MISSING`, `N/A` (not needed for this model).

#### C) Gap Statistics

```
## Coverage: 28/31 contracts (90.3%)
- Universal: 13/13
- Conditional: 12/14 (2 missing)
- Model-specific: 0/2 (2 missing)
```

#### D) Missing Contract Details

For each MISSING contract, output:

```
### MISSING: llama-shapes-v1.yaml

- **Suggested filename**: `contracts/llama-shapes-v1.yaml`
- **depends_on**: `model-config-algebra-v1`
- **Key equations**: Q/K/V projection shapes, FFN shapes for LLaMA-3.1-8B config
- **Source config fields**: hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336
- **Priority**: High (model-specific shape verification)
```

### Step 6: Create missing model-specific YAML contracts

If any model-specific contracts are MISSING (`<model>-shapes-v1.yaml` or
`<model>-e2e-verification-v1.yaml`), create them using the extracted config fields.

**Use existing contracts as templates.** Read a similar contract from `contracts/`
(e.g., `qwen2-shapes-v1.yaml` for a new shapes contract) and adapt:

1. Replace config constants (hidden_size, num_attention_heads, etc.) with the
   target model's values from Step 2
2. Recompute all derived values (d_k = hidden/n_heads, gqa_ratio, expansion_ratio, etc.)
3. Adjust proof obligations, falsification tests, and kani harnesses for the new constants
4. Write to `contracts/<model>-shapes-v1.yaml` and/or `contracts/<model>-e2e-verification-v1.yaml`

**YAML structure** (required sections):

```yaml
metadata:
  version: "1.0.0"
  created: "<today>"
  author: "PAIML Engineering"
  description: "<model> concrete shape instantiation..."
  references: [...]
  depends_on: ["model-config-algebra-v1"]

equations:
  <name>:
    formula: "<equation with concrete values>"
    domain: "<model config description>"
    invariants: [...]

proof_obligations:
  - type: invariant|monotonicity|equivalence|bound|ordering|conservation
    property: "<human-readable name>"
    formal: "<math expression>"
    applies_to: all|simd

falsification_tests:
  - id: FALSIFY-<PREFIX>-NNN
    rule: "<obligation name>"
    prediction: "<expected result>"
    test: "<test method>"
    if_fails: "<what went wrong>"

kani_harnesses:
  - id: KANI-<PREFIX>-NNN
    obligation: <OBLIGATION-ID>
    property: "<description>"
    bound: N
    strategy: exhaustive|bounded_int
    solver: cadical  # optional
    harness: <function_name>

qa_gate:
  id: F-<PREFIX>-001
  name: "<Model> Contract"
  description: "..."
  checks: [...]
  pass_criteria: "All N falsification tests pass"
  falsification: "<example mutation>"
```

**Skip this step** for MISSING universal/conditional contracts — those are generic
and must be authored manually by the team.

### Step 7: Generate Rust artifacts

For each newly created model-specific YAML contract, generate Rust artifacts:

```bash
pv generate contracts/<model>-shapes-v1.yaml -o generated
pv generate contracts/<model>-e2e-verification-v1.yaml -o generated
```

This produces 4 files per contract:
- `generated/<model>-shapes-v1_scaffold.rs` — trait definition
- `generated/<model>-shapes-v1_kani.rs` — Kani proof harnesses
- `generated/<model>-shapes-v1_probar.rs` — property + falsification tests
- `generated/<model>-shapes-v1_book.md` — documentation page

### Step 8: Implement all generated stubs

**ZERO-TOLERANCE: Every `unimplemented!()`, `todo!()`, and SATD marker must be
replaced with complete implementations.** The pre-commit hook will reject any
commit containing these markers.

For each generated `.rs` file:

#### Probar files (`_probar.rs`)

1. Define model config constants at file top level (outside `#[cfg(test)]` module):
   ```rust
   #[allow(dead_code)]
   const HIDDEN: usize = <hidden_size>;
   const N_HEADS: usize = <num_attention_heads>;
   // ... all config constants from the YAML
   ```

2. Replace every `unimplemented!("Wire up: ...")` with deterministic arithmetic:
   - **Shape invariants**: `assert_eq!(N_HEADS * D_K, HIDDEN)`
   - **Divisibility**: `assert_eq!(N_HEADS % N_KV_HEADS, 0)`
   - **RoPE monotonicity**: compute freq vector, assert `freqs[i] > freqs[i+1]`
   - **Parameter count**: sum all weight shapes, assert within expected range
   - **Memory ordering**: compute Q4K/Q6K/F16/F32, assert strict ordering
   - **Throughput monotonicity**: tok/s = bandwidth / model_bytes, assert monotonic

3. Replace every `unimplemented!("Implement falsification test for ...")`:
   - **Deterministic tests**: assert literal constants match (e.g., `28 * 128 == 3584`)
   - **Parametric tests**: sweep over multiple configs/values to stress the property
   - **RoPE tests**: sweep over multiple (base, d_k) pairs

#### Kani files (`_kani.rs`)

1. **Shape harnesses**: use symbolic `kani::any()` with bounded assumptions:
   ```rust
   let n_h: usize = kani::any();
   kani::assume(n_h >= 1 && n_h <= 64);
   kani::assume(n_h % n_kv == 0);
   // ... verify algebraic properties hold for ALL valid configs
   ```

2. **Param count harnesses**: use concrete model constants, verify total in expected range

3. **Quant ordering harnesses**: use symbolic `n`, verify `n*9 < n*13 < n*32 < n*64`
   (bits scaled by 2 to avoid floating point)

#### Scaffold files (`_scaffold.rs`)

1. Add config constants after the trait definition
2. Add a concrete struct (e.g., `pub struct LlamaShapesVerifier;`)
3. Implement all trait methods with real computation writing results to `output`

### Step 9: Compile and verify

Compile all generated files standalone to confirm no stubs remain:

```bash
# Probar files — compile and run tests
for f in generated/<model>-*_probar.rs; do
    rustc --edition 2021 --test "$f" -o /tmp/test_bin && /tmp/test_bin
done

# Scaffold files — compile as library
for f in generated/<model>-*_scaffold.rs; do
    rustc --edition 2021 --crate-type lib "$f" -o /tmp/lib_bin
done

# Kani files — syntax check only (kani::any not available without cargo kani)
# These are verified when `cargo kani` is run separately

# Final scan — zero stubs remaining
grep -rnH 'unimplemented!\|todo!\|FIXME\|HACK\|XXX' generated/<model>-*.rs
# Expected: no output
```

All probar tests must pass. All scaffolds must compile with zero warnings.

### Step 10: Output final summary

After implementation, output the final status:

```
## Pipeline Complete: $ARGUMENTS

### Contracts Created
- contracts/<model>-shapes-v1.yaml (NEW)
- contracts/<model>-e2e-verification-v1.yaml (NEW)

### Artifacts Generated & Implemented
| File | Tests | Status |
|------|-------|--------|
| <model>-shapes-v1_probar.rs | N property + N falsification | ALL PASS |
| <model>-e2e-verification-v1_probar.rs | N property + N falsification | ALL PASS |
| <model>-shapes-v1_kani.rs | N harnesses | COMPILES |
| <model>-e2e-verification-v1_kani.rs | N harnesses | COMPILES |
| <model>-shapes-v1_scaffold.rs | concrete impl | COMPILES |
| <model>-e2e-verification-v1_scaffold.rs | concrete impl | COMPILES |

### Coverage Update
- Before: X/Y contracts (Z%)
- After:  X/Y contracts (Z%)
- Stubs remaining: 0
```

## Non-LLM architectures

If `model_type` indicates a non-LLM architecture (e.g., `vit`, `clip`, `whisper`,
`wav2vec2`), report:

> This model (`model_type=vit`) is not a decoder-only LLM. The contract mapping
> is designed for autoregressive language models. Some contracts (attention-kernel,
> matmul-kernel, etc.) may still apply but the universal set assumes causal decoding.

Still attempt the mapping but flag that coverage analysis may be incomplete.

## Important: SATD Zero-Tolerance Policy

This skill MUST NOT leave any stub markers in generated code. The pre-commit hook
enforces `PMAT_MAX_SATD_COMMENTS=0` and scans all staged `.rs` files for:

- `unimplemented!()` / `todo!()`
- `FIXME` / `HACK` / `XXX`
- `TODO: Replace` / `Wire up:`

If `pv generate` produces scaffolds with these markers, **Steps 8-9 are mandatory**
before the skill is considered complete. Never stop at gap analysis alone.
