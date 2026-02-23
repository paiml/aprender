# Enforce Provable Design by Contract Across the Sovereign AI Stack

**Reference**: Meyer, B. (1992). "Applying 'Design by Contract'." *IEEE Computer*, 25(10), 40-51.

**Status**: SPECIFICATION — filed as GitHub issues for tracking
**Date**: 2026-02-23
**Scope**: trueno, realizar, aprender, entrenar, batuta, provable-contracts, apr-playbook

---

## 1. Meyer's Formal Framework (1992)

Meyer defines three mechanisms that together constitute a provable contract:

| Mechanism | Definition | Rust Equivalent |
|-----------|-----------|-----------------|
| **Precondition** | "Requirements that any call must satisfy to be correct" (p.42) | Function arguments validated before use; `Result::Err` on violation |
| **Postcondition** | "Properties that are ensured in return by the execution of the call" (p.42) | Return type guarantees; `debug_assert!` on output invariants |
| **Class Invariant** | "A property that applies to all instances of the class" (p.45) | Struct fields that are always valid after construction; newtype wrappers |

Meyer's **No Hidden Clauses** rule (p.41): "Every 'obligation' entry also in a certain sense describes a 'benefit' by stating that the constraints given are the only relevant ones." In Rust: if a function accepts `GGUFConfig`, the config MUST be self-contained — the function must NOT reach outside it for silent fallbacks.

Meyer's **rejection of defensive programming** (p.41): "Such blind and often redundant checking causes much of the complexity and unwieldiness that often characterizes software." In Rust: `.unwrap_or(151645)` is exactly the "blind checking" Meyer warns against — it masks the bug (missing EOS) instead of surfacing it.

Meyer's **Who should check?** principle (p.44): "The rejection of defensive programming means that the client and supplier are not both held responsible for a consistency condition. Either the condition is part of the precondition and must be guaranteed by the client, or it is not stated in the precondition, and must be handled by the supplier." In Rust: either the CALLER validates metadata before constructing a config, or the CONFIG CONSTRUCTOR validates — never both, and never neither.

---

## 2. Ecosystem-Wide Falsification Results

Comprehensive sweep across all 7 repositories on 2026-02-23:

| Repository | CRITICAL | HIGH | MEDIUM | LOW | Total |
|------------|----------|------|--------|-----|-------|
| **realizar** | 3 | 7 | 5 | 1 | 16 |
| **aprender + apr-cli** | 8 | 17 | 6 | 3 | 34 |
| **trueno** | 0 | 13 | 14 | 9 | 36 |
| **entrenar** | 3 | 3 | 4 | 0 | 10 |
| **batuta** | 0 | 1 | 5 | 5 | 11 |
| **provable-contracts** | 3 gaps | — | — | — | 3 |
| **TOTAL** | **17** | **41** | **34** | **18** | **110** |

---

## 3. The 11 CRITICAL Findings

Each violates Meyer's framework in a specific, provable way.

### C-01: Inconsistent Default Architecture — "llama" vs "qwen2"

**Location**: `realizar/src/gguf/loading.rs:183` vs `realizar/src/gguf/loader_apr_quantized.rs:171`

**Pattern**:
```rust
// loading.rs — defaults to "llama"
.unwrap_or_else(|| "llama".to_string())

// loader_apr_quantized.rs — defaults to "qwen2"
.unwrap_or_else(|| "qwen2".to_string())
```

**Meyer violation**: Class Invariant. The same model produces different `GGUFConfig.architecture` depending on which code path loads it. Meyer (p.45): "The invariant must be preserved by every exported routine of the class." Two loaders constructing the same class produce inconsistent state.

**Fix**: `architecture` must be `Result<String>`, not `Option<String>` with divergent defaults. If architecture cannot be determined, loading MUST fail.

---

### C-02: Inconsistent rope_theta — 10,000 vs 1,000,000

**Location**: 12 sites across 10 files in realizar

**Pattern**:
```rust
// GGUF APR loaders:
rope_theta.unwrap_or(1_000_000.0)  // Qwen2

// Every other path:
rope_theta.unwrap_or(10000.0)      // LLaMA
```

**Meyer violation**: No Hidden Clauses. The contract for `GGUFConfig.rope_theta` implicitly depends on which loader was called — a hidden clause. The 100x difference produces completely different positional encodings.

**Fix**: `rope_theta` must be derived from `architecture` via contract lookup, not from per-loader defaults. Add to `arch-constraints-v1.yaml`.

---

### C-03: Qwen2-1.5B Dimensions as Universal Defaults

**Location**: `realizar/src/gguf/loading.rs:169-173` and `loader_apr_quantized.rs:158-162`

**Pattern**:
```rust
hidden_size.unwrap_or(1536)       // Qwen2-1.5B
num_layers.unwrap_or(28)          // Qwen2-1.5B
num_heads.unwrap_or(12)           // Qwen2-1.5B
num_kv_heads.unwrap_or(2)         // Qwen2-1.5B
intermediate_size.unwrap_or(8960) // Qwen2-1.5B
```

**Meyer violation**: Precondition. Meyer (p.44): "The stronger the precondition, the heavier the burden on the client, and the easier for the supplier." These missing fields should be a STRONG precondition (must be present) rather than silently defaulted. The current code has NO precondition — any `None` is silently accepted.

**Fix**: All five fields must return `Result::Err` when absent. No model should load without explicit dimensions.

---

### C-04: Hardcoded Qwen2 EOS in apr-cli Chat Pipeline

**Location**: `apr-cli/src/commands/chat_generate_session_02.rs:315`

**Pattern**:
```rust
let eos_token_id = self.extract_apr_eos_token().unwrap_or(151645);
```

**Meyer violation**: No Hidden Clauses. The chat session's contract claims to work with any model, but silently assumes Qwen2's EOS token. On LLaMA (EOS=2), generation never stops at the correct token.

**Fix**: EOS must come from model config. If model has no EOS, chat session must refuse to start (precondition violation).

---

### C-05: Unconditional Hardcoded EOS for SafeTensors

**Location**: `apr-cli/src/commands/chat_generate_safetensors.rs:12`

**Pattern**:
```rust
let eos_id = 151645u32; // Qwen2 EOS token
```

**Meyer violation**: Precondition eliminated entirely. This isn't even a fallback — it's an unconditional hardcoded constant for ALL SafeTensors models. Not `.unwrap_or()`, just a raw assignment.

**Fix**: Read `eos_token_id` from `config.json` (which already has it — see `SafetensorsConfig.eos_token_id`). Fail if not present.

---

### C-06: Hardcoded Qwen2 Stop Tokens in Chat Generation

**Location**: `apr-cli/src/commands/chat_generate_session_02.rs:223,420`

**Pattern**:
```rust
stop_tokens: vec![151645, 151643], // <|im_end|>, <|endoftext|>
```

**Meyer violation**: Class Invariant. `QuantizedGenerateConfig.stop_tokens` is supposed to represent "tokens that stop generation for THIS model" but instead contains Qwen2-specific constants for ALL models. The invariant (stop_tokens matches the loaded model) is violated at construction.

**Fix**: Populate `stop_tokens` from the model's config (GGUFConfig or AprTransformerConfig `eos_token_id`), not from hardcoded constants.

---

### C-07: LLaMA-7B Dimensions as Universal Converter Defaults

**Location**: `aprender/src/format/converter/metadata.rs:9-17`

**Pattern**:
```rust
let hidden_size = apr_metadata.hidden_size.unwrap_or(4096);
let num_layers = apr_metadata.num_layers.unwrap_or(32);
let num_heads = apr_metadata.num_heads.unwrap_or(32);
let vocab_size = apr_metadata.vocab_size.unwrap_or(32000);
let intermediate_size = apr_metadata.intermediate_size.unwrap_or(11008);
let rope_theta = apr_metadata.rope_theta.unwrap_or(1_000_000.0);
```

**Meyer violation**: Precondition. Nine critical parameters silently default to a mixture of LLaMA-7B dimensions (4096/32/32000/11008) and Qwen2 rope_theta (1M). A Qwen2 model with missing metadata exports as a LLaMA-7B-shaped GGUF file — silent, plausible, completely wrong.

**Fix**: `apr_to_gguf_metadata()` must require all fields or return `Err`.

---

### C-08: Qwen2-0.5B Dimensions in CLI Extract

**Location**: `apr-cli/src/commands/gguf.rs:415-418,426`

**Pattern**:
```rust
let hidden_dim = mapped.model.embedding_dim().unwrap_or(896);
let num_heads = mapped.model.num_heads().unwrap_or(14);
let num_kv_heads = mapped.model.num_kv_heads().unwrap_or(2);
let num_layers = mapped.model.num_layers().unwrap_or(28);
// ...
t.dims.first().copied().unwrap_or(4864) as usize
```

**Meyer violation**: Postcondition. The function `extract_model_dims` claims to extract dimensions from a model but actually fabricates them when extraction fails. Meyer (p.44): "A postcondition violation is a bug in the supplier. The routine failed to deliver on its promises."

**Fix**: Return `Result<ModelDims>` instead of silently fabricating dimensions.

---

### C-09: SpecialTokens::default() Hardcodes Qwen2

**Location**: `aprender/src/demo/mod.rs:138-142`

**Pattern**:
```rust
impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_id: 151643,
            eos_id: 151645,
            pad_id: 151643,
            im_start_id: 151644,
            im_end_id: 151645,
        }
    }
}
```

**Meyer violation**: Class Invariant. Meyer (p.45): "Every creation procedure of the class must yield an object satisfying the invariant." The `Default` creation procedure produces an object whose invariant ("these token IDs match the loaded model") is violated for every non-Qwen2 model. Since `Default` is Rust's zero-config path, this is the most dangerous implicit assumption.

**Fix**: Remove `Default` impl. Require explicit construction from model metadata. `SpecialTokens::from_config(config: &ModelConfig) -> Result<Self>`.

---

### C-10: entrenar Silent Vocab Size Fallback

**Location**: `entrenar/src/config/train/loader.rs:423`

**Pattern**:
```rust
unwrap_or(QWEN_VOCAB_SIZE as u64)  // 151,936
```

**Meyer violation**: Precondition. Training with wrong vocab size corrupts the embedding matrix — a 32K-vocab model trained with 151K-vocab assumptions wastes memory and produces garbage embeddings.

**Fix**: `vocab_size` must be a required field in training config. Return error if missing.

---

### C-11: entrenar Silent Hidden Size Fallback

**Location**: `entrenar/src/config/train/loader.rs:403`

**Pattern**:
```rust
unwrap_or(QWEN_HIDDEN_SIZE as u64)  // 896
```

**Meyer violation**: Precondition. Training a model with wrong hidden_size produces shape mismatches in every linear layer. This should be a hard error, not a silent default to Qwen2-0.5B's dimension.

**Fix**: `hidden_size` must be a required field. Return error if missing.

---

## 4. Three Missing Provable Contracts

The `provable-contracts/` repository has 89 YAML contracts (26 kernel, 20 architecture, 16 ML, 8 time-series, 9 data/format, 4 E2E). Three critical gaps:

### Gap 1: `special-tokens-registry-v1.yaml` — MISSING

No contract maps architecture to BOS/EOS/PAD token IDs. This is WHY token IDs are hardcoded across 30+ sites.

**Required content**:
```yaml
architectures:
  qwen2:
    bos_token_id: 151643
    eos_token_id: 151645
    pad_token_id: 151643
  llama:
    bos_token_id: 128000
    eos_token_id: 128001
    pad_token_id: 128001
  mistral:
    bos_token_id: 1
    eos_token_id: 2
    pad_token_id: 0
  # ... every supported architecture
```

### Gap 2: `model-metadata-bounds-v1.yaml` — MISSING

No contract defines valid ranges for model dimensions, rope_theta, or epsilon. This is WHY arbitrary defaults are scattered.

**Required content**:
```yaml
fields:
  hidden_dim:
    required: true
    min: 64
    max: 65536
    must_be_divisible_by: num_heads
  vocab_size:
    required: true
    min: 100
    max: 1000000
  rope_theta:
    required: false
    default_by_architecture:
      qwen2: 1000000.0
      llama: 10000.0
      mistral: 10000.0
      # ...
```

### Gap 3: `tokenizer-vocab-v1.yaml` — MISSING

No contract ties tokenizer type to vocabulary size and special token mapping.

---

## 5. Systemic Root Causes

### Root Cause 1: `Option<T>` Without Validation Constructor

Every critical finding traces to the same pattern:

```rust
pub struct GGUFConfig {
    pub hidden_dim: usize,  // Set by unwrap_or(1536)
    pub vocab_size: usize,  // Set by unwrap_or(32000)
    // ...
}
```

The struct has no validation — any combination of values is accepted. Meyer's solution: validated construction.

**Fix pattern — ValidatedModelConfig newtype**:
```rust
/// Invariant: all fields are present, consistent, and within bounds.
/// Cannot be constructed except through `validate()`.
pub struct ValidatedModelConfig(ModelConfigInner);

impl ValidatedModelConfig {
    pub fn validate(raw: RawModelMetadata) -> Result<Self, ContractViolation> {
        let hidden_dim = raw.hidden_dim
            .ok_or(ContractViolation::MissingField("hidden_dim"))?;
        let num_heads = raw.num_heads
            .ok_or(ContractViolation::MissingField("num_heads"))?;

        // Class invariant: hidden_dim divisible by num_heads
        if hidden_dim % num_heads != 0 {
            return Err(ContractViolation::Invariant(
                format!("hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads})")
            ));
        }

        Ok(Self(ModelConfigInner { hidden_dim, num_heads, /* ... */ }))
    }
}
```

### Root Cause 2: Multiple Loaders With Independent Defaults

There are 5+ code paths that construct `GGUFConfig` (loading.rs, loader_apr_quantized.rs, config.rs, embedding.rs, safetensors), each with its own set of `.unwrap_or()` defaults. The defaults diverge (C-01, C-02).

**Fix pattern — single canonical constructor**:
```rust
impl GGUFConfig {
    /// The ONLY way to construct a GGUFConfig.
    /// All other construction paths must call this.
    pub fn from_validated(v: ValidatedModelConfig, constraints: ArchConstraints) -> Self {
        // Single source of truth for all field mappings
    }
}
```

### Root Cause 3: Contracts Exist But Are Not Wired

trueno has `contracts.rs` with `validate_weight_buffer()`, `validate_gemv_shapes()`, `validate_f32_buffer()` — but **zero callers from kernel code**. provable-contracts has 89 YAML files — but `special-tokens-registry-v1.yaml` (the one that would fix C-04 through C-09) doesn't exist.

**Fix**: Every YAML contract must have a build.rs codegen step that produces const arrays, and every loader/kernel must use the generated code.

---

## 6. Implementation Roadmap

### Phase 1: Stop the Bleeding (Week 1)
- [ ] Create `special-tokens-registry-v1.yaml` with codegen
- [ ] Replace all 11 CRITICAL sites with contract lookups
- [ ] File GitHub issues for each finding (C-01 through C-11)

### Phase 2: Validated Constructors (Week 2-3)
- [ ] Create `ValidatedModelConfig` newtype in realizar
- [ ] Funnel all `GGUFConfig` construction through single validated path
- [ ] Create `model-metadata-bounds-v1.yaml` with codegen
- [ ] Replace all `.unwrap_or(dimension)` with `ok_or_else()?`

### Phase 3: Kernel Contract Enforcement (Week 4)
- [ ] Wire `trueno/contracts.rs` validators into kernel dispatch
- [ ] Replace ad-hoc `expect()` in Q4K/Q6K kernels with contract validation
- [ ] Deprecate colmajor kernel exports with `#[deprecated]`

### Phase 4: Full Provability (Week 5-6)
- [ ] Create `tokenizer-vocab-v1.yaml`
- [ ] Create `chat-template-semantics-v1.yaml`
- [ ] Audit all 110 findings resolved
- [ ] Add CI gate: `grep -r "unwrap_or(151" src/` must return zero matches

---

## 7. Falsification Gate

The following command must produce ZERO matches to consider DbC complete:

```bash
# Magic number fallbacks (model-specific constants as silent defaults)
rg 'unwrap_or\((151|128|50256|32000|4096|1536|896|768|8960|11008)' \
   --type rust -g '!*test*' -g '!*bench*' -g '!*example*' \
   ~/src/{trueno,realizar,aprender,entrenar,batuta}/src/

# Architecture string heuristics in production code
rg 'contains\("(qwen|llama|mistral|phi|gpt|bert|gemma)"' \
   --type rust -g '!*test*' -g '!*bench*' -g '!*example*' \
   ~/src/{trueno,realizar,aprender,entrenar,batuta}/src/

# Hardcoded token IDs outside contract definitions
rg '\b151645\b|\b151643\b|\b128001\b|\b128000\b' \
   --type rust -g '!*test*' -g '!*bench*' -g '!*example*' -g '!*default_eos*' -g '!*registry*' \
   ~/src/{trueno,realizar,aprender,entrenar,batuta}/src/
```

Each must return **zero lines** for the ecosystem to be considered provably DbC-compliant per Meyer 1992.

---

## References

1. Meyer, B. (1992). "Applying 'Design by Contract'." *IEEE Computer*, 25(10), 40-51.
2. Meyer, B. (1988). *Object-Oriented Software Construction*. Prentice Hall.
3. Hoare, C.A.R. (1969). "An Axiomatic Basis for Computer Programming." *Comm. ACM*, 12(10), 576-580.
4. Floyd, R.W. (1967). "Assigning Meanings to Programs." *Proc. Am. Math. Soc. Symp. in Applied Math.*, 19, 19-31.
