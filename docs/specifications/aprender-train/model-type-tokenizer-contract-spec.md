# SPEC-MODEL-TYPE: Model Type Taxonomy and Tokenizer Embedding Contract

Version: 1.0
Status: proposed
Date: 2026-04-10

**Document ID:** SPEC-MODEL-TYPE-001
**Version:** 1.0.0
**Status:** PROPOSED
**Author:** PAIML Engineering
**Date:** 2026-04-10
**Priority:** P0 -- Defect discovered during SHIP-TWO (PMAT-521 ALB-010 teacher loading)
**Parent:** SPEC-SHIP-TWO-001
**Contracts:** `contracts/aprender/model-format-conversion-v1.yaml`, `contracts/aprender/tensor-layout-v1.yaml`
**Existing Gate:** F-APR-SELF-CONTAINED-001 (currently over-broad)
**New Contract:** `contracts/aprender/model-type-taxonomy-v1.yaml` (F-MODEL-TYPE-001)
**PMAT Epic:** PMAT-526 (subtasks: PMAT-527..531)
**Cookbook Validation:** `apr-cookbook/examples/creation/create_apr_linear_regression.rs` (ML model, no tokenizer)

**Citations:**
- [C1] Baltrusaitis, Ahuja, Morency (2019). "Multimodal Machine Learning: A Survey and Taxonomy." arXiv:1705.09406
- [C2] Patro & Agneeswaran (2026). "LLMOrbit: A Circular Taxonomy of Large Language Models." arXiv:2601.14053
- [C3] Casey, Damian, Cotaj, Santos (2025). "An Empirical Study of Safetensors' Usage Trends." arXiv:2501.02170

---

## 1. Abstract

The existing tokenizer embedding contract (F-APR-SELF-CONTAINED-001) states "APR files
MUST embed tokenizer" but does not distinguish between model categories. This caused two
failures:

1. **False negative (P0):** `qwen3-coder-30b-q4k.apr` was produced without a tokenizer
   and passed all validation gates. Inference failed at runtime with "Tokenizer encode
   failed" (PMAT-172). The contract existed but had no enforcement in the F32 write path.

2. **Over-broad enforcement (latent):** `write_apr_file_raw` has
   `#[requires(tokenizer.is_some())]` which would break non-LLM models (Whisper, Bert)
   if they ever used this code path.

Root cause: the contract system has no concept of **model categories**. It treats all
APR files identically regardless of whether the model is a text-generating LLM, an audio
encoder, or an embedding model.

---

## 2. Five Whys

1. ALB-010 teacher model fails inference → "Tokenizer encode failed"
2. APR file has no embedded tokenizer → `write_apr_file()` accepted `tokenizer=None`
3. No assertion on the F32 write path → `write_apr_file_raw` has `#[requires]` but
   `write_apr_file` does not
4. Contract F-APR-SELF-CONTAINED-001 says "all APR files" but doesn't distinguish model
   categories → enforcement is either too strict (breaks Whisper) or too loose (misses LLMs)
5. No model category taxonomy in the contract system → `Architecture` enum exists in code
   but contracts don't reference it

---

## 3. Prior Art: How Other Frameworks Handle This

No major framework or format enforces tokenizer requirements at the serialization level.
This is a gap across the entire ecosystem — and an opportunity for APR to lead.

### 3.1 HuggingFace Transformers (PyTorch)

HuggingFace uses **task-based dispatch** via `AutoModelFor*` classes. The taxonomy is
implicit in the class name — each maps `config.model_type` to both a model class and
a preprocessor class:

| Task Class | Preprocessor | Examples |
|------------|-------------|----------|
| `AutoModelForCausalLM` | `AutoTokenizer` (required) | GPT, LLaMA, Qwen |
| `AutoModelForImageClassification` | `AutoImageProcessor` (no tokenizer) | ViT, DINOv2 |
| `AutoModelForAudioClassification` | `AutoFeatureExtractor` (no tokenizer) | Wav2Vec2 |
| `AutoModelForSpeechSeq2Seq` | `AutoProcessor` (tokenizer + audio) | Whisper |

**Key insight:** The preprocessor requirement is implicit in the API class, not declared
in the model format. `config.json` has `model_type` but no `requires_tokenizer` field.

### 3.2 Candle (Rust)

119 model modules with **zero formal categorization**. No shared trait, no `AutoModel`
equivalent. Each example binary manually decides what to load. Tokenizer handling is
per-example: `dinov2` loads no tokenizer; `whisper` loads tokenizer + mel features.

### 3.3 Unsloth

Runtime detection hierarchy: Vision-Language → Audio → Embedding → Language Model
(default). Detection uses heuristics (presence of `vision_config`, special audio tokens,
`sentence-transformers` Hub tag). All categories still load a tokenizer/processor —
unsloth is LLM-focused and doesn't support classical ML.

### 3.4 GGUF Format

`general.architecture` identifies model family but **no metadata key indicates input
modality or tokenizer requirements**. Tokenizer data (`tokenizer.ggml.*`) is embedded in
every file — even Whisper, which primarily uses mel spectrograms. GGUF was designed for
LLMs; Whisper is the only non-text architecture.

### 3.5 SafeTensors Format

Stores only tensors and tensor metadata (name, shape, dtype). **Zero model-level
metadata** [C3]. All classification comes from companion `config.json`.

### 3.6 Gap in the Ecosystem

Model formats are **modality-blind** [C1]. They store weights and optionally tokenizer
data but never formally declare the model category or preprocessor requirements. This is
always left to the application layer.

**APR can be the first format to embed category metadata** — declaring `category: llm`
vs `category: ml` vs `category: audio` in the file itself. This makes validation
self-contained: `apr validate` can check tokenizer presence against declared category
without needing external `config.json`.

### 3.7 apr-model-qa-playbook: Kernel Classes vs Model Categories

The QA playbook (`paiml/apr-model-qa-playbook`) classifies models by **kernel
equivalence class** (A-F + SSM + Linear), grouping architectures that share identical
compute kernels:

| Class | Kernels | Families |
|-------|---------|----------|
| A | GQA + RMSNorm + SiLU + SwiGLU + RoPE | LLaMA, Qwen, Mistral, Phi-3/4, DeepSeek |
| B | MHA + LayerNorm + GELU | GPT-2, OPT, **Whisper**, **BERT**, Phi-2 |
| C | MQA + LayerNorm + GELU + ALiBi | BLOOM, Falcon-40B |
| D | Mixed LayerNorm (gegelu/SiLU) | Phi-3-small, **Moonshine** |
| E | MoE + GQA + RMSNorm + SwiGLU | Mixtral, Qwen-MoE |
| F | RMSNorm + GELU + GatedMlp + RoPE | Gemma |
| SSM | Selective scan, no attention | Mamba |
| Linear | WKV recurrence, no softmax | RWKV7 |

**Gap discovered:** Whisper and BERT are in Class B alongside GPT-2 — correct for
kernel dispatch but **incorrect for tokenizer requirements**. The QA playbook has
256 model playbooks, all LLMs. Zero non-LLM playbooks. The G0 gateway checks
integrity/layout/format but does not distinguish model categories for tokenizer
validation.

**Cross-cutting concern:** Kernel class (compute dispatch) is orthogonal to model
category (input modality). A model can be Class B (MHA + LayerNorm + GELU) and
simultaneously category `audio` (Whisper) or category `embedding` (BERT). The
taxonomy in this spec adds the missing modality dimension.

**Integration point:** `apr-qa` gateway G0 should use `is_llm()` from this spec
to decide whether tokenizer absence is a G0 failure (LLM) or acceptable (audio/ML).
Contract: `contracts/aprender/model-type-taxonomy-v1.yaml` equation `validate_tokenizer_presence`.

---

## 4. Model Category Taxonomy

### 4.1 Architecture Enum (source of truth)

From `crates/aprender-core/src/format/converter_types.rs:100-124`:

```rust
pub enum Architecture {
    Auto,       // Auto-detect from tensor names
    Whisper,    // OpenAI Whisper (audio)
    Llama,      // Meta LLaMA family
    Bert,       // Google BERT (embedding/classification)
    Qwen2,      // Alibaba Qwen2/2.5
    Qwen3,      // Alibaba Qwen3
    Qwen3_5,    // Alibaba Qwen3.5 (hybrid attention)
    Gpt2,       // OpenAI GPT-2 / StarCoder
    Phi,        // Microsoft Phi-3/4
    GptNeoX,    // EleutherAI GPT-NeoX / Pythia
    Opt,        // Meta OPT
}
```

### 4.2 Four Model Categories

| Category | Architectures / Models | Text Tokenizer Required? | Rationale |
|----------|----------------------|--------------------------|-----------|
| **LLM** (text generation) | Llama, Qwen2, Qwen3, Qwen3_5, Gpt2, Phi, GptNeoX, Opt | **YES** | Causal LM: tokenizer encodes prompt, decodes output |
| **ML** (classical/statistical) | LinearRegression, LogisticRegression, DecisionTree, RandomForest, GBM, NaiveBayes, KNN, SVM, KMeans, PCA, ARIMA, ICA, GLM, BayesianLR | **NO** | Numeric input (Vec<f32> / Matrix); no text encoding |
| **Audio** (speech-to-text) | Whisper, Moonshine | **NO** | Speech feature extraction, mel spectrogram input |
| **Embedding** (classification) | Bert | **NO** | No text generation; classification head output |

The **ML category** is the largest by model count. These are aprender's core v0.4-v0.7
models — they operate on numeric feature vectors, not text. They may be serialized as
APR files (model weights, hyperparameters) but never require a tokenizer. This is the
primary reason `write_apr_file` accepts `tokenizer: Option` — it was designed for ML
models first, LLM support came later.

### 4.3 Model Family Contracts

17 YAML contracts in `contracts/model-families/`:

| Family | Category | Contract File |
|--------|----------|---------------|
| llama | LLM | `llama.yaml` |
| qwen2 | LLM | `qwen2.yaml` |
| qwen3 | LLM | `qwen3.yaml` |
| qwen3_5 | LLM | `qwen3_5.yaml` |
| gpt2 | LLM | `gpt2.yaml` |
| phi | LLM | `phi.yaml` |
| deepseek | LLM | `deepseek.yaml` |
| mistral | LLM | `mistral.yaml` |
| falcon_h1 | LLM | `falcon_h1.yaml` |
| gemma | LLM | `gemma.yaml` |
| openelm | LLM | `openelm.yaml` |
| mamba | LLM | `mamba.yaml` |
| rwkv7 | LLM | `rwkv7.yaml` |
| whisper | Audio | `whisper.yaml` |
| moonshine | Audio | `moonshine.yaml` |
| bert | Embedding | `bert.yaml` |

---

## 5. Three Format Paths

| Format | Source | Tokenizer Availability | Write Function |
|--------|--------|----------------------|----------------|
| **GGUF** | llama.cpp | Always embedded in metadata | `write_apr_file_raw()` |
| **SafeTensors** | HuggingFace | Sibling `tokenizer.json` — may be absent | `write_apr_file()` |
| **APR** | Aprender native | Self-contained if source had it | N/A (already APR) |

---

## 6. Contract Gap Analysis

### 5.1 Current State (Broken)

| Write Function | `#[requires(tokenizer.is_some())]` | Runtime Guard | Result |
|----------------|-----------------------------------|---------------|--------|
| `write_apr_file_raw` (GGUF path) | YES (line 116) | None | Correct for GGUF but wrong abstraction level |
| `write_apr_file` (F32/SafeTensors path) | **NO** | None | **DEFECT: LLMs can be written without tokenizer** |

### 5.2 Required State

The guard belongs at the **caller level**, conditioned on model category:

| Caller | Model Category | Tokenizer Contract |
|--------|---------------|-------------------|
| `apr import model.gguf` | Any | GGUF always has tokenizer — pass through |
| `apr import model.safetensors` (LLM) | LLM | **MUST** find `tokenizer.json` or **FAIL** |
| `apr import model.safetensors` (Audio) | Audio | Tokenizer optional — speech models use mel features |
| `apr import model.safetensors` (Embedding) | Embedding | Tokenizer optional — no generation |
| `apr convert` | Any | Preserve tokenizer from source format |
| `apr quantize` | Any | Preserve tokenizer from input APR |
| `apr merge` | LLM | All inputs must have tokenizer; output must too |

---

## 7. Acceptance Criteria

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| AC-MT-001 | `Architecture` enum has `is_llm()` method | Returns true for all LLM variants | Unit test |
| AC-MT-002 | LLM import from SafeTensors fails without `tokenizer.json` | Error, not silent success | Integration test |
| AC-MT-003 | Audio import from SafeTensors succeeds without `tokenizer.json` | No error | Integration test |
| AC-MT-004 | GGUF import always produces APR with tokenizer | All GGUF families | Existing tests |
| AC-MT-005 | `apr validate` warns on LLM APR missing tokenizer | Warning in output | CLI test |
| AC-MT-006 | Model family contracts include `category` field | All 17 YAML files | Schema validation |
| AC-MT-007 | F-APR-SELF-CONTAINED-001 scoped to LLM category only | Contract YAML updated | Contract review |

---

## 8. Falsification Tests

| ID | Hypothesis Falsified If... | Mitigation |
|----|---------------------------|------------|
| FALSIFY-MT-001 | LLM SafeTensors import succeeds without tokenizer.json | Add category check in import pipeline |
| FALSIFY-MT-002 | Whisper import fails when no text tokenizer provided | Ensure audio category skips tokenizer requirement |
| FALSIFY-MT-003 | `apr validate` reports LLM APR as valid when tokenizer missing | Add tokenizer presence check for LLM category |
| FALSIFY-MT-004 | `is_llm()` returns false for any LLM architecture | Exhaustive match test |
| FALSIFY-MT-005 | `is_llm()` returns true for Whisper or Bert | Exhaustive match test |
| FALSIFY-MT-006 | GGUF import produces tokenizer-less APR for any architecture | GGUF always has tokenizer — this is a converter bug |
| FALSIFY-MT-007 | `AprConverter::to_apr()` fails without tokenizer | ML model creation path must not require tokenizer |
| FALSIFY-MT-008 | apr-cookbook `create_apr_*` examples fail | Tokenizer contract broke ML model creation workflow |

---

## 9. Implementation Plan

### Phase 1: Taxonomy (code changes)

Add `is_llm()` method to `Architecture` enum in
`crates/aprender-core/src/format/converter_types.rs`:

```rust
impl Architecture {
    /// Returns true for text-generating LLM architectures that require a tokenizer.
    /// Audio (Whisper) and embedding (Bert) models do not require text tokenizers.
    pub fn is_llm(&self) -> bool {
        matches!(self, Self::Llama | Self::Qwen2 | Self::Qwen3 | Self::Qwen3_5
            | Self::Gpt2 | Self::Phi | Self::GptNeoX | Self::Opt | Self::Auto)
    }
}
```

Note: `Auto` returns `true` because auto-detected models from GGUF/SafeTensors
default to LLM assumption (fail-safe). Classical ML models never flow through
the GGUF/SafeTensors import path — they use aprender's native serialization
(`Estimator::save()` / `Estimator::load()`), which does not call `write_apr_file`.

### Phase 2: Import guard

In the SafeTensors import path, after architecture detection, add:

```rust
if options.architecture.is_llm() && tokenizer.is_none() {
    return Err(anyhow!("F-APR-SELF-CONTAINED-001: LLM model requires tokenizer. \
        Provide --tokenizer <path> or place tokenizer.json alongside the model."));
}
```

### Phase 3: Validate guard

In `apr validate`, check:
- If APR metadata contains LLM-like tensors (attention, ffn) but no tokenizer → warn
- If APR metadata has architecture field → use `is_llm()` directly

### Phase 4: Contract updates

1. Update `contracts/aprender/model-format-conversion-v1.yaml`:
   - Scope `apr_tokenizer_embedding` to LLM category
   - Add `model_category` field to equation preconditions

2. Update `contracts/aprender/tensor-layout-v1.yaml`:
   - Scope F-APR-SELF-CONTAINED-001 to LLM category
   - Add FALSIFY-MT-001..006 tests

3. Add `category` field to all 17 model family YAML contracts:
   ```yaml
   category: llm    # or: audio, embedding
   ```

---

## 10. Files to Modify

| File | Change |
|------|--------|
| `crates/aprender-core/src/format/converter_types.rs` | Add `is_llm()` to `Architecture` |
| `crates/aprender-core/src/format/converter/import.rs` | Guard SafeTensors LLM import |
| `crates/apr-cli/src/commands/validate.rs` | Warn on LLM APR without tokenizer |
| `contracts/aprender/model-format-conversion-v1.yaml` | Scope to LLM category |
| `contracts/aprender/tensor-layout-v1.yaml` | Scope F-APR-SELF-CONTAINED-001 |
| `contracts/model-families/*.yaml` (17 files) | Add `category` field |

---

## 11. References

| Reference | Location |
|-----------|----------|
| F-APR-SELF-CONTAINED-001 | `contracts/aprender/tensor-layout-v1.yaml:374` |
| apr_tokenizer_embedding equation | `contracts/aprender/model-format-conversion-v1.yaml:127` |
| Architecture enum | `crates/aprender-core/src/format/converter_types.rs:100` |
| write_apr_file (F32 path, no guard) | `crates/aprender-core/src/format/converter/write.rs:324` |
| write_apr_file_raw (GGUF path, has guard) | `crates/aprender-core/src/format/converter/write_model_config.rs:116` |
| PMAT-172 (missing tokenizer error) | `crates/aprender-serve/src/apr/loading.rs` |
| SHIP-TWO parent spec | `docs/specifications/aprender-train/ship-two-models-spec.md` |
| Provable contract | `contracts/aprender/model-type-taxonomy-v1.yaml` (F-MODEL-TYPE-001) |
| QA playbook gateway contract | `apr-model-qa-playbook/contracts/gateway-contract-v1.yaml` (G0-G4) |
| QA format invariants | `apr-model-qa-playbook/contracts/apr-format-invariants-v1.yaml` (I-1..I-5) |
| Kernel class taxonomy | `apr-model-qa-playbook/crates/apr-qa-gen/src/kernel_class.rs` (A-F + SSM + Linear) |
| apr-cookbook ML creation | `apr-cookbook/examples/creation/create_apr_linear_regression.rs` |
| apr-cookbook from-scratch | `apr-cookbook/examples/creation/create_apr_from_scratch.rs` |
| apr-cookbook roundtrip contract | `apr-cookbook/contracts/apr-format-roundtrip-v1.yaml` |

---

*End of specification SPEC-MODEL-TYPE-001.*
