# Realizar Integration Issues

## ISSUE-001: Add .apr format support (P0)

**Priority:** P0 - CRITICAL

**Summary:** realizar must support .apr (Aprender native format) as PRIMARY inference format.

**Current State:**
- realizar has GGUF parser (`src/gguf.rs`)
- realizar has safetensors parser (`src/safetensors.rs`)
- realizar does NOT load .apr format

**Required:**
```rust
// Add aprender as optional dependency
aprender = { version = "0.14", optional = true }

// Feature flag
apr-format = ["dep:aprender"]

// API
use realizar::apr::AprModel;
let model = AprModel::load("model.apr")?;
let output = model.generate(&tokens, &config)?;
```

**Acceptance Criteria:**
- [ ] Add aprender dependency (optional, feature-gated)
- [ ] Create `src/apr.rs` module for .apr loading
- [ ] Map aprender ModelType to realizar Model
- [ ] Add `Model::from_apr()` loader
- [ ] Tests with sample .apr files

---

## ISSUE-002: Complete GGUF→Model loader (P1)

**Priority:** P1 - HIGH

**Summary:** realizar has GGUF parser but no high-level `Model::from_gguf()` loader.

**Current State:**
- `GGUFModel::from_bytes()` parses GGUF files
- `GGUFModel::get_tensor_f32()` extracts tensors
- NO automatic weight loading into `Model` struct

**Required:**
```rust
use realizar::gguf::GGUFModel;
use realizar::layers::Model;

// Should work:
let model = Model::from_gguf("model.gguf")?;
let output = model.generate(&tokens, &config)?;
```

**Challenges:**
- Different model architectures (LLaMA, Qwen, Phi, etc.)
- Different tensor naming conventions
- Automatic ModelConfig extraction from metadata

**Acceptance Criteria:**
- [ ] Add `Model::from_gguf()` loader
- [ ] Support LLaMA-style tensor naming
- [ ] Support Qwen tensor naming
- [ ] Extract ModelConfig from GGUF metadata
- [ ] Tests with actual GGUF files

---

## ISSUE-003: Remove ollama workaround from single-shot-eval (P0)

**Priority:** P0 - CRITICAL

**Summary:** single-shot-eval uses ollama CLI instead of sovereign realizar.

**Current State:**
- baselines.rs shells out to `ollama run`
- NOT sovereign - depends on external tool

**Required:**
- Use realizar directly for inference
- Primary format: .apr
- Fallback: .gguf, .safetensors

**Blocked by:** ISSUE-001, ISSUE-002
