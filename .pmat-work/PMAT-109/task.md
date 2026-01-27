# PMAT-109: Qwen2.5-Coder-1.5B GGUF Produces Garbage Output

**Status:** ✅ COMPLETE
**Priority:** P0 (Blocker)
**Date:** 2026-01-27
**Fixed In:** realizar v0.6.10

## Problem Statement

```bash
$ apr run /home/noah/.cache/pacha/models/c8490f8cd005ac4e.gguf --prompt "Hi"
Output: akakakakakakakakakakakakakakakakakakakakakakakakakakakakakakakak
```

The 0.5B model worked correctly, but 1.5B produced garbage.

## Root Cause

The inference code detected model architecture and instruct status from the **filename**:

```rust
// WRONG: Cached models have hash filenames like "c8490f8cd005ac4e.gguf"
let is_instruct = model_name.to_lowercase().contains("instruct");
```

For cached models downloaded via `apr pull`, the filename is a hash, so:
1. Architecture was detected as "Transformer" (no "qwen" in hash)
2. `is_instruct` was false (no "instruct" in hash)
3. Chat template was NOT applied
4. Prompt "Hi" was tokenized as 1 raw token instead of ChatML format
5. Model produced garbage without proper chat context

## Fix

Modified `realizar/src/infer/mod.rs` to use GGUF metadata instead of filename:

```rust
// CORRECT: Get architecture from GGUF metadata (works with any filename)
let gguf_arch = mapped.model.architecture().unwrap_or("transformer");
let is_instruct_arch = matches!(
    gguf_arch.to_lowercase().as_str(),
    "qwen2" | "qwen" | "llama" | "mistral" | "phi" | "phi3"
);
```

Now all Qwen2/LLaMA/Mistral/Phi models automatically apply their chat template regardless of filename.

## Verification

```bash
# Before fix (garbage)
$ apr run c8490f8cd005ac4e.gguf --prompt "Hi"
Output: akakakakakakakakakakakakak

# After fix (correct)
$ apr run c8490f8cd005ac4e.gguf --prompt "Hi"
Architecture: Qwen2 [GGUF: qwen2] (28 layers, vocab_size=151936)
Output: I'm sorry, but I'm not sure what you're asking...

$ apr run c8490f8cd005ac4e.gguf --prompt "What is 2+2?"
Output: 2 + 2 equals 4.
```

## Acceptance Criteria

- [x] `apr run <cached-hash.gguf> --prompt "Hi"` produces coherent output
- [x] Architecture detected from GGUF metadata, not filename
- [x] Chat template applied for known instruct architectures (qwen2, llama, mistral, phi)
- [x] Works with both cached (hash) and original filenames
