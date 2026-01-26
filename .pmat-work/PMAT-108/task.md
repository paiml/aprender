# PMAT-108: Fix apr pull to Accept Both URI Formats

**Status:** ✅ COMPLETE
**Priority:** P0 (Blocker)
**Date:** 2026-01-26
**Methodology:** Extreme TDD (Red-Green-Refactor)

## Problem Statement

`apr pull` only accepted full URIs with filename:
```bash
# Works
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Failed (now works)
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
```

## Solution

Added `resolve_hf_uri()` function in `pull.rs` that:
1. If URI already has `.gguf` extension → use as-is
2. If URI is repo-only → query HF API for `.gguf` files → pick Q4_K_M

## Implementation

**File:** `crates/apr-cli/src/commands/pull.rs`

```rust
pub fn resolve_hf_uri(uri: &str) -> Result<String> {
    // If not HF URI or already has .gguf, return unchanged
    if !uri.starts_with("hf://") || uri.to_lowercase().ends_with(".gguf") {
        return Ok(uri.to_string());
    }

    // Query HuggingFace API for repo files
    // Priority: Q4_K_M > Q4_K_S > Q4_0 > Q8_0 > any
    ...
}
```

## Tests (4 passed, 1 integration ignored)

- `test_pmat_108_resolve_uri_with_gguf_extension_unchanged` ✅
- `test_pmat_108_resolve_uri_case_insensitive_gguf` ✅
- `test_pmat_108_resolve_non_hf_uri_unchanged` ✅
- `test_pmat_108_resolve_invalid_uri_fails` ✅
- `test_pmat_108_resolve_qwen_repo_finds_q4_k_m` (integration, ignored)

## Verification

```bash
$ apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
=== APR Pull ===
Model: hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
✓ Downloaded successfully
  Path: /home/noah/.cache/pacha/models/c8490f8cd005ac4e.gguf

$ apr run /home/noah/.cache/pacha/models/c8490f8cd005ac4e.gguf --prompt "Hi!"
Output: I'm just a computer program, but I'm here to help you...
```

## Acceptance Criteria

- [x] `apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` downloads Q4_K_M automatically
- [x] `apr pull hf://org/repo/file.gguf` works unchanged
- [x] `apr run <model>` produces correct output
- [x] All tests pass
