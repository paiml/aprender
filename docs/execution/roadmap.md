# PMAT Development Roadmap

## Completed: v0.4.0 llama.cpp Performance Parity
- **Duration**: 2026-01-01 to 2026-01-15
- **Priority**: P0
- **Status**: ✅ Complete

## Next Sprint: v0.9.0 Autograd Engine - PyTorch-Compatible Automatic Differentiation
- **Duration**: 2025-11-25 to 2025-12-16
- **Priority**: P0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

## Current Sprint: v0.26.0 GGUF Export Metadata Completeness
- **Duration**: 2026-02-08 to 2026-02-15
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 20, SATD = 0, Coverage ≥ 80%

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| GH-253-1 | Export complete tokenizer metadata (token_type, eos/bos/pad IDs, chat_template) | ✅ done | M | P0 |
| GH-253-2 | Fix vocab count mismatch (151665 vs 152064) in exported GGUF | ✅ done | S | P0 |
| GH-253-3 | Fix tokenizer.ggml.model: "bpe" → "gpt2" for byte-level BPE | ✅ done | S | P0 |
| GH-253-4 | Implement ValidatedGgufMetadata newtype (compile-time enforcement) | ✅ done | L | P0 |
| GH-253-5 | Verify exported 7B GGUF produces correct decode on CPU and GPU | ✅ done | M | P0 |
| GH-253-6 | Falsify F-CONTRACT-008 and F-CONTRACT-009 gates in showcase spec | ✅ done | S | P1 |

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated
- [ ] `apr export model.apr --format gguf` produces GGUF with all 26 metadata keys
- [ ] Exported GGUF decodes correctly (no byte-level BPE artifacts)
- [ ] `ValidatedGgufMetadata` prevents incomplete export at compile time

