# PMAT-4662 implementation receipt (aprender#4661, aprender#4662)

## Root cause
`BPETokenizer::new` (GGUF serve) did greedy longest-match with no control-token handling.
Qwen vocabs hold `?<` (75414) and `.<` (15757), so `…?<|im_end|>` became
`?<`, `|`, `i`, `m`, `_end`, `|`, `>Ċ`. llama.cpp renders the same Qwen3-Coder
chat prompt as 15 tokens with `<|im_end|>` = 151645.

## Change
1. `tokenizer.rs`: new `greedy_specials` (vocab entries `<|…|>`, len > 4), filled in `new()`.
   Greedy `encode` splits on them with `apr::tokenizer::split_by_special_tokens` (made
   `pub(crate)`, the merge path's own helper) and greedy-encodes only the text between them.
   `special_tokens` is untouched, so `is_special_token` (embedding pooling) does not change.
2. **Required companion: the `decode` extraction.** The pmat pre-commit hook runs
   `pmat analyze complexity --file <staged> --max-cyclomatic 30 --max-cognitive 25` on every
   staged file and refuses the commit on any error. At base 316dee2cd4, `tokenizer.rs`
   already reports `Errors: 1` (`decode`, cognitive 41), so no edit to this file can commit.
   The fleet rule is to decompose the blocker in the same commit, never `--no-verify`. The
   per-token body of `decode` moves **verbatim** into `decode_token_into` (the `continue`s
   become `return`s) and `decode_char_into` (the match on one char). After: `Errors: 0`.

## Evidence
- RED before the fix: `greedy_encode_keeps_control_tokens_whole` on base (intel), with only
  the test applied: left `[20, 2, 5, 7, 8, 9, 10, 11, 12, 5, 6, 17, 19, 15]`
  (`?<`,`|`,`i`,`m`,…), right `[20, 1, 18, 17, 19, 15]`. FAILED.
- Ladder, lambda RTX 4090, binary pinned by `scripts/apr_bin.sh` at de55c2f7af:
  - `--only inv:Qwen2.5-0.5B-Instruct-f16.gguf` → `[ OK ] qa cap+golden pass, backends cuda honoured`
  - `--only inv:Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` → `[ OK ] qa cap+golden pass, backends cuda honoured`
- `cargo test -p aprender-serve --tests` (intel): see PR body.
