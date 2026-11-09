# EG-1 tokenizer parity: provisional receipt (2026-10-08)

**Result: 1120 / 1120 text-file pairs give identical token ids** (8 GGUF files × 140 texts). Every apr run took the
`canonical` path and round-tripped (`roundtrip: true`).

| Item | Value |
|------|-------|
| apr | release build of `feat/apr-embed-001-eg1` @ 391318a784 (EG-1 slice 4), sha256 `106b0a678617fc78131e959113f2c54a1812285c04bc7c4b876305e70b232b68` |
| Oracle | llama.cpp b11490 (`9c2e0e491`) `llama-tokenize --ids --no-escape --no-bos`. **PROVISIONAL** (§7) |
| Pinned oracle | `scripts/llama_pin.toml` build_commit `d1d3c3396` refuses the file: `unknown model architecture: 'gemma-embedding2'`. Moving the pin is D-3, a §7 pin-change STOP, so it was not moved |
| Files | unsloth BF16, F16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL; ggml-org BF16, Q8_0 (text GGUFs; mmproj excluded) |
| Corpus | one file per row (last column) of `corpus/{batch,lengths,texts}.tsv`, `retrieval/{docs,queries}.tsv`, `finetune/heldout.tsv`, plus 3 edge files: double spaces, tabs, newline runs, accents, CJK, emoji, `<start_of_turn>`/`<end_of_turn>` specials, leading/trailing spaces, code with a trailing `\n\n\n` |

## Why this is not the gate receipt

`scripts/tokenizer_parity.sh` refuses any comparator other than the pinned commit, and it should. The comparison
above re-uses its library (`tp_llama_ids`, `tp_first_mismatch`, `tp_apr_status`) and its exact flags, but runs
against b11490. The comparator was checked to be live: `tp_first_mismatch "1 2 3" "1 9 3"` = 1 and
`tp_first_mismatch "1 2" "1 2 3"` = 2. The same files and corpus become the gate receipt once a pin that loads
`gemma-embedding2` is approved (D-3).
