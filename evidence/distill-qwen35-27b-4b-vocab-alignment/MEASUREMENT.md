# Qwen3.5 27B -> 4B distillation vocab alignment (0.72 R18)

Measured 2026-09-27T20:36Z on lambda by aprender-76, pv 0.70.0 (9f5609568).

    python3 scripts/distill_vocab_identity.py ~/models/Qwen3.5-27B-Q4_K_M.gguf ~/models/Qwen3.5-4B-Q4_K_M.gguf --out identity.json   # rc 0
    python3 scripts/distill_vocab_identity.py ~/models/Qwen3.5-27B-Q4_K_M.gguf ~/models/Qwen2.5-0.5B-Instruct-IQ4_XS.gguf --out identity-negative-qwen25-student.json   # rc 1

| file | sha256 | pinned in |
|---|---|---|
| Qwen3.5-27B-Q4_K_M.gguf | 84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806 | evidence/crux/hf-sources.yaml |
| Qwen3.5-4B-Q4_K_M.gguf | 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4 | evidence/crux/hf-sources.yaml |

- tokens (248320), merges (247587), token_type: byte-identical, so truncation is the identity (FT-VOCAB-ALIGN-005).
- Qwen2.5 student: 151936 < 248320 passes a size-only rule, but tokens differ from index 280 (FT-VOCAB-ALIGN-006).
- Only tokenizer.chat_template differs: thinking defaults are inverted (27B on, 4B off). Render KD data with one template.
- 27B embedding_length is 5120 and it has an untied output.weight; 4B is 2560 with tied embeddings.
  contracts/model-families/qwen3_5.yaml gives 27b hidden_dim 6144 (= heads x head_dim); not the embedding width.
