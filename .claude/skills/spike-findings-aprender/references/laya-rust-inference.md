# Laya Inference in Rust (ModernBERT + head + scorer + request path)

A from-scratch Rust port of Laya's decision forward and its request handling. It matches torch fp32 to 3.8e-6 on
probabilities, token ids included. It is **spike code outside the workspace**: it uses only
`trueno::blis::gemm_blis` from aprender, and aprender has no ModernBERT (SetFit's `setfit/encoder.rs` is
post-norm HF BERT with WordPiece — a different architecture). Productising it is build work (see Constraints).

## Requirements

From idea `llm-decision-classifier`:
- **Inference must be Rust on aprender**, with **probability parity to Python fp32** across the handoff.
- **Laya is evaluated as an alternative base to Kev** (English root + `typed-decisions`; decided 2026-09-25).

## How to Build It

### 1. Load the published artifacts directly (no conversion)
`model.safetensors` (205 F16 tensors + an F32 `temperature`), `tokenizer/tokenizer.json`, `encoder/config.json` and
`rl_agent_config.json` load as-is. Widen F16 to f32 at load (0.15 s from a warm cache, 1.7 GB resident). A
fine-tuned Laya is served by swapping these four files. Kev, by contrast, needed PEFT merge → llama.cpp → GGUF.

### 2. Reproduce these semantics exactly (each has a ladder rung)
Reference: transformers 5.17 `modeling_modernbert.py` (SDPA) and `laya/common.py`.

| piece | semantics |
|---|---|
| embeddings | `LayerNorm(tok_embeddings(ids))`, no bias, **no position embedding** |
| layer 0 | `attn_norm = Identity`; layers 1–27 are pre-norm LayerNorm without bias |
| attention | fused `Wqkv` [3d, d], no bias, viewed as `[3, heads, hd]`; rotate-half RoPE with theta **160k** on global layers (every 3rd, starting at 0) and **10k** on local layers; scale hd^-0.5 |
| **local window** | bidirectional, **\|i − j\| ≤ local_attention / 2 = 64, inclusive**, proven by mutation |
| MLP | `Wi` [2·2624, d] → `chunk(input, gate)` → `gelu(input) * gate`, exact erf (`libm::erf`) → `Wo` |
| head | `h + type_emb[qtype]` → 2 × `nn.TransformerEncoderLayer(norm_first=True)`: 16 heads, **relu** FFN 4d, with biases |
| scorer | on each `[MASK]` marker: `LayerNorm → Linear → GELU(erf) → Linear`; `p = softmax(z / clamp(T_bucket, 0.5, 5))`; buckets `<type>:2 / 3-5 / 6-10 / 11+` |
| builder | `[CLS] "<t> question: <ins>" [SEP] ([MASK] " "+opt, first 48 tokens)* [SEP] state [SEP]`; option budget `head_max_len` 192 (shrink evenly if the budget falls below 16); head kept to at least 8 tokens; `max_len` 512; literal `[MASK]` replaced with a space everywhere |

Code: `sources/025-laya-rust-forward-parity/src/laya.rs` (model, builder, temperature),
`sources/026-laya-mcp-default-lambda/src/lib.rs` (`render_options` port, `/v1/systemone`-shaped `decide` tool,
S3 loader).

### 3. GEMM layout (spike 020)
Every dense product is `Cᵀ = W · Xᵀ` on `gemm_blis`. The checkpoint's `[out, in]` weight is the A operand, so no
weight is transposed. Band W's output rows over the rayon pool (bands = 2 × threads, multiples of 8).

### 4. Verify with the parity ladder
`sources/025-laya-rust-forward-parity/tools/oracle.py` hooks Laya's own torch fp32 CPU path and dumps a JSON fixture:
14 rows covering a 3-question ticket, stance, emotion, a `[MASK]`/`[SEP]` injection probe, mixed-script unicode and
a 512-token truncation. It also dumps a binary ladder: embeddings, 28 layers, the final norm and 2 head layers on
a 158-token row. The driver (`src/main.rs`) checks ids, the ladder, marker states, logits and probabilities.
`WINDOW=63` / `65` must break layer 1 only.

### 5. Request path
- `render_options` covers: `choice` as an object or list, `score` as a list (`"level i: c"`), and `noul` with default
  or custom `labels` (defaults "no, the statement does not hold" / "yes, the statement holds").
- `serde_json` needs the **`preserve_order`** feature: criteria order is the label index.
- Unknown types, fewer than 2 choice options, or options beyond `head_max_len` are **refused**, not defaulted.

## What to Avoid
- **Reusing aprender's SetFit BERT encoder**: wrong architecture on every block.
- **Guessing the window boundary.** 63 or 65 degrade |dp| to 1.4e-2 / 2.8e-2 and flip an argmax at 65.
- **Treating torch's 62 ms on the Mac as the bar.** That's AMX. Rust vs Rust is the fair comparison; the Rust port is
  GEMM-bound at ~61 GFLOP/s per core (the NEON 8×6 ceiling).
- **Expecting a prefix cache.** A bidirectional encoder re-reads the state for every question, so a 3-question
  request is 3 forwards (1.07 s on M4, 3.8 s on Graviton2). Batch a request's rows into one GEMM pass instead.

## Constraints
- Latency (Rust, f32, 6 threads): 85–94 tokens **226–240 ms on M4**, **0.75 s on Graviton2**, 0.49 s on Graviton3;
  512 tokens 1.24 s M4 / 5.6 s G2. That's **1.6× faster than Rust Kev-0.8B** (360 ms at 87 tokens on M4).
- One unpadded row per call; f32 compute only (no bf16/f16 GEMM path); aarch64 tested, x86 not.
- **To productise** (not done):
  - decide the home (realizar-first says `aprender-serve` unless an exception is argued);
  - apply workspace lints (no `unwrap`, pedantic clippy);
  - write a `contracts/` YAML with falsification tests;
  - turn the ladder, tokenizer and mutation rungs into `cargo test`s wired into CI;
  - test a fine-tuned checkpoint and the `typed-decisions` and multilingual (mmBERT: 256k vocabulary, different
    RoPE) variants;
  - handle non-string criteria and padding/batching;
  - land the spike-016 upstream sync, which is where the trueno build comes from and is still local-only.

## Origin
Synthesized from spikes 025 (VALIDATED) and 026 (VALIDATED, request path).
Source files: `sources/025-laya-rust-forward-parity/` (crate, oracle, RUN-OUTPUT), `sources/026-laya-mcp-default-lambda/`.
The parity fixture (`laya-en_fixture.json`, 936 KB, committed) and the ladder binary (gitignored; regenerate with
the oracle) stay in `.planning/spikes/025-laya-rust-forward-parity/fixtures/`.
