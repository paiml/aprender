---
spike: 025
idea: llm-decision-classifier
name: laya-rust-forward-parity
type: standard
validates: "Given Laya's safetensors (ModernBERT-large + 2-layer head + marker scorer, F16), when the forward and Laya's sequence builder are ported to Rust on trueno's BLIS GEMM, then token ids equal Python's, probabilities match torch fp32 to ~1e-5, and CPU latency is measured against Rust Kev-0.8B"
verdict: VALIDATED
related: [024, 017, 020, 005]
tags: [laya, modernbert, parity, rust, tokenizer, gemm, latency]
---

# Spike 025: Laya forward in Rust, parity with torch fp32

## What This Validates
The Python-trains / Rust-serves handoff for Laya: the published (or fine-tuned) `model.safetensors` + `tokenizer.json`
load directly in Rust — **no conversion step** (Kev needed PEFT merge -> llama.cpp converter -> GGUF) — and the Rust
decision forward reproduces Laya's own torch fp32 CPU probabilities, including its sequence builder.

## Research
Reference = transformers 5.17 `modeling_modernbert.py` (what Laya 4066d5d runs, `attn_implementation="sdpa"`) and
`laya/common.py`. Semantics the port must hit, each checked by a ladder rung:

| piece | semantics |
|---|---|
| embeddings | `LayerNorm(tok_embeddings(ids))`, no bias, **no position embedding** |
| layer 0 | `attn_norm = Identity`; layers 1–27 pre-norm LayerNorm, no bias |
| attention | fused `Wqkv` (d -> 3d, no bias) viewed `[3, heads, hd]`; rotate-half RoPE, theta **160k** on global layers (every 3rd, from 0), **10k** on local; scale hd^-0.5 |
| local window | `|i - j| <= local_attention / 2 = 64` (inclusive), bidirectional |
| MLP | `Wi` (d -> 2x2624) -> `chunk(input, gate)` -> `gelu(input) * gate` (exact erf) -> `Wo` |
| head | `h + type_emb[qtype]` -> 2 x `nn.TransformerEncoderLayer(norm_first=True)`: 16 heads, **relu** FFN 4d, biases, eps 1e-5 |
| scorer | per `[MASK]` marker: `LayerNorm -> Linear -> GELU(erf) -> Linear`; `p = softmax(z / clamp(T_bucket, 0.5, 5))` |
| builder | `[CLS] "<t> question: <ins>" [SEP] ([MASK] " "+opt[:48 tok])* [SEP] state [SEP]`, option budget `head_max_len` 192, `max_len` 512, `[MASK]` text replaced by space |

## How to Run
```bash
cd ../024-laya-vs-kev-few-shot && uv run --with ./vendor/laya --with datasets python ../025-laya-rust-forward-parity/tools/oracle.py
cd ../025-laya-rust-forward-parity && CARGO_TARGET_DIR=<repo>/target cargo build --release
S=~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
RAYON_NUM_THREADS=6 <repo>/target/release/laya-parity $S fixtures/laya-en_fixture.json fixtures/laya-en_ladder.bin
WINDOW=63 ...   # falsification probe: must break layer 1
```
Needs the spike-016 worktree (`aprender-016-upstream-sync`, upstream trueno with the 8x6 NEON microkernel).
`fixtures/laya-en_ladder.bin` (20 MB) is gitignored — regenerate with the oracle.

## Investigation Trail
1. **Oracle** (`tools/oracle.py`): Laya's own `Agent._encode_state` + `DecisionModel` in fp32 on CPU, hooks on the
   embeddings, every encoder layer, both head layers and the scorer input. 14 question rows: a 3-question support
   ticket (choice / noul / score, 143–158 tokens — the ladder row, long enough for the 64-token window to cut),
   6 stance tweets, 2 emotion rows, a `[MASK]`/`[SEP]`/`[CLS]` injection probe, mixed-script unicode, and a state
   truncated to 512 tokens.
2. **Parity on the first run**, no numerical debugging: ids 14/14, probabilities 3.8e-6, argmax 14/14.
3. **The one ambiguity — inclusive window — was settled by mutation**: `WINDOW=63` and `WINDOW=65` leave layer 0
   (global) untouched and break **layer 1** (the first local layer) by 1.6 / 0.7 on rms 1.2; end-to-end |dp| degrades
   3.8e-6 -> 1.4e-2 / 2.8e-2 and 65 flips an argmax. The harness sees a one-token boundary error.
4. **Latency is GEMM-bound**: 1 thread 1,058 ms at 85 tokens = ~61 GFLOP/s (the single-core NEON 8x6 ceiling from
   spike 008); 6 threads 226 ms (4.7x); 14 threads 193 ms. Attention is < 2 % of the FLOPs at these lengths.
5. **torch's 62 ms on the same Mac is Apple AMX, not a better algorithm**: 85 tokens x ~0.74 GFLOP/token = 63 GFLOP
   in 62 ms = ~1 TFLOP/s, only reachable through Accelerate's AMX units — which Graviton does not have. The fair
   comparison for a Lambda target is Rust vs Rust.

## Results

**Parity (Rust f32 from F16 weights vs torch fp32 CPU)**

| rung | worst |
|---|---|
| token ids + marker positions (builder + `tokenizers` 0.23 on `tokenizer.json`) | **14/14 identical** |
| embeddings | 9.5e-7 abs |
| encoder layer 27 (hidden rms 128) | 5.7e-2 abs = 4.4e-4 relative |
| final-normed encoder output | 3.1e-5 abs |
| marker states (scorer input, rms ~300) | 4.6e-3 abs |
| raw logits | 2.6e-5 |
| served probabilities | **3.8e-6** |
| argmax | **14/14** |

**Latency, M4 Pro (`RUN-OUTPUT.md`)**

| row | Rust 1 thr | **Rust 6 thr** | Rust 14 thr | torch fp32 CPU (AMX) | Rust Kev-0.8B 6 thr (spike 020) |
|---|---|---|---|---|---|
| 39 tokens | 543 ms | 126 ms | 122 ms | 48 ms | 0.20 s @ 37 |
| 85–94 tokens | 1,058 ms | **226–240 ms** | 193 ms | 62 ms | 0.36 s @ 87 |
| 143–158 tokens | 1,700–1,900 ms | 343–382 ms | 282–307 ms | 82–121 ms | – |
| 512 tokens | 6,460 ms | 1,237 ms | 895 ms | 229 ms | ~2.2 s @ 512 (interpolated) |

Load: 0.84 GB F16 read + widen to f32 in 0.15 s from a warm page cache (1.7 GB f32 resident).

**Verdict: VALIDATED ✓.** Laya runs in ~400 lines of plain Rust on the existing trueno GEMM, with **probability
parity 3.8e-6 on the first run** and the tokenizer reproduced id-for-id. Rust Laya is **1.6x faster than Rust
Kev-0.8B** at the same row length (226 vs 360 ms, 6 threads) — the 0.73x matmul ratio plus no sequential
recurrence — from **3.6x fewer weight bytes** (0.84 GB F16, loaded without conversion).

**Signal for the build**
- **The handoff is simpler than Kev's**: a fine-tuned Laya checkpoint (`model.safetensors` + `tokenizer.json` +
  `rl_agent_config.json`) is the serving artifact as-is. Keep the F16 file; widen at load.
- Port `render_options` (choice dict / score list / noul defaults and custom labels) for real requests; the fixture
  used Python-rendered options.
- A bidirectional encoder has **no prefix cache**: a 3-question request pays the state 3 times (343+345+382 ms).
  Batching the three rows into one GEMM (M = 3 x 150) is the lever, not caching.
- Head attention reuses the encoder's attention routine without RoPE or window; it is 2 of 30 layers.
