# Qwen3.5 Decision Inference in Rust (Kev on upstream `Qwen35Model`)

The Python-trains / Rust-serves handoff, the hidden-state readout, and the batched prefill that makes
a Qwen3.5-based decision model servable on CPU. It applies to any readout-head model on the Qwen3.5
hybrid, not only Kev.

## Requirements

From idea `llm-decision-classifier`:

- **Training may stay in Python; inference must be Rust on aprender.** The served path (speed,
  security, AWS Lambda) is Rust.
- **The Python-to-Rust weight handoff is part of the contract**: a Kev checkpoint (adapter + head)
  must export to an artifact the Rust inference path loads, **with probability parity to Python fp32**.
- **Qwen3.5 support comes from upstream, not a fork-local re-port.** Extend upstream's
  `Qwen35Model` (OPS-03). Prerequisite: `upstream-sync.md`.
- **Kev needs a batched prefill before it is servable on CPU.** Token-at-a-time costs 5–7 s per
  decision.

## How to Build It

### 1. Back-office export (Python, once per checkpoint)

| Step | Tool | Output |
|---|---|---|
| Merge LoRA in fp32 (exact) | PEFT `merge_and_unload` via `Checkpoint.load(..., LoadOptions(backend="torch", dtype=float32, merge=True))` | merged backbone |
| Save as HF causal LM | `AutoModelForCausalLM.from_pretrained(meta.base, revision=meta.base_revision)` then `causal.model.load_state_dict(m.lm.state_dict(), strict=False)`. Assert only `lm_head` keys are missing | `Qwen3_5ForCausalLM` dir, 2.8 GB |
| Export head | `safetensors.save_file(head.state_dict(), metadata={"temperature": repr(T), "base_revision": …})` | `head.safetensors` 2 MB |
| Convert | llama.cpp `convert_hf_to_gguf.py --outtype f32` (22 s) | GGUF: f32 3.0 GB / bf16, f16 1.5 GB / q8_0 0.8 GB |

```bash
cd vendor/llama.cpp   # git clone --depth 1 https://github.com/ggml-org/llama.cpp  (master d2e5458 registers Qwen3_5ForCausalLM)
uv run --python 3.12 --with ./gguf-py --with "transformers>=5" --with torch --with numpy --with sentencepiece \
  --with safetensors --with protobuf python convert_hf_to_gguf.py <merged-dir> --outtype f32 --outfile kev-merged-f32.gguf
```

`sources/017-kev-rust-forward-parity/tools/export_and_oracle.py` does the export **and** writes the
parity fixture: token ids, readout offsets, hidden states at each readout, raw logits and probs, plus
a row-0 ladder (final hidden at every position). The fixture carries **token ids**, so the tokenizer
is a separate rung.

### 2. Two upstream patches (`upstream-qwen35-hidden-and-mtp.patch`, commit `12976da0a`)

- **Hidden-state readout.** Split `forward_single_qwen35` into `forward_single_qwen35_hidden`, which
  stops at the final RMS norm and returns `[hidden_dim]`. `forward_single_qwen35` calls it and then
  applies `lm_head`, so existing callers are unchanged (41/41 qwen35 tests pass).
- **MTP depth defect (upstream-worthy on its own).** llama.cpp writes `qwen35.block_count = 25` and
  `nextn_predict_layers = 1` for the 24-layer 0.8B, whose tensors stop at `blk.23`. llama.cpp's
  runtime subtracts the MTP block; upstream's loader did not, and failed with
  `Tensor 'blk.24.attn_norm.weight' not found`. **Every Qwen3.5 GGUF freshly converted from HF
  hits this.** Fix in `qwen35_load.rs`:
  ```rust
  let nextn = arch_u32(model, "nextn_predict_layers").unwrap_or(0) as usize;
  let num_layers = num_layers.checked_sub(nextn).filter(|&n| n > 0).ok_or_else(|| …)?;
  ```

### 3. Batched prefill (`upstream-qwen35-batched-prefill.patch`, commit `32103ba83`, ~280 lines)

A new `forward_qwen35_prefill.rs` provides `Qwen35Model::prefill_hidden(&ids, &mut state) -> Vec<f32>`,
the final-normed hidden at **every** position (`[L, d]`). A decision model never generates, so its
whole cost is prefill.

- **Projections as one GEMM per layer over the row.** GGUF stores W as `[out, in]` row-major, which
  is exactly the A operand of `Cᵀ = W · Xᵀ`. No weight is copied or transposed; only the small
  `[L, in]` activation is. Borrow the F32 bytes with `bytemuck::try_cast_slice::<u8, f32>`
  (the workspace forbids `unsafe`). Call `trueno::blis::gemm_blis` directly, because
  `fused_matmul_f32` is a per-row matvec loop, not a GEMM.
- **Band W's output rows across the rayon pool yourself**, one single-threaded `gemm_blis` per band
  (`bands = 2 × threads`, band rounded to a multiple of 8). trueno's own parallel GEMM caps a GEMM
  below 512 MFLOP at 4 threads, using tiers measured on a Threadripper.
- **32×32 blocked, parallel transposes** (`par_chunks_mut` over output bands).
- **DeltaNet in three stages** (the key fix; the arithmetic is identical):
  - A, **sequential in t**: causal conv1d + SiLU, per-head L2 norm of q/k, `dt = softplus(dt_raw + dt_bias) · a`, `β = σ(β_raw)`.
  - B, **parallel over value heads**: each head's 128×128 state is independent for the row. Make
    one-head calls of the **same** `delta_rule_recurrence_gqa` (key head `h % num_k_heads`), so the
    arithmetic and its order are unchanged.
  - C, **parallel over t**: gather heads, then `gated_rmsnorm`.
- **Attention layers**: causal attention from the row's own K/V, parallel over tokens, with the
  per-head q/k RMS norm, partial NeoX RoPE and sigmoid gate applied per token.

### 4. Pointer head in Rust (~40 lines)

```rust
// logits_j = k(h_opt_j) . q(h_decide) / sqrt(dp); probs = softmax(logits / T), with T from head metadata
let q = proj(&qw, &qb, h_decide);
let z: Vec<f32> = h_opts.iter().map(|h| dot(&proj(&kw, &kb, h), &q) / (dp as f32).sqrt()).collect();
let p = softmax(&z, temperature);
```

Read `h_decide = h[decide*d..]` and `h_opt_j = h[opt_j*d..]` from the `prefill_hidden` output.

### 5. Verify with the parity ladder

Final hidden at every position of row 0 → hidden at every readout → raw pointer logits → served
probabilities → argmax. Include a 3-question ticket (`choice` / `noul` / `score`), a
delimiter-injection probe (`SPECIAL` tokens inside the state), mixed-script unicode, and a
915-token state.

## What to Avoid

- **Don't serve upstream's token-at-a-time path.** It runs at 79 ms per token: 4.8–6.7 s for an
  85-token decision and 72 s for a 915-token state. Upstream's own comment: *"there is no batched
  prefill for the hybrid"*.
- **Weight dtype is not the latency lever.** BF16 halves the bytes and buys 1.8×. Q8_0 runs the same
  43 ms/token while its **RSS rises to 10.9 GB** (above the 3 GB f32 file), because the loader holds
  dequantised owned copies.
- **Don't trust a sampling profile that says "idle".** `sample` reported 61 % `__psynch_cvwait` and
  pointed at trueno's thread cap. Fixing the cap bought only 719 → 639 ms. Thread-local phase timers
  showed the **DeltaNet per-token loop at 288 ms on one thread** while five threads waited. Serial
  work shows up as other threads waiting, not as its own hot symbol.
- **Don't expect more threads to fix short rows.** At 87 tokens, 6 threads take 0.36 s and 14 take
  0.34 s. Threads pay off at long rows (915 tokens: 3.9 s at 6 threads, 3.0 s at 14).
- **Don't merge the phase timers.** `PHASES` / `take_prefill_phases` are spike instrumentation.
- **Don't judge Kev with upstream's parity bar.** Upstream checks llama.cpp argmax with a 0.25-logit
  near-tie allowance. A pointer head needs ~1e-5 hidden-state parity.

## Constraints

Parity (Kev-0.8B, f32 GGUF vs torch fp32 CPU, 12 rows): final hidden 1.2e-4 abs (rms 2.1); readouts
6.3e-5; raw logits 2.0e-5; **probabilities 1.3–1.5e-6; argmax 12/12**. It passed on the first run,
unchanged by the prefill (prefill vs token-at-a-time |Δh| 1.5e-4, from GEMM summation order).

| Row | Token-at-a-time (6 threads) | Prefill, 1 thread | **Prefill, 6 threads** | Prefill, 14 threads |
|---|---|---|---|---|
| 37 tokens | 2.9 s | 0.81 s | **0.20 s** | 0.20 s |
| 87 tokens | 6.6 s | 1.69 s | **0.36–0.38 s** | 0.34 s |
| 915 tokens | 73 s | 17.4 s | **3.9 s** | 3.0 s |

Where 6 threads go at 87 tokens: GEMM 252 ms (~340 GFLOP/s), DeltaNet 91 ms, transposes 20 ms,
attention 5 ms. At 915 tokens: GEMM 2.36 s, DeltaNet 0.96 s, attention 0.38 s (attention grows with L²).

Open work for the real build:
- **Prefill is F32-only.** BF16/F16/Q8_0 fall back to per-row `fused_matmul`. A **BF16 GEMM** is
  required for Kev-4B.
- **State-prefix cache**: reuse the state's recurrent/KV state across a request's questions. Without
  it, a 3-question ticket pays the state three times (711 ms).
- **Rust tokenizer**: upstream has `gguf/byte_level_bpe.rs`. Port Kev's escaping of `<|x|>` to `<¦x¦>`.
- Kev-4B is ~7× the matmul work (3.57B block params). Expect ~2.1–2.5 s per short decision on 6 M4
  threads (GEMM-scaled projection, not measured).
- The prefill also benefits upstream beyond Kev (time-to-first-token for `apr run` / `apr chat` on the
  hybrid). Offering it upstream is a checkpoint.

## Origin

Synthesized from spikes 017 (VALIDATED) and 020 (VALIDATED).
Source files: `sources/017-kev-rust-forward-parity/` (driver, `prefill_bound.rs`,
`export_and_oracle.py`, hidden/MTP patch, RUN-DTYPES, PREFILL-BOUND),
`sources/020-qwen35-batched-prefill/` (driver, prefill patch, RUN-SCALING).
Models (GGUFs, merged safetensors), `vendor/llama.cpp` and the 3.3 MB fixture stay in
`.planning/spikes/017-kev-rust-forward-parity/`.
