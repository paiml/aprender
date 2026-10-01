# Chronos Zero-Shot Ports: Chronos-Bolt (tiny/small) and Chronos-2

From-scratch Rust forwards for Amazon's zero-shot time-series foundation models — no torch, no
transformers, no tokenizer — proven to float32 parity against `chronos-forecasting==2.3.1` with
a bottom-up ladder (spikes 005, 009). Bolt-small runs through the Bolt code unchanged (spike 006).

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- **Chronos joins the idea as a THIRD forecaster with its own thin server** (one model per server);
  Chronos-Bolt first, Chronos-2 later (decided 2026-09-04).
- Build order (frontier session 2026-09-05): 008 (NEON GEMM) → 007 (Chronos thin server) →
  009 (Chronos-2) → 010. Every multi-row product in these ports routes through `blis::gemm_blis`.

## How to Build It

**Reference implementations to port, not rewrite:**

| Model | File | Size | Notes |
|---|---|---|---|
| Chronos-Bolt (tiny, small, base share the code) | `sources/007-chronos-mcp-thin-server/src/bolt.rs` | 21 KB | Evolved from 005: GEMM routing, `dot8` for single rows, ladder tensors exposed |
| Chronos-2 | `sources/009-chronos-2-parity/src/chronos2.rs` | 15 KB | Transposed weights only; RoPE; group attention collapsed |
| Safetensors loader (F32/F16/BF16 → f32, from a byte slice) | `sources/007-chronos-mcp-thin-server/src/safetensors.rs` | 2.5 KB | Shared by embedded and on-disk weights |

**1. Chronos-Bolt architecture** (`config.json`: T5 `d_model 256`, `d_ff 1024`, 4 heads × `d_kv 64`,
4+4 layers for tiny; ReLU FFN not gated; `layer_norm_epsilon 1e-6`; 32 relative buckets / max
distance 128; `vocab_size 2` = pad + REG; patch 16 / stride 16; context 2048; horizon 64; 9 quantiles):

1. Instance norm: `loc = nanmean`, `scale = sqrt(nanmean((x−loc)²))` (population std; 0 → 1e-5).
2. Truncate to the last 2048; left-pad with NaN to a multiple of 16.
3. Per patch `[16 values (NaN→0), 16 mask]` → `ResidualBlock(32→1024→256)`:
   `out(relu(hidden(x))) + residual(x)`. Append `shared[1]` as the REG token.
4. Encoder attention mask = patch has any observed value; REG always on.
5. T5 stack: `T5LayerNorm` = RMS **without** mean subtraction; attention scores **not** scaled
   (`scaling = 1.0`); relative position bias computed in layer 0 only and reused (bidirectional
   buckets encoder, causal decoder self-attention, zero bias for cross-attention); masked keys get
   `f32::MIN` added; pre-norm residual blocks `[self, (cross), FF]`; final layer norm.
6. Decoder input = `shared[0]`; decoder last hidden → `ResidualBlock(256→1024→576)` → `[9][64]`
   → `× scale + loc`.
7. **Rollout past 64 (2.3.1 = `main`):** extend each of the 9 quantile paths with its own quantile,
   run as a batch of 9, re-quantile the 81 values per step at the 9 levels with `torch.quantile`
   semantics (linear interpolation; `torch_quantile` in `bolt.rs`). The older median-only scheme is
   off by 0.5 — the control that pins the installed pipeline.

**2. Chronos-2 architecture** (119.5M f32, 170 tensors, 478 MB; differs from Bolt in every layer):

encoder-only; context → last 8192 → instance norm → **arcsinh** → left-NaN-pad to 16 → patches of
`[time(16) | values(16) | mask(16)]` with time `[-L, …, -1] / 8192` → residual patch embedding
(48 → 3072 → 768) → REG token → *future* patches `[t/8192 | 0 | 0]` through the same embedding,
one per output patch (`ceil(h/16)`, ≤ 64 ⇒ 1024 steps direct). Twelve blocks of: **time
self-attention with RoPE** (θ = 10 000, d_kv 64, `rotate_half`, positions 0…L−1 over the whole
sequence including future patches, no score scaling, all-NaN patches masked); **group
self-attention** across the batch (no RoPE) — for a single series the softmax is over one key, so
the layer is exactly `x += o(v(ln(x)))` (confirmed by hidden states, not assumed); T5 FFN. Final
RMS norm → residual head (768 → 3072 → 21×16) on future positions → `sinh` → unscale.

```rust
fn rope(&self, l: usize) -> (Vec<f32>, Vec<f32>) {
    let dk = self.cfg.d_kv; let half = dk / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / self.cfg.rope_theta.powf((2 * i) as f32 / dk as f32)).collect();
    // cos/sin tables [l × dk], each half duplicated: c[pos*dk + i] = c[pos*dk + half + i]
}
fn apply_rope(&self, x: &mut [f32], l: usize, cos: &[f32], sin: &[f32]) {
    // per position, per head: tmp = [-v[half..], v[..half]]; v = v*cos + tmp*sin
}
```

**3. Parity ladder — the method that made both ports match on the first run.** The oracle
(`tools/oracle.py`) dumps intermediate tensors from the model's own methods plus `model(...)` and
`pipeline.predict(...)` outputs and timings into one `fixtures/*_fixture.json`; the Rust driver
prints one ladder table and compares **bottom-up**:

`loc`/`scale` → patch features → patch embeddings (first/last/REG) → encoder hidden states
(first token, REG, last future) → decoder hidden → quantiles → rollout.

Then a second fixture of **edge probes** (`tools/probes.py`): NaN gaps inside the context, a
5-point series (one patch, 11 NaN left-pad), a leading all-NaN patch, a constant series
(`scale → 1e-5`), values × 1e6, a negated series, a 130-point series rolled to 130 steps, air
passengers rolled to 24. Attention masks must be identical on every case.

**4. Weights.** Keep **only the transposed `[in, out]` copies** (`transpose(w, out, inp)` at load);
the plain-loop A/B copies in spike 005/007 doubled memory (Chronos-2 peaked at 1.45 GB). Route every
multi-row product through `trueno::blis::gemm_blis` (projections, FFN, patch embeddings, per-head
attention scores and context); keep `dot8` (8 independent accumulators) for `rows == 1`.

**5. Tolerances to assert:** Bolt-tiny quantiles 9.5e-7 abs on Peyton (1.8e-4 on air at scale 119,
i.e. 0.0002 %); rollout 365 steps 1.5e-5; Bolt-small 1.9e-6. Chronos-2 quantiles 2e-5 **of the
series scale** (1.6e-5 at 64 steps, 2.7e-5 at 365, 4.0e-5 at 1024), five-point series 2.3e-4 of
scale (worst case, 12 layers amplify f32 rounding).

## What to Avoid

- **Do not port the rollout from the paper or an old notebook** — the median-only scheme is off by
  half a unit past step 64. The 9-path re-quantiled scheme is the installed pipeline.
- **Do not subtract the mean in T5 layer norm** or scale attention scores by `1/√d_kv` — T5 does
  neither.
- **Do not use `trueno::Matrix::matmul` or `gemm_blis` on aarch64 without the spike-008 kernel** —
  before it both ran at ~3.6 GFLOP/s, 4× slower than plain unrolled loops (scalar microkernel).
  After it `gemm_blis` is 65–77 GFLOP/s and 7× faster than loops.
- **Do not write dot products as a strict-order float reduction** — LLVM will not vectorise it;
  use 8 independent accumulators (5× on its own).
- **Do not keep both weight layouts in the product.** Debug copies only.
- **Do not expect `blis::gemm` (rayon, feature `parallel`) to help below ~500 tokens** — it splits
  M only in 128-row blocks (`blis/parallel.rs`): 1.1× at 133 tokens, 2.1× at 517. Partitioning
  along N for M ≤ 256 is the upstream fix.
- Chronos-2 long-horizon unrolling (> 64 output patches) and multivariate/covariate groups are
  **not ported**; refuse with a message (`predict(h = 1100)` → "69 patches > 64").

## Constraints

| | Chronos-2 | Bolt-small | Bolt-tiny |
|---|---|---|---|
| params / f32 file | 119.5M / 478 MB | 47.7M / 182 MB | 8.65M / 34.6 MB |
| quantiles / direct horizon / context | 21 / 1024 / 8192 | 9 / 64 / 2048 | 9 / 64 / 2048 |
| forward, 2048 pts, 1 thread, spike-008 kernel | 423 ms (64 GFLOP/s) | 98 ms | 18.5–21 ms |
| forward, 8192 pts | 1.67 s | – | – |
| torch reference (10 threads) | 47–73 ms | – | 5.9 ms |
| f16 weight penalty (of series scale) | 0.3–0.6 % (3 % on 5 pts) | – | 0.1 % |
| peak RSS at load (f32) | 1.45 GB (0.5 GB steady) | ~0.3 GB | ~0.1 GB |

- Loader holds raw bytes + decoded f32 + transposed copy at once; a streaming decode-and-transpose
  per tensor would bring Chronos-2's peak to ~0.5 GB.
- Weights are Apache-2.0 on the Hub (`amazon/chronos-bolt-tiny`, `-small`, `amazon/chronos-2`);
  `models/` is gitignored — download with
  `uv run --python 3.12 --with huggingface_hub python -c "from huggingface_hub import snapshot_download; print(snapshot_download('amazon/chronos-bolt-tiny'))"`.
- Oracle env: `uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors`
  (pulls torch + transformers). Fixtures are committed under each spike's `fixtures/`.

## Origin

Synthesized from spikes: 005, 009 (Bolt-small evidence from 006; GEMM routing from 007/008)
Source files available in: `sources/005-chronos-bolt-tiny-parity/`, `sources/009-chronos-2-parity/`,
`sources/007-chronos-mcp-thin-server/src/{bolt,safetensors}.rs`
Parity fixtures (not copied, 2.6 MB): `.planning/spikes/005-chronos-bolt-tiny-parity/fixtures/`,
`.planning/spikes/009-chronos-2-parity/fixtures/chronos2_fixture.json`
