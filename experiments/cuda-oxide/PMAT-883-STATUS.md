# PMAT-883 — oxide attention PTX artifact + 3-way parity gate (integration-prep)

**Status: COMPLETE — SAFE integration-prep step done. VERDICT: artifact + gate GREEN, live default UNCHANGED.**

Follows the PMAT-882 GO (cuda-oxide `attn_warp` kernel C is bit-parity-correct +
1.7-2.9x faster than the production hand-PTX on gx10 GB10 sm_121). This ticket is
the SAFE next step: emit a parity-gated, embeddable PTX artifact and a 3-way
parity gate. **It deliberately does NOT change the live default decode path** —
that is a later, separately-reviewed step (see "Go-live remaining step").

- gx10: NVIDIA GB10, compute_cap 12.1 (sm_121), driver 590.48.01,
  cargo-oxide 0.2.1, nightly-2026-04-03, LLVM-21, CUDA 13.0 (ptxas).
- Source kernel: `incremental-attention/src/main.rs` — NEW raw-ptr ABI entry
  `attn_warp_rawptr` (bit-identical compute to PMAT-882 kernel C `attn_warp`).

## 1. Emitted PTX artifact (source-of-record)

**Path:** `experiments/cuda-oxide/generated/attn_warp.sm121.ptx`
(548 lines, `.target sm_121`, exactly 1 `.visible .entry`, 0 extern `__nv_*`,
ptxas-verified → 18816-byte cubin).

**ABI (stable raw-pointer C-style, for `include_str!` → `CudaModule::from_ptx`):**
```
attn_warp_rawptr(
    q:   *const f32,   // [n_heads * head_dim]
    k:   *const f32,   // [kv_len * kv_dim]  (INTERLEAVED [seq, kv_dim])
    v:   *const f32,   // [kv_len * kv_dim]
    out: *mut   f32,   // [n_heads * head_dim]
    kv_len: u32, head_dim: u32, n_heads: u32, n_kv_heads: u32, scale: f32)
Launch: grid = (n_heads,1,1), block = (32*NW = 1024,1,1).  GQA mapping is RUNTIME
(kv_head = head/(n_heads/n_kv_heads)) — nothing is baked in, so this ONE PTX
serves every (head_dim<=128, n_heads, n_kv_heads) decode shape.
```

**Why it's NOT a single `cargo oxide pipeline` .ptx (and how it IS emitted):**
kernel C uses `f32::exp()` (softmax), which cuda-oxide lowers to a libdevice
`__nv_expf` call. The pipeline therefore emits **NVVM IR (.ll)** and *skips llc*,
leaving libNVVM lowering to the consumer at JIT (exactly how the 882 runtime
`kernels::load` ran it). To get a SELF-CONTAINED `.ptx` (no extern `__nv_*`) that
`CudaModule::from_ptx` can JIT directly, the source-of-record emit script
`incremental-attention/emit_ptx.sh` does:

```
1. cargo oxide pipeline --arch sm_121      -> NVVM IR (.ll)
2. llvm-link  .ll + libdevice.10.bc        -> resolve __nv_expf
3. opt internalize/nvvm-reflect/globaldce/O3
4. llc -mcpu=sm_121 -mtriple=nvptx64       -> full-module .ptx
5. trim to the single `attn_warp_rawptr` entry (+ ASCII sanitize)
6. ptxas -arch=sm_121                       -> verify it assembles
```
Run on gx10: `cd incremental-attention && ./emit_ptx.sh ../generated/attn_warp.sm121.ptx`.

This mirrors the q4k-matvec promotion path (`generated/q4k_matvec.sm121.ptx`); q4k
needed no libdevice so its pipeline emitted PTX directly, attention needs the
libdevice-link + llc lowering above.

## 2. 3-way parity gate (oxide-PTX == hand-PTX == CPU) — ✅ PASS (measured)

The gate (`run_3way_gate` in `incremental-attention/src/main.rs`, auto-runs when
`generated/attn_warp.sm121.ptx` is present) loads:
- **way 1 — emitted oxide PTX**: `load_module_from_ptx_src` → resolve
  `attn_warp_rawptr` → `cuLaunchKernel` (the exact aprender consumption path),
- **way 2 — hand-PTX baseline**: `multi_warp_attention` (committed
  `baseline-ptx/multiwarp_msl4096_nw{8,32}.sm121.ptx`; baked n_heads=32, so
  parity-valid only at heads=32),
- **way 3 — CPU reference**: `causal_attention_cached` (`cpu_incremental_attention`),

runs all three on identical Q/K/V across the 9 configs (seq{128,1024,4096} ×
heads{8,16,32}, head_dim=128, n_kv_heads=8 GQA), and asserts
**cos ≥ 0.99 AND maxdiff < 1e-4·max|ref|**.

This is **gx10-manual** (no sm_121 CI runner). Re-run:
```bash
ssh gx10; export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"; export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
cd /tmp/incattn883_spike   # or rsync incremental-attention/
./emit_ptx.sh generated/attn_warp.sm121.ptx
cargo oxide run            # runs the 3-way gate, then 882 parity/perf
```

**Measured result (GB10 sm_121, 2026-06-21):**

```
== PMAT-883 3-WAY PARITY GATE (oxide-PTX == hand-PTX == CPU) ==
   gate: cos>=0.99 AND maxdiff < 1e-4*max|ref|, all 3 ways, all 9 configs

   seq head  kv | oxide-PTX (raw-ptr entry)    | hand-PTX multi_warp          | verdict
   128    8   8 | cos=1.000000 md=4.77e-7      | n/a (baked 32h)              | PASS
  1024    8   8 | cos=1.000000 md=2.62e-6      | n/a (baked 32h)              | PASS
  4096    8   8 | cos=1.000000 md=8.64e-6      | n/a (baked 32h)              | PASS
   128   16   8 | cos=1.000000 md=5.36e-7      | n/a (baked 32h)              | PASS
  1024   16   8 | cos=1.000000 md=3.04e-6      | n/a (baked 32h)              | PASS
  4096   16   8 | cos=1.000000 md=8.11e-6      | n/a (baked 32h)              | PASS
   128   32   8 | cos=1.000000 md=5.36e-7      | cos=1.000000 md=5.96e-7      | PASS
  1024   32   8 | cos=1.000000 md=2.65e-6      | cos=1.000000 md=2.62e-6      | PASS
  4096   32   8 | cos=1.000000 md=9.54e-6      | cos=1.000000 md=9.54e-6      | PASS

PMAT-883 3-WAY PARITY GATE: PASS
```

- emitted oxide PTX vs CPU: **cos = 1.000000** at every config, maxdiff
  4.77e-7 … 9.54e-6 (tol ≈ 6.5e-5).
- emitted oxide PTX vs hand-PTX (at the parity-valid heads=32 shapes): both
  cos=1.000000, maxdiff within ~1e-7 of each other — i.e. **all three agree**.
- The 882 parity (A/B/C all 9 configs) + perf A/B re-pass in the same run
  (oxide C 0.33-0.60× vs hand-PTX NW=8; 0.60-1.01× vs NW=32 — unchanged from 882).

A GPU-free CI guard also ships: `crates/aprender-serve/src/cuda/executor/oxide_attention.rs::tests::embedded_ptx_is_self_contained_single_entry`
asserts the committed PTX is `.target sm_121`, 1 entry, 0 extern `__nv_*` — so a
bad re-emit fails on ANY host (`cargo test -p aprender-serve --features oxide-attention --lib oxide_attention` → PASS).

## 3. Integration scaffold (default OFF — NOT wired into live decode)

`crates/aprender-serve/src/cuda/executor/oxide_attention.rs`, gated on the
**new default-OFF feature `oxide-attention`** (`oxide-attention = ["cuda"]`;
NOT in `default`/`full`). It:
- `include_str!`s the source-of-record PTX,
- `compile_oxide_attention(exec)` → `CudaModule::from_ptx` (same loader the live
  executor uses — GH-480 sm_121 patch + disk cache; no cuda-oxide build dep),
- `launch_oxide_attention(...)` mirrors `incremental_attention_async`'s raw
  pointer-array launch with the 9-param ABI above.

It is registered as `mod oxide_attention;` in `executor/mod.rs` ONLY under
`cfg(all(feature="cuda", feature="oxide-attention"))`, and is **referenced by no
dispatch path**. Default builds (`cuda` without `oxide-attention`) don't even
compile it. Verified: `cargo check -p aprender-serve --features oxide-attention`
clean on lambda-vector (sm_89); scaffold unit test PASS.

### ⚠️ The one real integration gap: KV-cache layout

The live `incremental_attention_async` stores K/V in the **separate-head** layout
`[num_kv_heads, max_len, head_dim]` (kv_stride = max_len·head_dim). The oxide
kernel uses the **interleaved** `[seq, kv_dim]` layout (kv_dim = num_kv_heads·head_dim)
— the layout the CPU reference + the 3-way gate use, but NOT the live GPU cache.
Promotion must resolve this by EITHER:
- **(a, preferred)** give the oxide path an interleaved-layout GPU KV cache (it is
  the layout the CPU ref + the oxide kernel already use), OR
- **(b, mechanical)** add an oxide variant that indexes the existing separate-head
  cache: change `krow = pos*kv_dim + kv_base` to `kv_head*kv_stride + pos*head_dim`.

This is documented in the module header as the #1 promotion blocker.

## 4. Promotion criteria (the ONLY way this becomes the live default)

1. Resolve the KV-cache layout (3a or 3b) and **re-pass the on-device 3-way gate
   against the LIVE separate-head cache** (not just the interleaved gate inputs).
2. End-to-end decode tok/s on a real GQA model on GB10 ≥ the hand-PTX default
   (`apr run --gpu ... --max-tokens N`, measured, not projected).
3. CPU/GPU parity test passes with the feature ON
   (`cargo test --features cuda,oxide-attention --test gpu_cpu_trace_compare`).
4. A `cfg(feature="oxide-attention")` branch in `incremental_attention_async`
   selecting the oxide kernel behind a runtime env guard (default OFF).

## 5. Go-live remaining step (exact, separately-reviewed)

When promotion criteria 1-4 are GREEN, the live flip is a single reviewed PR:

```rust
// in CudaExecutor::incremental_attention_async (after K/V are in cache):
#[cfg(feature = "oxide-attention")]
if std::env::var("APR_OXIDE_ATTENTION").as_deref() == Ok("1") {
    let module = self.oxide_attn_module();           // cached compile_oxide_attention(self)
    oxide_attention::launch_oxide_attention(
        self, module, q_gpu, k_iface, v_iface, &out_buf,
        new_len as u32, head_dim as u32, num_heads as u32, num_kv_heads as u32,
        1.0 / (head_dim as f32).sqrt(),
    )?;
    return Ok((out_buf, new_len));
}
// ...existing hand-PTX path unchanged (the default)...
```
Then, after a soak, make it the default by selecting the oxide kernel when
`compute_cap >= (12,0)` and removing the env guard. **None of this is done in
PMAT-883** — the live default decode path is untouched.

## Files

- `experiments/cuda-oxide/generated/attn_warp.sm121.ptx` — emitted source-of-record PTX.
- `experiments/cuda-oxide/incremental-attention/src/main.rs` — `attn_warp_rawptr`
  kernel + `run_3way_gate` + `oxide_ptx_parity`.
- `experiments/cuda-oxide/incremental-attention/emit_ptx.sh` — the emit pipeline.
- `crates/aprender-serve/src/cuda/executor/oxide_attention.rs` — integration scaffold (OFF).
- `crates/aprender-serve/Cargo.toml` — new `oxide-attention` feature.
- `crates/aprender-serve/src/cuda/executor/mod.rs` — gated `mod oxide_attention;`.
