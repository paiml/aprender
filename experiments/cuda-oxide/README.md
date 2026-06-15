# cuda-oxide pure-Rust GPU kernels (source-of-record) — CAPABILITY ONLY, NOT a decode speedup

Pure-Rust `#[kernel]` → PTX kernels authored with [cuda-oxide](https://github.com/NVlabs/cuda-oxide)
(NVlabs), the rustc backend that compiles Rust device code directly to CUDA PTX. This proves the
**capability** (pure-Rust→PTX that loads + runs parity-correct on Blackwell sm_121, with no hand-PTX
GH-480 JIT workaround) — the north-star R&D bet.

> 🚨 **HONEST PERF CORRECTION (2026-06-15): cuda-oxide is NOT a decode speedup.** The A/B table below
> shows the oxide kernel beats `TiledQ4KGemv` ~1.4–2.85×, BUT `TiledQ4KGemv` is **NOT** the kernel
> production decode uses. On Blackwell the dispatch auto-selects **`HwDp4a`** (Q8_1 activations +
> half-warp DP4A hardware dot-product), and the decisive A/B vs HwDp4a shows oxide is **~4× SLOWER**
> (oxide/HwDp4a = 3.4–4.2× at every FFN shape K=6656/8960/11008). So wiring oxide into the decode path
> would make Q4K FFN GEMVs ~4× slower — the backend integration was **closed** (PR #2045) and the
> "P4 decode win" claim is **retracted**. cuda-oxide's value is the capability/R&D bet only. The
> `…-DRAFT.md` promotion plan is superseded.
>
> 🔒 **DP4A REVISIT — FALSIFIED AT THE CAPABILITY LAYER (2026-06-15, gx10 GB10 sm_121).** The revisit
> condition ("only if a DP4A-class oxide kernel can beat HwDp4a") **cannot be met** — not "we didn't
> try", but the toolchain provides **no path to `dp4a`**. Enumerating the entire cuda-oxide intrinsic
> universe (`cuda-device/src/` + the `dialect-nvvm/src/ops/` op set) shows **no `dp4a`/`dp2a`/`prmt`/
> `sad`/`byte_perm`/int8-dot op** of any kind, and **no user-facing inline PTX** (`core::arch::asm!`
> bodies are discarded in `mir-importer`; the internal `inline_asm_convergent` helper is backend-private
> and unreachable from `#[kernel]` code). HwDp4a is built entirely on chained `dp4a.u32.s32` over Q8_1
> int8 activations — exactly the instruction class oxide cannot reach — so oxide is confined to f32
> accumulation (~4× slower) as a **hard, terminal capability ceiling**. This only changes if NVlabs
> adds an integer-SIMD-dot intrinsic to `dialect-nvvm`+`cuda-device`, or exposes user inline PTX, upstream.

⚠️ **These projects build ONLY on gx10 (GB10 Blackwell)** with the cuda-oxide toolchain
(nightly-2026-04-03 + LLVM-21 + `cargo-oxide`). They are **isolated** from the aprender workspace
(each has its own `[workspace]`) so they NEVER affect the normal `cargo build`/CI. They are committed
here as the canonical **source-of-record** (the kernels previously lived only in `gx10:/tmp`,
which is ephemeral).

## Kernels

| dir | kernel | status |
|-----|--------|--------|
| `q4k-matvec/` | `q4k_matvec_atomic` — T=32 threads/row + `DeviceAtomicF32` reduction | **beats hand-PTX `TiledQ4KGemv` 1.23×–2.85× across decode-hotpath shapes** on GB10, bit-exact (maxrel 1.46e-5) |
| `q4k-matvec-reference/` | `q4k_matvec` — naive 1-thread/row (clean bit-exact reference) | correctness reference; bit-matches realizar `dequantize_q4_k` |

### A/B vs hand-PTX `TiledQ4KGemv` (GB10 Blackwell sm_121, same-data/same-run median; 2026-06-15)

| Shape (M×K) | Role | cuda-oxide T=32 (µs) | hand-PTX (µs) | speedup |
|---|---|---|---|---|
| 4096×2048 | baseline | 76.6 | 109.3 | **1.43×** |
| 1536×8960 | Qwen FFN down-proj | 120.2 | 342.2 | **2.85×** |
| 4096×4096 | attn/FFN square | 138.4 | 208.2 | **1.50×** |
| 151936×2048 | LM head (large-M) | 2625 | 3234 | **1.23×** |

cuda-oxide wins at every shape (T=32 optimal) AND avoids the hand-PTX GH-480 sm_121 JIT workaround.

Both do device-side f16 decode + 6-bit scale/min unpack + Q4K dequant (144-byte super-blocks),
matching `crates/aprender-serve/src/quantize/dequant_q4k.rs` + `simd.rs` (`extract_scale_min`/`read_f16`).

## Regenerate the embeddable PTX (on gx10)

```bash
ssh gx10
export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
cd <this dir>/q4k-matvec
cargo oxide pipeline          # emits target/.../q4k_matvec_atomic.ptx (.target sm_121)
# or: cargo oxide run         # build + launch + self-check on the GB10
```

The emitted `.ptx` is loadable via the existing `CudaModule::from_ptx` path
(`crates/aprender-gpu/src/driver/module.rs`) — **no cuda-oxide build dependency in aprender CI**.
Promotion (embed PTX as a static asset, raw-pointer ABI, 3-way parity gate) is scoped in the DRAFT doc.

## Generated artifacts (verified, shippable)

- `generated/q4k_matvec.sm121.ptx` — trimmed standalone PTX (245 lines, `.target sm_121`, entry
  `q4k_matvec`) emitted by `cargo oxide pipeline`. **Verified end-to-end through the exact aprender
  consumption path** (`include_str!` → `CudaModule::from_ptx` → resolve `q4k_matvec` → `cuLaunchKernel`),
  bit-exact vs the slice-ABI kernel + CPU reference, perf preserved (78µs/4096×2048, 125µs/1536×8960).
- `q4k-matvec/src/main_rawptr_abi.rs` — the raw-pointer-ABI `#[kernel]` source that produced it.
  Entry `q4k_matvec(data:*const u8, x:*const f32, y:*mut f32, m:u32, k:u32, t:u32)` — C-style ABI
  (3 ptr + 3 u32, no fat pointers), helpers `#[inline(always)]` so the entry has zero `call`.
  Launch: total=m*t, block=256, grid=ceil(total/256); y must be zeroed (kernel atom.global.add.f32).
  This .ptx + signature is what the aprender backend embeds (see the promotion DRAFT doc).
