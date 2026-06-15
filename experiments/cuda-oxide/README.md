# cuda-oxide pure-Rust GPU kernels (source-of-record)

Pure-Rust `#[kernel]` → PTX kernels authored with [cuda-oxide](https://github.com/NVlabs/cuda-oxide)
(NVlabs), the rustc backend that compiles Rust device code directly to CUDA PTX. This is the
north-star path to **replace hand-PTX GPU kernels** (escaping the recurring Blackwell sm_121 JIT
pain) — see `memory/reference_cuda_oxide_rust_to_ptx.md` and the promotion plan
`docs/specifications/cuda-oxide-q4k-backend-promotion-DRAFT.md`.

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
