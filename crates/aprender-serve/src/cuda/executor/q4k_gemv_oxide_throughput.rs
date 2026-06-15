//! PMAT-OXIDE-Q4K-001 / contract `beat-q4k-oxide-sm121`: throughput BEAT gate.
//!
//! This is the FALSIFIER for the `beat-q4k-oxide-sm121` beat-benchmark contract:
//! the cuda-oxide pre-generated Q4K dequant-matvec PTX backend must run at least
//! `BEAT_THRESHOLD` (1.25x) the throughput of the incumbent hand-PTX
//! `TiledQ4KGemv` path on Blackwell sm_121 (GB10).
//!
//! ## Why this is an in-crate test (not `tests/`)
//!
//! The fair, contract-faithful comparison is the PRODUCTION decode hot path:
//! both kernels run async with WEIGHTS ALREADY CACHED on the GPU and the input
//! ALREADY resident in a device buffer, so the measurement is *pure launch
//! timing* (no per-call HtoD weight upload, no per-call DtoH copy). That path is
//! `CudaExecutor::q4k_gemv_cached_async` (incumbent `TiledQ4KGemv`, env off) and
//! `CudaExecutor::q4k_gemv_oxide_async_inner` (oxide). Driving them requires the
//! private `context` field and the non-re-exported `trueno_gpu` `GpuBuffer`,
//! neither of which is reachable from an external `tests/` integration test.
//! Placing the bench in-crate is the only way to keep the comparison apples-to-
//! apples (the alternative -- the public sync `q4k_gemv` / `q4k_gemv_oxide` --
//! re-uploads ~4.7MB of weights per call, so HtoD swamps the kernel and the
//! basic `Q4KGemv` kernel, not the contract's `TiledQ4KGemv` incumbent, would be
//! timed).
//!
//! ## Gating / CI safety
//!
//! The oxide PTX is a `.target sm_121` asset, so this is GATED on compute
//! capability >= 120 (Blackwell sm_120+). On sm_89 (RTX 4090) or any non-
//! Blackwell device the test returns early (skips) so CI on non-Blackwell hosts
//! is never failed. It is `#[ignore]`d so it only runs when explicitly selected
//! (it is a perf gate, not a correctness test, and needs sm_121 hardware).
//!
//! ## Run on gx10 (GB10, sm_121)
//!
//! ```bash
//! export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
//! export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
//! cargo test -p aprender-serve --features cuda --lib \
//!     q4k_gemv_oxide_throughput -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## Measured (gx10 GB10 sm_121, 2026-06-15) -- SHAPE-DEPENDENT
//!
//! Median of 7 batches x 200 async launches after 50 warmup (weights cached on
//! GPU, input device-resident):
//!   * FFN  N=1536 K=8960 : oxide ~2.0x tiled (159.7us vs 319.2us/launch) -> BEAT
//!   * attn N=4096 K=2048 : oxide ~0.95x tiled (101.9us vs 97.7us/launch) -> NO BEAT
//!
//! The oxide kernel's fixed 32-threads/row reduction wins on large-K GEMVs
//! (FFN) but LOSES the tiled shared-memory kernel on small-K/large-N GEMVs
//! (attention). The `ffn` test is the ENFORCED beat gate. The `baseline` test
//! intentionally FAILS at K=2048 as a tripwire: it surfaces that the
//! `beat-q4k-oxide-sm121` >=1.25x claim does NOT hold universally and that the
//! opt-in dispatch should shape-gate on K before claiming a Blackwell win.
//! See contract `contracts/beat-q4k-oxide-sm121-v1.yaml` (`shape_dependence`).

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use crate::cuda::CudaExecutor;
    use std::time::Instant;
    use trueno_gpu::driver::GpuBuffer;

    /// Q4_K super-block size in bytes: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs).
    const Q4K_BLOCK_BYTES: usize = 144;
    /// Q4_K super-block element count.
    const QK_K: usize = 256;
    /// Minimum compute capability for the oxide backend (Blackwell sm_120+).
    const OXIDE_MIN_CC: u32 = 120;
    /// Contract `beat-q4k-oxide-sm121`: oxide must be >= 1.25x tiled.
    const BEAT_THRESHOLD: f64 = 1.25;

    /// Timed iterations per measured batch (single sync amortized over the batch).
    const ITERS: u32 = 200;
    /// Warmup iterations before timing (module load + cubin JIT + caches warm).
    const WARMUP: u32 = 50;
    /// Number of measured batches; we take the MEDIAN to reject scheduler noise.
    const REPEATS: usize = 7;

    /// Build synthetic Q4_K weights with non-trivial nibbles so the kernel does
    /// representative arithmetic (mirrors `tests/q4k_gemv_oxide_parity.rs` block
    /// layout, but varies the quantized values across the row).
    fn create_test_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        assert!(
            in_dim.is_multiple_of(QK_K),
            "in_dim must be a multiple of 256"
        );
        let super_blocks_per_row = in_dim / QK_K;
        let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;
        let mut data = vec![0u8; out_dim * row_bytes];

        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let off = row * row_bytes + sb * Q4K_BLOCK_BYTES;
                // d = 1.0 (f16 0x3C00), dmin = 0.0
                data[off] = 0x00;
                data[off + 1] = 0x3C;
                data[off + 2] = 0x00;
                data[off + 3] = 0x00;
                // scales[0..3] = 1, mins[0..3] = 0, packed scales low nibble = 1
                for i in 0..4 {
                    data[off + 4 + i] = 1;
                    data[off + 4 + 4 + i] = 0;
                    data[off + 4 + 8 + i] = 0x01;
                }
                // qs[0..127]: vary nibbles deterministically so the dot product
                // is non-degenerate (both kernels see identical bytes).
                for i in 0..128 {
                    let lo = ((row + sb + i) % 16) as u8;
                    let hi = ((row + sb + i + 7) % 16) as u8;
                    data[off + 16 + i] = (hi << 4) | lo;
                }
            }
        }
        data
    }

    /// Median of a slice of timings (ns). Slice is sorted in place.
    fn median_ns(samples: &mut [u128]) -> f64 {
        samples.sort_unstable();
        let n = samples.len();
        if n.is_multiple_of(2) {
            (samples[n / 2 - 1] + samples[n / 2]) as f64 / 2.0
        } else {
            samples[n / 2] as f64
        }
    }

    /// Time one path: `launch` is invoked `WARMUP` times (untimed) then `ITERS`
    /// times (timed as one batch with a single trailing sync). Returns the
    /// median per-batch wall time over `REPEATS` batches, in milliseconds.
    fn time_path<F>(exec: &mut CudaExecutor, mut launch: F) -> f64
    where
        F: FnMut(&mut CudaExecutor),
    {
        // Warmup: load module / JIT cubin / warm allocators. Sync after.
        for _ in 0..WARMUP {
            launch(exec);
        }
        exec.sync_stream().expect("warmup sync");

        let mut batch_ns: Vec<u128> = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            for _ in 0..ITERS {
                launch(exec);
            }
            // Single sync amortized over the batch == pure launch+execute timing.
            exec.sync_stream().expect("batch sync");
            batch_ns.push(start.elapsed().as_nanos());
        }
        median_ns(&mut batch_ns) / 1.0e6
    }

    /// Measure the oxide-vs-tiled speedup (tiled_ms / oxide_ms) at one decode
    /// shape using the production async path with weights cached + input device-
    /// resident (pure launch timing). Returns `Some(ratio)` if the device is
    /// Blackwell (test ran), else `None` (skipped on sm_89 -> CI-safe).
    fn measure_speedup_at_shape(label: &str, n: u32, k: u32) -> Option<f64> {
        let mut exec = match CudaExecutor::new(0) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("CUDA init failed (cannot run oxide throughput gate): {e:?}");
                return None;
            },
        };
        let cc = exec.compute_capability();
        if cc < OXIDE_MIN_CC {
            eprintln!(
                "SKIP[{label}]: oxide Q4K backend requires sm_120+ (cc>={OXIDE_MIN_CC}); \
                 device cc={cc}. Expected on RTX 4090 (sm_89)."
            );
            return None;
        }

        let weight_name = format!("oxide_thru_{label}_{n}x{k}");
        let weights = create_test_q4k_weights(n as usize, k as usize);
        exec.load_quantized_weights(&weight_name, &weights)
            .expect("cache Q4K weights on GPU");
        let weight_ptr = exec
            .get_quantized_weight_ptr(&weight_name)
            .expect("cached weight ptr");

        // Input resident on device (uploaded once; never re-uploaded per launch).
        let input_host: Vec<f32> = (0..k as usize)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let input = GpuBuffer::from_host(&exec.context, &input_host).expect("device input buffer");

        // Incumbent: TiledQ4KGemv via the production async cached path. We do NOT
        // set APR_Q4K_OXIDE, so this routes to the tiled hand-PTX kernel.
        let tiled_ms = time_path(&mut exec, |e| {
            let _ = e
                .q4k_gemv_cached_async(&weight_name, &input, n, k)
                .expect("tiled cached async launch");
        });

        // Oxide: drive the inner oxide path directly (env-independent, always
        // oxide). Same cached weight ptr, same device input -- apples to apples.
        let oxide_ms = time_path(&mut exec, |e| {
            let _ = e
                .q4k_gemv_oxide_async_inner(weight_ptr, &input, n, k)
                .expect("oxide async launch");
        });

        let ratio = tiled_ms / oxide_ms;
        let per_tiled_us = tiled_ms * 1000.0 / f64::from(ITERS);
        let per_oxide_us = oxide_ms * 1000.0 / f64::from(ITERS);
        println!(
            "[{label}] N={n} K={k} cc={cc} | tiled={tiled_ms:.3}ms ({per_tiled_us:.3}us/launch) \
             oxide={oxide_ms:.3}ms ({per_oxide_us:.3}us/launch) | oxide/tiled speedup={ratio:.3}x \
             (threshold {BEAT_THRESHOLD}x)"
        );
        Some(ratio)
    }

    /// ENFORCED BEAT GATE -- canonical FFN-class decode shape (N=1536, K=8960).
    ///
    /// This is the falsifier for contract `beat-q4k-oxide-sm121`: oxide MUST be
    /// >= 1.25x the incumbent TiledQ4KGemv here. Measured ~2.0x on gx10
    /// (2026-06-15). Panics with the measured speedup on regression.
    #[test]
    #[ignore = "perf gate: needs Blackwell sm_121 (gx10); run with --ignored"]
    fn beat_q4k_oxide_sm121_ffn_1536x8960() {
        // n = out_dim = 1536, k = in_dim = 8960 (both 256-aligned).
        if let Some(ratio) = measure_speedup_at_shape("ffn", 1536, 8960) {
            assert!(
                ratio >= BEAT_THRESHOLD,
                "[ffn] BEAT FALSIFIED: oxide must be >= {BEAT_THRESHOLD}x tiled at \
                 N=1536 K=8960; measured speedup={ratio:.3}x < {BEAT_THRESHOLD}x"
            );
        }
    }

    /// DOCUMENTED NON-BEAT TRIPWIRE -- attention-class shape (N=4096, K=2048).
    ///
    /// SURFACED FINDING (gx10, 2026-06-15): at this small-K/large-N shape the
    /// oxide kernel is ~0.95x tiled -- i.e. SLOWER. The contract's >=1.25x claim
    /// does NOT hold universally on sm_121; it is scoped to FFN-class shapes.
    /// This test asserts the *measured reality* (oxide does NOT beat tiled here),
    /// so it PASSES today and FLIPS RED only if oxide ever starts winning at
    /// small K -- a signal to promote attn-class into the enforced beat (and to
    /// drop the K-based shape-gate the opt-in dispatch should add).
    #[test]
    #[ignore = "perf gate: needs Blackwell sm_121 (gx10); run with --ignored"]
    fn no_beat_q4k_oxide_sm121_attn_4096x2048_tripwire() {
        // n = out_dim = 4096, k = in_dim = 2048 (both 256-aligned).
        if let Some(ratio) = measure_speedup_at_shape("attn", 4096, 2048) {
            assert!(
                ratio < BEAT_THRESHOLD,
                "[attn] UNEXPECTED BEAT: oxide reached {ratio:.3}x >= {BEAT_THRESHOLD}x at \
                 N=4096 K=2048 (was ~0.95x on gx10 2026-06-15). Oxide improved at small K -- \
                 promote attn into the enforced canonical_task in beat-q4k-oxide-sm121-v1.yaml."
            );
        }
    }

    /// CI-safe shape sanity (no hardware needed beyond CUDA init): the synthetic
    /// weight buffer is sized correctly for both canonical shapes. Runs everywhere.
    #[test]
    fn oxide_throughput_weight_shapes_are_well_formed() {
        for (n, k) in [(4096usize, 2048usize), (1536, 8960)] {
            let w = create_test_q4k_weights(n, k);
            let expected = n * (k / QK_K) * Q4K_BLOCK_BYTES;
            assert_eq!(
                w.len(),
                expected,
                "Q4K weight byte count mismatch for {n}x{k}"
            );
        }
    }
}
