/// #3968 (the test half of #3945): every GPU-whitelisted quant type, at every (k, n) the HELD
/// inventory uses, through the production dispatch (`bound_gemv`), against the CPU decoder.
///
/// WHY. The whitelist admits a TYPE; a GEMV kernel is exercised at a SHAPE. IQ4_XS was admitted on
/// one model's `[2560, 9216]` while another held model used eight other shapes, one of them with a
/// non-power-of-two super-block count (#3951). The rows here are GENERATED from the committed census
/// (`evidence/gpu-shape-census/census-*.json`, written by `scripts/lib/gguf_census.py`), never
/// hand-listed, so a newly held shape is tested as soon as it is censused.
///
/// ORACLE. The type's CPU dequantizer (several are proven against gguf-py: q2k/iq `*_gguf_py_parity`)
/// then an f64 dense matvec. Tolerance is condition-aware, as in #3951: each row's error is divided by
/// Σ|w_ij|·|x_j|, the largest value the row's dot product could take. fp32 reordering moves a result by
/// ~1e-7 of that; a wrong block, scale or index by O(1e-2) or more.
///
/// WEIGHTS. Random block bytes with every f16 scale field pinned to a finite value in [0.5, 1.5), so no
/// block decodes to inf/NaN and the kernel sees the full range of quant bits and sub-block scales.
///
/// NEGATIVE CONTROL. Per type, one block of the GPU copy is corrupted at the type's first shape. That
/// row MUST exceed the bound, or the harness cannot see a decoding error for that type and its greens
/// license nothing.
///
/// n is capped for host memory at `4096 + n % 256` when n > 4096 (a vocabulary matrix has 151 936
/// rows). A row's result depends on k; n only sets the CTA count, and the residue is kept for kernels
/// that pack several rows per CTA.
///
/// Run (needs a CUDA device):
///   CONFORMANCE_RECEIPT=<path> cargo test -p aprender-serve --features cuda --lib \
///     shape_conformance -- --ignored --nocapture
/// `scripts/gpu_shape_conformance.sh` wraps this under gpu-q and names the receipt by host.
#[cfg(test)]
#[cfg(feature = "cuda")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod shape_conformance_tests {
    use super::*;
    use crate::cuda::types::{BoundWeight, WeightQuantType};
    use std::collections::{BTreeMap, BTreeSet};

    const TOL: f32 = 1e-5;

    /// Types whose production dispatch QUANTIZES THE ACTIVATION before the dot product (the DP4A path:
    /// `hw_dp4a_q6k_gemv_into` → `q8_quantize_into`). Measured on GB10 (6e38f069a): Q6_K sat at ~7.4e-4 of the
    /// bound on all 30 shapes, the Q8 activation floor (≈1/127), while every other type matched an f32
    /// activation to ~1e-8. For these the oracle applies the SAME Q8_1 quantization, so the bound stays 1e-5
    /// and a corrupted block is still distinguishable from the design floor (#3945: "per-format tolerance").
    const Q8_ACTIVATION_TYPES: &[u32] = &[14];

    /// `aprender_gpu::kernels::quantize::Q8QuantizeKernel`, in f32 on the host: per 32-value block,
    /// scale = max|x| * (1/127); q = round-half-even(x / (scale + 1e-10)) clamped to ±127; x' = q * scale.
    fn q8_1_mirror(x: &[f32]) -> Vec<f32> {
        x.chunks(32)
            .flat_map(|b| {
                let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = amax * (1.0 / 127.0);
                let inv = 1.0 / (scale + 1e-10);
                // The kernel QUANTIZES with the f32 scale but STORES d as f16 (q8.rs `cvt_f16_f32`), and the dot
                // product dequantizes with that f16 d. A constant block max hid this; a varying one does not.
                let d = half::f16::from_f32(scale).to_f32();
                b.iter().map(move |v| (v * inv).round_ties_even().clamp(-127.0, 127.0) * d).collect::<Vec<f32>>()
            })
            .collect()
    }

    /// (elements per block, bytes per block, byte offsets of the f16 scale fields in a block).
    /// Float types are one element per "block" and have no scale field.
    fn layout(q: u32) -> Option<(usize, usize, &'static [usize])> {
        Some(match q {
            0 => (1, 4, &[]),
            1 | 30 => (1, 2, &[]),
            2 => (32, 18, &[0]),      // Q4_0  { d; qs[16] }
            3 => (32, 20, &[0, 2]),   // Q4_1  { d; m; qs[16] }
            6 => (32, 22, &[0]),      // Q5_0  { d; qh[4]; qs[16] }
            7 => (32, 24, &[0, 2]),   // Q5_1  { d; m; qh[4]; qs[16] }
            8 => (32, 34, &[0]),      // Q8_0  { d; qs[32] }
            10 => (256, 84, &[80, 82]), // Q2_K { scales[16]; qs[64]; d; dmin }
            12 => (256, 144, &[0, 2]),  // Q4_K { d; dmin; scales[12]; qs[128] }
            13 => (256, 176, &[0, 2]),  // Q5_K { d; dmin; scales[12]; qh[32]; qs[128] }
            14 => (256, 210, &[208]),   // Q6_K { ql[128]; qh[64]; scales[16]; d }
            16 => (256, 66, &[0]),      // IQ2_XXS
            18 => (256, 98, &[0]),      // IQ3_XXS
            20 => (32, 18, &[0]),       // IQ4_NL
            21 => (256, 110, &[0]),     // IQ3_S
            22 => (256, 82, &[0]),      // IQ2_S
            23 => (256, 136, &[0]),     // IQ4_XS
            _ => return None,
        })
    }

    fn dequant(q: u32, bytes: &[u8]) -> Vec<f32> {
        use crate::quantize as qz;
        let out = match q {
            0 => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            // BF16 widening is a pure 16-bit shift: the definition, not an implementation under test.
            30 => Ok(bytes.chunks_exact(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect()),
            1 => qz::dequantize_f16(bytes),
            2 => qz::dequantize_q4_0(bytes),
            3 => qz::dequantize_q4_1(bytes),
            6 => qz::dequantize_q5_0(bytes),
            7 => qz::dequantize_q5_1(bytes),
            8 => qz::dequantize_q8_0(bytes),
            10 => qz::dequantize_q2_k(bytes),
            12 => qz::dequantize_q4_k(bytes),
            13 => qz::dequantize_q5_k(bytes),
            14 => qz::dequantize_q6_k(bytes),
            16 => qz::dequantize_iq2_xxs(bytes),
            18 => qz::dequantize_iq3_xxs(bytes),
            20 => qz::iq4_nl::dequantize_iq4_nl(bytes),
            21 => qz::dequantize_iq3_s(bytes),
            22 => qz::dequantize_iq2_s(bytes),
            23 => qz::dequantize_iq4_xs(bytes),
            _ => panic!("no CPU oracle for ggml type {q}"),
        };
        out.unwrap_or_else(|e| panic!("CPU dequant of ggml type {q}: {e}"))
    }

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
    }

    /// f16 in [0.5, 1.5): exponent 14 or 15, any mantissa.
    fn finite_f16(r: &mut Lcg) -> [u8; 2] {
        let h: u16 = 0x3800 + (r.next() >> 16) as u16 % 0x0800;
        h.to_le_bytes()
    }

    fn weights(q: u32, n: usize, k: usize, seed: u32) -> Vec<u8> {
        let (per, bytes, scales) = layout(q).unwrap();
        let mut r = Lcg(seed);
        let elems = n * k;
        match q {
            // Floats: finite values in (-2, 2), written in the type's own encoding.
            0 => (0..elems).flat_map(|_| (((r.next() >> 8) as f32 / 16_777_216.0) * 4.0 - 2.0).to_le_bytes()).collect(),
            1 => (0..elems).flat_map(|_| {
                let v = ((r.next() >> 8) as f32 / 16_777_216.0) * 4.0 - 2.0;
                half::f16::from_f32(v).to_bits().to_le_bytes()
            }).collect(),
            30 => (0..elems).flat_map(|_| {
                let v = ((r.next() >> 8) as f32 / 16_777_216.0) * 4.0 - 2.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            }).collect(),
            _ => {
                let blocks = elems / per;
                let mut data = Vec::with_capacity(blocks * bytes);
                for _ in 0..blocks {
                    let start = data.len();
                    for _ in 0..bytes {
                        data.push((r.next() >> 24) as u8);
                    }
                    for &off in scales {
                        let h = finite_f16(&mut r);
                        data[start + off] = h[0];
                        data[start + off + 1] = h[1];
                    }
                }
                data
            }
        }
    }

    /// {one | pow2 | nonpow2} over super-blocks per row (aprender-70, from the IQ4_XS PTX). Kernel-
    /// specific stride-remainder and row-tail classes are added per kernel once read from its PTX.
    fn class(q: u32, k: usize) -> &'static str {
        let per = layout(q).unwrap().0;
        let spb = k / per;
        if spb == 1 { "one" } else if spb.is_power_of_two() { "pow2" } else { "nonpow2" }
    }

    fn census() -> BTreeSet<(u32, usize, usize)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let files: Vec<std::path::PathBuf> = match std::env::var("CONFORMANCE_CENSUS") {
            Ok(list) => list.split(',').map(std::path::PathBuf::from).collect(),
            Err(_) => std::fs::read_dir(root.join("evidence/gpu-shape-census"))
                .expect("evidence/gpu-shape-census")
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect(),
        };
        assert!(!files.is_empty(), "no census file: a harness with no rows proves nothing");
        let mut rows = BTreeSet::new();
        for f in files {
            let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            assert_eq!(doc["schema"], "gguf-census/v1", "{}: unknown census schema", f.display());
            for file in doc["files"].as_array().unwrap() {
                for s in file["shapes"].as_array().unwrap() {
                    let q = u32::try_from(s["qtype"].as_u64().unwrap()).unwrap();
                    // The DEVICE-INDEPENDENT list: a type excluded on this device (#4096) is still run, and
                    // recorded `excluded`, so a stale exclusion is visible (check_gpu_shape_conformance.sh).
                    if crate::gguf::gpu_unsupported_quant_qtype_on(q, None) {
                        continue; // not admitted: the whitelist refuses it, nothing to conform
                    }
                    rows.insert((q, s["k"].as_u64().unwrap() as usize, s["n"].as_u64().unwrap() as usize));
                }
            }
        }
        assert!(!rows.is_empty(), "the census holds no whitelisted 2-D tensor");
        rows
    }

    /// Worst condition-scaled error of the GPU against the oracle, `device` fed to the GPU and
    /// `bytes` to the CPU (equal for a real row; one corrupted block for the negative control).
    fn ab(exec: &mut CudaExecutor, q: u32, bytes: &[u8], device: &[u8], k: usize, n: usize) -> f32 {
        let wqt = WeightQuantType::from_ggml_type(q).unwrap_or_else(|| panic!("ggml {q} admitted but has no WeightQuantType"));
        let dense = dequant(q, bytes);
        assert_eq!(dense.len(), n * k, "ggml {q} k={k} n={n}: dequant length");
        // A golden-ratio sequence, NOT a scaled integer pattern. The Q8 scale is set by the block's own max, so any
        // x = m / c makes x/scale = m * 127 / m_max, independent of c: with m in -8..=8 every |m| = 4 is an exact
        // 63.5 tie (measured twice on gx10: /4 and /4.1 both left Q6_K at ~1-2e-5), which the device's approximate
        // reciprocal may round either way. Fractional parts of i*phi hit a tie with probability ~0.
        let input: Vec<f32> = (0..k).map(|i| ((i as f64 * 0.618_033_988_749_895).fract() as f32 - 0.5) * 4.0).collect();
        // What the kernel actually multiplies: the Q8_1-quantized activation for the DP4A types.
        let x_ref: Vec<f32> = if Q8_ACTIVATION_TYPES.contains(&q) { q8_1_mirror(&input) } else { input.clone() };

        let w_buf = GpuBuffer::from_host(&exec.context, device).unwrap();
        let x_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let y_buf = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
        let bound = BoundWeight::bind(w_buf.as_ptr(), device.len(), wqt, u32::try_from(n).unwrap(), u32::try_from(k).unwrap());
        // The executor caches the Q8-quantized activation until the forward pass invalidates it; every launch here
        // has a NEW input, so invalidate as production does per layer, or the kernel reads a stale activation.
        exec.q8_activation_valid = false;
        exec.bound_gemv(&bound, &x_buf, &y_buf).expect("bound_gemv launch");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        y_buf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        for row in 0..n {
            let w = &dense[row * k..(row + 1) * k];
            let want: f64 = w.iter().zip(&x_ref).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            let cond: f64 = w.iter().zip(&x_ref).map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs()).sum();
            let err = if got[row].is_finite() {
                ((f64::from(got[row]) - want).abs() / cond.max(1e-6)) as f32
            } else {
                f32::INFINITY // never written, or garbage: always a failure
            };
            worst = worst.max(err);
        }
        worst
    }

    #[test]
    #[ignore = "needs a CUDA device; run via scripts/gpu_shape_conformance.sh"]
    fn every_whitelisted_held_shape_conforms_on_the_device() {
        let mut exec = CudaExecutor::new(0).expect("a CUDA device: this is the one check that needs one");
        let rows = census();
        // Production initialises the executor's workspace at model load, and some kernels (the Q6_K DP4A path)
        // read its Q8 activation buffer; a bare executor panics there (#3178, measured on gx10 by this harness).
        // Initialise it exactly as production does, sized to the largest row length in the census.
        let max_k = rows.iter().map(|&(_, k, _)| k).max().unwrap_or(256);
        exec.init_workspace(max_k, max_k)
            .unwrap_or_else(|e| panic!("init_workspace({max_k}, {max_k}) failed on a bare executor: {e} — the harness cannot mirror production"));
        // The capability the production gate sees (the same query), and the types it refuses here.
        let cc_major = crate::gguf::device_cc_major();
        let excluded = |q: u32| crate::gguf::gpu_qtype_excluded_on(q, cc_major);
        let mut results = Vec::new();
        let mut controls = BTreeMap::new();
        let mut failed = 0usize;
        for &(q, k, n_held) in &rows {
            let (per, _, _) = layout(q).unwrap_or_else(|| panic!("ggml {q} is admitted but this harness has no layout for it"));
            assert_eq!(k % per, 0, "ggml {q}: k={k} is a partial block (GGUF forbids it; admission must refuse it)");
            let n = if n_held <= 4096 { n_held } else { 4096 + n_held % 256 };
            let bytes = weights(q, n, k, (q << 20) ^ (k as u32) ^ ((n as u32) << 8));
            // One row's panic (a kernel precondition, a launch error) is THAT row's failure, recorded with its
            // message; it must not lose every other row's result (the first gx10 run wrote no receipt).
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ab(&mut exec, q, &bytes, &bytes, k, n)));
            let (worst, panic_msg) = match run {
                Ok(w) => (w, None),
                Err(e) => (f32::INFINITY, Some(e.downcast_ref::<String>().cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_else(|| "panic".into()))),
            };
            let pass = worst <= TOL;
            // An excluded row never reaches the GPU in production: its failure is the RED the exclusion states,
            // not a conformance failure. Its PASS is what check_gpu_shape_conformance.sh reads as a stale exclusion.
            failed += usize::from(!pass && !excluded(q));
            eprintln!("#3968 ggml={q:2} k={k:6} n={n:6} (held {n_held:6}) class={:7} worst={worst:.3e} {}{}{}", class(q, k),
                      if pass { "ok" } else { "FAIL" }, if excluded(q) { " [excluded on this device]" } else { "" },
                      panic_msg.as_deref().map(|m| format!(" (panicked: {m})")).unwrap_or_default());
            results.push(serde_json::json!({"qtype": q, "k": k, "n_held": n_held, "n_tested": n, "class": class(q, k),
                "worst": if worst.is_finite() { serde_json::json!(worst) } else { serde_json::json!("inf") }, "pass": pass,
                "excluded": excluded(q), "panic": panic_msg}));

            if let std::collections::btree_map::Entry::Vacant(slot) = controls.entry(q) {
                // Negative control: corrupt block 0 of row 0 in the GPU copy only.
                let mut bad = bytes.clone();
                // Mid-block bytes: quant bits, never a scale field (those sit at the block edges).
                let (per_b, blk, _) = layout(q).unwrap();
                let (lo, hi) = if per_b == 1 { (0, blk * 8) } else { (blk / 2, blk / 2 + 4) };
                for b in &mut bad[lo..hi] {
                    *b ^= 0x5A;
                }
                let worst_bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ab(&mut exec, q, &bytes, &bad, k, n)))
                    .unwrap_or(f32::NAN); // a panicking control proves nothing: NaN is not > TOL, so it counts as BLIND
                let red = worst_bad > TOL;
                eprintln!("#3968 ggml={q:2} NEGATIVE CONTROL worst={worst_bad:.3e} {}", if red { "RED (good)" } else { "GREEN — this type's greens license nothing" });
                slot.insert(serde_json::json!({"k": k, "n": n, "worst": if worst_bad.is_finite() { serde_json::json!(worst_bad) } else { serde_json::json!(worst_bad.to_string()) }, "red": red}));
            }
        }
        let blind: Vec<u32> = controls.iter().filter(|(q, v)| v["red"] != true && !excluded(**q)).map(|(q, _)| *q).collect();
        let excluded_types: Vec<u32> = controls.keys().copied().filter(|&q| excluded(q)).collect();
        let receipt = serde_json::json!({
            "schema": "gpu-shape-conformance/v1",
            "tolerance": TOL,
            "cc_major": cc_major,
            "excluded_types": excluded_types,
            "rows": results,
            "negative_controls": controls.iter().map(|(q, v)| (q.to_string(), v.clone())).collect::<serde_json::Map<_, _>>(),
            "failed": failed,
            "blind_types": blind,
        });
        if let Ok(path) = std::env::var("CONFORMANCE_RECEIPT") {
            std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
            eprintln!("#3968 receipt: {path}");
        }
        assert!(blind.is_empty(), "negative control stayed GREEN for ggml types {blind:?}: the harness cannot see their errors");
        assert_eq!(failed, 0, "{failed} of {} held shapes disagree with the CPU decoder", rows.len());
    }
}
