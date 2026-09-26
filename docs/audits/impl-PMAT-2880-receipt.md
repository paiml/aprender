# impl receipt — PMAT-2880 (slices 1–5), branch 1c/2880-vnni-reachable

| Slice | Commit | Evidence |
|---|---|---|
| 1 | 7d95e87d5 | lambda Zen 4 7960X, single-thread min-of-40: VNNI 4-row is 1.48–1.98× SLOWER than lean AVX2 → dispatch unchanged (ignored bench `vnni_reach_bench`) |
| 2 | 5badbfb68 | mini M4: NEON Q4_K×Q8_K 4.5× vs scalar single-thread; `neon_matches_scalar_bitwise` bit-equal; mutant RED |
| 3 | 4c5562ff2 | mini M4: NEON Q6_K 6.91× (2048²), 6.99× (6144×2048), 6.90× (2048×32768); `q6k_neon_matches_f64_reference` (tol 1e-5·Σ\|w·a\|) passes, mutant `sc[2*k]` RED; x86 `cargo check` rc=0 |
| 4 | 3217f69d8 | `reachability_gate` passes on x86 7960X (Q4kF32 Avx2, Q4kQ8k Avx512Vnni, Q5kF32 Scalar=declared, Q6kF32 Avx2) and M4 (Q4kF32 Scalar=declared, Q4kQ8k Neon, Q5kF32 Scalar=declared, Q6kF32 Neon). Mutants: Q6kF32 forced Scalar → RED; Q4kF32/aarch64 gap row removed → RED. x86 clippy: 0 findings in touched files |
| 5 | 1c28b2c5d | Q5_K×f32 SIMD dot: AVX2+FMA (`fused_q5k_dot_avx2`) and NEON (`fused_q5k_dot_neon`), dispatched via `selected_dot_kernel(DotOp::Q5kF32)`; both Q5kF32 gap rows removed, so `reachability_gate` now requires the SIMD path. `q5k_simd_matches_f64_reference` (tol 1e-5·Σ\|w·a\|, dispatcher bit-equal to arch kernel) + per-value probe pass on x86 7960X and M4 (56/56 each). Mutants RED on both: NEON qh bit `sub^1`, AVX2 nibble half swapped. M4 bench 5.33–5.55×, x86 7960X 7.25–7.59× vs scalar (single-thread, min-of-20) |

The gate judges the per-row dot dispatchers. `fused_q4k_q8k_parallel_matvec_into` still returns through lean AVX2 before the VNNI 4-row path (slice 1).
