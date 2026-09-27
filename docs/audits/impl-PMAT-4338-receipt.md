# impl receipt — PMAT-4338 = PR #4338 layernorm delta (, commit dc1628670, base 74b77fd11)

**Intent.** `prop_layernorm_zero_mean` panicked in CI (run 36086744293, shard 3/3, both tries):
`output mean = -0.000110417604` for `v = [9.388971, 9.394853]`. Classify tolerance-edge vs kernel
defect and fix without weakening the property for inputs that could expose a real bug.

**Diagnosis (measured).** var_in = 8.6e-6 (below eps = 1e-5), inv_std = 232. The output mean of a
two-pass layernorm is (mean_true - mean_f32) * inv_std; one f32 ulp of the mean at |x| = 9.39 is
~9.5e-7, times 232 is ~1.1e-4 > the fixed 1e-4. The kernel (`layernorm_scalar`) is unchanged.
Reproduced on fresh random seeds: 2 of 3 local runs at PROPTEST_CASES=20000 failed the old test.

**Change.** Test-only. tol = 1e-4 + (n+1) * f32::EPSILON * max|x| * inv_std (a bound on f32
summation error of the mean, propagated through the kernel's own inv_std). Failure message now
prints tol and var_in. The proptest regression seed file gains the CI failing seed.

**Verification.** 20 runs x 5000 cases: 0 failures. Mutant `mean = sum / n * 1.001` in the kernel
-> RED (`output mean = 0.0010000765, tol 0.00010071525, var_in 20.2`), restored -> GREEN.
CI run 36089463229 on dc1628670: success (all shards).
