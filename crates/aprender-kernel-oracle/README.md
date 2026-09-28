# aprender-kernel-oracle

KTEST-02 (KTEST-001 §3.2): the independent f64 scalar oracle and the
per-element margin harness that judges a kernel's output against it.

Part of the [Aprender](https://github.com/paiml/aprender) monorepo.

- `oracle` — the mathematical definition of each op in f64 (dot, GEMV, GEMM,
  sum, max, softmax, RMSNorm, SiLU). No SIMD, no dependencies.
- `error_model` — the bound Bᵢ per output element from the kernel's declared
  error model (`EM-DOT`, `EM-RED`, `EM-DEQ`). A model without an implemented
  bound is refused, never guessed (STOP S-1).
- `margin` — mᵢ = |ŷᵢ − yᵢ| / Bᵢ. Pass iff max mᵢ ≤ 1, no stray NaN, and the
  Inf set and signs match. NMSE is reported alongside, llama.cpp-comparable,
  but it is not the gate: F-12 shows one bad tile in 10⁶ outputs passing NMSE
  while the margin is RED.
- `shapes` — KTEST-03 (§4): the dimension values to test, derived from the
  registered tile T and vector width V: 0, 1, V−1, V, V+1, 2V, 2V+1, T−1, T,
  T+1, 2T, 2T+1, plus device limits ±1, 2³¹ and production shapes. 2T+1 is
  there because the `cls4` model has six classes and the spec's set reaches five.
- `inputs` — KTEST-03 (§3.3): uniform, normal, wide-range, cancellation,
  all-equal, near-overflow, subnormal and NaN/Inf inputs, each reproducible
  from its seed (`rng`, splitmix64).
- Falsifiers: F-1 (tile-tail over-read) is RED on exactly the tail classes;
  F-6 (softmax without max-subtraction) is RED on the near-overflow class.
