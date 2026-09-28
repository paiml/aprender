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
