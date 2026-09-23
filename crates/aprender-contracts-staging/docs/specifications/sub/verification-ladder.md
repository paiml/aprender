# 2. The Verification Ladder

Two complementary hierarchies: **proof levels** (what we verify about
the math) and **enforcement layers** (how we enforce it in the build).

## Proof Levels (theoretical guarantees)

<!-- generated from ProofLevel; do not edit -->
| Level | Method |
|-------|--------|
| L5 | Lean 4 theorem proved + every binding verified implemented |
| L4 | Lean 4 theorem proved |
| L3 | Kani bounded model check |
| L2 | Falsification tests cover the obligations |
| L1 | Contract YAML with equations |

L4 and L5 are self-declared until PVL-001 EV-8b lands: the level is computed from the contract's own YAML, not from a checked Lean discharge summary.
<!-- end generated from ProofLevel -->

## Enforcement Layers (practical deployment, strictest first)

Enforcement layers are labelled **E0–E5** so they cannot be read as the proof
levels **L1–L5** above: a layer is *how* a build enforces something, a level is
*what* has been proved about a contract.

| Layer | What it catches | Mechanism | Coverage | Misses |
|-------|----------------|-----------|----------|--------|
| **E5** | Algorithm incorrect | Lean 4 proof (no sorry) | 3 theorems (softmax) | — |
| **E4** | Logic bugs, overflows | Kani `#[kani::proof]` BMC | 985 harnesses (YAML-defined) | Inputs > bound |
| **E3** | Violated invariants | `#[contract]` debug_assert | 18 functions (forjar: 4, paiml-mcp-agent-toolkit: 11, batuta: 3) | Release builds |
| **E2** | Renamed/deleted fns | Trait `impl` (§23) | 12/33 repos have trait tests | Logic bugs |
| **E1** | Missing bindings | build.rs AllImplemented | 660 real bindings | Ghost bindings |
| **E0.5** | Schema/audit/score | `pv lint` 7 gates | 315/315 contracts pass | Impl bugs |
| **E0** | Obvious bugs | Human review | — | Everything subtle |

**E0 through E2 enforce on every `cargo build` + `cargo test`** in
the 7 repos with build.rs (aprender, trueno, entrenar, realizar,
forjar, ruchy, simular). The other 26 repos have YAML bindings only.

E3 enforces on 18 annotated functions across forjar, paiml-mcp-agent-toolkit, and batuta
debug builds. E4 and E5 are defined in YAML but not yet run in CI.

> **Spec Falsification (2026-03-28, v2.2.0):** Round 3 stripped 28,206
> ghost bindings (mass-generated entries without `module_path`). Honest
> count: 660 real bindings. Previous claim of
> 20,366 bindings / "Grade A for 33 repos" was a scoring artifact of
> YAML inflation, not real integration. See §25 for corrected baseline.

## The Provability Claim

When we say a kernel is "provable," we mean:

1. **Types:** the type system prevents invalid construction (Poka-Yoke).
2. **L2:** falsification tests (probar/proptest) exercised the property for
   10,000+ random inputs.
3. **L3:** Kani exhaustively verified it for ALL inputs within the kernel's
   natural bound (super-block size, SIMD width).
4. **L4:** a Lean 4 theorem proves it unbounded; **L5** additionally requires
   every binding verified as implemented.

For fixed-size kernel operations — which ML inference IS — bounded
verification at the natural bound IS exhaustive. A Q4_K super-block is
always 256 elements. Verifying for all 256-element inputs IS verifying
for all inputs.

## The Provability Invariant (ENFORCED)

**If a contract has proof obligations, it MUST have Kani harnesses.**
No exceptions. No "we'll add them later." The test suite enforces this:

```
∀ contract C:
  |C.proof_obligations| > 0  →  |C.kani_harnesses| > 0
  |C.proof_obligations| > 0  →  |C.falsification_tests| >= |C.proof_obligations|
```

Contracts are classified into two categories:

| Category | Has equations | Has proof_obligations | Has kani_harnesses | Example |
|---|---|---|---|---|
| **Kernel contract** | REQUIRED | REQUIRED | REQUIRED | softmax-kernel-v1 |
| **Data registry** | REQUIRED | optional | optional | special-tokens-registry-v1 |

A data registry is a contract whose `equations` encode lookup tables,
enum definitions, or configuration bounds — not computable kernels.
Data registries are declared via a `registry: true` field in metadata.
All other contracts are kernel contracts and MUST have the full
provability chain: equations → obligations → falsification tests → Kani.

If we cannot enforce provability on our own contracts, the project has
no reason to exist.

## Where Each Tool Lives

| Obligation Type | Types (rustc) | L2 (probar/proptest) | L3 (Kani) | L4 (Lean) |
|---|---|---|---|---|
| Shape correctness | ValidatedTensor | N/A | N/A | N/A |
| Softmax sums to 1 | N/A | proptest random | kani::proof <=16 | Lean theorem |
| SIMD = scalar | N/A | proptest random | kani::proof <=256 | N/A |
| No overflow | N/A | proptest edges | Kani auto | N/A |
| Quantized bsums | N/A | proptest blocks | kani::proof exact | N/A |

## How Kani (L3) and Lean (L4) Compose

Lean and Kani are NOT alternatives — they verify different things about
the SAME obligation. See **[lean-kani-composition.md](lean-kani-composition.md)**
for the full design.

**Short version:** Lean proves the algorithm over ℝ. Kani proves the Rust
code over f32. The `stub_float` strategy bridges them: Kani replaces
transcendentals (exp, log) with arbitrary-but-constrained values (what
Lean proved valid), then verifies the surrounding code preserves the
invariant. This is compositional: Lean discharges the hard math, Kani
verifies the structural code.
