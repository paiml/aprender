# 2. The Verification Ladder

Version: 1.0
Status: proposed
Date: 2026-04-09

Two complementary hierarchies: **proof levels** (what we verify about
the math) and **enforcement layers** (how we enforce it in the build).

## Proof Levels (theoretical guarantees)

```
Level   Method                  Tool            Guarantee
-----   ------                  ----            ---------
  L5    Theorem proving         Lean 4          True for ALL inputs. Period.
  L4    Bounded model check     Kani            True for ALL inputs <= size N.
  L3    Property-based test     probar/proptest True for ~10,000 random inputs.
  L2    Falsification test      #[test]         True for specific edge cases.
  L1    Type system             rustc           True by construction.
  L0    Code review             Human eyes      "Looks right to me."
```

## Enforcement Layers (practical deployment, strictest first)

| Layer | What it catches | Mechanism | Coverage | Misses |
|-------|----------------|-----------|----------|--------|
| **L5** | Algorithm incorrect | Lean 4 proof (no sorry) | 3 theorems (softmax) | — |
| **L4** | Logic bugs, overflows | Kani `#[kani::proof]` BMC | 985 harnesses (YAML-defined) | Inputs > bound |
| **L3** | Violated invariants | `#[contract]` debug_assert | 18 functions (forjar: 4, paiml-mcp-agent-toolkit: 11, batuta: 3) | Release builds |
| **L2** | Renamed/deleted fns | Trait `impl` (§23) | 12/33 repos have trait tests | Logic bugs |
| **L1** | Missing bindings | build.rs AllImplemented | 660 real bindings | Ghost bindings |
| **L0.5** | Schema/audit/score | `pv lint` 7 gates | 315/315 contracts pass | Impl bugs |
| **L0** | Obvious bugs | Human review | — | Everything subtle |

**L0 through L2 enforce on every `cargo build` + `cargo test`** in
the 7 repos with build.rs (aprender, trueno, entrenar, realizar,
forjar, ruchy, simular). The other 26 repos have YAML bindings only.

L3 enforces on 18 annotated functions across forjar, paiml-mcp-agent-toolkit, and batuta
debug builds. L4 and L5 are defined in YAML but not yet run in CI.

> **Spec Falsification (2026-03-28, v2.2.0):** Round 3 stripped 28,206
> ghost bindings (mass-generated entries without `module_path`). Honest
> count: 660 real bindings. Previous claim of
> 20,366 bindings / "Grade A for 33 repos" was a scoring artifact of
> YAML inflation, not real integration. See §25 for corrected baseline.

## The Provability Claim

When we say a kernel is "provable," we mean:

1. **L1:** The type system prevents invalid construction (Poka-Yoke).
2. **L3:** probar tested the property for 10,000+ random inputs.
3. **L4:** Kani exhaustively verified for ALL inputs within the kernel's
   natural bound (super-block size, SIMD width).

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

| Obligation Type | L1 (Types) | L3 (probar) | L4 (Kani) | L5 (Lean) |
|---|---|---|---|---|
| Shape correctness | ValidatedTensor | N/A | N/A | N/A |
| Softmax sums to 1 | N/A | proptest random | kani::proof <=16 | Lean theorem |
| SIMD = scalar | N/A | proptest random | kani::proof <=256 | N/A |
| No overflow | N/A | proptest edges | Kani auto | N/A |
| Quantized bsums | N/A | proptest blocks | kani::proof exact | N/A |

## How L4 and L5 Compose

Lean and Kani are NOT alternatives — they verify different things about
the SAME obligation. See **[lean-kani-composition.md](lean-kani-composition.md)**
for the full design.

**Short version:** Lean proves the algorithm over ℝ. Kani proves the Rust
code over f32. The `stub_float` strategy bridges them: Kani replaces
transcendentals (exp, log) with arbitrary-but-constrained values (what
Lean proved valid), then verifies the surrounding code preserves the
invariant. This is compositional: Lean discharges the hard math, Kani
verifies the structural code.
