# Phase 2: Specify -- Math to YAML Contract

## Translation Rules

Each proof obligation becomes a contract entry:

| Math Concept | YAML Field |
|-------------|------------|
| Governing equation | `equations.<name>.formula` |
| Domain | `equations.<name>.domain` |
| Codomain | `equations.<name>.codomain` |
| Invariant | `proof_obligations[].type: invariant` |
| Equivalence | `proof_obligations[].type: equivalence` |
| Bound | `proof_obligations[].type: bound` |
| Tolerance | `proof_obligations[].tolerance` |
| Falsification | `falsification_tests[]` |

## Tolerance Selection

Tolerances are derived from the arithmetic, not guessed:

| Operation | Source of Error | Typical Tolerance |
|-----------|----------------|-------------------|
| f32 addition (n terms) | Catastrophic cancellation | `n * f32::EPSILON` |
| f32 multiply-accumulate | Rounding per FMA | `sqrt(n) * f32::EPSILON` |
| Quantized dot product | Dequantization error | `ULP_TOLERANCE * f32::EPSILON` per contract |
| Softmax normalization | Exp + division | `1e-6` absolute on sum |
| RMSNorm | Sqrt + division | `1e-4` absolute |
| SIMD vs scalar | Reassociation | `ULP_TOLERANCE` (format-specific, see qdot contract) |

## The Critical Rule

> **Every YAML entry must be traceable to a specific equation in the paper.**
> If you cannot point to a formula, you cannot write a contract for it.
> If you cannot write a contract, you cannot write a falsification test.
> If you cannot write a falsification test, you are guessing.
