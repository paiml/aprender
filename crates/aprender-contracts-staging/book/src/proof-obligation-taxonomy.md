# Proof Obligation Taxonomy

Every mathematical property from a paper falls into one of these categories:

## Invariant

**Definition:** A property that holds for ALL valid inputs.

**Pattern:** `for all x in Domain: P(f(x))` is true.

**Examples:**
- Softmax: `sum(output) = 1.0`
- RMSNorm: `rms(output) ~= 1.0` (before scaling)
- Attention weights: `sum(weights_per_query) = 1.0`

**Test strategy:** probar property test with random inputs.
**Proof strategy:** `#[kani::proof]` with `kani::any()` inputs up to natural bound.

## Equivalence

**Definition:** Two computations produce the same result.

**Pattern:** `for all x: f(x) = g(x)` within tolerance.

**Examples:**
- Softmax: `softmax(x) = softmax(x - max(x))`
- SIMD vs scalar: `f_avx2(x) ~= f_scalar(x)`
- GPU vs CPU: `f_gpu(x) ~= f_cpu(x)`

**Test strategy:** probar property test comparing two implementations.
**Proof strategy:** `#[kani::proof]` comparing both implementations on
`kani::any()`. For SIMD vs scalar, this is the highest-value Kani harness --
SIMD intrinsics are fully supported.

## Bound

**Definition:** Output is bounded within a range.

**Pattern:** `for all x: a <= f(x)_i <= b`

**Examples:**
- Softmax: `0 < output_i < 1`
- Sigmoid: `0 < output < 1`
- Tanh: `-1 < output < 1`
- ReLU: `output >= 0`

**Test strategy:** probar property test checking range.
**Proof strategy:** `#[kani::proof]` asserting bounds on all outputs. Kani
excels at this -- range checks are simple assertions.

## Monotonicity

**Definition:** Order is preserved (or reversed) through the function.

**Pattern:** `x_i > x_j implies f(x)_i > f(x)_j` (or `<` for reversed).

**Examples:**
- Softmax: order-preserving
- Negation: order-reversing

**Test strategy:** probar property test with ordered pairs.
**Proof strategy:** `#[kani::proof]` with two symbolic values where
`kani::assume(a > b)`, assert `f(a) > f(b)`. Requires stub_float for
transcendentals.

## Idempotency

**Definition:** Applying the function twice yields the same result as once.

**Pattern:** `f(f(x)) = f(x)`

**Examples:**
- ReLU: `relu(relu(x)) = relu(x)`
- Softmax: NOT idempotent (applying twice changes output)
- Normalization: NOT idempotent in general

**Test strategy:** probar property test composing function with itself.
**Proof strategy:** `#[kani::proof]` applying function twice, asserting
`f(f(x)) == f(x)`. Exact for integer operations (ReLU).

## Linearity / Homogeneity

**Definition:** Scaling input scales output proportionally.

**Pattern:** `f(ax) = a*f(x)` (homogeneous of degree 1).

**Examples:**
- Matrix multiply in V: `Attn(Q,K,aV) = a*Attn(Q,K,V)`
- ReLU: `relu(ax) = a*relu(x)` for a > 0
- Softmax: NOT homogeneous

**Test strategy:** probar metamorphic test with random scaling factor.
**Proof strategy:** `#[kani::proof]` with symbolic `alpha: f32`, verify
`f(alpha * x) == alpha * f(x)`. Requires stub_float if function uses
transcendentals.

## Symmetry / Antisymmetry

**Definition:** Function behavior under input permutation.

**Examples:**
- Dot product: `dot(a,b) = dot(b,a)` (symmetric)
- Softmax: NOT permutation-invariant on output (but set-invariant on output set)

## Associativity

**Definition:** Grouping doesn't matter.

**Pattern:** `(a + b) + c = a + (b + c)`

**Critical for SIMD:** Floating-point addition is NOT associative. SIMD
reassociation changes results. This is why we need ULP tolerances.

**Examples:**
- FP addition: NOT associative (SIMD source of error)
- Integer addition: associative (SIMD is exact)

## Conservation

**Definition:** A quantity is preserved through computation.

**Pattern:** `Q(state_before) = Q(state_after)`

**Examples:**
- EDD harmonic oscillator: `E = KE + PE` is constant
- Attention: total probability mass = 1.0 per query
- Residual connections: information is preserved (additive)
