# PyTorch Testing Patterns: Learnings for Trueno

**Date**: 2025-11-16
**Source**: PyTorch codebase analysis (v2.x)
**Relevance**: Testing infrastructure for multi-backend SIMD library

---

## Executive Summary

PyTorch provides a sophisticated testing infrastructure with patterns directly applicable to Trueno's multi-backend architecture. Key learnings include dtype-specific tolerance testing, property-based edge case coverage, dispatch mechanism validation, and memory layout testing critical for SIMD correctness.

---

## 1. Numerical Tolerance Testing

### Source: `torch/testing/_comparison.py`

**Pattern**: Dtype-specific tolerances with relative + absolute error formula

```python
# Lines 54-70: Default precision by dtype
_DTYPE_PRECISIONS = {
    torch.float16: (0.001, 1e-5),      # (rtol, atol)
    torch.bfloat16: (0.016, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}

# Formula: |actual - expected| <= atol + rtol * |expected|
```

**Application to Trueno**:

```rust
// Proposed addition to trueno/src/testing.rs
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
}

pub const DTYPE_TOLERANCES: &[(DType, Tolerance)] = &[
    (DType::F16, Tolerance { rtol: 1e-3, atol: 1e-5 }),
    (DType::F32, Tolerance { rtol: 1.3e-6, atol: 1e-5 }),
    (DType::F64, Tolerance { rtol: 1e-7, atol: 1e-7 }),
];

pub fn assert_close_with_tolerance(actual: f32, expected: f32, tol: &Tolerance) {
    let abs_diff = (actual - expected).abs();
    let threshold = tol.atol as f32 + tol.rtol as f32 * expected.abs();
    assert!(
        abs_diff <= threshold,
        "Values not close:\n  actual:   {}\n  expected: {}\n  diff:     {}\n  threshold: {}",
        actual, expected, abs_diff, threshold
    );
}
```

**Key Insight**: Current Trueno uses `assert_eq!` for f32 comparisons, which is too strict for floating-point operations.

---

## 2. Property-Based Testing with Extremal Values

### Source: `torch/testing/_internal/hypothesis_utils.py`

**Pattern**: Systematic generation of edge cases using Hypothesis

```python
# Lines 162-174: Shape generation
@st.composite
def array_shapes(draw, min_dims=1, max_dims=None, min_side=1, max_side=None, max_numel=None):
    """Return a strategy for array shapes (tuples of int >= 1)."""
    candidate = st.lists(st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims)
    if max_numel is not None:
        candidate = candidate.filter(lambda x: reduce(int.__mul__, x, 1) <= max_numel)
    return draw(candidate.map(tuple))

# Lines 99-103: Overflow prevention
def assume_not_overflowing(tensor, qparams):
    """Filter to avoid overflows with quantized tensors"""
    min_value, max_value = _get_valid_min_max(qparams)
    assume(tensor.min() >= min_value)
    assume(tensor.max() <= max_value)
```

**Application to Trueno**:

```rust
// Enhanced proptest strategies in trueno tests
use proptest::prelude::*;

// Current: uses full f32 range
prop_compose! {
    fn arb_f32()(val in -1e6_f32..1e6_f32) -> f32 {
        val  // Constrained to prevent overflow in accumulation
    }
}

// Addition: extremal value testing
#[test]
fn test_extremal_values() {
    let extremals = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    for &a in &extremals {
        for &b in &extremals {
            let va = Vector::from_slice(&[a, a]);
            let vb = Vector::from_slice(&[b, b]);

            // Verify NaN propagation
            let result = va.add(&vb).unwrap();
            if a.is_nan() || b.is_nan() {
                assert!(result.as_slice()[0].is_nan());
            }

            // Verify infinity handling
            if a.is_infinite() && b.is_infinite() && a.signum() != b.signum() {
                assert!(result.as_slice()[0].is_nan()); // inf - inf = NaN
            }
        }
    }
}
```

**Key Insight**: Trueno's current proptest suite doesn't test inf/nan edge cases systematically.

---

## 3. Memory Layout Testing for SIMD

### Source: `torch/testing/_internal/opinfo/core.py`, `test/test_ops.py`

**Pattern**: Test both contiguous and non-contiguous memory layouts

```python
# Lines 247-250: SampleInput noncontiguous variant
def noncontiguous(self):
    """Returns variant with noncontiguous tensors"""
    def to_noncontiguous(t):
        if isinstance(t, torch.Tensor):
            return noncontiguous_like(t)
        return t
    return self.transform(to_noncontiguous)

# test_ops.py Lines 420+: Noncontiguous testing
@ops(op_db)
def test_noncontiguous_samples(self, device, dtype, op):
    """Test with noncontiguous tensors (important for vectorization)"""
    for sample in op.sample_inputs(device, dtype):
        result_cont = op(sample.input, *sample.args, **sample.kwargs)

        sample_noncontig = sample.noncontiguous()
        result_noncontig = op(sample_noncontig.input, *sample_noncontig.args, **sample_noncontig.kwargs)

        torch.testing.assert_close(result_cont, result_noncontig)
```

**Application to Trueno**:

```rust
// CRITICAL: Trueno currently assumes contiguous Vec<f32>
// Should add support for strided access patterns

#[test]
fn test_noncontiguous_memory() {
    // Layout: [valid, garbage, valid, garbage, valid]
    let data = vec![1.0, 999.0, 2.0, 999.0, 3.0, 999.0, 4.0];

    // TODO: Add strided iterator support
    // let v = Vector::from_strided(&data, stride=2, len=4);
    // assert_eq!(v.sum().unwrap(), 10.0);  // 1 + 2 + 3 + 4

    // Current workaround: manually extract
    let extracted: Vec<f32> = data.iter().step_by(2).copied().collect();
    let v = Vector::from_slice(&extracted);
    assert_eq!(v.sum().unwrap(), 10.0);
}

#[test]
fn test_unaligned_memory() {
    // Test that SIMD backends handle unaligned starts correctly
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let v = Vector::from_slice(&data[1..]); // Offset by 4 bytes (unaligned for 16-byte SSE)
    assert_eq!(v.sum().unwrap(), 10.0);
}
```

**Key Insight**: Real-world data is often non-contiguous (e.g., matrix columns). SIMD code must handle this gracefully.

---

## 4. Backend Dispatch Testing

### Source: `test/test_dispatch.py`

**Pattern**: Test dispatch mechanism for commutativity and invariants

```python
# Lines 57-200: Commutativity testing
class TestDispatch(TestCase):
    def run_ops(self, name, ops, ctor_order=None, dtor_order=None):
        """
        Run operations in specified order, checking invariants at each step.
        Validates that dispatcher state is consistent regardless of order.
        """
        active_ops = set()

        def check_invariants(actual_provenance):
            C._dispatch_check_invariants(name)
            actual_state = C._dispatch_dump(f"{test_namespace}::{name}")
            expected_state = results.setdefault(
                frozenset(active_ops),
                Result(actual_state, actual_table, actual_provenance)
            )
            self.assertEqual(actual_state, expected_state.state)
```

**Application to Trueno**:

```rust
#[test]
fn test_backend_selection_deterministic() {
    // Current test (already in trueno/src/lib.rs:227)
    let backend1 = select_best_available_backend();
    let backend2 = select_best_available_backend();
    assert_eq!(backend1, backend2);
}

#[test]
fn test_backend_cross_validation() {
    // All backends should produce equivalent results (within tolerance)
    let input = &[1.0, 2.0, 3.0, 4.0];

    let backends = vec![
        Backend::Scalar,
        #[cfg(target_arch = "x86_64")]
        Backend::SSE2,
        #[cfg(target_arch = "x86_64")]
        Backend::AVX2,
    ];

    let mut results = vec![];
    for backend in backends {
        let v = Vector::from_slice_with_backend(input, backend);
        results.push(v.sum().unwrap());
    }

    // All backends should agree within tolerance
    for i in 1..results.len() {
        let diff = (results[0] - results[i]).abs();
        assert!(diff < 1e-5, "Backend mismatch: {} vs {}", results[0], results[i]);
    }
}

#[test]
fn test_backend_override_respected() {
    // Verify explicit backend selection works
    let v_scalar = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    #[cfg(target_arch = "x86_64")]
    let v_sse2 = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::SSE2);

    // Internal backend field should match request
    // (Would need to expose backend getter for this test)
}
```

**Key Insight**: Trueno should validate that all backends produce consistent results, not just test each in isolation.

---

## 5. Per-Backend Tolerance Configuration

### Source: `torch/testing/_internal/common_device_type.py`

**Pattern**: Different backends have different precision characteristics

```python
# Lines 1548-1565: Tolerance overrides per dtype/device
tol = namedtuple("tol", ["atol", "rtol"])

@toleranceOverride({
    torch.float: tol(atol=1e-2, rtol=1e-3),
    torch.double: tol(atol=1e-4, rtol=0)
})
def test_example(self, device, dtype, op):
    pass

# Lines 326-342: Dynamic tolerance management
class DeviceTypeTestBase(TestCase):
    precision = 1e-5  # atol
    rel_tol = 0

    @property
    def rel_tol(self):
        return self._tls.rel_tol
```

**Application to Trueno**:

```rust
// Proposed: trueno/src/testing.rs
pub const BACKEND_TOLERANCES: &[(Backend, Tolerance)] = &[
    (Backend::Scalar, Tolerance { rtol: 1e-7, atol: 1e-7 }),
    (Backend::SSE2, Tolerance { rtol: 1e-6, atol: 1e-5 }),
    (Backend::AVX2, Tolerance { rtol: 1e-6, atol: 1e-5 }),  // FMA may introduce rounding
    (Backend::NEON, Tolerance { rtol: 1e-6, atol: 1e-5 }),
    (Backend::WasmSIMD, Tolerance { rtol: 1e-6, atol: 1e-5 }),
];

pub fn get_backend_tolerance(backend: Backend) -> Tolerance {
    BACKEND_TOLERANCES
        .iter()
        .find(|(b, _)| *b == backend)
        .map(|(_, t)| *t)
        .unwrap_or(Tolerance { rtol: 1e-5, atol: 1e-5 })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dot_product_with_backend_tolerance() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);

        let backend = a.backend(); // Would need getter
        let tolerance = get_backend_tolerance(backend);

        let result = a.dot(&b).unwrap();
        let expected = 32.0;

        assert_close_with_tolerance(result, expected, &tolerance);
    }
}
```

**Key Insight**: AVX2's FMA (fused multiply-add) can produce different rounding than separate mul+add, requiring looser tolerances.

---

## 6. Denormal/Subnormal Number Handling

### Source: `torch/testing/_internal/hypothesis_utils.py`

**Pattern**: Test behavior near zero with denormal numbers

```python
# Lines 194-217: Tensor generation with range constraints
@st.composite
def tensor(draw, shapes=None, elements=None, qparams=None, dtype=np.float32):
    if elements is None:
        elements = floats(-1e6, 1e6, allow_nan=False, width=32)
    X = draw(stnp.arrays(dtype=dtype, elements=elements, shape=_shape))
    assume(not (np.isnan(X).any() or np.isinf(X).any()))
    return X, None
```

**Application to Trueno**:

```rust
#[test]
fn test_denormal_numbers() {
    // Smallest normal f32: 2^-126 ≈ 1.175e-38
    // Smallest subnormal: 2^-149 ≈ 1.4e-45

    let denorm = f32::MIN_POSITIVE / 2.0; // Subnormal
    assert!(denorm > 0.0 && denorm < f32::MIN_POSITIVE);

    let v = Vector::from_slice(&[denorm, denorm, denorm, denorm]);

    // Test addition doesn't flush to zero
    let result = v.add(&v).unwrap();
    assert!(result.as_slice()[0] > 0.0, "Denormal addition should not flush to zero");

    // Test multiplication behavior
    let scaled = v.mul(&v).unwrap();
    // Product may underflow to zero (expected for denorm^2)

    // Test sum accumulation
    let sum = v.sum().unwrap();
    assert!(sum > 0.0, "Denormal sum should not flush to zero");
}

#[test]
fn test_gradual_underflow() {
    // IEEE 754 guarantees gradual underflow (denormals)
    let small = f32::MIN_POSITIVE / 4.0;
    let v = Vector::from_slice(&[small, small, small, small]);

    let sum = v.sum().unwrap();
    assert_eq!(sum, small * 4.0);
}
```

**Key Insight**: SIMD implementations may flush denormals to zero (`FTZ` mode), affecting numerical accuracy.

---

## 7. Error Input Testing

### Source: `torch/testing/_internal/opinfo/core.py`

**Pattern**: Systematically test error paths

```python
# Lines 331-343: Error input specification
class ErrorInput:
    """A sample that will cause an error"""
    def __init__(self, sample_input, *, error_type=RuntimeError, error_regex):
        self.sample_input = sample_input
        self.error_type = error_type
        self.error_regex = error_regex

# Usage in tests:
error_inputs_func = error_inputs_add  # Function returning ErrorInput list
```

**Application to Trueno**:

```rust
#[test]
fn test_error_inputs() {
    struct ErrorCase {
        name: &'static str,
        a: Vec<f32>,
        b: Vec<f32>,
        error_pattern: &'static str,
    }

    let cases = vec![
        ErrorCase {
            name: "size mismatch",
            a: vec![1.0, 2.0],
            b: vec![1.0, 2.0, 3.0],
            error_pattern: "SizeMismatch",
        },
        ErrorCase {
            name: "empty vectors",
            a: vec![],
            b: vec![],
            error_pattern: "EmptyVector",
        },
    ];

    for case in cases {
        let va = Vector::from_slice(&case.a);
        let vb = Vector::from_slice(&case.b);

        let result = va.add(&vb);
        assert!(result.is_err(), "Expected error for case: {}", case.name);

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains(case.error_pattern),
            "Error message '{}' should contain '{}'",
            err_msg,
            case.error_pattern
        );
    }
}
```

**Key Insight**: Trueno has basic error tests but lacks systematic enumeration of all error paths.

---

## 8. Extremal Value Testing

### Source: `torch/testing/_internal/common_methods_invocations.py`

**Pattern**: Separate test suites for small/large/extremal values

```python
# Lines 2112-2135: Extremal value generation
def generate_elementwise_binary_extremal_value_tensors(op_info, device, dtype, **kwargs):
    """Generate test cases with extreme values for numerical stability testing"""
    _float_extremals = (float("inf"), float("-inf"), float("nan"))

    for item in product(_float_extremals, _float_extremals):
        yield SampleInput(
            make_tensor((2, 2), device=device, dtype=dtype, values=item[0]),
            make_tensor((2, 2), device=device, dtype=dtype, values=item[1])
        )

# Lines 2493-2530: Small/large value testing
def generate_elementwise_binary_small_value_tensors(...):
    """Generate test cases with small values near machine epsilon"""

def generate_elementwise_binary_large_value_tensors(...):
    """Generate test cases with large values near overflow"""
```

**Application to Trueno**:

```rust
#[test]
fn test_small_values() {
    let epsilon = f32::EPSILON; // 2^-23 ≈ 1.19e-7

    let v = Vector::from_slice(&[epsilon, epsilon * 2.0, epsilon * 3.0, epsilon * 4.0]);

    // Test that operations preserve precision
    let sum = v.sum().unwrap();
    let expected = epsilon * 10.0;

    // Require high relative precision for small values
    let rel_error = ((sum - expected) / expected).abs();
    assert!(rel_error < 1e-5, "Lost precision with small values: rel_error = {}", rel_error);
}

#[test]
fn test_large_values() {
    let large = f32::MAX / 10.0; // 3.4e37

    let v = Vector::from_slice(&[large, large]);

    // Test addition doesn't overflow
    let sum = v.sum().unwrap();
    assert!(sum.is_finite(), "Large value sum should not overflow");
    assert_eq!(sum, large * 2.0);

    // Test multiplication DOES overflow
    let product = v.mul(&v).unwrap();
    assert!(product.as_slice()[0].is_infinite(), "Expected overflow for large^2");
}

#[test]
fn test_mixed_magnitude_values() {
    // Kahan summation test: adding small values to large accumulator
    let large = 1e10_f32;
    let small = 1.0_f32;

    let v = Vector::from_slice(&[large, small, small, small]);

    // Naive summation loses precision: large + small + small + small ≈ large
    // Kahan summation should preserve: large + 3.0
    let sum = v.sum().unwrap();

    // Current sum() may lose precision, sum_kahan() should not
    let kahan_sum = v.sum_kahan().unwrap();
    assert_eq!(kahan_sum, large + 3.0);
}
```

**Key Insight**: Trueno has `sum_kahan()` for numerical stability but doesn't systematically test it against edge cases.

---

## 9. Variant Consistency Testing

### Source: `test/test_ops.py`

**Pattern**: Test function/method/inplace variants produce identical results

```python
# Lines 600+: Variant consistency
@ops(op_db)
def test_variant_consistency_eager(self, device, dtype, op):
    """Test function, method, and inplace variants are consistent"""
    # op.op() is the function
    # op.method_variant() is the Tensor method
    # op.inplace_variant() is the inplace version
    # All should produce identical results
```

**Application to Trueno**:

```rust
// Trueno currently only has method form (e.g., `v.add(&other)`)
// If adding function and inplace variants:

#[test]
fn test_variant_consistency() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);

    // Method form (current)
    let result_method = a.add(&b).unwrap();

    // Function form (future)
    // let result_function = vector::add(&a, &b).unwrap();

    // Inplace form (future)
    // let mut a_copy = a.clone();
    // a_copy.add_inplace(&b).unwrap();

    // All should produce identical results
    // assert_eq!(result_method.as_slice(), result_function.as_slice());
    // assert_eq!(result_method.as_slice(), a_copy.as_slice());
}

#[test]
fn test_inplace_efficiency() {
    // Inplace should not allocate
    let mut v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let other = Vector::from_slice(&[4.0, 5.0, 6.0]);

    let ptr_before = v.as_slice().as_ptr();
    // v.add_inplace(&other).unwrap();
    let ptr_after = v.as_slice().as_ptr();

    // assert_eq!(ptr_before, ptr_after, "Inplace should not reallocate");
}
```

**Key Insight**: When Trueno adds inplace variants, must test they don't allocate and produce same results as out-of-place.

---

## 10. Device & Dtype Parametrization Infrastructure

### Source: `torch/testing/_internal/common_device_type.py`

**Pattern**: Automatic test instantiation per backend/dtype

```python
# Lines 70-160: Instantiate device-type tests
# A template class instantiates separate test classes per device type
class TestClassFoo(TestCase):
    def test_bar(self, device):
        pass

# Becomes:
# TestClassFooCPU with test_bar_cpu() running test_bar('cpu')
# TestClassFooCUDA with test_bar_cuda() running test_bar('cuda:0')
```

**Application to Trueno**:

```rust
// Rust doesn't have Python's metaprogramming, but can use macros

macro_rules! backend_tests {
    ($test_name:ident, $body:expr) => {
        paste::paste! {
            #[test]
            fn [<$test_name _scalar>]() {
                let backend = Backend::Scalar;
                $body(backend);
            }

            #[cfg(target_arch = "x86_64")]
            #[test]
            fn [<$test_name _sse2>]() {
                let backend = Backend::SSE2;
                $body(backend);
            }

            #[cfg(target_arch = "x86_64")]
            #[test]
            fn [<$test_name _avx2>]() {
                if is_x86_feature_detected!("avx2") {
                    let backend = Backend::AVX2;
                    $body(backend);
                }
            }

            #[cfg(target_arch = "aarch64")]
            #[test]
            fn [<$test_name _neon>]() {
                let backend = Backend::NEON;
                $body(backend);
            }
        }
    };
}

// Usage:
backend_tests!(test_addition, |backend: Backend| {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0], backend);
    let b = Vector::from_slice_with_backend(&[3.0, 4.0], backend);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 6.0]);
});
```

**Key Insight**: Systematic backend coverage requires infrastructure, not just manual duplication.

---

## Summary: Priority Improvements for Trueno

### High Priority (Correctness-Critical)

1. **Tolerance-based assertions** - Replace `assert_eq!` with `assert_close` for floating-point
2. **Backend cross-validation** - All backends must agree within tolerance
3. **Extremal value tests** - Systematic inf/nan handling verification
4. **Non-contiguous memory** - Test strided access patterns (critical for real-world SIMD)

### Medium Priority (Robustness)

5. **Denormal number tests** - Verify no flush-to-zero issues
6. **Per-backend tolerances** - AVX2 FMA may need looser bounds
7. **Error path coverage** - Systematic `ErrorInput`-style testing
8. **Small/large value suites** - Numerical stability regression detection

### Low Priority (Infrastructure)

9. **Backend parametrization macro** - Reduce test duplication
10. **Property-based overflow prevention** - Constrain proptest input ranges

---

## Implementation Checklist

- [ ] Add `trueno/src/testing.rs` with tolerance helpers
- [ ] Replace critical `assert_eq!` calls with `assert_close`
- [ ] Add `test_backend_cross_validation()` to integration tests
- [ ] Add `test_extremal_values()` for inf/nan edge cases
- [ ] Add `test_denormal_numbers()` for subnormal handling
- [ ] Add `test_noncontiguous_memory()` (may need API extension)
- [ ] Add `test_small_values()` and `test_large_values()` suites
- [ ] Enhance error tests with `ErrorCase` struct pattern
- [ ] Consider backend parametrization macro for reduced duplication
- [ ] Document tolerance rationale in `docs/TESTING.md`

---

## References

| Component | PyTorch File |
|-----------|--------------|
| Numerical Comparison | `torch/testing/_comparison.py` |
| Tolerance Management | `torch/testing/_internal/common_device_type.py` |
| Hypothesis Strategies | `torch/testing/_internal/hypothesis_utils.py` |
| OpInfo Structure | `torch/testing/_internal/opinfo/core.py` |
| Sample Inputs | `torch/testing/_internal/common_methods_invocations.py` |
| Dispatch Testing | `test/test_dispatch.py` |
| Main Ops Tests | `test/test_ops.py` |
| Common Utils | `torch/testing/_internal/common_utils.py` |

---

**Document Status**: Initial draft based on PyTorch 2.x codebase analysis
**Next Steps**: Prioritize high-priority items for next sprint
**Owner**: To be assigned
