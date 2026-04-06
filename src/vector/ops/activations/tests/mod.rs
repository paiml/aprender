use crate::vector::Vector;
use crate::{Backend, TruenoError};

mod backend;
mod individual;
mod relu_dispatch;
mod softmax;

type ActFn = fn(&Vector<f32>) -> Result<Vector<f32>, TruenoError>;

/// Assert that an activation produces expected element-wise results on a given backend.
fn assert_activation_elementwise(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    expected_fn: fn(f32) -> f32,
    tolerance: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let exp = expected_fn(data[i]);
        assert!(
            (val - exp).abs() < tolerance,
            "{label} {backend:?} mismatch at index {i}: got {val} expected {exp}",
        );
    }
}

/// Assert that an activation on a given backend produces values within a range.
fn assert_activation_in_range(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    lo: f32,
    hi: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    for &val in result.as_slice() {
        assert!(val >= lo && val <= hi, "{label} {backend:?} out of range [{lo}, {hi}]: {val}");
    }
}

/// Assert that an activation at a specific index equals an expected value.
fn assert_activation_at(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    index: usize,
    expected: f32,
    tolerance: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    let got = result.as_slice()[index];
    assert!(
        (got - expected).abs() < tolerance,
        "{label} {backend:?} at index {index}: got {got} expected {expected}",
    );
}

/// Assert backend equivalence: compare Scalar result with SSE2 and AVX2.
#[cfg(target_arch = "x86_64")]
fn assert_backend_equivalence(data: &[f32], activation_fn: ActFn, tolerance: f32, label: &str) {
    let scalar = activation_fn(&Vector::from_slice_with_backend(data, Backend::Scalar)).unwrap();
    for &backend in &[Backend::SSE2] {
        let other = activation_fn(&Vector::from_slice_with_backend(data, backend)).unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(other.as_slice().iter()).enumerate() {
            assert!(
                (s - x).abs() < tolerance,
                "Scalar vs {backend:?} {label} mismatch at {i}: {s} vs {x}"
            );
        }
    }
    if is_x86_feature_detected!("avx2") {
        let avx2 = activation_fn(&Vector::from_slice_with_backend(data, Backend::AVX2)).unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
            assert!(
                (s - x).abs() < tolerance,
                "Scalar vs AVX2 {label} mismatch at {i}: {s} vs {x}"
            );
        }
    }
}

// Activation adapters: wrap method calls as fn pointers for helpers.
fn act_relu(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> {
    v.relu()
}
fn act_sigmoid(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> {
    v.sigmoid()
}
fn act_gelu(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> {
    v.gelu()
}
fn act_swish(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> {
    v.swish()
}

/// Activation spec for parametric tests: (adapter, label, zero_output).
fn activation_specs() -> [(ActFn, &'static str, f32); 4] {
    [
        (act_relu, "relu", 0.0),
        (act_sigmoid, "sigmoid", 0.5),
        (act_gelu, "gelu", 0.0),
        (act_swish, "swish", 0.0),
    ]
}
