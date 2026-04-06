use crate::{Backend, Result, Vector};

mod backend;
mod edge_cases;

type NormMethod = fn(&Vector<f32>) -> Result<f32>;

fn norm_l1(v: &Vector<f32>) -> Result<f32> {
    v.norm_l1()
}
fn norm_l2(v: &Vector<f32>) -> Result<f32> {
    v.norm_l2()
}
fn norm_linf(v: &Vector<f32>) -> Result<f32> {
    v.norm_linf()
}

fn assert_norm_with_backend(
    norm_fn: NormMethod,
    data: &[f32],
    expected: f32,
    tol: f32,
    backend: Backend,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = norm_fn(&v).unwrap();
    assert!((result - expected).abs() <= tol, "expected {expected} got {result} ({backend:?})");
}

fn assert_norm_backend_equivalence(norm_fn: NormMethod, data: &[f32], tol: f32) {
    let scalar = norm_fn(&Vector::from_slice_with_backend(data, Backend::Scalar)).unwrap();
    for &backend in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
        let val = norm_fn(&Vector::from_slice_with_backend(data, backend)).unwrap();
        assert!((scalar - val).abs() < tol, "Scalar vs {backend:?}: {scalar} vs {val}");
    }
    #[cfg(target_arch = "x86_64")]
    {
        let sse2 = norm_fn(&Vector::from_slice_with_backend(data, Backend::SSE2)).unwrap();
        assert!((scalar - sse2).abs() < tol, "Scalar vs SSE2: {scalar} vs {sse2}");
        if is_x86_feature_detected!("avx2") {
            let avx2 = norm_fn(&Vector::from_slice_with_backend(data, Backend::AVX2)).unwrap();
            assert!((scalar - avx2).abs() < tol, "Scalar vs AVX2: {scalar} vs {avx2}");
        }
    }
}

fn assert_norm_non_aligned(
    norm_fn: NormMethod,
    make_data: fn(usize) -> Vec<f32>,
    make_expected: fn(&[f32]) -> f32,
    tol: f32,
) {
    for size in [1, 2, 3, 5, 7, 9, 13, 15, 17, 31, 33] {
        let data = make_data(size);
        let result = norm_fn(&Vector::from_slice(&data)).unwrap();
        let expected = make_expected(&data);
        assert!((result - expected).abs() < tol, "size {size}: {result} vs {expected}");
    }
}

fn norm_specs() -> [(NormMethod, &'static str, &'static [f32], f32); 3] {
    [
        (norm_l1, "l1", &[3.0, -4.0, 5.0], 12.0),
        (norm_l2, "l2", &[3.0, 4.0, 0.0, 0.0], 5.0),
        (norm_linf, "linf", &[3.0, -7.0, 5.0, -2.0], 7.0),
    ]
}
