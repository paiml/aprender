use super::*;

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_norms_sse2_backend() {
    for (method, name, data, expected) in norm_specs() {
        assert_norm_with_backend(method, data, expected, 1e-5, Backend::SSE2);
        let _ = name;
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_norms_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    for (method, _name, data, expected) in norm_specs() {
        assert_norm_with_backend(method, data, expected, 1e-3, Backend::AVX2);
    }
}

#[test]
fn test_all_norms_fallback_backends() {
    for (method, _name, data, expected) in norm_specs() {
        for &b in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto, Backend::Scalar]
        {
            assert_norm_with_backend(method, data, expected, 1e-5, b);
        }
    }
}

#[test]
fn test_all_norms_backend_equivalence() {
    let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.13).sin()).collect();
    for (method, _name, _, _) in norm_specs() {
        assert_norm_backend_equivalence(method, &data, 1e-3);
    }
}

#[test]
fn test_all_norms_non_aligned_sizes() {
    let norms: [(NormMethod, fn(usize) -> Vec<f32>, fn(&[f32]) -> f32); 3] = [
        (
            norm_l2,
            |sz| (0..sz).map(|i| (i as f32 + 1.0) * 0.1).collect(),
            |d| d.iter().map(|x| x * x).sum::<f32>().sqrt(),
        ),
        (
            norm_l1,
            |sz| (0..sz).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
            |d| d.len() as f32,
        ),
        (norm_linf, |sz| (0..sz).map(|i| i as f32 + 1.0).collect(), |d| d.len() as f32),
    ];
    for (method, make_data, make_expected) in norms {
        assert_norm_non_aligned(method, make_data, make_expected, 1e-3);
    }
}
