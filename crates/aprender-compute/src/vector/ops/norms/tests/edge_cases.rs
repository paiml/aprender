use super::*;

fn edge_cases() -> Vec<(NormMethod, &'static str, Vec<f32>, f32)> {
    let l2_large: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let l2_large_exp = l2_large.iter().map(|x| x * x).sum::<f32>().sqrt();

    vec![
        (norm_l2, "l2-pythagorean", vec![3.0, 4.0], 5.0),
        (norm_l2, "l2-empty", vec![], 0.0),
        (norm_l2, "l2-unit", vec![1.0, 0.0, 0.0], 1.0),
        (norm_l2, "l2-single", vec![7.0], 7.0),
        (norm_l2, "l2-neg", vec![-5.0], 5.0),
        (norm_l2, "l2-zeros", vec![0.0, 0.0, 0.0, 0.0], 0.0),
        (norm_l2, "l2-mixed", vec![3.0, -4.0, 0.0], 5.0),
        (
            norm_l2,
            "l2-5elem",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f32).sqrt(),
        ),
        (norm_l2, "l2-large", l2_large, l2_large_exp),
        (norm_l1, "l1-basic", vec![3.0, -4.0, 5.0], 12.0),
        (norm_l1, "l1-empty", vec![], 0.0),
        (norm_l1, "l1-single-neg", vec![-7.0], 7.0),
        (norm_l1, "l1-zeros", vec![0.0, 0.0, 0.0], 0.0),
        (norm_l1, "l1-positive", vec![1.0, 2.0, 3.0, 4.0], 10.0),
        (norm_l1, "l1-all-neg", vec![-1.0, -2.0, -3.0], 6.0),
        (norm_l1, "l1-non-aligned", vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0], 28.0),
        (
            norm_l1,
            "l1-large",
            (0..512).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
            512.0,
        ),
        (norm_linf, "linf-basic", vec![3.0, -7.0, 5.0, -2.0], 7.0),
        (norm_linf, "linf-empty", vec![], 0.0),
        (norm_linf, "linf-all-neg", vec![-1.0, -5.0, -3.0], 5.0),
        (norm_linf, "linf-single", vec![-42.0], 42.0),
        (norm_linf, "linf-zeros", vec![0.0, 0.0, 0.0], 0.0),
        (norm_linf, "linf-end", vec![1.0, 2.0, 3.0, 100.0], 100.0),
        (norm_linf, "linf-begin", vec![-100.0, 2.0, 3.0, 4.0], 100.0),
        (norm_linf, "linf-equal", vec![5.0, 5.0, 5.0, 5.0], 5.0),
        (norm_linf, "linf-non-aligned", vec![1.0, -9.0, 3.0, -4.0, 5.0], 9.0),
        (
            norm_linf,
            "linf-large",
            {
                let mut d: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
                d[200] = -99.9;
                d
            },
            99.9,
        ),
    ]
}

#[test]
fn test_all_norm_edge_cases() {
    for (method, label, data, expected) in edge_cases() {
        let result = method(&Vector::from_slice(&data)).unwrap();
        let tol = if data.len() > 256 { 1e-2 } else { 1e-5 };
        assert!((result - expected).abs() <= tol, "{label}: expected {expected}, got {result}");
    }
}

#[test]
fn test_norm_l2_very_small_values() {
    let v = Vector::from_slice(&[1e-20, 1e-20, 1e-20]);
    let norm = v.norm_l2().unwrap();
    assert!(norm > 0.0 && norm < 1e-10);
}

#[test]
fn test_norm_ordering_property() {
    for data in [
        vec![3.0, -4.0, 5.0, -2.0, 1.0],
        (0..100).map(|i| ((i as f32) * 0.37).sin()).collect::<Vec<_>>(),
    ] {
        let v = Vector::from_slice(&data);
        let l1 = v.norm_l1().unwrap();
        let l2 = v.norm_l2().unwrap();
        let linf = v.norm_linf().unwrap();
        assert!(linf <= l2 + 1e-4, "L-inf ({linf}) should be <= L2 ({l2})");
        assert!(l2 <= l1 + 1e-4, "L2 ({l2}) should be <= L1 ({l1})");
    }
}
