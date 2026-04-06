
#[test]
fn test_softmax_simd_large_input() {
    let mut x: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();

    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}
