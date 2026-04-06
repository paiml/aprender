use super::*;

/// Helper: create iota data (1..=n) and verify tiled sum matches expected.
fn assert_iota_sum(width: usize, height: usize) {
    let n = width * height;
    let data: Vec<f32> = (1..=n as i32).map(|x| x as f32).collect();
    let sum = tiled_sum_2d(&data, width, height);
    let expected: f32 = (1..=n as i32).sum::<i32>() as f32;
    assert!((sum - expected).abs() < 1e-2, "sum={sum}, expected={expected}, dims={width}x{height}");
}

#[test]
fn test_tiled_sum_small() {
    assert_iota_sum(4, 4);
}

#[test]
fn test_tiled_sum_exact_tile() {
    assert_iota_sum(16, 16);
}

#[test]
fn test_tiled_sum_multiple_tiles() {
    assert_iota_sum(32, 32);
}

#[test]
fn test_tiled_sum_non_aligned() {
    assert_iota_sum(20, 20);
}

#[test]
fn test_tiled_max() {
    let data: Vec<f32> = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 8.0, 4.0, 6.0];
    assert!((tiled_max_2d(&data, 3, 3) - 9.0).abs() < 1e-5);
}

#[test]
fn test_tiled_max_large() {
    let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
    assert!((tiled_max_2d(&data, 16, 16) - 256.0).abs() < 1e-5);
}

#[test]
fn test_tiled_min() {
    let data: Vec<f32> = vec![5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0];
    assert!((tiled_min_2d(&data, 3, 3) - 1.0).abs() < 1e-5);
}

#[test]
fn test_tiled_min_negative() {
    let data: Vec<f32> = vec![-5.0, 3.0, -7.0, 1.0, -9.0, 2.0, 8.0, -4.0, 6.0];
    assert!((tiled_min_2d(&data, 3, 3) - (-9.0)).abs() < 1e-5);
}

#[test]
fn test_empty_data() {
    let data: Vec<f32> = vec![];
    assert!((tiled_sum_2d(&data, 0, 0) - 0.0).abs() < 1e-10);
    assert!(tiled_max_2d(&data, 0, 0) == f32::NEG_INFINITY);
    assert!(tiled_min_2d(&data, 0, 0) == f32::INFINITY);
}

#[test]
fn test_partial_results() {
    let data: Vec<f32> = vec![1.0; 32 * 32];
    let partial = tiled_reduce_partial::<SumOp>(&data, 32, 32);
    assert_eq!(partial.len(), 4);
    for &p in &partial {
        assert!((p - 256.0).abs() < 1e-5);
    }
}

#[test]
fn test_partial_results_non_aligned() {
    let data: Vec<f32> = vec![1.0; 20 * 20];
    let partial = tiled_reduce_partial::<SumOp>(&data, 20, 20);
    assert_eq!(partial.len(), 4);
    let total: f32 = partial.iter().sum();
    assert!((total - 400.0).abs() < 1e-5);
}

#[test]
fn test_single_element() {
    let data = vec![42.0];
    assert!((tiled_sum_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
    assert!((tiled_max_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
    assert!((tiled_min_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
}

#[test]
fn test_equivalence_with_simple_sum() {
    let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
    let tiled = tiled_sum_2d(&data, 50, 20);
    let simple: f32 = data.iter().sum();
    let rel_err = (tiled - simple).abs() / simple;
    assert!(rel_err < 1e-5, "rel_err={rel_err}");
}

#[test]
fn test_tile_boundaries() {
    assert_iota_sum(17, 17);
}

#[test]
fn test_wide_matrix() {
    assert_iota_sum(100, 5);
}

#[test]
fn test_tall_matrix() {
    assert_iota_sum(5, 100);
}
