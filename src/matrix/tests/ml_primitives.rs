use super::*;

// ===== ML Primitives Tests =====

#[test]
fn test_max_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .expect("valid input");

    // 2x2 kernel, 2x2 stride
    let pooled = input.max_pool2d((2, 2), (2, 2)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    assert_eq!(pooled.get(0, 0), Some(&6.0)); // max of [1,2,5,6]
    assert_eq!(pooled.get(0, 1), Some(&8.0)); // max of [3,4,7,8]
    assert_eq!(pooled.get(1, 0), Some(&14.0)); // max of [9,10,13,14]
    assert_eq!(pooled.get(1, 1), Some(&16.0)); // max of [11,12,15,16]
}

#[test]
fn test_max_pool2d_stride_1() {
    let input = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .expect("valid input");

    // 2x2 kernel, 1x1 stride (overlapping)
    let pooled = input.max_pool2d((2, 2), (1, 1)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    assert_eq!(pooled.get(0, 0), Some(&5.0)); // max of [1,2,4,5]
    assert_eq!(pooled.get(0, 1), Some(&6.0)); // max of [2,3,5,6]
}

#[test]
fn test_avg_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .expect("valid input");

    let pooled = input.avg_pool2d((2, 2), (2, 2)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    // avg of [1,2,5,6] = 14/4 = 3.5
    assert!((pooled.get(0, 0).unwrap() - 3.5).abs() < 1e-5);
    // avg of [3,4,7,8] = 22/4 = 5.5
    assert!((pooled.get(0, 1).unwrap() - 5.5).abs() < 1e-5);
}

#[test]
fn test_topk() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0]).expect("valid input");
    let (values, indices) = m.topk(3).expect("valid topk");
    assert_eq!(values, vec![6.0, 5.0, 4.0]);
    assert_eq!(indices, vec![4, 1, 5]);
}

#[test]
fn test_topk_empty() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
    let (values, indices) = m.topk(0).expect("valid topk");
    assert!(values.is_empty());
    assert!(indices.is_empty());
}

#[test]
fn test_gather_rows() {
    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid input");
    let gathered = m.gather(&[2, 0], 0).expect("valid gather");
    assert_eq!(gathered.shape(), (2, 2));
    assert_eq!(gathered.get(0, 0), Some(&5.0)); // Row 2, col 0
    assert_eq!(gathered.get(0, 1), Some(&6.0)); // Row 2, col 1
    assert_eq!(gathered.get(1, 0), Some(&1.0)); // Row 0, col 0
}

#[test]
fn test_gather_cols() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid input");
    let gathered = m.gather(&[2, 0], 1).expect("valid gather");
    assert_eq!(gathered.shape(), (2, 2));
    assert_eq!(gathered.get(0, 0), Some(&3.0)); // Row 0, col 2
    assert_eq!(gathered.get(0, 1), Some(&1.0)); // Row 0, col 0
}

#[test]
fn test_pad() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
    let padded = m.pad(((1, 1), (1, 1)), 0.0).expect("valid pad");
    assert_eq!(padded.shape(), (4, 4));
    assert_eq!(padded.get(0, 0), Some(&0.0)); // top-left padding
    assert_eq!(padded.get(1, 1), Some(&1.0)); // original (0,0)
    assert_eq!(padded.get(2, 2), Some(&4.0)); // original (1,1)
    assert_eq!(padded.get(3, 3), Some(&0.0)); // bottom-right padding
}

#[test]
fn test_pad_asymmetric() {
    let m = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("valid input");
    let padded = m.pad(((0, 1), (2, 0)), -1.0).expect("valid pad");
    assert_eq!(padded.shape(), (2, 4));
    assert_eq!(padded.get(0, 0), Some(&-1.0)); // left padding
    assert_eq!(padded.get(0, 2), Some(&1.0)); // original
    assert_eq!(padded.get(1, 0), Some(&-1.0)); // bottom padding
}
