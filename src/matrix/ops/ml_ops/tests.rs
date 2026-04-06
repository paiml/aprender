use super::*;

#[test]
fn test_convolve2d_basic() {
    let input = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();
    let kernel = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
    let result = input.convolve2d(&kernel).unwrap();
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.get(0, 0), Some(&4.0));
}

#[test]
fn test_embedding_lookup() {
    let embeddings =
        Matrix::from_vec(4, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
    let result = embeddings.embedding_lookup(&[1, 3]).unwrap();
    assert_eq!(result.rows(), 2);
    assert_eq!(result.get(0, 0), Some(&4.0));
    assert_eq!(result.get(1, 0), Some(&10.0));
}

#[test]
fn test_embedding_lookup_out_of_bounds() {
    let embeddings = Matrix::from_vec(4, 3, vec![0.0; 12]).unwrap();
    assert!(embeddings.embedding_lookup(&[5]).is_err());
}

#[test]
fn test_max_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .unwrap();
    let pooled = input.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(pooled.shape(), (2, 2));
    assert_eq!(pooled.get(0, 0), Some(&6.0));
    assert_eq!(pooled.get(1, 1), Some(&16.0));
}

#[test]
fn test_avg_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .unwrap();
    let pooled = input.avg_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(pooled.shape(), (2, 2));
    assert!((pooled.get(0, 0).unwrap() - 3.5).abs() < 1e-5);
}

#[test]
fn test_topk() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0]).unwrap();
    let (values, indices) = m.topk(2).unwrap();
    assert_eq!(values, vec![6.0, 5.0]);
    assert_eq!(indices, vec![4, 1]);
}

#[test]
fn test_gather_rows() {
    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let gathered = m.gather(&[2, 0], 0).unwrap();
    assert_eq!(gathered.shape(), (2, 2));
    assert_eq!(gathered.get(0, 0), Some(&5.0));
    assert_eq!(gathered.get(1, 0), Some(&1.0));
}

#[test]
fn test_pad() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let padded = m.pad(((1, 1), (1, 1)), 0.0).unwrap();
    assert_eq!(padded.shape(), (4, 4));
    assert_eq!(padded.get(0, 0), Some(&0.0));
    assert_eq!(padded.get(1, 1), Some(&1.0));
}
