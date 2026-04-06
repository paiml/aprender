use super::super::*;

// ===== Embedding Lookup Tests (Issue #61) =====

#[test]
fn test_embedding_lookup_basic() {
    // Create embedding table: 4 words, 3-dimensional embeddings
    let embeddings = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, // word 0
            4.0, 5.0, 6.0, // word 1
            7.0, 8.0, 9.0, // word 2
            10.0, 11.0, 12.0, // word 3
        ],
    )
    .unwrap();

    // Lookup embeddings for indices [1, 3, 0]
    let result = embeddings.embedding_lookup(&[1, 3, 0]).unwrap();

    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 3);

    // Check word 1 embedding
    assert_eq!(result.get(0, 0), Some(&4.0));
    assert_eq!(result.get(0, 1), Some(&5.0));
    assert_eq!(result.get(0, 2), Some(&6.0));

    // Check word 3 embedding
    assert_eq!(result.get(1, 0), Some(&10.0));
    assert_eq!(result.get(1, 1), Some(&11.0));
    assert_eq!(result.get(1, 2), Some(&12.0));

    // Check word 0 embedding
    assert_eq!(result.get(2, 0), Some(&1.0));
    assert_eq!(result.get(2, 1), Some(&2.0));
    assert_eq!(result.get(2, 2), Some(&3.0));
}

#[test]
fn test_embedding_lookup_single_index() {
    let embeddings = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = embeddings.embedding_lookup(&[1]).unwrap();

    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.get(0, 0), Some(&3.0));
    assert_eq!(result.get(0, 1), Some(&4.0));
}

#[test]
fn test_embedding_lookup_repeated_indices() {
    let embeddings = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Same index can appear multiple times
    let result = embeddings.embedding_lookup(&[0, 0, 1, 0]).unwrap();

    assert_eq!(result.rows(), 4);
    assert_eq!(result.cols(), 3);

    // All index-0 rows should be identical
    assert_eq!(result.get(0, 0), result.get(1, 0));
    assert_eq!(result.get(0, 0), result.get(3, 0));
}

#[test]
fn test_embedding_lookup_empty_indices() {
    let embeddings = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = embeddings.embedding_lookup(&[]).unwrap();

    assert_eq!(result.rows(), 0);
    assert_eq!(result.cols(), 2);
}

#[test]
fn test_embedding_lookup_out_of_bounds() {
    let embeddings = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Index 5 is out of bounds for 3-row table
    let result = embeddings.embedding_lookup(&[0, 5, 1]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn test_embedding_lookup_sparse() {
    let embeddings = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    // Lookup with repeated indices
    let (result, unique) = embeddings.embedding_lookup_sparse(&[1, 3, 1, 0, 3]).unwrap();

    assert_eq!(result.rows(), 5);
    assert_eq!(result.cols(), 2);

    // Unique indices should be sorted and deduplicated
    assert_eq!(unique, vec![0, 1, 3]);
}

#[test]
fn test_embedding_lookup_large_embeddings() {
    // Test with realistic NLP dimensions
    let vocab_size = 1000;
    let embed_dim = 256;
    let data: Vec<f32> = (0..vocab_size * embed_dim).map(|i| i as f32).collect();
    let embeddings = Matrix::from_vec(vocab_size, embed_dim, data).unwrap();

    // Lookup a sequence
    let indices: Vec<usize> = vec![0, 500, 999, 42, 100];
    let result = embeddings.embedding_lookup(&indices).unwrap();

    assert_eq!(result.rows(), 5);
    assert_eq!(result.cols(), embed_dim);

    // Verify first element of each row
    assert_eq!(result.get(0, 0), Some(&0.0)); // word 0
    assert_eq!(result.get(1, 0), Some(&(500.0 * 256.0))); // word 500
    assert_eq!(result.get(2, 0), Some(&(999.0 * 256.0))); // word 999
}

// ===== Batched Matrix Multiplication Tests =====

#[test]
fn test_batched_matmul_basic() {
    // [batch=2, m=2, k=3] @ [batch=2, k=3, n=2] -> [batch=2, m=2, n=2]
    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;

    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[184,202],[265,292]]
    let a_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1
    ];
    let b_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1
    ];

    let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n).unwrap();

    assert_eq!(result.len(), batch * m * n);

    // Verify batch 0
    assert!((result[0] - 22.0).abs() < 1e-5);
    assert!((result[1] - 28.0).abs() < 1e-5);
    assert!((result[2] - 49.0).abs() < 1e-5);
    assert!((result[3] - 64.0).abs() < 1e-5);

    // Verify batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]]
    // C[0,0] = 7*7 + 8*9 + 9*11 = 49 + 72 + 99 = 220
    // C[0,1] = 7*8 + 8*10 + 9*12 = 56 + 80 + 108 = 244
    // C[1,0] = 10*7 + 11*9 + 12*11 = 70 + 99 + 132 = 301
    // C[1,1] = 10*8 + 11*10 + 12*12 = 80 + 110 + 144 = 334
    assert!((result[4] - 220.0).abs() < 1e-5);
    assert!((result[5] - 244.0).abs() < 1e-5);
    assert!((result[6] - 301.0).abs() < 1e-5);
    assert!((result[7] - 334.0).abs() < 1e-5);
}

#[test]
fn test_batched_matmul_single_batch() {
    let batch = 1;
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity
    let b_data = vec![5.0, 6.0, 7.0, 8.0];

    let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n).unwrap();

    // Identity @ B = B
    assert!((result[0] - 5.0).abs() < 1e-5);
    assert!((result[1] - 6.0).abs() < 1e-5);
    assert!((result[2] - 7.0).abs() < 1e-5);
    assert!((result[3] - 8.0).abs() < 1e-5);
}

#[test]
fn test_batched_matmul_a_size_mismatch() {
    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0; 10]; // Wrong size (should be 12)
    let b_data = vec![1.0; batch * k * n];

    let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("A data size mismatch"));
}

#[test]
fn test_batched_matmul_b_size_mismatch() {
    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0; batch * m * k];
    let b_data = vec![1.0; 10]; // Wrong size (should be 12)

    let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("B data size mismatch"));
}

#[test]
fn test_batched_matmul_4d_basic() {
    // [batch=1, heads=2, m=2, k=2] @ [batch=1, heads=2, k=2, n=2]
    let batch = 1;
    let heads = 2;
    let m = 2;
    let k = 2;
    let n = 2;

    // Head 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    // Head 1: [[5,6],[7,8]] @ [[1,0],[0,1]] = [[5,6],[7,8]]
    let a_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // Head 0
        5.0, 6.0, 7.0, 8.0, // Head 1
    ];
    let b_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, // Head 0 (identity)
        1.0, 0.0, 0.0, 1.0, // Head 1 (identity)
    ];

    let result = Matrix::batched_matmul_4d(&a_data, &b_data, batch, heads, m, k, n).unwrap();

    assert_eq!(result.len(), batch * heads * m * n);

    // Head 0: A @ I = A
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
    assert!((result[2] - 3.0).abs() < 1e-5);
    assert!((result[3] - 4.0).abs() < 1e-5);

    // Head 1: A @ I = A
    assert!((result[4] - 5.0).abs() < 1e-5);
    assert!((result[5] - 6.0).abs() < 1e-5);
    assert!((result[6] - 7.0).abs() < 1e-5);
    assert!((result[7] - 8.0).abs() < 1e-5);
}

#[test]
fn test_batched_matmul_4d_attention_pattern() {
    // Simulate Q @ K^T for attention: [batch=1, heads=2, seq=4, head_dim=8]
    let batch = 1;
    let heads = 2;
    let seq_len = 4;
    let head_dim = 8;

    let q_data: Vec<f32> =
        (0..batch * heads * seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
    let kt_data: Vec<f32> =
        (0..batch * heads * head_dim * seq_len).map(|i| (i as f32) * 0.01).collect();

    let result =
        Matrix::batched_matmul_4d(&q_data, &kt_data, batch, heads, seq_len, head_dim, seq_len)
            .unwrap();

    // Output should be [batch, heads, seq, seq] = 1 * 2 * 4 * 4 = 32 elements
    assert_eq!(result.len(), batch * heads * seq_len * seq_len);
}

#[test]
fn test_batched_matmul_4d_a_size_mismatch() {
    let batch = 1;
    let heads = 2;
    let m = 4;
    let k = 8;
    let n = 4;

    let a_data = vec![1.0; 50]; // Wrong size
    let b_data = vec![1.0; batch * heads * k * n];

    let result = Matrix::batched_matmul_4d(&a_data, &b_data, batch, heads, m, k, n);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("A data size mismatch"));
}

#[test]
fn test_batched_matmul_4d_b_size_mismatch() {
    let batch = 1;
    let heads = 2;
    let m = 4;
    let k = 8;
    let n = 4;

    let a_data = vec![1.0; batch * heads * m * k];
    let b_data = vec![1.0; 50]; // Wrong size

    let result = Matrix::batched_matmul_4d(&a_data, &b_data, batch, heads, m, k, n);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("B data size mismatch"));
}
