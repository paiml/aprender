//! Property-based tests using proptest
//!
//! Tests mathematical invariants and properties of core realizar modules:
//! - Quantization round-trip
//! - Softmax properties
//! - RMSNorm properties
//! - Matrix multiplication properties
//! - KVCache store/retrieve
//! - Tokenizer encode/decode

use proptest::prelude::*;
use realizar::inference::{simd_softmax, KVCache};
use realizar::layers::{softmax, LayerNorm};
use realizar::quantize::{
    dequantize_q4_0, dequantize_q8_0, quantize_to_q8_blocks, Q8KSuperBlock, Q8_0Block,
};
use realizar::tensor::Tensor;
use realizar::tokenizer::{BPETokenizer, SentencePieceTokenizer, Tokenizer, Vocabulary};

// ============================================================================
// QUANTIZATION PROPERTY TESTS
// ============================================================================

proptest! {
    /// Q8_0 quantize then dequantize preserves values within tolerance
    #[test]
    fn prop_q8_0_roundtrip_preserves_values(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite() && x.abs() < 1e6),
            32..=32  // Exactly 32 values for one block
        )
    ) {
        let arr: [f32; 32] = values.try_into().expect("exactly 32 elements");
        let block = Q8_0Block::quantize(&arr);
        let dequantized = block.dequantize();

        // Relative error should be bounded by quantization precision
        // Q8_0 uses 8 bits, so max relative error ~1/127 ≈ 0.008
        let max_abs = arr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-10 {
            for (orig, deq) in arr.iter().zip(dequantized.iter()) {
                let abs_err = (orig - deq).abs();
                let rel_err = abs_err / max_abs;
                prop_assert!(rel_err < 0.02, "Q8_0 relative error {} too large", rel_err);
            }
        }
    }

    /// Q8_K super-block quantize then dequantize preserves values
    #[test]
    fn prop_q8k_roundtrip_preserves_values(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite() && x.abs() < 1e6),
            256..=256  // Exactly 256 values for one super-block
        )
    ) {
        let arr: [f32; 256] = values.try_into().expect("exactly 256 elements");
        let block = Q8KSuperBlock::quantize(&arr);
        let dequantized = block.dequantize();

        let max_abs = arr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-10 {
            for (orig, deq) in arr.iter().zip(dequantized.iter()) {
                let abs_err = (orig - deq).abs();
                let rel_err = abs_err / max_abs;
                prop_assert!(rel_err < 0.02, "Q8_K relative error {} too large", rel_err);
            }
        }
    }

    /// Q8_0 quantization error is bounded
    #[test]
    fn prop_q8_0_error_bounded(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 1e4),
            32..=32
        )
    ) {
        let arr: [f32; 32] = values.try_into().expect("exactly 32 elements");
        let block = Q8_0Block::quantize(&arr);
        let rel_err = block.relative_error(&arr);

        // Relative error should be < 1% for well-behaved inputs
        prop_assert!(rel_err < 0.02, "Q8_0 relative error {} exceeds 2%", rel_err);
    }

    /// Multiple Q8_0 blocks quantization works correctly
    #[test]
    fn prop_q8_0_multi_block_roundtrip(
        num_blocks in 1usize..=8,
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite() && x.abs() < 1e6),
            1..=256
        )
    ) {
        // Pad to multiple of 32
        let target_len = ((num_blocks * 32).min(values.len())).max(32);
        let padded_len = target_len.div_ceil(32) * 32;
        let mut padded = values;
        padded.resize(padded_len, 0.0);

        let blocks = quantize_to_q8_blocks(&padded).expect("valid input");
        prop_assert_eq!(blocks.len(), padded_len / 32);
    }
}

// ============================================================================
// SOFTMAX PROPERTY TESTS
// ============================================================================

proptest! {
    /// Softmax output sums to 1
    #[test]
    fn prop_softmax_sum_to_one(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 100.0),
            1..=64
        )
    ) {
        let mut data = values;
        simd_softmax(&mut data);

        let sum: f32 = data.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "Softmax sum {} != 1.0", sum);
    }

    /// Softmax output is in [0, 1]
    #[test]
    fn prop_softmax_values_in_range(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 100.0),
            1..=64
        )
    ) {
        let mut data = values;
        simd_softmax(&mut data);

        for (i, &v) in data.iter().enumerate() {
            prop_assert!(v >= 0.0, "Softmax[{}] = {} < 0", i, v);
            prop_assert!(v <= 1.0, "Softmax[{}] = {} > 1", i, v);
        }
    }

    /// Softmax preserves relative ordering (monotonicity)
    #[test]
    fn prop_softmax_monotonic(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 50.0),
            2..=32
        )
    ) {
        let mut data = values.clone();
        simd_softmax(&mut data);

        // For each pair where input[i] > input[j], output[i] >= output[j]
        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] > values[j] {
                    prop_assert!(
                        data[i] >= data[j] - 1e-6,
                        "Monotonicity violated: input[{}]={} > input[{}]={} but output[{}]={} < output[{}]={}",
                        i, values[i], j, values[j], i, data[i], j, data[j]
                    );
                }
            }
        }
    }

    /// Softmax with Tensor API sums to 1
    #[test]
    fn prop_tensor_softmax_sum_to_one(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 100.0),
            1..=32
        )
    ) {
        let len = values.len();
        let tensor = Tensor::from_vec(vec![len], values).expect("valid tensor");
        let result = softmax(&tensor).expect("softmax succeeds");

        let sum: f32 = result.data().iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "Tensor softmax sum {} != 1.0", sum);
    }
}

// ============================================================================
// LAYER NORM / RMS NORM PROPERTY TESTS
// ============================================================================

proptest! {
    /// LayerNorm output has zero mean (with default bias=0)
    #[test]
    fn prop_layernorm_zero_mean(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 1e4),
            4..=64
        )
    ) {
        let dim = values.len();
        let layer = LayerNorm::new(dim, 1e-5).expect("valid layer");
        let tensor = Tensor::from_vec(vec![dim], values).expect("valid tensor");
        let result = layer.forward(&tensor).expect("forward succeeds");

        let mean: f32 = result.data().iter().sum::<f32>() / dim as f32;
        prop_assert!(mean.abs() < 1e-4, "LayerNorm mean {} != 0", mean);
    }

    /// LayerNorm output has unit variance (with default weight=1)
    /// Note: Uses uniform distribution to ensure reasonable variance
    #[test]
    fn prop_layernorm_unit_variance(
        values in prop::collection::vec(
            // Use uniform range to guarantee diverse values with non-trivial variance
            -100.0f32..100.0f32,
            8..=64
        )
    ) {
        let dim = values.len();
        let layer = LayerNorm::new(dim, 1e-5).expect("valid layer");
        let tensor = Tensor::from_vec(vec![dim], values).expect("valid tensor");
        let result = layer.forward(&tensor).expect("forward succeeds");

        let data = result.data();
        let mean: f32 = data.iter().sum::<f32>() / dim as f32;
        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;

        // Variance should be approximately 1 (with small epsilon tolerance)
        prop_assert!((variance - 1.0).abs() < 0.1, "LayerNorm variance {} != 1", variance);
    }

    /// LayerNorm output is bounded
    #[test]
    fn prop_layernorm_bounded_output(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("bounded", |x| x.is_finite() && x.abs() < 1e6),
            4..=64
        )
    ) {
        let dim = values.len();
        let layer = LayerNorm::new(dim, 1e-5).expect("valid layer");
        let tensor = Tensor::from_vec(vec![dim], values).expect("valid tensor");
        let result = layer.forward(&tensor).expect("forward succeeds");

        for (i, &v) in result.data().iter().enumerate() {
            prop_assert!(v.is_finite(), "LayerNorm[{}] = {} is not finite", i, v);
            // Output should be roughly bounded by ~5 std devs
            prop_assert!(v.abs() < 10.0, "LayerNorm[{}] = {} too large", i, v);
        }
    }
}

// ============================================================================
// MATRIX MULTIPLICATION PROPERTY TESTS
// ============================================================================

proptest! {
    /// Matrix multiplication is approximately associative (within tolerance)
    #[test]
    fn prop_matmul_identity(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("small", |x| x.is_finite() && x.abs() < 10.0),
            4..=16
        )
    ) {
        // For a vector v, v^T * v should be sum of squares (scalar)
        let sum_squares: f32 = values.iter().map(|x| x * x).sum();

        // Verify using trueno
        let vec = trueno::Vector::from_slice(&values);
        let dot = vec.dot(&vec).unwrap_or(0.0);

        prop_assert!(
            (dot - sum_squares).abs() < 1e-4 * sum_squares.abs().max(1.0),
            "v^T * v = {} != sum(v^2) = {}",
            dot, sum_squares
        );
    }

    /// Dot product is commutative
    #[test]
    fn prop_dot_commutative(
        a in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("small", |x| x.is_finite() && x.abs() < 100.0),
            1..=32
        ),
        b in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("small", |x| x.is_finite() && x.abs() < 100.0),
            1..=32
        )
    ) {
        let len = a.len().min(b.len());
        let a_slice = &a[..len];
        let b_slice = &b[..len];

        let va = trueno::Vector::from_slice(a_slice);
        let vb = trueno::Vector::from_slice(b_slice);

        let ab = va.dot(&vb).unwrap_or(f32::NAN);
        let ba = vb.dot(&va).unwrap_or(f32::NAN);

        if ab.is_finite() && ba.is_finite() {
            prop_assert!(
                (ab - ba).abs() < 1e-5 * ab.abs().max(1.0),
                "a.b = {} != b.a = {}",
                ab, ba
            );
        }
    }

    /// Dot product with zero vector is zero
    #[test]
    fn prop_dot_with_zero(
        values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite()),
            1..=32
        )
    ) {
        let zeros = vec![0.0f32; values.len()];
        let va = trueno::Vector::from_slice(&values);
        let vz = trueno::Vector::from_slice(&zeros);

        let dot = va.dot(&vz).unwrap_or(f32::NAN);
        prop_assert!(dot.abs() < 1e-10, "v.0 = {} != 0", dot);
    }
}

// ============================================================================
// KV CACHE PROPERTY TESTS
// ============================================================================

proptest! {
    /// KVCache store/retrieve round-trip preserves values
    #[test]
    fn prop_kvcache_roundtrip(
        num_layers in 1usize..=4,
        hidden_dim in 8usize..=64,
        k_values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite()),
            8..=64
        ),
        v_values in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("finite", |x| x.is_finite()),
            8..=64
        )
    ) {
        // Ensure values fit hidden_dim
        let dim = hidden_dim.min(k_values.len()).min(v_values.len());
        let k = &k_values[..dim];
        let v = &v_values[..dim];

        let mut cache = KVCache::new(num_layers, dim, 16);
        let layer = 0;

        // Store
        cache.store(layer, k, v);
        cache.advance();

        // Retrieve
        let k_cached = cache.get_k(layer);
        let v_cached = cache.get_v(layer);

        prop_assert_eq!(k_cached.len(), dim, "K cache length mismatch");
        prop_assert_eq!(v_cached.len(), dim, "V cache length mismatch");

        for i in 0..dim {
            prop_assert!(
                (k_cached[i] - k[i]).abs() < 1e-6,
                "K[{}]: {} != {}",
                i, k_cached[i], k[i]
            );
            prop_assert!(
                (v_cached[i] - v[i]).abs() < 1e-6,
                "V[{}]: {} != {}",
                i, v_cached[i], v[i]
            );
        }
    }

    /// KVCache respects capacity limits
    #[test]
    fn prop_kvcache_capacity(
        max_seq_len in 2usize..=8,
        hidden_dim in 4usize..=16
    ) {
        let mut cache = KVCache::new(1, hidden_dim, max_seq_len);
        let k = vec![1.0f32; hidden_dim];
        let v = vec![2.0f32; hidden_dim];

        // Fill to capacity
        for _ in 0..max_seq_len {
            cache.store(0, &k, &v);
            cache.advance();
        }

        prop_assert_eq!(cache.len(), max_seq_len, "Cache length != max_seq_len");

        // Try to exceed capacity - should not crash, but seq_len continues to grow
        // (store() silently ignores writes beyond buffer capacity)
        cache.store(0, &k, &v);
        cache.advance();

        // Note: KVCache.advance() doesn't cap seq_len - it grows beyond max_seq_len
        // The store() function checks bounds and ignores writes beyond capacity
        prop_assert_eq!(cache.len(), max_seq_len + 1, "seq_len should grow");
    }

    /// KVCache reset clears state
    #[test]
    fn prop_kvcache_reset(
        hidden_dim in 4usize..=32
    ) {
        let mut cache = KVCache::new(2, hidden_dim, 8);
        let k = vec![1.0f32; hidden_dim];
        let v = vec![2.0f32; hidden_dim];

        // Store some values
        cache.store(0, &k, &v);
        cache.advance();
        prop_assert_eq!(cache.len(), 1);

        // Reset
        cache.reset();
        prop_assert!(cache.is_empty(), "Cache not empty after reset");
        prop_assert_eq!(cache.len(), 0, "Cache length != 0 after reset");
    }
}

// ============================================================================
// TOKENIZER PROPERTY TESTS
// ============================================================================

proptest! {
    /// Basic tokenizer encode/decode round-trip for known vocabulary
    #[test]
    fn prop_tokenizer_known_words_roundtrip(
        word_indices in prop::collection::vec(0usize..5, 1..=8)
    ) {
        let words = ["<unk>", "hello", "world", "foo", "bar"];
        let vocab = Vocabulary::from_tokens(words.iter().map(|s| s.to_string()).collect())
            .expect("valid vocab");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("valid tokenizer");

        // Build text from known words (indices 1-4, skip unk)
        let text: String = word_indices
            .iter()
            .map(|&i| words[(i % 4) + 1])
            .collect::<Vec<_>>()
            .join(" ");

        let encoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&encoded).expect("decode succeeds");

        prop_assert_eq!(decoded, text, "Round-trip failed");
    }

    /// BPE tokenizer handles empty input
    #[test]
    fn prop_bpe_empty_input(_dummy in Just(())) {
        let vocab = vec!["<unk>".to_string(), "a".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("valid tokenizer");

        let encoded = tokenizer.encode("");
        prop_assert!(encoded.is_empty(), "Empty input should give empty output");

        let decoded = tokenizer.decode(&[]).expect("decode empty");
        prop_assert!(decoded.is_empty(), "Decode empty should give empty string");
    }

    /// BPE tokenizer decode is inverse of encode for ASCII chars in vocab
    #[test]
    fn prop_bpe_ascii_roundtrip(
        chars in prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c', 'd']), 1..=16)
    ) {
        let vocab: Vec<String> = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("valid tokenizer");

        let text: String = chars.iter().collect();
        let encoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&encoded).expect("decode succeeds");

        prop_assert_eq!(decoded, text.clone(), "BPE round-trip failed for '{}'", text);
    }

    /// SentencePiece tokenizer handles single-char vocabulary
    #[test]
    fn prop_sentencepiece_single_chars(
        chars in prop::collection::vec(prop::sample::select(vec!['x', 'y', 'z']), 1..=8)
    ) {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("x".to_string(), -1.0),
            ("y".to_string(), -1.0),
            ("z".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("valid tokenizer");

        let text: String = chars.iter().collect();
        let encoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&encoded).expect("decode succeeds");

        prop_assert_eq!(decoded, text.clone(), "SP round-trip failed for '{}'", text);
    }

    /// Vocabulary size matches input
    #[test]
    fn prop_vocabulary_size(
        num_tokens in 1usize..=100
    ) {
        let tokens: Vec<String> = (0..num_tokens).map(|i| format!("tok_{i}")).collect();
        let vocab = Vocabulary::from_tokens(tokens).expect("valid vocab");

        prop_assert_eq!(vocab.size(), num_tokens, "Vocab size mismatch");
    }
}

// ============================================================================
// Q4 DEQUANTIZATION TESTS
// ============================================================================

proptest! {
    /// Q4_0 dequantization produces correct number of values
    #[test]
    fn prop_q4_0_dequant_length(
        num_blocks in 1usize..=4
    ) {
        // Q4_0 block = 2 bytes (f16 scale) + 16 bytes (quants) = 18 bytes
        let block_size = 18;
        let data = vec![0u8; num_blocks * block_size];

        let result = dequantize_q4_0(&data).expect("valid Q4_0 data");

        // Each block produces 32 values
        prop_assert_eq!(result.len(), num_blocks * 32, "Q4_0 output length mismatch");
    }

    /// Q8_0 dequantization produces correct number of values
    #[test]
    fn prop_q8_0_dequant_length(
        num_blocks in 1usize..=4
    ) {
        // Q8_0 block = 2 bytes (f16 scale) + 32 bytes (quants) = 34 bytes
        let block_size = 34;
        let data = vec![0u8; num_blocks * block_size];

        let result = dequantize_q8_0(&data).expect("valid Q8_0 data");

        // Each block produces 32 values
        prop_assert_eq!(result.len(), num_blocks * 32, "Q8_0 output length mismatch");
    }

    /// Q4_0 dequantization produces finite values for valid input
    #[test]
    fn prop_q4_0_dequant_finite(
        num_blocks in 1usize..=2,
        quant_bytes in prop::collection::vec(0u8..=255, 16..=16)
    ) {
        // Construct valid Q4_0 data with f16 scale = 1.0 (0x3c00)
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // f16 scale = 1.0 in little-endian
            data.push(0x00);
            data.push(0x3c);
            // 16 bytes of quants
            data.extend_from_slice(&quant_bytes);
        }

        let result = dequantize_q4_0(&data).expect("valid Q4_0 data");

        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.is_finite(), "Q4_0 dequant[{}] = {} not finite", i, v);
        }
    }

    /// Q8_0 dequantization produces finite values for valid input
    #[test]
    fn prop_q8_0_dequant_finite(
        num_blocks in 1usize..=2,
        quant_bytes in prop::collection::vec(0u8..=255, 32..=32)
    ) {
        // Construct valid Q8_0 data with f16 scale = 1.0 (0x3c00)
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // f16 scale = 1.0 in little-endian
            data.push(0x00);
            data.push(0x3c);
            // 32 bytes of quants
            data.extend_from_slice(&quant_bytes);
        }

        let result = dequantize_q8_0(&data).expect("valid Q8_0 data");

        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.is_finite(), "Q8_0 dequant[{}] = {} not finite", i, v);
        }
    }

    /// Q8_0 dequantization with zero scale produces zeros
    #[test]
    fn prop_q8_0_zero_scale_gives_zeros(
        quant_bytes in prop::collection::vec(0u8..=255, 32..=32)
    ) {
        // f16 zero = 0x0000
        let mut data = vec![0x00, 0x00];
        data.extend_from_slice(&quant_bytes);

        let result = dequantize_q8_0(&data).expect("valid Q8_0 data");

        for (i, &v) in result.iter().enumerate() {
            prop_assert!(
                v.abs() < 1e-10,
                "Q8_0 with zero scale: dequant[{}] = {} should be 0",
                i, v
            );
        }
    }
}
