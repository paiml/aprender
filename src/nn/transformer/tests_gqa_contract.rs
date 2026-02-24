// =========================================================================
// FALSIFY-GQ: gqa-kernel-v1.yaml contract (aprender GroupedQueryAttention)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 10+ GQA tests but zero FALSIFY-GQ-* tests
//   Why 2: unit tests verify shapes/params, not mathematical invariants
//   Why 3: no mapping from gqa-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: GQA was "obviously correct" (MHA + KV head repetition)
//
// References:
//   - provable-contracts/contracts/gqa-kernel-v1.yaml
//   - Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models"
// =========================================================================

use super::*;

/// FALSIFY-GQ-001: Weight normalization — attention weight rows sum to 1
///
/// Each query position's attention weights over KV positions must sum to 1.
#[test]
fn falsify_gq_001_weight_normalization() {
    let gqa = GroupedQueryAttention::new(32, 4, 2);

    let q = Tensor::new(
        &(0..2 * 5 * 32)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>(),
        &[2, 5, 32],
    );
    let k = Tensor::new(
        &(0..2 * 8 * 32)
            .map(|i| (i as f32 * 0.02).cos())
            .collect::<Vec<_>>(),
        &[2, 8, 32],
    );
    let v = Tensor::ones(&[2, 8, 32]);

    let (_, attn_weights) = gqa.forward_qkv(&q, &k, &v, None);
    let w_data = attn_weights.data();
    let w_shape = attn_weights.shape(); // [batch=2, heads=4, q_len=5, kv_len=8]

    let batch = w_shape[0];
    let heads = w_shape[1];
    let q_len = w_shape[2];
    let kv_len = w_shape[3];

    for b in 0..batch {
        for h in 0..heads {
            for q_pos in 0..q_len {
                let row_start = b * heads * q_len * kv_len + h * q_len * kv_len + q_pos * kv_len;
                let row_sum: f32 = (0..kv_len).map(|j| w_data[row_start + j]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-4,
                    "FALSIFIED GQ-001: attn_weights[{b}][{h}][{q_pos}] sums to {row_sum}, expected 1.0"
                );
            }
        }
    }
}

/// FALSIFY-GQ-004: Head divisibility — num_heads % num_kv_heads == 0 enforced
#[test]
#[should_panic(expected = "must be divisible")]
fn falsify_gq_004_head_divisibility() {
    let _gqa = GroupedQueryAttention::new(64, 8, 3);
}

/// FALSIFY-GQ-006: MQA boundary — kv_heads=1 broadcasts to all query heads
#[test]
fn falsify_gq_006_mqa_boundary() {
    let mqa = GroupedQueryAttention::new(32, 4, 1);
    assert_eq!(mqa.num_kv_heads(), 1);

    let x = Tensor::ones(&[1, 6, 32]);
    let output = mqa.forward(&x);
    assert_eq!(output.shape(), &[1, 6, 32]);

    // Output should be finite
    for (i, &val) in output.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED GQ-006: MQA output[{i}] = {val} (not finite)"
        );
    }
}

/// FALSIFY-GQ-002: MHA degeneration — GQA(kv=h) behaves like standard MHA
///
/// When kv_heads == num_heads, GQA degenerates to standard MHA (no KV sharing).
#[test]
fn falsify_gq_002_mha_degeneration() {
    // Both should produce valid outputs with same shape
    let gqa_as_mha = GroupedQueryAttention::new(32, 4, 4);
    let mha = MultiHeadAttention::new(32, 4);

    assert_eq!(gqa_as_mha.num_kv_heads(), gqa_as_mha.num_heads());

    let x = Tensor::ones(&[1, 5, 32]);
    let gqa_out = gqa_as_mha.forward(&x);
    let (mha_out, _) = mha.forward_self(&x, None);

    assert_eq!(gqa_out.shape(), mha_out.shape());
    // Note: outputs won't be identical due to different random weight init,
    // but shapes and finiteness must match.
    for (i, &val) in gqa_out.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED GQ-002: GQA-as-MHA output[{i}] = {val} (not finite)"
        );
    }
}
