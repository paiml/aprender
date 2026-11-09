//! Tests for `OwnedQuantizedModel::parallel_multihead_attention_windowed` (attention_windowed.rs),
//! the bidirectional windowed attention block of the embedding forward (APR-EMBED-001 EG-2).
//!
//! Tests cover:
//! - A hand-computed two-position case (the softmax weights and the weighted sum)
//! - The window: a position outside |i - j| <= w has no effect; one inside it does, in
//!   both directions (no causal mask)
//! - `None` equals a window at least the sequence length
//! - Grouped-query head mapping
//! - The scale reaches the scores
//! - Shape refusals

use crate::gguf::OwnedQuantizedModel;

/// Deterministic values in roughly [-1, 1]: no RNG, so a failure reproduces exactly.
fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    heads: (usize, usize),
    head_dim: usize,
    half_window: Option<usize>,
) -> Vec<f32> {
    OwnedQuantizedModel::parallel_multihead_attention_windowed(
        q,
        k,
        v,
        seq_len,
        heads.0,
        heads.1,
        head_dim,
        1.0,
        half_window,
    )
    .expect("valid shape")
}

/// One head of width 1: position 0 scores [0, ln 3], so its weights are [1/4, 3/4] and its
/// output is 4/4 + 3·8/4 = 7.
#[test]
fn two_positions_weigh_values_by_the_softmax_of_the_scores_eg2() {
    let q = [1.0, 1.0];
    let k = [0.0, 3.0f32.ln()];
    let v = [4.0, 8.0];
    let out = attend(&q, &k, &v, 2, (1, 1), 1, None);
    assert!((out[0] - 7.0).abs() < 1e-5, "{out:?}");
    // Position 1 scores the same keys with the same query, so it is the same mixture.
    assert!((out[1] - 7.0).abs() < 1e-5, "{out:?}");
}

/// Position 0 attends to position 1, which follows it: no causal mask.
#[test]
fn a_later_position_changes_an_earlier_output_eg2() {
    let (seq, d) = (4, 4);
    let (q, k, mut v) = (ramp(seq * d, 0.0), ramp(seq * d, 1.0), ramp(seq * d, 2.0));
    let before = attend(&q, &k, &v, seq, (1, 1), d, None);
    v[3 * d] += 5.0;
    let after = attend(&q, &k, &v, seq, (1, 1), d, None);
    assert_ne!(before[..d], after[..d], "row 0 ignored a later position");
}

/// With w = 1, row i reads rows i-1..=i+1. Changing row 5's key and value moves rows 4 and 5
/// only.
#[test]
fn a_position_outside_the_window_has_no_effect_eg2() {
    let (seq, d) = (6, 4);
    let (q, mut k, mut v) = (ramp(seq * d, 0.0), ramp(seq * d, 1.0), ramp(seq * d, 2.0));
    let before = attend(&q, &k, &v, seq, (1, 1), d, Some(1));
    for x in &mut k[5 * d..] {
        *x += 3.0;
    }
    for x in &mut v[5 * d..] {
        *x -= 4.0;
    }
    let after = attend(&q, &k, &v, seq, (1, 1), d, Some(1));
    assert_eq!(before[..4 * d], after[..4 * d], "rows 0..=3 read row 5");
    assert_ne!(
        before[4 * d..5 * d],
        after[4 * d..5 * d],
        "row 4 ignored row 5"
    );
    // The edge is inclusive and symmetric: row 0 reads row 1.
    let mut v1 = ramp(seq * d, 2.0);
    v1[d] += 5.0;
    let moved = attend(&q, &ramp(seq * d, 1.0), &v1, seq, (1, 1), d, Some(1));
    assert_ne!(
        before[..d],
        moved[..d],
        "row 0 ignored row 1, inside |i - j| <= 1"
    );
}

#[test]
fn no_window_equals_a_window_of_at_least_the_sequence_length_eg2() {
    let (seq, d) = (5, 8);
    let (q, k, v) = (
        ramp(seq * 2 * d, 0.0),
        ramp(seq * d, 1.0),
        ramp(seq * d, 2.0),
    );
    let full = attend(&q, &k, &v, seq, (2, 1), d, None);
    assert_eq!(full, attend(&q, &k, &v, seq, (2, 1), d, Some(seq)));
    assert_eq!(full, attend(&q, &k, &v, seq, (2, 1), d, Some(usize::MAX)));
    assert_ne!(full, attend(&q, &k, &v, seq, (2, 1), d, Some(1)));
}

/// Four query heads over two KV heads: heads 0 and 1 read KV head 0 (values all 1), heads 2
/// and 3 read KV head 1 (values all 2). The weights sum to 1, so each output is its value.
#[test]
fn query_heads_read_their_grouped_kv_head_eg2() {
    let (seq, d) = (3, 2);
    let q = ramp(seq * 4 * d, 0.0);
    let k = ramp(seq * 2 * d, 1.0);
    let v: Vec<f32> = (0..seq).flat_map(|_| [1.0, 1.0, 2.0, 2.0]).collect();
    let out = attend(&q, &k, &v, seq, (4, 2), d, None);
    for (i, x) in out.iter().enumerate() {
        let head = (i % (4 * d)) / d;
        let want = if head < 2 { 1.0 } else { 2.0 };
        assert!((x - want).abs() < 1e-5, "element {i} (head {head}) = {x}");
    }
}

/// Scale 0 makes every score 0, so each output is the plain mean of the values in its window.
#[test]
fn the_scale_multiplies_the_scores_eg2() {
    let (seq, d) = (3, 1);
    let (q, k) = (ramp(seq, 0.0), ramp(seq, 1.0));
    let v = [3.0, 6.0, 12.0];
    let out = OwnedQuantizedModel::parallel_multihead_attention_windowed(
        &q, &k, &v, seq, 1, 1, d, 0.0, None,
    )
    .expect("valid shape");
    for x in &out {
        assert!((x - 7.0).abs() < 1e-5, "{out:?}");
    }
    assert_ne!(
        out,
        attend(&q, &k, &v, seq, (1, 1), d, None),
        "scale 1 equals scale 0"
    );
}

#[test]
fn shapes_that_do_not_fit_are_refused_eg2() {
    let call = |q: usize, kv: usize, heads: usize, kv_heads: usize, head_dim: usize| {
        OwnedQuantizedModel::parallel_multihead_attention_windowed(
            &vec![0.0; q],
            &vec![0.0; kv],
            &vec![0.0; kv],
            2,
            heads,
            kv_heads,
            head_dim,
            1.0,
            None,
        )
    };
    assert!(call(16, 8, 4, 2, 2).is_ok(), "the control shape is valid");
    assert!(call(16, 12, 4, 3, 2).is_err(), "4 heads over 3 KV heads");
    assert!(call(0, 0, 0, 1, 2).is_err(), "no query heads");
    assert!(call(0, 0, 4, 0, 2).is_err(), "no KV heads");
    assert!(call(0, 0, 4, 2, 0).is_err(), "zero head width");
    assert!(call(15, 8, 4, 2, 2).is_err(), "q one short");
    assert!(call(16, 7, 4, 2, 2).is_err(), "k and v one short");
    let ragged = OwnedQuantizedModel::parallel_multihead_attention_windowed(
        &[0.0; 16], &[0.0; 8], &[0.0; 6], 2, 4, 2, 2, 1.0, None,
    );
    assert!(ragged.is_err(), "v shorter than k");
}
