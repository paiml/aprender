//! Falsifier: `add_mask` MUST broadcast a `[B,1,1,S]` additive attention mask over
//! `[B,heads,T,S]` scores correctly AND keep the autograd edge back to `scores`.
//!
//! Contract: `setfit-encoder-conformance-v1`, equation `apply_additive_mask`.
//! Amendment A-02. Requirement ENC-03.
//!
//! BUG (plan 01-09, PMAT-913/914/922 severed-graph class): the pre-existing
//! `add_mask` in `nn/transformer/positional_encoding.rs` read:
//!
//! ```ignore
//! if scores.shape() == mask.shape() { return scores.add(mask); }
//! let data = scores.data().iter().zip(mask.data().iter())
//!     .map(|(&s, &m)| s + m).collect();
//! Tensor::from_vec(data, scores.shape())
//! ```
//!
//! Two independent defects:
//!
//! 1. **No broadcast.** `.zip()` stops at the SHORTER iterator, so a `[B,1,1,S]` mask
//!    against `[B,heads,T,S]` scores yields `B*S` summed elements where `B*H*T*S` are
//!    required. `Tensor::from_vec` asserts `data.len() == shape.product()`, so the call
//!    does not even silently truncate — it PANICS. The masked attention path was
//!    therefore unreachable at every realistic shape, and no test covered it.
//!    In the one case where the element COUNTS coincide but the shapes differ, the
//!    zip does produce values, and they are indexed wrongly.
//!
//! 2. **No graph edge.** The fallback builds the result with a bare `Tensor::from_vec`
//!    and never calls `requires_grad_` / `set_grad_fn` / `with_graph`. Masking severed
//!    the autograd tape, freezing every parameter upstream of the mask.
//!
//! These tests pin BOTH halves: elementwise numeric correctness against hand-computed
//! values at B>1, heads>1, T != S, and graph preservation (requires_grad + grad_fn +
//! a finite backward gradient of exactly the scores' shape) on the broadcast path.
//!
//! Any future edit to `add_mask` must keep every test in this file green.

use crate::autograd::{self, Tensor};
use crate::nn::transformer::add_mask;

const B: usize = 2;
const H: usize = 4;
const T: usize = 3;
const S: usize = 5;

/// Row-major flat index into `[B,H,T,S]`.
fn idx4(b: usize, h: usize, t: usize, s: usize) -> usize {
    ((b * H + h) * T + t) * S + s
}

/// Deterministic, all-distinct score values so an index-confused implementation
/// cannot coincidentally match.
fn scores_data() -> Vec<f32> {
    let mut v = Vec::with_capacity(B * H * T * S);
    for b in 0..B {
        for h in 0..H {
            for t in 0..T {
                for s in 0..S {
                    v.push(
                        0.5 + 0.1 * (b as f32)
                            + 0.03 * (h as f32)
                            + 0.007 * (t as f32)
                            + 0.0011 * (s as f32),
                    );
                }
            }
        }
    }
    v
}

/// `[B,1,1,S]` additive mask with a DISTINCT value per (b, s) pair, so a broadcast
/// over the wrong axis produces a different answer.
fn mask_data() -> Vec<f32> {
    let mut v = Vec::with_capacity(B * S);
    for b in 0..B {
        for s in 0..S {
            v.push(-(1.0 + 10.0 * (b as f32) + (s as f32)));
        }
    }
    v
}

#[test]
fn attention_mask_broadcast_b11s_over_bhts_matches_hand_computed_values() {
    let sd = scores_data();
    let md = mask_data();
    let scores = Tensor::new(&sd, &[B, H, T, S]);
    let mask = Tensor::new(&md, &[B, 1, 1, S]);

    let out = add_mask(&scores, &mask);

    assert_eq!(
        out.shape(),
        &[B, H, T, S],
        "masked output must keep the scores' shape"
    );

    for b in 0..B {
        for h in 0..H {
            for t in 0..T {
                for s in 0..S {
                    let f = idx4(b, h, t, s);
                    let expected = sd[f] + md[b * S + s];
                    let got = out.data()[f];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "out[{b}][{h}][{t}][{s}] = {got}, expected {expected} \
                         (scores {} + mask[{b}][0][0][{s}] {})",
                        sd[f],
                        md[b * S + s]
                    );
                }
            }
        }
    }
}

#[test]
fn attention_mask_broadcast_is_uniform_across_the_head_and_query_axes() {
    // Every head and every query position of a given batch row must receive the
    // SAME mask row. Catches a broadcast that leaks the head or query index.
    let scores = Tensor::new(&vec![0.0f32; B * H * T * S], &[B, H, T, S]);
    let md = mask_data();
    let mask = Tensor::new(&md, &[B, 1, 1, S]);

    let out = add_mask(&scores, &mask);

    for b in 0..B {
        for h in 0..H {
            for t in 0..T {
                for s in 0..S {
                    let got = out.data()[idx4(b, h, t, s)];
                    let expected = md[b * S + s];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "zero scores: out[{b}][{h}][{t}][{s}] = {got}, expected mask value {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn attention_mask_broadcast_distinct_batch_rows_receive_distinct_masks() {
    // WRONG-AXIS CATCHER. Zero scores, and a mask whose row 0 differs from row 1 by a
    // constant. If the implementation broadcasts over the batch axis instead of the
    // key axis (or zips), the two batch blocks come out identical.
    let scores = Tensor::new(&vec![0.0f32; B * H * T * S], &[B, H, T, S]);
    let mut md = vec![0.0f32; B * S];
    for s in 0..S {
        md[s] = -1.0; // batch row 0
        md[S + s] = -7.0; // batch row 1
    }
    let mask = Tensor::new(&md, &[B, 1, 1, S]);

    let out = add_mask(&scores, &mask);

    for h in 0..H {
        for t in 0..T {
            for s in 0..S {
                let row0 = out.data()[idx4(0, h, t, s)];
                let row1 = out.data()[idx4(1, h, t, s)];
                assert!(
                    (row0 - (-1.0)).abs() < 1e-6,
                    "batch row 0 at [{h}][{t}][{s}] = {row0}, expected -1.0"
                );
                assert!(
                    (row1 - (-7.0)).abs() < 1e-6,
                    "batch row 1 at [{h}][{t}][{s}] = {row1}, expected -7.0"
                );
                assert!(
                    (row0 - row1).abs() > 1e-3,
                    "batch rows must NOT be identical — mask broadcast over the wrong axis"
                );
            }
        }
    }
}

#[test]
fn attention_mask_broadcast_preserves_graph_on_the_broadcast_path() {
    // THE graph-preservation gate: requires_grad AND a recorded grad_fn on the
    // BROADCAST path (not just the equal-shape fast path).
    autograd::clear_graph();

    let mut scores = Tensor::new(&scores_data(), &[B, H, T, S]);
    scores.requires_grad_(true);
    let mask = Tensor::new(&mask_data(), &[B, 1, 1, S]);

    let out = add_mask(&scores, &mask);

    assert!(
        out.requires_grad_enabled(),
        "masked scores lost requires_grad — the autograd graph was severed by add_mask"
    );
    assert!(
        out.grad_fn().is_some(),
        "masked scores carry NO grad_fn — add_mask recorded no backward edge"
    );
}

#[test]
fn attention_mask_broadcast_backward_yields_finite_grad_of_scores_shape() {
    // The mask is a CONSTANT, so d(sum(c * masked))/d(scores) == c exactly.
    // This is a real numeric check on the recorded edge, not an `is_some` tautology.
    autograd::clear_graph();

    let mut scores = Tensor::new(&scores_data(), &[B, H, T, S]);
    scores.requires_grad_(true);
    let sid = scores.id();
    let mask = Tensor::new(&mask_data(), &[B, 1, 1, S]);

    let out = add_mask(&scores, &mask);

    // Non-uniform detached coefficients so the gradient is non-degenerate.
    let c: Vec<f32> = (0..B * H * T * S)
        .map(|i| 0.21 + 0.013 * (i as f32))
        .collect();
    let c_tensor = Tensor::new(&c, &[B, H, T, S]);
    let loss = out.mul(&c_tensor).sum();
    loss.backward();

    let grad = autograd::get_grad(sid)
        .expect("scores received NO gradient — add_mask severed the autograd graph");

    assert_eq!(
        grad.shape(),
        &[B, H, T, S],
        "gradient shape must equal the scores' shape"
    );
    assert!(
        grad.data().iter().all(|v| v.is_finite()),
        "gradient contains non-finite values"
    );
    for (i, (&g, &want)) in grad.data().iter().zip(c.iter()).enumerate() {
        assert!(
            (g - want).abs() < 1e-4,
            "dL/dscores[{i}] = {g}, expected {want} (mask is constant, so grad == coeff)"
        );
    }
}

#[test]
fn attention_mask_broadcast_equal_shapes_keep_existing_behavior_and_graph() {
    // REGRESSION GUARD on the path that already worked before the repair.
    autograd::clear_graph();

    let sd = scores_data();
    let mut scores = Tensor::new(&sd, &[B, H, T, S]);
    scores.requires_grad_(true);

    let md: Vec<f32> = (0..B * H * T * S).map(|i| -0.01 * (i as f32)).collect();
    let mask = Tensor::new(&md, &[B, H, T, S]);

    let out = add_mask(&scores, &mask);

    assert_eq!(out.shape(), &[B, H, T, S]);
    for i in 0..sd.len() {
        let expected = sd[i] + md[i];
        assert!(
            (out.data()[i] - expected).abs() < 1e-6,
            "equal-shape path: out[{i}] = {}, expected {expected}",
            out.data()[i]
        );
    }
    assert!(
        out.requires_grad_enabled(),
        "equal-shape path lost requires_grad"
    );
    assert!(out.grad_fn().is_some(), "equal-shape path lost its grad_fn");
}

#[test]
fn attention_mask_broadcast_2d_causal_mask_spreads_over_batch_and_heads() {
    // Existing callers pass a 2D [T,S] causal mask. Right-aligned broadcasting must
    // spread it over BOTH the batch and the head axes, and -inf entries must survive.
    autograd::clear_graph();

    const N: usize = 4;
    let sd: Vec<f32> = (0..B * H * N * N)
        .map(|i| 0.25 + 0.01 * (i as f32))
        .collect();
    let mut scores = Tensor::new(&sd, &[B, H, N, N]);
    scores.requires_grad_(true);

    let mut md = vec![0.0f32; N * N];
    for t in 0..N {
        for s in 0..N {
            if s > t {
                md[t * N + s] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::new(&md, &[N, N]);

    let out = add_mask(&scores, &mask);

    assert_eq!(out.shape(), &[B, H, N, N]);
    for b in 0..B {
        for h in 0..H {
            for t in 0..N {
                for s in 0..N {
                    let f = ((b * H + h) * N + t) * N + s;
                    let got = out.data()[f];
                    if s > t {
                        assert!(
                            got == f32::NEG_INFINITY,
                            "causal mask: out[{b}][{h}][{t}][{s}] = {got}, expected -inf"
                        );
                    } else {
                        assert!(
                            (got - sd[f]).abs() < 1e-6,
                            "causal mask: kept position [{b}][{h}][{t}][{s}] = {got}, \
                             expected unchanged {}",
                            sd[f]
                        );
                    }
                }
            }
        }
    }
    assert!(
        out.grad_fn().is_some(),
        "2D causal mask path recorded no backward edge"
    );
}

#[test]
fn attention_mask_broadcast_handles_t_not_equal_to_s() {
    // T != S is the shape that a square-mask assumption silently gets wrong.
    const TT: usize = 3;
    const SS: usize = 7;
    let sd: Vec<f32> = (0..B * H * TT * SS)
        .map(|i| 0.5 + 0.002 * (i as f32))
        .collect();
    let scores = Tensor::new(&sd, &[B, H, TT, SS]);

    let md: Vec<f32> = (0..B * SS).map(|i| -(1.0 + (i as f32))).collect();
    let mask = Tensor::new(&md, &[B, 1, 1, SS]);

    let out = add_mask(&scores, &mask);

    assert_eq!(
        out.shape(),
        &[B, H, TT, SS],
        "T != S must not change the output shape"
    );
    for b in 0..B {
        for h in 0..H {
            for t in 0..TT {
                for s in 0..SS {
                    let f = ((b * H + h) * TT + t) * SS + s;
                    let expected = sd[f] + md[b * SS + s];
                    assert!(
                        (out.data()[f] - expected).abs() < 1e-6,
                        "T!=S: out[{b}][{h}][{t}][{s}] = {}, expected {expected}",
                        out.data()[f]
                    );
                }
            }
        }
    }
}
