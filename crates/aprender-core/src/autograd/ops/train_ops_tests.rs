//! Falsifiers for `contracts/modernbert-train-v1.yaml` (APR-LAYA-TRAIN-001 LT-2).
//!
//! Every differentiable op is gradchecked against a central finite difference whose
//! forward runs in f32 and whose loss/difference bookkeeping runs in f64
//! (`h = 1e-3`, loss = `sum(out * W)` with a fixed non-uniform `W`). Every gradcheck
//! also carries a MUTANT proof: the same checker, fed the sign-flipped analytical
//! gradient, must reject it — so a checker that accepts anything turns RED.

use super::{clip_grad_norm_, concat, local_window_mask, local_window_padding_mask, NEG_MASK};
use crate::autograd::{self, Tensor};

const H: f64 = 1e-3;
const TOL: f64 = 1e-2;

/// Deterministic pseudo-random values in roughly [-1, 1).
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// One differentiable input: data and shape.
struct Input {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn input(rows: usize, cols: usize, seed: u64) -> Input {
    Input {
        data: rand_vec(rows * cols, seed),
        shape: vec![rows, cols],
    }
}

type Fwd = dyn Fn(&[Tensor]) -> Vec<Tensor>;

/// Fixed non-uniform weights, one per output, sized from a graph-free forward.
fn weights(inputs: &[Input], fwd: &Fwd) -> Vec<Vec<f32>> {
    let outs = autograd::no_grad(|| fwd(&leaves(inputs, false)));
    outs.iter()
        .enumerate()
        .map(|(k, o)| rand_vec(o.numel(), 1000 + k as u64))
        .collect()
}

fn leaves(inputs: &[Input], grad: bool) -> Vec<Tensor> {
    inputs
        .iter()
        .map(|i| {
            let t = Tensor::new(&i.data, &i.shape);
            if grad {
                t.requires_grad()
            } else {
                t
            }
        })
        .collect()
}

/// f64 loss `sum_k sum(out_k * W_k)` of an f32 forward, without a graph.
fn loss_f64(inputs: &[Input], fwd: &Fwd, w: &[Vec<f32>]) -> f64 {
    let outs = autograd::no_grad(|| fwd(&leaves(inputs, false)));
    outs.iter()
        .zip(w)
        .flat_map(|(o, wk)| {
            o.data()
                .iter()
                .zip(wk)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
        })
        .sum()
}

/// Analytical gradients via the tape, one per input.
fn analytic(inputs: &[Input], fwd: &Fwd, w: &[Vec<f32>]) -> Vec<Vec<f32>> {
    autograd::clear_graph();
    let xs = leaves(inputs, true);
    let outs = fwd(&xs);
    let mut loss: Option<Tensor> = None;
    for (o, wk) in outs.iter().zip(w) {
        let term = o.mul(&Tensor::new(wk, o.shape())).sum();
        loss = Some(match loss {
            Some(l) => l.add(&term),
            None => term,
        });
    }
    loss.expect("forward produced no output").backward();
    let grads = xs
        .iter()
        .map(|x| {
            autograd::get_grad(x.id())
                .expect("input received no gradient: graph severed")
                .data()
                .to_vec()
        })
        .collect();
    autograd::clear_graph();
    grads
}

/// Central finite difference in f64 bookkeeping for every element of every input.
fn numeric(inputs: &[Input], fwd: &Fwd, w: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let mut work: Vec<Input> = inputs
        .iter()
        .map(|i| Input {
            data: i.data.clone(),
            shape: i.shape.clone(),
        })
        .collect();
    let mut grads = Vec::with_capacity(inputs.len());
    for a in 0..inputs.len() {
        let mut g = Vec::with_capacity(inputs[a].data.len());
        for k in 0..inputs[a].data.len() {
            let x0 = f64::from(inputs[a].data[k]);
            let (xp, xm) = ((x0 + H) as f32, (x0 - H) as f32);
            work[a].data[k] = xp;
            let lp = loss_f64(&work, fwd, w);
            work[a].data[k] = xm;
            let lm = loss_f64(&work, fwd, w);
            work[a].data[k] = inputs[a].data[k];
            g.push((lp - lm) / (f64::from(xp) - f64::from(xm)));
        }
        grads.push(g);
    }
    grads
}

/// The checker: `|a - n| <= TOL * max(1, |n|)` at every element.
fn grads_agree(analytic: &[Vec<f32>], numeric: &[Vec<f64>]) -> bool {
    analytic.iter().zip(numeric).all(|(a, n)| {
        a.len() == n.len()
            && a.iter()
                .zip(n)
                .all(|(&ai, &ni)| (f64::from(ai) - ni).abs() <= TOL * ni.abs().max(1.0))
    })
}

fn negate(g: &[Vec<f32>]) -> Vec<Vec<f32>> {
    g.iter().map(|v| v.iter().map(|x| -x).collect()).collect()
}

/// Gradcheck plus the sign-flip mutant proof.
fn gradcheck(name: &str, inputs: &[Input], fwd: &Fwd) {
    let w = weights(inputs, fwd);
    let a = analytic(inputs, fwd, &w);
    let n = numeric(inputs, fwd, &w);
    assert!(
        a.iter().flatten().any(|v| v.abs() > 1e-3),
        "{name}: analytical gradient is all ~zero, the mutant proof would be vacuous"
    );
    assert!(
        grads_agree(&a, &n),
        "{name}: analytical {a:?} != finite difference {n:?}"
    );
    assert!(
        !grads_agree(&negate(&a), &n),
        "{name}: MUTANT survived — the checker accepted a sign-flipped backward"
    );
}

// ---------------------------------------------------------------------------
// narrow / chunk / concat
// ---------------------------------------------------------------------------

#[test]
fn train_ops_narrow_dim1_gradcheck_and_mutant() {
    gradcheck("narrow dim1", &[input(3, 5, 1)], &|x| {
        vec![x[0].narrow(1, 1, 3)]
    });
}

#[test]
fn train_ops_narrow_dim0_gradcheck_and_mutant() {
    gradcheck("narrow dim0", &[input(4, 3, 2)], &|x| {
        vec![x[0].narrow(0, 2, 2)]
    });
}

#[test]
fn train_ops_narrow_forward_and_zero_outside_window() {
    autograd::clear_graph();
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).requires_grad();
    let y = x.narrow(1, 1, 2);
    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(y.data(), &[2.0, 3.0, 5.0, 6.0]);
    y.sum().backward();
    let g = autograd::get_grad(x.id()).expect("narrow must record a backward");
    assert_eq!(g.data(), &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    autograd::clear_graph();
}

#[test]
fn train_ops_chunk_fused_qkv_gradcheck_and_mutant() {
    // A fused Wqkv output [N, 3d] with d = 2.
    gradcheck("chunk(3)", &[input(3, 6, 3)], &|x| x[0].chunk(3));
}

#[test]
fn train_ops_chunk_tiles_columns() {
    let x = Tensor::new(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 6]);
    let parts = x.chunk(3);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].data(), &[0.0, 1.0]);
    assert_eq!(parts[1].data(), &[2.0, 3.0]);
    assert_eq!(parts[2].data(), &[4.0, 5.0]);
}

#[test]
fn train_ops_concat_dim1_gradcheck_and_mutant() {
    gradcheck("concat dim1", &[input(2, 3, 4), input(2, 2, 5)], &|x| {
        vec![concat(&[&x[0], &x[1]], 1)]
    });
}

#[test]
fn train_ops_concat_dim0_gradcheck_and_mutant() {
    gradcheck(
        "concat dim0",
        &[input(2, 3, 6), input(1, 3, 7), input(3, 3, 8)],
        &|x| vec![concat(&[&x[0], &x[1], &x[2]], 0)],
    );
}

#[test]
fn train_ops_concat_forward_layout() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0], &[2, 1]);
    let c = concat(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

// ---------------------------------------------------------------------------
// clamp / max_scalar
// ---------------------------------------------------------------------------

/// Values kept at least 0.05 away from the bounds +-0.4 so `x +- h` never crosses a kink.
fn away_from_kinks(rows: usize, cols: usize, seed: u64) -> Input {
    let mut i = input(rows, cols, seed);
    for v in &mut i.data {
        if (v.abs() - 0.4).abs() < 0.05 {
            *v += 0.1 * v.signum();
        }
    }
    i
}

#[test]
fn train_ops_clamp_both_bounds_gradcheck_and_mutant() {
    gradcheck("clamp both", &[away_from_kinks(3, 4, 9)], &|x| {
        vec![x[0].clamp(Some(-0.4), Some(0.4))]
    });
}

#[test]
fn train_ops_clamp_lower_only_gradcheck_and_mutant() {
    gradcheck("clamp lo", &[away_from_kinks(3, 4, 10)], &|x| {
        vec![x[0].clamp(Some(-0.4), None)]
    });
}

#[test]
fn train_ops_clamp_upper_only_gradcheck_and_mutant() {
    gradcheck("clamp hi", &[away_from_kinks(3, 4, 11)], &|x| {
        vec![x[0].clamp(None, Some(0.4))]
    });
}

#[test]
fn train_ops_clamp_max_scalar_gradcheck_and_mutant() {
    gradcheck("max_scalar", &[away_from_kinks(3, 4, 12)], &|x| {
        vec![x[0].max_scalar(0.4)]
    });
}

#[test]
fn train_ops_clamp_boundary_passes_gradient_like_torch() {
    // torch.clamp backward: grad passes where min <= x <= max (inclusive).
    autograd::clear_graph();
    let x = Tensor::new(&[-1.0, -0.5, 0.0, 0.5, 1.0], &[1, 5]).requires_grad();
    let y = x.clamp(Some(-0.5), Some(0.5));
    assert_eq!(y.data(), &[-0.5, -0.5, 0.0, 0.5, 0.5]);
    y.sum().backward();
    let g = autograd::get_grad(x.id()).expect("clamp must record a backward");
    assert_eq!(g.data(), &[0.0, 1.0, 1.0, 1.0, 0.0]);
    autograd::clear_graph();

    let x = Tensor::new(&[-1.0, 0.0, 2.0], &[1, 3]).requires_grad();
    let y = x.max_scalar(0.0);
    assert_eq!(y.data(), &[0.0, 0.0, 2.0]);
    y.sum().backward();
    let g = autograd::get_grad(x.id()).expect("max_scalar must record a backward");
    assert_eq!(g.data(), &[0.0, 1.0, 1.0]);
    autograd::clear_graph();
}

// ---------------------------------------------------------------------------
// rope_rotate_half
// ---------------------------------------------------------------------------

#[test]
fn train_ops_rope_local_theta_gradcheck_and_mutant() {
    // S = 3, heads = 2, head_dim = 4.
    gradcheck("rope theta=1e4", &[input(3, 8, 13)], &|x| {
        vec![x[0].rope_rotate_half(&[0, 3, 7], 4, 10_000.0)]
    });
}

#[test]
fn train_ops_rope_global_theta_gradcheck_and_mutant() {
    gradcheck("rope theta=1.6e5", &[input(3, 8, 14)], &|x| {
        vec![x[0].rope_rotate_half(&[1, 5, 11], 4, 160_000.0)]
    });
}

#[test]
fn train_ops_rope_position_zero_is_identity() {
    let x = Tensor::new(&rand_vec(8, 15), &[1, 8]);
    let y = x.rope_rotate_half(&[0], 4, 10_000.0);
    assert_eq!(y.data(), x.data());
}

#[test]
fn train_ops_rope_theta_changes_output_beyond_position_zero() {
    let x = Tensor::new(&rand_vec(16, 16), &[2, 8]);
    let local = x.rope_rotate_half(&[0, 5], 4, 10_000.0);
    let global = x.rope_rotate_half(&[0, 5], 4, 160_000.0);
    assert_eq!(
        &local.data()[..8],
        &global.data()[..8],
        "position 0 is theta-free"
    );
    assert!(
        local.data()[8..]
            .iter()
            .zip(&global.data()[8..])
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "theta must matter at position > 0"
    );
}

#[test]
fn train_ops_rope_rotate_half_layout_by_hand() {
    // head_dim = 2: x1 = x[0], x2 = x[1], angle = p * theta^0 = p.
    let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let y = x.rope_rotate_half(&[1], 2, 10_000.0);
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    assert!((y.data()[0] - (1.0 * c - 2.0 * s)).abs() < 1e-6);
    assert!((y.data()[1] - (2.0 * c + 1.0 * s)).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// local window mask
// ---------------------------------------------------------------------------

#[test]
fn train_ops_window_boundary_is_inclusive() {
    let (seq, w) = (8, 2);
    let m = local_window_mask(seq, w).expect("mask must build");
    assert_eq!(m.shape(), &[seq, seq]);
    assert!(!m.requires_grad_enabled(), "the window mask is a constant");
    let at = |i: usize, j: usize| m.data()[i * seq + j];
    assert_eq!(at(3, 3 + w), 0.0, "|i-j| == half_window is kept");
    assert_eq!(at(3, 3 - w), 0.0, "|i-j| == half_window is kept (left)");
    assert_eq!(
        at(3, 3 + w + 1),
        NEG_MASK,
        "|i-j| == half_window+1 is masked"
    );
    assert_eq!(
        at(3, 0),
        NEG_MASK,
        "|i-j| == half_window+1 is masked (left)"
    );
    for i in 0..seq {
        assert_eq!(at(i, i), 0.0, "diagonal always kept");
    }
}

#[test]
fn train_ops_window_off_by_one_mutants_differ() {
    let base = local_window_mask(8, 2).expect("mask must build");
    for mutant in [1, 3] {
        let m = local_window_mask(8, mutant).expect("mask must build");
        assert!(
            m.data().iter().zip(base.data()).any(|(a, b)| a != b),
            "half_window {mutant} must change at least one entry"
        );
    }
}

#[test]
fn train_ops_window_rejects_zero_seq() {
    assert!(local_window_mask(0, 1).is_err());
}

#[test]
fn train_ops_window_padding_combination() {
    // B = 2, S = 4; row 1 pads the last key.
    let mask = [1u8, 1, 1, 1, 1, 1, 1, 0];
    let m = local_window_padding_mask(&mask, 2, 4, 1).expect("mask must build");
    assert_eq!(m.shape(), &[2, 1, 4, 4]);
    let at = |b: usize, i: usize, j: usize| m.data()[b * 16 + i * 4 + j];
    assert_eq!(at(0, 3, 3), 0.0);
    assert_eq!(
        at(1, 3, 3),
        NEG_MASK,
        "padded key masked even on the diagonal"
    );
    assert_eq!(at(1, 2, 3), NEG_MASK, "padded key masked inside the window");
    assert_eq!(at(1, 2, 1), 0.0, "kept key inside the window");
    assert_eq!(at(1, 0, 2), NEG_MASK, "kept key outside the window");
    assert!(local_window_padding_mask(&[0, 0], 1, 2, 1).is_err());
}

// ---------------------------------------------------------------------------
// clip_grad_norm_
// ---------------------------------------------------------------------------

/// Two leaves with grads [3, 0] and [0, 4, 0]: global norm 5.
fn two_grads() -> (Tensor, Tensor) {
    autograd::clear_graph();
    let a = Tensor::new(&[1.0, 1.0], &[1, 2]).requires_grad();
    let b = Tensor::new(&[1.0, 1.0, 1.0], &[1, 3]).requires_grad();
    let la = a.mul(&Tensor::new(&[3.0, 0.0], &[1, 2])).sum();
    let lb = b.mul(&Tensor::new(&[0.0, 4.0, 0.0], &[1, 3])).sum();
    la.add(&lb).backward();
    (a, b)
}

fn global_norm(ts: &[&Tensor]) -> f64 {
    ts.iter()
        .flat_map(|t| {
            autograd::get_grad(t.id())
                .expect("grad present")
                .data()
                .to_vec()
        })
        .map(|v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn train_ops_clip_total_and_clipped_norm() {
    let (a, b) = two_grads();
    let total = clip_grad_norm_(&[a.id(), b.id()], 1.0);
    assert!(
        (total - 5.0).abs() < 1e-6,
        "pre-clip global norm, got {total}"
    );
    let after = global_norm(&[&a, &b]);
    assert!(
        (after - 1.0).abs() < 1e-4,
        "clipped norm must be max_norm, got {after}"
    );
    let ga = autograd::get_grad(a.id()).expect("grad present");
    assert!(
        (ga.data()[0] - 0.6).abs() < 1e-5,
        "scaled jointly, not per tensor"
    );
    autograd::clear_graph();
}

#[test]
fn train_ops_clip_below_threshold_is_unchanged() {
    let (a, b) = two_grads();
    let total = clip_grad_norm_(&[a.id(), b.id()], 10.0);
    assert!((total - 5.0).abs() < 1e-6);
    assert_eq!(
        autograd::get_grad(a.id()).expect("grad").data(),
        &[3.0, 0.0]
    );
    assert_eq!(
        autograd::get_grad(b.id()).expect("grad").data(),
        &[0.0, 4.0, 0.0]
    );
    autograd::clear_graph();
}
