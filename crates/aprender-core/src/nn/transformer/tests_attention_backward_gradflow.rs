//! Falsifier: MultiHeadAttention backward MUST flow gradient to the Q/K/V/out
//! projection weights — end-to-end transformer trainability.
//!
//! Obligation: OBLIG-ATTENTION-BACKWARD-GRAD-FLOW
//!
//! BUG (PMAT-914): the scaled-dot-product attention core built its intermediate
//! tensors via `Tensor::from_vec` / `Tensor::new`, which severs the autograd
//! graph:
//!   - `matmul_batched` (4D path)   -> `Tensor::from_vec`   (QK^T and attn@V)
//!   - `transpose_last_two`         -> `Tensor::from_vec`   (K^T)
//!   - `reshape_for_attention`      -> `Tensor::from_vec`   (split heads)
//!   - `reshape_from_attention`     -> `Tensor::from_vec`   (concat heads)
//!   - `nn::functional::softmax`    -> `Tensor::from_vec`   (attn weights)
//! After `loss.backward()`, `get_grad(q_proj.weight.id())` etc. were `None` —
//! the Q/K/V (and the attention-side path to out) projection weights never
//! received gradient, so a transformer attention block was NON-FINE-TUNABLE
//! despite the earlier norm / embedding / pool gradflow fixes.
//!
//! This test is a self-contained finite-difference gradcheck (no torch dep):
//! perturb each projection weight[k] by ±eps, recompute the scalar loss, and
//! compare the central difference against the analytic `.grad`. It also asserts
//! grad is non-None (the severed-graph guard). The finite-diff comparison
//! genuinely catches a wrong/missing gradient — not a tautological `is_some`.

use crate::autograd::{self, Tensor};
use crate::nn::transformer::MultiHeadAttention;
use crate::nn::Module;

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

/// Fixed, non-uniform upstream coefficient so dL/dW is non-degenerate.
fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.21 + 0.13 * (i as f32)).collect()
}

/// Scalar loss = sum over all output elements of c[feat] * out[.., feat],
/// where `c` is detached (no grad). The only live edges are through the
/// projection weights and the input.
fn scalar_loss(output: &Tensor, c: &[f32]) -> Tensor {
    let shape = output.shape();
    let feat = c.len();
    let batch = output.numel() / feat;
    let mut cvec = Vec::with_capacity(output.numel());
    for _ in 0..batch {
        cvec.extend_from_slice(c);
    }
    let c_tensor = Tensor::new(&cvec, shape);
    output.mul(&c_tensor).sum()
}

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let denom = analytic.abs().max(numeric.abs()).max(1.0);
    let rel = (analytic - numeric).abs() / denom;
    assert!(
        rel < TOL,
        "{what}: analytic grad {analytic} != finite-diff {numeric} (rel err {rel})"
    );
}

/// Deterministic [batch, seq, embed] input. Amplitudes are deliberately large
/// so the QK^T scores have a wide spread across keys — this makes the softmax
/// strongly NON-uniform, so the softmax Jacobian (and hence the Q/K gradient
/// edges) are well above the finite-difference tolerance and genuinely probed.
fn make_input(batch: usize, seq: usize, embed: usize) -> Vec<f32> {
    (0..batch * seq * embed)
        .map(|k| {
            let b = (k / (seq * embed)) as f32;
            let s = ((k / embed) % seq) as f32;
            let e = (k % embed) as f32;
            0.4 + 0.5 * e - 0.6 * s + 0.3 * b - 0.2 * (e * s)
        })
        .collect()
}

/// Deterministic projection weight matrix [out, in].
fn make_weight(out: usize, inp: usize, seed: f32) -> Vec<f32> {
    (0..out * inp)
        .map(|k| {
            let r = (k / inp) as f32;
            let c = (k % inp) as f32;
            seed + 0.35 * ((r - c) as f32) + 0.2 * c - 0.1 * r
        })
        .collect()
}

const SEEDS: [f32; 4] = [0.30, 0.45, 0.25, 0.35]; // q, k, v, out

/// Install deterministic, zero-bias projection weights on an MHA. `track_grad`
/// controls whether the weights require grad (true for the analytic forward,
/// false for the finite-difference reference forward).
fn install_weights(
    mha: &mut MultiHeadAttention,
    embed: usize,
    weights: &[Vec<f32>],
    track_grad: bool,
) {
    let mk = |data: &[f32]| {
        let t = Tensor::new(data, &[embed, embed]);
        if track_grad {
            t.requires_grad()
        } else {
            t
        }
    };
    let zero_bias = || Tensor::new(&vec![0.0f32; embed], &[embed]);

    mha.q_proj_mut().set_weight(mk(&weights[0]));
    mha.q_proj_mut().set_bias(zero_bias());
    mha.k_proj_mut().set_weight(mk(&weights[1]));
    mha.k_proj_mut().set_bias(zero_bias());
    mha.v_proj_mut().set_weight(mk(&weights[2]));
    mha.v_proj_mut().set_bias(zero_bias());
    mha.out_proj_mut().set_weight(mk(&weights[3]));
    mha.out_proj_mut().set_bias(zero_bias());
}

fn base_weights(embed: usize) -> Vec<Vec<f32>> {
    SEEDS
        .iter()
        .map(|&s| make_weight(embed, embed, s))
        .collect()
}

/// Recompute the scalar loss for a perturbed copy of one projection's weights,
/// WITHOUT building an autograd graph (pure function of the weights).
fn forward_loss_with(
    x: &Tensor,
    embed: usize,
    heads: usize,
    base: &[Vec<f32>],
    which: usize,
    perturbed: &[f32],
    c: &[f32],
) -> f32 {
    autograd::no_grad(|| {
        let mut mha = MultiHeadAttention::new(embed, heads);
        let mut w = base.to_vec();
        w[which] = perturbed.to_vec();
        install_weights(&mut mha, embed, &w, false);
        let (out, _) = mha.forward_self(x, None);
        scalar_loss(&out, c).item()
    })
}

/// Finite-difference gradcheck for one projection's weights against `.grad`.
fn gradcheck_proj(
    label: &str,
    x: &Tensor,
    embed: usize,
    heads: usize,
    base: &[Vec<f32>],
    which: usize,
    analytic_grad: &[f32],
    c: &[f32],
) {
    let w = &base[which];
    // Probe EVERY weight entry (embed is small; full O(n^2) forward cost is fine)
    // so a wrong gradient on any single entry is caught.
    for idx in 0..w.len() {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp[idx] += FD_EPS;
        wm[idx] -= FD_EPS;
        let lp = forward_loss_with(x, embed, heads, base, which, &wp, c);
        let lm = forward_loss_with(x, embed, heads, base, which, &wm, c);
        let numeric = (lp - lm) / (2.0 * FD_EPS);
        assert_close(
            analytic_grad[idx],
            numeric,
            &format!("{label} weight[{idx}]"),
        );
    }
}

#[test]
fn attention_backward_flows_grad_to_qkv_and_out_proj() {
    autograd::clear_graph();

    let embed = 4usize;
    let heads = 2usize;
    let batch = 2usize;
    let seq = 3usize;

    let x = Tensor::new(&make_input(batch, seq, embed), &[batch, seq, embed]);
    let c = coeff(embed);
    let base = base_weights(embed);

    let mut mha = MultiHeadAttention::new(embed, heads);
    install_weights(&mut mha, embed, &base, true);
    let q_id = mha.q_proj_mut().weight().id();
    let k_id = mha.k_proj_mut().weight().id();
    let v_id = mha.v_proj_mut().weight().id();
    let out_id = mha.out_proj_mut().weight().id();

    let (output, _) = mha.forward_self(&x, None);
    let loss = scalar_loss(&output, &c);
    loss.backward();

    // ---- Severed-graph guard: grad MUST be present and finite-nonzero. ----
    for (name, id) in [("q", q_id), ("k", k_id), ("v", v_id), ("out", out_id)] {
        let g = autograd::get_grad(id)
            .unwrap_or_else(|| panic!("{name}_proj weight grad is None — autograd graph SEVERED"));
        let gd = g.data();
        assert!(
            gd.iter().all(|v| v.is_finite()),
            "{name}_proj grad has non-finite entries"
        );
        assert!(
            gd.iter().any(|&v| v.abs() > 1e-8),
            "{name}_proj grad is all-zero — no gradient flowed"
        );
    }

    // ---- Finite-difference gradcheck for each projection. ----
    let qg = autograd::get_grad(q_id).expect("q grad");
    let kg = autograd::get_grad(k_id).expect("k grad");
    let vg = autograd::get_grad(v_id).expect("v grad");
    let og = autograd::get_grad(out_id).expect("out grad");

    gradcheck_proj("out_proj", &x, embed, heads, &base, 3, og.data(), &c);
    gradcheck_proj("v_proj", &x, embed, heads, &base, 2, vg.data(), &c);
    gradcheck_proj("k_proj", &x, embed, heads, &base, 1, kg.data(), &c);
    gradcheck_proj("q_proj", &x, embed, heads, &base, 0, qg.data(), &c);
}
