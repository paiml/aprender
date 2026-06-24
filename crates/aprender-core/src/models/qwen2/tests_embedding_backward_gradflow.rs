//! Falsifier: token `Embedding` backward MUST scatter-ADD gradient into the
//! embedding TABLE rows by index — so token embeddings are trainable.
//!
//! Obligation: OBLIG-EMBEDDING-BACKWARD-GRAD-FLOW.
//!
//! BUG (PMAT-913): `Embedding::forward` built its lookup output via
//! `Tensor::new`, severing the autograd graph: after `loss.backward()` the
//! weight table received NO gradient (`get_grad(weight.id()) == None`), making
//! token embeddings NON-TRAINABLE.
//!
//! Two checks:
//! 1. Severed-graph guard + finite-difference gradcheck of the scatter-add dW
//!    against a self-contained central difference on the embedding table.
//! 2. Scatter is ADD not overwrite: a REPEATED token id must accumulate the
//!    gradient from every position that referenced it.

use super::Embedding;
use crate::autograd::{self, Tensor};

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.29 + 0.17 * (i as f32)).collect()
}

fn scalar_loss(output: &Tensor, c: &[f32]) -> Tensor {
    let ct = Tensor::new(c, output.shape());
    output.mul(&ct).sum()
}

/// Forward + loss WITHOUT a graph, with weight value at `flat_idx` perturbed.
fn perturbed_loss(
    w_data: &[f32],
    vocab: usize,
    hidden: usize,
    ids: &[u32],
    flat_idx: usize,
    delta: f32,
    c: &[f32],
) -> f32 {
    autograd::no_grad(|| {
        let mut wd = w_data.to_vec();
        wd[flat_idx] += delta;
        let mut emb = Embedding::new(vocab, hidden);
        emb.set_weight(Tensor::new(&wd, &[vocab, hidden]));
        let y = emb.forward(ids);
        scalar_loss(&y, c).item()
    })
}

#[test]
fn embedding_backward_scatter_add_gradcheck() {
    autograd::clear_graph();
    let vocab = 6usize;
    let hidden = 4usize;
    // REPEATED index 2 (positions 0 and 3) exercises the scatter-ADD path.
    let ids = [2u32, 5, 0, 2];

    // Non-degenerate weight table.
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| 0.1 + 0.013 * (i as f32) - 0.002 * ((i * i) as f32))
        .collect();

    let mut emb = Embedding::new(vocab, hidden);
    emb.set_weight(Tensor::new(&w_data, &[vocab, hidden]).requires_grad());
    let wid = emb.weight().id();

    let y = emb.forward(&ids);
    let c = coeff(y.numel());
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let grad = autograd::get_grad(wid)
        .expect("Embedding weight received NO gradient — autograd graph severed");
    assert_eq!(grad.shape(), &[vocab, hidden]);
    assert!(grad.data().iter().all(|v| v.is_finite()));
    assert!(grad.data().iter().any(|&v| v.abs() > 1e-9));

    // Finite-difference gradcheck over the whole table.
    for i in 0..w_data.len() {
        let num = (perturbed_loss(&w_data, vocab, hidden, &ids, i, FD_EPS, &c)
            - perturbed_loss(&w_data, vocab, hidden, &ids, i, -FD_EPS, &c))
            / (2.0 * FD_EPS);
        let denom = grad.data()[i].abs().max(num.abs()).max(1.0);
        let rel = (grad.data()[i] - num).abs() / denom;
        assert!(
            rel < TOL,
            "Embedding dW[{i}]: analytic {} != finite-diff {num} (rel {rel})",
            grad.data()[i]
        );
    }

    // Rows never referenced (1, 3, 4) must have ZERO gradient.
    for &unused in &[1usize, 3, 4] {
        for j in 0..hidden {
            assert!(
                grad.data()[unused * hidden + j].abs() < 1e-9,
                "Embedding: unused row {unused} got nonzero grad"
            );
        }
    }
}

#[test]
fn embedding_backward_is_additive_for_repeated_index() {
    // Pure scatter-add check: grad_output row i = c[i*h..]. The dW row for the
    // repeated id MUST equal the SUM of both contributing positions, not just
    // the last (which an overwrite would yield).
    autograd::clear_graph();
    let vocab = 5usize;
    let hidden = 3usize;
    let ids = [4u32, 1, 4]; // id 4 appears at positions 0 and 2.

    let w_data: Vec<f32> = (0..vocab * hidden).map(|i| 0.01 * (i as f32)).collect();
    let mut emb = Embedding::new(vocab, hidden);
    emb.set_weight(Tensor::new(&w_data, &[vocab, hidden]).requires_grad());
    let wid = emb.weight().id();

    let y = emb.forward(&ids);
    let c = coeff(y.numel()); // length = 3 positions * 3 hidden = 9
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let grad = autograd::get_grad(wid).expect("severed");
    let g = grad.data();

    // dW[4] should be c[pos0] + c[pos2] (positions 0 and 2 in the [3,3] output).
    for j in 0..hidden {
        let c_pos0 = c[0 * hidden + j];
        let c_pos2 = c[2 * hidden + j];
        let expect = c_pos0 + c_pos2;
        let got = g[4 * hidden + j];
        assert!(
            (got - expect).abs() < 1e-5,
            "Embedding scatter must ADD: dW[4][{j}] = {got}, expected sum {expect} (overwrite would give {c_pos2})"
        );
        // The ADD sum is strictly larger than either single contribution.
        assert!(got > c_pos2 + 1e-6 && got > c_pos0 + 1e-6);
    }

    // dW[1] (single reference at position 1) = c[pos1].
    for j in 0..hidden {
        let expect = c[1 * hidden + j];
        assert!((g[1 * hidden + j] - expect).abs() < 1e-5);
    }
}
