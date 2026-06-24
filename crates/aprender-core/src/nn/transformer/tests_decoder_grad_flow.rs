//! PMAT-922: severed-graph sweep — grad-flow guards for the functional::*
//! activations used INSIDE nn layer forward paths.
//!
//! PMAT-921 proved the ENCODER FFN's `nn::functional::gelu` severed the autograd
//! graph (builds its output via `Tensor::from_vec`, no grad_fn) and fixed the
//! encoder by routing through the autograd-aware `Tensor::gelu`. This sweep
//! closes the CLASS: the DECODER layer has the exact same severed `gelu(&ff_out)`
//! call, and `Dropout::forward` severs in training mode via `Tensor::new`.
//!
//! Each guard below trains/back-props a layer and asserts a specific upstream
//! param receives a finite, non-zero gradient. RED on the severed version,
//! GREEN on the autograd-aware fix.

use crate::autograd;
use crate::autograd::Tensor;
use crate::nn::transformer::TransformerDecoderLayer;
use crate::nn::Module;

/// Deterministic small fill so the test is CI-stable (no RNG).
fn filled(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.013 + seed).sin() * 0.1)
        .collect()
}

/// Returns true iff the tensor with `id` received a finite, non-zero gradient.
fn saw_grad(id: autograd::TensorId) -> bool {
    autograd::get_grad(id).is_some_and(|g| {
        let gd = g.data();
        gd.iter().all(|v| v.is_finite()) && gd.iter().any(|&v| v.abs() > 1e-12)
    })
}

/// GELU-SEVER GUARD: with dropout OFF (eval), the only non-autograd op left on
/// the FFN path is `gelu`. If `gelu` severs the graph, gradient cannot reach
/// `linear1.weight` / `norm3.gamma` (everything UPSTREAM of the FFN's gelu),
/// even though `linear2` (downstream of gelu) still gets a gradient.
///
/// This is the decoder twin of the PMAT-921 encoder bug.
#[test]
fn decoder_ffn_gelu_grad_flows_to_linear1_and_norm3() {
    const D: usize = 8;
    const HEADS: usize = 2;
    const FFN: usize = 16;
    const SEQ: usize = 4;

    let mut layer = TransformerDecoderLayer::new(D, HEADS, FFN);
    // eval() => all Dropout::forward become identity clones (no sever), so the
    // ONLY remaining graph-severing candidate on the FFN path is gelu.
    layer.eval();

    autograd::clear_graph();

    let tgt = Tensor::new(&filled(SEQ * D, 0.1), &[1, SEQ, D]).requires_grad();
    let out = layer.forward(&tgt);
    let loss = out.sum();
    loss.backward();

    // linear1 (FFN in-projection) and norm3 (the pre-FFN LayerNorm) are UPSTREAM
    // of the FFN gelu. With a severed gelu they get NO gradient.
    // Decoder parameters() order: self_attn(q,k,v,out × {w,b}=8) + cross_attn(8) +
    // linear1{w,b}=16,17 + linear2{w,b}=18,19 + norm1{γ,β}=20,21 + norm2=22,23 +
    // norm3{γ,β}=24,25.
    let params = layer.parameters();
    assert_eq!(
        params.len(),
        26,
        "decoder param layout changed; update indices"
    );
    let linear1_w = params[16].id(); // linear1.weight (upstream of FFN gelu)
    let linear2_w = params[18].id(); // linear2.weight (downstream of gelu)
    let norm3_gamma = params[24].id(); // norm3.gamma (pre-FFN LayerNorm, upstream of gelu)

    assert!(
        saw_grad(linear2_w),
        "linear2 (downstream of gelu) must always get a gradient — if not, the \
         test wiring is wrong, not the gelu edge"
    );
    assert!(
        saw_grad(linear1_w),
        "linear1.weight got NO gradient — the FFN gelu SEVERED the autograd graph \
         (PMAT-922 decoder twin of PMAT-921). Route gelu through Tensor::gelu."
    );
    assert!(
        saw_grad(norm3_gamma),
        "norm3.gamma got NO gradient — the FFN gelu SEVERED the autograd graph \
         upstream of the pre-FFN LayerNorm (PMAT-922)."
    );
}

/// DROPOUT-SEVER GUARD (Dropout layer): in TRAINING mode with p>0, the layer's
/// forward must route gradient back to its input. The old `Tensor::new(scaled)`
/// path produced a fresh leaf with no grad_fn, severing the graph and freezing
/// every parameter upstream of any training-mode dropout. The mask+mul fix
/// records a MulBackward edge.
#[test]
fn dropout_layer_grad_flows_to_input_in_training_mode() {
    use crate::nn::Dropout;

    autograd::clear_graph();

    // Deterministic seed so exactly-zero-everywhere (which would also "sever"
    // benignly) cannot happen by chance: with p=0.5 over 64 elems some are kept.
    let mut d = Dropout::with_seed(0.5, 0xC0FFEE);
    d.train();

    let x = Tensor::new(&filled(64, 0.2), &[8, 8]).requires_grad();
    let y = d.forward(&x);
    let loss = y.sum();
    loss.backward();

    assert!(
        saw_grad(x.id()),
        "Dropout(train, p>0) SEVERED the autograd graph — its input got NO \
         gradient. Apply the mask via Tensor::mul, not Tensor::new (PMAT-922)."
    );
}

/// DROPOUT-SEVER GUARD (functional::dropout): the canonical functional dropout
/// (used by attention's `apply_dropout`) must also record a backward edge.
#[test]
fn functional_dropout_grad_flows_to_input() {
    autograd::clear_graph();

    let x = Tensor::new(&filled(64, 0.3), &[8, 8]).requires_grad();
    // training=true, p>0 => the masking branch (the severed branch pre-fix).
    let y = crate::nn::functional::dropout(&x, 0.5, true);
    let loss = y.sum();
    loss.backward();

    assert!(
        saw_grad(x.id()),
        "nn::functional::dropout SEVERED the autograd graph — input got NO \
         gradient. Apply the mask via Tensor::mul, not Tensor::from_vec (PMAT-922)."
    );
}
