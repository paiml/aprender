//! CPU-reference test for `nf4-backward-tensor-core-gemm-v1`: the backward pass
//! through NF4-dequantized weights. The fused tensor-core kernel is GPU-only.

use super::nf4::{dequantize_blockwise, quantize_blockwise};

/// dX = dY . W^T through dequantized NF4 weights stays within the NF4
/// quantisation error of the fp32 gradient (|dX_nf4 - dX_fp32| <= sum|dY| * eps_w).
#[test]
fn test_nf4_backward_gradient_within_quant_error() {
    let (k, n, blocksize) = (64usize, 16usize, 64usize);
    let w: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.173).sin()).collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).cos()).collect();

    let mut packed = vec![0u8; w.len() / 2];
    let mut absmax = vec![0.0f32; w.len() / blocksize];
    quantize_blockwise(&w, blocksize, &mut packed, &mut absmax);
    let mut wq = vec![0.0f32; w.len()];
    dequantize_blockwise(&packed, &absmax, blocksize, &mut wq);

    let eps_w = w.iter().zip(&wq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let l1_dy: f32 = dy.iter().map(|d| d.abs()).sum();
    assert!(eps_w > 0.0 && eps_w < 0.2, "NF4 weight error {eps_w} out of range");

    for row in 0..k {
        let g32: f32 = (0..n).map(|c| dy[c] * w[row * n + c]).sum();
        let gq: f32 = (0..n).map(|c| dy[c] * wq[row * n + c]).sum();
        assert!((g32 - gq).abs() <= l1_dy * eps_w + 1e-6, "row {row} gradient error too large");
    }
}
