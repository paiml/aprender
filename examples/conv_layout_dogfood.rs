//! Dogfood test for GH-159: Conv layout optimization with im2col+GEMM.
//!
//! ```bash
//! cargo run --release --example conv_layout_dogfood
//! ```

use aprender::autograd::Tensor;
use aprender::nn::{Conv1d, Conv2d, ConvDimensionNumbers, ConvLayout, KernelLayout, Module};

#[allow(clippy::cognitive_complexity)]
fn main() {
    println!("=== GH-159 Dogfood: Conv Layout Optimization ===\n");

    // Test 1: Conv2d im2col (default) produces correct shapes
    println!("Test 1: Conv2d im2col shape correctness");
    let conv2d = Conv2d::new(3, 16, 3);
    let x = Tensor::ones(&[4, 3, 32, 32]);
    let y = conv2d.forward(&x);
    assert_eq!(y.shape(), &[4, 16, 30, 30]);
    println!("  PASS: [4,3,32,32] -> [4,16,30,30]");

    // Test 2: Conv2d NHWC layout
    println!("Test 2: Conv2d NHWC layout");
    let conv2d_nhwc = Conv2d::with_layout(3, 16, (3, 3), (1, 1), (0, 0), true, ConvLayout::NHWC);
    let x_nhwc = Tensor::ones(&[4, 32, 32, 3]);
    let y_nhwc = conv2d_nhwc.forward(&x_nhwc);
    assert_eq!(y_nhwc.shape(), &[4, 30, 30, 16]);
    println!("  PASS: [4,32,32,3] NHWC -> [4,30,30,16] NHWC");

    // Test 3: Conv1d im2col
    println!("Test 3: Conv1d im2col shape correctness");
    let conv1d = Conv1d::new(16, 32, 5);
    let x1d = Tensor::ones(&[8, 16, 100]);
    let y1d = conv1d.forward(&x1d);
    assert_eq!(y1d.shape(), &[8, 32, 96]);
    println!("  PASS: [8,16,100] -> [8,32,96]");

    // Test 4: Conv1d NLC layout
    println!("Test 4: Conv1d NLC layout");
    let conv1d_nlc = Conv1d::with_layout(16, 32, 5, 1, 0, true, ConvLayout::NLC);
    let x_nlc = Tensor::ones(&[8, 100, 16]);
    let y_nlc = conv1d_nlc.forward(&x_nlc);
    assert_eq!(y_nlc.shape(), &[8, 96, 32]);
    println!("  PASS: [8,100,16] NLC -> [8,96,32] NLC");

    // Test 5: Numerical equivalence — manual element verification
    println!("Test 5: Numerical equivalence (manual element check)");
    let conv_a = Conv2d::with_options(2, 4, (3, 3), (1, 1), (1, 1), true);
    let weights = conv_a.parameters()[0].data().to_vec();
    let bias = conv_a.parameters()[1].data().to_vec();

    let input_data: Vec<f32> = (0..2 * 2 * 8 * 8).map(|i| (i as f32) * 0.01).collect();
    let input = Tensor::new(&input_data, &[2, 2, 8, 8]);

    let out = conv_a.forward(&input);
    let out_data = out.data();

    let mut expected = bias[0];
    for ic in 0..2usize {
        for kh in 0..3usize {
            for kw in 0..3usize {
                let ih = kh as i32 - 1;
                let iw = kw as i32 - 1;
                let val = if ih < 0 || ih >= 8 || iw < 0 || iw >= 8 {
                    0.0
                } else {
                    input_data[ic * 64 + (ih as usize) * 8 + (iw as usize)]
                };
                let w_idx = ic * 9 + kh * 3 + kw;
                expected += val * weights[w_idx];
            }
        }
    }
    let actual = out_data[0];
    assert!(
        (actual - expected).abs() < 1e-4,
        "expected {expected}, got {actual}"
    );
    println!("  PASS: output[0,0,0,0] = {actual:.6} (expected {expected:.6})");

    // Test 6: Larger conv to exercise GEMM path
    println!("Test 6: Larger conv (64->128 ch, 56x56)");
    let conv_big = Conv2d::with_options(64, 128, (3, 3), (1, 1), (1, 1), true);
    let x_big = Tensor::ones(&[1, 64, 56, 56]);
    let start = std::time::Instant::now();
    let y_big = conv_big.forward(&x_big);
    let elapsed = start.elapsed();
    assert_eq!(y_big.shape(), &[1, 128, 56, 56]);
    println!("  PASS: [1,64,56,56] -> [1,128,56,56] in {elapsed:?}");

    // Test 7: Public API types
    println!("Test 7: Layout types usable from public API");
    let cdn = ConvDimensionNumbers::default();
    assert_eq!(cdn.input_layout, ConvLayout::NCHW);
    assert_eq!(cdn.kernel_layout, KernelLayout::OIHW);
    println!("  PASS: ConvDimensionNumbers, KernelLayout, ConvLayout all importable");

    println!("\n=== All 7 dogfood tests PASSED ===");
}
