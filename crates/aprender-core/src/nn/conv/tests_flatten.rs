use super::*;

#[test]
fn test_flatten_from_dim_2d() {
    // When start_dim makes it a no-op
    let flatten = Flatten::from_dim(1);
    let x = Tensor::ones(&[4, 64]);
    let y = flatten.forward(&x);

    assert_eq!(y.shape(), &[4, 64]);
}

#[test]
fn test_flatten_5d_input() {
    let flatten = Flatten::new();
    let x = Tensor::ones(&[2, 3, 4, 5, 6]);
    let y = flatten.forward(&x);

    // Flatten from dim 1: [2, 3*4*5*6] = [2, 360]
    assert_eq!(y.shape(), &[2, 360]);
}

// Additional edge cases
#[test]
fn test_conv1d_single_channel() {
    let conv = Conv1d::new(1, 1, 3);
    let x = Tensor::ones(&[1, 1, 10]);
    let y = conv.forward(&x);

    assert_eq!(y.shape(), &[1, 1, 8]);
}

#[test]
fn test_conv2d_single_channel() {
    let conv = Conv2d::new(1, 1, 3);
    let x = Tensor::ones(&[1, 1, 8, 8]);
    let y = conv.forward(&x);

    assert_eq!(y.shape(), &[1, 1, 6, 6]);
}

#[test]
fn test_conv1d_large_stride() {
    let conv = Conv1d::with_stride(4, 8, 3, 4);
    let x = Tensor::ones(&[1, 4, 20]);
    let y = conv.forward(&x);

    // Output: (20 - 3) / 4 + 1 = 5
    assert_eq!(y.shape(), &[1, 8, 5]);
}

#[test]
fn test_conv2d_large_stride() {
    let conv = Conv2d::with_stride(4, 8, 3, 4);
    let x = Tensor::ones(&[1, 4, 20, 20]);
    let y = conv.forward(&x);

    // Output: (20 - 3) / 4 + 1 = 5
    assert_eq!(y.shape(), &[1, 8, 5, 5]);
}

#[test]
fn test_conv1d_full_options() {
    let conv = Conv1d::with_options(8, 16, 5, 2, 2, true);
    let x = Tensor::ones(&[4, 8, 50]);
    let y = conv.forward(&x);

    // Output: (50 + 2*2 - 5) / 2 + 1 = 25
    assert_eq!(y.shape(), &[4, 16, 25]);
}

#[test]
fn test_conv2d_full_options() {
    let conv = Conv2d::with_options(3, 32, (5, 5), (2, 2), (2, 2), true);
    let x = Tensor::ones(&[2, 3, 28, 28]);
    let y = conv.forward(&x);

    // Output: (28 + 2*2 - 5) / 2 + 1 = 14
    assert_eq!(y.shape(), &[2, 32, 14, 14]);
}

// ==========================================================================
// Layout, Permute, im2col Tests
// ==========================================================================

#[test]
fn test_im2col_2d_matches_naive() {
    // Create a Conv2d with known weights, run both im2col and naive, compare
    let in_c = 2;
    let out_c = 3;
    let kh = 3;
    let kw = 3;
    let batch = 2;
    let in_h = 8;
    let in_w = 8;
    let stride = 1;
    let padding = 1;

    let conv_im2col = Conv2d::with_options(
        in_c,
        out_c,
        (kh, kw),
        (stride, stride),
        (padding, padding),
        true,
    );

    // Create naive version with same weights
    let mut conv_naive = Conv2d::with_options(
        in_c,
        out_c,
        (kh, kw),
        (stride, stride),
        (padding, padding),
        true,
    );
    // Copy weights from im2col version
    let params_im2col = conv_im2col.parameters();
    let mut params_naive = conv_naive.parameters_mut();
    *params_naive[0] = params_im2col[0].clone();
    *params_naive[1] = params_im2col[1].clone();
    conv_naive.use_im2col = false;

    // Create input
    let input_data: Vec<f32> = (0..batch * in_c * in_h * in_w)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = Tensor::new(&input_data, &[batch, in_c, in_h, in_w]);

    let out_im2col = conv_im2col.forward(&input);
    let out_naive = conv_naive.forward(&input);

    assert_eq!(out_im2col.shape(), out_naive.shape());

    for (a, b) in out_im2col.data().iter().zip(out_naive.data().iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "im2col vs naive mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_im2col_1d_matches_naive() {
    let in_c = 3;
    let out_c = 4;
    let k = 5;
    let batch = 2;
    let in_l = 20;
    let stride = 2;
    let padding = 2;

    let conv_im2col = Conv1d::with_options(in_c, out_c, k, stride, padding, true);

    let mut conv_naive = Conv1d::with_options(in_c, out_c, k, stride, padding, true);
    let params_im2col = conv_im2col.parameters();
    let mut params_naive = conv_naive.parameters_mut();
    *params_naive[0] = params_im2col[0].clone();
    *params_naive[1] = params_im2col[1].clone();
    conv_naive.use_im2col = false;

    let input_data: Vec<f32> = (0..batch * in_c * in_l)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = Tensor::new(&input_data, &[batch, in_c, in_l]);

    let out_im2col = conv_im2col.forward(&input);
    let out_naive = conv_naive.forward(&input);

    assert_eq!(out_im2col.shape(), out_naive.shape());

    for (a, b) in out_im2col.data().iter().zip(out_naive.data().iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "im2col vs naive mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_nhwc_layout_conv2d() {
    let in_c = 2;
    let out_c = 3;
    let ksize = 3;
    let batch = 1;
    let h = 6;
    let w = 6;

    // Create NCHW version
    let conv_nchw = Conv2d::new(in_c, out_c, ksize);

    // Create NHWC version with same weights
    let mut conv_nhwc = Conv2d::with_layout(
        in_c,
        out_c,
        (ksize, ksize),
        (1, 1),
        (0, 0),
        true,
        layout::ConvLayout::NHWC,
    );
    let params_nchw = conv_nchw.parameters();
    let mut params_nhwc = conv_nhwc.parameters_mut();
    *params_nhwc[0] = params_nchw[0].clone();
    *params_nhwc[1] = params_nchw[1].clone();

    // Create NCHW input
    let nchw_data: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let nchw_input = Tensor::new(&nchw_data, &[batch, in_c, h, w]);

    // Create equivalent NHWC input via permute
    let nhwc_input = permute::permute(&nchw_input, &[0, 2, 3, 1]);
    assert_eq!(nhwc_input.shape(), &[batch, h, w, in_c]);

    let out_nchw = conv_nchw.forward(&nchw_input);
    let out_nhwc = conv_nhwc.forward(&nhwc_input);

    // NHWC output should be in NHWC layout
    let out_h = h - ksize + 1;
    let out_w = w - ksize + 1;
    assert_eq!(out_nhwc.shape(), &[batch, out_h, out_w, out_c]);

    // Convert NHWC output to NCHW for comparison
    let out_nhwc_as_nchw = permute::permute(&out_nhwc, &[0, 3, 1, 2]);

    assert_eq!(out_nchw.shape(), out_nhwc_as_nchw.shape());
    for (a, b) in out_nchw.data().iter().zip(out_nhwc_as_nchw.data().iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "NCHW vs NHWC mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_nlc_layout_conv1d() {
    let in_c = 3;
    let out_c = 4;
    let k = 3;
    let batch = 2;
    let length = 10;

    let conv_ncl = Conv1d::new(in_c, out_c, k);

    let mut conv_nlc = Conv1d::with_layout(in_c, out_c, k, 1, 0, true, layout::ConvLayout::NLC);
    let params_ncl = conv_ncl.parameters();
    let mut params_nlc = conv_nlc.parameters_mut();
    *params_nlc[0] = params_ncl[0].clone();
    *params_nlc[1] = params_ncl[1].clone();

    // NCL input
    let ncl_data: Vec<f32> = (0..batch * in_c * length)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let ncl_input = Tensor::new(&ncl_data, &[batch, in_c, length]);

    // NLC input via permute
    let nlc_input = permute::permute(&ncl_input, &[0, 2, 1]);
    assert_eq!(nlc_input.shape(), &[batch, length, in_c]);

    let out_ncl = conv_ncl.forward(&ncl_input);
    let out_nlc = conv_nlc.forward(&nlc_input);

    let out_l = length - k + 1;
    assert_eq!(out_nlc.shape(), &[batch, out_l, out_c]);

    // Convert NLC output to NCL for comparison
    let out_nlc_as_ncl = permute::permute(&out_nlc, &[0, 2, 1]);

    assert_eq!(out_ncl.shape(), out_nlc_as_ncl.shape());
    for (a, b) in out_ncl.data().iter().zip(out_nlc_as_ncl.data().iter()) {
        assert!((a - b).abs() < 1e-4, "NCL vs NLC mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_permute_roundtrip() {
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let t = Tensor::new(&data, &[2, 3, 4, 2]);

    // NCHW -> NHWC -> NCHW
    let nhwc = permute::permute(&t, &[0, 2, 3, 1]);
    let back = permute::permute(&nhwc, &[0, 3, 1, 2]);

    assert_eq!(back.shape(), t.shape());
    for (a, b) in t.data().iter().zip(back.data().iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_layout_parse_shape() {
    let (b, c, s) = layout::ConvLayout::NCHW.parse_shape(&[4, 3, 32, 32]);
    assert_eq!(b, 4);
    assert_eq!(c, 3);
    assert_eq!(s, vec![32, 32]);

    let (b, c, s) = layout::ConvLayout::NHWC.parse_shape(&[4, 32, 32, 3]);
    assert_eq!(b, 4);
    assert_eq!(c, 3);
    assert_eq!(s, vec![32, 32]);

    let (b, c, s) = layout::ConvLayout::NCL.parse_shape(&[2, 16, 100]);
    assert_eq!(b, 2);
    assert_eq!(c, 16);
    assert_eq!(s, vec![100]);
}

#[test]
fn test_conv2d_im2col_1x1_kernel() {
    // 1x1 kernel is an edge case for im2col
    let conv = Conv2d::new(3, 8, 1);
    let x = Tensor::ones(&[2, 3, 4, 4]);
    let y = conv.forward(&x);
    assert_eq!(y.shape(), &[2, 8, 4, 4]);
}

#[test]
fn test_conv1d_im2col_1x_kernel() {
    let conv = Conv1d::new(4, 8, 1);
    let x = Tensor::ones(&[2, 4, 10]);
    let y = conv.forward(&x);
    assert_eq!(y.shape(), &[2, 8, 10]);
}

#[test]
fn test_conv2d_im2col_large_stride_with_padding() {
    let conv = Conv2d::with_options(2, 4, (3, 3), (2, 2), (1, 1), true);

    let mut conv_naive = Conv2d::with_options(2, 4, (3, 3), (2, 2), (1, 1), true);
    let params = conv.parameters();
    let mut params_n = conv_naive.parameters_mut();
    *params_n[0] = params[0].clone();
    *params_n[1] = params[1].clone();
    conv_naive.use_im2col = false;

    let input_data: Vec<f32> = (0..2 * 2 * 7 * 7).map(|i| (i as f32) * 0.01).collect();
    let input = Tensor::new(&input_data, &[2, 2, 7, 7]);

    let out_im2col = conv.forward(&input);
    let out_naive = conv_naive.forward(&input);

    assert_eq!(out_im2col.shape(), out_naive.shape());
    for (a, b) in out_im2col.data().iter().zip(out_naive.data().iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_conv2d_with_layout_constructor() {
    let conv = Conv2d::with_layout(
        3,
        16,
        (3, 3),
        (1, 1),
        (0, 0),
        true,
        layout::ConvLayout::NHWC,
    );
    let debug_str = format!("{:?}", conv);
    assert!(debug_str.contains("NHWC"));
}

#[test]
fn test_conv1d_with_layout_constructor() {
    let conv = Conv1d::with_layout(8, 16, 3, 1, 0, true, layout::ConvLayout::NLC);
    let debug_str = format!("{:?}", conv);
    assert!(debug_str.contains("NLC"));
}
