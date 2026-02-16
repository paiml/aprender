//! im2col transforms for GEMM-based convolution.
//!
//! Converts convolution into matrix multiplication by unrolling input patches
//! into columns of a matrix. This enables use of BLAS/SIMD-accelerated matmul
//! for ~10-50x speedup over naive nested-loop convolution.
//!
//! # References
//!
//! - Chellapilla, K., Puri, S., & Simard, P. (2006). High performance
//!   convolutional neural networks for document processing.

/// Sample a single input value with padding boundary checks.
///
/// Returns `0.0` for positions that fall in the padding region,
/// otherwise returns the corresponding input element.
#[inline]
fn sample_padded(input: &[f32], c: usize, ih: usize, iw: usize, in_h: usize, in_w: usize, ph: usize, pw: usize) -> f32 {
    let in_bounds = ih >= ph && ih < in_h + ph && iw >= pw && iw < in_w + pw;
    if in_bounds {
        input[c * in_h * in_w + (ih - ph) * in_w + (iw - pw)]
    } else {
        0.0
    }
}

/// Fill one row of the im2col matrix for a given (channel, ky, kx) triple.
///
/// Each row corresponds to one element of the kernel applied across all
/// spatial output positions.
#[inline]
fn fill_col_row(
    col: &mut [f32],
    input: &[f32],
    row: usize,
    col_w: usize,
    c: usize,
    y: usize,
    x: usize,
    out_h: usize,
    out_w: usize,
    sh: usize,
    sw: usize,
    in_h: usize,
    in_w: usize,
    ph: usize,
    pw: usize,
) {
    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh * sh + y;
            let iw = ow * sw + x;
            col[row * col_w + oh * out_w + ow] = sample_padded(input, c, ih, iw, in_h, in_w, ph, pw);
        }
    }
}

/// Perform im2col for 2D convolution.
///
/// Converts a single batch element's input into a column matrix suitable
/// for GEMM-based convolution: `output = weight_matrix @ col_matrix + bias`.
///
/// # Arguments
///
/// * `input` - Input data for one batch element, shape `[C_in, H, W]` (flattened)
/// * `in_c` - Number of input channels
/// * `in_h` - Input height
/// * `in_w` - Input width
/// * `kh` - Kernel height
/// * `kw` - Kernel width
/// * `sh` - Stride height
/// * `sw` - Stride width
/// * `ph` - Padding height
/// * `pw` - Padding width
///
/// # Returns
///
/// `(col_data, col_height, col_width)` where:
/// - `col_data` is a flattened `[col_height, col_width]` matrix
/// - `col_height = in_c * kh * kw`
/// - `col_width = out_h * out_w`
#[must_use]
pub(crate) fn im2col_2d(
    input: &[f32],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = (in_h + 2 * ph - kh) / sh + 1;
    let out_w = (in_w + 2 * pw - kw) / sw + 1;

    let col_h = in_c * kh * kw;
    let col_w = out_h * out_w;
    let mut col = vec![0.0f32; col_h * col_w];

    for c in 0..in_c {
        for y in 0..kh {
            for x in 0..kw {
                let row = c * kh * kw + y * kw + x;
                fill_col_row(&mut col, input, row, col_w, c, y, x, out_h, out_w, sh, sw, in_h, in_w, ph, pw);
            }
        }
    }

    (col, col_h, col_w)
}

/// Perform im2col for 1D convolution.
///
/// Converts a single batch element's input into a column matrix suitable
/// for GEMM-based 1D convolution.
///
/// # Arguments
///
/// * `input` - Input data for one batch element, shape `[C_in, L]` (flattened)
/// * `in_c` - Number of input channels
/// * `in_l` - Input length
/// * `k` - Kernel size
/// * `s` - Stride
/// * `p` - Padding
///
/// # Returns
///
/// `(col_data, col_height, col_width)` where:
/// - `col_height = in_c * k`
/// - `col_width = out_l`
#[must_use]
pub(crate) fn im2col_1d(
    input: &[f32],
    in_c: usize,
    in_l: usize,
    k: usize,
    s: usize,
    p: usize,
) -> (Vec<f32>, usize, usize) {
    let out_l = (in_l + 2 * p - k) / s + 1;

    let col_h = in_c * k;
    let col_w = out_l;
    let mut col = vec![0.0f32; col_h * col_w];

    for c in 0..in_c {
        for ki in 0..k {
            let row = c * k + ki;
            for ol in 0..out_l {
                let il = ol * s + ki;

                let val = if il < p || il >= in_l + p {
                    0.0
                } else {
                    input[c * in_l + (il - p)]
                };

                col[row * col_w + ol] = val;
            }
        }
    }

    (col, col_h, col_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col_2d_no_padding() {
        // 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding
        // Input:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (col, col_h, col_w) = im2col_2d(&input, 1, 3, 3, 2, 2, 1, 1, 0, 0);

        // out_h = (3-2)/1+1 = 2, out_w = (3-2)/1+1 = 2
        assert_eq!(col_h, 1 * 2 * 2); // 4
        assert_eq!(col_w, 2 * 2); // 4

        // col should be:
        // row 0 (c=0, ky=0, kx=0): 1, 2, 4, 5
        // row 1 (c=0, ky=0, kx=1): 2, 3, 5, 6
        // row 2 (c=0, ky=1, kx=0): 4, 5, 7, 8
        // row 3 (c=0, ky=1, kx=1): 5, 6, 8, 9
        assert_eq!(col, vec![
            1.0, 2.0, 4.0, 5.0,
            2.0, 3.0, 5.0, 6.0,
            4.0, 5.0, 7.0, 8.0,
            5.0, 6.0, 8.0, 9.0,
        ]);
    }

    #[test]
    fn test_im2col_2d_with_padding() {
        // 1 channel, 2x2 input, 3x3 kernel, stride 1, padding 1
        let input = [1.0, 2.0, 3.0, 4.0];
        let (col, col_h, col_w) = im2col_2d(&input, 1, 2, 2, 3, 3, 1, 1, 1, 1);

        // out_h = (2+2-3)/1+1 = 2, out_w = (2+2-3)/1+1 = 2
        assert_eq!(col_h, 9); // 1*3*3
        assert_eq!(col_w, 4); // 2*2
    }

    #[test]
    fn test_im2col_2d_stride_2() {
        // 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let (col, col_h, col_w) = im2col_2d(&input, 1, 4, 4, 2, 2, 2, 2, 0, 0);

        // out_h = (4-2)/2+1 = 2, out_w = (4-2)/2+1 = 2
        assert_eq!(col_h, 4); // 1*2*2
        assert_eq!(col_w, 4); // 2*2

        // Patches at (0,0), (0,2), (2,0), (2,2)
        // row 0 (ky=0, kx=0): input[0,0]=1, input[0,2]=3, input[2,0]=9, input[2,2]=11
        assert_eq!(&col[0..4], &[1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn test_im2col_2d_multi_channel() {
        // 2 channels, 2x2 input each, 2x2 kernel, stride 1, no padding
        let input = [
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];
        let (col, col_h, col_w) = im2col_2d(&input, 2, 2, 2, 2, 2, 1, 1, 0, 0);

        // out_h = 1, out_w = 1 -> single output spatial element
        assert_eq!(col_h, 8); // 2*2*2
        assert_eq!(col_w, 1);

        // Column should contain all elements from both channels
        assert_eq!(col, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_im2col_1d_no_padding() {
        // 1 channel, length 5, kernel 3, stride 1, no padding
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (col, col_h, col_w) = im2col_1d(&input, 1, 5, 3, 1, 0);

        // out_l = (5-3)/1+1 = 3
        assert_eq!(col_h, 3); // 1*3
        assert_eq!(col_w, 3);

        // row 0 (k=0): 1, 2, 3
        // row 1 (k=1): 2, 3, 4
        // row 2 (k=2): 3, 4, 5
        assert_eq!(col, vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
        ]);
    }

    #[test]
    fn test_im2col_1d_with_padding() {
        // 1 channel, length 3, kernel 3, stride 1, padding 1
        let input = [1.0, 2.0, 3.0];
        let (col, col_h, col_w) = im2col_1d(&input, 1, 3, 3, 1, 1);

        // out_l = (3+2-3)/1+1 = 3
        assert_eq!(col_h, 3);
        assert_eq!(col_w, 3);

        // row 0 (k=0): 0(pad), 1, 2
        // row 1 (k=1): 1, 2, 3
        // row 2 (k=2): 2, 3, 0(pad)
        assert_eq!(col, vec![
            0.0, 1.0, 2.0,
            1.0, 2.0, 3.0,
            2.0, 3.0, 0.0,
        ]);
    }

    #[test]
    fn test_im2col_1d_stride_2() {
        // 1 channel, length 6, kernel 3, stride 2, no padding
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (col, col_h, col_w) = im2col_1d(&input, 1, 6, 3, 2, 0);

        // out_l = (6-3)/2+1 = 2
        assert_eq!(col_h, 3);
        assert_eq!(col_w, 2);

        // row 0 (k=0): 1, 3
        // row 1 (k=1): 2, 4
        // row 2 (k=2): 3, 5
        assert_eq!(col, vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);
    }

    #[test]
    fn test_im2col_1d_multi_channel() {
        // 2 channels, length 3 each, kernel 2, stride 1, no padding
        let input = [
            1.0, 2.0, 3.0, // channel 0
            4.0, 5.0, 6.0, // channel 1
        ];
        let (col, col_h, col_w) = im2col_1d(&input, 2, 3, 2, 1, 0);

        // out_l = (3-2)/1+1 = 2
        assert_eq!(col_h, 4); // 2*2
        assert_eq!(col_w, 2);

        // row 0 (c=0, k=0): 1, 2
        // row 1 (c=0, k=1): 2, 3
        // row 2 (c=1, k=0): 4, 5
        // row 3 (c=1, k=1): 5, 6
        assert_eq!(col, vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_im2col_2d_1x1_kernel() {
        // 1x1 kernel is a special case — col is just the reshaped input
        let input = [1.0, 2.0, 3.0, 4.0]; // 1 channel, 2x2
        let (col, col_h, col_w) = im2col_2d(&input, 1, 2, 2, 1, 1, 1, 1, 0, 0);

        assert_eq!(col_h, 1); // 1*1*1
        assert_eq!(col_w, 4); // 2*2
        assert_eq!(col, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_im2col_1d_1x_kernel() {
        let input = [1.0, 2.0, 3.0];
        let (col, col_h, col_w) = im2col_1d(&input, 1, 3, 1, 1, 0);

        assert_eq!(col_h, 1);
        assert_eq!(col_w, 3);
        assert_eq!(col, vec![1.0, 2.0, 3.0]);
    }
}
