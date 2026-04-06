//! N-D tensor permutation for layout conversion.
//!
//! Provides `permute()` which reorders dimensions of a tensor according
//! to a permutation vector, enabling NCHW <-> NHWC conversions.

use crate::autograd::Tensor;

/// Permute tensor dimensions according to `perm`.
///
/// `perm` must be a valid permutation of `0..tensor.ndim()`.
/// Returns a new tensor where dimension `i` of the output corresponds
/// to dimension `perm[i]` of the input.
///
/// # Panics
///
/// Panics if `perm` is not a valid permutation of `0..ndim`.
#[must_use]
pub(crate) fn permute(tensor: &Tensor, perm: &[usize]) -> Tensor {
    let shape = tensor.shape();
    let ndim = shape.len();

    assert_eq!(
        perm.len(),
        ndim,
        "permutation length {} != tensor ndim {}",
        perm.len(),
        ndim
    );

    // Validate perm is a valid permutation
    let mut seen = vec![false; ndim];
    for &p in perm {
        assert!(
            p < ndim,
            "permutation index {p} out of range for ndim {ndim}"
        );
        assert!(!seen[p], "duplicate index {p} in permutation");
        seen[p] = true;
    }

    // Compute new shape
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let total: usize = new_shape.iter().product();

    if total == 0 {
        return Tensor::new(&[], &new_shape);
    }

    // Compute input strides (row-major)
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    let data = tensor.data();
    let mut output = vec![0.0f32; total];

    // For each output index, compute the corresponding input index
    let mut out_coords = vec![0usize; ndim];
    for out_idx in 0..total {
        // Convert flat output index to coordinates
        let mut remaining = out_idx;
        for d in 0..ndim {
            out_coords[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        // Map output coords to input coords via permutation
        let mut in_idx = 0;
        for d in 0..ndim {
            in_idx += out_coords[d] * in_strides[perm[d]];
        }

        output[out_idx] = data[in_idx];
    }

    Tensor::new(&output, &new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_2d_transpose() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = permute(&t, &[1, 0]);
        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(p.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_permute_identity() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = permute(&t, &[0, 1]);
        assert_eq!(p.shape(), &[2, 3]);
        assert_eq!(p.data(), t.data());
    }

    #[test]
    fn test_permute_3d() {
        // Shape [2, 3, 4] -> permute [0, 2, 1] -> [2, 4, 3]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(&data, &[2, 3, 4]);
        let p = permute(&t, &[0, 2, 1]);
        assert_eq!(p.shape(), &[2, 4, 3]);

        // Check a known element: t[0,1,2] = 6.0 should be at p[0,2,1] = 0*12 + 2*3 + 1 = 7
        assert!((p.data()[7] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_permute_4d_nchw_to_nhwc() {
        // [N=1, C=2, H=3, W=4] -> [N=1, H=3, W=4, C=2]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(&data, &[1, 2, 3, 4]);
        let p = permute(&t, &[0, 2, 3, 1]);
        assert_eq!(p.shape(), &[1, 3, 4, 2]);

        // t[0,0,0,0] = 0.0 -> p[0,0,0,0] = 0.0
        assert!((p.data()[0] - 0.0).abs() < 1e-6);
        // t[0,1,0,0] = 12.0 -> p[0,0,0,1] = 12.0
        assert!((p.data()[1] - 12.0).abs() < 1e-6);
        // t[0,0,1,0] = 4.0 -> p[0,1,0,0] index = 1*8+0*2+0 = 8... wait
        // p shape [1,3,4,2], strides [24,8,2,1]
        // p[0,1,0,0] = 0*24 + 1*8 + 0*2 + 0 = 8
        // input: out_coords[0]=0, [1]=1, [2]=0, [3]=0
        // in_idx = 0*in_strides[perm[0]=0] + 1*in_strides[perm[1]=2] + 0*in_strides[perm[2]=3] + 0*in_strides[perm[3]=1]
        // in_strides = [12, 4, 1] wait, [24/2=12, 12/3=4, 4/4=1... no]
        // shape [1,2,3,4], strides = [24, 12, 4, 1]
        // in_idx = 0*24 + 1*4 + 0*1 + 0*12 = 4
        // t[0,0,1,0] = 4.0
        assert!((p.data()[8] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_permute_roundtrip_nchw_nhwc() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::new(&data, &[1, 2, 3, 4]);

        // NCHW -> NHWC -> NCHW
        let nhwc = permute(&t, &[0, 2, 3, 1]);
        let back = permute(&nhwc, &[0, 3, 1, 2]);

        assert_eq!(back.shape(), &[1, 2, 3, 4]);
        for (a, b) in t.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_permute_roundtrip_ncl_nlc() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::new(&data, &[2, 3, 2]);

        // NCL -> NLC -> NCL
        let nlc = permute(&t, &[0, 2, 1]);
        let back = permute(&nlc, &[0, 2, 1]);

        assert_eq!(back.shape(), &[2, 3, 2]);
        for (a, b) in t.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn test_permute_wrong_length() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = permute(&t, &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "duplicate index")]
    fn test_permute_duplicate_index() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = permute(&t, &[0, 0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_permute_out_of_range() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = permute(&t, &[0, 3]);
    }
}
