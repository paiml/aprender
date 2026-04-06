use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_convolve2d_output_size(
        input_rows in 3usize..20,
        input_cols in 3usize..20,
        kernel_rows in 1usize..5,
        kernel_cols in 1usize..5,
    ) {
        // Property: Output size is always (input - kernel + 1) for valid padding
        if kernel_rows <= input_rows && kernel_cols <= input_cols {
            let input = Matrix::from_vec(input_rows, input_cols, vec![1.0; input_rows * input_cols]).unwrap();
            let kernel = Matrix::from_vec(kernel_rows, kernel_cols, vec![1.0; kernel_rows * kernel_cols]).unwrap();

            let result = input.convolve2d(&kernel).unwrap();

            prop_assert_eq!(result.rows(), input_rows - kernel_rows + 1);
            prop_assert_eq!(result.cols(), input_cols - kernel_cols + 1);
        }
    }

    #[test]
    fn test_convolve2d_identity_kernel(
        input_rows in 3usize..10,
        input_cols in 3usize..10,
        values in prop::collection::vec(-100.0f32..100.0, 9..100)
    ) {
        // Property: 1x1 identity kernel preserves input
        if values.len() >= input_rows * input_cols {
            let data: Vec<f32> = values.iter().take(input_rows * input_cols).copied().collect();
            let input = Matrix::from_vec(input_rows, input_cols, data.clone()).unwrap();
            let kernel = Matrix::from_vec(1, 1, vec![1.0]).unwrap();

            let result = input.convolve2d(&kernel).unwrap();

            prop_assert_eq!(result.rows(), input_rows);
            prop_assert_eq!(result.cols(), input_cols);
            prop_assert_eq!(result.as_slice(), input.as_slice());
        }
    }

    #[test]
    fn test_convolve2d_zero_kernel(
        input_rows in 3usize..10,
        input_cols in 3usize..10,
        kernel_rows in 1usize..4,
        kernel_cols in 1usize..4,
    ) {
        // Property: Zero kernel produces zero output
        if kernel_rows <= input_rows && kernel_cols <= input_cols {
            let input = Matrix::from_vec(input_rows, input_cols, vec![5.0; input_rows * input_cols]).unwrap();
            let kernel = Matrix::from_vec(kernel_rows, kernel_cols, vec![0.0; kernel_rows * kernel_cols]).unwrap();

            let result = input.convolve2d(&kernel).unwrap();

            for &val in result.as_slice() {
                prop_assert!((val - 0.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_convolve2d_scalar_multiplication(
        input_rows in 3usize..10,
        input_cols in 3usize..10,
        scalar in -10.0f32..10.0,
    ) {
        // Property: Convolving with scalar * kernel = scalar * (convolve with kernel)
        let input = Matrix::from_vec(input_rows, input_cols, vec![2.0; input_rows * input_cols]).unwrap();
        let kernel = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();
        let kernel_scaled = Matrix::from_vec(3, 3, vec![scalar; 9]).unwrap();

        let result1 = input.convolve2d(&kernel).unwrap();
        let result2 = input.convolve2d(&kernel_scaled).unwrap();

        for (v1, v2) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
            prop_assert!((v1 * scalar - v2).abs() < 1e-3);
        }
    }
}
