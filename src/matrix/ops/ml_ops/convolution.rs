//! 2D convolution operations for Matrix

use crate::TruenoError;

use super::super::super::Matrix;

impl Matrix<f32> {
    /// Perform 2D convolution with a kernel
    ///
    /// Applies a 2D convolution operation using "valid" padding (no padding),
    /// resulting in an output smaller than the input.
    ///
    /// # Arguments
    ///
    /// * `kernel` - Convolution kernel (filter) to apply
    ///
    /// # Returns
    ///
    /// Convolved matrix with dimensions:
    /// - rows: `input.rows - kernel.rows + 1`
    /// - cols: `input.cols - kernel.cols + 1`
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if:
    /// - Kernel is larger than input in any dimension
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Matrix;
    ///
    /// // 5x5 input image
    /// let input = Matrix::from_vec(
    ///     5, 5,
    ///     vec![
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 9.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///     ]
    /// )?;
    ///
    /// // 3x3 averaging kernel
    /// let kernel_val = 1.0 / 9.0;
    /// let kernel = Matrix::from_vec(
    ///     3, 3,
    ///     vec![kernel_val; 9]
    /// )?;
    ///
    /// let result = input.convolve2d(&kernel)?;
    /// assert_eq!(result.rows(), 3); // 5 - 3 + 1
    /// assert_eq!(result.cols(), 3);
    /// # Ok(())
    /// # }
    /// ```
    // =========================================================================
    // HOT PATH - PERFORMANCE CRITICAL
    // =========================================================================
    // This function processes millions of elements for typical image sizes.
    // Any changes to the inner loop REQUIRE benchmark verification.
    // =========================================================================
    pub fn convolve2d(&self, kernel: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        // Validate kernel size
        if kernel.rows > self.rows || kernel.cols > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}x{}) larger than input ({}x{})",
                kernel.rows, kernel.cols, self.rows, self.cols
            )));
        }

        // Calculate output dimensions (valid padding)
        let output_rows = self.rows - kernel.rows + 1;
        let output_cols = self.cols - kernel.cols + 1;

        // Initialize output matrix (reuse parent's backend)
        let mut result = Matrix::zeros_with_backend(output_rows, output_cols, self.backend);

        // GPU acceleration for large convolutions
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 10_000;

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if output_rows * output_cols >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuBackend;

                if GpuBackend::is_available() {
                    if let Ok(gpu_result) =
                        self.convolve2d_gpu(kernel, &mut result, output_rows, output_cols)
                    {
                        return Ok(gpu_result);
                    }
                }
            }
        }

        // Scalar baseline implementation - optimized with direct indexing
        let input_data = self.as_slice();
        let kernel_data = kernel.as_slice();
        let result_data = result.data.as_mut_slice();
        let input_cols = self.cols;
        let kernel_cols = kernel.cols;
        let result_cols = output_cols;

        for out_row in 0..output_rows {
            for out_col in 0..output_cols {
                let mut sum = 0.0;

                for k_row in 0..kernel.rows {
                    let in_row = out_row + k_row;
                    let input_row_offset = in_row * input_cols;
                    let kernel_row_offset = k_row * kernel_cols;

                    for k_col in 0..kernel.cols {
                        let in_col = out_col + k_col;
                        sum += input_data[input_row_offset + in_col]
                            * kernel_data[kernel_row_offset + k_col];
                    }
                }

                result_data[out_row * result_cols + out_col] = sum;
            }
        }

        Ok(result)
    }

    /// GPU-accelerated 2D convolution helper
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn convolve2d_gpu(
        &self,
        kernel: &Matrix<f32>,
        result: &mut Matrix<f32>,
        _output_rows: usize,
        _output_cols: usize,
    ) -> Result<Matrix<f32>, TruenoError> {
        use crate::backends::gpu::GpuDevice;

        let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;

        gpu.convolve2d(
            self.as_slice(),
            kernel.as_slice(),
            result.data.as_mut_slice(),
            self.rows,
            self.cols,
            kernel.rows,
            kernel.cols,
        )
        .map_err(TruenoError::InvalidInput)?;

        Ok(result.clone())
    }
}

#[cfg(test)]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod gpu_tests {
    use crate::Matrix;

    /// Helper: compute expected 2D convolution (scalar reference) for verification
    fn reference_convolve2d(
        input: &[f32],
        kernel: &[f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Vec<f32> {
        let output_rows = input_rows - kernel_rows + 1;
        let output_cols = input_cols - kernel_cols + 1;
        let mut result = vec![0.0f32; output_rows * output_cols];

        for out_row in 0..output_rows {
            for out_col in 0..output_cols {
                let mut sum = 0.0;
                for k_row in 0..kernel_rows {
                    for k_col in 0..kernel_cols {
                        sum += input[(out_row + k_row) * input_cols + (out_col + k_col)]
                            * kernel[k_row * kernel_cols + k_col];
                    }
                }
                result[out_row * output_cols + out_col] = sum;
            }
        }
        result
    }

    /// Test convolve2d_gpu via public API with input large enough to exceed
    /// GPU_THRESHOLD (output_rows * output_cols >= 10_000).
    /// Uses a 102x102 input with a 3x3 kernel => 100x100 = 10,000 output elements.
    #[test]
    fn test_convolve2d_gpu_identity_kernel() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_convolve2d_gpu_identity_kernel");
            return;
        }

        // 102x102 input, 3x3 kernel => 100x100 = 10,000 output (meets GPU_THRESHOLD)
        let input_rows = 102;
        let input_cols = 102;
        let kernel_rows = 3;
        let kernel_cols = 3;

        // Fill input with sequential values for easy verification
        let input_data: Vec<f32> =
            (0..input_rows * input_cols).map(|i| (i % 100) as f32 * 0.1).collect();

        // Identity-like kernel: center element = 1.0, rest = 0.0
        let mut kernel_data = vec![0.0f32; kernel_rows * kernel_cols];
        kernel_data[4] = 1.0; // center of 3x3

        let input = Matrix::from_vec(input_rows, input_cols, input_data.clone())
            .expect("valid input matrix");
        let kernel = Matrix::from_vec(kernel_rows, kernel_cols, kernel_data.clone())
            .expect("valid kernel matrix");

        let result = input.convolve2d(&kernel).expect("convolve2d should succeed");

        assert_eq!(result.rows(), 100);
        assert_eq!(result.cols(), 100);

        // With identity kernel, output[r][c] = input[r+1][c+1]
        let expected = reference_convolve2d(
            &input_data,
            &kernel_data,
            input_rows,
            input_cols,
            kernel_rows,
            kernel_cols,
        );

        for i in 0..expected.len() {
            assert!(
                (result.as_slice()[i] - expected[i]).abs() < 1e-3,
                "Mismatch at index {}: gpu={}, expected={}",
                i,
                result.as_slice()[i],
                expected[i]
            );
        }
    }

    /// Test convolve2d_gpu with a uniform averaging kernel on a large input.
    #[test]
    fn test_convolve2d_gpu_averaging_kernel() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_convolve2d_gpu_averaging_kernel");
            return;
        }

        // 105x105 input, 5x5 kernel => 101x101 = 10,201 output (exceeds GPU_THRESHOLD)
        let input_rows = 105;
        let input_cols = 105;
        let kernel_rows = 5;
        let kernel_cols = 5;

        // All-ones input
        let input_data = vec![1.0f32; input_rows * input_cols];

        // Averaging kernel: each element = 1/25
        let kernel_data = vec![1.0f32 / 25.0; kernel_rows * kernel_cols];

        let input =
            Matrix::from_vec(input_rows, input_cols, input_data).expect("valid input matrix");
        let kernel =
            Matrix::from_vec(kernel_rows, kernel_cols, kernel_data).expect("valid kernel matrix");

        let result = input.convolve2d(&kernel).expect("convolve2d should succeed");

        assert_eq!(result.rows(), 101);
        assert_eq!(result.cols(), 101);

        // With all-ones input and averaging kernel, every output should be ~1.0
        for i in 0..result.as_slice().len() {
            assert!(
                (result.as_slice()[i] - 1.0).abs() < 1e-3,
                "Mismatch at index {}: gpu={}, expected=1.0",
                i,
                result.as_slice()[i]
            );
        }
    }

    /// Test convolve2d_gpu with a gradient input and edge-detection kernel.
    #[test]
    fn test_convolve2d_gpu_edge_detection() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_convolve2d_gpu_edge_detection");
            return;
        }

        // 103x103 input, 3x3 kernel => 101x101 = 10,201 output
        let input_rows = 103;
        let input_cols = 103;
        let kernel_rows = 3;
        let kernel_cols = 3;

        // Horizontal gradient: each row is constant, increases per row
        let input_data: Vec<f32> =
            (0..input_rows * input_cols).map(|i| (i / input_cols) as f32).collect();

        // Vertical edge detection kernel (Sobel-like simplified)
        let kernel_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];

        let input = Matrix::from_vec(input_rows, input_cols, input_data.clone())
            .expect("valid input matrix");
        let kernel = Matrix::from_vec(kernel_rows, kernel_cols, kernel_data.clone())
            .expect("valid kernel matrix");

        let result = input.convolve2d(&kernel).expect("convolve2d should succeed");

        // Verify against scalar reference
        let expected = reference_convolve2d(
            &input_data,
            &kernel_data,
            input_rows,
            input_cols,
            kernel_rows,
            kernel_cols,
        );

        assert_eq!(result.as_slice().len(), expected.len());

        // Sample verification (check first 100 elements and last 100)
        for i in 0..100.min(expected.len()) {
            assert!(
                (result.as_slice()[i] - expected[i]).abs() < 1e-2,
                "Mismatch at index {}: gpu={}, expected={}",
                i,
                result.as_slice()[i],
                expected[i]
            );
        }
        let len = expected.len();
        for i in (len.saturating_sub(100))..len {
            assert!(
                (result.as_slice()[i] - expected[i]).abs() < 1e-2,
                "Mismatch at index {}: gpu={}, expected={}",
                i,
                result.as_slice()[i],
                expected[i]
            );
        }
    }

    /// Test convolve2d_gpu directly via the private helper method.
    #[test]
    fn test_convolve2d_gpu_direct() {
        use crate::backends::gpu::GpuDevice;

        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test_convolve2d_gpu_direct");
            return;
        }

        // 4x4 input, 2x2 kernel -> 3x3 output (test the private method directly)
        let input_rows = 4;
        let input_cols = 4;
        let kernel_rows = 2;
        let kernel_cols = 2;
        let output_rows = 3;
        let output_cols = 3;

        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let kernel_data = vec![1.0, 0.0, 0.0, 1.0]; // diagonal sum kernel

        let input =
            Matrix::from_vec(input_rows, input_cols, input_data).expect("valid input matrix");
        let kernel =
            Matrix::from_vec(kernel_rows, kernel_cols, kernel_data).expect("valid kernel matrix");

        let mut result = Matrix::zeros(output_rows, output_cols);

        let gpu_result = input
            .convolve2d_gpu(&kernel, &mut result, output_rows, output_cols)
            .expect("convolve2d_gpu should succeed");

        // output[0,0] = input[0,0]*1 + input[0,1]*0 + input[1,0]*0 + input[1,1]*1 = 1+6 = 7
        assert!(
            (gpu_result.as_slice()[0] - 7.0).abs() < 1e-3,
            "Expected 7.0, got {}",
            gpu_result.as_slice()[0]
        );
        // output[0,1] = input[0,1]*1 + input[1,2]*1 = 2+7 = 9
        assert!(
            (gpu_result.as_slice()[1] - 9.0).abs() < 1e-3,
            "Expected 9.0, got {}",
            gpu_result.as_slice()[1]
        );
    }
}
