//! Pooling operations (max pool, average pool) for Matrix

use crate::TruenoError;

use super::super::super::Matrix;

impl Matrix<f32> {
    /// 2D Max Pooling operation for CNN downsampling
    ///
    /// Applies max pooling over a 2D input tensor with specified kernel size and stride.
    ///
    /// # Arguments
    /// * `kernel` - (kernel_height, kernel_width) pooling window size
    /// * `stride` - (stride_height, stride_width) step size
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::matrix::Matrix;
    /// let input = Matrix::from_vec(4, 4, vec![
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// ])?;
    /// let pooled = input.max_pool2d((2, 2), (2, 2))?;
    /// assert_eq!(pooled.shape(), (2, 2));
    /// assert_eq!(pooled.get(0, 0), Some(&6.0));  // max of [1,2,5,6]
    /// assert_eq!(pooled.get(1, 1), Some(&16.0)); // max of [11,12,15,16]
    /// # Ok(())
    /// # }
    /// ```
    pub fn max_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Matrix<f32>, TruenoError> {
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(TruenoError::InvalidInput(
                "Kernel and stride dimensions must be positive".into(),
            ));
        }

        if kh > self.rows || kw > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh, kw, self.rows, self.cols
            )));
        }

        let out_h = (self.rows - kh) / sh + 1;
        let out_w = (self.cols - kw) / sw + 1;
        let mut result = Matrix::new(out_h, out_w);

        for i in 0..out_h {
            for j in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let val = self.data[(i * sh + ki) * self.cols + (j * sw + kj)];
                        max_val = max_val.max(val);
                    }
                }
                result.data[i * out_w + j] = max_val;
            }
        }

        Ok(result)
    }

    /// 2D Average Pooling operation for CNN downsampling
    ///
    /// Applies average pooling over a 2D input tensor with specified kernel size and stride.
    ///
    /// # Arguments
    /// * `kernel` - (kernel_height, kernel_width) pooling window size
    /// * `stride` - (stride_height, stride_width) step size
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::matrix::Matrix;
    /// let input = Matrix::from_vec(4, 4, vec![
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// ])?;
    /// let pooled = input.avg_pool2d((2, 2), (2, 2))?;
    /// assert_eq!(pooled.shape(), (2, 2));
    /// assert!((pooled.get(0, 0).unwrap_or(&0.0) - 3.5).abs() < 1e-5);  // avg of [1,2,5,6]
    /// # Ok(())
    /// # }
    /// ```
    pub fn avg_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Matrix<f32>, TruenoError> {
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(TruenoError::InvalidInput(
                "Kernel and stride dimensions must be positive".into(),
            ));
        }

        if kh > self.rows || kw > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh, kw, self.rows, self.cols
            )));
        }

        let out_h = (self.rows - kh) / sh + 1;
        let out_w = (self.cols - kw) / sw + 1;
        let kernel_size = (kh * kw) as f32;
        let mut result = Matrix::new(out_h, out_w);

        for i in 0..out_h {
            for j in 0..out_w {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += self.data[(i * sh + ki) * self.cols + (j * sw + kj)];
                    }
                }
                result.data[i * out_w + j] = sum / kernel_size;
            }
        }

        Ok(result)
    }
}
