
impl MaxPool2d {
    /// Create a new `MaxPool2d` layer with square kernel.
    #[must_use]
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: kernel_size,
            stride_w: kernel_size,
        }
    }

    /// Create `MaxPool2d` with custom stride.
    #[must_use]
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
        }
    }

    /// Create `MaxPool2d` with rectangular kernel and stride.
    #[must_use]
    pub fn with_options(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: stride.0,
            stride_w: stride.1,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "MaxPool2d expects 4D input [N, C, H, W]");

        let shape = input.shape();
        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (in_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w - self.kernel_w) / self.stride_w + 1;

        let mut output = vec![f32::NEG_INFINITY; batch_size * channels * out_h * out_w];

        let input_data = input.data();

        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh;
                                let iw = ow * self.stride_w + kw;
                                let val = input_data
                                    [n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                                max_val = max_val.max(val);
                            }
                        }

                        output
                            [n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] =
                            max_val;
                    }
                }
            }
        }

        Tensor::new(&output, &[batch_size, channels, out_h, out_w])
    }
}

/// Average Pooling 2D.
///
/// Applies average pooling over a 2D input signal.
#[derive(Debug)]
pub struct AvgPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl AvgPool2d {
    /// Create a new `AvgPool2d` layer with square kernel.
    #[must_use]
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: kernel_size,
            stride_w: kernel_size,
        }
    }

    /// Create `AvgPool2d` with custom stride.
    #[must_use]
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "AvgPool2d expects 4D input [N, C, H, W]");

        let shape = input.shape();
        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (in_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w - self.kernel_w) / self.stride_w + 1;

        let mut output = vec![0.0; batch_size * channels * out_h * out_w];
        let kernel_area = (self.kernel_h * self.kernel_w) as f32;

        let input_data = input.data();

        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh;
                                let iw = ow * self.stride_w + kw;
                                sum += input_data
                                    [n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                            }
                        }

                        output
                            [n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] =
                            sum / kernel_area;
                    }
                }
            }
        }

        Tensor::new(&output, &[batch_size, channels, out_h, out_w])
    }
}

/// Global Average Pooling 2D.
///
/// Pools over the entire spatial dimension, outputting one value per channel.
///
/// # Shape
///
/// - Input: `(N, C, H, W)`
/// - Output: `(N, C)`
#[derive(Debug, Default)]
pub struct GlobalAvgPool2d;

impl GlobalAvgPool2d {
    /// Create a new `GlobalAvgPool2d` layer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Module for GlobalAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "GlobalAvgPool2d expects 4D input [N, C, H, W]"
        );

        let shape = input.shape();
        let (batch_size, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial_size = (h * w) as f32;

        let mut output = vec![0.0; batch_size * channels];
        let input_data = input.data();

        for n in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0;
                for i in 0..h {
                    for j in 0..w {
                        sum += input_data[n * channels * h * w + c * h * w + i * w + j];
                    }
                }
                output[n * channels + c] = sum / spatial_size;
            }
        }

        Tensor::new(&output, &[batch_size, channels])
    }
}

/// Flatten layer.
///
/// Flattens contiguous dimensions of a tensor.
///
/// # Shape
///
/// - Input: `(N, *dims)`
/// - Output: `(N, prod(dims))`
#[derive(Debug, Default)]
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    /// Create a new Flatten layer.
    ///
    /// By default, flattens from dimension 1 onwards (preserving batch).
    #[must_use]
    pub fn new() -> Self {
        Self { start_dim: 1 }
    }

    /// Create Flatten with custom start dimension.
    #[must_use]
    pub fn from_dim(start_dim: usize) -> Self {
        Self { start_dim }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();

        if shape.len() <= self.start_dim + 1 {
            return input.clone();
        }

        let mut new_shape: Vec<usize> = shape[..self.start_dim].to_vec();
        let flattened_size: usize = shape[self.start_dim..].iter().product();
        new_shape.push(flattened_size);

        Tensor::new(input.data(), &new_shape)
    }
}

#[cfg(test)]
mod tests;
