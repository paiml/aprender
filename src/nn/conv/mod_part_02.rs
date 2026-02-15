
impl Conv2d {
    /// Create a new Conv2d layer with square kernel.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the square convolving kernel
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            true,
        )
    }

    /// Create Conv2d with custom options.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - (height, width) of the kernel
    /// * `stride` - (height, width) stride
    /// * `padding` - (height, width) padding
    /// * `bias` - If true, adds a learnable bias
    #[must_use]
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;

        // Kaiming initialization
        let fan_in = in_channels * kernel_h * kernel_w;
        let weight = kaiming_uniform(
            &[out_channels, in_channels, kernel_h, kernel_w],
            fan_in,
            None,
        )
        .requires_grad();

        let bias_tensor = if bias {
            Some(zeros(&[out_channels]).requires_grad())
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h: stride.0,
            stride_w: stride.1,
            padding_h: padding.0,
            padding_w: padding.1,
            layout: ConvLayout::NCHW,
            use_im2col: true,
        }
    }

    /// Create Conv2d with a specific data layout.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - (height, width) of the kernel
    /// * `stride` - (height, width) stride
    /// * `padding` - (height, width) padding
    /// * `bias` - If true, adds a learnable bias
    /// * `layout` - Data layout for input/output tensors
    #[must_use]
    pub fn with_layout(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        layout: ConvLayout,
    ) -> Self {
        let mut conv = Self::with_options(in_channels, out_channels, kernel_size, stride, padding, bias);
        conv.layout = layout;
        conv
    }

    /// Create Conv2d with stride.
    #[must_use]
    pub fn with_stride(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (0, 0),
            true,
        )
    }

    /// Create Conv2d with padding.
    #[must_use]
    pub fn with_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (padding, padding),
            true,
        )
    }

    /// Get kernel size as (height, width).
    #[must_use]
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel_h, self.kernel_w)
    }

    /// Get stride as (height, width).
    #[must_use]
    pub fn stride(&self) -> (usize, usize) {
        (self.stride_h, self.stride_w)
    }

    /// Get padding as (height, width).
    #[must_use]
    pub fn padding(&self) -> (usize, usize) {
        (self.padding_h, self.padding_w)
    }
}

impl Conv2d {
    /// Naive 7-loop convolution (fallback path).
    fn forward_naive(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch_size, in_channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (in_h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;

        let mut output = vec![0.0; batch_size * self.out_channels * out_h * out_w];

        let input_data = input.data();
        let weight_data = self.weight.data();

        for n in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_h {
                                for kw in 0..self.kernel_w {
                                    let ih = oh * self.stride_h + kh;
                                    let iw = ow * self.stride_w + kw;

                                    let val = if ih < self.padding_h
                                        || ih >= in_h + self.padding_h
                                        || iw < self.padding_w
                                        || iw >= in_w + self.padding_w
                                    {
                                        0.0
                                    } else {
                                        let actual_ih = ih - self.padding_h;
                                        let actual_iw = iw - self.padding_w;
                                        input_data[n * in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + actual_ih * in_w
                                            + actual_iw]
                                    };

                                    let w_idx =
                                        oc * self.in_channels * self.kernel_h * self.kernel_w
                                            + ic * self.kernel_h * self.kernel_w
                                            + kh * self.kernel_w
                                            + kw;
                                    sum += val * weight_data[w_idx];
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            sum += bias.data()[oc];
                        }

                        output[n * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow] = sum;
                    }
                }
            }
        }

        Tensor::new(&output, &[batch_size, self.out_channels, out_h, out_w])
    }

    /// im2col + GEMM convolution (fast path via trueno SIMD matmul).
    fn forward_im2col(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch_size, in_channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (in_h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;

        // Weight reshaped to [out_channels, in_channels * kH * kW]
        let weight_2d = Tensor::new(
            self.weight.data(),
            &[self.out_channels, self.in_channels * self.kernel_h * self.kernel_w],
        );

        let input_data = input.data();
        let batch_spatial = in_channels * in_h * in_w;

        let mut all_output = Vec::with_capacity(batch_size * self.out_channels * out_h * out_w);

        for n in 0..batch_size {
            let batch_input = &input_data[n * batch_spatial..(n + 1) * batch_spatial];

            let (col_data, col_h, col_w) = im2col::im2col_2d(
                batch_input,
                in_channels,
                in_h,
                in_w,
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
                self.padding_h,
                self.padding_w,
            );

            let col_tensor = Tensor::new(&col_data, &[col_h, col_w]);
            let result = weight_2d.matmul(&col_tensor);

            // result shape: [out_channels, out_h * out_w]
            let mut result_data = result.data().to_vec();

            // Add bias
            if let Some(ref bias) = self.bias {
                let bias_data = bias.data();
                let spatial = out_h * out_w;
                for oc in 0..self.out_channels {
                    for s in 0..spatial {
                        result_data[oc * spatial + s] += bias_data[oc];
                    }
                }
            }

            all_output.extend_from_slice(&result_data);
        }

        Tensor::new(&all_output, &[batch_size, self.out_channels, out_h, out_w])
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "Conv2d expects 4D input [N, C, H, W], got {}D",
            input.ndim()
        );

        // Handle layout: convert to NCHW if needed
        let nchw_input = if self.layout == ConvLayout::NHWC {
            permute::permute(input, &ConvLayout::NHWC.permutation_to(ConvLayout::NCHW))
        } else {
            input.clone()
        };

        let shape = nchw_input.shape();
        let in_channels = shape[1];

        assert_eq!(
            in_channels, self.in_channels,
            "Expected {} input channels, got {}",
            self.in_channels, in_channels
        );

        let result = if self.use_im2col {
            self.forward_im2col(&nchw_input)
        } else {
            self.forward_naive(&nchw_input)
        };

        // Convert output back to original layout if needed
        if self.layout == ConvLayout::NHWC {
            permute::permute(&result, &ConvLayout::NCHW.permutation_to(ConvLayout::NHWC))
        } else {
            result
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match &self.bias {
            Some(b) => vec![&self.weight, b],
            None => vec![&self.weight],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        match &mut self.bias {
            Some(b) => vec![&mut self.weight, b],
            None => vec![&mut self.weight],
        }
    }
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &(self.kernel_h, self.kernel_w))
            .field("stride", &(self.stride_h, self.stride_w))
            .field("padding", &(self.padding_h, self.padding_w))
            .field("bias", &self.bias.is_some())
            .field("layout", &self.layout)
            .field("use_im2col", &self.use_im2col)
            .finish_non_exhaustive()
    }
}

/// Max Pooling 1D.
///
/// Applies max pooling over a 1D input signal.
///
/// # Shape
///
/// - Input: `(N, C, L)`
/// - Output: `(N, C, L_out)` where `L_out` = (L - `kernel_size`) / stride + 1
#[derive(Debug)]
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool1d {
    /// Create a new `MaxPool1d` layer.
    #[must_use]
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
        }
    }

    /// Create `MaxPool1d` with custom stride.
    #[must_use]
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 3, "MaxPool1d expects 3D input [N, C, L]");

        let shape = input.shape();
        let (batch_size, channels, in_length) = (shape[0], shape[1], shape[2]);

        let out_length = (in_length - self.kernel_size) / self.stride + 1;
        let mut output = vec![f32::NEG_INFINITY; batch_size * channels * out_length];

        let input_data = input.data();

        for n in 0..batch_size {
            for c in 0..channels {
                for ol in 0..out_length {
                    let mut max_val = f32::NEG_INFINITY;

                    for k in 0..self.kernel_size {
                        let il = ol * self.stride + k;
                        let val = input_data[n * channels * in_length + c * in_length + il];
                        max_val = max_val.max(val);
                    }

                    output[n * channels * out_length + c * out_length + ol] = max_val;
                }
            }
        }

        Tensor::new(&output, &[batch_size, channels, out_length])
    }
}

/// Max Pooling 2D.
///
/// Applies max pooling over a 2D input signal (image).
///
/// # Shape
///
/// - Input: `(N, C, H, W)`
/// - Output: `(N, C, H_out, W_out)`
#[derive(Debug)]
pub struct MaxPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}
