//! Convolutional layers for neural networks.
//!
//! Implements 1D and 2D convolutions following the `PyTorch` API.
//!
//! # References
//!
//! - `LeCun`, Y., et al. (1998). Gradient-based learning applied to document
//!   recognition. Proceedings of the IEEE.
//! - He, K., et al. (2015). Delving deep into rectifiers: Surpassing
//!   human-level performance on `ImageNet` classification. ICCV.

pub(crate) mod im2col;
pub(crate) mod layout;
pub(crate) mod permute;

use super::init::{kaiming_uniform, zeros};
use super::module::Module;
use crate::autograd::Tensor;
pub use layout::{ConvDimensionNumbers, ConvLayout, KernelLayout};

/// 1D Convolution layer.
///
/// Applies a 1D convolution over an input signal composed of several input planes.
///
/// # Shape
///
/// - Input: `(N, C_in, L)` where N is batch size, `C_in` is input channels, L is length
/// - Output: `(N, C_out, L_out)` where `L_out` = (L + 2*padding - `kernel_size`) / stride + 1
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Conv1d, Module};
/// use aprender::autograd::Tensor;
///
/// let conv = Conv1d::new(16, 32, 3);  // 16 in channels, 32 out channels, kernel size 3
/// let x = Tensor::randn(&[4, 16, 100]);  // batch of 4, 16 channels, length 100
/// let y = conv.forward(&x);  // [4, 32, 98]
/// ```
pub struct Conv1d {
    /// Weight tensor, shape: [`out_channels`, `in_channels`, `kernel_size`]
    weight: Tensor,
    /// Bias tensor, shape: [`out_channels`], or None
    bias: Option<Tensor>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Data layout for input/output (default: NCL)
    layout: ConvLayout,
    /// Whether to use im2col+GEMM path (default: true)
    use_im2col: bool,
}

impl Conv1d {
    /// Create a new Conv1d layer.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, 0, true)
    }

    /// Create Conv1d with custom options.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    /// * `stride` - Stride of the convolution
    /// * `padding` - Zero-padding added to both sides
    /// * `bias` - If true, adds a learnable bias
    #[must_use]
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        // Kaiming initialization (He et al., 2015)
        let fan_in = in_channels * kernel_size;
        let weight = kaiming_uniform(&[out_channels, in_channels, kernel_size], fan_in, None)
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
            kernel_size,
            stride,
            padding,
            layout: ConvLayout::NCL,
            use_im2col: true,
        }
    }

    /// Create Conv1d with a specific data layout.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    /// * `stride` - Stride of the convolution
    /// * `padding` - Zero-padding added to both sides
    /// * `bias` - If true, adds a learnable bias
    /// * `layout` - Data layout for input/output tensors
    #[must_use]
    pub fn with_layout(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        layout: ConvLayout,
    ) -> Self {
        let mut conv = Self::with_options(in_channels, out_channels, kernel_size, stride, padding, bias);
        conv.layout = layout;
        conv
    }

    /// Create Conv1d with specific stride.
    #[must_use]
    pub fn with_stride(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, stride, 0, true)
    }

    /// Create Conv1d with padding.
    #[must_use]
    pub fn with_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, padding, true)
    }

    /// Get kernel size.
    #[must_use]
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get stride.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get padding.
    #[must_use]
    pub fn padding(&self) -> usize {
        self.padding
    }
}

impl Conv1d {
    /// Naive 5-loop convolution (fallback path).
    fn forward_naive(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch_size, in_channels, in_length) = (shape[0], shape[1], shape[2]);

        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output = vec![0.0; batch_size * self.out_channels * out_length];

        let input_data = input.data();
        let weight_data = self.weight.data();

        for n in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0;

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;

                            let val = if il < self.padding || il >= in_length + self.padding {
                                0.0
                            } else {
                                let actual_il = il - self.padding;
                                input_data[n * in_channels * in_length + ic * in_length + actual_il]
                            };

                            let w_idx = oc * self.in_channels * self.kernel_size
                                + ic * self.kernel_size
                                + k;
                            sum += val * weight_data[w_idx];
                        }
                    }

                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[oc];
                    }

                    output[n * self.out_channels * out_length + oc * out_length + ol] = sum;
                }
            }
        }

        Tensor::new(&output, &[batch_size, self.out_channels, out_length])
    }

    /// im2col + GEMM convolution (fast path via trueno SIMD matmul).
    fn forward_im2col(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch_size, in_channels, in_length) = (shape[0], shape[1], shape[2]);

        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Weight reshaped to [out_channels, in_channels * kernel_size]
        let weight_2d = Tensor::new(
            self.weight.data(),
            &[self.out_channels, self.in_channels * self.kernel_size],
        );

        let input_data = input.data();
        let batch_spatial = in_channels * in_length;

        let mut all_output = Vec::with_capacity(batch_size * self.out_channels * out_length);

        for n in 0..batch_size {
            let batch_input = &input_data[n * batch_spatial..(n + 1) * batch_spatial];

            let (col_data, col_h, col_w) = im2col::im2col_1d(
                batch_input,
                in_channels,
                in_length,
                self.kernel_size,
                self.stride,
                self.padding,
            );

            let col_tensor = Tensor::new(&col_data, &[col_h, col_w]);
            let result = weight_2d.matmul(&col_tensor);

            // result shape: [out_channels, out_length]
            let mut result_data = result.data().to_vec();

            // Add bias
            if let Some(ref bias) = self.bias {
                let bias_data = bias.data();
                for oc in 0..self.out_channels {
                    for ol in 0..out_length {
                        result_data[oc * out_length + ol] += bias_data[oc];
                    }
                }
            }

            all_output.extend_from_slice(&result_data);
        }

        Tensor::new(&all_output, &[batch_size, self.out_channels, out_length])
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            3,
            "Conv1d expects 3D input [N, C, L], got {}D",
            input.ndim()
        );

        // Handle layout: convert to NCL if needed
        let ncl_input = if self.layout == ConvLayout::NLC {
            permute::permute(input, &ConvLayout::NLC.permutation_to(ConvLayout::NCL))
        } else {
            input.clone()
        };

        let shape = ncl_input.shape();
        let in_channels = shape[1];

        assert_eq!(
            in_channels, self.in_channels,
            "Expected {} input channels, got {}",
            self.in_channels, in_channels
        );

        let result = if self.use_im2col {
            self.forward_im2col(&ncl_input)
        } else {
            self.forward_naive(&ncl_input)
        };

        // Convert output back to original layout if needed
        if self.layout == ConvLayout::NLC {
            permute::permute(&result, &ConvLayout::NCL.permutation_to(ConvLayout::NLC))
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

impl std::fmt::Debug for Conv1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv1d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("bias", &self.bias.is_some())
            .field("layout", &self.layout)
            .field("use_im2col", &self.use_im2col)
            .finish_non_exhaustive()
    }
}

/// 2D Convolution layer.
///
/// Applies a 2D convolution over an input image composed of several input planes.
///
/// # Shape
///
/// - Input: `(N, C_in, H, W)` where N is batch, `C_in` is channels, H is height, W is width
/// - Output: `(N, C_out, H_out, W_out)`
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Conv2d, Module};
/// use aprender::autograd::Tensor;
///
/// let conv = Conv2d::new(3, 64, 3);  // 3 in channels (RGB), 64 out channels, 3x3 kernel
/// let x = Tensor::randn(&[4, 3, 32, 32]);  // batch of 4, 3 channels, 32x32 images
/// let y = conv.forward(&x);  // [4, 64, 30, 30]
/// ```
pub struct Conv2d {
    /// Weight tensor, shape: [`out_channels`, `in_channels`, `kernel_h`, `kernel_w`]
    weight: Tensor,
    /// Bias tensor, shape: [`out_channels`], or None
    bias: Option<Tensor>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel height
    kernel_h: usize,
    /// Kernel width
    kernel_w: usize,
    /// Stride height
    stride_h: usize,
    /// Stride width
    stride_w: usize,
    /// Padding height
    padding_h: usize,
    /// Padding width
    padding_w: usize,
    /// Data layout for input/output (default: NCHW)
    layout: ConvLayout,
    /// Whether to use im2col+GEMM path (default: true)
    use_im2col: bool,
}

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
