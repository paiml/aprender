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

use super::init::{kaiming_uniform, zeros};
use super::module::Module;
use crate::autograd::Tensor;

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
        }
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

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            3,
            "Conv1d expects 3D input [N, C, L], got {}D",
            input.ndim()
        );

        let shape = input.shape();
        let (batch_size, in_channels, in_length) = (shape[0], shape[1], shape[2]);

        assert_eq!(
            in_channels, self.in_channels,
            "Expected {} input channels, got {}",
            self.in_channels, in_channels
        );

        // Calculate output length
        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Perform convolution
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

                            // Handle padding
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

                    // Add bias
                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[oc];
                    }

                    output[n * self.out_channels * out_length + oc * out_length + ol] = sum;
                }
            }
        }

        Tensor::new(&output, &[batch_size, self.out_channels, out_length])
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
        }
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

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "Conv2d expects 4D input [N, C, H, W], got {}D",
            input.ndim()
        );

        let shape = input.shape();
        let (batch_size, in_channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        assert_eq!(
            in_channels, self.in_channels,
            "Expected {} input channels, got {}",
            self.in_channels, in_channels
        );

        // Calculate output dimensions
        let out_h = (in_h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;

        // Perform convolution
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

                                    // Handle padding
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

                        // Add bias
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
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_shape() {
        let conv = Conv1d::new(16, 32, 3);
        let x = Tensor::ones(&[4, 16, 100]);
        let y = conv.forward(&x);

        // Output length = (100 - 3) / 1 + 1 = 98
        assert_eq!(y.shape(), &[4, 32, 98]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let conv = Conv1d::with_padding(16, 32, 3, 1);
        let x = Tensor::ones(&[4, 16, 100]);
        let y = conv.forward(&x);

        // Output length = (100 + 2*1 - 3) / 1 + 1 = 100
        assert_eq!(y.shape(), &[4, 32, 100]);
    }

    #[test]
    fn test_conv1d_with_stride() {
        let conv = Conv1d::with_stride(16, 32, 3, 2);
        let x = Tensor::ones(&[4, 16, 100]);
        let y = conv.forward(&x);

        // Output length = (100 - 3) / 2 + 1 = 49
        assert_eq!(y.shape(), &[4, 32, 49]);
    }

    #[test]
    fn test_conv1d_parameters() {
        let conv = Conv1d::new(16, 32, 3);
        let params = conv.parameters();

        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].shape(), &[32, 16, 3]); // weight
        assert_eq!(params[1].shape(), &[32]); // bias
    }

    #[test]
    fn test_conv2d_shape() {
        let conv = Conv2d::new(3, 64, 3);
        let x = Tensor::ones(&[4, 3, 32, 32]);
        let y = conv.forward(&x);

        // Output: (32 - 3) / 1 + 1 = 30
        assert_eq!(y.shape(), &[4, 64, 30, 30]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::with_padding(3, 64, 3, 1);
        let x = Tensor::ones(&[4, 3, 32, 32]);
        let y = conv.forward(&x);

        // Output: (32 + 2 - 3) / 1 + 1 = 32 (same size)
        assert_eq!(y.shape(), &[4, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_with_stride() {
        let conv = Conv2d::with_stride(3, 64, 3, 2);
        let x = Tensor::ones(&[4, 3, 32, 32]);
        let y = conv.forward(&x);

        // Output: (32 - 3) / 2 + 1 = 15
        assert_eq!(y.shape(), &[4, 64, 15, 15]);
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 64, 3);
        let params = conv.parameters();

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[64, 3, 3, 3]); // weight
        assert_eq!(params[1].shape(), &[64]); // bias
    }

    #[test]
    fn test_maxpool1d_shape() {
        let pool = MaxPool1d::new(2);
        let x = Tensor::ones(&[4, 16, 100]);
        let y = pool.forward(&x);

        assert_eq!(y.shape(), &[4, 16, 50]);
    }

    #[test]
    fn test_maxpool2d_shape() {
        let pool = MaxPool2d::new(2);
        let x = Tensor::ones(&[4, 64, 32, 32]);
        let y = pool.forward(&x);

        assert_eq!(y.shape(), &[4, 64, 16, 16]);
    }

    #[test]
    fn test_maxpool2d_values() {
        let pool = MaxPool2d::new(2);
        // Create input with known values
        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        );
        let y = pool.forward(&x);

        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        // Max of 2x2 windows:
        // [1,2,5,6] -> 6, [3,4,7,8] -> 8
        // [9,10,13,14] -> 14, [11,12,15,16] -> 16
        assert_eq!(y.data(), &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avgpool2d_shape() {
        let pool = AvgPool2d::new(2);
        let x = Tensor::ones(&[4, 64, 32, 32]);
        let y = pool.forward(&x);

        assert_eq!(y.shape(), &[4, 64, 16, 16]);
    }

    #[test]
    fn test_avgpool2d_values() {
        let pool = AvgPool2d::new(2);
        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        );
        let y = pool.forward(&x);

        // Avg of 2x2 windows:
        // [1,2,5,6] -> 3.5, [3,4,7,8] -> 5.5
        // [9,10,13,14] -> 11.5, [11,12,15,16] -> 13.5
        assert_eq!(y.data(), &[3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_global_avg_pool2d() {
        let pool = GlobalAvgPool2d::new();
        let x = Tensor::ones(&[2, 64, 7, 7]);
        let y = pool.forward(&x);

        assert_eq!(y.shape(), &[2, 64]);
        // All ones, so average is 1.0
        assert!(y.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_flatten() {
        let flatten = Flatten::new();
        let x = Tensor::ones(&[4, 64, 7, 7]);
        let y = flatten.forward(&x);

        assert_eq!(y.shape(), &[4, 64 * 7 * 7]);
    }

    #[test]
    fn test_flatten_from_dim() {
        let flatten = Flatten::from_dim(2);
        let x = Tensor::ones(&[4, 64, 7, 7]);
        let y = flatten.forward(&x);

        assert_eq!(y.shape(), &[4, 64, 49]);
    }
}
