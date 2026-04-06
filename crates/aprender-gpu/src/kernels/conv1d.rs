//! WAPR-PERF-012: 1D Convolution Kernel for Whisper Frontend
//!
//! GPU implementation of Conv1d for audio processing.
//! Target: Move Whisper's 588ms CPU conv to GPU (<50ms target).
//!
//! # Whisper Conv Configuration
//!
//! - conv1: 80 → 384 channels, kernel=3, stride=1, padding=1
//! - conv2: 384 → 384 channels, kernel=3, stride=2, padding=1
//!
//! # Parallelization Strategy
//!
//! Grid: (ceil(out_seq_len/32), ceil(out_channels/8), 1)
//! Block: (32, 8, 1) = 256 threads
//!
//! Each thread computes one output element.

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// 1D Convolution Kernel Configuration
#[derive(Debug, Clone)]
pub struct Conv1dKernel {
    /// Input channels
    pub in_channels: u32,
    /// Output channels
    pub out_channels: u32,
    /// Kernel size
    pub kernel_size: u32,
    /// Stride
    pub stride: u32,
    /// Padding
    pub padding: u32,
    /// Include bias
    pub bias: bool,
    /// Threads per block (x dimension)
    pub block_x: u32,
    /// Threads per block (y dimension - output channels)
    pub block_y: u32,
}

impl Conv1dKernel {
    /// Create Conv1d kernel for Whisper conv1 (mel → hidden)
    ///
    /// 80 → 384 channels, kernel=3, stride=1, padding=1
    #[must_use]
    pub fn whisper_conv1() -> Self {
        Self {
            in_channels: 80,
            out_channels: 384,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            bias: true,
            block_x: 32,
            block_y: 8,
        }
    }

    /// Create Conv1d kernel for Whisper conv2 (hidden → hidden with stride)
    ///
    /// 384 → 384 channels, kernel=3, stride=2, padding=1
    #[must_use]
    pub fn whisper_conv2() -> Self {
        Self {
            in_channels: 384,
            out_channels: 384,
            kernel_size: 3,
            stride: 2,
            padding: 1,
            bias: true,
            block_x: 32,
            block_y: 8,
        }
    }

    /// Create custom Conv1d kernel
    #[must_use]
    pub fn new(
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias: true,
            block_x: 32,
            block_y: 8,
        }
    }

    /// Disable bias
    #[must_use]
    pub const fn without_bias(mut self) -> Self {
        self.bias = false;
        self
    }

    /// Set block dimensions
    #[must_use]
    pub const fn with_block_dims(mut self, block_x: u32, block_y: u32) -> Self {
        self.block_x = block_x;
        self.block_y = block_y;
        self
    }
}

impl Kernel for Conv1dKernel {
    fn name(&self) -> &str {
        "conv1d"
    }

    fn build_ptx(&self) -> PtxKernel {
        let in_channels = self.in_channels;
        let out_channels = self.out_channels;
        let kernel_size = self.kernel_size;
        let stride = self.stride;
        let padding = self.padding;
        let has_bias = self.bias;

        // Parameters:
        // - input_ptr: [seq_len, in_channels] row-major
        // - weight_ptr: [out_channels, in_channels, kernel_size] row-major
        // - bias_ptr: [out_channels] (optional)
        // - output_ptr: [out_seq_len, out_channels] row-major
        // - seq_len: input sequence length
        //
        // Grid: (ceil(out_seq_len/block_x), ceil(out_channels/block_y), 1)
        // Block: (block_x, block_y, 1)
        //
        // Each thread computes output[out_pos, out_ch]:
        //   sum = bias[out_ch]
        //   for k in 0..kernel_size:
        //     in_pos = out_pos * stride + k - padding
        //     if 0 <= in_pos < seq_len:
        //       for in_ch in 0..in_channels:
        //         sum += weight[out_ch, in_ch, k] * input[in_pos, in_ch]
        //   output[out_pos, out_ch] = gelu(sum)

        PtxKernel::new("conv1d")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "weight_ptr")
            .param(PtxType::U64, "bias_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "seq_len")
            .build(move |ctx| {
                // Thread indices
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);

                // Global output position
                let out_pos = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);
                let out_ch = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);

                // Load parameters
                let seq_len_param = ctx.load_param_u32("seq_len");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let weight_ptr = ctx.load_param_u64("weight_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Calculate output sequence length: (seq_len + 2*padding - kernel_size) / stride + 1
                let two_padding = ctx.mov_u32_imm(2 * padding);
                let seq_plus_pad = ctx.add_u32_reg(seq_len_param, two_padding);
                let kernel_u32 = ctx.mov_u32_imm(kernel_size);
                let numerator = ctx.sub_u32_reg(seq_plus_pad, kernel_u32);
                let div_result = ctx.div_u32(numerator, stride);
                let one_imm = 1u32;
                let out_seq_len = ctx.add_u32(div_result, one_imm);

                // Bounds check
                let out_channels_imm = ctx.mov_u32_imm(out_channels);
                let out_ch_oob = ctx.setp_ge_u32(out_ch, out_channels_imm);
                ctx.branch_if(out_ch_oob, "exit");

                let out_pos_oob = ctx.setp_ge_u32(out_pos, out_seq_len);
                ctx.branch_if(out_pos_oob, "exit");

                // Initialize sum with bias (if enabled)
                let sum = if has_bias {
                    let bias_ptr = ctx.load_param_u64("bias_ptr");
                    let four = ctx.mov_u32_imm(4);
                    let bias_offset = ctx.mul_wide_u32_reg(out_ch, four);
                    let bias_addr = ctx.add_u64(bias_ptr, bias_offset);
                    ctx.ld_global_f32(bias_addr)
                } else {
                    ctx.mov_f32_imm(0.0)
                };

                // Convolution loop
                // for k in 0..kernel_size
                let k = ctx.mov_u32_imm(0);
                let stride_u32 = ctx.mov_u32_imm(stride);

                ctx.label("kernel_loop");
                let k_done = ctx.setp_ge_u32(k, kernel_u32);
                ctx.branch_if(k_done, "apply_activation");

                // Calculate input position: in_pos = out_pos * stride + k - padding
                let out_pos_times_stride = ctx.mul_lo_u32(out_pos, stride_u32);
                let pos_plus_k = ctx.add_u32_reg(out_pos_times_stride, k);
                let padding_u32 = ctx.mov_u32_imm(padding);

                // Check if in_pos would be negative (pos_plus_k < padding)
                let before_start = ctx.setp_lt_u32(pos_plus_k, padding_u32);
                ctx.branch_if(before_start, "skip_kernel_pos");

                // in_pos = pos_plus_k - padding
                let in_pos = ctx.sub_u32_reg(pos_plus_k, padding_u32);

                // Check if in_pos >= seq_len
                let after_end = ctx.setp_ge_u32(in_pos, seq_len_param);
                ctx.branch_if(after_end, "skip_kernel_pos");

                // Inner loop: for in_ch in 0..in_channels
                let in_ch = ctx.mov_u32_imm(0);
                let in_channels_u32 = ctx.mov_u32_imm(in_channels);

                ctx.label("channel_loop");
                let ch_done = ctx.setp_ge_u32(in_ch, in_channels_u32);
                ctx.branch_if(ch_done, "channel_loop_end");

                // Load input[in_pos, in_ch]
                // input index = in_pos * in_channels + in_ch
                let input_row_offset = ctx.mul_wide_u32_reg(in_pos, in_channels_u32);
                let input_col_offset = ctx.cvt_u64_u32(in_ch);
                let input_idx = ctx.add_u64(input_row_offset, input_col_offset);
                let input_byte_offset = ctx.mul_u64(input_idx, 4);
                let input_addr = ctx.add_u64(input_ptr, input_byte_offset);
                let input_val = ctx.ld_global_f32(input_addr);

                // Load weight[out_ch, in_ch, k]
                // weight index = out_ch * (in_channels * kernel_size) + in_ch * kernel_size + k
                let in_ch_k = ctx.mov_u32_imm(in_channels * kernel_size);
                let out_ch_stride = ctx.mul_wide_u32_reg(out_ch, in_ch_k);
                let in_ch_offset = ctx.mul_wide_u32_reg(in_ch, kernel_u32);
                let k_offset = ctx.cvt_u64_u32(k);
                let weight_idx = ctx.add_u64(out_ch_stride, in_ch_offset);
                let weight_idx = ctx.add_u64(weight_idx, k_offset);
                let weight_byte_offset = ctx.mul_u64(weight_idx, 4);
                let weight_addr = ctx.add_u64(weight_ptr, weight_byte_offset);
                let weight_val = ctx.ld_global_f32(weight_addr);

                // sum += input_val * weight_val (FMA)
                ctx.fma_f32_inplace(sum, input_val, weight_val);

                // in_ch++
                ctx.add_u32_inplace(in_ch, 1);
                ctx.branch("channel_loop");

                ctx.label("channel_loop_end");

                ctx.label("skip_kernel_pos");
                // k++
                ctx.add_u32_inplace(k, 1);
                ctx.branch("kernel_loop");

                // Apply GELU activation
                ctx.label("apply_activation");
                // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                // Simplified: tanh approximation using fast formula
                // Use x * sigmoid(1.702 * x) approximation for speed
                let gelu_coef = ctx.mov_f32_imm(1.702);
                let _half = ctx.mov_f32_imm(0.5); // Reserved for future GELU variants
                let one_f32 = ctx.mov_f32_imm(1.0);

                // sigmoid_arg = 1.702 * sum
                let sigmoid_arg = ctx.mul_f32(gelu_coef, sum);

                // sigmoid(x) = 1 / (1 + exp(-x))
                // Use approximation: neg_arg = -sigmoid_arg
                let neg_arg = ctx.neg_f32(sigmoid_arg);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E); // log2(e)
                let scaled_neg = ctx.mul_f32(neg_arg, log2e);
                let exp_neg = ctx.ex2_f32(scaled_neg); // exp(x) ≈ 2^(x * log2(e))
                let one_plus_exp = ctx.add_f32(one_f32, exp_neg);
                let sigmoid = ctx.div_f32(one_f32, one_plus_exp);

                // gelu = sum * sigmoid
                let gelu_result = ctx.mul_f32(sum, sigmoid);

                // Store output[out_pos, out_ch]
                // output index = out_pos * out_channels + out_ch
                let out_channels_u32 = ctx.mov_u32_imm(out_channels);
                let output_row_offset = ctx.mul_wide_u32_reg(out_pos, out_channels_u32);
                let output_col_offset = ctx.cvt_u64_u32(out_ch);
                let output_idx = ctx.add_u64(output_row_offset, output_col_offset);
                let output_byte_offset = ctx.mul_u64(output_idx, 4);
                let output_addr = ctx.add_u64(output_ptr, output_byte_offset);
                ctx.st_global_f32(output_addr, gelu_result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Fused Conv1d + GELU kernel with shared memory tiling
///
/// Uses shared memory to cache input tiles for better memory access patterns.
/// Each block processes a tile of output positions with all output channels.
#[derive(Debug, Clone)]
pub struct TiledConv1dKernel {
    /// Input channels
    pub in_channels: u32,
    /// Output channels
    pub out_channels: u32,
    /// Kernel size
    pub kernel_size: u32,
    /// Stride
    pub stride: u32,
    /// Padding
    pub padding: u32,
    /// Tile size for output positions
    pub tile_size: u32,
}

impl TiledConv1dKernel {
    /// Create tiled Conv1d for Whisper conv1
    #[must_use]
    pub fn whisper_conv1() -> Self {
        Self {
            in_channels: 80,
            out_channels: 384,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            tile_size: 64,
        }
    }

    /// Create tiled Conv1d for Whisper conv2
    #[must_use]
    pub fn whisper_conv2() -> Self {
        Self {
            in_channels: 384,
            out_channels: 384,
            kernel_size: 3,
            stride: 2,
            padding: 1,
            tile_size: 32,
        }
    }
}

impl Kernel for TiledConv1dKernel {
    fn name(&self) -> &str {
        "conv1d_tiled"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Tiled version with shared memory
        // Delegates to the non-tiled kernel; shared-memory tiling is not yet implemented
        let simple = Conv1dKernel::new(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        );
        simple.build_ptx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_whisper_conv1() {
        let kernel = Conv1dKernel::whisper_conv1();
        assert_eq!(kernel.in_channels, 80);
        assert_eq!(kernel.out_channels, 384);
        assert_eq!(kernel.kernel_size, 3);
        assert_eq!(kernel.stride, 1);
        assert_eq!(kernel.padding, 1);
    }

    #[test]
    fn test_conv1d_whisper_conv2() {
        let kernel = Conv1dKernel::whisper_conv2();
        assert_eq!(kernel.in_channels, 384);
        assert_eq!(kernel.out_channels, 384);
        assert_eq!(kernel.kernel_size, 3);
        assert_eq!(kernel.stride, 2);
        assert_eq!(kernel.padding, 1);
    }

    #[test]
    fn test_conv1d_ptx_generation() {
        let kernel = Conv1dKernel::whisper_conv1();
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry conv1d"), "Should have conv1d entry");
        assert!(
            ptx.contains(".param .u64 input_ptr"),
            "Should have input_ptr"
        );
        assert!(
            ptx.contains(".param .u64 weight_ptr"),
            "Should have weight_ptr"
        );
        assert!(ptx.contains(".param .u64 bias_ptr"), "Should have bias_ptr");
        assert!(
            ptx.contains(".param .u64 output_ptr"),
            "Should have output_ptr"
        );
        assert!(ptx.contains(".param .u32 seq_len"), "Should have seq_len");
    }

    #[test]
    fn test_conv1d_kernel_name() {
        let kernel = Conv1dKernel::whisper_conv1();
        assert_eq!(kernel.name(), "conv1d");
    }

    #[test]
    fn test_conv1d_custom() {
        let kernel = Conv1dKernel::new(16, 32, 5, 2, 2);
        assert_eq!(kernel.in_channels, 16);
        assert_eq!(kernel.out_channels, 32);
        assert_eq!(kernel.kernel_size, 5);
        assert_eq!(kernel.stride, 2);
        assert_eq!(kernel.padding, 2);
    }

    #[test]
    fn test_conv1d_without_bias() {
        let kernel = Conv1dKernel::whisper_conv1().without_bias();
        assert!(!kernel.bias);
    }

    #[test]
    fn test_tiled_conv1d_whisper_conv1() {
        let kernel = TiledConv1dKernel::whisper_conv1();
        assert_eq!(kernel.in_channels, 80);
        assert_eq!(kernel.out_channels, 384);
    }

    #[test]
    fn test_tiled_conv1d_ptx() {
        let kernel = TiledConv1dKernel::whisper_conv2();
        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }
}
