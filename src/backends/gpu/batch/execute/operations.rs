//! Operation dispatch for GPU batch execution
//!
//! Contains `encode_operation()` which routes each `GpuOp` variant to the
//! appropriate cached pipeline via `encode_unary_op`, `encode_binary_op`,
//! or `encode_matmul_op`.

use super::super::{GpuCommandBatch, GpuOp};
use super::dispatch::PipelineCache;

impl GpuCommandBatch {
    /// Encode a single GPU operation into the command encoder.
    ///
    /// Uses the pipeline cache to avoid recompiling shaders for repeated operations.
    pub(crate) fn encode_operation(
        &self,
        op: &GpuOp,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
    ) -> Result<(), String> {
        use crate::backends::gpu::shaders;

        match op {
            GpuOp::Relu { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer =
                    input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
                let output_buffer =
                    output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

                self.encode_unary_op::<()>(
                    encoder,
                    cache,
                    shaders::RELU_SHADER,
                    "ReLU",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )?;
            }

            GpuOp::Scale { input, output, scalar } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer =
                    input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
                let output_buffer =
                    output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct ScaleParams {
                    scalar: f32,
                    _padding: [f32; 3],
                }

                let params = ScaleParams { scalar: *scalar, _padding: [0.0; 3] };

                self.encode_unary_op(
                    encoder,
                    cache,
                    shaders::SCALE_SHADER,
                    "Scale",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    Some(&params),
                )?;
            }

            GpuOp::Add { a, b, output } => {
                self.encode_binary_op_for(
                    encoder,
                    cache,
                    shaders::VEC_ADD_SHADER,
                    "Add",
                    a,
                    b,
                    output,
                )?;
            }

            GpuOp::Mul { a, b, output } => {
                self.encode_binary_op_for(
                    encoder,
                    cache,
                    shaders::VEC_MUL_SHADER,
                    "Mul",
                    a,
                    b,
                    output,
                )?;
            }

            GpuOp::Dot { a, b, output } => {
                self.encode_binary_op_for(
                    encoder,
                    cache,
                    shaders::DOT_PRODUCT_SHADER,
                    "Dot",
                    a,
                    b,
                    output,
                )?;
            }

            GpuOp::Sigmoid { input, output } => {
                self.encode_unary_op_for(
                    encoder,
                    cache,
                    shaders::SIGMOID_SHADER,
                    "Sigmoid",
                    input,
                    output,
                )?;
            }

            GpuOp::Tanh { input, output } => {
                self.encode_unary_op_for(
                    encoder,
                    cache,
                    shaders::TANH_SHADER,
                    "Tanh",
                    input,
                    output,
                )?;
            }

            GpuOp::Swish { input, output } => {
                self.encode_unary_op_for(
                    encoder,
                    cache,
                    shaders::SWISH_SHADER,
                    "Swish",
                    input,
                    output,
                )?;
            }

            GpuOp::Gelu { input, output } => {
                self.encode_unary_op_for(
                    encoder,
                    cache,
                    shaders::GELU_SHADER,
                    "GELU",
                    input,
                    output,
                )?;
            }

            GpuOp::Sub { a, b, output } => {
                self.encode_binary_op_for(
                    encoder,
                    cache,
                    shaders::VEC_SUB_SHADER,
                    "Sub",
                    a,
                    b,
                    output,
                )?;
            }

            GpuOp::Matmul { a, b, output, m, k, n } => {
                self.encode_matmul_op(
                    encoder,
                    cache,
                    shaders::MATMUL_SHADER,
                    "Matmul",
                    a,
                    b,
                    output,
                    *m,
                    *k,
                    *n,
                )?;
            }
        }

        Ok(())
    }

    /// Helper to extract buffers and encode a unary operation (no params)
    fn encode_unary_op_for(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
        shader_source: &str,
        label: &str,
        input: &super::super::BufferId,
        output: &super::super::BufferId,
    ) -> Result<(), String> {
        let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let input_buffer = input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        self.encode_unary_op::<()>(
            encoder,
            cache,
            shader_source,
            label,
            input_buffer,
            output_buffer,
            input_info.size,
            None,
        )
    }

    /// Helper to extract buffers and encode a binary operation
    fn encode_binary_op_for(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
        shader_source: &str,
        label: &str,
        a: &super::super::BufferId,
        b: &super::super::BufferId,
        output: &super::super::BufferId,
    ) -> Result<(), String> {
        let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
        let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
        let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        self.encode_binary_op(
            encoder,
            cache,
            shader_source,
            label,
            a_buffer,
            b_buffer,
            output_buffer,
            a_info.size,
        )
    }
}
