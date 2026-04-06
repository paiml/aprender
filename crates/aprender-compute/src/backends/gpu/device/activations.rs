//! GPU activation functions and softmax operations
//!
//! Element-wise activation functions (ReLU, sigmoid, tanh, etc.) and
//! multi-pass softmax/log_softmax implementations.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// Execute ReLU activation on GPU: result[i] = max(0, input[i]) (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn relu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async {
            self.execute_element_wise_op("ReLU", shaders::RELU_SHADER, input, result, None).await
        })
    }

    /// Execute ReLU activation on GPU (async, works on all platforms)
    pub async fn relu_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("ReLU", shaders::RELU_SHADER, input, result, None).await
    }

    /// Execute leaky ReLU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn leaky_relu(
        &self,
        input: &[f32],
        result: &mut [f32],
        negative_slope: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.leaky_relu_async(input, result, negative_slope))
    }

    /// Execute leaky ReLU activation on GPU (async, works on all platforms)
    pub async fn leaky_relu_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        negative_slope: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LeakyReluParams {
            negative_slope: f32,
        }

        let params = LeakyReluParams { negative_slope };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "LeakyReLU",
            shaders::LEAKY_RELU_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await
    }

    /// Execute ELU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn elu(&self, input: &[f32], result: &mut [f32], alpha: f32) -> Result<(), String> {
        runtime::block_on(self.elu_async(input, result, alpha))
    }

    /// Execute ELU activation on GPU (async, works on all platforms)
    pub async fn elu_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        alpha: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct EluParams {
            alpha: f32,
        }

        let params = EluParams { alpha };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op("ELU", shaders::ELU_SHADER, input, result, Some(uniform_data))
            .await
    }

    /// Execute sigmoid activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn sigmoid(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.sigmoid_async(input, result))
    }

    /// Execute sigmoid activation on GPU (async, works on all platforms)
    pub async fn sigmoid_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Sigmoid", shaders::SIGMOID_SHADER, input, result, None).await
    }

    /// Execute tanh activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tanh(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.tanh_async(input, result))
    }

    /// Execute tanh activation on GPU (async, works on all platforms)
    pub async fn tanh_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Tanh", shaders::TANH_SHADER, input, result, None).await
    }

    /// Execute swish activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn swish(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.swish_async(input, result))
    }

    /// Execute swish activation on GPU (async, works on all platforms)
    pub async fn swish_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Swish", shaders::SWISH_SHADER, input, result, None).await
    }

    /// Execute GELU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gelu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.gelu_async(input, result))
    }

    /// Execute GELU activation on GPU (async, works on all platforms)
    pub async fn gelu_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("GELU", shaders::GELU_SHADER, input, result, None).await
    }

    /// Execute clip (clamp) operation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn clip(
        &self,
        input: &[f32],
        result: &mut [f32],
        min_val: f32,
        max_val: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.clip_async(input, result, min_val, max_val))
    }

    /// Execute clip (clamp) operation on GPU (async, works on all platforms)
    pub async fn clip_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        min_val: f32,
        max_val: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ClipParams {
            min_val: f32,
            max_val: f32,
        }

        let params = ClipParams { min_val, max_val };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "Clip",
            shaders::CLIP_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await
    }

    /// Execute softmax on GPU (sync, native only)
    ///
    /// Multi-pass implementation:
    /// 1. Find max value (parallel reduction)
    /// 2. Compute exp(x - max) (element-wise)
    /// 3. Sum exp values (parallel reduction)
    /// 4. Normalize by sum (element-wise)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn softmax(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.softmax_async(input, result).await })
    }

    /// Execute softmax on GPU (async, works on all platforms)
    pub async fn softmax_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        // Pass 1: Find max value
        let max_val = self.reduce_max(input).await?;

        // Pass 2: Compute exp(x - max)
        let exp_vals = self.compute_exp_subtract(input, max_val).await?;

        // Pass 3: Sum exp values
        let sum_exp = self.reduce_sum(&exp_vals).await?;

        // Pass 4: Normalize by sum
        self.normalize_by_sum(&exp_vals, result, sum_exp).await?;

        Ok(())
    }

    /// Execute log_softmax on GPU (sync, native only)
    ///
    /// Multi-pass implementation:
    /// 1. Find max value (parallel reduction)
    /// 2. Compute exp(x - max) (element-wise)
    /// 3. Sum exp values (parallel reduction)
    /// 4. Compute log_softmax (element-wise)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn log_softmax(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.log_softmax_async(input, result).await })
    }

    /// Execute log_softmax on GPU (async, works on all platforms)
    pub async fn log_softmax_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        // Pass 1: Find max value
        let max_val = self.reduce_max(input).await?;

        // Pass 2: Compute exp(x - max)
        let exp_vals = self.compute_exp_subtract(input, max_val).await?;

        // Pass 3: Sum exp values
        let sum_exp = self.reduce_sum(&exp_vals).await?;

        // Pass 4: Compute log_softmax = x - max - log(sum_exp)
        let log_sum_exp = sum_exp.max(f32::EPSILON).ln();

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LogSoftmaxParams {
            max_val: f32,
            log_sum_exp: f32,
        }

        let params = LogSoftmaxParams { max_val, log_sum_exp };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "LogSoftmax",
            shaders::LOG_SOFTMAX_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await?;

        Ok(())
    }

    /// Helper: Compute exp(input[i] - max_val)
    pub(super) async fn compute_exp_subtract(
        &self,
        input: &[f32],
        max_val: f32,
    ) -> Result<Vec<f32>, String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MaxValue {
            max_val: f32,
        }

        let params = MaxValue { max_val };
        let uniform_data = bytemuck::bytes_of(&params);

        let mut result = vec![0.0f32; input.len()];
        self.execute_element_wise_op(
            "SoftmaxExp",
            shaders::SOFTMAX_EXP_SHADER,
            input,
            &mut result,
            Some(uniform_data),
        )
        .await?;

        Ok(result)
    }

    /// Helper: Normalize by sum
    pub(super) async fn normalize_by_sum(
        &self,
        input: &[f32],
        result: &mut [f32],
        sum_val: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SumValue {
            sum_val: f32,
        }

        let params = SumValue { sum_val };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "SoftmaxNormalize",
            shaders::SOFTMAX_NORMALIZE_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await?;

        Ok(())
    }
}
