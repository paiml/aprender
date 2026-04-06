//! Per-channel vs Per-tensor Quantization Granularity
//!
//! Provides quantization at different granularities:
//! - **Per-tensor**: Single scale/zero-point for entire tensor (fastest, least accurate)
//! - **Per-channel**: Separate scale/zero-point per channel (slower, more accurate)
//! - **Per-group**: Scale/zero-point per group of values (balance of speed/accuracy)
//!
//! Per-channel is critical for weight quantization where channels have different ranges.

mod calibrate;
mod metrics;
mod params;
mod quantize;
#[cfg(test)]
mod tests;
mod types;

pub use calibrate::{calibrate_per_channel, calibrate_per_group, calibrate_per_tensor};
pub use metrics::{compare_granularities, quantization_mse};
pub use params::{QuantParams, QuantizedTensor};
pub use quantize::{
    dequantize_tensor, dequantize_with_params, quantize_tensor, quantize_with_params,
};
pub use types::{QuantGranularity, QuantMode};
