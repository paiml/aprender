use crate::driver::CudaContext;
use crate::memory::resident::GpuResidentTensor;

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                $crate::testing::require_gpu::skip_or_panic("CUDA test", &e);
                return;
            }
        }
    };
}

mod decoder_config;
mod encoder_block;
mod kv_cache;
