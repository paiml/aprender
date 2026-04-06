//! DEBUG: Isolate which operation crashes
//! PHASE 4: Full GPU Encoder Block (Total Offload)

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    reset_transfer_counters, total_d2h_transfers, total_h2d_transfers, GpuResidentTensor,
};

mod encoder_block;
mod kernel_isolation;
