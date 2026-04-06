//! Shared helpers and constants for FKR-101-DEBUG tests.

pub use std::ffi::c_void;
pub use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
pub use trueno_gpu::kernels::lz4::{LZ4_HASH_SIZE, PAGE_SIZE};
pub use trueno_gpu::ptx::{
    PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxMemory, PtxModule, PtxReg, PtxType,
};

pub fn cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}

// Constants matching the LZ4 kernel
pub const WARP_SMEM_SIZE: u32 = PAGE_SIZE + LZ4_HASH_SIZE * 2 + 256; // 12544
pub const STATE_OFFSET: u32 = PAGE_SIZE + 8192 + 128 + 4; // 12420
pub const HASH_TABLE_OFFSET: u32 = PAGE_SIZE; // 4096
pub const LZ4_PRIME: u32 = 0x9E3779B1;
pub const LZ4_MAX_OFFSET: u32 = 65535;
pub const LZ4_MIN_MATCH: u32 = 4;

// Debug marker constants
pub const MARKER_LOOP_ENTRY: u32 = 0x01000000;
pub const MARKER_STATE_LOADED: u32 = 0x02000000;
pub const MARKER_HASH_COMPUTED: u32 = 0x03000000;
pub const MARKER_HASH_LOOKUP: u32 = 0x04000000;
pub const MARKER_FOUND_MATCH: u32 = 0x05000000;
pub const MARKER_COPY_LIT: u32 = 0x06000000;
pub const MARKER_NO_MATCH: u32 = 0x07000000;
pub const MARKER_EMIT_REMAIN: u32 = 0x08000000;
