//! PMAT-008: PTX Debugger Implementation Tests (REQ-001 to REQ-010)
//!
//! Falsification tests per FKR-008 specification.
//! Verifies PTX debugger handles all PTX 8.0 constructs.
//!
//! Citations:
//! - [Betts et al. 2012] "GPUVerify: A Verifier for GPU Kernels" DOI:10.1145/2384616.2384625
//! - [Li & Gopalakrishnan 2010] "SMT-Based GPU Kernel Verification" DOI:10.1145/1882291.1882320
//! - [Leung et al. 2012] "Loop Tiling for GPGPU" DOI:10.1145/2259016.2259067

mod req001_to_req005;
mod req006_to_req010;
