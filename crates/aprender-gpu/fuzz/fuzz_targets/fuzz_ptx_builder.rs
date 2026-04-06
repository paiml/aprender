//! Fuzz target for PTX builder
//!
//! Tests that arbitrary inputs to PTX module builder don't cause panics
//! or invalid PTX output.

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};

/// Fuzz input for PTX module generation
#[derive(Debug, Arbitrary)]
struct PtxFuzzInput {
    /// PTX version major (clamped to valid range)
    version_major: u8,
    /// PTX version minor
    version_minor: u8,
    /// Kernel name (will be sanitized)
    kernel_name: String,
    /// Number of parameters (clamped)
    num_params: u8,
    /// Parameter types
    param_types: Vec<u8>,
}

impl PtxFuzzInput {
    /// Sanitize kernel name to be valid PTX identifier
    fn sanitized_kernel_name(&self) -> String {
        let name: String = self.kernel_name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
            .take(64)
            .collect();

        if name.is_empty() || name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
            format!("kernel_{}", name)
        } else {
            name
        }
    }

    /// Get clamped version
    fn clamped_version(&self) -> (u8, u8) {
        let major = self.version_major.clamp(7, 8);
        let minor = self.version_minor.clamp(0, 5);
        (major, minor)
    }

    /// Convert param type index to PtxType
    fn param_type(&self, idx: usize) -> PtxType {
        match self.param_types.get(idx).unwrap_or(&0) % 6 {
            0 => PtxType::U32,
            1 => PtxType::U64,
            2 => PtxType::F32,
            3 => PtxType::F64,
            4 => PtxType::S32,
            _ => PtxType::B32,
        }
    }
}

fuzz_target!(|input: PtxFuzzInput| {
    // Build PTX module with fuzzed input
    let (major, minor) = input.clamped_version();
    let kernel_name = input.sanitized_kernel_name();
    let num_params = (input.num_params % 16) as usize; // Max 16 params

    // Create kernel with fuzzed parameters
    let mut kernel = PtxKernel::new(&kernel_name);

    for i in 0..num_params {
        let param_type = input.param_type(i);
        let param_name = format!("param_{}", i);
        kernel = kernel.param(param_type, &param_name);
    }

    // Build module
    let module = PtxModule::new()
        .version(major as u32, minor as u32)
        .target("sm_70")
        .address_size(64)
        .add_kernel(kernel);

    // Emit PTX - should never panic
    let ptx = module.emit();

    // Validate basic PTX structure
    assert!(ptx.contains(".version"), "PTX must have version");
    assert!(ptx.contains(".target"), "PTX must have target");
    assert!(ptx.contains(".entry"), "PTX must have entry point");

    // Validate kernel name appears
    assert!(ptx.contains(&kernel_name), "PTX must contain kernel name");
});
