//! Quantization parity oracle.
//!
//! Verifies that CPU and GPU quantization implementations produce
//! equivalent results within format-specific tolerances.
//!
//! # Supported Formats
//!
//! | Format | Bits | Tolerance | Levels |
//! |--------|------|-----------|--------|
//! | `Q4_K` | 4 | 0.05 | 16 |
//! | `Q5_K` | 5 | 0.02 | 32 |
//! | `Q6_K` | 6 | 0.01 | 64 |
//! | `Q8_0` | 8 | 0.005 | 256 |
//! | `F16` | 16 | 0.001 | 65536 |
//! | `F32` | 32 | ε | 2³² |
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::quant_oracle::{
//!     QuantFormat, BoundaryValueGenerator, ParityConfig, check_values_parity
//! };
//!
//! // Generate boundary test values
//! let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
//! let boundaries = gen.all_boundaries();
//! assert!(boundaries.iter().any(|v| v.is_nan()));
//!
//! // Check parity between CPU and GPU results
//! let cpu = vec![1.0, 2.0, 3.0];
//! let gpu = vec![1.001, 2.001, 3.001];
//! let config = ParityConfig::new(QuantFormat::Q4K);
//! let report = check_values_parity(&cpu, &gpu, &config);
//! assert!(report.passed()); // within 0.05 tolerance
//! ```

pub mod boundary;
pub mod parity;
pub mod roundtrip;

pub use boundary::{BoundaryValueGenerator, QuantFormat};
pub use parity::{check_values_parity, ParityConfig, ParityReport, ParityViolation};
pub use roundtrip::{roundtrip_idempotence, MockQuantizer, Quantizer, RoundtripResult};
