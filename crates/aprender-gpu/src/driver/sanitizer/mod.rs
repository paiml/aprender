//! GPU Memory Sanitizer Wrapper
//!
//! Wraps NVIDIA compute-sanitizer to provide enhanced error reporting with
//! semantic context from trueno-gpu's knowledge of buffer allocations.
//!
//! # Features
//!
//! - **Address Registry**: Maps device addresses to buffer names/types
//! - **Semantic Errors**: Translates raw addresses to meaningful buffer offsets
//! - **PTX Source Mapping**: Maps SASS offsets to PTX source lines
//!
//! # Usage
//!
//! ```ignore
//! use trueno_gpu::driver::sanitizer::{AddressRegistry, SanitizedLaunch};
//!
//! // Register buffers before kernel launch
//! AddressRegistry::global().register("input_data", buf.as_ptr(), buf.size_bytes());
//!
//! // Run with sanitizer
//! let result = SanitizedLaunch::new(&stream, &module, "my_kernel")
//!     .with_ptx_source(ptx_source)
//!     .launch(&config, &args)?;
//! ```

mod parser;
mod types;

pub use parser::{PtxSourceMap, SanitizerParser};
pub use types::{
    AddressRegistry, BufferInfo, MemoryViolation, MemoryViolationType, SourceLocation,
};

use std::process::{Command, Stdio};

// ============================================================================
// Enhanced Sanitizer Report
// ============================================================================

/// A complete sanitizer report with enhanced context
#[derive(Debug)]
pub struct SanitizerReport {
    /// All violations detected
    pub violations: Vec<MemoryViolation>,
    /// Whether the kernel completed successfully
    pub success: bool,
    /// Raw sanitizer output
    pub raw_output: String,
}

impl SanitizerReport {
    /// Format the report with full context
    pub fn format(&self) -> String {
        if self.success {
            return "✅ No memory violations detected".to_string();
        }

        let registry = AddressRegistry::global()
            .lock()
            .expect("address registry lock poisoned");
        let mut output = String::new();

        output.push_str(&format!(
            "🚨 SANITIZER REPORT: {} violation(s) detected\n\n",
            self.violations.len()
        ));

        for (i, violation) in self.violations.iter().enumerate() {
            output.push_str(&format!("━━━ Violation {} ━━━\n", i + 1));
            output.push_str(&violation.format_with_registry(&registry));
            output.push_str("\n\n");
        }

        output
    }

    /// Format with PTX source context
    pub fn format_with_ptx(&self, ptx_map: &PtxSourceMap) -> String {
        if self.success {
            return "✅ No memory violations detected".to_string();
        }

        let registry = AddressRegistry::global()
            .lock()
            .expect("address registry lock poisoned");
        let mut output = String::new();

        output.push_str(&format!(
            "🚨 SANITIZER REPORT: {} violation(s) detected\n\n",
            self.violations.len()
        ));

        for (i, violation) in self.violations.iter().enumerate() {
            output.push_str(&format!("━━━ Violation {} ━━━\n", i + 1));
            output.push_str(&violation.format_with_registry(&registry));
            output.push('\n');

            // Try to find relevant PTX context
            // Note: SASS offset doesn't directly map to PTX line, but we can show
            // the relevant labels based on the kernel name
            if let Some(label) = ptx_map.label_at_line(1) {
                if let Some(context) = ptx_map.context_around_label(label, 5) {
                    output.push_str("\n📜 PTX Context:\n");
                    output.push_str(&context);
                }
            }

            output.push_str("\n\n");
        }

        output
    }
}

// ============================================================================
// Sanitizer Runner
// ============================================================================

/// Check if compute-sanitizer is available
pub fn sanitizer_available() -> bool {
    Command::new("compute-sanitizer")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a command under compute-sanitizer and parse the output
pub fn run_with_sanitizer(args: &[&str]) -> Result<SanitizerReport, std::io::Error> {
    let mut cmd = Command::new("compute-sanitizer");
    cmd.arg("--tool").arg("memcheck");
    cmd.args(args);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    let violations = SanitizerParser::parse(&combined);
    let success = violations.is_empty() && output.status.success();

    Ok(SanitizerReport {
        violations,
        success,
        raw_output: combined,
    })
}

#[cfg(test)]
mod tests;
