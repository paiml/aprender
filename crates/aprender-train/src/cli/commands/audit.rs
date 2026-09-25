//! Audit command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::AuditArgs;

/// #2477: every audit type used to return literals (bias rates 0.72/0.78,
/// calibration 0.05, privacy/security a fixed PASS that never took `args`), so
/// the verdict was the same for a JSONL and a 151 MB safetensors. An auditor
/// that cannot fail is worse than none: refuse until one reads its input.
pub(crate) const AUDIT_UNIMPLEMENTED: &str = "entrenar audit is not implemented: \
     it has no analysis that reads the input, so it refuses rather than print a \
     verdict (aprender#2477). For dataset quality use `apr data audit`.";

pub fn run_audit(args: AuditArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Auditing: {}", args.input.display()));

    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    Err(format!("{} (requested: {} audit)", AUDIT_UNIMPLEMENTED, args.audit_type))
}
