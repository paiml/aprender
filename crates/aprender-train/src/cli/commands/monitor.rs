//! Monitor command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::MonitorArgs;

/// #2477: the PSI used to be computed from two literal bucket vectors, never
/// from the input or the baseline, so the drift verdict was a constant.
/// Refuse until a monitor reads the distributions it compares.
pub(crate) const MONITOR_UNIMPLEMENTED: &str = "entrenar monitor is not implemented: \
     it has no drift computation that reads the input, so it refuses rather than \
     print a PSI (aprender#2477). To attach to a training run use `apr monitor`.";

pub fn run_monitor(args: MonitorArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Monitoring: {}", args.input.display()));

    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    Err(MONITOR_UNIMPLEMENTED.to_string())
}
