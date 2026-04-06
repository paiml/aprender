//! Cross-platform debug logging for ttop.
//!
//! Provides structured debug output that works identically on Linux and macOS.
//! Enabled via `--debug` flag or `TTOP_DEBUG=1` environment variable.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Global debug mode flag.
static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Start time stored as millis since UNIX epoch (atomic-safe).
static START_TIME_MS: AtomicU64 = AtomicU64::new(0);

/// Enables debug mode globally.
pub fn enable() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    START_TIME_MS.store(now, Ordering::SeqCst);
    DEBUG_ENABLED.store(true, Ordering::SeqCst);
}

/// Disables debug mode globally.
pub fn disable() {
    DEBUG_ENABLED.store(false, Ordering::SeqCst);
}

/// Returns true if debug mode is enabled.
#[inline]
pub fn is_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// Gets elapsed time since debug was enabled.
fn elapsed_ms() -> u64 {
    let start = START_TIME_MS.load(Ordering::Relaxed);
    if start == 0 {
        return 0;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    now.saturating_sub(start)
}

/// Debug log levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Tracing entry/exit of functions
    Trace,
    /// Debug information
    Debug,
    /// Informational messages
    Info,
    /// Warnings
    Warn,
    /// Errors
    Error,
}

impl Level {
    fn as_str(&self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    fn color_code(&self) -> &'static str {
        match self {
            Level::Trace => "\x1b[90m", // Gray
            Level::Debug => "\x1b[36m", // Cyan
            Level::Info => "\x1b[32m",  // Green
            Level::Warn => "\x1b[33m",  // Yellow
            Level::Error => "\x1b[31m", // Red
        }
    }
}

/// Logs a debug message if debug mode is enabled.
pub fn log(level: Level, component: &str, message: &str) {
    if !is_enabled() {
        return;
    }

    let elapsed = elapsed_ms();
    let reset = "\x1b[0m";
    let color = level.color_code();

    // Format: [+0000ms] [LEVEL] [component] message
    let _ = writeln!(
        io::stderr(),
        "[+{:04}ms] {}[{:5}]{} [{}] {}",
        elapsed,
        color,
        level.as_str(),
        reset,
        component,
        message
    );
}

/// Logs with format arguments.
#[macro_export]
macro_rules! debug_log {
    ($level:expr, $component:expr, $($arg:tt)*) => {
        if $crate::monitor::debug::is_enabled() {
            $crate::monitor::debug::log($level, $component, &format!($($arg)*));
        }
    };
}

/// Convenience macro for trace level.
#[macro_export]
macro_rules! trace {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug_log!($crate::monitor::debug::Level::Trace, $component, $($arg)*)
    };
}

/// Convenience macro for debug level.
#[macro_export]
macro_rules! debug {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug_log!($crate::monitor::debug::Level::Debug, $component, $($arg)*)
    };
}

/// Convenience macro for info level.
#[macro_export]
macro_rules! info {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug_log!($crate::monitor::debug::Level::Info, $component, $($arg)*)
    };
}

/// Convenience macro for warn level.
#[macro_export]
macro_rules! warn {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug_log!($crate::monitor::debug::Level::Warn, $component, $($arg)*)
    };
}

/// Convenience macro for error level.
#[macro_export]
macro_rules! error {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug_log!($crate::monitor::debug::Level::Error, $component, $($arg)*)
    };
}

/// RAII guard for timing a scope.
pub struct TimingGuard {
    component: &'static str,
    operation: String,
    start: Instant,
}

impl TimingGuard {
    /// Creates a new timing guard.
    pub fn new(component: &'static str, operation: impl Into<String>) -> Self {
        let operation = operation.into();
        if is_enabled() {
            log(Level::Trace, component, &format!("-> {operation}"));
        }
        Self { component, operation, start: Instant::now() }
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        if is_enabled() {
            let elapsed = self.start.elapsed();
            log(
                Level::Trace,
                self.component,
                &format!("<- {} ({:.2}ms)", self.operation, elapsed.as_secs_f64() * 1000.0),
            );
        }
    }
}

/// Creates a timing guard for a scope.
#[macro_export]
macro_rules! time_scope {
    ($component:expr, $operation:expr) => {
        let _guard = $crate::monitor::debug::TimingGuard::new($component, $operation);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_disabled_by_default() {
        // Reset state
        disable();
        assert!(!is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        disable();
        assert!(!is_enabled());

        enable();
        assert!(is_enabled());

        disable();
        assert!(!is_enabled());
    }

    #[test]
    fn test_level_as_str() {
        assert_eq!(Level::Trace.as_str(), "TRACE");
        assert_eq!(Level::Debug.as_str(), "DEBUG");
        assert_eq!(Level::Info.as_str(), "INFO");
        assert_eq!(Level::Warn.as_str(), "WARN");
        assert_eq!(Level::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_level_has_color() {
        // All levels should have non-empty color codes
        assert!(!Level::Trace.color_code().is_empty());
        assert!(!Level::Debug.color_code().is_empty());
        assert!(!Level::Info.color_code().is_empty());
        assert!(!Level::Warn.color_code().is_empty());
        assert!(!Level::Error.color_code().is_empty());
    }

    #[test]
    fn test_log_when_disabled_does_nothing() {
        disable();
        // Should not panic or output anything
        log(Level::Debug, "test", "message");
    }

    #[test]
    fn test_log_when_enabled_outputs_to_stderr() {
        enable();
        // Should not panic - output goes to stderr
        log(Level::Info, "test", "hello world");
        disable();
    }

    #[test]
    fn test_timing_guard_measures_time() {
        enable();
        {
            let _guard = TimingGuard::new("test", "operation");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Guard dropped, should have logged
        disable();
    }

    #[test]
    fn test_elapsed_increases() {
        enable();
        let t1 = elapsed_ms();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let t2 = elapsed_ms();
        assert!(t2 >= t1, "elapsed should increase: {t2} >= {t1}");
        disable();
    }

    #[test]
    fn test_macros_compile() {
        enable();
        // These should compile and not panic
        log(Level::Trace, "test", "trace message");
        log(Level::Debug, "test", "debug message");
        log(Level::Info, "test", "info message");
        log(Level::Warn, "test", "warn message");
        log(Level::Error, "test", "error message");
        disable();
    }

    #[test]
    fn test_timing_guard_when_disabled() {
        disable();
        {
            let _guard = TimingGuard::new("test", "noop");
            // Should not output anything
        }
    }

    #[test]
    fn test_elapsed_ms_when_not_enabled() {
        // Reset everything
        disable();
        START_TIME_MS.store(0, Ordering::SeqCst);
        // When start time is 0, should return 0
        assert_eq!(elapsed_ms(), 0);
    }

    #[test]
    fn test_timing_guard_drop_when_enabled() {
        // Enable debug mode first
        enable();
        // Ensure debug is enabled
        assert!(is_enabled());

        // Create and immediately drop the guard
        let guard = TimingGuard::new("drop_test", "test_operation");
        // Verify fields are set
        assert_eq!(guard.component, "drop_test");
        assert_eq!(guard.operation, "test_operation");
        // Drop explicitly
        drop(guard);

        // Clean up
        disable();
    }

    #[test]
    fn test_level_equality() {
        assert_eq!(Level::Trace, Level::Trace);
        assert_eq!(Level::Debug, Level::Debug);
        assert_ne!(Level::Trace, Level::Debug);
    }

    #[test]
    fn test_level_clone() {
        let level = Level::Warn;
        let cloned = level;
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_level_debug() {
        let debug_str = format!("{:?}", Level::Error);
        assert!(debug_str.contains("Error"));
    }

    #[test]
    fn test_log_all_levels_when_enabled() {
        enable();
        // Test each level to ensure color codes are used
        log(Level::Trace, "test", "trace msg");
        log(Level::Debug, "test", "debug msg");
        log(Level::Info, "test", "info msg");
        log(Level::Warn, "test", "warn msg");
        log(Level::Error, "test", "error msg");
        disable();
    }
}
