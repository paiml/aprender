// PMAT-540 Phase 5: Tests for train command handler functions
// Contract: co-evolution with apr-cli-commands-v1.yaml

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // classify_exit_code
    // ========================================================================

    #[test]
    fn classify_exit_code_success() {
        assert_eq!(classify_exit_code(0), "success");
    }

    #[test]
    fn classify_exit_code_error() {
        assert_eq!(classify_exit_code(1), "error");
    }

    #[test]
    fn classify_exit_code_usage() {
        assert_eq!(classify_exit_code(2), "usage");
    }

    #[test]
    fn classify_exit_code_oom() {
        assert_eq!(classify_exit_code(137), "oom");
    }

    #[test]
    fn classify_exit_code_sigsegv() {
        assert_eq!(classify_exit_code(139), "sigsegv");
    }

    #[test]
    fn classify_exit_code_sigabrt() {
        assert_eq!(classify_exit_code(134), "sigabrt");
    }

    #[test]
    fn classify_exit_code_sigbus() {
        assert_eq!(classify_exit_code(135), "sigbus");
    }

    #[test]
    fn classify_exit_code_generic_signal() {
        assert_eq!(classify_exit_code(143), "signal"); // SIGTERM
        assert_eq!(classify_exit_code(129), "signal"); // SIGHUP
    }

    #[test]
    fn classify_exit_code_unknown() {
        assert_eq!(classify_exit_code(42), "unknown");
        assert_eq!(classify_exit_code(-1), "unknown");
    }

    // ========================================================================
    // format_archive_size
    // ========================================================================

    #[test]
    fn format_archive_size_bytes() {
        assert_eq!(format_archive_size(0), "0 B");
        assert_eq!(format_archive_size(512), "512 B");
        assert_eq!(format_archive_size(1023), "1023 B");
    }

    #[test]
    fn format_archive_size_kb() {
        assert_eq!(format_archive_size(1024), "1.0 KB");
        assert_eq!(format_archive_size(1536), "1.5 KB");
    }

    #[test]
    fn format_archive_size_mb() {
        assert_eq!(format_archive_size(1_048_576), "1.0 MB");
        assert_eq!(format_archive_size(10_485_760), "10.0 MB");
    }

    #[test]
    fn format_archive_size_gb() {
        assert_eq!(format_archive_size(1_073_741_824), "1.0 GB");
        assert_eq!(format_archive_size(4_294_967_296), "4.0 GB");
    }

    // ========================================================================
    // watch_max_restarts_exceeded
    // ========================================================================

    #[test]
    fn watch_max_restarts_exceeded_returns_error() {
        let err = watch_max_restarts_exceeded(5, true);
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("5"), "Should mention restart count");
                assert!(msg.contains("exceeded"), "Should mention exceeded");
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    // ========================================================================
    // classify_not_available
    // ========================================================================

    #[test]
    fn classify_not_available_returns_validation_failed() {
        let err = classify_not_available();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("classify"), "Should mention classify");
                assert!(msg.contains("entrenar"), "Should mention entrenar dep");
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }
}
