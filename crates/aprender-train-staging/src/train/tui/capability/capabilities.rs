//! Terminal capability detection.

use super::TerminalMode;

/// Detected terminal capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    /// Terminal width in columns
    pub width: u16,
    /// Terminal height in rows
    pub height: u16,
    /// Supports Unicode characters
    pub unicode: bool,
    /// Supports ANSI color codes
    pub ansi_color: bool,
    /// Supports 24-bit true color
    pub true_color: bool,
    /// Is interactive TTY
    pub is_tty: bool,
}

impl Default for TerminalCapabilities {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            unicode: true,
            ansi_color: true,
            true_color: false,
            is_tty: true,
        }
    }
}

impl TerminalCapabilities {
    /// Detect terminal capabilities from environment.
    pub fn detect() -> Self {
        use std::env;
        use std::io::{stdout, IsTerminal};

        let is_tty = stdout().is_terminal();

        // Get size from environment or default
        let (width, height) = Self::get_size();

        // Check for Unicode support (most modern terminals)
        let lang = env::var("LANG").unwrap_or_default();
        let unicode = lang.contains("UTF") || lang.contains("utf");

        // Check for ANSI color support
        let term = env::var("TERM").unwrap_or_default();
        let ansi_color = !term.is_empty() && term != "dumb";

        // Check for true color support
        let colorterm = env::var("COLORTERM").unwrap_or_default();
        let true_color = colorterm == "truecolor" || colorterm == "24bit";

        Self { width, height, unicode, ansi_color, true_color, is_tty }
    }

    /// Get terminal size.
    pub(crate) fn get_size() -> (u16, u16) {
        use std::env;

        // 1. Check environment variables (CI/headless)
        if let (Ok(cols), Ok(rows)) = (env::var("COLUMNS"), env::var("LINES")) {
            if let (Ok(c), Ok(r)) = (cols.parse(), rows.parse()) {
                return (c, r);
            }
        }

        // 2. Try ioctl on Unix
        #[cfg(unix)]
        {
            use std::io::{stdout, IsTerminal};
            if stdout().is_terminal() {
                // Use libc directly for TIOCGWINSZ
                #[repr(C)]
                struct WinSize {
                    ws_row: u16,
                    ws_col: u16,
                    ws_xpixel: u16,
                    ws_ypixel: u16,
                }
                extern "C" {
                    fn ioctl(fd: i32, request: u64, ...) -> i32;
                }
                const TIOCGWINSZ: u64 = 0x5413; // Linux
                let mut ws = WinSize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
                // SAFETY: ioctl with TIOCGWINSZ is safe for reading terminal size
                #[allow(unsafe_code)]
                if unsafe { ioctl(1, TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 {
                    return (ws.ws_col, ws.ws_row);
                }
            }
        }

        // 3. Fallback
        (80, 24)
    }

    /// Get recommended terminal mode based on capabilities.
    pub fn recommended_mode(&self) -> TerminalMode {
        if !self.is_tty {
            TerminalMode::Ascii
        } else if self.true_color {
            TerminalMode::Ansi
        } else if self.unicode {
            TerminalMode::Unicode
        } else {
            TerminalMode::Ascii
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_capabilities_default() {
        let caps = TerminalCapabilities::default();
        assert_eq!(caps.width, 80);
        assert_eq!(caps.height, 24);
        assert!(caps.unicode);
        assert!(caps.ansi_color);
        assert!(!caps.true_color);
        assert!(caps.is_tty);
    }

    #[test]
    fn test_terminal_capabilities_clone() {
        let caps = TerminalCapabilities::default();
        let cloned = caps;
        assert_eq!(caps.width, cloned.width);
        assert_eq!(caps.unicode, cloned.unicode);
    }

    #[test]
    fn test_terminal_capabilities_eq() {
        let caps1 = TerminalCapabilities::default();
        let caps2 = TerminalCapabilities::default();
        assert_eq!(caps1, caps2);

        let caps3 = TerminalCapabilities { width: 120, ..Default::default() };
        assert_ne!(caps1, caps3);
    }

    #[test]
    fn test_terminal_capabilities_debug() {
        let caps = TerminalCapabilities::default();
        let debug = format!("{caps:?}");
        assert!(debug.contains("TerminalCapabilities"));
        assert!(debug.contains("width: 80"));
    }

    #[test]
    fn test_recommended_mode_not_tty() {
        let caps = TerminalCapabilities { is_tty: false, ..Default::default() };
        assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);
    }

    #[test]
    fn test_recommended_mode_true_color() {
        let caps = TerminalCapabilities {
            is_tty: true,
            true_color: true,
            unicode: true,
            ..Default::default()
        };
        assert_eq!(caps.recommended_mode(), TerminalMode::Ansi);
    }

    #[test]
    fn test_recommended_mode_unicode() {
        let caps = TerminalCapabilities {
            is_tty: true,
            true_color: false,
            unicode: true,
            ..Default::default()
        };
        assert_eq!(caps.recommended_mode(), TerminalMode::Unicode);
    }

    #[test]
    fn test_recommended_mode_ascii_fallback() {
        let caps = TerminalCapabilities {
            is_tty: true,
            true_color: false,
            unicode: false,
            ansi_color: false,
            ..Default::default()
        };
        assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);
    }

    #[test]
    fn test_detect_returns_valid_capabilities() {
        // Note: actual values depend on environment
        let caps = TerminalCapabilities::detect();
        assert!(caps.width > 0);
        assert!(caps.height > 0);
    }

    #[test]
    fn test_get_size_returns_valid_size() {
        let (width, height) = TerminalCapabilities::get_size();
        assert!(width > 0);
        assert!(height > 0);
    }
}
