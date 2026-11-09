//! Python exception emulation.
//!
//! The Python judge declines (rc 2) on `OSError`, `ValueError` and `KeyError`
//! and their subclasses, and crashes (rc 1) on anything else. The port keeps
//! the exception *type name* and *message* so the decline line on stderr is
//! byte-identical: `decline: <Type>: <msg>`.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyErr {
    pub kind: &'static str,
    pub msg: String,
}

pub type PyResult<T> = Result<T, PyErr>;

impl PyErr {
    pub fn new(kind: &'static str, msg: impl Into<String>) -> Self {
        Self {
            kind,
            msg: msg.into(),
        }
    }
    pub fn value(msg: impl Into<String>) -> Self {
        Self::new("ValueError", msg)
    }
    pub fn type_err(msg: impl Into<String>) -> Self {
        Self::new("TypeError", msg)
    }
    pub fn attr(msg: impl Into<String>) -> Self {
        Self::new("AttributeError", msg)
    }
    pub fn index(msg: impl Into<String>) -> Self {
        Self::new("IndexError", msg)
    }
    /// `KeyError(k)`: `str()` of a KeyError is the repr of its key.
    pub fn key(repr_of_key: impl Into<String>) -> Self {
        Self::new("KeyError", repr_of_key)
    }

    /// `except OSError` (and its errno subclasses).
    pub fn is_os(&self) -> bool {
        matches!(
            self.kind,
            "OSError"
                | "FileNotFoundError"
                | "IsADirectoryError"
                | "PermissionError"
                | "NotADirectoryError"
                | "FileExistsError"
        )
    }
    /// `except ValueError` (JSONDecodeError and the Unicode errors subclass it).
    pub fn is_value(&self) -> bool {
        matches!(
            self.kind,
            "ValueError" | "JSONDecodeError" | "UnicodeDecodeError" | "UnicodeEncodeError"
        )
    }
    pub fn is_key(&self) -> bool {
        self.kind == "KeyError"
    }
    pub fn is_type(&self) -> bool {
        self.kind == "TypeError"
    }
    /// `except (OSError, ValueError, KeyError)` — the set `main` declines on.
    pub fn declines(&self) -> bool {
        self.is_os() || self.is_value() || self.is_key()
    }
    /// `except (OSError, ValueError, KeyError, TypeError)` — the judge's
    /// "unreadable" catch.
    pub fn unreadable(&self) -> bool {
        self.declines() || self.is_type()
    }
}

impl fmt::Display for PyErr {
    /// `str(exc)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}
