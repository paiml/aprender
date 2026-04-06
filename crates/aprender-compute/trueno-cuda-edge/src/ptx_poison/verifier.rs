//! PTX structural verification.
//!
//! [`PtxVerifier`] performs 6 structural checks on PTX source before it is
//! submitted to the CUDA driver. [`VerifiedPtx`] is an opaque wrapper that
//! can only be created through successful verification.

use serde::{Deserialize, Serialize};

use crate::error::EdgeError;

/// Structural verification errors specific to PTX source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtxVerificationError {
    /// Missing `.version` directive.
    MissingVersion,
    /// Missing `.target` directive.
    MissingTarget,
    /// Missing `.address_size` directive.
    MissingAddressSize,
    /// No `.entry` or `.func` defined.
    NoEntryPoint,
    /// Unbalanced braces in the PTX source.
    UnbalancedBraces {
        /// Number of opening braces.
        open: usize,
        /// Number of closing braces.
        close: usize,
    },
    /// Empty PTX source.
    EmptySource,
}

impl std::fmt::Display for PtxVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVersion => write!(f, "missing .version directive"),
            Self::MissingTarget => write!(f, "missing .target directive"),
            Self::MissingAddressSize => write!(f, "missing .address_size directive"),
            Self::NoEntryPoint => write!(f, "no .entry or .func defined"),
            Self::UnbalancedBraces { open, close } => {
                write!(f, "unbalanced braces: {open} open, {close} close")
            }
            Self::EmptySource => write!(f, "empty PTX source"),
        }
    }
}

/// PTX structural verifier that performs 6 checks.
#[derive(Debug, Default)]
pub struct PtxVerifier;

impl PtxVerifier {
    /// Create a new PTX verifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Verify PTX source and return a [`VerifiedPtx`] on success.
    ///
    /// # Checks performed
    ///
    /// 1. Source is non-empty
    /// 2. Contains `.version` directive
    /// 3. Contains `.target` directive
    /// 4. Contains `.address_size` directive
    /// 5. Contains at least one `.entry` or `.func`
    /// 6. Braces are balanced
    ///
    /// # Errors
    ///
    /// Returns [`EdgeError::PtxVerification`] describing the first failing check.
    pub fn verify(&self, source: &str) -> crate::error::Result<VerifiedPtx> {
        let errors = self.check_all(source);
        if let Some(first) = errors.into_iter().next() {
            return Err(EdgeError::PtxVerification { reason: first.to_string() });
        }
        Ok(VerifiedPtx { source: source.to_string() })
    }

    /// Run all checks and return all errors found.
    #[must_use]
    pub fn check_all(&self, source: &str) -> Vec<PtxVerificationError> {
        let mut errors = Vec::new();

        if source.trim().is_empty() {
            errors.push(PtxVerificationError::EmptySource);
            return errors;
        }

        if !source.contains(".version") {
            errors.push(PtxVerificationError::MissingVersion);
        }
        if !source.contains(".target") {
            errors.push(PtxVerificationError::MissingTarget);
        }
        if !source.contains(".address_size") {
            errors.push(PtxVerificationError::MissingAddressSize);
        }
        if !source.contains(".entry") && !source.contains(".func") {
            errors.push(PtxVerificationError::NoEntryPoint);
        }

        let open = source.chars().filter(|c| *c == '{').count();
        let close = source.chars().filter(|c| *c == '}').count();
        if open != close {
            errors.push(PtxVerificationError::UnbalancedBraces { open, close });
        }

        errors
    }
}

/// Opaque wrapper for PTX source that has passed structural verification.
///
/// Cannot be constructed outside this module — the only way to obtain one
/// is through [`PtxVerifier::verify`].
#[derive(Debug, Clone)]
pub struct VerifiedPtx {
    source: String,
}

impl VerifiedPtx {
    /// Returns the verified PTX source.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// Minimal valid PTX source for testing purposes.
pub const MINIMAL_VALID_PTX: &str = "\
.version 7.0
.target sm_80
.address_size 64
.entry test_kernel() {
    ret;
}
";

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn minimal_valid_ptx_passes() {
        let verifier = PtxVerifier::new();
        let result = verifier.verify(MINIMAL_VALID_PTX);
        assert!(result.is_ok());
    }

    #[test]
    fn empty_source_fails() {
        let verifier = PtxVerifier::new();
        let result = verifier.verify("");
        assert!(result.is_err());
    }

    #[test]
    fn missing_version_fails() {
        let src = ".target sm_80\n.address_size 64\n.entry k() {\nret;\n}\n";
        let verifier = PtxVerifier::new();
        let errors = verifier.check_all(src);
        assert!(errors.contains(&PtxVerificationError::MissingVersion));
    }

    #[test]
    fn missing_target_fails() {
        let src = ".version 7.0\n.address_size 64\n.entry k() {\nret;\n}\n";
        let verifier = PtxVerifier::new();
        let errors = verifier.check_all(src);
        assert!(errors.contains(&PtxVerificationError::MissingTarget));
    }

    #[test]
    fn missing_address_size_fails() {
        let src = ".version 7.0\n.target sm_80\n.entry k() {\nret;\n}\n";
        let verifier = PtxVerifier::new();
        let errors = verifier.check_all(src);
        assert!(errors.contains(&PtxVerificationError::MissingAddressSize));
    }

    #[test]
    fn no_entry_point_fails() {
        let src = ".version 7.0\n.target sm_80\n.address_size 64\n";
        let verifier = PtxVerifier::new();
        let errors = verifier.check_all(src);
        assert!(errors.contains(&PtxVerificationError::NoEntryPoint));
    }

    #[test]
    fn unbalanced_braces_fails() {
        let src = ".version 7.0\n.target sm_80\n.address_size 64\n.entry k() {\nret;\n";
        let verifier = PtxVerifier::new();
        let errors = verifier.check_all(src);
        assert!(errors.iter().any(|e| matches!(e, PtxVerificationError::UnbalancedBraces { .. })));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn verified_ptx_exposes_source() {
        let verifier = PtxVerifier::new();
        let verified = verifier.verify(MINIMAL_VALID_PTX).unwrap();
        assert!(verified.source().contains(".version"));
    }

    #[test]
    fn func_is_accepted_as_entry_point() {
        let src = ".version 7.0\n.target sm_80\n.address_size 64\n.func helper() {\nret;\n}\n";
        let verifier = PtxVerifier::new();
        let result = verifier.verify(src);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn whitespace_only_is_empty(ws in r"\s{0,100}") {
            let verifier = PtxVerifier::new();
            let errors = verifier.check_all(&ws);
            prop_assert!(errors.contains(&PtxVerificationError::EmptySource));
        }
    }
}
