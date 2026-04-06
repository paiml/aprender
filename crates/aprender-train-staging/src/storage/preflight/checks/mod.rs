//! Preflight check definitions and built-in checks.

mod data_integrity;
mod environment;

#[cfg(test)]
mod tests;

use super::{CheckMetadata, CheckResult, CheckType, PreflightContext};

/// Check function type
pub(crate) type CheckFn = Box<dyn Fn(&[Vec<f64>], &PreflightContext) -> CheckResult + Send + Sync>;

/// A single preflight check
pub struct PreflightCheck {
    /// Name of the check
    pub name: String,
    /// Type of check
    pub check_type: CheckType,
    /// Description of what this check validates
    pub description: String,
    /// Whether this check is required (failure blocks training)
    pub required: bool,
    /// The check function
    pub(crate) check_fn: CheckFn,
}

impl From<&PreflightCheck> for CheckMetadata {
    fn from(check: &PreflightCheck) -> Self {
        Self {
            name: check.name.clone(),
            check_type: check.check_type.clone(),
            description: check.description.clone(),
            required: check.required,
        }
    }
}

impl std::fmt::Debug for PreflightCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreflightCheck")
            .field("name", &self.name)
            .field("check_type", &self.check_type)
            .field("description", &self.description)
            .field("required", &self.required)
            .finish_non_exhaustive()
    }
}

impl PreflightCheck {
    /// Create a new check
    pub fn new<F>(
        name: impl Into<String>,
        check_type: CheckType,
        description: impl Into<String>,
        check_fn: F,
    ) -> Self
    where
        F: Fn(&[Vec<f64>], &PreflightContext) -> CheckResult + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            check_type,
            description: description.into(),
            required: true,
            check_fn: Box::new(check_fn),
        }
    }

    /// Make this check optional (warning only)
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Run this check
    pub fn run(&self, data: &[Vec<f64>], context: &PreflightContext) -> CheckResult {
        (self.check_fn)(data, context)
    }
}
