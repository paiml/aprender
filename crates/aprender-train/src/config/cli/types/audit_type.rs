//! Audit type for CLI commands.

/// Audit type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AuditType {
    #[default]
    Bias,
    Fairness,
    Privacy,
    Security,
}

impl std::str::FromStr for AuditType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bias" => Ok(AuditType::Bias),
            "fairness" => Ok(AuditType::Fairness),
            "privacy" => Ok(AuditType::Privacy),
            "security" => Ok(AuditType::Security),
            _ => Err(format!(
                "Unknown audit type: {s}. Valid types: bias, fairness, privacy, security"
            )),
        }
    }
}

impl std::fmt::Display for AuditType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditType::Bias => write!(f, "bias"),
            AuditType::Fairness => write!(f, "fairness"),
            AuditType::Privacy => write!(f, "privacy"),
            AuditType::Security => write!(f, "security"),
        }
    }
}
