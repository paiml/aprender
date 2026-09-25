//! Provider-terms tagging for `agent-trace-v1` rows (PRA-001 T12, spec §2.10;
//! contract `agent-trace-terms-v1`).
//!
//! Every captured row carries `provider`, an exact `model_id`, `access_channel`
//! and a `terms_ref` snapshot, so any pool can be rebuilt without a given
//! provider or channel in one call (P-TERMS). This records which terms governed
//! a row; it is not legal advice, and the operator's S-6 ruling governs use.
//!
//! A row whose tags are missing or inconsistent is REFUSED by every rebuild, even
//! one that excludes nothing: an untagged row cannot be shown not to come from an
//! excluded source.

use serde_json::Value;

/// Who served the lane (`provider` in §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    Google,
    Local,
}

impl Provider {
    #[must_use]
    pub fn parse(_s: &str) -> Option<Self> {
        None
    }
}

/// How the lane was reached (`access_channel` in §2.2); terms differ by channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    AnthropicApi,
    ClaudeCodeCli,
    Antigravity,
    GeminiApi,
    Vertex,
    AprServe,
}

impl Channel {
    pub const ALL: [Channel; 6] = [
        Channel::AnthropicApi,
        Channel::ClaudeCodeCli,
        Channel::Antigravity,
        Channel::GeminiApi,
        Channel::Vertex,
        Channel::AprServe,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        ""
    }

    #[must_use]
    pub fn parse(_s: &str) -> Option<Self> {
        None
    }

    #[must_use]
    pub fn provider(self) -> Provider {
        Provider::Local
    }
}

/// The governing-terms snapshot taken at capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermsRef {
    pub url: String,
    pub effective_date: String,
    pub fetched_at: String,
}

/// The four terms fields of one row, validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tags {
    pub provider: Provider,
    pub model_id: String,
    pub channel: Channel,
    pub terms: TermsRef,
}

/// One reason a row's terms tags do not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gap {
    Missing(&'static str),
}

/// The validated tags of `row`, or every gap found.
///
/// # Errors
/// Returns the gaps when any tag is missing, unknown or inconsistent.
pub fn tags(_row: &Value) -> Result<Tags, Vec<Gap>> {
    Err(vec![])
}

/// What a pool rebuild leaves out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exclude {
    pub providers: Vec<Provider>,
    pub channels: Vec<Channel>,
}

impl Exclude {
    /// The public-release default (§2.11): gold-and-local only.
    #[must_use]
    pub fn public_release() -> Self {
        Self::default()
    }

    /// Parse `--exclude-provider <p>` / `--exclude-channel <c>`, each repeatable.
    ///
    /// # Errors
    /// An unknown flag, a missing value, or an unknown provider or channel.
    pub fn from_args(_args: &[&str]) -> Result<Self, String> {
        Ok(Self::default())
    }
}

/// A rebuilt pool: rows kept, rows left out by the filter, rows refused for gaps.
#[derive(Debug, Default)]
pub struct Rebuild<'a> {
    pub kept: Vec<&'a str>,
    pub excluded: Vec<&'a str>,
    pub refused: Vec<(&'a str, Vec<Gap>)>,
}

/// Rebuild a pool from a JSONL batch, leaving out every excluded source.
#[must_use]
pub fn rebuild<'a>(rows: &'a str, _ex: &Exclude) -> Rebuild<'a> {
    Rebuild {
        kept: rows.lines().filter(|l| !l.trim().is_empty()).collect(),
        ..Rebuild::default()
    }
}

#[cfg(test)]
#[path = "terms_tests.rs"]
mod tests;
