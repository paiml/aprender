//! One exit-code convention for the whole `apr *-lint` family.
//!
//! Every `apr *-lint` command reads a captured observation (JSON, NDJSON, CSV
//! or a trace directory), runs pure classifiers over it, and reports whether
//! the contract gates held. They are documented identically and are meant to be
//! driven from the same CI harness, so they must agree on what an exit code
//! means:
//!
//! | exit | meaning |
//! |-----:|---------|
//! | 0 | every gate the observation exercised passed |
//! | 3 | the named input does not exist |
//! | 4 | the input exists but is not a usable observation: wrong kind of path, empty, unparseable, or containing none of the sections the gates need |
//! | 5 | the observation was usable and a contract gate rejected it |
//! | 7 | the input exists but could not be read (OS error) |
//!
//! The distinction that matters to a CI wrapper is **4 vs 5**: 4 means *your
//! capture step is broken*, 5 means *the system under test violated the
//! contract*. Collapsing both onto 1 — which half the family used to do — makes
//! that undecidable.
//!
//! See `docs/reference/lint-exit-codes.md`.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::CliError;

/// A failure from an `apr *-lint` command, classified into the family's
/// exit-code convention.
#[derive(Debug)]
pub enum LintError {
    /// The input the user named does not exist. -> exit 3
    MissingInput(PathBuf),
    /// The input exists but could not be read. -> exit 7
    Unreadable(String),
    /// The input is not a usable observation: unparseable, empty, or it
    /// contains none of the sections the gates need. -> exit 4
    UnusableInput(String),
    /// The observation was usable and at least one contract gate rejected it.
    /// -> exit 5
    GateFailed(String),
}

impl LintError {
    /// The input exists but is not a usable observation.
    pub fn unusable(msg: impl Into<String>) -> Self {
        Self::UnusableInput(msg.into())
    }

    /// At least one contract gate rejected an otherwise usable observation.
    pub fn gate_failed(msg: impl Into<String>) -> Self {
        Self::GateFailed(msg.into())
    }
}

impl fmt::Display for LintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingInput(p) => write!(f, "File not found: {}", p.display()),
            Self::Unreadable(m) | Self::UnusableInput(m) | Self::GateFailed(m) => {
                write!(f, "{m}")
            }
        }
    }
}

impl std::error::Error for LintError {}

impl From<LintError> for CliError {
    fn from(e: LintError) -> Self {
        match e {
            LintError::MissingInput(p) => Self::FileNotFound(p),
            LintError::Unreadable(m) => Self::Io(std::io::Error::other(m)),
            LintError::UnusableInput(m) => Self::InvalidInput(m),
            LintError::GateFailed(m) => Self::ValidationFailed(m),
        }
    }
}

/// Read and parse a captured JSON observation, classifying every failure into
/// the family convention.
///
/// `falsify_id` is the contract stamp (e.g. `FALSIFY-CRUX-B-08`) that prefixes
/// the diagnostic so the message still names the contract that wanted the file.
pub fn load_json_observation(
    observation_file: &str,
    falsify_id: &str,
) -> std::result::Result<serde_json::Value, LintError> {
    let path = Path::new(observation_file);
    if !path.exists() {
        return Err(LintError::MissingInput(path.to_path_buf()));
    }
    let raw = std::fs::read_to_string(path).map_err(|e| {
        LintError::Unreadable(format!("{falsify_id}: failed to read observation: {e}"))
    })?;
    if raw.trim().is_empty() {
        return Err(LintError::unusable(format!(
            "{falsify_id}: observation file is empty"
        )));
    }
    serde_json::from_str(&raw).map_err(|e| {
        LintError::unusable(format!("{falsify_id}: observation is not valid JSON: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_input_maps_to_exit_3() {
        let e = LintError::MissingInput(PathBuf::from("/nope/x.json"));
        assert_eq!(CliError::from(e).exit_code_value(), 3);
    }

    #[test]
    fn unusable_input_maps_to_exit_4() {
        let e = LintError::unusable("FALSIFY-X: observation is not valid JSON: eof");
        assert_eq!(CliError::from(e).exit_code_value(), 4);
    }

    #[test]
    fn gate_failure_maps_to_exit_5() {
        let e = LintError::gate_failed("FALSIFY-X-001 quality gate failed");
        assert_eq!(CliError::from(e).exit_code_value(), 5);
    }

    #[test]
    fn unreadable_input_maps_to_exit_7() {
        let e = LintError::Unreadable("FALSIFY-X: failed to read observation: EACCES".into());
        assert_eq!(CliError::from(e).exit_code_value(), 7);
    }

    #[test]
    fn missing_input_message_matches_the_rest_of_the_cli() {
        let e = LintError::MissingInput(PathBuf::from("/nope/x.json"));
        assert_eq!(e.to_string(), "File not found: /nope/x.json");
        assert_eq!(
            CliError::from(LintError::MissingInput(PathBuf::from("/nope/x.json"))).to_string(),
            "File not found: /nope/x.json"
        );
    }

    #[test]
    fn a_json_observation_failure_never_claims_to_be_an_apr_model() {
        let dir = std::env::temp_dir().join(format!("apr-lint-err-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let f = dir.join("bad.json");
        std::fs::write(&f, "not json at all").expect("write");
        let err = load_json_observation(&f.to_string_lossy(), "FALSIFY-CRUX-B-08")
            .expect_err("must reject non-JSON");
        let rendered = CliError::from(err).to_string();
        assert!(
            !rendered.contains("APR"),
            "a captured JSON observation is not an APR model file; got: {rendered}"
        );
        assert!(rendered.contains("not valid JSON"), "got: {rendered}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_observation_is_unusable_input_not_a_gate_failure() {
        let dir = std::env::temp_dir().join(format!("apr-lint-err-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let f = dir.join("empty.json");
        std::fs::write(&f, "   \n").expect("write");
        let err = load_json_observation(&f.to_string_lossy(), "FALSIFY-CRUX-B-08")
            .expect_err("must reject empty");
        assert_eq!(CliError::from(err).exit_code_value(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_observation_is_reported_before_any_parse_attempt() {
        let err = load_json_observation("/definitely/not/here.json", "FALSIFY-CRUX-B-08")
            .expect_err("must reject missing");
        assert!(matches!(err, LintError::MissingInput(_)));
        assert_eq!(CliError::from(err).exit_code_value(), 3);
    }
}
