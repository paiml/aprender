//! Eval rows attached to a model (EXT-001 row EXT-09, aprender#4391).
//!
//! EXT-001 §3.1: pacha is the system of record for evals. A row is keyed on the
//! evaluated file's sha256, not on a pacha model id, so `apr eval` on a file the
//! registry has never seen still records (and warns): registering it later links
//! the rows through `ModelCard.extra["sha256"]`, which EXT-05 writes.

use crate::error::{PachaError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One eval result: `model_sha, suite, suite_manifest_sha, score, n, engine identity, host, ts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecord {
    /// sha256 of the evaluated file, 64 lowercase hex.
    pub model_sha: String,
    /// Suite name, e.g. `perplexity/wikitext2` or `apr-qa`.
    pub suite: String,
    /// sha256 of what the suite ran (its items or gate list), 64 lowercase hex.
    pub suite_manifest_sha: String,
    /// The suite's own score. Its direction and unit belong to the suite.
    pub score: f64,
    /// Items (tokens, gates, problems) the score is over.
    pub n: u64,
    /// I-5 engine identity: the producing `apr` version.
    pub engine_version: String,
    /// I-5 engine identity: the git sha or crates.io tarball sha of that `apr`.
    pub engine_sha: String,
    /// Host the eval ran on.
    pub host: String,
    /// When the eval finished.
    pub ts: DateTime<Utc>,
}

/// True for 64 lowercase hex characters.
#[must_use]
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl EvalRecord {
    /// Refuse a row that could not be traced back to a file, a suite and an engine.
    ///
    /// # Errors
    ///
    /// Returns `PachaError::Validation` naming the first bad field.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str| Err(PachaError::Validation(format!("eval row: {what}")));
        if !is_sha256_hex(&self.model_sha) {
            return bad("model_sha is not 64 lowercase hex");
        }
        if !is_sha256_hex(&self.suite_manifest_sha) {
            return bad("suite_manifest_sha is not 64 lowercase hex");
        }
        if self.suite.trim().is_empty() {
            return bad("suite is empty");
        }
        if !self.score.is_finite() {
            return bad("score is not finite");
        }
        if self.engine_version.trim().is_empty() || self.engine_sha.trim().is_empty() {
            return bad("engine identity is incomplete");
        }
        if self.host.trim().is_empty() {
            return bad("host is empty");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Registry, RegistryConfig};
    use tempfile::TempDir;

    fn sha(c: char) -> String {
        std::iter::repeat(c).take(64).collect()
    }

    fn row(model: char, suite: &str, score: f64) -> EvalRecord {
        EvalRecord {
            model_sha: sha(model),
            suite: suite.to_string(),
            suite_manifest_sha: sha('e'),
            score,
            n: 128,
            engine_version: "0.71.0".to_string(),
            engine_sha: "0123456789abcdef0123456789abcdef01234567".to_string(),
            host: "lambda-vector".to_string(),
            ts: Utc::now(),
        }
    }

    fn registry() -> (TempDir, Registry) {
        let dir = TempDir::new().unwrap();
        let reg = Registry::open(RegistryConfig::new(dir.path())).unwrap();
        (dir, reg)
    }

    #[test]
    fn ext_09_eval_rows_round_trip_per_model() {
        let (_d, reg) = registry();
        let a1 = row('a', "perplexity/wikitext2", 7.25);
        let a2 = row('a', "apr-qa", 1.0);
        reg.record_eval(&a1).unwrap();
        reg.record_eval(&a2).unwrap();
        reg.record_eval(&row('b', "apr-qa", 0.5)).unwrap();

        let got = reg.list_evals(&sha('a')).unwrap();
        assert_eq!(got.len(), 2, "only model a's rows");
        assert!(got.contains(&a1) && got.contains(&a2), "every field survives: {got:?}");
        assert_eq!(reg.list_evals(&sha('b')).unwrap().len(), 1);
        assert!(reg.list_evals(&sha('c')).unwrap().is_empty());
        // Re-opening sees the same rows: the table is persistent, not per-connection.
        let reg2 = Registry::open(reg.config().clone()).unwrap();
        assert_eq!(reg2.list_evals(&sha('a')).unwrap().len(), 2);
    }

    #[test]
    fn ext_09_bad_rows_are_refused_and_not_written() {
        let (_d, reg) = registry();
        type Plant = (&'static str, fn(&mut EvalRecord));
        let plants: [Plant; 7] = [
            ("short model_sha", |r| r.model_sha.truncate(63)),
            ("uppercase model_sha", |r| r.model_sha = r.model_sha.to_uppercase()),
            ("bad manifest sha", |r| r.suite_manifest_sha = "sha256:x".into()),
            ("empty suite", |r| r.suite = " ".into()),
            ("NaN score", |r| r.score = f64::NAN),
            ("no engine sha", |r| r.engine_sha.clear()),
            ("no host", |r| r.host.clear()),
        ];
        for (what, plant) in plants {
            let mut r = row('a', "apr-qa", 1.0);
            plant(&mut r);
            assert!(reg.record_eval(&r).is_err(), "{what} must be refused");
        }
        assert!(reg.list_evals(&sha('a')).unwrap().is_empty(), "a refused row was written");
    }

    #[test]
    fn ext_09_registered_means_a_model_card_carries_the_sha256() {
        use crate::model::{ModelCard, ModelVersion};
        let (_d, reg) = registry();
        assert!(!reg.is_model_sha_registered(&sha('a')).unwrap());
        let mut card = ModelCard::new("produced");
        card.extra.insert("sha256".into(), serde_json::json!(sha('a')));
        reg.register_model("m", &ModelVersion::new(1, 0, 0), b"weights", card).unwrap();
        assert!(reg.is_model_sha_registered(&sha('a')).unwrap());
        assert!(!reg.is_model_sha_registered(&sha('b')).unwrap(), "another sha is not registered");
    }
}
