//! The one door into the filesystem for the `apr *-lint` command family.
//!
//! Every member of the family consumes an *observation* — a JSON/JSONL/CSV body
//! already captured from some other `apr` command — and classifies it. None of
//! them ever opens a model. Before #2377 each command hand-rolled the same four
//! steps (exists / read / non-empty / parse) and the family drifted into two
//! incompatible dialects:
//!
//! * Ten commands (`awq`, `gptq`, `imatrix`, `fp8`, `embeddings`, `nf4`,
//!   `registry-quota`, `rm-gc`, `shared-cache`, `unified-search`) declared
//!   `run(..) -> Result<(), String>`. A `String` cannot *represent* an error
//!   class, so `dispatch_analysis` stamped every one of them
//!   `CliError::Aprender` and a missing file, a malformed body, a directory and
//!   a genuinely failing falsifier all came back as exit 1 — indistinguishable
//!   to CI.
//! * Sixteen commands reported a malformed JSON observation as
//!   `CliError::InvalidFormat`, whose Display is hardcoded to
//!   `"Invalid APR format"`. It sent users hunting for a corrupt model file when
//!   what failed to parse was, say, a captured `apr profile --kv-timeline`
//!   body — and none of these commands even accepts a model path.
//!
//! Both defects were possible only because loading was open-coded. The ten
//! commands that had no representable error class now load through this module,
//! so their exit codes are decided in one place; the rest already classified
//! correctly and only needed `InvalidInput` in place of `InvalidFormat`.
//! `lint_family_guard` keeps both defects from being written again.
//!
//! The dialect this module defines, matching `CliError::exit_code()`:
//!
//! | input                          | error                     | exit |
//! |--------------------------------|---------------------------|------|
//! | path does not exist            | `FileNotFound`            | 3    |
//! | unreadable (directory, EPERM…) | `Io`                      | 7    |
//! | empty / not parseable          | `InvalidInput`            | 4    |
//! | parsed, falsifier rejected it  | `ValidationFailed` (caller)| 5   |

use std::path::Path;

use serde_json::Value;

use crate::error::{CliError, Result};

/// Read an observation file as UTF-8 text.
///
/// # Errors
///
/// [`CliError::FileNotFound`] (exit 3) if `path` does not exist;
/// [`CliError::Io`] (exit 7) if it exists but cannot be read — a directory
/// handed to `--observation-file` lands here.
pub(crate) fn read_observation_text(path: &Path) -> Result<String> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    std::fs::read_to_string(path).map_err(CliError::Io)
}

/// Read an observation file and parse it as a single JSON document.
///
/// # Errors
///
/// Everything [`read_observation_text`] returns, plus [`CliError::InvalidInput`]
/// (exit 4) when the body is empty or is not valid JSON. Never
/// [`CliError::InvalidFormat`]: that variant's Display names the APR *model*
/// format, and no member of this family takes a model path.
pub(crate) fn read_json_observation(cmd: &str, path: &Path) -> Result<Value> {
    let text = read_observation_text(path)?;
    if text.trim().is_empty() {
        return Err(CliError::InvalidInput(format!(
            "{cmd}: observation file {} is empty",
            path.display()
        )));
    }
    serde_json::from_str(&text).map_err(|e| {
        CliError::InvalidInput(format!(
            "{cmd}: failed to parse JSON from {}: {e}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::ExitCode;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("apr-lint-input-{tag}"));
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    #[test]
    fn missing_file_is_exit_3_not_collapsed() {
        let err = read_json_observation("apr x-lint", Path::new("/no/such/observation.json"))
            .expect_err("missing file must fail");
        assert!(matches!(err, CliError::FileNotFound(_)), "got {err:?}");
        assert_eq!(err.exit_code(), ExitCode::from(3));
    }

    #[test]
    fn directory_is_exit_7_not_collapsed() {
        let d = tmpdir("isdir");
        let err = read_json_observation("apr x-lint", &d).expect_err("a directory is not a body");
        assert!(matches!(err, CliError::Io(_)), "got {err:?}");
        assert_eq!(err.exit_code(), ExitCode::from(7));
    }

    #[test]
    fn malformed_json_is_exit_4_and_never_says_apr_format() {
        let d = tmpdir("bad");
        let p = d.join("bad.json");
        std::fs::write(&p, "{{{not json").expect("write");
        let err = read_json_observation("apr x-lint", &p).expect_err("malformed body must fail");
        assert_eq!(err.exit_code(), ExitCode::from(4));
        let msg = err.to_string();
        assert!(
            !msg.contains("Invalid APR format"),
            "a JSON observation is not an APR model: {msg}"
        );
        assert!(msg.contains("apr x-lint"), "must name the linter: {msg}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn empty_file_is_rejected_not_treated_as_empty_object() {
        let d = tmpdir("empty");
        let p = d.join("empty.json");
        std::fs::write(&p, "   \n").expect("write");
        let err = read_json_observation("apr x-lint", &p).expect_err("empty body must fail");
        assert_eq!(err.exit_code(), ExitCode::from(4));
        assert!(err.to_string().contains("is empty"), "{err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn the_four_input_classes_have_four_distinct_exit_codes() {
        // The whole point of #2377-8: a CI job must be able to tell "you gave me
        // no file" from "your falsifier failed". Collapsing any two of these
        // back together turns this red.
        let d = tmpdir("distinct");
        let bad = d.join("b.json");
        std::fs::write(&bad, "nope").expect("write");
        let good = d.join("g.json");
        std::fs::write(&good, "{}").expect("write");

        let missing = read_json_observation("apr x-lint", Path::new("/no/such.json"))
            .expect_err("missing")
            .exit_code();
        let isdir = read_json_observation("apr x-lint", &d)
            .expect_err("dir")
            .exit_code();
        let malformed = read_json_observation("apr x-lint", &bad)
            .expect_err("malformed")
            .exit_code();
        let gate = CliError::ValidationFailed("gate rejected".into()).exit_code();

        let codes = [missing, isdir, malformed, gate];
        for (i, a) in codes.iter().enumerate() {
            for b in codes.iter().skip(i + 1) {
                assert_ne!(a, b, "two input classes share an exit code: {codes:?}");
            }
        }
        assert!(read_json_observation("apr x-lint", &good).is_ok());
        std::fs::remove_file(&bad).ok();
        std::fs::remove_file(&good).ok();
    }

    #[test]
    fn text_reader_agrees_with_json_reader_on_missing_and_dir() {
        let d = tmpdir("text");
        assert_eq!(
            read_observation_text(Path::new("/no/such.csv"))
                .expect_err("missing")
                .exit_code(),
            ExitCode::from(3)
        );
        assert_eq!(
            read_observation_text(&d).expect_err("dir").exit_code(),
            ExitCode::from(7)
        );
    }
}
