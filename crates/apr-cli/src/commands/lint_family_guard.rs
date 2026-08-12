//! Poka-yoke for the `apr *-lint` family's error surface (#2377-8, #2377-9).
//!
//! The two defects fixed in #2377 were not typos, they were *representable
//! states*:
//!
//! * `pub fn run(..) -> Result<(), String>` cannot carry an error class, so the
//!   dispatcher had nothing to map and collapsed missing-file, malformed-body,
//!   is-a-directory and failing-falsifier all to exit 1.
//! * `CliError::InvalidFormat` renders as `"Invalid APR format"`. Applied to a
//!   captured JSON observation it names an artifact that was never involved —
//!   no member of this family accepts a model path.
//!
//! Fixing the fourteen-odd instances would leave the *class* alive: the next
//! `*_lint.rs` file added by copy-paste reintroduces it, exactly as the first
//! ten did. These tests scan the family's source instead, so a reintroduction
//! fails `cargo test -p apr-cli --lib` rather than shipping.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// Every `*_lint.rs` under `commands/`, as (file name, source text).
    fn lint_family() -> Vec<(String, String)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir)
            .expect("commands/ is readable")
            .flatten()
        {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.ends_with("_lint.rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("lint source is readable");
            out.push((name.to_string(), src));
        }
        out.sort();
        out
    }

    /// Strip `//`-comments so a doc-comment *explaining* the ban does not trip it.
    fn code_only(src: &str) -> String {
        src.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_family_is_not_empty() {
        // Guard the guard: a scan that silently matches nothing is theater.
        let family = lint_family();
        assert!(
            family.len() >= 25,
            "expected the *-lint family, found {} files — did the scan path move?",
            family.len()
        );
        assert!(
            family.iter().any(|(n, _)| n == "nf4_lint.rs"),
            "nf4_lint.rs must be in scope"
        );
        assert!(
            family.iter().any(|(n, _)| n == "kv_timeline_lint.rs"),
            "kv_timeline_lint.rs must be in scope"
        );
    }

    #[test]
    fn no_lint_command_returns_a_bare_string_error() {
        // #2377-8. `Result<(), String>` makes the error class unrepresentable,
        // so `dispatch_analysis` can only stamp `CliError::Aprender` (exit 1) on
        // it and a CI job cannot tell "no such file" from "your falsifier
        // failed". The `run` of a lint command must return `crate::error::Result`.
        let mut offenders = Vec::new();
        for (name, src) in lint_family() {
            for line in code_only(&src).lines() {
                let t = line.trim();
                if t.starts_with("pub fn run") || t.starts_with("pub(crate) fn run") {
                    if t.contains("Result<(), String>") {
                        offenders.push(format!("{name}: {t}"));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "lint `run` must return a typed error so the exit code survives dispatch:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn no_lint_command_blames_the_apr_model_format_for_an_observation() {
        // #2377-9. `CliError::InvalidFormat` Displays as "Invalid APR format".
        // Every command in this family reads a captured JSON/JSONL/CSV body and
        // none of them takes a model path, so that wording sends the user to
        // the wrong artifact. `CliError::InvalidInput` exists for exactly this
        // and shares exit code 4.
        let mut offenders = Vec::new();
        for (name, src) in lint_family() {
            for line in code_only(&src).lines() {
                if line.contains("InvalidFormat") {
                    offenders.push(format!("{name}: {}", line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "use CliError::InvalidInput — a lint observation is not an APR model:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn the_ten_collapsed_commands_are_wired_without_the_aprender_stamp() {
        // The other half of #2377-8: even a typed `run` is neutered if the
        // dispatch arm re-collapses it. `.map_err(CliError::Aprender)` on a
        // `*_lint::run` call throws the class away again on the way out.
        let dispatch = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/dispatch_analysis.rs");
        assert!(Path::new(&dispatch).is_file(), "dispatch_analysis.rs moved");
        let src = std::fs::read_to_string(&dispatch).expect("readable");
        let code = code_only(&src);

        let lines: Vec<&str> = code.lines().collect();
        let mut offenders = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("CliError::Aprender") {
                continue;
            }
            // A dispatch arm for a lint command is at most a handful of lines;
            // 15 covers the widest one (`registry_quota_lint`) with margin.
            let start = i.saturating_sub(15);
            if let Some(hit) = lines[start..i].iter().find(|l| l.contains("_lint::run(")) {
                offenders.push(hit.trim().to_string());
            }
        }
        assert!(
            offenders.is_empty(),
            "these lint dispatch arms discard the error class (exit 1 for everything):\n{}",
            offenders.join("\n")
        );
    }

    /// The black-box statement of #2377-8, run against the commands themselves.
    ///
    /// Four *different* things a user can get wrong must not come back as the
    /// same exit code. Before the fix all ten of these commands answered `1` to
    /// every row of this table, so a CI job that ran `apr nf4-lint …` could not
    /// distinguish "the path you gave me does not exist" from "your NF4
    /// codebook diverges". Reverting the fix turns this red on the first row.
    #[test]
    fn every_converted_lint_gives_four_input_classes_four_exit_codes() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // exists, is a directory
        let tmp = std::env::temp_dir().join("apr-lint-family-guard");
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let malformed = tmp.join("malformed.json");
        std::fs::write(&malformed, "{{{not json").expect("write");
        let vacuous = tmp.join("vacuous.json");
        std::fs::write(&vacuous, "{}").expect("write");

        macro_rules! probe {
            ($m:ident, $a:ident) => {{
                use crate::commands::$m::{run, $a};
                let mk = |p: &std::path::Path| $a {
                    observation_file: p.to_string_lossy().into_owned(),
                    json: false,
                };
                (
                    stringify!($m),
                    run(mk(std::path::Path::new("/no/such/observation.json")))
                        .expect_err("a missing observation must fail")
                        .exit_code_value(),
                    run(mk(&dir))
                        .expect_err("a directory is not a body")
                        .exit_code_value(),
                    run(mk(&malformed))
                        .expect_err("a malformed body must fail")
                        .exit_code_value(),
                    run(mk(&vacuous))
                        .expect_err("an observation with no gates must fail")
                        .exit_code_value(),
                )
            }};
        }

        let rows = [
            probe!(awq_lint, AwqLintArgs),
            probe!(embeddings_lint, EmbeddingsLintArgs),
            probe!(fp8_lint, Fp8LintArgs),
            probe!(gptq_lint, GptqLintArgs),
            probe!(imatrix_lint, ImatrixLintArgs),
            probe!(nf4_lint, Nf4LintArgs),
            probe!(registry_quota_lint, RegistryQuotaLintArgs),
            probe!(rm_gc_lint, RmGcLintArgs),
            probe!(shared_cache_lint, SharedCacheLintArgs),
            probe!(unified_search_lint, UnifiedSearchLintArgs),
        ];

        assert_eq!(rows.len(), 10, "all ten converted commands must be probed");
        for (name, missing, isdir, malformed_code, vacuous_code) in rows {
            // The table in commands/lint_error.rs, made executable.
            assert_eq!(missing, 3, "{name}: missing observation must be exit 3");
            assert_eq!(isdir, 7, "{name}: an unreadable input must be exit 7");
            assert_eq!(
                malformed_code, 4,
                "{name}: unparseable observation must be exit 4"
            );
            // An observation containing none of the sections the gates need is a
            // BROKEN CAPTURE, not a contract violation: nothing was measured, so
            // nothing can have been rejected. It shares 4 with "unparseable" on
            // purpose - both mean "your capture step is broken" - and that is the
            // 4-vs-5 distinction the family exists to keep decidable.
            assert_eq!(
                vacuous_code, 4,
                "{name}: a gateless observation must be exit 4"
            );

            // The property that actually matters, stated without naming numbers:
            // the three distinct input classes may not be confusable.
            let codes = [missing, isdir, malformed_code];
            for (i, a) in codes.iter().enumerate() {
                for b in codes.iter().skip(i + 1) {
                    assert_ne!(a, b, "{name}: two input classes share an exit code");
                }
            }
            // ...and none of them may collide with a gate verdict (5), which is
            // the collapse #2377-8 was about.
            for c in codes {
                assert_ne!(c, 5, "{name}: an input error is reported as a gate verdict");
            }
        }

        std::fs::remove_file(&malformed).ok();
        std::fs::remove_file(&vacuous).ok();
    }

    /// #2377-9 stated as behaviour rather than as a source scan.
    #[test]
    fn no_lint_command_says_invalid_apr_format_for_a_malformed_observation() {
        let tmp = std::env::temp_dir().join("apr-lint-family-guard-msg");
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let bad = tmp.join("bad.json");
        std::fs::write(&bad, "{{{not json").expect("write");
        let p = bad.as_path();

        let msgs = vec![
            (
                "kv-timeline-lint",
                crate::commands::kv_timeline_lint::run(p, 0.9, false)
                    .expect_err("malformed")
                    .to_string(),
            ),
            (
                "oom-lint",
                crate::commands::oom_lint::run(p, None, false)
                    .expect_err("malformed")
                    .to_string(),
            ),
            (
                "nf4-lint",
                crate::commands::nf4_lint::run(crate::commands::nf4_lint::Nf4LintArgs {
                    observation_file: p.to_string_lossy().into_owned(),
                    json: false,
                })
                .expect_err("malformed")
                .to_string(),
            ),
        ];

        for (name, msg) in msgs {
            assert!(
                !msg.contains("Invalid APR format"),
                "{name} blamed the APR model format for a JSON observation: {msg}"
            );
        }
        std::fs::remove_file(&bad).ok();
    }
}
