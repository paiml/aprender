//! `apr test llm parity`, the front door of `apr_test::llm::parity`
//! (#4971 V3-d): the perf041 batched-parity probe, ported from
//! `scripts/perf041_batched_parity_probe.py` (C301).
//!
//! It decodes one prompt twice alone (`m = 1`, the reference), then as `c`
//! identical concurrent requests for each `c` of the ladder, reads the batch
//! each band formed from the server's log, and writes the witness that
//! `apr test llm shape-check` reads. Every number it judges by comes from
//! `scripts/perf-matrix.yaml`, compiled in; only the ladder can be overridden.
//!
//! The script's exit codes (0 PASS, 1 FAIL, 2 UNMEASURABLE) map onto apr's:
//!
//! | Verdict | apr exit |
//! |---|---|
//! | PASS | 0 |
//! | FAIL: a batch's slots parted before `declared_min`, or a slot froze | 5 (`ValidationFailed`) |
//! | UNMEASURABLE: the run could not decide | 8 (`InferenceFailed`) |
//! | a server log, binary or model that is not a readable file | 3 (`FileNotFound` / `NotAFile`) or 7 (`Io`) |
//!
//! FAIL and UNMEASURABLE never share a code, and the witness is written
//! before the exit, whichever verdict it carries. A file that cannot be read
//! refuses the run before any request is fired.
//!
//! This is not a gate. No merge, queue or release job runs it.

use std::path::Path;

use apr_test::llm::client::LlmClient;
use apr_test::llm::{ParityProbe, ParityRun};
use apr_test::perf_gate::parity::{env_block, prompt_sha256, PROBE, WITNESS_VERSION};
use apr_test::perf_gate::receipt::sha256_file;
use apr_test::perf_gate::{BatchInvariance, ProbeBand, ProbeMatrix, ProbeModel, ProbeWitness};

use crate::error::{CliError, Result};
use crate::LlmSubcommand;

/// Who and what was measured, as the witness records it. A field nobody gave
/// stays `None`, and the V3 shape check names it.
#[derive(Debug, Clone, PartialEq)]
struct Identity {
    host: Option<String>,
    commit: Option<String>,
    binary_sha256: Option<String>,
    model: ProbeModel,
}

/// Run the probe, print every band, write the witness when asked, and exit
/// by the verdict.
///
/// # Errors
/// FAIL is `ValidationFailed` and UNMEASURABLE `InferenceFailed`, each after
/// the witness is written. A server log, binary or model that is not a
/// readable file refuses the run first.
pub(crate) fn run(command: &LlmSubcommand) -> Result<()> {
    let LlmSubcommand::Parity {
        url,
        served_model,
        server_log,
        prompt,
        ladder,
        json,
        host,
        commit,
        binary,
        model,
    } = command
    else {
        return Err(CliError::InvalidInput(
            "`apr test llm`: only `parity` routes to the parity probe".to_string(),
        ));
    };
    let probe = configure(prompt, ladder, server_log)?;
    let identity = identify(
        host.as_deref(),
        commit.as_deref(),
        binary.as_deref(),
        model.as_deref(),
    )?;
    let client = LlmClient::new(url.as_str(), served_model.as_str());
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("tokio runtime: {e}")))?;
    let witness = witness(&probe, identity, rt.block_on(probe.run(&client)));
    println!("{}", report(&witness));
    if let Some(path) = json {
        write(path, &witness)?;
    }
    outcome(&witness)
}

/// The probe as the matrix configures it. A non-empty `ladder` replaces the
/// declared one.
///
/// # Errors
/// `FileNotFound` / `NotAFile` for a server log that is not a file, since a
/// log the probe cannot read would turn every band UNMEASURABLE.
/// `InvalidInput` for a matrix that cannot configure a run.
fn configure(prompt: &str, ladder: &[u32], server_log: &Path) -> Result<ParityProbe> {
    readable_file(server_log)?;
    let matrix = ProbeMatrix::from_matrix().map_err(CliError::InvalidInput)?;
    Ok(ParityProbe {
        prompt: prompt.to_string(),
        ladder: if ladder.is_empty() {
            matrix.ladder
        } else {
            ladder.to_vec()
        },
        policy: matrix.policy,
        sampler: matrix.sampler,
        server_log: server_log.to_path_buf(),
    })
}

/// The identity block. `host` falls back to `PERF041_HOST`, then this host's
/// name; `commit` to `PERF041_COMMIT`. The model is recorded by file name
/// only, never a host path, beside its sha256.
///
/// # Errors
/// As [`hash`], for `binary` or `model`.
fn identify(
    host: Option<&str>,
    commit: Option<&str>,
    binary: Option<&Path>,
    model: Option<&Path>,
) -> Result<Identity> {
    Ok(Identity {
        host: given_or_env(host, "PERF041_HOST").or_else(this_host),
        commit: given_or_env(commit, "PERF041_COMMIT"),
        binary_sha256: binary.map(hash).transpose()?,
        model: ProbeModel {
            path: model
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned()),
            sha256: model.map(hash).transpose()?,
        },
    })
}

/// The flag's value, else the environment's; an empty value counts as unset.
fn given_or_env(given: Option<&str>, key: &str) -> Option<String> {
    given
        .map(str::to_string)
        .or_else(|| std::env::var(key).ok())
        .filter(|value| !value.is_empty())
}

/// This host's name, or `None`, so a witness never records a made-up host.
fn this_host() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|name| name.trim().to_string())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .filter(|name| !name.is_empty())
}

/// `path`'s sha256.
///
/// # Errors
/// `FileNotFound` / `NotAFile` when it is not a file; `Io` when it cannot be read.
fn hash(path: &Path) -> Result<String> {
    readable_file(path)?;
    Ok(sha256_file(path)?)
}

/// Refuse a path that is not a file, naming which way.
fn readable_file(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

/// The witness for one run.
fn witness(probe: &ParityProbe, identity: Identity, run: ParityRun) -> ProbeWitness {
    ProbeWitness {
        witness_version: WITNESS_VERSION,
        probe: PROBE.to_string(),
        host: identity.host,
        commit: identity.commit,
        binary_sha256: identity.binary_sha256,
        model: identity.model,
        prompt_sha256: prompt_sha256(&probe.prompt),
        sampler: probe.sampler,
        declared_min: probe.policy.declared_min,
        env: env_block(),
        reference: run.reference,
        bands: run.bands,
        result: run.result,
    }
}

/// The printed run: the verdict, the reference, then one line per band.
fn report(witness: &ProbeWitness) -> String {
    let reference = &witness.reference;
    let head = format!(
        "perf041 batched parity: {} (declared_min {}, n_predict {})",
        witness.result.wire_token(),
        witness.declared_min,
        witness.sampler.max_tokens
    );
    let reference_line = format!(
        "  m=1 reference: {} tokens={} self_divergence_at={}{}",
        if reference.stable {
            "stable"
        } else {
            "NOT STABLE"
        },
        or_dash(reference.tokens),
        or_dash(reference.self_divergence_at),
        reason(reference.reason.as_deref())
    );
    [head, reference_line]
        .into_iter()
        .chain(witness.bands.iter().map(band_line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One band, as printed.
fn band_line(band: &ProbeBand) -> String {
    format!(
        "  c={} m_formed={} {} intra_agree_to={} divergence_at={} top2_margin={}{}",
        band.c,
        band.m_formed,
        band.result.wire_token(),
        or_dash(band.intra_agree_to),
        or_dash(band.divergence_at),
        or_dash(band.top2_margin_at_divergence),
        reason(band.reason.as_deref())
    )
}

/// A value, or `-` when there is none.
fn or_dash(value: Option<impl std::fmt::Display>) -> String {
    value.map_or_else(|| "-".to_string(), |v| v.to_string())
}

/// ` (reason)`, or nothing.
fn reason(reason: Option<&str>) -> String {
    reason.map_or_else(String::new, |r| format!(" ({r})"))
}

/// Write the witness as pretty JSON.
///
/// # Errors
/// `Io` when the file cannot be written.
fn write(path: &Path, witness: &ProbeWitness) -> Result<()> {
    let json = serde_json::to_string_pretty(witness)
        .map_err(|e| CliError::InvalidInput(format!("perf041 witness as JSON: {e}")))?;
    std::fs::write(path, json + "\n")?;
    println!("witness: {}", path.display());
    Ok(())
}

/// The verdict as apr's result. Only PASS is `Ok`.
fn outcome(witness: &ProbeWitness) -> Result<()> {
    match witness.result {
        BatchInvariance::Pass => Ok(()),
        BatchInvariance::Fail => Err(CliError::ValidationFailed(format!(
            "perf041 FAIL: {}",
            why(witness)
        ))),
        BatchInvariance::Unmeasurable => Err(CliError::InferenceFailed(format!(
            "perf041 UNMEASURABLE: {}",
            why(witness)
        ))),
    }
}

/// The first reason behind the verdict: a band that carries it, else the
/// reference, else the one case that has neither (no band at `c > 1`).
fn why(witness: &ProbeWitness) -> String {
    witness
        .bands
        .iter()
        .filter(|band| band.result == witness.result)
        .find_map(|band| band.reason.as_ref().map(|r| format!("c={}: {r}", band.c)))
        .or_else(|| {
            witness
                .reference
                .reason
                .as_ref()
                .map(|r| format!("m=1 reference: {r}"))
        })
        .unwrap_or_else(|| "no band at c > 1 was measured".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apr_test::perf_gate::ProbeReference;
    use clap::{FromArgMatches, Subcommand};

    fn reference(stable: bool, reason: Option<&str>) -> ProbeReference {
        ProbeReference {
            stable,
            tokens: Some(128),
            self_divergence_at: None,
            reason: reason.map(str::to_string),
        }
    }

    fn band(c: u32, result: BatchInvariance, reason: Option<&str>) -> ProbeBand {
        ProbeBand {
            c,
            m_formed: c,
            result,
            divergence_at: Some(128),
            top2_margin_at_divergence: None,
            intra_agree_to: Some(128),
            max_constant_run: Some(1),
            declared_min: 64,
            max_constant_run_declared: 16,
            reason: reason.map(str::to_string),
            slots: vec![],
        }
    }

    fn log() -> tempfile::NamedTempFile {
        let log = tempfile::NamedTempFile::new().expect("temp log");
        std::fs::write(log.path(), "server starting\n").expect("seed log");
        log
    }

    fn witness_of(run: ParityRun) -> ProbeWitness {
        let log = log();
        let probe = configure("p", &[], log.path()).expect("configures");
        let identity = identify(Some("box"), Some("abc123"), None, None).expect("identity");
        witness(&probe, identity, run)
    }

    #[test]
    fn the_three_verdicts_exit_apart() {
        let pass = witness_of(ParityRun {
            reference: reference(true, None),
            bands: vec![band(4, BatchInvariance::Pass, None)],
            result: BatchInvariance::Pass,
        });
        assert!(outcome(&pass).is_ok());

        let fail = witness_of(ParityRun {
            reference: reference(true, None),
            bands: vec![
                band(4, BatchInvariance::Pass, None),
                band(8, BatchInvariance::Fail, Some("slot 3 froze")),
            ],
            result: BatchInvariance::Fail,
        });
        let err = outcome(&fail).expect_err("FAIL");
        assert_eq!(err.exit_code_value(), 5, "{err}");
        assert!(err.to_string().contains("c=8: slot 3 froze"), "{err}");

        let unstable = witness_of(ParityRun {
            reference: reference(false, Some("the two decodes parted at 2")),
            bands: vec![],
            result: BatchInvariance::Unmeasurable,
        });
        let err = outcome(&unstable).expect_err("UNMEASURABLE");
        assert_eq!(err.exit_code_value(), 8, "{err}");
        assert!(
            err.to_string()
                .contains("m=1 reference: the two decodes parted at 2"),
            "{err}"
        );

        let no_batch = witness_of(ParityRun {
            reference: reference(true, None),
            bands: vec![band(1, BatchInvariance::Pass, None)],
            result: BatchInvariance::Unmeasurable,
        });
        let err = outcome(&no_batch).expect_err("UNMEASURABLE");
        assert_eq!(err.exit_code_value(), 8, "{err}");
        assert!(err.to_string().contains("no band at c > 1"), "{err}");
    }

    #[test]
    fn a_server_log_that_is_not_a_file_refuses_the_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("server.log");
        assert!(matches!(
            configure("p", &[], &missing),
            Err(CliError::FileNotFound(p)) if p == missing
        ));
        assert!(matches!(
            configure("p", &[], dir.path()),
            Err(CliError::NotAFile(_))
        ));
    }

    #[test]
    fn the_matrix_configures_the_probe_and_the_ladder_flag_replaces_its_ladder() {
        let log = log();
        let matrix = ProbeMatrix::from_matrix().expect("the shipped matrix");
        let declared = configure("p", &[], log.path()).expect("configures");
        assert_eq!(declared.ladder, matrix.ladder);
        assert_eq!(declared.policy, matrix.policy);
        assert_eq!(declared.sampler, matrix.sampler);
        assert_eq!(declared.server_log, log.path());
        let overridden = configure("p", &[2, 3], log.path()).expect("configures");
        assert_eq!(overridden.ladder, vec![2, 3]);
        assert_eq!(overridden.policy, matrix.policy);
    }

    #[test]
    fn identity_records_the_model_by_file_name_and_hashes_both_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("model.gguf");
        let binary = dir.path().join("apr");
        std::fs::write(&model, "abc").expect("model");
        std::fs::write(&binary, "").expect("binary");
        let identity =
            identify(Some("box"), Some("c0ffee"), Some(&binary), Some(&model)).expect("identity");
        assert_eq!(identity.host.as_deref(), Some("box"));
        assert_eq!(identity.commit.as_deref(), Some("c0ffee"));
        assert_eq!(identity.model.path.as_deref(), Some("model.gguf"));
        assert_eq!(
            identity.model.sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            identity.binary_sha256.as_deref(),
            Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        let missing = dir.path().join("nope.gguf");
        assert!(matches!(
            identify(Some("box"), None, None, Some(&missing)),
            Err(CliError::FileNotFound(p)) if p == missing
        ));
        let bare = identify(Some("box"), Some(""), None, None).expect("identity");
        assert_eq!(
            bare.model,
            ProbeModel {
                path: None,
                sha256: None
            }
        );
        assert_eq!(given_or_env(Some(""), "PERF041_NO_SUCH_KEY"), None);
    }

    #[test]
    fn the_report_names_the_verdict_the_reference_and_every_band() {
        let w = witness_of(ParityRun {
            reference: reference(true, None),
            bands: vec![
                band(1, BatchInvariance::Pass, None),
                band(4, BatchInvariance::Unmeasurable, Some("no batch formed")),
            ],
            result: BatchInvariance::Unmeasurable,
        });
        let text = report(&w);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert!(lines[0].contains("UNMEASURABLE"), "{text}");
        assert!(
            lines[1].contains("m=1 reference: stable tokens=128"),
            "{text}"
        );
        assert!(lines[2].starts_with("  c=1 m_formed=1 PASS"), "{text}");
        assert!(lines[3].contains("c=4 m_formed=4 UNMEASURABLE"), "{text}");
        assert!(
            lines[3].ends_with("top2_margin=- (no batch formed)"),
            "{text}"
        );
    }

    #[test]
    fn the_witness_is_written_as_json_that_reads_back_whole() {
        let w = witness_of(ParityRun {
            reference: reference(true, None),
            bands: vec![band(4, BatchInvariance::Pass, None)],
            result: BatchInvariance::Pass,
        });
        assert_eq!(w.witness_version, WITNESS_VERSION);
        assert_eq!(w.probe, PROBE);
        assert_eq!(w.prompt_sha256, prompt_sha256("p"));
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("witness.json");
        write(&path, &w).expect("written");
        let back: ProbeWitness =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parses");
        assert_eq!(back, w);
    }

    fn parse(argv: &[&str]) -> std::result::Result<LlmSubcommand, clap::Error> {
        let matches = LlmSubcommand::augment_subcommands(clap::Command::new("llm"))
            .try_get_matches_from(std::iter::once("llm").chain(argv.iter().copied()))?;
        LlmSubcommand::from_arg_matches(&matches)
    }

    #[test]
    fn the_flags_parse_with_the_scripts_defaults() {
        let parsed = parse(&[
            "parity",
            "--server-log",
            "/tmp/server.log",
            "--ladder",
            "1,4",
        ])
        .expect("parses");
        let LlmSubcommand::Parity {
            url,
            served_model,
            prompt,
            ladder,
            json,
            host,
            ..
        } = parsed
        else {
            panic!("not `apr test llm parity`");
        };
        assert_eq!(url.as_str(), "http://127.0.0.1:8080");
        assert_eq!(served_model.as_str(), "q");
        assert_eq!(prompt.as_str(), "Write an essay on compilers.");
        assert_eq!(ladder, vec![1, 4]);
        assert!(json.is_none());
        assert!(host.is_none());
        assert!(parse(&["parity", "--server-log", "x", "--ladder", "0"]).is_err());
        assert!(parse(&["parity"]).is_err());
    }
}
