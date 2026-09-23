//! `apr capability` — the model-capability registry, on the CLI surface (#3856 Row 2).
//!
//! Operator, verbatim (2026-09-22): "capability.rs need deep pv SHACL support and
//! exposure on all surfaces verb + transport (http/mcp/cli)". This is the verb.
//!
//! THE FACTS ARE NOT HELD HERE. They are embedded from
//! `crates/apr-cli/contracts/apr-model-capability-v1.yaml`, the packaged mirror of
//! `contracts/apr-model-capability-v1.yaml`, which is the linted source. A second
//! hand-written copy in this file is precisely the defect the contract removes —
//! #3418/#3421 (quant dispatch duplicated across ~30 files), #3852 (a refusal
//! enumerating in prose while the code enumerates in a `match`).
//!
//! `include_str!` is compile-time, so the shipped binary carries the bytes it was
//! BUILT from. A runtime read would answer from whatever file happened to be on the
//! box, which for a published binary is no answer at all.

use crate::error::{CliError, Result};

/// The contract, embedded at compile time from the packaged mirror.
///
/// `crates/apr-cli/contracts/` and not the workspace root: `include_str!` cannot
/// escape the crate directory at package time, and the root `contracts/` is in no
/// crate's package. `tests/capability_mirror.rs` asserts the two are byte-identical,
/// in both directions, so embedding the mirror embeds the linted source.
const CAPABILITY_CONTRACT: &str = include_str!("../../contracts/apr-model-capability-v1.yaml");

/// Parsed once per invocation. A parse failure is an error, never a default: an
/// empty registry rendered as "nothing is supported" would be a confident wrong
/// answer, which is worse than a refusal.
fn contract() -> Result<serde_yaml::Value> {
    serde_yaml::from_str(CAPABILITY_CONTRACT).map_err(|e| {
        CliError::ValidationFailed(format!(
            "the embedded capability contract did not parse: {e} \
             (crates/apr-cli/contracts/apr-model-capability-v1.yaml)"
        ))
    })
}

fn rows<'a>(doc: &'a serde_yaml::Value, key: &str) -> Result<&'a Vec<serde_yaml::Value>> {
    doc.get(key).and_then(|v| v.as_sequence()).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "the embedded capability contract has no `{key}` sequence"
        ))
    })
}

fn str_of(v: &serde_yaml::Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// `apr capability` — print what this build can and cannot do, and why.
///
/// `--json` emits the contract's own sections verbatim rather than a re-shaped
/// summary, so a consumer reads the same field names the contract declares.
pub fn run(json: bool) -> Result<()> {
    let doc = contract()?;
    let ops = rows(&doc, "ops")?;
    let quants = rows(&doc, "quant_types")?;

    let cc = this_device_cc_major();
    if json {
        let out = serde_json::json!({
            "source": "contracts/apr-model-capability-v1.yaml",
            "embedded_from": "crates/apr-cli/contracts/apr-model-capability-v1.yaml",
            "this_device_cc_major": cc,
            "ops": ops,
            "quant_types": quants,
            "op_implementation": doc.get("op_implementation"),
        });
        let text = serde_json::to_string_pretty(&out).map_err(|e| {
            CliError::ValidationFailed(format!("capability registry did not serialize: {e}"))
        })?;
        println!("{text}");
        return Ok(());
    }

    println!("GPU-supported operations");
    for r in ops {
        let name = str_of(r, "op");
        let ok = r
            .get("gpu_supported")
            .and_then(serde_yaml::Value::as_bool)
            .unwrap_or(false);
        let why = str_of(r, "reason");
        if ok {
            println!("  yes  {name}");
        } else {
            // The REASON travels with the negative fact. These used to live in a
            // comment trailing an unrelated line, which is why nothing could print
            // them (#3075).
            println!("  no   {name} — {why}");
        }
    }

    let (sup, unsup): (Vec<_>, Vec<_>) = quants.iter().partition(|r| {
        r.get("gpu_supported")
            .and_then(serde_yaml::Value::as_bool)
            .unwrap_or(false)
    });

    println!("\nQuantizations with a verified GPU kernel");
    for r in &sup {
        let (name, id) = (
            str_of(r, "name"),
            r.get("ggml_type")
                .and_then(serde_yaml::Value::as_u64)
                .unwrap_or_default(),
        );
        println!("  {}", quant_line(&name, id, excluded_from(r), cc));
    }
    println!(
        "\n{} further ggml type(s) have no verified GPU kernel and run on the CPU.",
        unsup.len()
    );
    println!("Run with --json for every row, including the reason each one carries.");
    Ok(())
}

/// `gpu_excluded_from_cc_major`: the compute-capability major at and above which this type's kernels do not load
/// and it runs on the CPU (#3968/#4096). The dispatch gate reads the same fact from `GPU_QTYPES_UNLOADABLE_AT_CC`;
/// FALSIFY-CAP's row test binds the two.
fn excluded_from(r: &serde_yaml::Value) -> Option<i64> {
    r.get("gpu_excluded_from_cc_major")
        .and_then(serde_yaml::Value::as_i64)
}

/// One supported-quant line. A per-capability exclusion is RED, never a footnote: it is stated on every host, and on
/// a host at or above the threshold the line says `no`, because there the type runs on the CPU.
fn quant_line(name: &str, id: u64, excluded_from: Option<i64>, cc: Option<i32>) -> String {
    match excluded_from {
        None => format!("yes  {name:<8} ggml type {id}"),
        Some(m) if cc.is_some_and(|c| i64::from(c) >= m) => format!(
            "no   {name:<8} ggml type {id} — refused on THIS device (compute capability {}; its kernels do not load at >= {m}, #4096): runs on the CPU",
            cc.unwrap_or_default()
        ),
        Some(m) => format!(
            "yes  {name:<8} ggml type {id} — except compute capability >= {m}, where its kernels do not load (#4096) and it runs on the CPU"
        ),
    }
}

/// The capability the dispatch gate itself sees (`realizar::gguf::device_cc_major`), or `None` in a build without
/// inference or on a host without a device.
fn this_device_cc_major() -> Option<i32> {
    #[cfg(feature = "inference")]
    {
        realizar::gguf::device_cc_major()
    }
    #[cfg(not(feature = "inference"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #3968/#4096: a type excluded at cc >= 12 is RED on such a device and stated as an exception elsewhere; a
    /// type with no exclusion is unaffected by the device.
    #[test]
    fn a_per_capability_exclusion_renders_red_on_a_device_at_or_above_it() {
        assert!(quant_line("IQ3_S", 21, Some(12), Some(12)).starts_with("no   IQ3_S"));
        assert!(quant_line("IQ3_S", 21, Some(12), Some(12)).contains("compute capability 12"));
        assert!(quant_line("IQ3_S", 21, Some(12), Some(8)).starts_with("yes  IQ3_S"));
        assert!(
            quant_line("IQ3_S", 21, Some(12), Some(8)).contains("except compute capability >= 12")
        );
        assert!(quant_line("IQ3_S", 21, Some(12), None).contains("except compute capability >= 12"));
        assert_eq!(
            quant_line("Q4_K", 12, None, Some(12)),
            "yes  Q4_K     ggml type 12"
        );
    }

    /// The embedded contract carries the exclusion for exactly IQ3_S and IQ2_S — so the CLI renders what the gate does.
    #[test]
    fn the_embedded_contract_excludes_iq3s_and_iq2s_at_cc12() {
        let doc = contract().expect("embedded contract parses");
        let got: Vec<(String, i64)> = rows(&doc, "quant_types")
            .expect("quant_types")
            .iter()
            .filter_map(|r| excluded_from(r).map(|m| (str_of(r, "name"), m)))
            .collect();
        assert_eq!(
            got,
            vec![("IQ3_S".to_string(), 12), ("IQ2_S".to_string(), 12)]
        );
    }

    /// The embedded bytes must parse. If this fails the binary ships a contract it
    /// cannot read, and every caller gets an error instead of an answer.
    #[test]
    fn the_embedded_contract_parses_and_is_not_empty() {
        let doc = contract().expect("embedded contract parses");
        assert!(
            !rows(&doc, "ops").expect("ops").is_empty(),
            "the embedded contract declares no ops"
        );
        assert!(
            !rows(&doc, "quant_types").expect("quant_types").is_empty(),
            "the embedded contract declares no quant types"
        );
    }

    /// Anti-vacuity: the embedded bytes are the CONTRACT, not some other YAML that
    /// happens to parse. Keyed on content the contract is required to carry.
    #[test]
    fn the_embedded_bytes_are_the_capability_contract() {
        assert!(
            CAPABILITY_CONTRACT.contains("apr-model-capability"),
            "the embedded file does not name itself — include_str! may be pointed \
             at the wrong path"
        );
        let doc = contract().expect("parses");
        let ops = rows(&doc, "ops").expect("ops");
        assert!(
            ops.iter().any(|r| str_of(r, "op") == "LayerNorm"),
            "LayerNorm is absent from the embedded ops — this build would report a \
             capability set that the contract does not declare"
        );
    }

    /// Every unsupported op prints a reason. An op that renders as
    /// `no   Foo — ` puts the user back where the comment left them.
    #[test]
    fn every_unsupported_op_has_a_reason_to_print() {
        let doc = contract().expect("parses");
        for r in rows(&doc, "ops").expect("ops") {
            let ok = r
                .get("gpu_supported")
                .and_then(serde_yaml::Value::as_bool)
                .unwrap_or(false);
            if !ok {
                assert!(
                    !str_of(r, "reason").is_empty(),
                    "op {} is unsupported with nothing to print",
                    str_of(r, "op")
                );
            }
        }
    }
}
