//! `apr model gate`: model gates M0..M6 over a release dir (EXT-001 §3.6, row EXT-12, aprender#4394).
//!
//! The gate reads three things and writes one:
//! - `model-release-v1.json` in the release dir (§3.4; EXT-11's `apr model pack` writes it);
//! - an evidence file for what cannot be recomputed here: the M1 llama.cpp parity
//!   measurement, the M2 sealed-suite results with the C7 arms, the M3 probe answers and
//!   the receipt ids the card may cite;
//! - pacha, through [`GateEnv`], for the lineage runs (M0) and dataset manifests (M4);
//! - it writes `model-gate-receipt-v1`, with one row per gate and no timestamp, so the
//!   same inputs give the same bytes.
//!
//! Missing evidence is RED, never a skip. M-CR (EXT-13) and M7 (EXT-15) are separate rows.

use super::model_gate_m1b::{self, M1bEvidence, M1bReport};
use super::model_gate_m2::arms::{gate as m2_gate, Arm, GateReport};
use super::model_gate_m2::{M2Prereg, ReleaseClass, Suite};
use pacha::data::{AdmittedManifest, SealedItems};
use pacha::registry::is_sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

/// The manifest file the gate reads.
pub(crate) const MANIFEST: &str = "model-release-v1.json";
/// The receipt file the gate writes.
pub(crate) const RECEIPT: &str = "model-gate-receipt-v1.json";
/// §3.6 M1: cosine floor against llama.cpp (`ds.yaml`, `[C]`).
pub(crate) const M1_MIN_COSINE: f64 = 0.98;
/// §3.6 M1: the llama.cpp commit parity is measured against.
pub(crate) const M1_LLAMA_CPP_PIN: &str = "d1d3c3396";
/// The files a release dir may hold beside those the manifest lists: the gate's own
/// inputs and output.
pub(crate) const GATE_FILES: [&str; 2] = [MANIFEST, RECEIPT];

/// One file of the release (§3.4 `files`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReleaseFile {
    pub name: String,
    pub format: String,
    #[serde(default)]
    pub quant: Option<String>,
    pub bytes: u64,
    pub sha256: String,
}

/// §3.4 `base`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Base {
    pub hf_id: String,
    pub revision: String,
    pub sha256: String,
}

/// §3.4 `engine`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Engine {
    pub apr_version: String,
    pub crate_tarball_sha256: String,
}

/// §3.4 `license`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct License {
    pub spdx_or_name: String,
    pub upstream_notice_sha256: String,
}

/// The fields of `model-release-v1.json` the gates read. `gates` and `recipe` are not
/// read: the first is what this command produces, the second no gate checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReleaseManifest {
    pub line: String,
    pub version: String,
    pub channel: String,
    pub files: Vec<ReleaseFile>,
    pub base: Base,
    pub lineage: Vec<String>,
    pub datasets: Vec<String>,
    pub engine: Engine,
    pub license: License,
}

/// M1 evidence: the parity measurement on the release's exact bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct M1Evidence {
    pub cosine: f64,
    pub llama_cpp_commit: String,
    /// Report-only.
    #[serde(default)]
    pub greedy_token_agreement: Option<f64>,
}

/// M2 evidence: the sealed suites against the incumbent, and the C7 arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct M2Evidence {
    pub class: ReleaseClass,
    pub suites: Vec<Suite>,
    pub arms: Vec<Arm>,
}

/// One probe of the fixed M3 set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Probe {
    pub id: String,
    pub run_answered: bool,
    pub serve_answered: bool,
}

/// M3 evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct M3Evidence {
    pub probes: Vec<Probe>,
    /// Review-lane verdict parse rate: reported, never blocks.
    #[serde(default)]
    pub parse_rate: Option<f64>,
}

/// What the gate cannot recompute. Every section is optional in the file, and every
/// absent section is RED.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GateEvidence {
    #[serde(default)]
    pub m1: Option<M1Evidence>,
    /// M1b artifact quality (EXT-27): KL and top-1 vs BF16, per arm, ratcheted.
    #[serde(default)]
    pub m1b: Option<M1bEvidence>,
    #[serde(default)]
    pub m2: Option<M2Evidence>,
    #[serde(default)]
    pub m3: Option<M3Evidence>,
    /// Receipt ids a card number may cite (I-10).
    #[serde(default)]
    pub receipt_ids: Vec<String>,
}

/// The pacha lookups the gates need.
pub(crate) trait GateEnv {
    /// Whether a lineage run id resolves to a recorded run.
    fn run_resolves(&self, run_id: &str) -> Result<bool, String>;
    /// The registered dataset manifest with this canonical hash.
    fn dataset_manifest(&self, canonical_sha256: &str) -> Result<Option<AdmittedManifest>, String>;
}

/// One gate's row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct GateRow {
    pub gate: &'static str,
    pub green: bool,
    /// Why it is red; for a green gate, what it checked.
    pub findings: Vec<String>,
}

/// `model-gate-receipt-v1`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct GateReceipt {
    pub schema: &'static str,
    pub line: String,
    pub version: String,
    pub manifest_sha256: String,
    pub gates: Vec<GateRow>,
    pub m1: Option<M1Evidence>,
    pub m1b: Option<M1bReport>,
    pub m2: Option<GateReport>,
    pub m3_parse_rate: Option<f64>,
    pub sealed_items_checked: usize,
    pub sealed_set_sha256: String,
    pub all_green: bool,
}

/// What the gate was pointed at.
pub(crate) struct GateInputs<'a> {
    pub dir: &'a Path,
    pub evidence: &'a GateEvidence,
    pub sealed: &'a SealedItems,
    /// The released `apr` crate tarball, re-hashed against `engine.crate_tarball_sha256`.
    pub engine_tarball: Option<&'a Path>,
    pub env: &'a dyn GateEnv,
}

pub(crate) fn sha256_file(path: &Path) -> std::io::Result<(u64, String)> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let n = std::io::copy(&mut f, &mut h)?;
    Ok((n, hex_lower(&h.finalize())))
}

fn hex_lower(b: &[u8]) -> String {
    use std::fmt::Write;
    b.iter()
        .fold(String::with_capacity(b.len() * 2), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
}

pub(crate) fn row(gate: &'static str, findings: Vec<String>, checked: String) -> GateRow {
    if findings.is_empty() {
        GateRow {
            gate,
            green: true,
            findings: vec![checked],
        }
    } else {
        GateRow {
            gate,
            green: false,
            findings,
        }
    }
}

fn clean_name(n: &str) -> bool {
    !n.is_empty() && !n.contains('/') && !n.contains('\\') && n != "." && n != ".."
}

/// M0 identity: every file hashed, the engine a verified released tarball, the base
/// pinned, the lineage resolving (I-1, I-2, I-5).
fn m0(m: &ReleaseManifest, inp: &GateInputs<'_>) -> GateRow {
    let mut red = Vec::new();
    if m.files.is_empty() {
        red.push("manifest lists no files".into());
    }
    let mut listed = BTreeSet::new();
    for f in &m.files {
        if !clean_name(&f.name) || !listed.insert(f.name.as_str()) {
            red.push(format!("{:?}: not a unique plain file name", f.name));
            continue;
        }
        if GATE_FILES.contains(&f.name.as_str()) {
            red.push(format!(
                "{}: the gate's own file cannot be a release file",
                f.name
            ));
            continue;
        }
        match sha256_file(&inp.dir.join(&f.name)) {
            Err(e) => red.push(format!("{}: {e}", f.name)),
            Ok((n, sha)) => {
                if n != f.bytes {
                    red.push(format!("{}: {n} bytes, manifest says {}", f.name, f.bytes));
                }
                if sha != f.sha256 {
                    red.push(format!(
                        "{}: sha256 {sha}, manifest says {}",
                        f.name, f.sha256
                    ));
                }
            }
        }
    }
    match std::fs::read_dir(inp.dir) {
        Err(e) => red.push(format!("release dir: {e}")),
        Ok(entries) => {
            let mut extra: Vec<String> = entries
                .filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !listed.contains(n.as_str()) && !GATE_FILES.contains(&n.as_str()))
                .collect();
            extra.sort();
            for n in extra {
                red.push(format!(
                    "{n}: in the release dir but not hashed in the manifest"
                ));
            }
        }
    }
    if m.engine.apr_version.trim().is_empty() {
        red.push("engine.apr_version is empty (I-5)".into());
    }
    if !is_sha256_hex(&m.engine.crate_tarball_sha256) {
        red.push("engine.crate_tarball_sha256 is not 64 lowercase hex (I-5)".into());
    }
    match inp.engine_tarball {
        None => red.push("engine tarball not supplied: the producing apr is unverified".into()),
        Some(p) => match sha256_file(p) {
            Err(e) => red.push(format!("engine tarball {}: {e}", p.display())),
            Ok((_, sha)) if sha != m.engine.crate_tarball_sha256 => red.push(format!(
                "engine tarball sha256 {sha}, manifest says {}",
                m.engine.crate_tarball_sha256
            )),
            Ok(_) => {}
        },
    }
    let b = &m.base;
    if b.hf_id.trim().is_empty() || b.revision.trim().is_empty() || !is_sha256_hex(&b.sha256) {
        red.push("base needs hf_id, revision and a sha256".into());
    }
    if m.lineage.is_empty() {
        red.push("lineage is empty: an orphan model (I-1)".into());
    }
    for id in &m.lineage {
        match inp.env.run_resolves(id) {
            Ok(true) => {}
            Ok(false) => red.push(format!("lineage run {id} does not resolve (I-1)")),
            Err(e) => red.push(format!("lineage run {id}: {e}")),
        }
    }
    let checked = format!(
        "{} files hashed, engine {} verified, {} lineage runs resolve",
        m.files.len(),
        m.engine.apr_version,
        m.lineage.len()
    );
    row("M0", red, checked)
}

/// M1 parity: cosine ≥ 0.98 against the pinned llama.cpp.
fn m1(ev: Option<&M1Evidence>) -> GateRow {
    let Some(e) = ev else {
        return row("M1", vec!["no M1 parity evidence".into()], String::new());
    };
    let mut red = Vec::new();
    if !e.llama_cpp_commit.starts_with(M1_LLAMA_CPP_PIN) {
        red.push(format!(
            "parity measured against llama.cpp {}, the pin is {M1_LLAMA_CPP_PIN}",
            e.llama_cpp_commit
        ));
    }
    // A NaN cosine is not ≥ the floor.
    if e.cosine.is_nan() || e.cosine < M1_MIN_COSINE || e.cosine > 1.0 + 1e-9 {
        red.push(format!(
            "cosine {} is below {M1_MIN_COSINE} or not a cosine",
            e.cosine
        ));
    }
    row(
        "M1",
        red,
        format!("cosine {} vs llama.cpp {M1_LLAMA_CPP_PIN}", e.cosine),
    )
}

/// M2 quality with the C7 stock arm.
fn m2(pre: &M2Prereg, ev: Option<&M2Evidence>) -> (GateRow, Option<GateReport>) {
    let Some(e) = ev else {
        return (
            row(
                "M2",
                vec!["no M2 sealed-suite evidence".into()],
                String::new(),
            ),
            None,
        );
    };
    match m2_gate(pre, e.class, &e.suites, &e.arms) {
        Err(msg) => (row("M2", vec![msg], String::new()), None),
        Ok(r) => {
            let red = if r.verdict == super::model_gate_m2::Verdict::Promote {
                Vec::new()
            } else {
                vec![format!("M2 verdict {}", verdict_text(&r.verdict))]
            };
            let checked = format!("{} suites, {} arms: promote", e.suites.len(), e.arms.len());
            (row("M2", red, checked), Some(r))
        }
    }
}

fn verdict_text(v: &super::model_gate_m2::Verdict) -> String {
    use super::model_gate_m2::Verdict;
    match v {
        Verdict::Reject { reason } => format!("reject: {reason}"),
        other => other.to_string(),
    }
}

/// M3 smoke: every probe answered by both `apr run` and `apr serve`.
fn m3(ev: Option<&M3Evidence>) -> GateRow {
    let Some(e) = ev else {
        return row("M3", vec!["no M3 probe evidence".into()], String::new());
    };
    let mut red = Vec::new();
    if e.probes.is_empty() {
        red.push("the probe set is empty".into());
    }
    let mut seen = BTreeSet::new();
    for p in &e.probes {
        if !seen.insert(p.id.as_str()) {
            red.push(format!("probe {} appears twice", p.id));
        }
        if !p.run_answered {
            red.push(format!("probe {}: apr run did not answer", p.id));
        }
        if !p.serve_answered {
            red.push(format!("probe {}: apr serve did not answer", p.id));
        }
    }
    let rate = e.parse_rate.map_or("unmeasured".into(), |r| r.to_string());
    row(
        "M3",
        red,
        format!(
            "{} probes answered by run and serve; parse rate {rate}",
            e.probes.len()
        ),
    )
}

/// M4 contamination: every training dataset re-admitted against today's sealed set (I-11).
fn m4(m: &ReleaseManifest, inp: &GateInputs<'_>) -> GateRow {
    let mut red = Vec::new();
    if inp.sealed.is_empty() {
        red.push("no sealed items: I-11 would check nothing".into());
    }
    for sha in &m.datasets {
        match inp.env.dataset_manifest(sha) {
            Err(e) => red.push(format!("dataset {sha}: {e}")),
            Ok(None) => red.push(format!("dataset {sha} is not registered")),
            Ok(Some(a)) => {
                if let Err(e) = a.manifest.admit(inp.sealed) {
                    red.push(format!("dataset {sha}: {e}"));
                }
            }
        }
    }
    let checked = format!(
        "{} datasets against {} sealed items (set {})",
        m.datasets.len(),
        inp.sealed.len(),
        inp.sealed.set_sha256()
    );
    row("M4", red, checked)
}

fn file_starting<'m>(m: &'m ReleaseManifest, stem: &str) -> Option<&'m ReleaseFile> {
    m.files
        .iter()
        .find(|f| f.name.to_ascii_uppercase().starts_with(stem))
}

/// M5 license: the upstream LICENSE and NOTICE ship, hashed, the NOTICE the one recorded (I-13).
fn m5(m: &ReleaseManifest) -> GateRow {
    let mut red = Vec::new();
    if m.license.spdx_or_name.trim().is_empty() {
        red.push("license.spdx_or_name is empty".into());
    }
    if file_starting(m, "LICENSE").is_none() {
        red.push("no LICENSE file in the release (I-13)".into());
    }
    match file_starting(m, "NOTICE") {
        None => red.push("no NOTICE file in the release (I-13)".into()),
        Some(f) if f.sha256 != m.license.upstream_notice_sha256 => red.push(format!(
            "{} sha256 {}, the upstream NOTICE is {}",
            f.name, f.sha256, m.license.upstream_notice_sha256
        )),
        Some(_) => {}
    }
    row(
        "M5",
        red,
        format!(
            "license {} with upstream LICENSE and NOTICE",
            m.license.spdx_or_name
        ),
    )
}

/// A card line's receipt markers: `[receipt:<id>]`.
pub(crate) fn receipt_markers(line: &str) -> (String, Vec<&str>) {
    let mut ids = Vec::new();
    let mut rest = String::with_capacity(line.len());
    let mut s = line;
    while let Some(i) = s.find("[receipt:") {
        rest.push_str(&s[..i]);
        let after = &s[i + "[receipt:".len()..];
        match after.find(']') {
            Some(j) => {
                ids.push(after[..j].trim());
                s = &after[j + 1..];
            }
            None => {
                s = after;
                break;
            }
        }
    }
    rest.push_str(s);
    (rest, ids)
}

/// Whether a line states a figure: a digit that is not part of an identifier (the
/// `3.5` in `Qwen3.5-4B`, the `0` in `v0.1.0`) and not an ordered-list marker.
pub(crate) fn states_a_figure(text: &str) -> bool {
    let body = text.trim_start();
    let body = match body.find(". ") {
        Some(i) if i > 0 && body[..i].bytes().all(|b| b.is_ascii_digit()) => &body[i + 2..],
        _ => body,
    };
    let b = body.as_bytes();
    let ident = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.' || b[i] == b'-') {
                i += 1;
            }
            let before = start.checked_sub(1).map(|k| b[k]);
            let after = b.get(i).copied();
            let glued = before.is_some_and(|c| ident(c) || c == b'-' || c == b'.')
                || after.is_some_and(ident);
            if !glued {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// M6 card truth (I-10): every card line that states a figure cites a known receipt id.
/// The comparator-block and `[X]` label checks belong to EXT-16's card lint.
fn m6(m: &ReleaseManifest, inp: &GateInputs<'_>) -> GateRow {
    let Some(card) = m.files.iter().find(|f| f.name == "README.md") else {
        return row(
            "M6",
            vec!["no README.md card in the release".into()],
            String::new(),
        );
    };
    let text = match std::fs::read_to_string(inp.dir.join(&card.name)) {
        Ok(t) => t,
        Err(e) => return row("M6", vec![format!("README.md: {e}")], String::new()),
    };
    let known: BTreeSet<&str> = inp
        .evidence
        .receipt_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut red = Vec::new();
    let mut cited = 0usize;
    for (n, line) in text.lines().enumerate() {
        let (rest, ids) = receipt_markers(line);
        for id in &ids {
            if !known.contains(id) {
                red.push(format!(
                    "README.md:{}: receipt {id:?} is not a known receipt",
                    n + 1
                ));
            }
        }
        if states_a_figure(&rest) {
            if ids.is_empty() {
                red.push(format!(
                    "README.md:{}: a figure with no receipt id (I-10)",
                    n + 1
                ));
            } else {
                cited += 1;
            }
        }
    }
    row(
        "M6",
        red,
        format!("{cited} card lines with figures, each citing a known receipt"),
    )
}

/// Run M0..M6, with M1b after M1.
///
/// # Errors
///
/// Only when the manifest cannot be read or parsed: without it there is nothing to gate.
/// Every other failure is a RED row in the receipt.
pub(crate) fn run(pre: &M2Prereg, inp: &GateInputs<'_>) -> Result<GateReceipt, String> {
    let path = inp.dir.join(MANIFEST);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m: ReleaseManifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let (m1b_row, m1b_report) = model_gate_m1b::gate(&m, inp.evidence.m1b.as_ref());
    let (m2_row, m2_report) = m2(pre, inp.evidence.m2.as_ref());
    let gates = vec![
        m0(&m, inp),
        m1(inp.evidence.m1.as_ref()),
        m1b_row,
        m2_row,
        m3(inp.evidence.m3.as_ref()),
        m4(&m, inp),
        m5(&m),
        m6(&m, inp),
    ];
    let all_green = gates.iter().all(|g| g.green);
    Ok(GateReceipt {
        schema: "model-gate-receipt-v1",
        line: m.line,
        version: m.version,
        manifest_sha256: hex_lower(&Sha256::digest(&bytes)),
        gates,
        m1: inp.evidence.m1.clone(),
        m1b: m1b_report,
        m2: m2_report,
        m3_parse_rate: inp.evidence.m3.as_ref().and_then(|e| e.parse_rate),
        sealed_items_checked: inp.sealed.len(),
        sealed_set_sha256: inp.sealed.set_sha256(),
        all_green,
    })
}

#[cfg(test)]
#[path = "model_gate_tests.rs"]
pub(crate) mod tests;
