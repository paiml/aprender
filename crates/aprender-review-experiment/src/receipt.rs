//! REX-03 receipt (contract `review-experiment-receipt-v1`, spec §2.3/§2.4).
//!
//! One JSONL row per (item, cell, arm). A row is admissible only if it
//! deserializes with every required field, carries no empty or `unknown`
//! identity field, names the locked prereg sha and the corpus version under
//! analysis, and (when it ran) carries token counts, timings, the raw-output
//! path+sha and a load snapshot. An inadmissible row is never a pass: the
//! scorer counts its item as not run.

use crate::corpus::{Class, Loc, Split, Stratum};
use serde::{Deserialize, Serialize};

/// Schema name every row carries.
pub const SCHEME: &str = "review-experiment-receipt-v1";

/// The lane under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Arm {
    /// Qwen3.5-4B-Q4_K_M through `apr serve` (primary).
    #[serde(rename = "apr-4b")]
    Apr4b,
    /// Qwen3.5-9B-Q4_K_M through `apr serve` (control, H3).
    #[serde(rename = "apr-9b")]
    Apr9b,
    /// Haiku 4.5 via `claude -p` (baseline).
    #[serde(rename = "haiku")]
    Haiku,
    /// The agy lane (baseline).
    #[serde(rename = "agy")]
    Agy,
}

impl Arm {
    /// Hosted arms have no apr binary or local weights; they record `hosted`.
    #[must_use]
    pub fn is_hosted(self) -> bool {
        matches!(self, Self::Haiku | Self::Agy)
    }
}

/// Why a (item, cell, arm) produced no verdict. Never correct, never a FAIL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NotRun {
    /// No forjar-declared executor for the cell (R-4).
    NoDeclaredExecutor,
    /// The diff does not fit the context; it is never truncated.
    ContextOverflow,
    /// The backend refused to load or serve the model on this cell.
    Refused,
    /// The serve answered with an error that is not a context overflow.
    ServeError,
    /// Assigned by the scorer to an item whose receipt is inadmissible.
    Inadmissible,
}

/// Parsed verdict (§2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "reason")]
pub enum Verdict {
    Pass,
    Fail,
    /// The output had no well-formed `VERDICT:` line. Counted as incorrect.
    Unparsed,
    NotRun(NotRun),
}

impl Verdict {
    /// The row executed a request (parsed or not).
    #[must_use]
    pub fn executed(self) -> bool {
        !matches!(self, Self::NotRun(_))
    }

    /// Correct per the analysis plan: a parsed FAIL on a defect, a parsed
    /// PASS on a good item. `Unparsed` and `NotRun` are never correct (R-6).
    #[must_use]
    pub fn correct(self, defect: bool) -> bool {
        match self {
            Self::Fail => defect,
            Self::Pass => !defect,
            Self::Unparsed | Self::NotRun(_) => false,
        }
    }
}

/// Decoding settings (fixed across every run, §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Decoding {
    pub temperature: f64,
    pub seed: u64,
    pub max_tokens: u32,
}

/// Token counts from the serve's `usage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokens {
    pub prompt: u64,
    pub completion: u64,
}

/// Server-measured phase timings (the serve's `timings`, PP-LLAMA-001 §3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ServerTimings {
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub prompt_per_second: Option<f64>,
    pub predicted_per_second: Option<f64>,
}

/// Client wall-clock (request sent → verdict parsed) plus the server's own
/// timings. `server: null` means the serving path did not measure them; the
/// TTFT/tok/s columns are then `NotRun{NoSensor}`, never zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub wall_ms: f64,
    pub server: Option<ServerTimings>,
}

/// Host load at request time (§2.3). `co_running` and `peak_anon_bytes`
/// come from the executor unit (infra#1088); `null` where it does not report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Load {
    pub loadavg1: f64,
    pub cpus: u32,
    pub co_running: Option<Vec<String>>,
    pub peak_anon_bytes: Option<u64>,
}

/// Where the raw model output was written, and its sha.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    pub path: String,
    pub sha256: String,
}

/// One `review-experiment-receipt-v1` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    pub item_id: String,
    pub item_sha256: String,
    pub class: Class,
    pub stratum: Stratum,
    pub split: Split,
    pub cell: String,
    pub host: String,
    pub backend: String,
    pub arm: Arm,
    pub apr_tag: String,
    pub apr_sha256: String,
    pub model_id: String,
    pub weights_sha256: String,
    pub prompt_sha256: String,
    /// sha of the exact request body sent (prompt composed with the diff).
    pub request_sha256: String,
    pub decoding: Decoding,
    pub corpus_version: String,
    pub prereg_sha: String,
    /// RFC 3339 UTC.
    pub started_at: String,
    /// First request after the unit started (§2.4); reported separately.
    pub cold: bool,
    /// Member of the 10 % determinism re-run.
    pub rerun: bool,
    pub verdict: Verdict,
    pub tokens: Option<Tokens>,
    pub timings: Option<Timings>,
    pub output: Option<Output>,
    pub load: Option<Load>,
}

/// What the analysis requires every row to name.
#[derive(Debug, Clone, Copy)]
pub struct Expect<'a> {
    pub prereg_sha: &'a str,
    pub corpus_version: &'a str,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Parse and check one JSONL row. `Err` lists every reason it is inadmissible.
///
/// # Errors
/// The row does not deserialize, or fails any admissibility rule.
pub fn admissible(line: &str, expect: Expect<'_>) -> Result<Receipt, Vec<String>> {
    let r: Receipt = serde_json::from_str(line).map_err(|e| vec![format!("schema: {e}")])?;
    let bad = problems(&r, expect);
    if bad.is_empty() {
        Ok(r)
    } else {
        Err(bad)
    }
}

fn problems(r: &Receipt, expect: Expect<'_>) -> Vec<String> {
    let mut bad = Vec::new();
    if r.schema != SCHEME {
        bad.push(format!("schema {:?} != {SCHEME}", r.schema));
    }
    let named = [
        ("item_id", &r.item_id),
        ("cell", &r.cell),
        ("host", &r.host),
        ("backend", &r.backend),
        ("apr_tag", &r.apr_tag),
        ("model_id", &r.model_id),
        ("started_at", &r.started_at),
    ];
    for (k, v) in named {
        if v.trim().is_empty() || v.eq_ignore_ascii_case("unknown") {
            bad.push(format!("{k} is empty or unknown"));
        }
    }
    let mut shas = vec![
        ("item_sha256", &r.item_sha256),
        ("prompt_sha256", &r.prompt_sha256),
        ("request_sha256", &r.request_sha256),
        ("prereg_sha", &r.prereg_sha),
    ];
    if r.arm.is_hosted() {
        for (k, v) in [
            ("apr_sha256", &r.apr_sha256),
            ("weights_sha256", &r.weights_sha256),
        ] {
            if v != "hosted" {
                bad.push(format!("{k} must be `hosted` for a hosted arm"));
            }
        }
    } else {
        shas.push(("apr_sha256", &r.apr_sha256));
        shas.push(("weights_sha256", &r.weights_sha256));
    }
    for (k, v) in shas {
        if !is_hex64(v) {
            bad.push(format!("{k} is not a sha256"));
        }
    }
    if r.prereg_sha != expect.prereg_sha {
        bad.push("prereg_sha is not the locked one (exploratory data)".into());
    }
    if r.corpus_version != expect.corpus_version {
        bad.push(format!(
            "corpus_version {:?} is not {:?}",
            r.corpus_version, expect.corpus_version
        ));
    }
    if r.decoding.temperature != 0.0 {
        bad.push("decoding is not greedy (temperature != 0)".into());
    }
    if r.verdict.executed() {
        for (k, missing) in [
            ("tokens", r.tokens.is_none()),
            ("timings", r.timings.is_none()),
            ("output", r.output.is_none()),
            ("load", r.load.is_none()),
        ] {
            if missing {
                bad.push(format!("{k} missing on an executed row"));
            }
        }
    }
    if let Some(o) = &r.output {
        if o.path.is_empty() || !is_hex64(&o.sha256) {
            bad.push("output path/sha malformed".into());
        }
    }
    bad
}

/// The model's answer with any `<think>…</think>` reasoning removed: the
/// verdict is read after the last `</think>` when one is present.
#[must_use]
pub fn answer(output: &str) -> &str {
    output.rsplit_once("</think>").map_or(output, |(_, a)| a)
}

/// §2.3: the verdict of the first `VERDICT:` line of the answer. Markdown
/// decoration (`**`, `#`, `>`, `-`) is ignored. The first word must be
/// exactly `PASS` or `FAIL`, and a line naming both (an echo of the prompt,
/// "PASS or FAIL") is `Unparsed`. No `VERDICT:` line → `Unparsed`.
#[must_use]
pub fn parse_verdict(output: &str) -> Verdict {
    for line in answer(output).lines() {
        let t = line
            .trim()
            .trim_start_matches(['*', '#', '>', '-', ' ', '`']);
        let Some(rest) = t.strip_prefix("VERDICT:") else {
            continue;
        };
        let rest = rest.trim_start().trim_start_matches(['*', '`', ' ']);
        let word: String = rest.chars().take_while(char::is_ascii_alphabetic).collect();
        let tail = &rest[word.len()..];
        return match word.as_str() {
            "PASS" if !tail.contains("FAIL") => Verdict::Pass,
            "FAIL" if !tail.contains("PASS") => Verdict::Fail,
            _ => Verdict::Unparsed,
        };
    }
    Verdict::Unparsed
}

/// Accepted spellings of a defect file `[A]`: the full repo path, or its
/// last two components (`src/foo.rs`). A bare file name is not enough —
/// `mod.rs`/`lib.rs` would match anything.
fn spellings(file: &str) -> Vec<&str> {
    let mut v = vec![file];
    let cut: Vec<usize> = file.match_indices('/').map(|(i, _)| i).collect();
    if cut.len() >= 2 {
        v.push(&file[cut[cut.len() - 2] + 1..]);
    }
    v
}

/// §2.3 localization: some finding after the verdict line names a defect
/// file. Only meaningful on a FAIL of a defect item.
#[must_use]
pub fn localized(output: &str, defect: &[Loc]) -> bool {
    let a = answer(output);
    let findings = a
        .find("VERDICT:")
        .and_then(|i| a[i..].find('\n').map(|j| &a[i + j..]))
        .unwrap_or("");
    defect
        .iter()
        .any(|l| spellings(&l.file).iter().any(|s| findings.contains(s)))
}

#[cfg(test)]
#[path = "receipt_tests.rs"]
pub(crate) mod tests;
