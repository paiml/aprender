//! 0.71 V3 — is a perf041 witness a serving-shape parity receipt? (#4971 V3-d)
//!
//! perf041 (PP-26 v3.1) answers one question: is the batch consistent with
//! itself? A V3 receipt asks more of the same witness, and this module is the
//! check. It ports the hand-run `v3_shape_check.py` from the spec branch
//! (`docs/lookahead/0.71-v3-serving-shape-parity.md`, draft contract
//! `serving-shape-parity-v1`, branch `la-71/v3-serving-shape-parity`) and
//! carries its case table. No Python ships with it.
//!
//! | Rule | The witness holds |
//! |---|---|
//! | S1 | the blessed Qwen3.5-4B, with its sha256 |
//! | S2 | every band `c` in {1, 4, 8, 16} |
//! | S3 | `m_formed == c` in every band: a band that formed fewer served another shape |
//! | S4 | identity: `binary_sha256`, `commit`, `host`, `prompt_sha256` |
//! | S5 | agreement with the `m = 1` reference to `declared_min`, unless a recorded top-2 margin below ε explains the divergence as a near-tie |
//! | S6 | perf041 PASS in every band |
//! | S7 | FALSIFY-CB-004: per-request decode at c=4 at least half of c=1's, read from the bands' `decode_tok_s` |
//! | floor | every band's `declared_min` is at least 64, the perf matrix's `witness.min_agree_tokens` |
//! | cap | at most one of c=4, 8, 16 is admitted through the near-tie clause |
//!
//! S7, the floor and the cap are the V3 bar's (C314 round 2, contract
//! `serving-shape-parity-v1`), as is ε = 0.05 logit, now fixed. A witness
//! with no per-request decode timing at c=1 or c=4 is not admissible: S7 is
//! unmeasured, never assumed. The probe times a request only on a stream the
//! server declared live ([`super::parity::live_decode_tok_s`]).
//!
//! The port closes the draft contract's findings:
//! * **F1** an unreadable witness is RED (exit 2), never NOT ADMISSIBLE;
//! * **F2** fields are typed, so `true` is not a batch width and `false` is not a margin;
//! * **F3** a band with no `declared_min` is not admissible;
//! * **F4** when a margin was recorded, the S5 reason names it and ε;
//! * **F5** a band with no divergence measurement reports only its result;
//! * **F6** a band `c` recorded twice is not admissible, and every copy is checked.
//!
//! This is not a gate. No merge, queue or release job runs it; putting it
//! beside `check_perf041_marker.sh` is a gate change with its own ruling.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

/// S1: the blessed model's file name starts with this, compared without case.
pub const BLESSED_MODEL_PREFIX: &str = "qwen3.5-4b";

/// S2: the bands `apr serve` advertises.
pub const V3_BANDS: [u32; 4] = [1, 4, 8, 16];

/// S5: ε in logits, fixed by the V3 bar. A recorded top-2 margin explains a
/// divergence only when `0 <= margin < ε`.
pub const NEAR_TIE_EPS: f64 = 0.05;

/// The cap: batched bands (c > 1) that may be admitted through the near-tie
/// clause. Bands c=4, 8 and 16 share one batched path, so more than one early
/// divergence there points at the path, not at independent near-ties.
pub const NEAR_TIE_CAP: usize = 1;

/// The floor under every band's `declared_min`: the perf matrix's
/// `witness.min_agree_tokens`, so a witness cannot declare its own bar.
pub const DECLARED_MIN_FLOOR: u32 = 64;

/// S7 (FALSIFY-CB-004): per-request decode at c=4 over c=1's, at least.
pub const S7_MIN_RATIO: f64 = 0.5;

/// The check's answer. Exit codes follow the Python reference: 0, 1, 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeVerdict {
    /// Every rule S1–S7, the floor and the cap hold.
    Admissible,
    /// The witness was read, and these rules failed.
    NotAdmissible(Vec<String>),
    /// The witness could not be read, or it measured nothing.
    Red(String),
}

impl ShapeVerdict {
    /// 0 ADMISSIBLE, 1 NOT ADMISSIBLE, 2 RED.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Admissible => 0,
            Self::NotAdmissible(_) => 1,
            Self::Red(_) => 2,
        }
    }

    /// The verdict word printed on the last line.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Admissible => "ADMISSIBLE",
            Self::NotAdmissible(_) => "NOT ADMISSIBLE",
            Self::Red(_) => "RED",
        }
    }

    /// Every reason, in rule order. Empty only for [`Self::Admissible`].
    #[must_use]
    pub fn reasons(&self) -> Vec<String> {
        match self {
            Self::Admissible => Vec::new(),
            Self::NotAdmissible(reasons) => reasons.clone(),
            Self::Red(reason) => vec![reason.clone()],
        }
    }
}

/// The fields of a perf041 witness that V3 reads. Every other field is
/// allowed and ignored: the witness belongs to the probe, not to this check.
#[derive(Debug, Deserialize)]
struct ShapeWitness {
    model: Option<ShapeModel>,
    binary_sha256: Option<String>,
    commit: Option<String>,
    host: Option<String>,
    prompt_sha256: Option<String>,
    bands: Option<Vec<ShapeBand>>,
}

#[derive(Debug, Deserialize)]
struct ShapeModel {
    path: Option<String>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ShapeBand {
    c: u32,
    result: Option<String>,
    m_formed: Option<u32>,
    divergence_at: Option<u32>,
    declared_min: Option<u32>,
    top2_margin_at_divergence: Option<f64>,
    decode_tok_s: Option<f64>,
}

/// Check the witness file at `path`. A file that cannot be read is RED (F1).
#[must_use]
pub fn check_v3_shape_file(path: &Path) -> ShapeVerdict {
    match std::fs::read_to_string(path) {
        Ok(text) => check_v3_shape(&text),
        Err(e) => ShapeVerdict::Red(format!("RED: cannot read witness {}: {e}", path.display())),
    }
}

/// Check one witness, given as JSON text.
#[must_use]
pub fn check_v3_shape(text: &str) -> ShapeVerdict {
    if text.trim().is_empty() {
        return ShapeVerdict::Red("RED: empty witness".to_string());
    }
    let witness: ShapeWitness = match serde_json::from_str(text) {
        Ok(witness) => witness,
        Err(e) => return ShapeVerdict::Red(format!("RED: unreadable witness: {e}")),
    };
    let Some(bands) = witness.bands.as_deref().filter(|bands| !bands.is_empty()) else {
        return ShapeVerdict::Red("RED: the witness measured no bands".to_string());
    };
    let reasons: Vec<String> = model_reasons(witness.model.as_ref())
        .into_iter()
        .chain(identity_reasons(&witness))
        .chain(bands_reasons(bands))
        .collect();
    if reasons.is_empty() {
        ShapeVerdict::Admissible
    } else {
        ShapeVerdict::NotAdmissible(reasons)
    }
}

/// S1: the blessed model, and its sha256.
fn model_reasons(model: Option<&ShapeModel>) -> Vec<String> {
    let path = model.and_then(|m| m.path.as_deref()).unwrap_or("");
    let mut out = Vec::new();
    if !path.to_ascii_lowercase().starts_with(BLESSED_MODEL_PREFIX) {
        out.push(format!("model {path:?} is not the blessed Qwen3.5-4B"));
    }
    if model.and_then(|m| m.sha256.as_deref()).is_none_or(blank) {
        out.push("model sha256 missing".to_string());
    }
    out
}

/// S4: every identity field present and not blank.
fn identity_reasons(witness: &ShapeWitness) -> Vec<String> {
    [
        ("binary_sha256", &witness.binary_sha256),
        ("commit", &witness.commit),
        ("host", &witness.host),
        ("prompt_sha256", &witness.prompt_sha256),
    ]
    .into_iter()
    .filter(|(_, value)| value.as_deref().is_none_or(blank))
    .map(|(name, _)| format!("identity field {name} missing"))
    .collect()
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// S2, F6 and the per-band rules in band order, then the cap and S7.
fn bands_reasons(bands: &[ShapeBand]) -> Vec<String> {
    let mut by_c: BTreeMap<u32, Vec<&ShapeBand>> = BTreeMap::new();
    for band in bands {
        by_c.entry(band.c).or_default().push(band);
    }
    let mut out = Vec::new();
    for c in V3_BANDS {
        let copies = by_c.get(&c).map_or(&[][..], Vec::as_slice);
        if copies.is_empty() {
            out.push(format!("band c={c} not measured"));
        }
        if copies.len() > 1 {
            out.push(format!(
                "band c={c} recorded {} times; a witness has one band per c",
                copies.len()
            ));
        }
        for band in copies {
            out.extend(band_reasons(band));
        }
    }
    out.extend(near_tie_cap_reason(&by_c));
    out.extend(s7_reason(&by_c));
    out
}

/// S6, S3, the floor and S5 for one band.
fn band_reasons(band: &ShapeBand) -> Vec<String> {
    let c = band.c;
    let result = band.result.as_deref();
    let mut out = Vec::new();
    if result != Some("PASS") {
        out.push(format!(
            "c={c}: perf041 result {}",
            result.unwrap_or("not recorded")
        ));
    }
    match band.m_formed {
        Some(m) if m == c => {}
        Some(m) => out.push(format!("c={c}: m_formed {m} != c (shape not served)")),
        None => out.push(format!("c={c}: m_formed not recorded (shape not shown)")),
    }
    if let Some(min) = band.declared_min.filter(|&min| min < DECLARED_MIN_FLOOR) {
        out.push(format!(
            "c={c}: declared_min {min} is below the floor of {DECLARED_MIN_FLOOR} \
             (witness.min_agree_tokens)"
        ));
    }
    out.extend(divergence_reason(band, result == Some("PASS")));
    out
}

/// S5's near-tie clause: a recorded top-2 margin in `[0, ε)`.
fn near_tie(margin: f64) -> bool {
    (0.0..NEAR_TIE_EPS).contains(&margin)
}

/// True when S5 holds for `band` only through the near-tie clause: it parts
/// from m=1 before `declared_min`, at a recorded near-tie.
fn admitted_by_near_tie(band: &ShapeBand) -> bool {
    match (
        band.declared_min,
        band.divergence_at,
        band.top2_margin_at_divergence,
    ) {
        (Some(min), Some(at), Some(margin)) => at < min && near_tie(margin),
        _ => false,
    }
}

/// The cap: at most [`NEAR_TIE_CAP`] batched V3 bands admitted through the
/// near-tie clause. c=1 is not batched, and a band outside V3 is ignored, as
/// everywhere else.
fn near_tie_cap_reason(by_c: &BTreeMap<u32, Vec<&ShapeBand>>) -> Option<String> {
    let tied: Vec<String> = V3_BANDS
        .iter()
        .filter(|&&c| c > 1)
        .flat_map(|c| by_c.get(c).into_iter().flatten())
        .filter(|band| admitted_by_near_tie(band))
        .map(|band| format!("c={}", band.c))
        .collect();
    (tied.len() > NEAR_TIE_CAP).then(|| {
        format!(
            "near-tie cap: {} batched bands ({}) are admitted only by a near-tie margin; \
             at most {NEAR_TIE_CAP} of c=4, 8, 16 may be",
            tied.len(),
            tied.join(", ")
        )
    })
}

/// S7 (FALSIFY-CB-004): per-request decode at c=4 at least [`S7_MIN_RATIO`]
/// of c=1's. Read only when c=1 and c=4 each have exactly one band: a missing
/// or doubled band already has its reason, and which copy's rate S7 would read
/// is not defined. A band with no rate, or a rate that is not positive, leaves
/// S7 unmeasured, and an unmeasured S7 is not admissible.
fn s7_reason(by_c: &BTreeMap<u32, Vec<&ShapeBand>>) -> Option<String> {
    let one = |c: u32| match by_c.get(&c).map(Vec::as_slice) {
        Some([band]) => Some(*band),
        _ => None,
    };
    let (c1, c4) = (one(1)?, one(4)?);
    let rate = |band: &ShapeBand| band.decode_tok_s.filter(|&r| r > 0.0);
    let (Some(d1), Some(d4)) = (rate(c1), rate(c4)) else {
        let unmeasured: Vec<&str> = [(c1, "c=1"), (c4, "c=4")]
            .into_iter()
            .filter(|&(band, _)| rate(band).is_none())
            .map(|(_, name)| name)
            .collect();
        return Some(format!(
            "S7 (FALSIFY-CB-004) is unmeasured: no positive per-request decode_tok_s at {}",
            unmeasured.join(" and ")
        ));
    };
    (d4 < S7_MIN_RATIO * d1).then(|| {
        format!(
            "S7 (FALSIFY-CB-004): per-request decode at c=4 is {d4} tok/s, {:.1}% of c=1's \
             {d1} tok/s; needs at least {}%",
            100.0 * d4 / d1,
            100.0 * S7_MIN_RATIO
        )
    })
}

/// S5, with F3 (no `declared_min`), F4 (name the recorded margin) and F5 (no
/// divergence measurement on a band that did not pass: its result says it).
fn divergence_reason(band: &ShapeBand, passed: bool) -> Option<String> {
    let c = band.c;
    let Some(min) = band.declared_min else {
        return Some(format!(
            "c={c}: declared_min not recorded, so agreement with m=1 cannot be checked"
        ));
    };
    let Some(at) = band.divergence_at else {
        return passed.then(|| {
            format!("c={c}: divergence_at not recorded on a PASS band, so agreement with m=1 is unmeasured")
        });
    };
    if at >= min {
        return None;
    }
    match band.top2_margin_at_divergence {
        None => Some(format!(
            "c={c}: diverges from m=1 at {at} < {min} with no near-tie margin recorded"
        )),
        Some(margin) if near_tie(margin) => None,
        Some(margin) => Some(format!(
            "c={c}: diverges from m=1 at {at} < {min}; the recorded top-2 margin {margin} \
             is not a near-tie (needs 0 <= margin < {NEAR_TIE_EPS})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// The admissible control, as `fixtures/v3/admissible.json` on the spec
    /// branch has it, plus S7's per-request decode rates (c=4 at 75% of c=1).
    /// Every case below changes one thing in it.
    fn admissible() -> Value {
        let band = |c: u32, rate: f64| json!({"c": c, "m_formed": c, "result": "PASS", "declared_min": 64, "divergence_at": 127, "decode_tok_s": rate});
        json!({
            "_planted": "FIXTURE: not a measurement",
            "binary_sha256": "a".repeat(64),
            "commit": "f".repeat(40),
            "host": "lambda",
            "prompt_sha256": "b".repeat(64),
            "model": {"path": "Qwen3.5-4B-Q4_K_M.gguf", "sha256": "c".repeat(64)},
            "bands": [band(1, 40.0), band(4, 30.0), band(8, 25.0), band(16, 20.0)],
        })
    }

    /// The band at `index` of the control (0..4 is c = 1, 4, 8, 16).
    fn band_mut(w: &mut Value, index: usize) -> &mut serde_json::Map<String, Value> {
        w["bands"][index]
            .as_object_mut()
            .expect("a band is an object")
    }

    fn set(index: usize, key: &str, value: Value) -> impl Fn(&mut Value) {
        let key = key.to_string();
        move |w: &mut Value| {
            band_mut(w, index).insert(key.clone(), value.clone());
        }
    }

    fn remove_top(key: &'static str) -> impl Fn(&mut Value) {
        move |w: &mut Value| {
            w.as_object_mut().expect("object").remove(key);
        }
    }

    fn drop_band(index: usize) -> impl Fn(&mut Value) {
        move |w: &mut Value| {
            w["bands"].as_array_mut().expect("bands").remove(index);
        }
    }

    fn diverge(index: usize, at: u32, margin: Option<Value>) -> impl Fn(&mut Value) {
        move |w: &mut Value| {
            let band = band_mut(w, index);
            band.insert("divergence_at".into(), json!(at));
            if let Some(margin) = &margin {
                band.insert("top2_margin_at_divergence".into(), margin.clone());
            }
        }
    }

    fn remove(index: usize, key: &'static str) -> impl Fn(&mut Value) {
        move |w: &mut Value| {
            band_mut(w, index).remove(key);
        }
    }

    /// Several changes, in order.
    fn both(a: impl Fn(&mut Value), b: impl Fn(&mut Value)) -> impl Fn(&mut Value) {
        move |w: &mut Value| {
            a(w);
            b(w);
        }
    }

    type Mutation = Box<dyn Fn(&mut Value)>;

    /// One planted case: a change to the control, the exit code it must give,
    /// and its exact reasons. The first 19 rows are the spec branch's
    /// `cases.json` (its `empty` and `blank` rows are in the RED test below);
    /// rows marked F2–F6 are the findings, and their wording is this port's.
    fn cases() -> Vec<(&'static str, Mutation, i32, Vec<&'static str>)> {
        vec![
            ("admissible", Box::new(|_: &mut Value| {}), 0, vec![]),
            ("near-tie", Box::new(diverge(1, 3, Some(json!(0.01)))), 0, vec![]),
            ("short-band", Box::new(set(3, "m_formed", json!(14))), 1, vec!["c=16: m_formed 14 != c (shape not served)"]),
            (
                "wrong-model",
                Box::new(|w: &mut Value| w["model"]["path"] = json!("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")),
                1,
                vec!["model \"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf\" is not the blessed Qwen3.5-4B"],
            ),
            ("unexplained-divergence", Box::new(diverge(1, 3, None)), 1, vec!["c=4: diverges from m=1 at 3 < 64 with no near-tie margin recorded"]),
            ("no-identity-binary-sha256", Box::new(remove_top("binary_sha256")), 1, vec!["identity field binary_sha256 missing"]),
            ("no-identity-commit", Box::new(remove_top("commit")), 1, vec!["identity field commit missing"]),
            ("no-identity-host", Box::new(remove_top("host")), 1, vec!["identity field host missing"]),
            ("no-identity-prompt-sha256", Box::new(remove_top("prompt_sha256")), 1, vec!["identity field prompt_sha256 missing"]),
            (
                "no-model-sha256",
                Box::new(|w: &mut Value| {
                    w["model"].as_object_mut().expect("model").remove("sha256");
                }),
                1,
                vec!["model sha256 missing"],
            ),
            (
                "margin-at-epsilon (F4 wording)",
                Box::new(diverge(1, 3, Some(json!(0.05)))),
                1,
                vec!["c=4: diverges from m=1 at 3 < 64; the recorded top-2 margin 0.05 is not a near-tie (needs 0 <= margin < 0.05)"],
            ),
            ("failed-band", Box::new(set(2, "result", json!("FAIL"))), 1, vec!["c=8: perf041 result FAIL"]),
            ("missing-band-c1", Box::new(drop_band(0)), 1, vec!["band c=1 not measured"]),
            ("missing-band-c4", Box::new(drop_band(1)), 1, vec!["band c=4 not measured"]),
            ("missing-band-c8", Box::new(drop_band(2)), 1, vec!["band c=8 not measured"]),
            ("missing-band-c16", Box::new(drop_band(3)), 1, vec!["band c=16 not measured"]),
            ("divergence-at-min", Box::new(diverge(1, 64, None)), 0, vec![]),
            ("no-bands", Box::new(|w: &mut Value| w["bands"] = json!([])), 2, vec!["RED: the witness measured no bands"]),
            (
                "unmeasurable-band (F5: its result only)",
                Box::new(|w: &mut Value| {
                    let band = band_mut(w, 2);
                    band.insert("result".into(), json!("UNMEASURABLE"));
                    band.insert("divergence_at".into(), Value::Null);
                }),
                1,
                vec!["c=8: perf041 result UNMEASURABLE"],
            ),
            // F2: typed fields. A boolean is not a width and not a margin.
            ("F2 m_formed true", Box::new(set(0, "m_formed", json!(true))), 2, vec![]),
            ("F2 margin false", Box::new(diverge(1, 3, Some(json!(false)))), 2, vec![]),
            ("F2 c as a string", Box::new(set(1, "c", json!("4"))), 2, vec![]),
            // F3: no declared_min, so S5 cannot hold.
            (
                "F3 no declared_min",
                Box::new(|w: &mut Value| {
                    band_mut(w, 1).remove("declared_min");
                }),
                1,
                vec!["c=4: declared_min not recorded, so agreement with m=1 cannot be checked"],
            ),
            // F6: a FAIL first and a PASS second; the FAIL is not masked.
            (
                "F6 duplicate c",
                Box::new(|w: &mut Value| {
                    let mut failed = w["bands"][2].clone();
                    failed["result"] = json!("FAIL");
                    w["bands"].as_array_mut().expect("bands").insert(2, failed);
                }),
                1,
                vec!["band c=8 recorded 2 times; a witness has one band per c", "c=8: perf041 result FAIL"],
            ),
            // The mutants the Python table let survive, and the other boundaries.
            (
                "prefix, not substring",
                Box::new(|w: &mut Value| w["model"]["path"] = json!("not-qwen3.5-4b.gguf")),
                1,
                vec!["model \"not-qwen3.5-4b.gguf\" is not the blessed Qwen3.5-4B"],
            ),
            ("divergence one below min", Box::new(diverge(1, 63, None)), 1, vec!["c=4: diverges from m=1 at 63 < 64 with no near-tie margin recorded"]),
            ("margin just below epsilon", Box::new(diverge(1, 3, Some(json!(0.0499)))), 0, vec![]),
            ("margin zero (an exact tie)", Box::new(diverge(1, 3, Some(json!(0)))), 0, vec![]),
            (
                "negative margin",
                Box::new(diverge(1, 3, Some(json!(-0.01)))),
                1,
                vec!["c=4: diverges from m=1 at 3 < 64; the recorded top-2 margin -0.01 is not a near-tie (needs 0 <= margin < 0.05)"],
            ),
            (
                "PASS band without divergence_at",
                Box::new(set(3, "divergence_at", Value::Null)),
                1,
                vec!["c=16: divergence_at not recorded on a PASS band, so agreement with m=1 is unmeasured"],
            ),
            ("m_formed null", Box::new(set(1, "m_formed", Value::Null)), 1, vec!["c=4: m_formed not recorded (shape not shown)"]),
            (
                "result missing",
                Box::new(|w: &mut Value| {
                    band_mut(w, 0).remove("result");
                }),
                1,
                vec!["c=1: perf041 result not recorded"],
            ),
            (
                "blank identity",
                Box::new(|w: &mut Value| w["host"] = json!("  ")),
                1,
                vec!["identity field host missing"],
            ),
            (
                "model missing",
                Box::new(remove_top("model")),
                1,
                vec!["model \"\" is not the blessed Qwen3.5-4B", "model sha256 missing"],
            ),
            (
                "a band outside V3 is ignored",
                Box::new(|w: &mut Value| {
                    let mut extra = w["bands"][0].clone();
                    extra["c"] = json!(2);
                    extra["result"] = json!("FAIL");
                    w["bands"].as_array_mut().expect("bands").push(extra);
                }),
                0,
                vec![],
            ),
            // FALSIFY-SSP-017, S7: the bar's falsifier, its boundary, and a
            // witness that cannot say.
            (
                "S7 c=4 at 49% of c=1 (the bar's falsifier)",
                Box::new(set(1, "decode_tok_s", json!(19.6))),
                1,
                vec!["S7 (FALSIFY-CB-004): per-request decode at c=4 is 19.6 tok/s, 49.0% of c=1's 40 tok/s; needs at least 50%"],
            ),
            ("S7 c=4 at exactly 50%", Box::new(set(1, "decode_tok_s", json!(20.0))), 0, vec![]),
            ("S7 reads c=4, not c=8 or c=16", Box::new(both(set(2, "decode_tok_s", json!(1.0)), set(3, "decode_tok_s", json!(1.0)))), 0, vec![]),
            (
                "S7 no timing at c=4",
                Box::new(remove(1, "decode_tok_s")),
                1,
                vec!["S7 (FALSIFY-CB-004) is unmeasured: no positive per-request decode_tok_s at c=4"],
            ),
            (
                "S7 null timing at c=1",
                Box::new(set(0, "decode_tok_s", Value::Null)),
                1,
                vec!["S7 (FALSIFY-CB-004) is unmeasured: no positive per-request decode_tok_s at c=1"],
            ),
            (
                "S7 zero rates are not rates",
                Box::new(both(set(0, "decode_tok_s", json!(0.0)), set(1, "decode_tok_s", json!(0.0)))),
                1,
                vec!["S7 (FALSIFY-CB-004) is unmeasured: no positive per-request decode_tok_s at c=1 and c=4"],
            ),
            (
                "S7 is not read beside a missing c=1 (its band says it)",
                Box::new(both(drop_band(0), set(0, "decode_tok_s", Value::Null))),
                1,
                vec!["band c=1 not measured"],
            ),
            ("F2 decode_tok_s as a string", Box::new(set(1, "decode_tok_s", json!("30"))), 2, vec![]),
            // FALSIFY-SSP-018, the cap: one near-tie batched band is chance,
            // two are the batched path.
            (
                "near-tie cap: c=4 and c=8",
                Box::new(both(diverge(1, 3, Some(json!(0.01))), diverge(2, 3, Some(json!(0.01))))),
                1,
                vec!["near-tie cap: 2 batched bands (c=4, c=8) are admitted only by a near-tie margin; at most 1 of c=4, 8, 16 may be"],
            ),
            (
                "near-tie cap: all three batched bands",
                Box::new(|w: &mut Value| {
                    for index in 1..4 {
                        diverge(index, 3, Some(json!(0.01)))(w);
                    }
                }),
                1,
                vec!["near-tie cap: 3 batched bands (c=4, c=8, c=16) are admitted only by a near-tie margin; at most 1 of c=4, 8, 16 may be"],
            ),
            ("near-tie at c=16 alone", Box::new(diverge(3, 3, Some(json!(0.01)))), 0, vec![]),
            ("near-tie cap: c=1 is not batched", Box::new(both(diverge(0, 3, Some(json!(0.01))), diverge(1, 3, Some(json!(0.01))))), 0, vec![]),
            (
                "near-tie cap: a margin on a band that agreed to declared_min is not counted",
                Box::new(both(diverge(1, 64, Some(json!(0.01))), diverge(2, 3, Some(json!(0.01))))),
                0,
                vec![],
            ),
            // FALSIFY-SSP-019, the floor: a witness does not set its own bar.
            (
                "floor: declared_min 1",
                Box::new(both(set(1, "declared_min", json!(1)), diverge(1, 3, None))),
                1,
                vec!["c=4: declared_min 1 is below the floor of 64 (witness.min_agree_tokens)"],
            ),
            (
                "floor: declared_min 63",
                Box::new(set(2, "declared_min", json!(63))),
                1,
                vec!["c=8: declared_min 63 is below the floor of 64 (witness.min_agree_tokens)"],
            ),
        ]
    }

    /// FALSIFY-SSP-001..006, 008..019 and F1–F6: every planted case exits as
    /// its row says and prints exactly its row's reasons. A RED row with no
    /// reasons listed checks the exit code and the `RED:` prefix.
    #[test]
    fn every_planted_case_exits_as_its_row_says_with_exactly_its_reasons() {
        let cases = cases();
        assert!(
            cases.len() >= 19,
            "the case table holds at least the spec's 19 object rows"
        );
        let mut ids: Vec<_> = cases.iter().map(|(id, ..)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), cases.len(), "case ids are unique");
        for (id, mutate, want_exit, want_reasons) in cases {
            let mut w = admissible();
            mutate(&mut w);
            let verdict = check_v3_shape(&w.to_string());
            assert_eq!(verdict.exit_code(), want_exit, "{id}: {verdict:?}");
            if want_exit == 2 {
                let reasons = verdict.reasons();
                assert!(
                    reasons.len() == 1 && reasons[0].starts_with("RED: "),
                    "{id}: {reasons:?}"
                );
            }
            if want_exit < 2 || !want_reasons.is_empty() {
                assert_eq!(verdict.reasons(), want_reasons, "{id}");
            }
        }
    }

    /// F1 and FALSIFY-SSP-006/016: empty, blank, malformed, a JSON value that
    /// is not an object, and a file that does not exist are all RED.
    #[test]
    fn an_unreadable_witness_is_red_never_not_admissible() {
        for text in ["", "\n", "  \t\n", "{", "[]", "3", "null", "{\"bands\": 4}"] {
            let verdict = check_v3_shape(text);
            assert_eq!(verdict.exit_code(), 2, "{text:?}: {verdict:?}");
            assert_eq!(verdict.label(), "RED");
        }
        assert_eq!(
            check_v3_shape("{}"),
            ShapeVerdict::Red("RED: the witness measured no bands".to_string())
        );
        let missing = Path::new("/nonexistent/la-0.71.1a/v3-shape/witness.json");
        let verdict = check_v3_shape_file(missing);
        assert_eq!(verdict.exit_code(), 2, "{verdict:?}");
        assert!(verdict.reasons()[0].starts_with("RED: cannot read witness "));
    }

    /// The file entry point reads what the text entry point checks.
    #[test]
    fn the_file_entry_point_checks_the_file() {
        let dir = std::env::temp_dir().join(format!("v3-shape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("witness.json");
        std::fs::write(&path, admissible().to_string()).expect("write");
        assert_eq!(check_v3_shape_file(&path), ShapeVerdict::Admissible);
        std::fs::remove_file(&path).expect("remove");
        let _ = std::fs::remove_dir(&dir);
    }

    /// The floor is the shipped perf matrix's `witness.min_agree_tokens`, not
    /// a second copy of it that can drift.
    #[test]
    fn the_floor_is_the_shipped_matrix_min_agree_tokens() {
        let shipped = super::super::protocol::witness_min_agree_tokens_from(
            super::super::protocol::PERF_MATRIX_SOURCE,
        )
        .expect("the shipped witness block");
        assert_eq!(DECLARED_MIN_FLOOR, shipped);
    }

    #[test]
    fn exit_codes_and_labels_follow_the_python_reference() {
        let rows = [
            (ShapeVerdict::Admissible, 0, "ADMISSIBLE"),
            (
                ShapeVerdict::NotAdmissible(vec!["x".into()]),
                1,
                "NOT ADMISSIBLE",
            ),
            (ShapeVerdict::Red("RED: x".into()), 2, "RED"),
        ];
        for (verdict, code, label) in rows {
            assert_eq!(verdict.exit_code(), code);
            assert_eq!(verdict.label(), label);
        }
        assert!(ShapeVerdict::Admissible.reasons().is_empty());
    }
}
