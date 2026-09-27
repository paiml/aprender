//! OBS-14 epochs (APR-OBS-001 v1.3 §2.9, §10 E3/E9; contract `apr-perf-epoch-v1`).
//!
//! An epoch is a maximal run of nights with a constant
//! `(model_sha256, comparator.build_commit, comparator.flags)`, accelerator driver and
//! workload (`band.workload_id`). The apr binary — including a `rustc` bump — is NOT
//! part of the key: that is the signal being measured (C19).
//!
//! - A model or workload change opens an epoch that recalibrates from scratch
//!   (14 nights, report-only until then; never bridged).
//! - A comparator or driver change opens an epoch that must be bridged: exactly 3 nights
//!   measuring old and new side by side (E9). A passing bridge carries θ and the baseline
//!   (`L_tag + Δ_e`); a failing one arms the gate in provisional mode with `θ_prov(n_e)`
//!   until E3 calibration at `n_e = 14` replaces it (C17).
//! - An epoch change on a gated host inside `T−14` nights of a scheduled final is refused
//!   (C18, S-12).
//! - Only `apr-perf-ledger-v1` rows enter a series; a backfill row (`apr-perf-backfill-v1`)
//!   is refused, never skipped (§2.2: backfill is never read by RED, baseline or view series).
//!
//! The per-night statistic `L_n` (E2) is an input here: this module does not re-derive it.

use chrono::NaiveDate;
use serde_json::Value;

/// `schema` of the only rows a series may contain.
pub const PERF_SCHEMA: &str = "apr-perf-ledger-v1";
/// `schema` of an imported historical row (OBS-14 importer).
pub const BACKFILL_SCHEMA: &str = "apr-perf-backfill-v1";

/// E3 threshold multiplier `k`.
pub const K: f64 = 3.0;
/// `σ̂ = 1.4826 · MAD` (Gaussian consistency constant).
pub const MAD_TO_SIGMA: f64 = 1.4826;
/// Nights in a calibration window (E3) and at which provisional mode graduates (E9).
pub const CALIBRATION_NIGHTS: usize = 14;
/// Bridge nights on a comparator or driver change (E9).
pub const BRIDGE_NIGHTS: usize = 3;
/// Freeze window before a scheduled final, in nights (C18).
pub const FREEZE_NIGHTS: i64 = 14;
/// Relative SE of the MAD-based σ̂ is ≈ 1.166/√n (E9 `[C]`).
pub const MAD_REL_SE: f64 = 1.166;
/// SE of a median ≈ 1.2533 σ/√n (E9 `[C]`).
pub const MEDIAN_SE: f64 = 1.2533;

/// `θ_min = −ln 0.95` (E3).
#[must_use]
pub fn theta_min() -> f64 {
    -(0.95_f64.ln())
}

/// What an epoch is keyed on (§2.9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochKey {
    /// Weights identity.
    pub model_sha256: String,
    /// `comparator.build_commit`.
    pub comparator_build_commit: String,
    /// `comparator.flags`, as canonical JSON text.
    pub comparator_flags: String,
    /// `build_identity.driver`; `None` on a host with no accelerator driver.
    pub driver: Option<String>,
    /// `band.workload_id`.
    pub workload_id: String,
}

fn str_at<'a>(row: &'a Value, path: &[&str]) -> Result<&'a str, String> {
    let mut v = row;
    for p in path {
        v = v
            .get(p)
            .ok_or_else(|| format!("`{}` absent", path.join(".")))?;
    }
    v.as_str()
        .filter(|s| !s.is_empty() && *s != "unknown")
        .ok_or_else(|| format!("`{}` is not a known string", path.join(".")))
}

/// Read the epoch key of one nightly row.
///
/// # Errors
/// Names the first key field that is absent or unknown.
pub fn epoch_key(row: &Value) -> Result<EpochKey, String> {
    let flags = row
        .get("comparator")
        .and_then(|c| c.get("flags"))
        .ok_or("`comparator.flags` absent")?;
    let driver = match row.get("build_identity").and_then(|b| b.get("driver")) {
        None => return Err("`build_identity.driver` absent".to_string()),
        Some(Value::Null) => None,
        Some(_) => Some(str_at(row, &["build_identity", "driver"])?.to_string()),
    };
    Ok(EpochKey {
        model_sha256: str_at(row, &["model_sha256"])?.to_string(),
        comparator_build_commit: str_at(row, &["comparator", "build_commit"])?.to_string(),
        comparator_flags: canonical(flags),
        driver,
        workload_id: str_at(row, &["band", "workload_id"])?.to_string(),
    })
}

/// Canonical JSON text: object keys sorted at every level, no insignificant whitespace.
#[must_use]
pub fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", Value::String((*k).clone()), canonical(&m[*k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        other => other.to_string(),
    }
}

/// What a key change between two consecutive nights means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Same epoch.
    Same,
    /// Model or workload changed: fresh calibration, no carried baseline.
    Recalibrate(Vec<&'static str>),
    /// Comparator or driver changed (and nothing that forces recalibration): bridge.
    Bridge(Vec<&'static str>),
}

/// Classify the change from `prev` to `next`.
#[must_use]
pub fn classify(prev: &EpochKey, next: &EpochKey) -> Change {
    let mut recal = Vec::new();
    if prev.model_sha256 != next.model_sha256 {
        recal.push("model");
    }
    if prev.workload_id != next.workload_id {
        recal.push("workload");
    }
    let mut bridge = Vec::new();
    if prev.comparator_build_commit != next.comparator_build_commit
        || prev.comparator_flags != next.comparator_flags
    {
        bridge.push("comparator");
    }
    if prev.driver != next.driver {
        bridge.push("driver");
    }
    if !recal.is_empty() {
        recal.extend(bridge);
        Change::Recalibrate(recal)
    } else if !bridge.is_empty() {
        Change::Bridge(bridge)
    } else {
        Change::Same
    }
}

/// One detected epoch of a series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Epoch {
    /// 0-based position in the series.
    pub index: usize,
    /// What opened it (`Change::Same` for the first epoch).
    pub opened_by: Change,
    /// Its key.
    pub key: EpochKey,
    /// Nights (`YYYY-MM-DD`) in order.
    pub nights: Vec<String>,
}

/// Why a series could not be segmented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentError {
    /// A backfill row was offered to a series (§2.2).
    BackfillInSeries(usize),
    /// A row that is not a nightly row.
    WrongSchema(usize, String),
    /// A row without a readable epoch key or night.
    Unkeyed(usize, String),
}

fn night_of(row: &Value) -> Result<NaiveDate, String> {
    let ts = row.get("ts").and_then(Value::as_str).ok_or("`ts` absent")?;
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|t| t.with_timezone(&chrono::Utc).date_naive())
        .map_err(|_| "`ts` is not RFC 3339".to_string())
}

fn keyed(i: usize, row: &Value) -> Result<(NaiveDate, EpochKey), SegmentError> {
    match row.get("schema").and_then(Value::as_str) {
        Some(PERF_SCHEMA) => {}
        Some(BACKFILL_SCHEMA) => return Err(SegmentError::BackfillInSeries(i)),
        other => return Err(SegmentError::WrongSchema(i, format!("{other:?}"))),
    }
    let night = night_of(row).map_err(|e| SegmentError::Unkeyed(i, e))?;
    let key = epoch_key(row).map_err(|e| SegmentError::Unkeyed(i, e))?;
    Ok((night, key))
}

/// Split one series' nightly rows into epochs, in night order.
///
/// # Errors
/// Refuses the whole series on a backfill row, a non-nightly row or an unkeyed row: a
/// series with a hole in its key cannot say where its epochs are.
pub fn segment(rows: &[Value]) -> Result<Vec<Epoch>, SegmentError> {
    let mut nights = rows
        .iter()
        .enumerate()
        .map(|(i, r)| keyed(i, r))
        .collect::<Result<Vec<_>, _>>()?;
    nights.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out: Vec<Epoch> = Vec::new();
    for (night, key) in nights {
        let change = out.last().map_or(Change::Same, |e| classify(&e.key, &key));
        let night = night.format("%Y-%m-%d").to_string();
        match (out.last_mut(), change) {
            (Some(e), Change::Same) => e.nights.push(night),
            (_, opened_by) => out.push(Epoch {
                index: out.len(),
                opened_by,
                key,
                nights: vec![night],
            }),
        }
    }
    Ok(out)
}

/// Median of a non-empty slice (mean of the middle two when even).
#[must_use]
pub fn median(v: &[f64]) -> Option<f64> {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(s[n / 2]),
        _ => Some((s[n / 2 - 1] + s[n / 2]) / 2.0),
    }
}

/// `σ̂ = 1.4826 · med |x − med x|`.
#[must_use]
pub fn sigma_hat(v: &[f64]) -> Option<f64> {
    let m = median(v)?;
    let dev: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    Some(MAD_TO_SIGMA * median(&dev)?)
}

/// E3 calibration over a window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// `L̃`.
    pub median: f64,
    /// `σ̂`.
    pub sigma_hat: f64,
    /// `θ = max(θ_min, kσ̂)`.
    pub theta: f64,
}

/// E3 over the first `CALIBRATION_NIGHTS` of `nights`; `None` until that many exist.
#[must_use]
pub fn calibrate(nights: &[f64]) -> Option<Calibration> {
    let w = nights.get(..CALIBRATION_NIGHTS)?;
    let sigma = sigma_hat(w)?;
    Some(Calibration {
        median: median(w)?,
        sigma_hat: sigma,
        theta: theta_min().max(K * sigma),
    })
}

/// E9 bridge result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bridge {
    /// `Δ_e = med(L^{e'} − L^e)`.
    pub delta_e: f64,
    /// `σ̂` of the bridge differences.
    pub sigma_bridge: f64,
    /// `σ̂_bridge ≤ σ̂_e`.
    pub bridge_ok: bool,
}

/// Why a bridge was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    /// Not exactly `BRIDGE_NIGHTS` paired nights.
    NightCount {
        /// Old-configuration nights.
        old: usize,
        /// New-configuration nights.
        new: usize,
    },
    /// A non-finite `L` or `σ̂_e`.
    NonFinite,
}

/// E9: compare 3 side-by-side nights of the old (`e`) and new (`e′`) configuration.
///
/// # Errors
/// Refuses anything other than exactly 3 paired finite nights and a finite `σ̂_e ≥ 0`.
pub fn bridge(old: &[f64], new: &[f64], sigma_e: f64) -> Result<Bridge, BridgeError> {
    if old.len() != BRIDGE_NIGHTS || new.len() != BRIDGE_NIGHTS {
        return Err(BridgeError::NightCount {
            old: old.len(),
            new: new.len(),
        });
    }
    let all_finite = old.iter().chain(new).all(|x| x.is_finite());
    if !all_finite || !sigma_e.is_finite() || sigma_e < 0.0 {
        return Err(BridgeError::NonFinite);
    }
    let d: Vec<f64> = new.iter().zip(old).map(|(n, o)| n - o).collect();
    let sigma_bridge = sigma_hat(&d).ok_or(BridgeError::NonFinite)?;
    Ok(Bridge {
        delta_e: median(&d).ok_or(BridgeError::NonFinite)?,
        sigma_bridge,
        bridge_ok: sigma_bridge <= sigma_e,
    })
}

/// E9 carried baseline: `L_tag(e′) = L_tag(e) + Δ_e`.
#[must_use]
pub fn carried_baseline(l_tag: f64, b: &Bridge) -> f64 {
    l_tag + b.delta_e
}

/// E9 provisional threshold with `n_e ≥ 3` nights in the new epoch (bridge nights count).
///
/// `θ_prov = max(θ_min, k σ̂_{e′}(n_e)(1 + 2·1.166/√n_e)) + k·1.2533·σ̂_bridge/√3`
#[must_use]
pub fn theta_prov(n_e: usize, sigma_new: f64, sigma_bridge: f64) -> Option<f64> {
    if n_e < BRIDGE_NIGHTS {
        return None;
    }
    let n = n_e as f64;
    let widened = K * sigma_new * (1.0 + 2.0 * MAD_REL_SE / n.sqrt());
    let u_delta = K * MEDIAN_SE * sigma_bridge / (BRIDGE_NIGHTS as f64).sqrt();
    Some(theta_min().max(widened) + u_delta)
}

/// How an epoch was entered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Entry {
    /// The series' first epoch, or a model/workload change: calibrate from scratch.
    Fresh,
    /// A comparator/driver change with its bridge result.
    Bridged(Bridge),
}

/// Why an epoch has no admissible entry.
#[derive(Debug, Clone, PartialEq)]
pub enum EntryError {
    /// A comparator or driver change without a valid 3-night bridge.
    Unbridged(Vec<&'static str>, Option<BridgeError>),
}

/// The entry of an epoch opened by `change`, given its bridge (if one was measured).
///
/// # Errors
/// A `Change::Bridge` epoch without a valid bridge is refused, never admitted.
pub fn entry(
    change: &Change,
    measured: Option<Result<Bridge, BridgeError>>,
) -> Result<Entry, EntryError> {
    match (change, measured) {
        (Change::Same | Change::Recalibrate(_), _) => Ok(Entry::Fresh),
        (Change::Bridge(_), Some(Ok(b))) => Ok(Entry::Bridged(b)),
        (Change::Bridge(c), m) => Err(EntryError::Unbridged(c.clone(), m.and_then(Result::err))),
    }
}

/// Gate state of a series in its current epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Fewer than 14 nights in a fresh epoch: every rule is report-only.
    ReportOnly {
        /// Nights so far.
        nights: usize,
    },
    /// Failed bridge, fewer than 14 nights: armed with `θ_prov`.
    Provisional {
        /// `n_e`.
        n_e: usize,
        /// `θ_prov(n_e)`.
        theta: f64,
    },
    /// Armed with a full threshold (E3, or carried through a passing bridge).
    Calibrated {
        /// `θ`.
        theta: f64,
    },
}

/// The mode of a series given how its epoch was entered, its nights so far (`L_n`, the
/// bridge nights' `L^{e′}` first) and, for a passing bridge, the carried θ.
#[must_use]
pub fn mode(entry: Entry, nights: &[f64], carried_theta: Option<f64>) -> Mode {
    if let Some(c) = calibrate(nights) {
        return Mode::Calibrated { theta: c.theta };
    }
    match (entry, carried_theta) {
        (Entry::Bridged(b), Some(theta)) if b.bridge_ok => Mode::Calibrated { theta },
        (Entry::Bridged(b), _) => {
            let theta = sigma_hat(nights).and_then(|s| theta_prov(nights.len(), s, b.sigma_bridge));
            match theta {
                Some(theta) => Mode::Provisional {
                    n_e: nights.len(),
                    theta,
                },
                None => Mode::ReportOnly {
                    nights: nights.len(),
                },
            }
        }
        (Entry::Fresh, _) => Mode::ReportOnly {
            nights: nights.len(),
        },
    }
}

/// An epoch change refused by the freeze (S-12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frozen {
    /// The final it would land inside the window of.
    pub final_night: NaiveDate,
    /// Nights between the change and that final.
    pub nights_before: i64,
}

/// C18: refuse an epoch change on a gated host inside `T−14` nights of a scheduled final
/// (the change night `c` with `0 ≤ T − c ≤ 14`).
///
/// # Errors
/// Returns the nearest final whose freeze window contains `change_night`.
pub fn freeze_check(
    change_night: NaiveDate,
    finals: &[NaiveDate],
    gated: bool,
) -> Result<(), Frozen> {
    if !gated {
        return Ok(());
    }
    let hit = finals
        .iter()
        .map(|t| (*t, (*t - change_night).num_days()))
        .filter(|(_, d)| (0..=FREEZE_NIGHTS).contains(d))
        .min_by_key(|(_, d)| *d);
    match hit {
        Some((final_night, nights_before)) => Err(Frozen {
            final_night,
            nights_before,
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
#[path = "obs_epoch_tests.rs"]
mod tests;
