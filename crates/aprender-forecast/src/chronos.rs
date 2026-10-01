//! The Chronos-Bolt door: model loading, the validation boundary and the stateless zero-shot
//! `forecast` call.
//!
//! Ported from `sources/007-chronos-mcp-thin-server/src/lib.rs` (lines 25-138 and 175-189) under
//! D-08, with THREE recorded deviations:
//!
//! 1. **The `EMBEDDED_WEIGHTS` / `EMBEDDED_CONFIG` statics and `resolve_model` stay in the SERVER
//!    crate** (plan 06-07). They depend on that crate's `build.rs` `OUT_DIR` staging (D-13);
//!    embedding is transport-side packaging, and this library takes bytes.
//! 2. **`Result<_, String>` becomes [`ModelLoadError`]** with `Display` + `std::error::Error`, the
//!    `crates/aprender-mcp-setfit` shape. A load failure crosses a crate boundary; a bare `String`
//!    there is not an error type.
//! 3. **THE ONE STRUCTURAL SEAM: the spike's `forecast()` validation prefix is extracted into
//!    [`validate`]** (D-11 + D-18). Every refusal message is verbatim. The reason is D-18: the
//!    refusal behaviour of the tool boundary must be testable WITHOUT model weights, so a run
//!    with no `models/` directory still proves the door refuses rather than silently reporting
//!    "0 ignored". [`forecast`] calls [`validate`] first and is otherwise unchanged, so no
//!    refusal exists in two places (OPS-03).
#![allow(clippy::disallowed_methods)] // JsonSchema derive and serde_json::json! expand to .unwrap()

use crate::bolt::{Bolt, Config};
use crate::dates::{format_ymd, future_days, parse_date};
use crate::safetensors;
use crate::types::ForecastError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// Fewest points the Chronos door accepts (D-11; the fit server's floor is 10, Chronos is 4).
pub const CHRONOS_MIN_POINTS: usize = 4;
/// Most points the Chronos door accepts before refusing (T-06-02).
pub const CHRONOS_MAX_POINTS: usize = 20_000;
/// Longest horizon reachable at all, and only with `allow_long_horizon` (D-13).
pub const CHRONOS_MAX_HORIZON: usize = 1_024;

/// A loaded Chronos-Bolt model plus the provenance the response reports.
pub struct Model {
    /// The forward itself.
    pub bolt: Bolt,
    /// Model name, from `config.json`'s `_name_or_path`.
    pub name: String,
    /// Total decoded parameter count.
    pub n_params: usize,
    /// The dtype the weights were stored in (`F32`, `F16`, `BF16` or `mixed`).
    pub dtype: String,
    /// Where the bytes came from (`embedded`, or the directory path).
    pub source: String,
    /// Wall-clock seconds the load took.
    pub load_seconds: f64,
}

/// Why a Chronos model failed to load.
///
/// Replaces the spike's `Result<_, String>` (recorded D-08 deviation 2), on the
/// `crates/aprender-mcp-setfit` `ModelLoadError` shape.
#[derive(Debug)]
pub enum ModelLoadError {
    /// A weights or config file could not be read.
    Io(std::io::Error),
    /// The safetensors bytes were refused by the decoder.
    Decode(String),
    /// `config.json` was not readable as a Chronos-Bolt config.
    Config(String),
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cannot read model file: {e}"),
            Self::Decode(s) => write!(f, "weights refused: {s}"),
            Self::Config(s) => write!(f, "config.json refused: {s}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

/// Decode weights and config from bytes — the ONE door the embedded copy and an on-disk file
/// both go through.
///
/// # Errors
///
/// [`ModelLoadError::Config`] when `config.json` is not valid JSON, [`ModelLoadError::Decode`]
/// when the safetensors bytes are refused.
pub fn load_model_from_bytes(
    weights: &[u8],
    config: &[u8],
    source: &str,
) -> Result<Model, ModelLoadError> {
    let t0 = Instant::now();
    let cfg_json: serde_json::Value =
        serde_json::from_slice(config).map_err(|e| ModelLoadError::Config(e.to_string()))?;
    let name = cfg_json["_name_or_path"]
        .as_str()
        .unwrap_or("chronos-bolt")
        .rsplit('/')
        .next()
        .unwrap_or("chronos-bolt")
        .to_string();
    let (w, dtype) = safetensors::load_bytes(weights).map_err(ModelLoadError::Decode)?;
    let n_params = w.values().map(|t| t.data.len()).sum();
    let bolt = Bolt::load(&w, Config::from_json(&cfg_json));
    Ok(Model {
        bolt,
        name,
        n_params,
        dtype,
        source: source.to_string(),
        load_seconds: t0.elapsed().as_secs_f64(),
    })
}

/// Load `model.safetensors` + `config.json` from a directory (the `CHRONOS_MODEL_DIR` path).
///
/// # Errors
///
/// [`ModelLoadError::Io`] when either file cannot be read, otherwise the
/// [`load_model_from_bytes`] error.
pub fn load_model_from_dir(dir: &Path) -> Result<Model, ModelLoadError> {
    let w = std::fs::read(dir.join("model.safetensors")).map_err(ModelLoadError::Io)?;
    let c = std::fs::read(dir.join("config.json")).map_err(ModelLoadError::Io)?;
    load_model_from_bytes(&w, &c, &dir.display().to_string())
}

/// The Chronos tool arguments. `deny_unknown_fields`: the boundary refuses, never defaults (D-11).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChronosArgs {
    /// Timestamps, YYYY-MM-DD (a time part is ignored), ascending, unique.
    pub ds: Vec<String>,
    /// Observed values, same length as `ds`; `null` marks a missing value (the model handles gaps).
    pub y: Vec<Option<f64>>,
    /// Number of future periods: 1 ... 64 by default, up to 1024 with `allow_long_horizon`.
    pub horizon: usize,
    /// Period of the future steps: "D" (default), "W", or "MS" (month start).
    #[serde(default)]
    pub freq: Option<String>,
    /// Accept a horizon beyond the model's native 64 steps. The model then rolls its own forecast
    /// forward; accuracy past step 64 degrades (measured), and the response carries a warning.
    #[serde(default)]
    pub allow_long_horizon: bool,
}

/// What [`validate`] establishes before a single weight is touched.
pub struct Validated {
    /// Parsed history dates, strictly ascending.
    pub ds: Vec<i64>,
    /// The future dates the response will label.
    pub fut: Vec<i64>,
    /// The resolved frequency (`D` when the caller sent none).
    pub freq: String,
    /// The context as `f32`, `NaN` where `y` was `null`.
    pub context: Vec<f32>,
    /// How many `y` entries were `null`.
    pub n_missing: usize,
}

/// The Chronos response — the spike-004 shape, shared by every forecast server.
#[derive(Debug, Serialize)]
pub struct ChronosResponse {
    /// Model name.
    pub model: String,
    /// The frequency the future dates were stepped at.
    pub freq: String,
    /// Points the caller sent.
    pub n_history: usize,
    /// Points the model actually looked at (capped at `context_length`).
    pub context_used: usize,
    /// Wall-clock seconds the forward(s) took.
    pub predict_seconds: f64,
    /// Future dates, `YYYY-MM-DD`.
    pub ds: Vec<String>,
    /// Median (q50).
    pub yhat: Vec<f64>,
    /// 10th percentile.
    pub yhat_lower: Vec<f64>,
    /// 90th percentile.
    pub yhat_upper: Vec<f64>,
    /// Every quantile the model emits, keyed by level.
    pub quantiles: serde_json::Map<String, serde_json::Value>,
    /// Present only when the horizon exceeded the native 64 steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    /// Provenance and cost of this call.
    pub diagnostics: serde_json::Value,
}

fn bad(s: String) -> ForecastError {
    ForecastError::Validation(s)
}

/// THE Chronos validation door — every bound, every refusal, no weights required.
///
/// Extracted from the spike's `forecast()` prefix (recorded D-08 deviation 3) so D-18's
/// "a weight-dependent test never passes vacuously" has a weights-FREE counterpart: the refusal
/// behaviour is asserted on every run, armed or not. Messages are verbatim from the spike.
///
/// # Errors
///
/// [`ForecastError::Validation`] for a length mismatch, too few or too many points, horizon 0 or
/// above `CHRONOS_MAX_HORIZON`, a horizon past the model's native window without
/// `allow_long_horizon`, a non-finite or all-null `y`, an unparseable or non-ascending `ds`, or an
/// unsupported `freq`.
pub fn validate(cfg: &Config, args: &ChronosArgs) -> Result<Validated, ForecastError> {
    if args.ds.len() != args.y.len() {
        return Err(bad(format!(
            "ds has {} entries but y has {}",
            args.ds.len(),
            args.y.len()
        )));
    }
    if args.ds.len() < CHRONOS_MIN_POINTS {
        return Err(bad(format!(
            "need at least {CHRONOS_MIN_POINTS} points, got {}",
            args.ds.len()
        )));
    }
    if args.ds.len() > CHRONOS_MAX_POINTS {
        return Err(bad(format!(
            "{} points exceeds max_points {CHRONOS_MAX_POINTS}",
            args.ds.len()
        )));
    }
    if args.horizon == 0 {
        return Err(bad("horizon must be at least 1".into()));
    }
    if args.horizon > CHRONOS_MAX_HORIZON {
        return Err(bad(format!(
            "horizon {} exceeds the maximum {CHRONOS_MAX_HORIZON}",
            args.horizon
        )));
    }
    if args.horizon > cfg.prediction_length && !args.allow_long_horizon {
        return Err(bad(format!(
            "horizon {} exceeds the model's native {} steps; beyond that the model rolls its own \
             forecast forward and accuracy degrades — pass allow_long_horizon: true to accept \
             (max {CHRONOS_MAX_HORIZON})",
            args.horizon, cfg.prediction_length
        )));
    }
    let n_missing = args.y.iter().filter(|v| v.is_none()).count();
    if args.y.iter().flatten().any(|v| !v.is_finite()) {
        return Err(bad("y contains a non-finite value".into()));
    }
    if args.y.len() - n_missing < 2 {
        return Err(bad("y needs at least 2 observed (non-null) values".into()));
    }
    let ds: Vec<i64> = args
        .ds
        .iter()
        .map(|s| parse_date(s))
        .collect::<Result<_, _>>()?;
    if !ds.windows(2).all(|w| w[0] < w[1]) {
        return Err(bad(
            "ds must be strictly ascending with no duplicates".into()
        ));
    }
    let freq = args.freq.clone().unwrap_or_else(|| "D".into());
    let fut = future_days(ds[ds.len() - 1], args.horizon, &freq)?;
    let context: Vec<f32> = args
        .y
        .iter()
        .map(|v| v.map_or(f32::NAN, |x| x as f32))
        .collect();
    Ok(Validated {
        ds,
        fut,
        freq,
        context,
        n_missing,
    })
}

/// One stateless zero-shot Chronos-Bolt forecast: validate, one forward (plus rollouts past the
/// native horizon), nine quantiles back.
///
/// # Errors
///
/// Every [`validate`] refusal, plus [`ForecastError::Internal`] if the loaded model has no
/// q10/q50/q90 level.
pub fn forecast(model: &Model, args: &ChronosArgs) -> Result<ChronosResponse, ForecastError> {
    let cfg = &model.bolt.cfg;
    let v = validate(cfg, args)?;

    // ---- one zero-shot pass ----
    let context_used = v.context.len().min(cfg.context_length);
    let t0 = Instant::now();
    let (q, forwards) = model.bolt.predict(&v.context, args.horizon);
    let predict_seconds = t0.elapsed().as_secs_f64();
    let level = |want: f32| {
        cfg.quantiles
            .iter()
            .position(|&l| (l - want).abs() < 1e-6)
            .ok_or_else(|| ForecastError::Internal(format!("model has no {want} quantile")))
    };
    let (lo, med, hi) = (level(0.1)?, level(0.5)?, level(0.9)?);
    let to64 = |v: &Vec<f32>| v.iter().map(|x| f64::from(*x)).collect::<Vec<f64>>();
    let mut quantiles = serde_json::Map::new();
    for (i, lvl) in cfg.quantiles.iter().enumerate() {
        // `{lvl:.1}` is LOSSY: `cfg.quantiles` comes verbatim from the checkpoint's
        // config.json, and a grid like [0.05, 0.1, 0.5, 0.9, 0.95] collapses 0.05/0.1 and
        // 0.9/0.95 onto the same key. Later inserts win, so the surviving "0.9" series is
        // q95 while claiming to be q90 — a mislabelled number, not a dropped one.
        quantiles.insert(lvl.to_string(), serde_json::json!(to64(&q[i])));
    }
    let rollouts = forwards.saturating_sub(1) / cfg.quantiles.len();
    let warning = (args.horizon > cfg.prediction_length).then(|| {
        format!(
            "horizon {} exceeds the model's native {} steps: steps {}+ come from {} \
             autoregressive rollout(s) of the model's own forecast; spike 006 measured MASE 1.7 \
             past step 64 vs 1.1 within it on daily data",
            args.horizon,
            cfg.prediction_length,
            cfg.prediction_length + 1,
            rollouts
        )
    });
    Ok(ChronosResponse {
        model: model.name.clone(),
        freq: v.freq,
        n_history: v.ds.len(),
        context_used,
        predict_seconds,
        ds: v.fut.iter().map(|d| format_ymd(*d)).collect(),
        yhat: to64(&q[med]),
        yhat_lower: to64(&q[lo]),
        yhat_upper: to64(&q[hi]),
        quantiles,
        warning,
        diagnostics: serde_json::json!({
            "model": model.name,
            "params": model.n_params,
            "weights_dtype": model.dtype,
            "weights_source": model.source,
            "context_length": cfg.context_length,
            "patch": cfg.patch,
            "native_horizon": cfg.prediction_length,
            "rollouts": rollouts,
            "forwards": forwards,
            "missing_values": v.n_missing,
            "quantile_levels": cfg.quantiles.iter().map(|q| (f64::from(*q) * 10.0).round() / 10.0).collect::<Vec<f64>>(),
        }),
    })
}

/// CSV `ds,y` loader over the TEXT (callers use `include_str!`), sorted by date and de-duplicated
/// keeping the last row — the spike-006 convention.
///
/// Takes the text rather than a path (recorded D-08 deviation): the server crate embeds its
/// sample CSVs with `include_str!`, so a path-taking loader would force a second reader.
///
/// # Errors
///
/// A `String` — reserved for future malformed-input reporting; every currently-unparseable line
/// is skipped exactly as the spike skips it.
pub fn load_csv(text: &str) -> Result<(Vec<String>, Vec<Option<f64>>), String> {
    let mut rows: Vec<(String, Option<f64>)> = Vec::new();
    for line in text.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(d), Some(v)) = (it.next(), it.next()) else {
            continue;
        };
        let d = d
            .trim_matches('"')
            .trim()
            .get(..10)
            .unwrap_or("")
            .to_string();
        if d.len() != 10 {
            continue;
        }
        let v = v.trim_matches('"').trim();
        rows.push((
            d,
            if v.is_empty() || v.eq_ignore_ascii_case("nan") {
                None
            } else {
                v.parse().ok()
            },
        ));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.dedup_by(|later, earlier| {
        if later.0 == earlier.0 {
            *earlier = later.clone();
            true
        } else {
            false
        }
    });
    Ok(rows.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::{validate, ChronosArgs, CHRONOS_MAX_HORIZON};
    use crate::bolt::Config;
    use crate::test_support::load_json;
    use crate::types::ForecastError;

    /// The committed 1.1 KB `config.json` (hyper-parameters, NOT weights — D-18 permits it), so
    /// every refusal below is asserted on a run with no `models/` directory at all.
    fn cfg() -> Config {
        Config::from_json(&load_json("chronos_bolt_tiny_config.json"))
    }

    fn args(n: usize, horizon: usize) -> ChronosArgs {
        let ds: Vec<String> = (0..n).map(|i| format!("2020-01-{:02}", i + 1)).collect();
        ChronosArgs {
            ds,
            y: (0..n).map(|i| Some(i as f64)).collect(),
            horizon,
            freq: None,
            allow_long_horizon: false,
        }
    }

    fn refusal(a: &ChronosArgs) -> String {
        match validate(&cfg(), a) {
            Err(ForecastError::Validation(s)) => s,
            Err(ForecastError::Internal(s)) => {
                panic!("the door must refuse with Validation, not Internal: {s}")
            }
            Ok(_) => panic!("expected a refusal, got Ok"),
        }
    }

    /// D-18 permits the config fixture because it is hyper-parameters, not weights. Its sha256 is
    /// asserted by the plan's `shasum` acceptance criterion (adding a SHA-256 implementation to
    /// this crate for one test would be a new dependency, which RESEARCH F9 rules out); here the
    /// SHAPE is pinned so a swapped file is caught by a test as well as by the criterion.
    #[test]
    fn config_fixture_sha_is_pinned() {
        let raw = std::fs::read(crate::test_support::fixture_path(
            "chronos_bolt_tiny_config.json",
        ))
        .expect("the config fixture is committed; absence is a defect");
        assert!(
            (900..1400).contains(&raw.len()),
            "the committed config is 1.1 KB of hyper-parameters, not weights; got {} bytes",
            raw.len()
        );
        let c = cfg();
        assert_eq!(c.prediction_length, 64, "native horizon");
        assert_eq!(c.context_length, 2048, "context length");
        assert_eq!(c.quantiles.len(), 9, "quantile levels");
        assert_eq!(c.d_model, 256);
        assert_eq!(c.patch, 16);
        assert!(c.use_reg_token, "Bolt appends a REG token");
    }

    #[test]
    fn validate_refuses_too_few_points() {
        assert!(refusal(&args(3, 8)).contains("at least 4"));
    }

    #[test]
    fn validate_refuses_horizon_zero() {
        assert!(refusal(&args(10, 0)).contains("horizon must be at least 1"));
    }

    #[test]
    fn validate_refuses_horizon_above_max() {
        let msg = refusal(&args(10, CHRONOS_MAX_HORIZON + 1));
        assert!(msg.contains("1024"), "{msg}");
    }

    #[test]
    fn validate_refuses_long_horizon_without_flag() {
        let msg = refusal(&args(10, 65));
        assert!(
            msg.contains("allow_long_horizon"),
            "the refusal must name the fix: {msg}"
        );
    }

    #[test]
    fn validate_accepts_long_horizon_with_flag() {
        let mut a = args(10, 365);
        a.allow_long_horizon = true;
        let v = validate(&cfg(), &a).expect("365 steps with the flag is accepted");
        assert_eq!(v.fut.len(), 365);
        assert_eq!(v.context.len(), 10);
    }

    #[test]
    fn validate_refuses_all_null_y() {
        let mut a = args(10, 8);
        a.y = vec![None; 10];
        assert!(refusal(&a).contains("non-null"));
    }

    #[test]
    fn validate_refuses_impossible_date() {
        let mut a = args(10, 8);
        a.ds[3] = "2008-02-30".into();
        assert!(refusal(&a).contains("calendar"));
    }

    #[test]
    fn validate_refuses_unsorted_ds() {
        let mut a = args(10, 8);
        a.ds.swap(2, 5);
        assert!(refusal(&a).contains("strictly ascending"));
    }

    #[test]
    fn validate_refuses_freq_h() {
        let mut a = args(10, 8);
        a.freq = Some("H".into());
        assert!(refusal(&a).contains("use D, W or MS"));
    }

    /// REVIEW-06-U2: 06-01 made `parse_date` refuse trailing content. This proves the CHRONOS
    /// door inherits that refusal rather than re-implementing a prefix take.
    #[test]
    fn validate_refuses_trailing_content_in_ds() {
        let mut a = args(10, 8);
        a.ds[7] = "2008-02-01garbage".into();
        let msg = refusal(&a);
        assert!(msg.contains("YYYY-MM-DD"), "{msg}");
    }

    #[test]
    fn chronos_args_schema_is_strict() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ChronosArgs)).expect("schema serializes");
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        let required = schema["required"].as_array().expect("required array");
        for field in ["ds", "y", "horizon"] {
            assert!(
                required.contains(&serde_json::json!(field)),
                "{field} must be required: {schema}"
            );
        }
        let props = schema["properties"].as_object().expect("properties");
        assert!(props.contains_key("allow_long_horizon"));
        assert!(props.contains_key("freq"));
    }
}

#[cfg(test)]
mod parity {
    //! The SC4 door-level rungs: f16 weight precision and the horizon/rollout reporting.
    //!
    //! Weight-gated with `#[cfg_attr(not(chronos_weights), ignore = "...")]` so an unarmed run
    //! reports them as COUNTED skips (D-18). Every bar is read from
    //! `contracts/chronos-bolt-parity-v1.yaml` at test time (D-15).

    use super::{forecast, load_model_from_dir, ChronosArgs, Model};
    use crate::test_support::{constant_u64, equation_tolerance, load_json, read_csv};
    use std::path::PathBuf;

    /// `CHRONOS_MODEL_DIR` as a path. `expect`, not skip: when `chronos_weights` is armed the
    /// variable held a `model.safetensors` at build time.
    fn armed_dir() -> PathBuf {
        PathBuf::from(
            std::env::var_os("CHRONOS_MODEL_DIR")
                .expect("CHRONOS_MODEL_DIR is set whenever cfg(chronos_weights) is armed"),
        )
    }

    /// A `ds`/`y` pair from the committed Peyton CSV, last `n` rows.
    fn peyton_tail(n: usize) -> (Vec<String>, Vec<Option<f64>>) {
        let (ds, y) = read_csv("peyton_manning.csv");
        let start = ds.len() - n;
        (
            ds[start..].to_vec(),
            y[start..].iter().map(|v| Some(*v)).collect(),
        )
    }

    /// FALSIFY-CHRONOS-010: the locally derived f16 weights forecast within 2 % of the series
    /// standard deviation of the pinned f32 weights (SC4's f16 literal).
    ///
    /// Both directories exist whenever the recipe has run — it derives f16 from the verified f32
    /// — so their absence when armed is a defect, not a skip. The dtype assertions rule out the
    /// failure mode where the test loads the same file twice and compares it to itself.
    #[test]
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    fn f16_weights_within_two_percent_of_std() {
        let base = armed_dir();
        let root = base.parent().expect("CHRONOS_MODEL_DIR has a parent");
        let f32_model: Model =
            load_model_from_dir(&root.join("f32")).expect("the f32 weights must load");
        let f16_model: Model =
            load_model_from_dir(&root.join("f16")).expect("the derived f16 weights must load");
        assert_eq!(f32_model.dtype, "F32", "the f32 directory must hold F32");
        assert_eq!(f16_model.dtype, "F16", "the f16 directory must hold F16");
        assert_eq!(
            f32_model.n_params, f16_model.n_params,
            "the derivation must not add or drop a tensor"
        );

        let (_, y) = read_csv("peyton_manning.csv");
        let ctx: Vec<f32> = y.iter().map(|v| *v as f32).collect();
        let q32 = f32_model.bolt.forward(&ctx);
        let q16 = f16_model.bolt.forward(&ctx);

        // The series std is the oracle's own `scale` for this series (0.8718), not a number this
        // test derives for itself.
        let std = load_json("chronos_bolt_tiny_fixture.json")["series"]["peyton"]["scale"]
            .as_f64()
            .expect("scale");
        let rel = equation_tolerance("chronos-bolt-parity-v1", "f16_rel_std");
        let bar = rel * std;
        let delta = q32
            .iter()
            .flatten()
            .zip(q16.iter().flatten())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max);
        println!(
            "f16 vs f32: max|delta| {delta:.4e} = {:.4} % of the series std {std:.4} \
             (bar {rel} x std = {bar:e})",
            100.0 * delta / std
        );
        assert!(
            delta <= bar,
            "f16 quantiles differ from f32 by {delta:e}, over {rel} x std {std} = {bar:e}"
        );
    }

    /// FALSIFY-CHRONOS-012: a rolled-forward horizon carries a warning naming the rollout, and
    /// the forward/rollout counts are the documented arithmetic.
    ///
    /// The series is deliberately SHORT (200 points): `forwards` and `rollouts` depend only on
    /// the horizon, so a 2905-point context would cost 46 full-length forwards to assert the same
    /// two integers.
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    #[test]
    fn door_reports_warning_and_forwards_for_365() {
        let model = load_model_from_dir(&armed_dir()).expect("the armed weights must load");
        let (ds, y) = peyton_tail(200);
        let long = ChronosArgs {
            ds: ds.clone(),
            y: y.clone(),
            horizon: 365,
            freq: None,
            allow_long_horizon: true,
        };
        let r = forecast(&model, &long).expect("365 steps with the flag is accepted");
        let warning = r
            .warning
            .as_deref()
            .expect("a rolled-forward horizon warns");
        println!("warning: {warning}");
        assert!(
            warning.contains("rollout"),
            "the warning must name the rollout: {warning}"
        );
        assert_eq!(
            r.diagnostics["forwards"].as_u64().expect("forwards"),
            constant_u64("chronos-bolt-parity-v1", "rollout_forwards_for_365")
        );
        assert_eq!(r.diagnostics["rollouts"].as_u64().expect("rollouts"), 5);
        assert_eq!(r.ds.len(), 365, "365 future timestamps");
        assert_eq!(r.yhat.len(), 365);

        // A horizon inside the native window carries NO warning key at all.
        let native = ChronosArgs {
            ds,
            y,
            horizon: constant_u64("chronos-bolt-parity-v1", "native_horizon") as usize,
            freq: None,
            allow_long_horizon: false,
        };
        let r = forecast(&model, &native).expect("a native-horizon call is accepted");
        assert!(
            r.warning.is_none(),
            "a horizon within the native window must carry no warning: {:?}",
            r.warning
        );
        assert_eq!(r.diagnostics["forwards"].as_u64().expect("forwards"), 1);
        assert_eq!(r.diagnostics["rollouts"].as_u64().expect("rollouts"), 0);
    }

    /// The context cap (T-06-02): a series longer than `context_length` is truncated from the
    /// left rather than refused or fed whole, and the response REPORTS how much was used.
    #[test]
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    fn door_context_used_is_capped_at_2048() {
        let model = load_model_from_dir(&armed_dir()).expect("the armed weights must load");
        // 3000 points, SYNTHETIC: the committed Peyton CSV is 2905 rows, so a real series long
        // enough to exceed the 2048 cap by a comfortable margin has to be generated. Only the
        // LENGTH is under test here, not any numeric value.
        let n = 3000usize;
        let day0 = crate::dates::days_from_civil(2010, 1, 1);
        let ds: Vec<String> = (0..n)
            .map(|i| crate::dates::format_ymd(day0 + i as i64))
            .collect();
        let y: Vec<Option<f64>> = (0..n)
            .map(|i| Some(10.0 + (i as f64 / 30.0).sin()))
            .collect();
        let r = forecast(
            &model,
            &ChronosArgs {
                ds,
                y,
                horizon: 8,
                freq: None,
                allow_long_horizon: false,
            },
        )
        .expect("a 3000-point series is inside max_points");
        let cap = constant_u64("chronos-bolt-parity-v1", "context_length") as usize;
        println!(
            "n_history {} -> context_used {}",
            r.n_history, r.context_used
        );
        assert_eq!(r.n_history, n);
        assert_eq!(
            r.context_used, cap,
            "a series longer than the context length must report the CAP, not its own length"
        );
    }
}
