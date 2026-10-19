//! SIM-2 (FLOW-005, ARB-SOT-001 §3): the arbiter's `pr-flow --params` surface.
//!
//! Arbiter writes one params file per repo; this module reads it through ONE adapter
//! ([`parse_params`]), runs the PR-flow model for the repo as measured (`actual`), for the
//! design (`optimum`), and for each design change alone (`one_change`), and names the binding
//! station as the single change that moves merged/h the most.
//!
//! The params shape is provisional until infra pins `arbiter-sim-params-v1`: only
//! [`parse_params`] knows it, so the pin edits one function.

use super::pr_flow::{preset, run_pr_flow, PrFlowConfig, PrFlowResult};
use serde_json::{json, Value};

/// Params fields ARB-SOT-001 §3 item 1 lists, in the order it lists them.
const FIELDS: [&str; 14] = [
    "n",
    "t_author",
    "cap",
    "r_pr",
    "pr_runner_min",
    "mq_runner_min",
    "p_red",
    "t_fix",
    "t_arm",
    "builds",
    "batch",
    "f",
    "p_dirty",
    "t_rebase",
];

/// Fields the model does not read yet (SIM-6 adds them); reported, never silently dropped.
const UNMODELLED: [&str; 2] = ["p_dirty", "t_rebase"];

/// A repo's parsed params.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub repo: String,
    pub design: String,
    pub cfg: PrFlowConfig,
    /// Fields carrying a measured value, out of [`FIELDS`], in percent.
    pub measured_pct: f64,
    /// Names of the fields that fell back to the design value.
    pub design_default: Vec<String>,
}

/// Read a params file: `{"repo", "design", "params": {field: number | {"value", "source", "window"}}}`.
/// A missing field, or one whose value is not a number, is `design_default`.
///
/// # Errors
/// A message naming the problem: not JSON, no `repo`, unknown `design`, a negative or non-finite value.
pub fn parse_params(text: &str) -> Result<Params, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("params: not JSON: {e}"))?;
    let repo = v["repo"]
        .as_str()
        .ok_or("params: missing `repo`")?
        .to_owned();
    let design = v["design"].as_str().unwrap_or("D").to_owned();
    let mut cfg = preset(&design).ok_or_else(|| format!("params: unknown design `{design}`"))?;
    let mut measured = 0usize;
    let mut design_default = Vec::new();
    for name in FIELDS {
        let raw = &v["params"][name];
        let val = raw.get("value").unwrap_or(raw).as_f64();
        match val {
            Some(x) if x.is_finite() && x >= 0.0 => {
                measured += 1;
                set_field(&mut cfg, name, x);
            }
            Some(x) => {
                return Err(format!(
                    "params: `{name}` = {x} is not a non-negative number"
                ))
            }
            None => design_default.push(name.to_owned()),
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let measured_pct = measured as f64 / FIELDS.len() as f64 * 100.0;
    Ok(Params {
        repo,
        design,
        cfg,
        measured_pct,
        design_default,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn set_field(c: &mut PrFlowConfig, name: &str, x: f64) {
    match name {
        "n" => c.n = x,
        "t_author" => c.t_author_min = x,
        "cap" => c.cap = x as usize,
        "r_pr" => c.r_pr = x as usize,
        "pr_runner_min" => c.pr_runner_min = x,
        "mq_runner_min" => c.mq_runner_min = x,
        "p_red" => c.p_red = x,
        "t_fix" => c.t_fix_min = x,
        "t_arm" => c.t_arm_min = x,
        "builds" => c.builds = x as usize,
        "batch" => c.mq_batch = x as usize,
        "f" => c.f = x,
        _ => {} // p_dirty, t_rebase: see UNMODELLED
    }
}

/// Mean over seeds `1..=seeds`.
fn run_mean(c: &PrFlowConfig, seeds: u64, hours: f64, warmup_h: f64) -> (PrFlowResult, f64) {
    let rs: Vec<PrFlowResult> = (1..=seeds)
        .map(|s| run_pr_flow(c, s, hours, warmup_h))
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n = rs.len() as f64;
    let mean = |f: fn(&PrFlowResult) -> f64| rs.iter().map(f).sum::<f64>() / n;
    let m = mean(|r| r.merged_per_h);
    let var = if rs.len() > 1 {
        rs.iter().map(|r| (r.merged_per_h - m).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let r = PrFlowResult {
        merged_per_h: m,
        w_mean_min: mean(|r| r.w_mean_min),
        w_p90_min: mean(|r| r.w_p90_min),
        l: mean(|r| r.l),
        prs_per_pass: mean(|r| r.prs_per_pass),
        pr_util: mean(|r| r.pr_util),
        mq_util: mean(|r| r.mq_util),
        runs_per_pr: mean(|r| r.runs_per_pr),
        rho_a: mean(|r| r.rho_a),
        run_wall_min: mean(|r| r.run_wall_min),
        checks_min: mean(|r| r.checks_min),
        arm_min: mean(|r| r.arm_min),
        mq_min: mean(|r| r.mq_min),
        pass_wait_min: mean(|r| r.pass_wait_min),
        pass_wall_min: mean(|r| r.pass_wall_min),
    };
    (r, 1.96 * (var / n).sqrt())
}

/// The design changes, each as "move this repo's supply fields to the design's".
fn changes(actual: &PrFlowConfig, design: &PrFlowConfig) -> Vec<(&'static str, PrFlowConfig)> {
    type Apply = fn(&mut PrFlowConfig, &PrFlowConfig);
    let table: [(&'static str, Apply); 8] = [
        ("cap", |a, d| a.cap = d.cap),
        ("r_pr", |a, d| a.r_pr = d.r_pr),
        ("r_mq", |a, d| a.r_mq = d.r_mq),
        ("builds", |a, d| a.builds = d.builds),
        ("runner_min", |a, d| {
            a.pr_runner_min = d.pr_runner_min;
            a.mq_runner_min = d.mq_runner_min;
        }),
        ("p_red", |a, d| a.p_red = d.p_red),
        ("flake", |a, d| a.f = d.f),
        ("setup", |a, d| {
            a.pr_setup_min = d.pr_setup_min;
            a.mq_setup_min = d.mq_setup_min;
        }),
    ];
    table
        .into_iter()
        .filter_map(|(name, apply)| {
            let mut c = actual.clone();
            apply(&mut c, design);
            (&c != actual).then_some((name, c))
        })
        .collect()
}

/// Run the model for one repo and build the §3 item 5 row (`actual`, `optimum`, `one_change`).
#[must_use]
pub fn evaluate(p: &Params, seeds: u64, hours: f64, warmup_h: f64) -> Value {
    let design = preset(&p.design).unwrap_or_else(|| p.cfg.clone());
    let (actual, ci95) = run_mean(&p.cfg, seeds, hours, warmup_h);
    let moves = changes(&p.cfg, &design);
    let one_change: Vec<Value> = moves
        .iter()
        .map(|(name, c)| json!({"change": name, "merged_h": run_mean(c, seeds, hours, warmup_h).0.merged_per_h}))
        .collect();
    let mut opt_cfg = p.cfg.clone();
    for (_, c) in &moves {
        opt_cfg = merge_supply(&opt_cfg, c, &p.cfg);
    }
    let (opt, _) = run_mean(&opt_cfg, seeds, hours, warmup_h);
    let binding = one_change
        .iter()
        .max_by(|a, b| {
            a["merged_h"]
                .as_f64()
                .partial_cmp(&b["merged_h"].as_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(Value::Null, |c| c["change"].clone());
    json!({
        "repo": p.repo,
        "params_measured_pct": p.measured_pct,
        "design_default": p.design_default,
        "unmodelled": UNMODELLED,
        "actual": {
            "merged_h": actual.merged_per_h, "ci95_h": ci95,
            "W_min_mean": actual.w_mean_min, "W_min_p90": actual.w_p90_min, "L": actual.l,
            "pr_pool_util_pct": actual.pr_util * 100.0, "mq_util_pct": actual.mq_util * 100.0,
            "runs_per_pr": actual.runs_per_pr,
            "rho_a": actual.rho_a
        },
        "binding_station": binding,
        "optimum": {
            "scenario": p.design, "merged_h": opt.merged_per_h,
            "changes": moves.iter().map(|(n, _)| *n).collect::<Vec<_>>()
        },
        "one_change": one_change
    })
}

/// `base` with every field that `changed` moved off `orig` taken from `changed`.
fn merge_supply(base: &PrFlowConfig, changed: &PrFlowConfig, orig: &PrFlowConfig) -> PrFlowConfig {
    let mut c = base.clone();
    macro_rules! take { ($($f:ident),*) => { $( if changed.$f != orig.$f { c.$f = changed.$f; } )* } }
    take!(
        cap,
        r_pr,
        r_mq,
        builds,
        pr_runner_min,
        mq_runner_min,
        p_red,
        f,
        pr_setup_min,
        mq_setup_min
    );
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    const D_ALL: &str = r#"{"repo":"aprender","design":"D","params":{"n":11.1,"cap":10,"r_pr":28,
        "builds":1,"f":0.12,"p_red":{"value":0.62,"source":"jobs.tsv","window":"7d"}}}"#;

    #[test]
    fn missing_fields_are_design_defaults_and_counted() {
        let p = parse_params(D_ALL).expect("parse");
        assert_eq!(p.cfg.cap, 10);
        assert!((p.cfg.p_red - 0.62).abs() < 1e-12);
        assert_eq!(p.cfg.pr_runner_min, preset("D").expect("D").pr_runner_min);
        assert!(p.design_default.contains(&"t_arm".to_owned()));
        assert!((p.measured_pct - 6.0 / 14.0 * 100.0).abs() < 1e-9);
    }

    #[test]
    fn bad_params_are_refused_by_name() {
        assert!(parse_params("nope").is_err());
        assert!(parse_params(r#"{"design":"D"}"#)
            .unwrap_err()
            .contains("repo"));
        assert!(parse_params(r#"{"repo":"x","design":"Z"}"#)
            .unwrap_err()
            .contains("design"));
        assert!(parse_params(r#"{"repo":"x","params":{"cap":-1}}"#)
            .unwrap_err()
            .contains("cap"));
    }

    #[test]
    fn a_design_point_repo_has_no_changes_and_equals_the_reference() {
        let p = parse_params(r#"{"repo":"x","design":"E"}"#).expect("parse");
        let v = evaluate(&p, 5, 400.0, 20.0);
        assert!(v["one_change"].as_array().expect("arr").is_empty());
        assert!(v["binding_station"].is_null());
        let m = v["actual"]["merged_h"].as_f64().expect("num");
        assert!((m - 21.37).abs() <= 21.37 * 0.05, "{m}");
        assert_eq!(v["params_measured_pct"], 0.0);
    }

    #[test]
    fn the_binding_change_is_the_one_that_moves_merged_h_most() {
        let p = parse_params(D_ALL).expect("parse");
        let v = evaluate(&p, 3, 200.0, 10.0);
        let rows = v["one_change"].as_array().expect("arr");
        let best = rows
            .iter()
            .max_by(|a, b| {
                a["merged_h"]
                    .as_f64()
                    .partial_cmp(&b["merged_h"].as_f64())
                    .expect("cmp")
            })
            .expect("row");
        assert_eq!(v["binding_station"], best["change"]);
        assert!(
            v["optimum"]["merged_h"].as_f64().expect("n")
                > v["actual"]["merged_h"].as_f64().expect("n")
        );
    }

    #[test]
    fn same_params_same_bytes() {
        let p = parse_params(D_ALL).expect("parse");
        assert_eq!(
            evaluate(&p, 2, 100.0, 5.0).to_string(),
            evaluate(&p, 2, 100.0, 5.0).to_string()
        );
    }
}
