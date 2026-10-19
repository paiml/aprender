//! Scenario "repo-flow": the FLOW-006 repo template (SIM-6, #5511).
//!
//! The PR-flow line with the
//! merge queue optional (`builds == 0` merges on arm) and the conflict step: an armed PR is found
//! dirty with `p_dirty` and waits `t_rebase_min` for a rebase and a rerun. One parameter file per
//! repo; inputs come from the KPI report, never typed.

use super::pr_flow::{run_pr_flow, PrFlowConfig, PrFlowResult, ReviewMode};

/// A named repo case: `(name, supply N, t_author, cap, R_pr, runner-min, jobs, setup, p_red, p_dirty, t_rebase)`.
type Case = (
    &'static str,
    usize,
    f64,
    usize,
    usize,
    f64,
    usize,
    f64,
    f64,
    f64,
    f64,
);

const CASES: [Case; 7] = [
    (
        "rmedia-design",
        9,
        25.0,
        10,
        9,
        17.1,
        3,
        3.0,
        0.1,
        0.05,
        60.0,
    ),
    (
        "infra-today",
        7,
        540.0,
        10,
        6,
        40.0,
        1,
        5.0,
        0.2,
        0.3,
        10000.0,
    ),
    ("infra-rule", 7, 540.0, 10, 6, 40.0, 1, 5.0, 0.2, 0.3, 60.0),
    (
        "infra-design",
        16,
        45.0,
        15,
        6,
        10.0,
        1,
        10.0,
        0.1,
        0.05,
        60.0,
    ),
    ("infra-33", 16, 45.0, 15, 6, 33.0, 1, 10.0, 0.1, 0.05, 60.0),
    ("small-10", 16, 45.0, 10, 12, 20.0, 1, 3.0, 0.1, 0.05, 60.0),
    ("small-15", 16, 45.0, 15, 12, 20.0, 1, 3.0, 0.1, 0.05, 60.0),
];

/// Names of the FLOW-006 repo cases.
#[must_use]
pub fn repo_cases() -> Vec<&'static str> {
    CASES.iter().map(|c| c.0).collect()
}

/// The configuration of a named repo case; no merge queue (`builds = 0`).
#[must_use]
pub fn repo_preset(name: &str) -> Option<PrFlowConfig> {
    let c = CASES.iter().find(|c| c.0 == name)?;
    Some(PrFlowConfig {
        cap: c.3,
        n: c.1 as f64,
        t_author_min: c.2,
        r_pr: c.4,
        pr_runner_min: c.5,
        pr_jobs: c.6,
        pr_setup_min: c.7,
        p_red: c.8,
        t_fix_min: 10.0,
        t_arm_min: 2.0,
        r_mq: 1,
        mq_runner_min: 0.0,
        mq_jobs: 1,
        mq_setup_min: 0.0,
        builds: 0,
        mq_batch: 1,
        mq_wait_min: 0.0,
        f: 0.0,
        p_dirty: c.9,
        t_rebase_min: c.10,
        t_rev_min: 0.0,
        rev_mode: ReviewMode::Serial,
        rev_lanes: 0,
        rev_tiers: Vec::new(),
    })
}

/// The reference run of a repo case: seeds 1..=5, 400 h, 20 h warm-up, averaged.
#[must_use]
pub fn run_repo_reference(c: &PrFlowConfig) -> PrFlowResult {
    let mut acc = PrFlowResult::default();
    for seed in 1..=5 {
        let r = run_pr_flow(c, seed, 400.0, 20.0);
        acc.merged_per_h += r.merged_per_h / 5.0;
        acc.w_mean_min += r.w_mean_min / 5.0;
        acc.w_p90_min += r.w_p90_min / 5.0;
        acc.l += r.l / 5.0;
        acc.pr_util += r.pr_util / 5.0;
        acc.runs_per_pr += r.runs_per_pr / 5.0;
    }
    acc
}

/// Override the case with the ledger's measured inputs (a JSON object); unknown keys are refused.
///
/// # Errors
/// Returns the offending key or type when the object holds something the model does not read.
pub fn apply_inputs(c: &mut PrFlowConfig, v: &serde_json::Value) -> Result<(), String> {
    let obj = v.as_object().ok_or("inputs must be a JSON object")?;
    for (k, x) in obj {
        let f = x.as_f64().ok_or_else(|| format!("{k}: not a number"))?;
        match k.as_str() {
            "cap" => c.cap = f as usize,
            "N" => c.n = f,
            "t_author_min" => c.t_author_min = f,
            "R_pr" => c.r_pr = f as usize,
            "pr_runner_min" => c.pr_runner_min = f,
            "p_red" => c.p_red = f,
            "t_fix_min" => c.t_fix_min = f,
            "t_arm_min" => c.t_arm_min = f,
            "p_dirty" => c.p_dirty = f,
            "t_rebase_min" => c.t_rebase_min = f,
            _ => return Err(format!("unknown input: {k}")),
        }
    }
    Ok(())
}

/// The `sim` block fields this scenario owns: scenario, seeds, hours, inputs, predicted, binding.
#[must_use]
pub fn sim_block(case: &str, c: &PrFlowConfig, r: &PrFlowResult) -> serde_json::Value {
    let binding = if r.pr_util > 0.85 {
        "ci_pool"
    } else if r.l >= c.cap as f64 * 0.95 {
        "cap"
    } else {
        "supply"
    };
    serde_json::json!({
        "scenario": format!("repo-flow:{case}"),
        "seeds": 5, "hours": 400, "warmup_h": 20,
        "inputs": {
            "cap": c.cap, "N": c.n, "t_author_min": c.t_author_min, "R_pr": c.r_pr,
            "pr_runner_min": c.pr_runner_min, "p_red": c.p_red, "t_fix_min": c.t_fix_min,
            "t_arm_min": c.t_arm_min, "p_dirty": c.p_dirty, "t_rebase_min": c.t_rebase_min,
        },
        "predicted": {
            "merged_h": r.merged_per_h, "W_min_mean": r.w_mean_min, "W_min_p90": r.w_p90_min,
            "L": r.l, "pr_pool_util_pct": r.pr_util * 100.0, "runs_per_pr": r.runs_per_pr,
        },
        "binding_station": binding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FLOW-006 reference bands: (case, merged/h, tolerance).
    const WANT: [(&str, f64, f64); 7] = [
        ("rmedia-design", 21.13, 0.05),
        ("infra-today", 0.142, 0.10),
        ("infra-rule", 0.78, 0.05),
        ("infra-design", 21.22, 0.05),
        ("infra-33", 9.71, 0.05),
        ("small-10", 17.48, 0.05),
        ("small-15", 21.38, 0.05),
    ];

    fn got(name: &str) -> PrFlowResult {
        run_repo_reference(&repo_preset(name).expect("case"))
    }

    #[test]
    fn merges_per_hour_match_the_reference_bands() {
        for (name, want, tol) in WANT {
            let r = got(name);
            assert!(
                (r.merged_per_h - want).abs() <= want * tol,
                "{name}: {:.3} vs {want}",
                r.merged_per_h
            );
        }
    }

    #[test]
    fn littles_law_holds_within_half_a_percent() {
        // infra-today strands PRs past the horizon: they sit in L but never in W, so Little is
        // not defined for that unsteady case.
        for (name, _, _) in WANT.iter().filter(|w| w.0 != "infra-today") {
            let r = got(name);
            let lam_w = r.merged_per_h * r.w_mean_min / 60.0;
            assert!(
                (lam_w - r.l).abs() <= r.l * 0.005,
                "{name}: {lam_w:.3} vs {:.3}",
                r.l
            );
        }
    }

    #[test]
    fn same_seed_same_output() {
        let c = repo_preset("infra-rule").expect("case");
        assert_eq!(
            run_pr_flow(&c, 3, 100.0, 5.0),
            run_pr_flow(&c, 3, 100.0, 5.0)
        );
    }

    #[test]
    fn keep_fresh_rule_alone_lifts_infra_fivefold() {
        let (a, b) = (got("infra-today"), got("infra-rule"));
        assert!(b.merged_per_h > 4.0 * a.merged_per_h);
    }

    #[test]
    fn a_reverted_rule_shows_red() {
        let mut c = repo_preset("infra-rule").expect("case");
        c.t_rebase_min = 10000.0;
        let r = run_repo_reference(&c);
        assert!(
            (r.merged_per_h - 0.78).abs() > 0.78 * 0.05,
            "mutant passed at {:.3}",
            r.merged_per_h
        );
    }

    #[test]
    fn unknown_case_is_none() {
        assert!(repo_preset("nope").is_none());
    }
}
