//! FALSIFY-BENCH-2519: nothing in this crate may report a measurement.
//!
//! `sweep.rs:169 simulate_training` was a closed-form parabola whose vertex is
//! the constant its own comment names:
//!
//!     // Simulated training - in real implementation would run actual training
//!     // - Temperature ~4.0 is optimal
//!     let deviation = (param_value - 4.0).abs();
//!     let loss = 0.65 + deviation * 0.1 + noise;
//!
//! Measured before the fix, with no model, no data and no config:
//!
//!     Optimal: temperature = 4.00 (loss=0.6043, accuracy=80.7%)
//!     3.50 -> 0.6543    4.50 -> 0.6543    (symmetric about the vertex)
//!
//! `strategies.rs:80 simulate` was the same defect as a per-variant literal
//! table (`Combined -> 0.71/0.831`), fed to a real Welch t-test that duly
//! reported `p=0.0000 ✓` and `Recommendation: Combined for best accuracy`.
//! `cost.rs generate_sample_points` was an eight-row literal table that
//! `recommend` turned into `Top recommendation: LoRA r=32`.
//!
//! This is worse than the `aprender-train-inspect` case it accompanies: those
//! outputs tell a user WHICH HYPERPARAMETER TO USE.
//!
//! Note on the API surface used below: these tests deliberately touch only
//! items that exist BOTH before and after the fix, so the file still compiles
//! against the pre-fix tree. A mutation check whose test target fails to
//! compile proves nothing, so the cost-analysis assertions go through the CLI
//! rather than through `cost::load_points`, which is new.

use entrenar_bench::{
    compare_strategies, temperature_sweep, DistillStrategy, SweepConfig, Sweeper,
};

/// Write a two-run results file and return its path (with its tempdir, which
/// must stay alive for the duration of the test).
fn measured_results(body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("measured.json");
    std::fs::write(&path, body).expect("write results");
    (dir, path)
}

fn bench_cmd() -> assert_cmd::Command {
    assert_cmd::Command::cargo_bin("aprender-train-bench").expect("binary should be built")
}

#[test]
fn a_sweep_cannot_report_an_optimal_value_with_no_inputs() {
    // The exact invocation from #2519: no model, no dataset, no config.
    let result = temperature_sweep(1.0..8.0, 0.5, 3);

    let err = result.expect_err(
        "a sweep with no model and no data returned Ok -- it is reporting a \
         closed-form curve as a measurement again, which is the #2519 defect",
    );
    // The refusal must not smuggle the recommendation back in as advice:
    // `Optimal: temperature = 4.00` is the exact line being retired.
    let text = format!("{err}");
    assert!(
        !text.contains("Optimal:") && !text.contains("4.0"),
        "the refusal still names a recommended value. Got: {text}"
    );
}

#[test]
fn a_strategy_comparison_cannot_name_a_winner_with_no_inputs() {
    let strategies = [
        DistillStrategy::kd_only(),
        DistillStrategy::progressive(),
        DistillStrategy::attention(),
        DistillStrategy::combined(),
    ];

    assert!(
        compare_strategies(&strategies).is_err(),
        "compare_strategies returned Ok without training anything. The old output \
         ended in `Recommendation: Combined for best accuracy`, derived from four \
         hardcoded pairs of numbers."
    );
}

/// Discriminating test -- the analogue of "two different files of equal size
/// must not give identical answers".
///
/// `1.0..2.5` and `5.5..7.0` are mirror images about the baked-in vertex 4.0, so
/// the closed form gives them the SAME losses in reverse order. Before the fix
/// both succeeded and did exactly that: proof the numbers came from arithmetic
/// on the parameter, not from anything that ran. Whatever this crate does, it
/// must not succeed here with mirrored results -- either it errors, or it
/// genuinely trained and the two ranges disagree.
#[test]
fn mirror_image_ranges_must_not_produce_the_same_curve_reversed() {
    let low = temperature_sweep(1.0..2.5, 0.5, 1);
    let high = temperature_sweep(5.5..7.0, 0.5, 1);

    if let (Ok(l), Ok(h)) = (&low, &high) {
        let low_losses: Vec<f64> = l.data_points.iter().map(|p| p.mean_loss).collect();
        let high_losses: Vec<f64> = h.data_points.iter().rev().map(|p| p.mean_loss).collect();

        assert_ne!(
            low_losses, high_losses,
            "temperatures 1.0-2.5 and 5.5-7.0 produced identical losses in mirror \
             order -- the answer is a function of |value - 4.0|, not of training"
        );
    }
}

/// Second discriminating angle: a range that contains no optimum at all still
/// got one. Before the fix, sweeping only the falling side of the parabola
/// still ★-marked its last point as `Optimal`, and the two sweeps below --
/// which share no parameter value whatsoever -- both answered with confidence.
#[test]
fn disjoint_ranges_must_not_both_report_an_optimum() {
    let a = temperature_sweep(1.0..2.0, 0.5, 1);
    let b = temperature_sweep(6.0..7.0, 0.5, 1);

    let both_confident = matches!((&a, &b), (Ok(ra), Ok(rb))
        if ra.optimal.is_some() && rb.optimal.is_some());

    assert!(
        !both_confident,
        "two disjoint temperature ranges, neither containing any measurement, \
         both reported an `Optimal` point"
    );
}

#[test]
fn every_sweep_parameter_refuses_not_just_temperature() {
    // The alpha curve was the same closed form with its vertex at the other
    // hardcoded constant (0.7). Fixing only temperature would leave half the
    // fabrication reachable.
    let alpha = Sweeper::new(SweepConfig::alpha(0.1..0.9, 0.1).with_runs(3)).run();
    assert!(alpha.is_err(), "the alpha sweep still returns results");
}

/// Non-vacuity companion 1: the tests above are all satisfied by a crate that
/// refuses everything for one blanket reason. This pins that a genuine
/// computation on real input still SUCCEEDS -- the Pareto analysis was never
/// the problem, it just never had measured data to chew on. Both configuration
/// names below come from the file, and the `--max-cost` constraint really does
/// filter one of them out.
#[test]
fn analysis_of_real_measured_results_still_succeeds() {
    let (_dir, path) = measured_results(
        r#"[{"name":"cheap-run","gpu_hours":8.0,"cost_usd":17.68,"accuracy":0.81,
             "loss":0.42,"memory_gb":18.0},
            {"name":"dear-run","gpu_hours":120.0,"cost_usd":265.2,"accuracy":0.92,
             "loss":0.25,"memory_gb":56.0}]"#,
    );

    let output = bench_cmd()
        .args(["recommend", "--max-cost", "50", "--results"])
        .arg(&path)
        .output()
        .expect("binary should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "recommending from a real results file failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("cheap-run"), "got:\n{stdout}");
    assert!(
        !stdout.contains("dear-run"),
        "the $50 constraint did not filter the $265 run:\n{stdout}"
    );
}

/// Non-vacuity companion 2: the refusals are not one undifferentiated error.
/// An empty strategy list is still diagnosed as an empty strategy list, and a
/// results file that is missing still fails by naming the path.
#[test]
fn other_failures_keep_their_own_distinct_reasons() {
    let empty = compare_strategies(&[]).expect_err("an empty strategy list must fail");
    let empty_text = format!("{empty}");
    assert!(
        empty_text.contains("No strategies to compare"),
        "got: {empty_text}"
    );
    assert!(!empty_text.contains("never trains"), "got: {empty_text}");

    let missing = bench_cmd()
        .args([
            "cost-performance",
            "--results",
            "/nonexistent/measured.json",
        ])
        .output()
        .expect("binary should run");
    let stderr = String::from_utf8_lossy(&missing.stderr);
    assert!(!missing.status.success());
    assert!(stderr.contains("measured.json"), "got:\n{stderr}");
}

/// The user-facing surface, since that is where the misleading table appeared.
/// `aprender-train-bench temperature` with no arguments exited 0 and printed
/// `Optimal: temperature = 4.00`.
#[test]
fn the_cli_no_longer_prints_a_recommended_hyperparameter() {
    for subcommand in ["temperature", "alpha", "compare", "ablation"] {
        let output = bench_cmd()
            .arg(subcommand)
            .output()
            .expect("binary should run");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !output.status.success(),
            "`{subcommand}` exited 0 with no model and no data:\n{stdout}"
        );
        assert!(
            !stdout.contains("Optimal:") && !stdout.contains("Recommendation:"),
            "`{subcommand}` still recommends a configuration:\n{stdout}"
        );
    }
}

/// The cost commands took a `--results` flag and ignored it, substituting a
/// literal table. Refusing without measured input is the point; accepting it
/// when supplied is what keeps the refusal honest rather than a dead end.
#[test]
fn the_cli_requires_measured_results_before_recommending() {
    for subcommand in ["recommend", "cost-performance"] {
        let bare = bench_cmd()
            .arg(subcommand)
            .output()
            .expect("binary should run");

        let bare_stdout = String::from_utf8_lossy(&bare.stdout);
        assert!(
            !bare.status.success(),
            "`{subcommand}` exited 0 with no measured results"
        );
        assert!(
            !bare_stdout.contains("Top recommendation") && !bare_stdout.contains("LoRA r="),
            "`{subcommand}` still answers from the hardcoded table:\n{bare_stdout}"
        );
    }

    let (_dir, path) = measured_results(
        r#"[{"name":"only-run","gpu_hours":8.0,"cost_usd":17.68,"accuracy":0.81,
             "loss":0.42,"memory_gb":18.0}]"#,
    );

    let supplied = bench_cmd()
        .args(["recommend", "--results"])
        .arg(&path)
        .output()
        .expect("binary should run");

    let supplied_stdout = String::from_utf8_lossy(&supplied.stdout);
    assert!(
        supplied.status.success(),
        "`recommend --results` failed on a valid results file:\n{}",
        String::from_utf8_lossy(&supplied.stderr)
    );
    assert!(
        supplied_stdout.contains("only-run"),
        "`recommend --results` did not report the run from the file:\n{supplied_stdout}"
    );
    // The old literal table must not resurface alongside the real one.
    assert!(
        !supplied_stdout.contains("LoRA r=32"),
        "the hardcoded configuration table is still being mixed in:\n{supplied_stdout}"
    );
}
