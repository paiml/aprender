//! GH-2391 falsifiers — a NaN or out-of-domain threshold must never reach a gate.
//!
//! `threshold_arg` shipped in the *-lint family and five callers used it. The
//! issue is broader: *any* CLI flag whose value ends up on one side of a
//! pass/fail comparison disarms that gate when it is NaN, because IEEE-754 makes
//! every comparison involving NaN false. An enumeration of `apr-cli`'s clap
//! definitions found 52 `f32`/`f64` flags, 23 of which feed a gate comparison;
//! 8 were guarded, 15 were not.
//!
//! These tests are the *user-visible* proof: the exact command line an operator
//! would type must be refused, and the refusal must name the flag.
//!
//! `apr grad-norm` is the mutation witness. It already had a
//! `spike_multiplier <= 0.0` check — which is itself NaN-blind, since
//! `NaN <= 0.0` is false — and its cap check is
//! `grad_norm_clipped > cap + 1e-6`. Against a NaN cap that is false for every
//! step, so the command printed a report and exited 0 over telemetry that
//! blew straight through the cap.

use super::threshold_arg::{guard, guard_f32, guard_opt, COSINE, FRACTION, TOLERANCE};
use crate::error::CliError;

/// `Commands` is a very large enum; building the clap command tree needs more
/// stack than the 2 MiB a test thread gets by default.
fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("join");
}

/// Every gate-feeding flag the enumeration found unguarded, as
/// `(base command line, flag, legitimate value, disarming values)`.
///
/// Each row is checked BOTH ways on purpose. `try_parse_from(...).is_err()` on
/// its own proves nothing: it is equally satisfied by a missing positional
/// argument, so a typo in the test's own command line would fake a pass. The
/// legitimate value pins the base command line as parseable, and only then does
/// the rejection of the disarming value mean what it claims.
type ThresholdCase = (
    &'static [&'static str],
    &'static str,
    &'static str,
    &'static [&'static str],
);

const GATE_FLAGS: &[ThresholdCase] = &[
    // apr diff --quant-roundtrip: "any tensor cosine < threshold" (CRUX-B-20).
    (
        &["apr", "diff", "a.apr", "b.apr"],
        "--threshold",
        "0.95",
        &["nan", "1.5", "-2"],
    ),
    // apr eval: perplexity pass/fail threshold.
    (
        &["apr", "eval", "m.apr"],
        "--threshold",
        "20.0",
        &["nan", "-1"],
    ),
    // apr profile: achieved-GFLOPS floor, then the three CI assertions.
    (
        &["apr", "profile", "m.apr"],
        "--threshold",
        "10.0",
        &["nan", "-1"],
    ),
    (
        &["apr", "profile", "m.apr"],
        "--assert-throughput",
        "100",
        &["nan", "-1", "inf"],
    ),
    (
        &["apr", "profile", "m.apr"],
        "--assert-p99",
        "50",
        &["nan", "-1"],
    ),
    (
        &["apr", "profile", "m.apr"],
        "--assert-p50",
        "25",
        &["nan", "-1"],
    ),
    // apr qa: the project's own release gate.
    (
        &["apr", "qa", "m.gguf"],
        "--assert-tps",
        "100",
        &["nan", "-1", "inf"],
    ),
    (
        &["apr", "qa", "m.gguf"],
        "--assert-speedup",
        "1.0",
        &["nan", "-1"],
    ),
    (
        &["apr", "qa", "m.gguf"],
        "--assert-gpu-speedup",
        "2.0",
        &["nan", "-1"],
    ),
    (
        &["apr", "qa", "m.gguf"],
        "--regression-threshold",
        "0.10",
        &["nan", "-1", "42"],
    ),
    // apr compare-hf: "max_abs_diff < threshold".
    (
        &["apr", "compare-hf", "m.apr", "--hf", "org/repo"],
        "--threshold",
        "1e-5",
        &["nan", "-1"],
    ),
    // apr grad-norm: cap-violation check and spike detection.
    (
        &["apr", "grad-norm", "--history-file", "h.json"],
        "--max-grad-norm",
        "1.0",
        &["nan", "-1"],
    ),
    (
        &["apr", "grad-norm", "--history-file", "h.json"],
        "--spike-multiplier",
        "10.0",
        &["nan", "-1"],
    ),
    // apr cbtop --ci: minimum throughput.
    (&["apr", "cbtop"], "--throughput", "225.0", &["nan", "-1"]),
    // apr probar tensor --assert: cosine floor for the golden comparison.
    (
        &["apr", "probar", "tensor", "m.apr"],
        "--tolerance",
        "0.98",
        &["nan", "2.0", "-2"],
    ),
    // apr rosetta: verify tolerance, inference mismatch rate, sigma threshold.
    (
        &["apr", "rosetta", "verify", "m.apr"],
        "--tolerance",
        "1e-5",
        &["nan", "-1"],
    ),
    (
        &["apr", "rosetta", "compare-inference", "a.gguf", "b.apr"],
        "--tolerance",
        "0.1",
        &["nan", "-1", "42"],
    ),
    (
        &["apr", "rosetta", "validate-stats", "m.apr"],
        "--threshold",
        "3.0",
        &["nan", "-1"],
    ),
];

/// `apr pretrain` only exists under the `training` feature, so its row is kept
/// out of the always-on table rather than making the whole table's "must parse"
/// premise feature-dependent.
#[cfg(feature = "training")]
const TRAINING_GATE_FLAGS: &[ThresholdCase] = &[(
    &[
        "apr",
        "pretrain",
        "--dataset",
        "d.bin",
        "--tokenizer",
        "tok/",
        "--run-dir",
        "run/",
    ],
    "--target-val-loss",
    "2.2",
    &["nan", "-1"],
)];

#[test]
fn cli_refuses_a_disarming_threshold_on_every_newly_guarded_gate_flag() {
    on_big_stack(cli_gate_flag_table_body);
}

fn cli_gate_flag_table_body() {
    use clap::Parser;

    let mut cases: Vec<&ThresholdCase> = GATE_FLAGS.iter().collect();
    #[cfg(feature = "training")]
    cases.extend(TRAINING_GATE_FLAGS.iter());

    for (base, flag, good, bad_values) in cases {
        let mut ok_argv: Vec<&str> = base.to_vec();
        ok_argv.push(flag);
        ok_argv.push(good);
        assert!(
            crate::Cli::try_parse_from(ok_argv.iter().copied()).is_ok(),
            "the test's own command line must parse, or the rejections below \
             prove nothing: {ok_argv:?}"
        );

        for bad in *bad_values {
            let mut argv: Vec<&str> = base.to_vec();
            argv.push(flag);
            argv.push(bad);
            assert!(
                crate::Cli::try_parse_from(argv.iter().copied()).is_err(),
                "clap accepted a gate-disarming threshold: {argv:?}"
            );
        }
    }
}

/// The whole family the fix touches must still accept its own documented
/// defaults. Guarding a flag is only correct if it did not narrow the domain,
/// and every `good` value in the table above is either the flag's shipped
/// default or a value the falsification suites pass on the command line.
#[test]
fn cli_still_accepts_the_documented_values_for_those_same_flags() {
    on_big_stack(cli_defaults_body);
}

fn cli_defaults_body() {
    use clap::Parser;

    // Combining every flag of a command in one invocation catches a guard that
    // only works when the flag is passed alone.
    let combined: &[&[&str]] = &[
        &[
            "apr",
            "profile",
            "m.apr",
            "--threshold",
            "10.0",
            "--assert-throughput",
            "100",
            "--assert-p99",
            "50",
            "--assert-p50",
            "25",
        ],
        &[
            "apr",
            "qa",
            "m.gguf",
            "--assert-tps",
            "100",
            "--assert-speedup",
            "1.0",
            "--assert-gpu-speedup",
            "2.0",
            "--regression-threshold",
            "0.10",
        ],
        &[
            "apr",
            "grad-norm",
            "--history-file",
            "h.json",
            "--max-grad-norm",
            "1.0",
            "--spike-multiplier",
            "10.0",
        ],
    ];

    for argv in combined {
        assert!(
            crate::Cli::try_parse_from(argv.iter().copied()).is_ok(),
            "clap rejected a legitimate combination: {argv:?}"
        );
    }
}

/// MUTATION WITNESS for the `guard()` half of the fix (the half a non-clap
/// caller reaches). `apr grad-norm` is handed telemetry whose clipped norm is
/// 99.0 against a NaN `--max-grad-norm`. `max_exceeds_cap` is
/// `clipped > cap + 1e-6`, false against NaN, so without the guard the command
/// prints its report and returns `Ok(())`: a cap violation reported as a clean
/// run. With the guard it refuses, before it even opens the file.
#[test]
fn grad_norm_refuses_a_nan_cap_instead_of_reporting_a_clean_run() {
    let dir = std::env::temp_dir().join(format!(
        "apr-gh2391-gradnorm-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let history = dir.join("history.json");
    // grad_norm_clipped = 99.0 blows through any sane cap; with a finite cap
    // this file makes the command exit non-zero.
    std::fs::write(
        &history,
        r#"[{"step":0,"grad_norm":100.0,"grad_norm_clipped":99.0,"loss":2.0}]"#,
    )
    .expect("write history");

    let err = super::grad_norm::run(&history, Some(f64::NAN), 16, 10.0, false)
        .expect_err("a NaN --max-grad-norm must not produce a clean run");
    let msg = err.to_string();
    assert!(
        msg.contains("--max-grad-norm"),
        "the refusal must name the flag the operator typed, got: {msg}"
    );
    assert!(
        msg.contains("NaN"),
        "the refusal must say why NaN is not a threshold, got: {msg}"
    );

    // Control: the same file against a finite cap still reaches the gate and
    // still fails it, so the guard did not replace the gate.
    let gate_err = super::grad_norm::run(&history, Some(1.0), 16, 10.0, false)
        .expect_err("clipped 99.0 exceeds a cap of 1.0");
    assert!(
        gate_err.to_string().contains("exceeds --max-grad-norm cap"),
        "expected the cap gate to fire, got: {gate_err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The NaN-blind range check `spike_multiplier <= 0.0` that shipped in
/// `grad_norm` is the whole bug class in one line: it looks like validation and
/// admits NaN. The guard must catch what it misses.
#[test]
fn grad_norm_refuses_a_nan_spike_multiplier_that_its_own_range_check_admits() {
    let nan_multiplier = "nan".parse::<f64>().expect("NaN literal");
    assert!(
        !(nan_multiplier <= 0.0),
        "premise: the shipped `spike_multiplier <= 0.0` check is false for NaN"
    );
    let err = guard("--spike-multiplier", f64::NAN, TOLERANCE)
        .expect_err("NaN must be refused by the guard the range check missed");
    assert!(err.to_string().contains("--spike-multiplier"), "{err}");
}

/// `guard_opt` distinguishes "no assertion" from "an assertion that cannot
/// fail". `None` is a legitimate absence of a gate; `Some(NaN)` is a gate that
/// was armed and then disarmed.
#[test]
fn guard_opt_admits_none_and_refuses_some_nan() {
    guard_opt("--assert-tps", None, TOLERANCE).expect("no assertion is not a disarmed gate");
    guard_opt("--assert-tps", Some(100.0), TOLERANCE).expect("100 tok/s is a real floor");

    let err = guard_opt("--assert-tps", Some(f64::NAN), TOLERANCE)
        .expect_err("Some(NaN) must be refused");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(msg.contains("--assert-tps"), "got: {msg}");
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // A negative throughput floor is satisfied by every measurement, including
    // a broken one, so it is a disarm even though it is finite.
    guard_opt("--assert-tps", Some(-1.0), TOLERANCE)
        .expect_err("a negative floor disarms the gate");
}

/// The `f32` half of the family. Half of apr-cli's gate thresholds are `f32`;
/// widening for the domain check must not change the verdict.
#[test]
fn guard_f32_refuses_the_same_values_as_the_f64_guard() {
    guard_f32("--threshold", f32::NAN, COSINE).expect_err("NaN cosine floor");
    guard_f32("--threshold", f32::INFINITY, TOLERANCE).expect_err("infinite floor");
    guard_f32("--tolerance", 2.0, COSINE).expect_err("cosine cannot exceed 1.0");
    guard_f32("--tolerance", -0.5, FRACTION).expect_err("a fraction cannot be negative");

    guard_f32("--threshold", 0.95, COSINE).expect("0.95 is a real cosine floor");
    guard_f32("--threshold", 3.0, TOLERANCE).expect("3 sigma is a real threshold");
    guard_f32("--tolerance", 0.1, FRACTION).expect("10% mismatch is a real tolerance");
}

/// Parsing an `f32` flag through the `f64` domain check must not silently
/// widen what the flag accepts: `1e-300` is representable as `f64` and rounds
/// to `0.0` as `f32`, which would hand a "tolerance" gate a zero it never asked
/// for. Parsing as `f32` first keeps the value the user typed and the value the
/// gate sees identical.
#[test]
fn f32_parsers_do_not_launder_a_value_through_f64() {
    use super::threshold_arg::parse_tolerance_f32;

    let parsed = parse_tolerance_f32("1e-300").expect("finite and non-negative");
    assert_eq!(
        parsed, 0.0f32,
        "an f32 flag must report the f32 value the gate will actually use"
    );

    assert_eq!(parse_tolerance_f32("0.95"), Ok(0.95f32));
    assert_eq!(parse_tolerance_f32("nan"), parse_tolerance_f32("NaN"));
    parse_tolerance_f32("inf").expect_err("an infinite tolerance disarms the gate");
    parse_tolerance_f32("banana").expect_err("not a float literal");
}
