//! PMAT-4105 (#3606 follow-up): the stage falsifier through the call sites that SHIP.
//!
//! The helper tests above prove `timed()` and `planted_delay()` in isolation. That is how a `load`
//! plant that moved no field got through once (`inference_result.rs`, the load-plant comment): the
//! hand-rolled wiring in `run_gguf_inference` slept OUTSIDE the measured window, and every helper
//! test stayed green. So every case here goes through [`run_inference`] on a real GGUF file, once
//! per planted stage. It asserts the plant moves ITS field by at least the plant, and moves no
//! other field (`unattributed_ms` included) by more than its tolerance, measured against a clean
//! baseline run of the same model. The CPU legs use the fixed [`TOL_MS`]; the GPU legs widen each
//! field's tolerance by the noise it showed across clean runs (see [`Band`], #4105).
//!
//! What each path CAN measure, and where each case runs:
//! - CPU (dense or qwen35): only `load`. Every other plant has no site on this path, so it must
//!   land NOWHERE: no field moves and the wall clock does not grow by the plant. Runs in CI on
//!   a generated fixture.
//! - CUDA dense: load, h2d, validate, prefill, decode. `#[ignore]`d. Runs on a GPU runner with
//!   `APR_STAGE_CUDA_GGUF=<dense gguf>`. Run the GPU legs from a `--release` build: in debug, the
//!   guard's CPU reference forward takes tens of seconds and varies run to run by far more than
//!   [`TOL_MS`], so a debug run can fail on noise that has nothing to do with attribution.
//! - CUDA qwen35: the dense set plus validate_ref/validate_probe, which are INSIDE validate (so
//!   their plant moves validate too). `#[ignore]`d. Needs `APR_STAGE_CUDA_QWEN35_GGUF`, and forces
//!   a fresh F2 guard into a private receipt dir, so a cached receipt cannot skip the guard.
//!
//! The wall-clock boundary these fields close against is `run_gguf_inference`'s `run_start`
//! (just before the load stage), NOT `apr run`'s `inference_time_ms`, which also covers resolving
//! the model and preparing tokens. `apr run --json` reports the difference as `outside_wall_ms`.

use super::with_delay;
use crate::infer::stage_timings::StageTimings;
use crate::infer::{run_inference, InferenceConfig};

/// The plant. Far larger than this fixture's whole CPU run (single-digit ms), so a plant that
/// lands is unmistakable and one that lands nowhere cannot hide in noise. It is large against
/// [`TOL_MS`] too: a plant in the WRONG field moves that field by ~`PLANT_MS`, which is more than
/// three tolerances away from any jitter.
const PLANT_MS: u64 = 1000;
/// How far any field OTHER than the planted one may move between the planted run and the clean
/// baseline. Stated once, here: the falsifier's tolerance is a claim about attribution, never
/// about arithmetic (the books close exactly by construction; see `StageTimings::close`).
/// Sized from a measurement: on the CUDA leg (lambda, RTX 4090, qwen2.5-coder-0.5b q4_k_m, release)
/// `load_ms` alone varied by up to 113.7 ms between two clean runs (the mmap + prefault of a 400 MB
/// file against the page cache), which failed an earlier 100 ms tolerance on noise alone.
const TOL_MS: f64 = 300.0;

/// Every attributable field, including the residual. A plant that lands in `unattributed_ms`
/// failed to be attributed, and that is exactly the defect this file exists to catch.
const FIELDS: [&str; 8] = [
    "load_ms",
    "h2d_ms",
    "validate_ms",
    "validate_ref_ms",
    "validate_probe_ms",
    "prefill_ms",
    "decode_ms",
    "unattributed_ms",
];

fn field(s: &StageTimings, name: &str) -> Option<f64> {
    match name {
        "load_ms" => s.load_ms,
        "h2d_ms" => s.h2d_ms,
        "validate_ms" => s.validate_ms,
        "validate_ref_ms" => s.validate_ref_ms,
        "validate_probe_ms" => s.validate_probe_ms,
        "prefill_ms" => s.prefill_ms,
        "decode_ms" => s.decode_ms,
        "unattributed_ms" => s.unattributed_ms,
        other => panic!("unknown field {other}"),
    }
}

/// How many clean runs a GPU leg measures its noise from (after a warm-up it discards).
const CLEAN_RUNS: usize = 4;
/// A field's tolerance on a GPU leg is [`TOL_MS`] plus this many times the spread it showed across
/// the clean runs. The spread of a handful of samples underestimates the true range, hence > 1.
const NOISE_MARGIN: f64 = 2.0;
/// The plant is at least this many times the widest tolerance, so a plant in the WRONG field
/// still moves that field far outside its band.
const PLANT_OVER_TOL: f64 = 3.0;

/// What one plant is judged against: the size of the plant, and how far each field may move on
/// noise alone.
///
/// #4105: a fixed [`TOL_MS`] cannot hold on the GPU legs. The F2 guard's CPU reference forward
/// runs on every qwen35 run (the receipt is forced fresh), and on a loaded host `validate_ms`
/// varied by ~2 s and `load_ms` by ~1 s between CLEAN runs. That is above both the tolerance and
/// the 1000 ms plant, so every plant read as misattribution. The band is therefore MEASURED:
/// each field's spread across clean runs widens its tolerance, and the plant grows with the
/// widest tolerance so attribution stays unambiguous.
#[derive(Debug, Clone)]
struct Band {
    plant_ms: u64,
    tol_ms: [f64; FIELDS.len()],
}

impl Band {
    /// The CPU legs and the checker's own tests: the stated constants, with no measurement.
    fn fixed() -> Self {
        Band {
            plant_ms: PLANT_MS,
            tol_ms: [TOL_MS; FIELDS.len()],
        }
    }

    /// Sized from clean runs of the SAME model on the SAME host.
    fn measured(clean: &[StageTimings]) -> Self {
        let mut tol_ms = [TOL_MS; FIELDS.len()];
        for (tol, f) in tol_ms.iter_mut().zip(FIELDS) {
            let seen: Vec<f64> = clean.iter().filter_map(|s| field(s, f)).collect();
            let lo = seen.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = seen.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if hi >= lo {
                *tol = TOL_MS + NOISE_MARGIN * (hi - lo);
            }
        }
        let widest = tol_ms.iter().copied().fold(TOL_MS, f64::max);
        // Round up to whole half-seconds so the plant spec in the log reads cleanly.
        let wanted = (PLANT_OVER_TOL * widest / 500.0).ceil() as u64 * 500;
        Band {
            plant_ms: PLANT_MS.max(wanted),
            tol_ms,
        }
    }

    fn tol(&self, name: &str) -> f64 {
        let i = FIELDS
            .iter()
            .position(|f| *f == name)
            .unwrap_or_else(|| panic!("unknown field {name}"));
        self.tol_ms[i]
    }

    /// The wall clock is not a field; it may move by as much as the noisiest field.
    fn widest(&self) -> f64 {
        self.tol_ms.iter().copied().fold(TOL_MS, f64::max)
    }
}

/// The field a stage's plant must move, plus the parent it is nested inside (a plant in a
/// validate HALF is inside `validate_ms` too, and moving it is not misattribution).
fn target(stage: &str) -> (&'static str, Option<&'static str>) {
    match stage {
        "load" => ("load_ms", None),
        "h2d" => ("h2d_ms", None),
        "validate" => ("validate_ms", None),
        "validate_ref" => ("validate_ref_ms", Some("validate_ms")),
        "validate_probe" => ("validate_probe_ms", Some("validate_ms")),
        "prefill" => ("prefill_ms", None),
        "decode" => ("decode_ms", None),
        other => panic!("unknown stage {other}"),
    }
}

/// `Ok` when the plant for `stage` moved its own field by at least the band's plant, and no other
/// field by more than that field's tolerance. `Err` names the stage and each field that broke the rule.
fn plant_lands_in_its_stage(
    stage: &str,
    band: &Band,
    base: &StageTimings,
    planted: &StageTimings,
) -> Result<(), String> {
    let plant = band.plant_ms;
    let (own, parent) = target(stage);
    let mut why = Vec::new();
    match (field(base, own), field(planted, own)) {
        (_, None) => why.push(format!("{own} was not measured in the planted run")),
        (b, Some(p)) => {
            // Two readings of "moved by the plant". The field must CONTAIN the whole sleep, which
            // is exact: the sleep is inside the window or it is not. And against the baseline it must
            // move by the plant, less the same run-to-run jitter every other field is allowed
            // (measured: 249.8 ms on a then-250 ms plant, the unslept part of load varying by 0.2 ms).
            if p < plant as f64 {
                why.push(format!(
                    "{own} is {p:.1} ms, less than the {plant} ms plant it must contain"
                ));
            }
            let moved = p - b.unwrap_or(0.0);
            if moved < plant as f64 - band.tol(own) {
                why.push(format!(
                    "{own} moved {moved:.1} ms against the baseline, less than the {plant} ms plant"
                ));
            }
        },
    }
    for f in FIELDS {
        if f == own || Some(f) == parent {
            continue;
        }
        match (field(base, f), field(planted, f)) {
            (Some(b), Some(p)) if (p - b).abs() > band.tol(f) => {
                why.push(format!(
                    "{f} moved {:.1} ms (tolerance {:.0} ms)",
                    p - b,
                    band.tol(f)
                ));
            },
            (None, Some(p)) => {
                why.push(format!("{f} appeared ({p:.1} ms) only in the planted run"));
            },
            (Some(_), None) => why.push(format!("{f} vanished in the planted run")),
            _ => {},
        }
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(format!("plant `{stage}:{plant}`: {}", why.join("; ")))
    }
}

/// `Ok` when a plant for a stage this PATH has no site for lands nowhere: no field moves past
/// its tolerance, and the wall clock does not grow by the plant (it did not sleep somewhere unmeasured).
fn plant_lands_nowhere(
    stage: &str,
    band: &Band,
    base: &StageTimings,
    planted: &StageTimings,
) -> Result<(), String> {
    let plant = band.plant_ms;
    let mut why = Vec::new();
    for f in FIELDS {
        match (field(base, f), field(planted, f)) {
            (Some(b), Some(p)) if (p - b).abs() > band.tol(f) => {
                why.push(format!("{f} moved {:.1} ms", p - b));
            },
            (None, Some(_)) => why.push(format!("{f} appeared")),
            (Some(_), None) => why.push(format!("{f} vanished")),
            _ => {},
        }
    }
    let grew = planted.wall_ms.unwrap_or(0.0) - base.wall_ms.unwrap_or(0.0);
    if grew > band.widest() {
        why.push(format!(
            "wall grew {grew:.1} ms: the plant slept somewhere this path does not measure"
        ));
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "plant `{stage}:{plant}` on a path with no {stage} site: {}",
            why.join("; ")
        ))
    }
}

fn run(config: &InferenceConfig, plant: Option<(&str, u64)>) -> StageTimings {
    let spec = plant.map(|(s, ms)| format!("{s}:{ms}"));
    let stages = with_delay(spec.as_deref(), || {
        run_inference(config).expect("the fixture must run").stages
    });
    assert!(
        stages.is_closed(),
        "the books were never closed: {stages:?}"
    );
    stages
}

/// The generated model every CPU case runs: a real GGUF file through the real loader.
fn cpu_fixture() -> (tempfile::NamedTempFile, InferenceConfig) {
    use std::io::Write;
    let bytes = crate::gguf::test_factory::build_minimal_llama_gguf(32, 256, 512, 4, 4);
    let mut file = tempfile::Builder::new()
        .suffix(".gguf")
        .tempfile()
        .expect("tempfile");
    file.write_all(&bytes).expect("write fixture");
    file.flush().expect("flush fixture");
    let mut config = InferenceConfig::new(file.path())
        .with_input_tokens(vec![1, 2, 3])
        .with_max_tokens(2)
        .with_temperature(0.0);
    config.no_gpu = true;
    (file, config)
}

/// Control: two clean runs agree within the tolerance, and the CPU path measures `load` and only
/// `load`. Without this, "the plant moved no other field" could be true of a report that moves
/// on its own.
#[test]
fn cpu_a_clean_run_is_stable_and_measures_load_only() {
    let (_f, config) = cpu_fixture();
    let _warm = run(&config, None);
    let a = run(&config, None);
    let b = run(&config, None);
    assert_eq!(a.measured(), vec!["load_ms"], "{a:?}");
    assert_eq!(a.backend, "cpu", "{a:?}");
    for f in FIELDS {
        if let (Some(x), Some(y)) = (field(&a, f), field(&b, f)) {
            assert!(
                (x - y).abs() <= TOL_MS,
                "{f}: {x:.1} vs {y:.1} on two clean runs"
            );
        }
    }
}

/// The plant that slipped through once: through `run_gguf_inference`, `load` moves `load_ms`
/// and nothing else, and `unattributed_ms` does NOT absorb it.
#[test]
fn cpu_a_load_plant_moves_load_and_only_load() {
    let (_f, config) = cpu_fixture();
    let _warm = run(&config, None);
    let base = run(&config, None);
    let band = Band::fixed();
    let planted = run(&config, Some(("load", band.plant_ms)));
    plant_lands_in_its_stage("load", &band, &base, &planted).unwrap_or_else(|e| panic!("{e}"));
}

/// Every stage the CPU path has no site for: its plant must land NOWHERE. A plant that moved
/// `unattributed_ms` or the wall clock here would be a sleep in a place nobody measures.
#[test]
fn cpu_plants_for_stages_the_cpu_path_cannot_measure_land_nowhere() {
    let (_f, config) = cpu_fixture();
    let _warm = run(&config, None);
    let base = run(&config, None);
    let band = Band::fixed();
    let mut broke = Vec::new();
    for stage in [
        "h2d",
        "validate",
        "validate_ref",
        "validate_probe",
        "prefill",
        "decode",
    ] {
        let planted = run(&config, Some((stage, band.plant_ms)));
        if let Err(e) = plant_lands_nowhere(stage, &band, &base, &planted) {
            broke.push(e);
        }
    }
    assert!(broke.is_empty(), "{}", broke.join("\n"));
}

/// The checker is not vacuous: a report where the plant landed in a NEIGHBOUR (or in the
/// residual, the historical `load` bug) is rejected, naming the stage. The code-level mutant is
/// in the PMAT-4105 receipt; this pins the judgement it relies on.
#[test]
fn the_checker_rejects_a_plant_attributed_to_a_neighbour_or_to_nobody() {
    let base = StageTimings {
        load_ms: Some(5.0),
        h2d_ms: Some(10.0),
        validate_ms: Some(20.0),
        prefill_ms: Some(3.0),
        decode_ms: Some(7.0),
        unattributed_ms: Some(1.0),
        wall_ms: Some(46.0),
        ..StageTimings::default()
    };
    let band = Band::fixed();
    let plant = band.plant_ms as f64;
    // correct: h2d moved by the plant
    let good = StageTimings {
        h2d_ms: Some(10.0 + plant),
        ..base.clone()
    };
    assert!(plant_lands_in_its_stage("h2d", &band, &base, &good).is_ok());
    // misattributed to the neighbour
    let neighbour = StageTimings {
        validate_ms: Some(20.0 + plant),
        ..base.clone()
    };
    let e = plant_lands_in_its_stage("h2d", &band, &base, &neighbour)
        .expect_err("neighbour must be RED");
    assert!(e.contains("h2d") && e.contains("validate_ms moved"), "{e}");
    // attributed to nobody: the residual took it (the load bug)
    let nobody = StageTimings {
        unattributed_ms: Some(1.0 + plant),
        ..base.clone()
    };
    let e =
        plant_lands_in_its_stage("load", &band, &base, &nobody).expect_err("residual must be RED");
    assert!(
        e.contains("load_ms is 5.0 ms") && e.contains("unattributed_ms moved"),
        "{e}"
    );
    // a validate half may move its parent, and only its parent
    let half = StageTimings {
        validate_ref_ms: Some(plant),
        validate_ms: Some(20.0 + plant),
        ..base.clone()
    };
    assert!(plant_lands_in_its_stage("validate_ref", &band, &base, &half).is_ok());
    // a sleep on a path with no site that still grew the wall clock is RED
    let hidden = StageTimings {
        unattributed_ms: Some(1.0 + plant),
        wall_ms: Some(46.0 + plant),
        ..base.clone()
    };
    assert!(plant_lands_nowhere("prefill", &band, &base, &hidden).is_err());
    assert!(plant_lands_nowhere("prefill", &band, &base, &base.clone()).is_ok());
}

/// #4105: a measured band follows the noise it was given. A field that jittered by 2 s on
/// clean runs gets a tolerance above 2 s, a quiet field keeps [`TOL_MS`], and the plant grows so
/// that a plant in the WRONG field is still rejected.
#[test]
fn a_measured_band_widens_only_the_noisy_fields_and_still_rejects_a_neighbour() {
    let clean = |validate: f64, load: f64| StageTimings {
        load_ms: Some(load),
        h2d_ms: Some(10.0),
        validate_ms: Some(validate),
        prefill_ms: Some(3.0),
        decode_ms: Some(7.0),
        unattributed_ms: Some(1.0),
        wall_ms: Some(validate + load + 21.0),
        ..StageTimings::default()
    };
    let runs = [
        clean(15_000.0, 900.0),
        clean(17_000.0, 1_900.0),
        clean(16_000.0, 1_200.0),
    ];
    let band = Band::measured(&runs);
    assert_eq!(
        band.tol("h2d_ms"),
        TOL_MS,
        "a quiet field keeps the fixed tolerance"
    );
    assert!(band.tol("validate_ms") > 2_000.0, "{band:?}");
    assert!(band.tol("load_ms") > 1_000.0, "{band:?}");
    let plant = band.plant_ms as f64;
    assert!(
        plant >= PLANT_OVER_TOL * band.tol("validate_ms"),
        "{band:?}"
    );

    let base = &runs[2];
    // noise alone inside the band, plus a correct h2d plant: GREEN
    let noisy_good = StageTimings {
        h2d_ms: Some(10.0 + plant),
        validate_ms: Some(17_500.0),
        load_ms: Some(500.0),
        ..base.clone()
    };
    plant_lands_in_its_stage("h2d", &band, base, &noisy_good).unwrap_or_else(|e| panic!("{e}"));
    // the same plant landing in validate instead: RED, even on the noisiest field
    let neighbour = StageTimings {
        validate_ms: Some(16_000.0 + plant),
        ..base.clone()
    };
    let e = plant_lands_in_its_stage("h2d", &band, base, &neighbour).expect_err("neighbour");
    assert!(e.contains("validate_ms moved"), "{e}");
}

/// The GPU legs share one driver: every stage `stages` names, each through the real CUDA path,
/// against a clean baseline. The run must STAY on CUDA; a fallback is a failure, never a pass.
#[cfg(feature = "cuda")]
fn cuda_leg(env_var: &str, want_backend: &str, stages: &[&str]) {
    let path = std::env::var(env_var)
        .unwrap_or_else(|_| panic!("set {env_var} to a GGUF this GPU runner can serve"));
    let config = InferenceConfig::new(path)
        .with_prompt("Hi")
        .with_max_tokens(4)
        .with_temperature(0.0);
    let _warm = run(&config, None);
    let clean: Vec<StageTimings> = (0..CLEAN_RUNS).map(|_| run(&config, None)).collect();
    for c in &clean {
        assert_eq!(
            c.backend, want_backend,
            "a clean run did not stay on {want_backend}: {c:?}"
        );
    }
    let band = Band::measured(&clean);
    eprintln!("[4105] {want_backend} band from {CLEAN_RUNS} clean runs: {band:?}");
    let base = &clean[CLEAN_RUNS - 1];
    let mut broke = Vec::new();
    for stage in stages {
        let planted = run(&config, Some((stage, band.plant_ms)));
        if planted.backend != want_backend {
            broke.push(format!(
                "plant `{stage}`: the run fell back to {}",
                planted.backend
            ));
        } else if let Err(e) = plant_lands_in_its_stage(stage, &band, base, &planted) {
            broke.push(e);
        }
    }
    assert!(broke.is_empty(), "{}", broke.join("\n"));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU runner: set APR_STAGE_CUDA_GGUF to a dense GGUF (e.g. qwen2.5-coder-0.5b q4_k_m)"]
fn cuda_dense_each_plant_moves_its_own_stage() {
    // This leg measures ATTRIBUTION, so the run must stay on CUDA. On sm_89 the dense path's FP8
    // prefill fails the F2 guard on small Qwen2.5 models (#3602's 2x2, #3483) and the run falls back
    // to CPU, where h2d/prefill/decode have no site. FP16 prefill is the same stage wiring. A caller
    // who sets FP8_PREFILL keeps their value.
    if std::env::var_os("FP8_PREFILL").is_none() {
        std::env::set_var("FP8_PREFILL", "0");
    }
    cuda_leg(
        "APR_STAGE_CUDA_GGUF",
        "cuda",
        &["load", "h2d", "validate", "prefill", "decode"],
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU runner: set APR_STAGE_CUDA_QWEN35_GGUF to a Qwen3.5 GGUF"]
fn cuda_qwen35_each_plant_moves_its_own_stage() {
    // A matching F2 receipt skips the guard, and then the validate halves are never measured.
    // Force a fresh guard, into a private dir so no user receipt is read or written.
    let dir = tempfile::tempdir().expect("receipt dir");
    std::env::set_var("APR_F2_RECEIPT_DIR", dir.path());
    std::env::set_var("APR_F2_REVALIDATE", "1");
    cuda_leg(
        "APR_STAGE_CUDA_QWEN35_GGUF",
        "cuda-qwen35",
        &[
            "load",
            "h2d",
            "validate",
            "validate_ref",
            "validate_probe",
            "prefill",
            "decode",
        ],
    );
}
