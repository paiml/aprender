// #3821: GPU-correct(model_sha256, host, apr_version) is ONE subject, not a
// per-surface opinion.
//
// Two surfaces decide it, in the same binary, on the same host, for the same
// model, and until now neither recorded the other's answer:
//
//   * `apr run --gpu` — the F2 parity guard. REFUSES a diverging GPU
//     (`BackendUnavailable`, exit 14).
//   * `apr qa` `golden_output` — generates on the GPU and scores the text.
//     It calls `generate_gpu_resident` directly, so it never reaches the
//     guard (#3804: grepping a failing run's stderr for the guard's own lines
//     returns zero occurrences).
//
// On qwen2.5-coder-0.5b-instruct-q4_k_m at `a9502d992` the guard refused at
// cosine 0.4153 while the gate shipped the symptom as ``` ``` ``` repetition.
// Both happened to say "bad", so nothing noticed they were never compared.
// The dangerous ordering is the other one: the guard refusing while the gate
// PASSES, which `APR_F2_PROBE_PATH=serial` produces on demand — serial parity
// is cosine >= 0.9978 at all 16 positions, so the guard accepts, and
// generation still runs the batched prefill that emits the fragment.
//
// REPORT_ONLY for 0.69.1 (#3821's ruling): this reconciles and NAMES, it does
// not yet block. Flipping it to blocking is a one-line change at the caller,
// deliberately deferred until it has been green on a real release commit —
// a gate never green on its own target is not a gate (#3731).

/// One surface's verdict about the subject, with the evidence it decided on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SurfaceVerdict {
    /// The surface that decided, named as the user would invoke it.
    pub surface: &'static str,
    pub verdict: provable_contracts::ontology::verdict::Verdict,
    /// What it measured. Printed verbatim in a contradiction.
    pub detail: String,
}

impl SurfaceVerdict {
    pub fn new(
        surface: &'static str,
        verdict: provable_contracts::ontology::verdict::Verdict,
        detail: impl Into<String>,
    ) -> Self {
        Self { surface, verdict, detail: detail.into() }
    }

    /// A surface that was never evaluated on this subject. Absence is
    /// `Unknown(NotRun)`, never agreement (#3821 done_when 3).
    pub fn not_run(surface: &'static str) -> Self {
        use provable_contracts::ontology::verdict::{Reason, Verdict};
        Self::new(surface, Verdict::Unknown(Reason::NotRun), "not evaluated on this subject")
    }
}

/// The reconciled answer for one subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubjectReport {
    /// The ONT-6 meet over every surface. `Fail < Unknown(_) < Pass`, so this
    /// is `min` and a single Fail carries.
    pub verdict: provable_contracts::ontology::verdict::Verdict,
    /// Set iff at least one surface said Pass AND at least one said Fail.
    /// Names every surface and its value — the point of the row is that a
    /// disagreement is reported rather than silently minned away.
    pub contradiction: Option<String>,
    /// Surfaces that never evaluated this subject, named.
    pub not_run: Vec<&'static str>,
}

/// Reconcile every surface's verdict about ONE subject.
///
/// Pure and GPU-free, so the contradiction logic is testable on a host with no
/// CUDA — which is the only way this row could have been tested at all.
pub(crate) fn reconcile_gpu_correct(subject: &str, surfaces: &[SurfaceVerdict]) -> SubjectReport {
    use provable_contracts::ontology::verdict::{Reason, Verdict};

    if surfaces.is_empty() {
        return SubjectReport {
            verdict: Verdict::Unknown(Reason::NotRun),
            contradiction: None,
            not_run: Vec::new(),
        };
    }

    let verdict = surfaces.iter().map(|s| s.verdict).fold(Verdict::Pass, Verdict::meet);

    let passed: Vec<&SurfaceVerdict> =
        surfaces.iter().filter(|s| s.verdict == Verdict::Pass).collect();
    let failed: Vec<&SurfaceVerdict> =
        surfaces.iter().filter(|s| s.verdict == Verdict::Fail).collect();

    let contradiction = (!passed.is_empty() && !failed.is_empty()).then(|| {
        let mut lines = vec![format!(
            "CONTRADICTION on {subject}: GPU-correct is one subject and {} of {} surfaces \
             disagree about it.",
            passed.len() + failed.len(),
            surfaces.len()
        )];
        for s in surfaces {
            lines.push(format!("  - {}: {:?} — {}", s.surface, s.verdict, s.detail));
        }
        lines.push(
            "  Both cannot be right about the same model on the same host in the same binary."
                .to_string(),
        );
        lines.join("\n")
    });

    let not_run = surfaces
        .iter()
        .filter(|s| s.verdict == Verdict::Unknown(Reason::NotRun))
        .map(|s| s.surface)
        .collect();

    SubjectReport { verdict, contradiction, not_run }
}

/// The subject tuple, rendered. Keyed on the model's CONTENT, not its path —
/// two files with the same name are not the same subject, and the same bytes
/// under two names are.
pub(crate) fn gpu_correct_subject(model_sha256: &str, host: &str, apr_version: &str) -> String {
    let short = model_sha256.get(..12).unwrap_or(model_sha256);
    format!("GPU-correct(model {short}, host {host}, apr {apr_version})")
}

// NOTE ON NAMING: `Makefile`'s coverage target passes `--skip gpu_`, and
// libtest matches that substring against the WHOLE test path. Every name below
// therefore avoids the sequence `gpu_` — otherwise these rows would compile,
// pass locally, and be invisible to the coverage gate. The module is named
// `subject_verdict_tests` for the same reason (#3839).
// The golden gate's own leg, expressed in the same lattice as the guard so the
// two can be compared at all. #3821: until this existed the two surfaces spoke
// different languages — one returned bool, the other a five-variant enum — and
// "compare them" had no meaning.
impl crate::commands::qa::GpuGoldenLeg {
    /// This leg's verdict ABOUT THE SUBJECT, which is not the same as whether
    /// the gate passed. A leg that never ran, errored, or ran out of budget
    /// says nothing about `GPU-correct`; only `Passed` and `WrongAnswer` do.
    pub(crate) fn subject_verdict(&self) -> provable_contracts::ontology::verdict::Verdict {
        use provable_contracts::ontology::verdict::{Reason, Verdict};
        match self {
            Self::Passed => Verdict::Pass,
            Self::WrongAnswer(_) => Verdict::Fail,
            // Attempted, but produced nothing judgeable. Not evidence either way.
            Self::Errored(_) | Self::Unclosed { .. } | Self::NotRun(_) => {
                Verdict::Unknown(Reason::NotRun)
            }
        }
    }

    /// What this leg measured, for a contradiction message.
    pub(crate) fn subject_detail(&self) -> String {
        match self {
            Self::Passed => "golden patterns matched on the device".to_string(),
            Self::WrongAnswer(why) => why.clone(),
            Self::Errored(e) => format!("errored: {e}"),
            Self::Unclosed { budget, generated_chars } => {
                format!("still reasoning at the {budget}-token budget ({generated_chars} chars)")
            }
            Self::NotRun(why) => (*why).to_string(),
        }
    }
}

/// The surface names, as a user would invoke each one.
#[cfg(all(feature = "inference", feature = "cuda"))]
const F2_SURFACE: &str = "apr run --gpu (F2 parity guard)";
#[cfg(all(feature = "inference", feature = "cuda"))]
const GOLDEN_SURFACE: &str = "apr qa golden_output (GPU leg)";

/// The F2 guard's outcome as a verdict ABOUT THE SUBJECT, in the lattice the
/// golden leg speaks. `NotMeasured` is `Unknown(NotRun)`: `apr run` proceeds on
/// it, flagged UNVALIDATED (#3973), but it is not evidence that the GPU is correct
/// — absence is not agreement.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn f2_surface_verdict(f2: &realizar::infer::F2Outcome) -> SurfaceVerdict {
    use provable_contracts::ontology::verdict::{Reason, Verdict};
    use realizar::infer::F2Outcome;
    match f2 {
        F2Outcome::Validated { min_cosine } => SurfaceVerdict::new(
            F2_SURFACE,
            Verdict::Pass,
            format!("accepted, min cosine {min_cosine:.4} over the real positions"),
        ),
        F2Outcome::Mismatch => SurfaceVerdict::new(
            F2_SURFACE,
            Verdict::Fail,
            "refused: GPU logits diverge from the CPU reference (or the GPU forward failed)",
        ),
        F2Outcome::NotMeasured { reason } => SurfaceVerdict::new(
            F2_SURFACE,
            Verdict::Unknown(Reason::NotRun),
            format!("not measured: {reason}"),
        ),
    }
}

/// Reconcile the guard and the golden leg on ONE subject and print a
/// contradiction to stderr. REPORT_ONLY (see the header): the gate's own result
/// is unchanged. The model is hashed only when there is a contradiction to name,
/// so an agreeing run pays nothing for the digest.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn report_gpu_correct_subject(
    model_bytes: &[u8],
    f2: &realizar::infer::F2Outcome,
    leg: &crate::commands::qa::GpuGoldenLeg,
) {
    let surfaces = [
        f2_surface_verdict(f2),
        SurfaceVerdict::new(GOLDEN_SURFACE, leg.subject_verdict(), leg.subject_detail()),
    ];
    if reconcile_gpu_correct("", &surfaces).contradiction.is_none() {
        return;
    }
    let digest = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(model_bytes))
    };
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|h| h.trim().to_string())
        .unwrap_or_else(|_| "unknown-host".to_string());
    let subject = gpu_correct_subject(&digest, &host, env!("CARGO_PKG_VERSION"));
    if let Some(contradiction) = reconcile_gpu_correct(&subject, &surfaces).contradiction {
        eprintln!("[#3821 REPORT_ONLY] {contradiction}");
    }
}

#[cfg(test)]
mod subject_verdict_tests {
    use super::{gpu_correct_subject, reconcile_gpu_correct, SurfaceVerdict};
    use provable_contracts::ontology::verdict::{Reason, Verdict};

    const RUN: &str = "apr run --gpu (F2 parity guard)";
    const QA: &str = "apr qa golden_output";

    fn subject() -> String {
        gpu_correct_subject("d98cdcbd03e17ce47681435b5150e34c", "lambda", "0.69.1")
    }

    /// The row's whole reason for existing. `meet` already yields Fail, so a
    /// silent `min` would look like an ordinary failure — indistinguishable
    /// from both surfaces agreeing the model is bad. A contradiction is a
    /// different fact and must READ differently.
    #[test]
    fn a_pass_and_a_fail_on_one_subject_are_named_as_a_contradiction() {
        let r = reconcile_gpu_correct(
            &subject(),
            &[
                SurfaceVerdict::new(RUN, Verdict::Fail, "diverges at position 1, cosine 0.4153"),
                SurfaceVerdict::new(QA, Verdict::Pass, "3 golden test cases passed"),
            ],
        );
        assert_eq!(r.verdict, Verdict::Fail, "the meet still carries the Fail");
        let msg = r.contradiction.expect("a Pass beside a Fail is a contradiction");
        assert!(msg.contains("CONTRADICTION"));
        assert!(msg.contains(RUN) && msg.contains(QA), "both surfaces must be named: {msg}");
        assert!(
            msg.contains("0.4153") && msg.contains("3 golden test cases passed"),
            "both VALUES must be named, not just the surfaces: {msg}"
        );
    }

    /// Agreement is not a contradiction, in either direction. A row that fired
    /// on agreement would be noise and would be turned off.
    #[test]
    fn surfaces_that_agree_are_not_reported_as_contradicting() {
        for v in [Verdict::Pass, Verdict::Fail] {
            let r = reconcile_gpu_correct(
                &subject(),
                &[SurfaceVerdict::new(RUN, v, "x"), SurfaceVerdict::new(QA, v, "y")],
            );
            assert_eq!(r.verdict, v);
            assert!(r.contradiction.is_none(), "agreement on {v:?} is not a contradiction");
        }
    }

    /// #3821 done_when 3. One surface passing while the other never ran must
    /// NOT read as a pass — that is how a gate nobody executed becomes
    /// "green". The lattice gives Unknown, and the unrun surface is named.
    #[test]
    fn absence_is_not_agreement() {
        let r = reconcile_gpu_correct(
            &subject(),
            &[SurfaceVerdict::new(RUN, Verdict::Pass, "admitted"), SurfaceVerdict::not_run(QA)],
        );
        assert_eq!(
            r.verdict,
            Verdict::Unknown(Reason::NotRun),
            "a Pass beside an unrun surface is Unknown, never Pass"
        );
        assert!(!r.verdict.arm(), "Unknown must not arm");
        assert_eq!(r.not_run, vec![QA], "the unrun surface must be named");
        assert!(r.contradiction.is_none(), "not running is not disagreeing");
    }

    /// The meet is the ONT-6 lattice min over all three levels, and a Fail
    /// beside an Unknown is still a Fail — the strongest claim wins.
    #[test]
    fn the_verdict_is_the_lattice_meet() {
        let cases: &[(Verdict, Verdict, Verdict)] = &[
            (Verdict::Pass, Verdict::Pass, Verdict::Pass),
            (Verdict::Pass, Verdict::Fail, Verdict::Fail),
            (Verdict::Fail, Verdict::Unknown(Reason::NotRun), Verdict::Fail),
            (Verdict::Pass, Verdict::Unknown(Reason::NotRun), Verdict::Unknown(Reason::NotRun)),
            (Verdict::Fail, Verdict::Fail, Verdict::Fail),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(a, b, want)| {
                let got = reconcile_gpu_correct(
                    &subject(),
                    &[SurfaceVerdict::new(RUN, *a, ""), SurfaceVerdict::new(QA, *b, "")],
                )
                .verdict;
                (got != *want).then(|| format!("\n  - {a:?} meet {b:?}: expected {want:?}, got {got:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "the meet is not the lattice min:{}", wrong.join(""));
    }

    /// Nothing measured is Unknown, never Pass. An empty surface set is how a
    /// subject nobody looked at would otherwise arrive as green.
    #[test]
    fn a_subject_no_surface_evaluated_is_unknown_not_pass() {
        let r = reconcile_gpu_correct(&subject(), &[]);
        assert_eq!(r.verdict, Verdict::Unknown(Reason::NotRun));
        assert!(!r.verdict.arm());
        assert!(r.contradiction.is_none());
    }

    /// The subject is keyed on the model's CONTENT. Two files with the same
    /// name are not one subject; the same bytes under two names are.
    #[test]
    fn the_subject_is_keyed_on_content_and_host_and_version() {
        let a = gpu_correct_subject("aaaaaaaaaaaabbbb", "lambda", "0.69.1");
        assert_eq!(a, gpu_correct_subject("aaaaaaaaaaaabbbb", "lambda", "0.69.1"));
        assert_ne!(a, gpu_correct_subject("ccccccccccccbbbb", "lambda", "0.69.1"));
        assert_ne!(a, gpu_correct_subject("aaaaaaaaaaaabbbb", "gx10", "0.69.1"));
        assert_ne!(a, gpu_correct_subject("aaaaaaaaaaaabbbb", "lambda", "0.69.0"));
        assert!(a.contains("aaaaaaaaaaaa"), "the sha must appear: {a}");
        assert!(!a.contains("bbbb"), "only the first 12 hex are printed: {a}");
    }

    /// Only `Passed` and `WrongAnswer` are evidence about the subject. The
    /// other three mean the gate produced nothing judgeable, and a gate that
    /// produced nothing must not read as agreement with whatever the guard
    /// said — that is how an unrun surface becomes a green one.
    #[test]
    fn only_a_judged_answer_is_evidence_about_the_subject() {
        use crate::commands::qa::GpuGoldenLeg;
        let cases: Vec<(GpuGoldenLeg, Verdict)> = vec![
            (GpuGoldenLeg::Passed, Verdict::Pass),
            (GpuGoldenLeg::WrongAnswer("gibberish".into()), Verdict::Fail),
            (GpuGoldenLeg::Errored("CUDA init".into()), Verdict::Unknown(Reason::NotRun)),
            (
                GpuGoldenLeg::Unclosed { budget: 16, generated_chars: 40 },
                Verdict::Unknown(Reason::NotRun),
            ),
            (GpuGoldenLeg::NotRun("no device"), Verdict::Unknown(Reason::NotRun)),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(leg, want)| {
                let got = leg.subject_verdict();
                (got != *want).then(|| format!("\n  - {leg:?}: expected {want:?}, got {got:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "leg-to-verdict mapping drifted:{}", wrong.join(""));
    }

    /// The detail is what a contradiction message prints, so a leg that says
    /// nothing useful makes the contradiction unreadable.
    #[test]
    fn every_leg_states_what_it_measured() {
        use crate::commands::qa::GpuGoldenLeg;
        for leg in [
            GpuGoldenLeg::Passed,
            GpuGoldenLeg::WrongAnswer("fragment repeats".into()),
            GpuGoldenLeg::Errored("init".into()),
            GpuGoldenLeg::Unclosed { budget: 16, generated_chars: 40 },
            GpuGoldenLeg::NotRun("no device"),
        ] {
            let d = leg.subject_detail();
            assert!(!d.trim().is_empty(), "{leg:?} produced an empty detail");
        }
        assert!(GpuGoldenLeg::WrongAnswer("fragment repeats".into())
            .subject_detail()
            .contains("fragment repeats"), "the reason must survive into the detail");
    }

    /// A short or empty sha must not panic on the 12-char slice.
    #[test]
    fn a_short_digest_does_not_panic() {
        assert!(gpu_correct_subject("abc", "h", "v").contains("abc"));
        assert!(gpu_correct_subject("", "h", "v").contains("host h"));
    }
}

// The guard's side of the comparison. `F2Outcome` exists only under cuda, so these
// rows compile there; names avoid `gpu_` for the coverage `--skip` (see above).
#[cfg(all(test, feature = "inference", feature = "cuda"))]
mod f2_surface_tests {
    use super::{f2_surface_verdict, reconcile_gpu_correct, SurfaceVerdict, GOLDEN_SURFACE};
    use crate::commands::qa::GpuGoldenLeg;
    use provable_contracts::ontology::verdict::{Reason, Verdict};
    use realizar::infer::F2Outcome;

    #[test]
    fn each_f2_outcome_maps_to_one_lattice_value() {
        let validated = f2_surface_verdict(&F2Outcome::Validated { min_cosine: 0.9978 });
        assert_eq!(validated.verdict, Verdict::Pass);
        assert!(validated.detail.contains("0.9978"), "{}", validated.detail);
        assert_eq!(f2_surface_verdict(&F2Outcome::Mismatch).verdict, Verdict::Fail);
        let unmeasured = f2_surface_verdict(&F2Outcome::NotMeasured { reason: "empty prompt".into() });
        // `apr run` proceeds on NotMeasured; that is not evidence of correctness.
        assert_eq!(unmeasured.verdict, Verdict::Unknown(Reason::NotRun));
        assert!(unmeasured.detail.contains("empty prompt"), "{}", unmeasured.detail);
    }

    /// The ordering #3821 exists for: the guard refuses, the gate passes.
    #[test]
    fn guard_refusing_while_the_golden_leg_passes_is_named() {
        let leg = GpuGoldenLeg::Passed;
        let surfaces = [
            f2_surface_verdict(&F2Outcome::Mismatch),
            SurfaceVerdict::new(GOLDEN_SURFACE, leg.subject_verdict(), leg.subject_detail()),
        ];
        let report = reconcile_gpu_correct("s", &surfaces);
        assert_eq!(report.verdict, Verdict::Fail);
        let c = report.contradiction.expect("a Fail and a Pass on one subject must be named");
        assert!(c.contains("F2 parity guard") && c.contains("golden_output"), "{c}");
    }

    #[test]
    fn an_unmeasured_guard_does_not_contradict_a_passing_leg() {
        let leg = GpuGoldenLeg::Passed;
        let surfaces = [
            f2_surface_verdict(&F2Outcome::NotMeasured { reason: "r".into() }),
            SurfaceVerdict::new(GOLDEN_SURFACE, leg.subject_verdict(), leg.subject_detail()),
        ];
        let report = reconcile_gpu_correct("s", &surfaces);
        assert!(report.contradiction.is_none());
        assert_eq!(report.verdict, Verdict::Unknown(Reason::NotRun));
    }
}
