//! PERF-062 / #2790 — what the run ASKED FOR and what it actually GOT.
//!
//! # The defect this closes
//!
//! `apr run --gpu` on a 7B Q4_K_M reaches the CUDA path, the F2 parity gate
//! rejects it, the process falls through wgpu to CPU, and **nothing that
//! survives the run says so**. Measured on this box against an `apr` built from
//! `a866988e4` with `--features cuda`:
//!
//! ```text
//! stdout: {"tok_s": 0.4, "tokens": 16, "latency_ms": 39505.73}
//! stderr: warning: GPU output diverges from CPU at position 59 (... cosine 0.9403) ...
//!         Backend: wgpu (Vulkan)
//!         warning: GPU (wgpu) path rejected ... cosine vs CPU = 0.722249 (< 0.99)
//! ```
//!
//! Two things are wrong with that, and only the first is "silence":
//!
//! 1. **stdout carries no compute class at all**, so a receipt built from the
//!    machine-readable surface cannot tell a CUDA run from a CPU one. Any
//!    tok/s taken from here is a CPU number wearing whatever label the operator
//!    typed.
//! 2. **The last backend statement on stderr is `Backend: wgpu (Vulkan)`, and
//!    it is FALSE** — wgpu was rejected two lines later and the run executed on
//!    CPU. The final `Backend: CPU (SIMD-accelerated)` line is `--verbose`-gated,
//!    so on a normal run the log's own last word is wrong. That is worse than
//!    silence: `scripts/parity_host_receipt.sh::apr_class_from_log()` greps for
//!    `wgpu` and returns **`wgpu`** for that log. The provenance producer emits
//!    a fabricated `compute_class` for a run that ran on CPU.
//!
//! # The rule
//!
//! I-2 (resolved-versus-requested) and I-17 (explicit wins). A run states, in
//! one place, on one line, what was asked for and what was taken. The reader
//! never infers a class from the presence of a banner, because a banner is
//! printed by an ATTEMPT and says nothing about whether the attempt survived.
//!
//! This is the same shape as `gpu_layers_requested` / `gpu_layers_resolved`:
//! a boolean cannot express a partial honouring, and neither can a banner.

use std::fmt;

/// The canonical machine-readable resolution line's prefix.
///
/// A log reader keys on this and nothing else. It is deliberately not a
/// `Backend:` line: every `Backend:` line in this crate is printed by an
/// ATTEMPT, so a reader that keys on them classifies a rejected attempt as a
/// success. There is exactly one of these per run, and it is printed after the
/// cascade has settled.
pub const RESOLUTION_LINE_PREFIX: &str = "apr-compute:";

/// The dispatch path actually taken, in `bench_receipt.py`'s vocabulary.
///
/// The token set is `COMPUTE_CLASSES` in `scripts/lib/bench_receipt.py`. It is
/// duplicated here rather than derived because a receipt schema and an
/// inference engine cannot share a type across a language boundary — but the
/// guard `scripts/check_compute_class_vocabulary.sh` fails when they diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeClass {
    /// SIMD CPU path.
    #[default]
    Cpu,
    /// CUDA / cudarc path.
    Cuda,
    /// wgpu (Vulkan/Metal/DX) compute path.
    Wgpu,
    /// Apple Metal path.
    Metal,
}

impl ComputeClass {
    /// The wire token, identical to the receipt's `provenance.compute_class`.
    #[must_use]
    pub fn wire(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }

    /// Is this an accelerator, as opposed to the CPU fallback?
    #[must_use]
    pub fn is_accelerator(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

impl fmt::Display for ComputeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire())
    }
}

/// What the operator asked for, as typed.
///
/// `Auto` and `Accelerator` are DIFFERENT requests and the difference is the
/// whole point: before this type existed, `apr run --gpu` and `apr run` reached
/// the engine as the same `no_gpu: false`, so no downstream code could tell a
/// silent downgrade from a default CPU run. A request that cannot be
/// distinguished from no request cannot be reported as unhonoured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeRequest {
    /// No compute flag given — the engine picks, and any choice honours it.
    #[default]
    Auto,
    /// `--gpu`: any accelerator, the engine picks which.
    Accelerator,
    /// `--no-gpu` or `--backend cpu`: CPU deliberately.
    Cpu,
    /// `--backend cuda|wgpu|metal`: this accelerator by name.
    Named(ComputeClass),
}

impl ComputeRequest {
    /// Build from the two flags `apr run` / `apr chat` / `apr serve` accept.
    ///
    /// `--no-gpu` wins over `--gpu` because that is what the existing CLI does
    /// (both set `no_gpu`), and a request to run on CPU is never upgraded.
    #[must_use]
    pub fn from_flags(gpu: bool, no_gpu: bool, backend: Option<&str>) -> Self {
        if no_gpu {
            return Self::Cpu;
        }
        match backend {
            Some("cpu") => Self::Cpu,
            Some("cuda") => Self::Named(ComputeClass::Cuda),
            Some("wgpu") => Self::Named(ComputeClass::Wgpu),
            Some("metal") => Self::Named(ComputeClass::Metal),
            _ if gpu => Self::Accelerator,
            _ => Self::Auto,
        }
    }

    /// The wire token for the receipt's `requested` half.
    #[must_use]
    pub fn wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Accelerator => "accelerator",
            Self::Cpu => "cpu",
            Self::Named(c) => c.wire(),
        }
    }

    /// Did the operator explicitly name a compute path?
    ///
    /// The resolution line is printed for every explicit request — including
    /// one that was honoured — because a producer needs the class on a GREEN
    /// run too. `Auto` stays quiet so a casual `apr run` gains no new noise.
    #[must_use]
    pub fn is_explicit(self) -> bool {
        !matches!(self, Self::Auto)
    }

    /// Does `taken` honour this request?
    #[must_use]
    pub fn honoured_by(self, taken: ComputeClass) -> bool {
        match self {
            Self::Auto => true,
            Self::Accelerator => taken.is_accelerator(),
            Self::Cpu => taken == ComputeClass::Cpu,
            Self::Named(want) => taken == want,
        }
    }
}

impl fmt::Display for ComputeRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire())
    }
}

/// One refused accelerator, with the reason the engine gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// Which path was refused.
    pub class: ComputeClass,
    /// Why, in the engine's own words. Never a summary — the reason a reader
    /// needs is the cosine and the position, not "parity failed".
    pub reason: String,
}

/// What a run asked for, what it took, and every accelerator refused on the way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputeResolution {
    /// As typed by the operator.
    pub requested: ComputeRequest,
    /// The path the tokens were actually produced on.
    pub resolved: ComputeClass,
    /// Every accelerator that was tried and declined, in the order tried.
    pub refusals: Vec<Refusal>,
}

impl ComputeResolution {
    /// A resolution that has not settled yet: the request, and CPU until an
    /// accelerator proves otherwise. Fail-closed — an engine that forgets to
    /// record its success reports the slow path, never a fast one it did not
    /// take.
    #[must_use]
    pub fn pending(requested: ComputeRequest) -> Self {
        Self {
            requested,
            resolved: ComputeClass::Cpu,
            refusals: Vec::new(),
        }
    }

    /// A resolution that has already settled, with no refusals collected.
    ///
    /// For the cascades that know WHICH path produced the tokens but do not
    /// collect the reasons the others were declined (the APR and SafeTensors
    /// paths). The class is still the path taken; the empty `refusals` list is
    /// reported as "not recorded" rather than as "none", because those are
    /// different facts and only one of them is known here.
    #[must_use]
    pub fn settled(requested: ComputeRequest, resolved: ComputeClass) -> Self {
        Self {
            requested,
            resolved,
            refusals: Vec::new(),
        }
    }

    /// Record an accelerator that was tried and declined.
    pub fn refused(&mut self, class: ComputeClass, reason: impl Into<String>) {
        self.refusals.push(Refusal {
            class,
            reason: reason.into(),
        });
    }

    /// Record the path the run actually took.
    pub fn settled_on(&mut self, class: ComputeClass) {
        self.resolved = class;
    }

    /// Was the request honoured?
    #[must_use]
    pub fn honoured(&self) -> bool {
        self.requested.honoured_by(self.resolved)
    }

    /// The one canonical line a log reader parses.
    ///
    /// Stable by contract: `scripts/parity_host_receipt.sh` and
    /// `scripts/check_compute_resolution_line.sh` both key on this exact shape.
    #[must_use]
    pub fn wire_line(&self) -> String {
        format!(
            "{RESOLUTION_LINE_PREFIX} requested={} resolved={} honoured={} refused={}",
            self.requested.wire(),
            self.resolved.wire(),
            self.honoured(),
            if self.refusals.is_empty() {
                "-".to_string()
            } else {
                self.refusals
                    .iter()
                    .map(|r| r.class.wire())
                    .collect::<Vec<_>>()
                    .join(",")
            }
        )
    }

    /// The human andon, or `None` when the request was honoured.
    ///
    /// Unconditional at the call site — NOT `--verbose`-gated. A user who typed
    /// `--gpu` and received CPU is about to read a throughput number that is off
    /// by an order of magnitude, and they need to know regardless of verbosity.
    /// This is the same argument `gpu_forward_failure_msg` already makes for the
    /// CUDA forward error, applied to the outcome rather than to one cause.
    #[must_use]
    pub fn andon(&self) -> Option<String> {
        if self.honoured() {
            return None;
        }
        let mut out = format!(
            "warning: {} was requested; this run resolved to compute_class={}.",
            match self.requested {
                ComputeRequest::Accelerator => "--gpu".to_string(),
                ComputeRequest::Named(c) => format!("--backend {}", c.wire()),
                ComputeRequest::Cpu => "--no-gpu".to_string(),
                ComputeRequest::Auto => "no compute flag".to_string(),
            },
            self.resolved.wire()
        );
        if self.refusals.is_empty() {
            // NOT "nothing was refused". This cascade does not collect the
            // reasons, and saying so is the difference between an unknown and
            // an assertion.
            out.push_str(
                "\n         which paths were refused, and why, was not recorded on this cascade",
            );
        }
        for refusal in &self.refusals {
            out.push_str(&format!(
                "\n         refused {}: {}",
                refusal.class.wire(),
                refusal.reason
            ));
        }
        out.push_str("\n         Throughput from this run is a compute_class=");
        out.push_str(self.resolved.wire());
        out.push_str(
            " number. Labelling it as the requested class is the provenance \
             defect aprender#2790 exists to remove.",
        );
        Some(out)
    }

    /// Emit the resolution: the andon when unhonoured, then the wire line.
    ///
    /// The wire line is printed for every EXPLICIT request and for every
    /// downgrade. An `Auto` run that got what auto picked stays quiet: it made
    /// no claim, so there is nothing to contradict.
    pub fn report(&self) {
        if let Some(msg) = self.andon() {
            eprintln!("{msg}");
        }
        if self.requested.is_explicit() || !self.honoured() {
            eprintln!("{}", self.wire_line());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE DEFECT, as a test. `--gpu` that ends on CPU must not read as honoured.
    ///
    /// Mutation: make `honoured_by` return `true` for `Accelerator` regardless of
    /// `taken` — the shape a silent fallback has — and this goes RED.
    #[test]
    fn an_accelerator_request_that_lands_on_cpu_is_not_honoured() {
        let mut r = ComputeResolution::pending(ComputeRequest::Accelerator);
        r.refused(ComputeClass::Cuda, "cosine 0.9403 at position 59");
        r.refused(ComputeClass::Wgpu, "cosine 0.722249 < 0.99 at step 1/3");
        r.settled_on(ComputeClass::Cpu);
        assert!(!r.honoured(), "--gpu resolved to cpu is a downgrade");
        assert_eq!(r.resolved.wire(), "cpu");
    }

    /// THE DISCRIMINATION CASE. Without it the rule could be "nothing is ever
    /// honoured" and every assertion above would still read green.
    #[test]
    fn an_accelerator_request_that_lands_on_cuda_is_honoured_and_silent() {
        let mut r = ComputeResolution::pending(ComputeRequest::Accelerator);
        r.settled_on(ComputeClass::Cuda);
        assert!(r.honoured());
        assert!(
            r.andon().is_none(),
            "a honoured request must raise no andon: {:?}",
            r.andon()
        );
    }

    /// `--no-gpu` is a request too, and CPU honours it.
    #[test]
    fn a_deliberate_cpu_request_is_honoured_by_cpu_and_broken_by_an_accelerator() {
        let mut r = ComputeResolution::pending(ComputeRequest::Cpu);
        r.settled_on(ComputeClass::Cpu);
        assert!(r.honoured());
        let mut up = ComputeResolution::pending(ComputeRequest::Cpu);
        up.settled_on(ComputeClass::Cuda);
        assert!(!up.honoured(), "--no-gpu must never be silently upgraded");
    }

    /// A named backend is not satisfied by a DIFFERENT accelerator.
    ///
    /// This is #2779's shape: a build reporting a backend it did not enable.
    /// `--backend cuda` served by wgpu is a downgrade even though both are GPUs.
    #[test]
    fn a_named_backend_is_not_honoured_by_a_different_accelerator() {
        let mut r = ComputeResolution::pending(ComputeRequest::Named(ComputeClass::Cuda));
        r.refused(ComputeClass::Cuda, "F2 parity gate");
        r.settled_on(ComputeClass::Wgpu);
        assert!(!r.honoured());
        assert!(r.andon().is_some());
    }

    /// `Auto` made no claim, so nothing can contradict it — and it stays quiet.
    #[test]
    fn auto_is_honoured_by_every_class_and_prints_nothing() {
        for class in [
            ComputeClass::Cpu,
            ComputeClass::Cuda,
            ComputeClass::Wgpu,
            ComputeClass::Metal,
        ] {
            let mut r = ComputeResolution::pending(ComputeRequest::Auto);
            r.settled_on(class);
            assert!(r.honoured(), "auto must accept {class}");
            assert!(r.andon().is_none());
        }
        assert!(!ComputeRequest::Auto.is_explicit());
    }

    /// A request that cannot be distinguished from NO request cannot be
    /// reported as unhonoured. Before `ComputeRequest` existed, `--gpu` and no
    /// flag both reached the engine as `no_gpu: false` — this is the assertion
    /// that keeps them apart.
    #[test]
    fn gpu_and_no_flag_are_different_requests() {
        assert_eq!(
            ComputeRequest::from_flags(true, false, None),
            ComputeRequest::Accelerator
        );
        assert_eq!(
            ComputeRequest::from_flags(false, false, None),
            ComputeRequest::Auto
        );
        assert_ne!(
            ComputeRequest::from_flags(true, false, None),
            ComputeRequest::from_flags(false, false, None)
        );
    }

    /// `--no-gpu` wins over `--gpu`: a CPU request is never upgraded.
    #[test]
    fn no_gpu_wins_over_gpu() {
        assert_eq!(
            ComputeRequest::from_flags(true, true, None),
            ComputeRequest::Cpu
        );
        assert_eq!(
            ComputeRequest::from_flags(false, false, Some("cpu")),
            ComputeRequest::Cpu
        );
        assert_eq!(
            ComputeRequest::from_flags(false, false, Some("cuda")),
            ComputeRequest::Named(ComputeClass::Cuda)
        );
    }

    /// The wire line is what a producer parses. Its shape is a contract.
    #[test]
    fn the_wire_line_names_both_halves_and_every_refusal() {
        let mut r = ComputeResolution::pending(ComputeRequest::Accelerator);
        r.refused(ComputeClass::Cuda, "F2");
        r.refused(ComputeClass::Wgpu, "parity");
        r.settled_on(ComputeClass::Cpu);
        let line = r.wire_line();
        assert!(line.starts_with(RESOLUTION_LINE_PREFIX), "{line}");
        assert!(line.contains("requested=accelerator"), "{line}");
        assert!(line.contains("resolved=cpu"), "{line}");
        assert!(line.contains("honoured=false"), "{line}");
        assert!(line.contains("refused=cuda,wgpu"), "{line}");
        // ONE line. A reader that does `read().lines().last()` must not get
        // half of it.
        assert!(!line.contains('\n'), "the wire line must be a single line");
    }

    /// The green wire line is DIFFERENT from the red one in the field a reader
    /// keys on. Without this the line could be a constant.
    #[test]
    fn the_wire_line_of_an_honoured_run_differs_where_it_matters() {
        let mut ok = ComputeResolution::pending(ComputeRequest::Accelerator);
        ok.settled_on(ComputeClass::Cuda);
        assert!(ok.wire_line().contains("resolved=cuda"));
        assert!(ok.wire_line().contains("honoured=true"));
        assert!(ok.wire_line().contains("refused=-"));
    }

    /// The andon must carry the ENGINE'S reason, not a summary of it. A reader
    /// diagnosing #2790 needs the cosine and the position.
    #[test]
    fn the_andon_quotes_every_refusal_verbatim() {
        let mut r = ComputeResolution::pending(ComputeRequest::Accelerator);
        r.refused(
            ComputeClass::Cuda,
            "GPU output diverges from CPU at position 59 (cosine 0.9403)",
        );
        r.settled_on(ComputeClass::Cpu);
        let msg = r.andon().expect("a downgrade must raise the andon");
        assert!(msg.contains("--gpu was requested"), "{msg}");
        assert!(msg.contains("compute_class=cpu"), "{msg}");
        assert!(msg.contains("position 59"), "{msg}");
        assert!(msg.contains("cosine 0.9403"), "{msg}");
        assert!(
            msg.contains("2790"),
            "the andon must cite the defect: {msg}"
        );
    }

    /// `pending` fails CLOSED: an engine that forgets to record its success
    /// reports the slow path, never a fast one it did not take.
    #[test]
    fn a_resolution_nobody_settled_reports_cpu() {
        let r = ComputeResolution::pending(ComputeRequest::Accelerator);
        assert_eq!(r.resolved, ComputeClass::Cpu);
        assert!(!r.honoured());
    }

    /// The class tokens ARE `bench_receipt.py`'s vocabulary. A token that is
    /// not in `COMPUTE_CLASSES` cannot be written into a receipt.
    #[test]
    fn every_class_token_is_in_the_receipt_vocabulary() {
        const RECEIPT: [&str; 5] = ["cpu", "cuda", "metal", "wgpu", "unknown"];
        for class in [
            ComputeClass::Cpu,
            ComputeClass::Cuda,
            ComputeClass::Wgpu,
            ComputeClass::Metal,
        ] {
            assert!(
                RECEIPT.contains(&class.wire()),
                "{} is not a bench_receipt.py compute_class",
                class.wire()
            );
        }
    }
}
