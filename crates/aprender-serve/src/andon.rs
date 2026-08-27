//! PERF-006 (aprender#2706) — the andon lamp: ONE compute class, ONE
//! admission bound, three surfaces.
//!
//! APR-PERF-GATE-001 v2.2 §4 lists the Andon row as *"one `compute_class()`
//! feeds the serve banner, `/health`, and `provenance.compute_class`"* and
//! marks it **pending**. Before this module there were three answers to
//! "what is this process running on, and how many requests will it run at
//! once":
//!
//! | surface | said | vocabulary |
//! |---|---|---|
//! | `apr bench --json` receipt | `compute_class` from apr-cli's own `cfg!` | `cpu/cuda/metal/wgpu/unknown` |
//! | `GET /health` | `compute_mode` from realizar's `AppState` | `cpu/gpu` |
//! | serve banner | nothing | — |
//!
//! Two spellings of the same fact and one silence. This module is the single
//! definition all three now read.
//!
//! # Why `max_in_flight` is here and not in a metrics endpoint
//!
//! §14 calls `max_in_flight` *"the cheapest confirmation of defect #2"* —
//! defect #2 being that apr does not batch. `contracts/batch-admission-v1.yaml`
//! records the measurement: wall time linear in concurrency to two decimals
//! (1.00 / 1.97 / 3.94 / 7.86 at N=1/2/4/8), aggregate 0.097x the comparator
//! at c=16 while per-user decode read 1.554x. A server that will only run one
//! generation at a time must SAY so, on the path where that is true, without
//! anyone running a sweep.
//!
//! [`Admission::Serialized`] is therefore the **default**, not a fallback that
//! only appears once something proves it. #2696's shape — a field that appears
//! only when the good case is active, and is silent on the failure it exists
//! to expose — is the thing this module is built to avoid. A bound above 1 has
//! to be *recorded by the code that creates it*; nothing else can raise the
//! lamp.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The vocabulary `provenance.compute_class` admits.
///
/// Mirrors `contracts/apr-bench-receipt-v1.yaml`:
/// `compute_class in { cpu, cuda, metal, wgpu, unknown }`. An unrecognised
/// value is worse than a missing one: it reads as measured.
pub const COMPUTE_CLASSES: [&str; 5] = ["cpu", "cuda", "metal", "wgpu", "unknown"];

/// How many generations this process will run through the model **at once**,
/// decided by construction rather than by measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// One generation at a time. This is the andon lamp.
    ///
    /// Every model handle the server installs behind a lock holds it across
    /// the whole generate call — `cuda_model` / `gpu_model` under
    /// `RwLock::write` (`api/cuda_chat_backend.rs:148` and `:204`, the
    /// fallback taken whenever no batch scheduler was wired;
    /// `api/openai_handlers.rs:715`; `api/gpu_completions_handler.rs:52`),
    /// `safetensors_cuda_model` under `Mutex::lock`
    /// (`api/cuda_chat_backend.rs:24`). A second request cannot start until
    /// the first finishes.
    Serialized,
    /// A scheduler admits this many generations concurrently.
    ///
    /// Recorded by the spawn function itself — see
    /// [`record_admission`] — so a caller cannot forget to report it and a
    /// caller cannot invent it.
    Batched(usize),
}

impl Admission {
    /// The number a receipt, a banner and `/health` all print.
    ///
    /// Always a number, on both arms. A `null` on the serialized arm would
    /// let "we do not batch" read as "not applicable".
    #[must_use]
    pub const fn max_in_flight(self) -> usize {
        match self {
            Self::Serialized => 1,
            Self::Batched(n) => n,
        }
    }

    /// Machine-readable tag: `"serialized"` or `"batched"`.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Serialized => "serialized",
            Self::Batched(_) => "batched",
        }
    }
}

/// `0` is the "nothing has recorded a bound" sentinel; [`admission`] reads it
/// as [`Admission::Serialized`], which is what a process with no scheduler is.
static ADMISSION_SLOTS: AtomicUsize = AtomicUsize::new(0);

/// Record the admission bound. Called by the code that CREATES the bound.
///
/// `slots <= 1` is recorded as [`Admission::Serialized`]: a scheduler with one
/// slot is a queue, and calling it "batched" would put the lamp out.
pub fn record_admission(slots: usize) {
    ADMISSION_SLOTS.store(slots, Ordering::Relaxed);
}

/// The admission regime this process is running under, right now.
#[must_use]
pub fn admission() -> Admission {
    match ADMISSION_SLOTS.load(Ordering::Relaxed) {
        0 | 1 => Admission::Serialized,
        n => Admission::Batched(n),
    }
}

/// How many generations this process will run at once. Never absent.
#[must_use]
pub fn max_in_flight() -> usize {
    admission().max_in_flight()
}

/// Is a CUDA runtime actually present, or did this build only *compile* for
/// one?
///
/// Verification discipline #2: never label a run by intent. A binary built
/// `--features cuda` on a host with no driver silently falls back, and a
/// receipt that claims `cuda` there is the fabricated-comparator class.
#[cfg(feature = "cuda")]
fn cuda_runtime_present() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The dispatch path this process can actually take.
///
/// NOT the hardware present: the path TAKEN. A CUDA-capable host running a
/// default-features build is `cpu`, because the code that would dispatch to
/// CUDA was never compiled in. That distinction is the whole point of the
/// field — a receipt proving WHICH BINARY ran but not WHICH PATH it took
/// catches the wrong-binary class (five in-tree rediscoveries) and misses the
/// wrong-compute-class one entirely, which is how a CPU-only apr side measured
/// against a CUDA comparator validates cleanly.
///
/// Feature gates are read FIRST and are decisive when absent: a build without
/// the feature cannot take that path, whatever `nvidia-smi` says.
///
/// # `metal` and `wgpu` are in the vocabulary and nothing emits them
///
/// Deliberate, and the reason this function moved out of `apr-cli`. The old
/// copy returned `"wgpu"` from `cfg!(feature = "wgpu")` — but apr-cli's `wgpu`
/// feature is declared `wgpu = ["inference"]`, so it enables no wgpu dispatch
/// anywhere; it only widens `ensure_accelerator_available`. That receipt field
/// named a path the binary could not take. realizar's own `gpu` feature is no
/// better a signal: it is **default-on** and gates trueno's wgpu primitives,
/// not an LLM decode path, so keying `"wgpu"` off it would report a GPU class
/// for every default build. Rule: never hand-assign a value a probe is
/// supposed to decide. When a wgpu or Metal decode path lands, this function
/// is the one place to extend, and `/health`, the banner and the receipt
/// follow without being touched.
#[must_use]
pub fn compute_class() -> &'static str {
    // Built for CUDA. It still only counts as `cuda` if the runtime is
    // actually there; otherwise this build silently fell back and every
    // surface must say so rather than claim the fast path.
    #[cfg(feature = "cuda")]
    {
        if cuda_runtime_present() {
            "cuda"
        } else {
            "cpu"
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        "cpu"
    }
}

/// The one line the serve banner prints.
///
/// Renders exactly the two facts `/health` returns and the receipt records, so
/// a reader comparing a terminal against a JSON body is comparing one source
/// against itself.
#[must_use]
pub fn andon_line() -> String {
    let a = admission();
    format!(
        "Compute: compute_class={} max_in_flight={} ({})",
        compute_class(),
        a.max_in_flight(),
        a.tag()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The class must be one of the values the receipt schema admits.
    #[test]
    fn compute_class_is_in_the_schema_vocabulary() {
        assert!(
            COMPUTE_CLASSES.contains(&compute_class()),
            "compute_class must be one of {COMPUTE_CLASSES:?}, got {}",
            compute_class()
        );
    }

    /// A build without a GPU feature cannot take a GPU path, whatever hardware
    /// is attached. This is the assertion that refuses a cross-class ratio.
    #[test]
    fn a_build_without_cuda_is_cpu() {
        if cfg!(feature = "cuda") {
            assert!(
                ["cuda", "cpu"].contains(&compute_class()),
                "a cuda build is `cuda` when the runtime is there and `cpu` when it is not"
            );
        } else {
            assert_eq!(
                compute_class(),
                "cpu",
                "no cuda compiled in, so no cuda path exists to report"
            );
        }
    }

    /// THE POINT OF THE MODULE. The banner line is not allowed to render a
    /// class of its own; if it ever does, this fails.
    #[test]
    fn the_banner_line_renders_the_same_class_and_bound_as_the_accessors() {
        let line = andon_line();
        assert!(
            line.contains(&format!("compute_class={}", compute_class())),
            "banner line disagrees with compute_class(): {line}"
        );
        assert!(
            line.contains(&format!("max_in_flight={}", max_in_flight())),
            "banner line disagrees with max_in_flight(): {line}"
        );
    }

    /// The default is the DEFECT, not the success case. A process that wired
    /// no scheduler reports 1, and says `serialized` while saying it.
    #[test]
    fn nothing_recorded_reports_the_serialized_lamp() {
        // Reads the sentinel directly rather than the global, so a test that
        // ran `record_admission` first cannot make this one pass or fail.
        assert_eq!(
            match 0usize {
                0 | 1 => Admission::Serialized,
                n => Admission::Batched(n),
            },
            Admission::Serialized
        );
        assert_eq!(Admission::Serialized.max_in_flight(), 1);
        assert_eq!(Admission::Serialized.tag(), "serialized");
    }

    /// A one-slot scheduler is a queue. Calling it "batched" would put the
    /// lamp out on exactly the configuration that needs it lit.
    #[test]
    fn a_single_slot_scheduler_is_still_serialized() {
        // Pure mapping, not the global: `record_admission(1)` must not be
        // able to report `batched`.
        for (slots, expected) in [
            (0usize, Admission::Serialized),
            (1, Admission::Serialized),
            (2, Admission::Batched(2)),
            (4, Admission::Batched(4)),
        ] {
            let got = match slots {
                0 | 1 => Admission::Serialized,
                n => Admission::Batched(n),
            };
            assert_eq!(got, expected, "slots={slots}");
        }
    }

    /// `record_admission` is observable through `admission()` — otherwise the
    /// scheduler could record a bound nothing ever reads.
    ///
    /// Restores the sentinel so the process-global default is not left dirty
    /// for the tests above (which read it only through pure mappings, but the
    /// next reader may not).
    #[test]
    fn a_recorded_bound_reaches_every_reader() {
        record_admission(6);
        assert_eq!(admission(), Admission::Batched(6));
        assert_eq!(max_in_flight(), 6);
        assert!(andon_line().contains("max_in_flight=6"));
        assert!(andon_line().contains("(batched)"));
        record_admission(0);
        assert_eq!(admission(), Admission::Serialized);
        assert!(andon_line().contains("(serialized)"));
    }
}
