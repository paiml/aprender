//! FALSIFY-GPUTRAIN-004 / INV-GPUTRAIN-004 / GATE-GPUTRAIN-005 — algorithm-level
//! PARTIAL discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §14
//! (task #132 CUDA training backend gap).
//!
//! Contract: `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 → v1.1.0
//! adds `GATE-GPUTRAIN-005` PARTIAL-discharge evidence binding the pure
//! dispatch invariant: choosing `Device::Cpu` never invokes CUDA code, and
//! choosing `Device::Cuda(_)` never invokes CPU-only code.
//!
//! INV-GPUTRAIN-004 states: "CPU fallback path remains fully functional (peer
//! GATE-TRAIN-001..010 still PASS)." The symmetric dual is also in scope:
//! the silent-CPU-fallback-when-CUDA-was-requested class (task #126's defect)
//! must be a ship blocker. Together they bound the `requested → dispatched`
//! relation:
//!
//!   - `Device::Cpu` requested ⇒ `Device::Cpu` dispatched. Full stop.
//!   - `Device::Cuda(_)` requested ⇒ `Device::Cuda(_)` dispatched.
//!   - Any cross-assignment is a silent-promotion / silent-fallback bug.
//!   - Any unknown label is conservatively Fail (no implicit defaults).
//!
//! The compute-heavy portion (actually running a training step through either
//! `TransformerTrainer` or `CudaTransformerTrainer`) is intentionally out of
//! scope here; the decision rule is what the live dispatch wrapper calls
//! *before* instantiating any trainer, and changing either dispatch set or
//! the equality rule breaks this test before any trainer is allocated.

/// Set of device-tag labels that count as "the CPU training path."
/// Pinned as a slice of byte-literals so a mutation that merged the CPU
/// and CUDA sets (e.g. a typo in device dispatch switch) is caught
/// immediately by the provenance pin in the mutation survey.
pub const AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS: &[&str] = &["Cpu"];

/// Set of device-tag labels that count as "the CUDA training path."
/// Mirrors the `Device::Cuda` enum variant name. If the enum gets a
/// third variant (e.g. `Rocm`, `Metal`), the survey's provenance pin
/// forces a contract amendment in lockstep.
pub const AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS: &[&str] = &["Cuda"];

/// Binary verdict for FALSIFY-GPUTRAIN-004 / GATE-GPUTRAIN-005.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gputrain004Verdict {
    /// The requested device matches the dispatched device's equivalence
    /// class (both CPU or both CUDA).
    Pass,
    /// Requested vs dispatched are in different classes, or either is
    /// unknown / empty — both are ship-blocking bugs.
    Fail,
}

/// Algorithm-level verdict rule for INV-GPUTRAIN-004: given the operator-
/// requested device class and the class the dispatch wrapper actually
/// selected, Pass iff both labels live in the same set
/// (CPU ↔ CPU, CUDA ↔ CUDA). Every other combination — including any
/// unknown label on either side — is a conservative Fail because silent
/// promotion (cpu→cuda) and silent fallback (cuda→cpu) are both recorded
/// defect classes (task #126 is the silent-fallback proof-of-existence).
#[must_use]
pub fn verdict_from_dispatch_label(requested: &str, dispatched: &str) -> Gputrain004Verdict {
    let req_cpu = AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS.contains(&requested);
    let req_cuda = AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS.contains(&requested);
    let dis_cpu = AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS.contains(&dispatched);
    let dis_cuda = AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS.contains(&dispatched);

    match (req_cpu, req_cuda, dis_cpu, dis_cuda) {
        (true, false, true, false) => Gputrain004Verdict::Pass, // cpu → cpu
        (false, true, false, true) => Gputrain004Verdict::Pass, // cuda → cuda
        _ => Gputrain004Verdict::Fail,
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GPUTRAIN-004 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-GPUTRAIN-004 algorithm-level PARTIAL discharge: prove the
    /// CPU-fallback-preserved + no-silent-promotion dispatch invariant.
    /// Any mutation that swaps the equality rule, quietly accepts a
    /// cross-class dispatch, or defaults unknown labels to Pass must
    /// break this test before the live trainer is instantiated.
    #[test]
    fn falsify_gputrain_004_cpu_fallback_preserved_logic() {
        // Section 1: cpu → cpu — baseline Pass. The CPU-only host path
        // (INV-GPUTRAIN-004's primary rule) must round-trip.
        assert_eq!(
            verdict_from_dispatch_label("Cpu", "Cpu"),
            Gputrain004Verdict::Pass,
            "cpu → cpu must Pass (primary INV-GPUTRAIN-004 case)",
        );

        // Section 2: cuda → cuda — baseline Pass on the CUDA side. Any
        // mutation that flipped the Pass case to Fail here would break
        // every GPU training dispatch, so catching it is free.
        assert_eq!(
            verdict_from_dispatch_label("Cuda", "Cuda"),
            Gputrain004Verdict::Pass,
            "cuda → cuda must Pass (symmetric CUDA path preservation)",
        );

        // Section 3: cpu → cuda — silent GPU promotion bug. Catches a
        // refactor that accidentally upgrades a `--device cpu` request
        // to GPU because a helper preferred CUDA when it was "available."
        // Must Fail — operators who asked for CPU parity runs MUST get
        // CPU dispatch (reproducibility, debug isolation).
        assert_eq!(
            verdict_from_dispatch_label("Cpu", "Cuda"),
            Gputrain004Verdict::Fail,
            "cpu → cuda must Fail (silent GPU promotion bug)",
        );

        // Section 4: cuda → cpu — silent CPU fallback bug. THIS IS THE
        // EXACT DEFECT TASK #126 HIT. 14 minutes of compute at 0 MiB GPU
        // memory because the CLI accepted `--device cuda` and ran CPU.
        // Must Fail — that's the whole point of INV-GPUTRAIN-002.
        assert_eq!(
            verdict_from_dispatch_label("Cuda", "Cpu"),
            Gputrain004Verdict::Fail,
            "cuda → cpu must Fail (this is the task #126 silent-CPU-fallback bug)",
        );

        // Section 5: unknown label on either side. Catches a refactor
        // that added a third device (e.g. "Metal", "Rocm", "Xpu")
        // without an explicit contract amendment. Conservative Fail
        // because default-behaviour on unknown labels is what made
        // task #126 possible in the first place.
        assert_eq!(
            verdict_from_dispatch_label("xpu", "Cpu"),
            Gputrain004Verdict::Fail,
            "unknown requested label must Fail (no implicit default)",
        );
        assert_eq!(
            verdict_from_dispatch_label("Cpu", "metal"),
            Gputrain004Verdict::Fail,
            "unknown dispatched label must Fail (no implicit default)",
        );
        assert_eq!(
            verdict_from_dispatch_label("rocm", "rocm"),
            Gputrain004Verdict::Fail,
            "unknown label on both sides must Fail even if matched \
             (only the two recorded sets are allowed)",
        );

        // Section 6: empty string on either side. A stringly-typed
        // dispatch wrapper is easy to break on uninitialized default
        // values; an empty label must never Pass.
        assert_eq!(
            verdict_from_dispatch_label("", "Cpu"),
            Gputrain004Verdict::Fail,
            "empty requested label must Fail",
        );
        assert_eq!(
            verdict_from_dispatch_label("Cpu", ""),
            Gputrain004Verdict::Fail,
            "empty dispatched label must Fail",
        );
        assert_eq!(
            verdict_from_dispatch_label("", ""),
            Gputrain004Verdict::Fail,
            "both-empty must Fail (degenerate input, no implicit default)",
        );

        // Section 7: case-sensitivity and whitespace. A mutation that
        // relaxed to case-insensitive compare (e.g. `.eq_ignore_ascii_
        // case()`) would silently accept "CPU" vs "Cpu" which are NOT
        // in the pinned slice. Fail-closed is required.
        assert_eq!(
            verdict_from_dispatch_label("CPU", "Cpu"),
            Gputrain004Verdict::Fail,
            "uppercase `CPU` is not in the pinned slice — must Fail",
        );
        assert_eq!(
            verdict_from_dispatch_label("cpu", "Cpu"),
            Gputrain004Verdict::Fail,
            "lowercase `cpu` is not in the pinned slice — must Fail \
             (slice contents are the authoritative grammar)",
        );
        assert_eq!(
            verdict_from_dispatch_label("Cpu ", "Cpu"),
            Gputrain004Verdict::Fail,
            "trailing whitespace must Fail (no implicit trim)",
        );

        // Provenance pin on slice contents. Load-bearing: if a new
        // Device variant is added (e.g. Rocm, Metal), both the enum
        // side and this slice must move together. The byte-literal
        // shape prevents accidental merger into a single set.
        assert_eq!(
            AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS,
            &["Cpu"],
            "CPU dispatch set must be exactly [\"Cpu\"] \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-004)",
        );
        assert_eq!(
            AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS,
            &["Cuda"],
            "CUDA dispatch set must be exactly [\"Cuda\"] \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-004)",
        );
        // Explicitly prove the two sets are disjoint. Any mutation that
        // merged them (e.g. both contained "Cpu") would trivially make
        // every verdict Pass.
        for cpu in AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS {
            assert!(
                !AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS.contains(cpu),
                "CPU/CUDA dispatch sets must be disjoint",
            );
        }
    }
}
