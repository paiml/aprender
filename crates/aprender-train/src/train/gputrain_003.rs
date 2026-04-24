//! FALSIFY-GPUTRAIN-003 / INV-GPUTRAIN-003 / GATE-GPUTRAIN-003 — algorithm-level
//! PARTIAL discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §14
//! (task #132 CUDA training backend gap).
//!
//! Contract: `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 → v1.1.0
//! adds `GATE-GPUTRAIN-003` PARTIAL-discharge evidence binding the pure
//! decision rule for the `nvidia-smi` residency probe.
//!
//! INV-GPUTRAIN-003 states: "When the resolved backend is CUDA, a GPU process
//! MUST be visible via `nvidia-smi --query-compute-apps`. A post-init probe
//! runs within 5 seconds of step 0 and asserts `pid == training_pid AND
//! used_memory_mib > 0`." This file discharges the *decision rule* at
//! `PARTIAL_ALGORITHM_LEVEL`:
//!
//!   1. `parse_nvidia_smi_compute_apps(output) -> Result<Vec<…>, ()>`
//!      — strict parser for `nvidia-smi --query-compute-apps=pid,used_memory
//!      --format=csv,noheader,nounits` output. Each line is `<pid>, <used_mib>`
//!      (comma-space). Both fields must be non-negative integers; empty lines
//!      are skipped; any parse failure is a conservative `Err(())`.
//!
//!   2. `verdict_from_residency(training_pid, apps) -> Gputrain003Verdict`
//!      — aggregate rule: `Pass` iff any `app.pid == training_pid AND
//!      app.used_memory_mib >= AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB`. An empty
//!      slice is `Fail` (no active processes ⇒ no GPU residency proof).
//!
//! The compute-heavy portion of the AC (`spawn nvidia-smi` + 5-second poll
//! window + capture stdout) is intentionally out of scope here; the threshold
//! rule is what the live probe must call, and changing either constant
//! (the 5-second poll window, the 1-MiB floor) breaks this test before any
//! subprocess is launched.

/// Maximum seconds between step 0 and the first `nvidia-smi` probe call.
/// Pinned to the INV-GPUTRAIN-003 rule text so that any drift in the
/// probe-window timing (e.g. tightening to 1 s or relaxing to 30 s) must
/// move this constant in lockstep.
pub const AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS: u32 = 5;

/// Minimum `used_memory_mib` that counts as "GPU actually allocated
/// something on behalf of this pid." Zero mem means the pid registered
/// with the driver but never cudaMalloc'd — the exact symptom task #126
/// caught.
pub const AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB: u64 = 1;

/// A single row from `nvidia-smi --query-compute-apps=pid,used_memory
/// --format=csv,noheader,nounits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvidiaSmiComputeApp {
    /// Process ID the CUDA driver reports as owning the allocation.
    pub pid: u32,
    /// MiB of GPU memory currently resident for that pid.
    pub used_memory_mib: u64,
}

/// Binary verdict for FALSIFY-GPUTRAIN-003 / GATE-GPUTRAIN-003.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gputrain003Verdict {
    /// Residency proof held: some app in the probe output matches our pid
    /// AND is holding at least the minimum amount of GPU memory.
    Pass,
    /// Residency proof failed: no match, zero-mem match, or empty output.
    Fail,
}

/// Strict parser for `nvidia-smi --query-compute-apps=pid,used_memory
/// --format=csv,noheader,nounits` output.
///
/// Format (documented by NVIDIA): one line per compute app, `<pid>, <mib>`
/// with a literal comma-space separator and no header row. Blank lines
/// between records are tolerated (some `nvidia-smi` versions emit a
/// trailing newline).
///
/// # Errors
///
/// Returns `Err(())` on any line that is not `<digits>, <digits>` —
/// wrong separator, non-digit chars, missing field, or integer overflow.
/// The error is intentionally empty-payload so callers treat it as a
/// single opaque "parse failed" signal and conservatively map to Fail.
pub fn parse_nvidia_smi_compute_apps(output: &str) -> Result<Vec<NvidiaSmiComputeApp>, ()> {
    let mut apps = Vec::new();
    for line in output.lines() {
        if line.is_empty() {
            continue;
        }
        // Split on the exact ", " separator — any other whitespace is
        // rejected because nvidia-smi emits comma-space canonically.
        let (pid_s, mem_s) = line.split_once(", ").ok_or(())?;
        let pid: u32 = pid_s.parse().map_err(|_| ())?;
        let used_memory_mib: u64 = mem_s.parse().map_err(|_| ())?;
        apps.push(NvidiaSmiComputeApp { pid, used_memory_mib });
    }
    Ok(apps)
}

/// Algorithm-level verdict rule for INV-GPUTRAIN-003: given our training
/// pid and the parsed nvidia-smi compute-app list, Pass iff some app has
/// `app.pid == training_pid AND app.used_memory_mib >=
/// AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB`.
///
/// Returns [`Gputrain003Verdict::Fail`] conservatively for every negative
/// case: empty slice, no matching pid, or matching pid with zero memory.
#[must_use]
pub fn verdict_from_residency(
    training_pid: u32,
    apps: &[NvidiaSmiComputeApp],
) -> Gputrain003Verdict {
    for app in apps {
        if app.pid == training_pid && app.used_memory_mib >= AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB {
            return Gputrain003Verdict::Pass;
        }
    }
    Gputrain003Verdict::Fail
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GPUTRAIN-003 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-GPUTRAIN-003 algorithm-level PARTIAL discharge: prove the
    /// residency-probe parse + aggregate rule binding `nvidia-smi` output
    /// to the training pid. Any mutation that relaxes the zero-memory
    /// rejection, accepts a non-matching pid, or silently treats empty
    /// input as Pass must break this test before any subprocess is
    /// launched.
    #[test]
    fn falsify_gputrain_003_residency_proof_logic() {
        // Section 1: happy path — our pid, 5000 MiB allocated. Baseline
        // Pass case. Catches any mutation that swaps comparison operators
        // (== vs !=, >= vs <) because any such flip breaks this.
        let training_pid = 12345;
        let output_happy = "12345, 5000\n";
        let apps = parse_nvidia_smi_compute_apps(output_happy).expect("canonical line parses");
        assert_eq!(apps, vec![NvidiaSmiComputeApp { pid: 12345, used_memory_mib: 5000 }],);
        assert_eq!(
            verdict_from_residency(training_pid, &apps),
            Gputrain003Verdict::Pass,
            "matching pid with non-zero mem must Pass",
        );

        // Section 2: our pid but zero memory. Exactly the task #126
        // defect: process registered with the CUDA driver but never
        // cudaMalloc'd a weight tensor. Must Fail — otherwise a silent
        // CPU training run would pass the residency gate.
        let output_zero = "12345, 0\n";
        let apps = parse_nvidia_smi_compute_apps(output_zero).expect("zero-mem line parses");
        assert_eq!(
            verdict_from_residency(training_pid, &apps),
            Gputrain003Verdict::Fail,
            "matching pid with 0 MiB must Fail (GPU allocated nothing)",
        );

        // Section 3: different pid owns all the memory. Catches the
        // "wrong pid captured" class where the probe read pid from a
        // stale env var or a different child process. Must Fail.
        let output_other = "99999, 5000\n";
        let apps = parse_nvidia_smi_compute_apps(output_other).expect("other-pid line parses");
        assert_eq!(
            verdict_from_residency(training_pid, &apps),
            Gputrain003Verdict::Fail,
            "non-matching pid must Fail even if it holds lots of memory",
        );

        // Section 4: empty output. Catches the CUDA-present-but-no-
        // active-compute-apps case where nvidia-smi returns an empty
        // record set. Must Fail — we need POSITIVE proof that our
        // process is resident, not absence of negative evidence.
        let apps_empty: Vec<NvidiaSmiComputeApp> =
            parse_nvidia_smi_compute_apps("").expect("empty input parses as zero-length slice");
        assert!(apps_empty.is_empty());
        assert_eq!(
            verdict_from_residency(training_pid, &apps_empty),
            Gputrain003Verdict::Fail,
            "empty compute-app list must Fail",
        );

        // Section 5: multi-process output. Catches the case where our
        // training pid is one of many on a shared GPU host (e.g.
        // jupyter kernel + `apr pretrain` + `nvidia-settings` probe).
        // Must Pass because our row has non-zero mem; must NOT
        // short-circuit on the first row if that row is a sibling
        // process. Tests both orderings.
        let output_multi_ours_first = "12345, 2000\n99999, 1500\n";
        let apps =
            parse_nvidia_smi_compute_apps(output_multi_ours_first).expect("two-line output parses");
        assert_eq!(apps.len(), 2);
        assert_eq!(
            verdict_from_residency(training_pid, &apps),
            Gputrain003Verdict::Pass,
            "multi-process output with our pid first must Pass",
        );
        let output_multi_ours_last = "99999, 1500\n12345, 2000\n";
        let apps =
            parse_nvidia_smi_compute_apps(output_multi_ours_last).expect("two-line output parses");
        assert_eq!(
            verdict_from_residency(training_pid, &apps),
            Gputrain003Verdict::Pass,
            "multi-process output with our pid last must Pass \
             (loop must not short-circuit on non-matching rows)",
        );

        // Section 6: malformed lines. Each of these is a parse failure
        // that must surface as Err(()) at the parser boundary. On Err
        // the caller conservatively treats apps as empty, which then
        // falls through to verdict Fail per Section 4.
        let no_comma = "12345 5000\n";
        assert_eq!(parse_nvidia_smi_compute_apps(no_comma), Err(()));
        let wrong_separator = "12345,5000\n"; // no space after comma
        assert_eq!(parse_nvidia_smi_compute_apps(wrong_separator), Err(()));
        let extra_whitespace = "12345,  5000\n"; // two spaces
        assert_eq!(parse_nvidia_smi_compute_apps(extra_whitespace), Err(()));
        let non_digit_pid = "abc, 5000\n";
        assert_eq!(parse_nvidia_smi_compute_apps(non_digit_pid), Err(()));
        let non_digit_mem = "12345, xyz\n";
        assert_eq!(parse_nvidia_smi_compute_apps(non_digit_mem), Err(()));
        let missing_field = "12345,\n"; // no mem after comma
        assert_eq!(parse_nvidia_smi_compute_apps(missing_field), Err(()));
        // Conservative fallback: on parse Err, verdict of empty slice is Fail.
        assert_eq!(
            verdict_from_residency(training_pid, &[]),
            Gputrain003Verdict::Fail,
            "conservative Fail when parse errored and caller passed empty slice",
        );

        // Section 7: u32::MAX / u64::MAX boundary sanity. Catches any
        // mutation that replaces unsigned comparison with a signed-cast
        // helper or that overflows on extreme inputs. MAX memory must
        // Pass; zero memory at MAX pid must Fail.
        let max_pid_max_mem =
            vec![NvidiaSmiComputeApp { pid: u32::MAX, used_memory_mib: u64::MAX }];
        assert_eq!(
            verdict_from_residency(u32::MAX, &max_pid_max_mem),
            Gputrain003Verdict::Pass,
            "u32::MAX pid + u64::MAX mem must Pass",
        );
        let max_pid_zero_mem = vec![NvidiaSmiComputeApp { pid: u32::MAX, used_memory_mib: 0 }];
        assert_eq!(
            verdict_from_residency(u32::MAX, &max_pid_zero_mem),
            Gputrain003Verdict::Fail,
            "u32::MAX pid + 0 MiB must Fail (zero-mem rule is exceptionless)",
        );

        // Provenance pin — both constants are load-bearing and
        // lockstep with the YAML contract. Any spec drift (tightening
        // the 5-s window or relaxing the 1-MiB floor) must move these
        // constants together with the YAML rule text.
        assert_eq!(
            AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS, 5,
            "INV-GPUTRAIN-003 poll window is 5 seconds \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-003)",
        );
        assert_eq!(
            AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB, 1,
            "INV-GPUTRAIN-003 min-mem floor is 1 MiB \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-003)",
        );
    }
}
