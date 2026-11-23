//! #4374 falsifiers: a Qwen3.5 serve reports where it runs and what its F2
//! guard measured. Before the fix `/v1/effective-config` said `unknown` with an
//! empty `backend_loaded`, and `parity` said "no GPU model loaded (cpu
//! residency)" while the hybrid decoded on CUDA behind a passed F2 guard.

use super::*;
use crate::gguf::qwen35_session::Qwen35F2Parity;

#[test]
fn a_guard_run_maps_to_a_status_that_never_overstates_it() {
    let pass = Qwen35F2Parity::judged(true, "fresh", Some(0.997), 65);
    assert_eq!(
        (pass.status, pass.cosine, pass.positions),
        ("PASS", Some(0.997), 65)
    );
    assert_eq!(
        Qwen35F2Parity::judged(true, "receipt", None, 65).status,
        "PASS"
    );
    // Accepted without comparing anything is not a pass.
    assert_eq!(
        Qwen35F2Parity::judged(true, "not-judged", None, 0).status,
        "not-run"
    );
    // A rejection is a FAIL whatever it was keyed on.
    assert_eq!(
        Qwen35F2Parity::judged(false, "fresh", Some(0.41), 65).status,
        "FAIL"
    );
    assert_eq!(
        Qwen35F2Parity::judged(false, "not-judged", None, 0).status,
        "FAIL"
    );
}

#[test]
fn falsify_4374_a_passed_f2_guard_reports_its_measured_cosine() {
    let f2 = Qwen35F2Parity::judged(true, "fresh", Some(0.9931), 65);
    let p = qwen35_parity(Some(&f2), true);
    assert_eq!(p.status, "PASS");
    assert_eq!(
        p.cosine,
        Some(0.9931),
        "the cosine the guard measured, verbatim"
    );
    assert_eq!(p.positions, 65);
    assert!((p.threshold - crate::infer::F2_GATE_COSINE_MIN).abs() < f32::EPSILON);
    assert!(
        p.basis.contains("F2 guard") && p.basis.contains("source=fresh"),
        "{}",
        p.basis
    );
    assert!(!p.basis.contains("cpu residency"), "{}", p.basis);
}

#[test]
fn a_receipt_hit_is_a_pass_without_a_cosine_and_says_so() {
    let f2 = Qwen35F2Parity::judged(true, "receipt", None, 65);
    let p = qwen35_parity(Some(&f2), true);
    assert_eq!((p.status.as_str(), p.cosine), ("PASS", None));
    assert!(p.basis.contains("not re-measured"), "{}", p.basis);
}

#[test]
fn no_guard_run_is_not_run_with_the_reason_for_each_residency() {
    let gpu = qwen35_parity(None, true);
    assert_eq!(
        (gpu.status.as_str(), gpu.cosine, gpu.positions),
        ("not-run", None, 0)
    );
    assert!(gpu.basis.contains("first prompt"), "{}", gpu.basis);
    let cpu = qwen35_parity(None, false);
    assert_eq!(cpu.status, "not-run");
    assert!(cpu.basis.contains("CPU"), "{}", cpu.basis);
}

#[test]
fn a_rejected_or_since_fallen_back_session_is_never_a_clean_pass() {
    let fail = qwen35_parity(
        Some(&Qwen35F2Parity::judged(false, "fresh", Some(0.41), 65)),
        false,
    );
    assert_eq!((fail.status.as_str(), fail.cosine), ("FAIL", Some(0.41)));
    let pass = Qwen35F2Parity::judged(true, "fresh", Some(0.99), 65);
    let moved = qwen35_parity(Some(&pass), false);
    assert!(moved.basis.contains("moved to the CPU"), "{}", moved.basis);
}
