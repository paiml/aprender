//! FALSIFY-KTEST-04-TRUENO-*: `APR_FORCE_ISA` caps trueno's backend selection the way
//! aprender-serve's `isa` gate caps its dispatch sites.

use crate::{cap_backend_for_forced_isa, Backend};

#[test]
fn forced_isa_caps_the_detected_backend() {
    use Backend::{Scalar, AVX, AVX2, AVX512, NEON, SSE2};
    let table: &[(Option<&str>, Backend, Backend)] = &[
        (None, AVX2, AVX2),
        (Some(""), AVX512, AVX512),
        (Some("native"), AVX512, AVX512),
        (Some("avx512"), AVX512, AVX512),
        (Some("avx2"), AVX512, AVX2),
        (Some("avx2"), AVX2, AVX2),
        // Capping only lowers: a host below AVX2 keeps what it has.
        (Some("avx2"), AVX, AVX),
        (Some("avx2"), SSE2, SSE2),
        (Some(" scalar "), AVX512, Scalar),
        (Some("scalar"), AVX2, Scalar),
        (Some("scalar"), NEON, Scalar),
        (Some("neon"), NEON, NEON),
    ];
    for (forced, detected, want) in table {
        assert_eq!(
            cap_backend_for_forced_isa(*forced, *detected),
            *want,
            "{forced:?} on {detected:?}"
        );
    }
}

#[test]
#[should_panic(expected = "is not one of")]
fn an_unknown_forced_isa_is_refused() {
    let _ = cap_backend_for_forced_isa(Some("AVX2"), Backend::AVX2);
}

const TAG: &str = "KTEST04-TRUENO";

/// Runs only as the child of `forcing_reaches_backend_selection`, under its `APR_FORCE_ISA`.
#[test]
#[ignore = "child process of forcing_reaches_backend_selection; meaningless on its own"]
fn forced_selection_child() {
    let v = crate::Vector::from_slice(&[1.0f32, 2.0, 3.0]);
    println!(
        "{TAG} best={:?} vector={:?} compute_bound={:?}",
        crate::select_best_available_backend(),
        v.backend(),
        crate::select_backend_for_operation(crate::OperationType::ComputeBound)
    );
}

fn child(force: &str) -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "force_isa_tests::forced_selection_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("APR_FORCE_ISA", force)
        .output()
        .expect("spawn the child test");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "child {force}: {}\n{stdout}",
        String::from_utf8_lossy(&out.stderr)
    );
    // libtest prints `test <name> ... ` without a newline before the child's own output.
    stdout
        .lines()
        .find_map(|l| l.find(TAG).map(|i| &l[i..]))
        .unwrap_or_else(|| panic!("child {force} printed no probe line:\n{stdout}"))
        .to_string()
}

/// FALSIFY-KTEST-04-TRUENO-ENGAGE: a forced-scalar process selects `Scalar` everywhere, and
/// the same probe unforced does not — so the scalar arm is not what this host picks anyway.
#[cfg(target_arch = "x86_64")]
#[test]
fn forcing_reaches_backend_selection() {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "this host has no AVX2+FMA backend to force off, so the test would prove nothing"
    );
    assert_eq!(child("scalar"), format!("{TAG} best=Scalar vector=Scalar compute_bound=Scalar"));
    let native = child("native");
    assert!(native.starts_with(&format!("{TAG} best=AVX2 vector=AVX2 ")), "native: {native}");
    let avx2 = child("avx2");
    assert!(!avx2.contains("AVX512"), "avx2 forced: {avx2}");
}
