//! FALSIFY-KTEST-04-TRUENO-*: `APR_FORCE_ISA` caps trueno's backend selection the way
//! aprender-serve's `isa` gate caps its dispatch sites.

use crate::{cap_backend_for_forced_isa, refuse_unrunnable_forced_isa, Backend};

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

fn host(feats: &'static [&'static str]) -> impl Fn(&str) -> bool {
    move |f| feats.contains(&f)
}

/// FALSIFY-KTEST-04-TRUENO-S3: a level the host cannot run is refused, never capped to what it
/// has — the same answers as aprender-serve's `isa::parse` (`isa_tests` checks the two agree).
#[test]
fn an_unrunnable_forced_isa_is_refused_by_the_host_check() {
    let x86 = !cfg!(target_arch = "aarch64");
    let table: &[(Option<&str>, &[&str], Option<&str>)] = &[
        (None, &[], None),
        (Some("native"), &[], None),
        (Some(" scalar "), &[], None),
        (Some("avx512"), &["avx2", "fma", "avx512f"], (!x86).then_some("is not one of")),
        (
            Some("avx512"),
            &["avx2", "fma"],
            Some(if x86 { "lacks `avx512f`" } else { "is not one of" }),
        ),
        (Some("avx2"), &["avx2", "fma"], (!x86).then_some("is not one of")),
        (Some("avx2"), &["avx2"], Some(if x86 { "lacks `fma`" } else { "is not one of" })),
        (Some("avx2"), &[], Some(if x86 { "lacks `avx2`" } else { "is not one of" })),
        (Some("neon"), &["neon"], x86.then_some("is not one of")),
        (Some("neon"), &[], Some(if x86 { "is not one of" } else { "lacks `neon`" })),
        (Some("AVX2"), &["avx2", "fma"], Some("is not one of")),
    ];
    for (forced, feats, want) in table {
        let got = refuse_unrunnable_forced_isa(*forced, host(feats));
        match (want, &got) {
            (None, Ok(())) => {}
            (Some(w), Err(e)) if e.contains(w) => {}
            _ => panic!("{forced:?} on {feats:?}: want {want:?}, got {got:?}"),
        }
    }
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

fn child_output(force: &str) -> std::process::Output {
    let exe = std::env::current_exe().expect("test binary path");
    std::process::Command::new(exe)
        .args([
            "--exact",
            "force_isa_tests::forced_selection_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("APR_FORCE_ISA", force)
        .output()
        .expect("spawn the child test")
}

fn child(force: &str) -> String {
    let out = child_output(force);
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

/// FALSIFY-KTEST-04-TRUENO-REFUSE: a value naming another arch's level panics at backend
/// selection. Before the host check it ran as the detected backend under that name.
#[test]
fn an_unrunnable_forced_isa_panics_at_backend_selection() {
    let other = if cfg!(target_arch = "aarch64") { "avx2" } else { "neon" };
    let out = child_output(other);
    let text =
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "child {other} was not refused:\n{text}");
    assert!(
        text.contains(&format!("APR_FORCE_ISA={other:?} is not one of")),
        "child {other} failed, but not on the host check:\n{text}"
    );
}
