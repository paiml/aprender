//! FALSIFY-KTEST-04-*: `APR_FORCE_ISA` parses fail-closed, reaches every dispatch site, and
//! demonstrably changes which path runs.

use super::{parse, within, Ceiling};

fn host(feats: &'static [&'static str]) -> impl Fn(&str) -> bool {
    move |f| feats.contains(&f)
}

const AVX512_HOST: &[&str] = &["avx2", "fma", "avx512f"];
const AVX2_HOST: &[&str] = &["avx2", "fma"];

/// FALSIFY-KTEST-04-PARSE: accepted values, and the S-3 refusals (a level the host lacks is an
/// error, never a silent `native`).
#[cfg(target_arch = "x86_64")]
#[test]
fn force_isa_values_parse_fail_closed() {
    let ok: &[(Option<&str>, &[&str], Ceiling)] = &[
        (None, AVX2_HOST, Ceiling::Native),
        (Some(""), AVX2_HOST, Ceiling::Native),
        (Some("native"), &[], Ceiling::Native),
        (Some(" scalar "), AVX512_HOST, Ceiling::Scalar),
        (Some("scalar"), &[], Ceiling::Scalar),
        (Some("avx2"), AVX512_HOST, Ceiling::Simd),
        (Some("avx2"), AVX2_HOST, Ceiling::Simd),
        (Some("avx512"), AVX512_HOST, Ceiling::Native),
    ];
    for (v, h, want) in ok {
        assert_eq!(parse(*v, host(h)), Ok(*want), "{v:?} on {h:?}");
    }
    let refused: &[(&str, &[&str], &str)] = &[
        ("avx512", AVX2_HOST, "lacks `avx512f`"),
        ("avx2", &["avx2"], "lacks `fma`"),
        ("avx2", &[], "lacks `avx2`"),
        ("neon", AVX512_HOST, "is not one of"),
        ("AVX2", AVX512_HOST, "is not one of"),
        ("sse2", AVX512_HOST, "is not one of"),
        ("avx-512", AVX512_HOST, "is not one of"),
    ];
    for (v, h, why) in refused {
        let e = parse(Some(v), host(h)).expect_err(v);
        assert!(e.contains(why), "{v} on {h:?}: {e}");
    }
}

/// FALSIFY-KTEST-04-LADDER: what each ceiling lets through. An unclassified feature is off
/// below `native`, so a new `cpu_feature!("…")` cannot slip past a forced run.
#[test]
fn ceilings_admit_only_their_rung() {
    let table: &[(&str, [bool; 3])] = &[
        // feature        scalar  simd   native
        ("avx2", [false, true, true]),
        ("fma", [false, true, true]),
        ("f16c", [false, true, true]),
        ("sse4.1", [false, true, true]),
        ("neon", [false, true, true]),
        ("avx512f", [false, false, true]),
        ("avx512bw", [false, false, true]),
        ("avx512vnni", [false, false, true]),
        ("dotprod", [false, false, true]),
        ("amx-tile", [false, false, true]),
    ];
    for (f, want) in table {
        let got = [Ceiling::Scalar, Ceiling::Simd, Ceiling::Native].map(|c| within(c, f));
        assert_eq!(&got, want, "{f}");
    }
}

/// A line of Rust that asks the CPU for a feature without going through the gate.
fn raw_detection(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or("");
    code.contains("is_x86_feature_detected!(") || code.contains("is_aarch64_feature_detected!(")
}

#[test]
fn raw_detection_case_table() {
    let cases: &[(&str, bool)] = &[
        (r#"        if is_x86_feature_detected!("avx2") {"#, true),
        (
            r#"    let a = std::arch::is_x86_feature_detected!("fma");"#,
            true,
        ),
        (
            r#"("neon", std::arch::is_aarch64_feature_detected!("neon")),"#,
            true,
        ),
        (
            r#"if is_x86_feature_detected!("avx2") && x { // gate later"#,
            true,
        ),
        (r#"        if crate::isa::cpu_feature!("avx2") {"#, false),
        (
            "            // SAFETY: AVX2 verified by is_x86_feature_detected!(\"avx2\")",
            false,
        ),
        (
            "/// 1. AVX2 available (use `is_x86_feature_detected!`)",
            false,
        ),
        (
            r#"    foo(); // was is_x86_feature_detected!("avx2")"#,
            false,
        ),
    ];
    for (line, want) in cases {
        assert_eq!(raw_detection(line), *want, "{line}");
    }
}

/// Files that may probe the CPU directly: the gate itself, and the two registry probes, which
/// filter through `isa::permits` (asserted below). Test files choose paths on purpose.
const PROBE_FILES: &[&str] = &["isa.rs", "kernel_registry.rs", "kernel_registry_parity.rs"];

/// `tests/` dirs and `*tests*.rs` files — this crate's naming for test-only code.
fn is_test_file(rel: &str) -> bool {
    rel.contains("tests")
}

/// A `.rs` file of library code: the files the ISA gate scan reads.
fn is_library_source(p: &std::path::Path, rel: &str) -> bool {
    p.extension().is_some_and(|ext| ext == "rs") && !is_test_file(rel)
}

/// One library file of the gate scan: a probe file other than the gate itself must still
/// filter through `isa::permits`; any other file's raw detections are appended to `raw`.
fn scan_library_file(rel: &str, text: &str, raw: &mut Vec<String>) {
    if PROBE_FILES.contains(&rel) {
        assert!(
            rel == "isa.rs" || text.contains("crate::isa::permits("),
            "{rel} probes the CPU but no longer filters through isa::permits"
        );
        return;
    }
    raw.extend(
        text.lines()
            .enumerate()
            .filter(|(_, line)| raw_detection(line))
            .map(|(i, line)| format!("src/{rel}:{}: {}", i + 1, line.trim())),
    );
}

/// FALSIFY-KTEST-04-GUARD: no runtime ISA decision in this crate bypasses `APR_FORCE_ISA`. A
/// raw `is_x86_feature_detected!` in library code is a path a forced run cannot reach, so its
/// receipts would describe a path that did not run (KTEST-001 §4, S-2).
#[test]
fn every_runtime_isa_decision_goes_through_the_gate() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![src.clone()];
    let (mut scanned, mut gated, mut raw) = (0usize, 0usize, Vec::new());
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).expect("read src dir") {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let rel = p
                .strip_prefix(&src)
                .expect("under src")
                .to_string_lossy()
                .replace('\\', "/");
            if !is_library_source(&p, &rel) {
                continue;
            }
            let text = std::fs::read_to_string(&p).expect("read .rs");
            scanned += 1;
            gated += text.matches("isa::cpu_feature!(").count();
            scan_library_file(&rel, &text, &mut raw);
        }
    }
    // Anti-vacuity: the walk found the crate, and the gate is actually in use.
    assert!(scanned > 500, "scanned only {scanned} files under {src:?}");
    assert!(
        gated >= 50,
        "only {gated} cpu_feature! sites: the rewrite was undone?"
    );
    assert!(
        raw.is_empty(),
        "runtime ISA checks that bypass APR_FORCE_ISA — use crate::isa::cpu_feature!:\n{}",
        raw.join("\n")
    );
}

#[cfg(target_arch = "x86_64")]
mod forced {
    const IN_DIM: usize = 4096;
    const OUT_DIM: usize = 64;
    const TAG: &str = "KTEST04-PROBE";

    /// Q8_0 rows (f16 scale + 32 × i8) and activations from a fixed xorshift stream.
    fn workload() -> (Vec<u8>, Vec<f32>) {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let blocks = IN_DIM / 32;
        let mut w = Vec::with_capacity(OUT_DIM * blocks * 34);
        for _ in 0..OUT_DIM * blocks {
            #[allow(clippy::cast_precision_loss)]
            let scale = 0.01 + (next() % 1000) as f32 * 4e-5;
            w.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            w.extend((0..32).map(|_| (next() & 0xFF) as u8));
        }
        #[allow(clippy::cast_precision_loss)]
        let x = (0..IN_DIM)
            .map(|_| ((next() % 20_001) as f32 - 10_000.0) * 3e-4)
            .collect();
        (w, x)
    }

    /// Runs only as the child of `forcing_changes_which_path_runs`, under its `APR_FORCE_ISA`.
    #[test]
    #[ignore = "child process of forcing_changes_which_path_runs; meaningless on its own"]
    fn forced_dispatch_child() {
        let (w, x) = workload();
        let got = crate::quantize::fused_q8_0_q8_0_parallel_matvec(&w, &x, IN_DIM, OUT_DIM)
            .expect("matvec");
        let (scales, quants) = crate::quantize::quantize_activations_q8_0(&x);
        let row = w.len() / OUT_DIM;
        let equal = (0..OUT_DIM)
            .filter(|&o| {
                let want = crate::quantize::fused_q8_0_q8_0_dot_scalar(
                    &w[o * row..(o + 1) * row],
                    &scales,
                    &quants,
                    IN_DIM,
                );
                got[o].to_bits() == want.to_bits()
            })
            .count();
        // The registry's view of this CPU must narrow with the dispatch sites (KTEST-04 §4).
        let host = format!("{:?}", crate::kernel_registry::Target::host());
        println!(
            "{TAG} ceiling={} refused={} equal={equal} rows={OUT_DIM} registry_avx2={}",
            crate::isa::ceiling().name(),
            crate::isa::refused(),
            host.contains("\"avx2\"")
        );
    }

    fn child(force: &str) -> (String, u64, usize, bool) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--exact",
                "isa::tests::forced::forced_dispatch_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(super::super::ENV, force)
            .output()
            .expect("spawn the child test");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child {force}: {}\n{stdout}",
            String::from_utf8_lossy(&out.stderr)
        );
        // libtest prints `test <name> ... ` without a newline before the child's own output.
        let line = stdout
            .lines()
            .find_map(|l| l.find(TAG).map(|i| &l[i..]))
            .unwrap_or_else(|| panic!("child {force} printed no probe line:\n{stdout}"));
        let field = |k: &str| {
            line.split_whitespace()
                .find_map(|t| t.strip_prefix(k))
                .unwrap_or_else(|| panic!("no {k} in {line}"))
                .to_string()
        };
        (
            field("ceiling="),
            field("refused=").parse().expect("refused"),
            field("equal=").parse().expect("equal"),
            field("registry_avx2=").parse().expect("registry_avx2"),
        )
    }

    /// FALSIFY-KTEST-04-ENGAGE (CLAUDE.md verification rule 2: prove the mechanism engaged).
    /// Forced `scalar`, the public matvec is bit-for-bit the scalar reference on every row and
    /// the gate refused live sites; `native`, nothing is refused and the AVX2 path gives
    /// different bits on some row — so the workload can tell the two paths apart, and the
    /// scalar arm's agreement is not an accident of the inputs.
    #[test]
    fn forcing_changes_which_path_runs() {
        assert!(
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"),
            "this host has no AVX2+FMA path to force off, so the test would prove nothing"
        );
        let (c, refused, equal, registry_avx2) = child("scalar");
        assert_eq!(c, "scalar");
        assert!(
            !registry_avx2,
            "scalar forced, yet the registry's host target still reports avx2"
        );
        assert!(
            refused > 0,
            "scalar forced, yet no dispatch site refused a feature"
        );
        assert_eq!(
            equal, OUT_DIM,
            "scalar forced, yet {equal}/{OUT_DIM} rows match scalar"
        );

        let (c, refused, equal, registry_avx2) = child("native");
        assert_eq!(c, "native");
        assert!(
            registry_avx2,
            "native: the registry's host target lacks avx2"
        );
        assert_eq!(refused, 0, "native refused {refused} features");
        assert!(
            equal < OUT_DIM,
            "native: all {OUT_DIM} rows bit-equal to scalar, so this workload cannot tell the \
             AVX2 path from the scalar one and the scalar arm proves nothing"
        );
    }
}
