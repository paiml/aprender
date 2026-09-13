// Section 13: Trueno Compute (F-TRUENO-*)
// =============================================================================

#[test]
fn f_trueno_001_runtime_backend_detection_works() {
    // F-TRUENO-001: Structural check — Backend enum has runtime detection methods
    let loading_path = crate_dir("aprender-core")
        .join("src")
        .join("loading")
        .join("mod.rs");
    let content = std::fs::read_to_string(&loading_path).expect("loading/mod.rs must exist");
    assert!(
        content.contains("Backend"),
        "F-TRUENO-001: Backend enum must exist in loading module"
    );
    assert!(
        content.contains("CpuSimd") || content.contains("Gpu") || content.contains("Cuda"),
        "F-TRUENO-001: Backend must have hardware-specific variants"
    );
}

#[test]
fn f_trueno_002_q4k_dequantize_matches_reference() {
    // F-TRUENO-002: Structural check — Q4K dequant function exists in trueno
    // #2522: `../trueno` was archived by APR-MONO; trueno is the [lib] name of
    // crates/aprender-compute. The sibling lookup could only ever miss, and the
    // miss branch `return`s, which the harness reports as `ok`.
    let trueno_dir = crate_dir("aprender-compute").join("src");
    let mut has_q4k_dequant = false;
    for path in collect_rs_files(&trueno_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        if content.contains("dequantize") && content.contains("q4") {
            has_q4k_dequant = true;
            break;
        }
    }
    assert!(
        has_q4k_dequant,
        "F-TRUENO-002: trueno must have Q4K dequantize function"
    );
}

#[test]
fn f_trueno_003_trueno_quant_used_by_both_projects() {
    // F-TRUENO-003: trueno-quant dependency in both Cargo.toml
    let aprender_toml = project_root().join("Cargo.toml");
    let aprender_content =
        std::fs::read_to_string(&aprender_toml).expect("aprender Cargo.toml readable");

    assert!(
        aprender_content.contains("trueno"),
        "F-TRUENO-003: aprender Cargo.toml must depend on trueno"
    );

    // Check realizar if it exists as a sibling
    let realizar_toml = project_root()
        .parent()
        .expect("parent")
        .join("realizar")
        .join("Cargo.toml");
    if realizar_toml.exists() {
        let realizar_content =
            std::fs::read_to_string(&realizar_toml).expect("realizar Cargo.toml readable");
        assert!(
            realizar_content.contains("trueno"),
            "F-TRUENO-003: realizar Cargo.toml must depend on trueno"
        );
    }
}

#[test]
fn f_trueno_004_cuda_ptx_compiles_and_runs() {
    // F-TRUENO-004: CUDA PTX compilation works and produces correct inference
    // Verified by running GPU inference end-to-end: trueno compiles PTX,
    // realizar uses it for fused matmul kernels, apr bench reports throughput.

    // 1. Structural: trueno-gpu has PTX compilation pipeline
    // #2522: `../trueno/trueno-gpu` was archived by APR-MONO; the PTX pipeline
    // is crates/aprender-gpu. The sibling miss branch returned `ok`.
    let trueno_ptx = crate_dir("aprender-gpu").join("src").join("ptx");
    assert!(
        trueno_ptx.join("mod.rs").exists(),
        "F-TRUENO-004: PTX module must exist at {}",
        trueno_ptx.display()
    );

    // 2. Verify CUDA hardware is available
    let gpu_available = Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !gpu_available {
        eprintln!("SKIP: no NVIDIA GPU available");
        return;
    }

    // 3. Runtime: GPU inference works (proves PTX compiled and ran)
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (_ok, stdout, stderr) = run_apr(&[
        "bench",
        gguf.to_str().unwrap(),
        "--iterations",
        "1",
        "--warmup",
        "0",
        "--max-tokens",
        "5",
        "--fast",
    ]);
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("GPU") || combined.contains("CUDA"),
        "F-TRUENO-004: bench --fast must use CUDA GPU. output: {}",
        combined
    );
    let tps: f64 = combined
        .lines()
        .find(|l| l.contains("Throughput:") && l.contains("tok/s"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    assert!(
        tps > 10.0,
        "F-TRUENO-004: CUDA inference must produce >10 tok/s (got {:.1})",
        tps
    );
}

#[test]
fn f_trueno_005_jidoka_guard_catches_nan() {
    // F-TRUENO-005: Jidoka guard types exist in trueno
    // #2522: `../trueno` was archived by APR-MONO; trueno is the [lib] name of
    // crates/aprender-compute. The sibling lookup could only ever miss, and the
    // miss branch `return`s, which the harness reports as `ok`.
    let trueno_dir = crate_dir("aprender-compute").join("src");

    // Search for JidokaGuard in trueno source
    let mut found_jidoka = false;
    for path in collect_rs_files(&trueno_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        if content.contains("JidokaGuard") || content.contains("Jidoka") {
            found_jidoka = true;
            break;
        }
    }

    assert!(
        found_jidoka,
        "F-TRUENO-005: trueno must have Jidoka guard types"
    );
}

#[test]
fn f_trueno_006_gpu_threshold_prevents_small_dispatch() {
    // F-TRUENO-006: Structural check — GPU threshold logic exists
    // #2522: `../trueno` was archived by APR-MONO; trueno is the [lib] name of
    // crates/aprender-compute. The sibling lookup could only ever miss, and the
    // miss branch `return`s, which the harness reports as `ok`.
    let trueno_dir = crate_dir("aprender-compute").join("src");
    let mut has_threshold = false;
    for path in collect_rs_files(&trueno_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        if content.contains("threshold") && (content.contains("gpu") || content.contains("Gpu")) {
            has_threshold = true;
            break;
        }
    }
    assert!(
        has_threshold,
        "F-TRUENO-006: trueno must have GPU dispatch threshold logic"
    );
}

#[test]
fn f_trueno_007_row_col_major_kernels_exist_separately() {
    // F-TRUENO-007: trueno provides BOTH row-major and col-major Q4K kernels
    // Structural check: verify both functions exist in trueno source
    // #2522: this looked for a SIBLING `../trueno` checkout, archived by
    // APR-MONO. It always missed, and its fallback asserted only that the string
    // "trueno" appears in the root Cargo.toml -- which is true of a workspace
    // that ships no Q4K kernel at all. trueno is `crates/aprender-compute`.
    let trueno_q4k_dir = crate_dir("aprender-compute")
        .join("src")
        .join("backends")
        .join("q4k");
    assert!(
        trueno_q4k_dir.is_dir(),
        "F-TRUENO-007: no q4k backend at {}",
        trueno_q4k_dir.display()
    );

    let mut has_row = false;
    let mut has_col = false;
    for path in collect_rs_files(&trueno_q4k_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for line in content.lines() {
            let trimmed = line.trim();
            has_row |= is_row_major_q4k_kernel(trimmed);
            has_col |= trimmed.contains("colmajor") && trimmed.contains("fn ");
        }
        // Module-level re-exports also count as "the kernel exists here".
        has_col |= content.contains("pub use colmajor::");
    }

    assert!(
        has_row,
        "F-TRUENO-007: trueno must have row-major Q4K kernel"
    );
    assert!(
        has_col,
        "F-TRUENO-007: trueno must have col-major Q4K kernel (for GGML compat)"
    );
}

/// A row-major Q4K matmul declaration (i.e. not one of the colmajor variants).
fn is_row_major_q4k_kernel(trimmed: &str) -> bool {
    let declares = trimmed.contains("fn matmul_q4k_f32(")
        || trimmed.contains("fn matmul_q4k_f32_scalar(")
        || trimmed.contains("fn matmul_q4k_f32_dispatch(");
    declares && !trimmed.contains("colmajor")
}

#[test]
fn f_trueno_008_wgsl_matmul_shader_correct() {
    // F-TRUENO-008: WGSL matmul shader exists and has correct structure
    // The wgpu backend uses WGSL shaders for cross-platform GPU compute.
    // Runtime execution verified by GPU inference tests (F-PERF-003, F-TRUENO-004).
    // #2522: `../trueno` was archived by APR-MONO, and `shaders.rs` became the
    // `shaders/` directory. Both misses returned `ok`.
    let trueno_dir = crate_dir("aprender-compute");
    let shaders_dir = trueno_dir
        .join("src")
        .join("backends")
        .join("gpu")
        .join("shaders");
    let content = wgsl_shader_corpus(&shaders_dir);

    // Verify matmul shader exists with correct WGSL structure
    assert!(
        content.contains("@compute") || content.contains("@workgroup_size"),
        "F-TRUENO-008: WGSL shader must have @compute or @workgroup_size attribute"
    );
    assert!(
        content.contains("fn main") || content.contains("fn matmul"),
        "F-TRUENO-008: WGSL shader must have main or matmul entry point"
    );
    assert!(
        content.contains("storage") || content.contains("@group"),
        "F-TRUENO-008: WGSL shader must use storage buffers or binding groups"
    );

    // Verify wgpu dependency exists in trueno
    let cargo_toml = trueno_dir.join("Cargo.toml");
    let toml_content = std::fs::read_to_string(&cargo_toml).expect("trueno Cargo.toml");
    assert!(
        toml_content.contains("wgpu"),
        "F-TRUENO-008: trueno must depend on wgpu for WGSL execution"
    );
}

// =============================================================================

/// Every WGSL shader source in the gpu shaders module, concatenated.
///
/// `backends/gpu/shaders.rs` became `backends/gpu/shaders/` at some point; the
/// gate read the file and skipped when it was missing, so it never once looked
/// at a shader (#2522).
fn wgsl_shader_corpus(shaders_dir: &Path) -> String {
    assert!(
        shaders_dir.is_dir(),
        "F-TRUENO-008: no gpu shaders module at {}",
        shaders_dir.display()
    );
    let files = collect_rs_files(shaders_dir);
    assert!(
        !files.is_empty(),
        "F-TRUENO-008: gpu shaders module at {} holds no sources",
        shaders_dir.display()
    );
    files
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect::<Vec<_>>()
        .join("\n")
}
