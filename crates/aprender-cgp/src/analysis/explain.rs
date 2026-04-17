//! `cgp explain` — Static code analysis for PTX, SIMD assembly, and WGSL shaders.
//! Spec section 2.7: wraps trueno-explain or performs inline analysis.
//! Detects register pressure, instruction mix, and common performance pitfalls.

use anyhow::Result;
use std::path::Path;

/// Analyze a PTX file for performance-relevant patterns.
pub fn analyze_ptx(source: &str) -> PtxAnalysis {
    let lines: Vec<&str> = source.lines().collect();
    let total_instructions = count_ptx_instructions(&lines);

    let mut counters = PtxCounters::default();
    for line in &lines {
        classify_ptx_line(line, &mut counters);
    }
    append_ptx_pressure_warnings(&mut counters);

    let compute_memory_ratio = if counters.memory_ops > 0 {
        f64::from(counters.compute_ops) / f64::from(counters.memory_ops)
    } else {
        f64::INFINITY
    };

    PtxAnalysis {
        total_instructions,
        memory_ops: counters.memory_ops,
        compute_ops: counters.compute_ops,
        control_ops: counters.control_ops,
        sync_ops: counters.sync_ops,
        shared_ops: counters.shared_ops,
        registers_declared: counters.registers_declared,
        has_wmma: counters.has_wmma,
        has_fma: counters.has_fma,
        compute_memory_ratio,
        warnings: counters.warnings,
    }
}

/// Mutable accumulator used while walking PTX lines.
#[derive(Default)]
struct PtxCounters {
    memory_ops: u32,
    compute_ops: u32,
    control_ops: u32,
    sync_ops: u32,
    shared_ops: u32,
    registers_declared: u32,
    has_wmma: bool,
    has_fma: bool,
    warnings: Vec<String>,
}

/// Count non-trivial PTX instructions (skip blanks, comments, directives, braces).
fn count_ptx_instructions(lines: &[&str]) -> u32 {
    u32::try_from(
        lines
            .iter()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with("//")
                    && !t.starts_with('.')
                    && !t.starts_with('{')
                    && !t.starts_with('}')
            })
            .count(),
    )
    .unwrap_or(u32::MAX)
}

/// Dispatch a single PTX line to all per-category counters.
fn classify_ptx_line(line: &str, counters: &mut PtxCounters) {
    let trimmed = line.trim();
    tally_register_declaration(trimmed, counters);
    tally_memory_op(trimmed, counters);
    tally_compute_op(trimmed, counters);
    tally_control_op(trimmed, counters);
    if trimmed.starts_with("bar.") {
        counters.sync_ops += 1;
    }
    if trimmed.contains("wmma.") || trimmed.contains("mma.") {
        counters.has_wmma = true;
    }
}

/// Parse `.reg .TYPE %name<N>;` and accumulate the declared register count.
fn tally_register_declaration(trimmed: &str, counters: &mut PtxCounters) {
    if !trimmed.starts_with(".reg") {
        return;
    }
    let Some(count_str) = trimmed.split('<').nth(1).and_then(|s| s.split('>').next()) else {
        return;
    };
    let Ok(count) = count_str.parse::<u32>() else {
        return;
    };
    counters.registers_declared += count;
}

/// Track `ld.*` / `st.*` operations, flagging shared-memory traffic separately.
fn tally_memory_op(trimmed: &str, counters: &mut PtxCounters) {
    if !(trimmed.starts_with("ld.") || trimmed.starts_with("st.")) {
        return;
    }
    counters.memory_ops += 1;
    if trimmed.contains(".shared") {
        counters.shared_ops += 1;
    }
}

/// Track arithmetic ops and record whether FMA-class instructions are present.
fn tally_compute_op(trimmed: &str, counters: &mut PtxCounters) {
    let is_compute = trimmed.starts_with("add.")
        || trimmed.starts_with("mul.")
        || trimmed.starts_with("mad.")
        || trimmed.starts_with("fma.");
    if !is_compute {
        return;
    }
    counters.compute_ops += 1;
    if trimmed.starts_with("fma.") || trimmed.starts_with("mad.") {
        counters.has_fma = true;
    }
}

/// Track branches and predicated branches; warn on data-dependent divergence.
fn tally_control_op(trimmed: &str, counters: &mut PtxCounters) {
    if !(trimmed.starts_with("bra") || trimmed.starts_with('@')) {
        return;
    }
    counters.control_ops += 1;
    if trimmed.starts_with("@%p") && trimmed.contains("bra") {
        counters
            .warnings
            .push("Data-dependent branch may cause warp divergence".to_string());
    }
}

/// Add register-pressure and sync-overhead warnings after counting is complete.
fn append_ptx_pressure_warnings(counters: &mut PtxCounters) {
    if counters.registers_declared > 128 {
        counters.warnings.push(format!(
            "High register usage ({}) may limit occupancy",
            counters.registers_declared
        ));
    }
    if counters.sync_ops > 2 {
        counters.warnings.push(format!(
            "{} barrier syncs — review if all are necessary",
            counters.sync_ops
        ));
    }
}

/// Result of PTX static analysis.
#[derive(Debug)]
pub struct PtxAnalysis {
    pub total_instructions: u32,
    pub memory_ops: u32,
    pub compute_ops: u32,
    pub control_ops: u32,
    pub sync_ops: u32,
    pub shared_ops: u32,
    pub registers_declared: u32,
    pub has_wmma: bool,
    pub has_fma: bool,
    pub compute_memory_ratio: f64,
    pub warnings: Vec<String>,
}

/// Analyze a WGSL shader for compute patterns.
pub fn analyze_wgsl(source: &str) -> WgslAnalysis {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = u32::try_from(lines.len()).unwrap_or(u32::MAX);

    let mut counters = WgslCounters::default();
    for line in &lines {
        classify_wgsl_line(line, &mut counters);
    }
    append_wgsl_workgroup_warnings(&mut counters);

    WgslAnalysis {
        total_lines,
        workgroup_size: counters.workgroup_size,
        bindings: counters.bindings,
        has_atomics: counters.has_atomics,
        has_shared: counters.has_shared,
        warnings: counters.warnings,
    }
}

/// Mutable accumulator used while walking WGSL lines.
#[derive(Default)]
struct WgslCounters {
    workgroup_size: Option<String>,
    bindings: u32,
    has_atomics: bool,
    has_shared: bool,
    warnings: Vec<String>,
}

/// Dispatch a single WGSL line to each feature detector.
fn classify_wgsl_line(line: &str, counters: &mut WgslCounters) {
    let trimmed = line.trim();
    capture_workgroup_size(trimmed, counters);
    if trimmed.contains("@binding") {
        counters.bindings += 1;
    }
    if trimmed.contains("atomicAdd") || trimmed.contains("atomicStore") {
        counters.has_atomics = true;
    }
    if trimmed.contains("var<workgroup>") {
        counters.has_shared = true;
    }
}

/// Extract the literal arguments of `@workgroup_size(...)` when present.
fn capture_workgroup_size(trimmed: &str, counters: &mut WgslCounters) {
    if !trimmed.contains("@workgroup_size") {
        return;
    }
    let Some(start) = trimmed.find('(').map(|i| i + 1) else {
        return;
    };
    let Some(end) = trimmed.find(')') else {
        return;
    };
    counters.workgroup_size = Some(trimmed[start..end].to_string());
}

/// Emit warnings when the workgroup size is too small or too large for typical GPUs.
fn append_wgsl_workgroup_warnings(counters: &mut WgslCounters) {
    let Some(ref ws) = counters.workgroup_size else {
        return;
    };
    let total: u32 = ws
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .product();
    if total < 64 {
        counters.warnings.push(format!(
            "Workgroup size ({ws}) = {total} threads — consider >=64 for GPU occupancy"
        ));
    }
    if total > 1024 {
        counters.warnings.push(format!(
            "Workgroup size ({ws}) = {total} threads — exceeds common hardware limit (1024)"
        ));
    }
}

/// Result of WGSL static analysis.
#[derive(Debug)]
pub struct WgslAnalysis {
    pub total_lines: u32,
    pub workgroup_size: Option<String>,
    pub bindings: u32,
    pub has_atomics: bool,
    pub has_shared: bool,
    pub warnings: Vec<String>,
}

/// Run the explain command.
pub fn run_explain(target: &str, kernel: Option<&str>) -> Result<()> {
    println!("\n=== CGP Explain: {target} ===\n");

    match target {
        "ptx" => {
            let kernel_name = kernel.unwrap_or("*");
            println!("  Target: PTX (CUDA assembly)");
            println!("  Kernel: {kernel_name}");

            // Try to find PTX files
            let ptx_path = find_ptx_file(kernel_name);
            match ptx_path {
                Some(path) => {
                    let source = std::fs::read_to_string(&path)?;
                    let analysis = analyze_ptx(&source);
                    println!("  File: {path}");
                    render_ptx_analysis(&analysis);
                }
                None => {
                    println!("  No PTX file found for kernel '{kernel_name}'.");
                    println!("  Generate with: cargo build -p trueno-gpu --features cuda");
                    println!("  Or provide path: cgp explain ptx --kernel path/to/kernel.ptx");
                }
            }
        }
        "wgsl" | "shader" => {
            let shader_path = kernel.unwrap_or("*.wgsl");
            println!("  Target: WGSL (WebGPU shader)");

            if Path::new(shader_path).exists() {
                let source = std::fs::read_to_string(shader_path)?;
                let analysis = analyze_wgsl(&source);
                println!("  File: {shader_path}");
                render_wgsl_analysis(&analysis);
            } else {
                println!("  Shader file not found: {shader_path}");
                println!(
                    "  Provide path: cgp explain wgsl --kernel src/backends/gpu/shaders/gemm.wgsl"
                );
            }
        }
        "simd" => {
            println!("  Target: SIMD (x86/ARM assembly analysis)");
            println!("  Analysis: instruction mix, vectorization rate, register usage");
            println!(
                "  Use: cgp profile simd --function <fn> --arch avx2 for runtime SIMD analysis"
            );
        }
        _ => {
            println!("  Unknown target: {target}");
            println!("  Supported: ptx, wgsl, simd");
        }
    }

    println!();
    Ok(())
}

/// Render PTX analysis results.
fn render_ptx_analysis(analysis: &PtxAnalysis) {
    println!("\n  Instruction Mix:");
    println!("    Total instructions: {}", analysis.total_instructions);
    println!("    Compute ops:       {}", analysis.compute_ops);
    println!("    Memory ops:        {}", analysis.memory_ops);
    println!("    Control flow:      {}", analysis.control_ops);
    println!("    Sync barriers:     {}", analysis.sync_ops);
    println!("    Shared memory ops: {}", analysis.shared_ops);

    println!(
        "\n  Compute/Memory Ratio: {:.2}",
        analysis.compute_memory_ratio
    );
    if analysis.compute_memory_ratio < 1.0 {
        println!("    Status: MEMORY-INTENSIVE (more loads than compute)");
    } else if analysis.compute_memory_ratio > 4.0 {
        println!("    Status: COMPUTE-INTENSIVE (good arithmetic density)");
    } else {
        println!("    Status: BALANCED");
    }

    println!("\n  Features:");
    println!("    Registers declared: {}", analysis.registers_declared);
    println!(
        "    Tensor cores (WMMA/MMA): {}",
        if analysis.has_wmma { "YES" } else { "no" }
    );
    println!(
        "    FMA instructions: {}",
        if analysis.has_fma { "YES" } else { "no" }
    );

    if !analysis.warnings.is_empty() {
        println!("\n  Warnings:");
        for w in &analysis.warnings {
            println!("    \x1b[33m[WARN]\x1b[0m {w}");
        }
    }
}

/// Render WGSL analysis results.
fn render_wgsl_analysis(analysis: &WgslAnalysis) {
    println!("\n  Shader Info:");
    println!("    Lines: {}", analysis.total_lines);
    println!(
        "    Workgroup size: {}",
        analysis
            .workgroup_size
            .as_deref()
            .unwrap_or("not specified")
    );
    println!("    Bindings: {}", analysis.bindings);
    println!(
        "    Atomics: {}",
        if analysis.has_atomics { "YES" } else { "no" }
    );
    println!(
        "    Shared memory: {}",
        if analysis.has_shared { "YES" } else { "no" }
    );

    if !analysis.warnings.is_empty() {
        println!("\n  Warnings:");
        for w in &analysis.warnings {
            println!("    \x1b[33m[WARN]\x1b[0m {w}");
        }
    }
}

/// Find a PTX file for a given kernel name.
fn find_ptx_file(kernel_name: &str) -> Option<String> {
    // Check if kernel_name is already a path
    if Path::new(kernel_name).exists() {
        return Some(kernel_name.to_string());
    }

    // Search common locations
    let search_dirs = ["src/backends/gpu/kernels", "trueno-gpu/src", "."];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".ptx")
                    && (kernel_name == "*" || name_str.contains(kernel_name))
                {
                    return Some(entry.path().display().to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_ptx_basic() {
        let ptx = r#"
.version 8.0
.target sm_89
.entry gemm_kernel {
    .reg .f32 %f<32>;
    .reg .pred %p<4>;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    fma.rn.f32 %f3, %f1, %f2, %f0;
    st.global.f32 [%rd3], %f3;
    bar.sync 0;
}
"#;
        let analysis = analyze_ptx(ptx);
        assert!(analysis.memory_ops >= 3); // 2 loads + 1 store
        assert!(analysis.compute_ops >= 1); // fma
        assert!(analysis.has_fma);
        assert!(analysis.sync_ops >= 1);
        assert!(analysis.registers_declared >= 32);
    }

    #[test]
    fn test_analyze_ptx_wmma() {
        let ptx = "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16 {a}, {b}, {c};";
        let analysis = analyze_ptx(ptx);
        assert!(analysis.has_wmma);
    }

    #[test]
    fn test_analyze_ptx_high_register_warning() {
        let ptx = ".reg .f32 %f<256>;";
        let analysis = analyze_ptx(ptx);
        assert!(analysis.registers_declared >= 256);
        assert!(!analysis.warnings.is_empty());
    }

    #[test]
    fn test_analyze_wgsl_basic() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    b[gid.x] = a[gid.x] * 2.0;
}
"#;
        let analysis = analyze_wgsl(wgsl);
        assert_eq!(analysis.bindings, 2);
        assert_eq!(analysis.workgroup_size.as_deref(), Some("256, 1, 1"));
        assert!(!analysis.has_atomics);
    }

    #[test]
    fn test_analyze_wgsl_small_workgroup() {
        let wgsl = "@compute @workgroup_size(8, 1, 1)\nfn main() {}";
        let analysis = analyze_wgsl(wgsl);
        assert!(!analysis.warnings.is_empty());
    }

    #[test]
    fn test_run_explain_ptx() {
        let result = run_explain("ptx", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_explain_simd() {
        let result = run_explain("simd", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_explain_unknown() {
        let result = run_explain("unknown_target", None);
        assert!(result.is_ok());
    }
}
