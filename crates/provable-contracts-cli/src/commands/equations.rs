use std::path::Path;

use provable_contracts::latex::{latex_escape, math_to_latex};
use provable_contracts::schema::{Contract, Equation, parse_contract};

/// Supported output formats for equation rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain-text tabular output
    Text,
    /// LaTeX document fragment with equation environments
    Latex,
    /// NVIDIA PTX kernel stub with equation comments
    Ptx,
    /// x86-64 assembly stub with SIMD register setup
    Asm,
}

impl OutputFormat {
    /// Parse a format name string into an `OutputFormat` variant
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(Self::Text),
            "latex" => Ok(Self::Latex),
            "ptx" => Ok(Self::Ptx),
            "asm" => Ok(Self::Asm),
            other => Err(format!(
                "unknown format '{other}', expected 'text', 'latex', 'ptx', or 'asm'"
            )),
        }
    }
}

/// Execute the equations command, rendering contract equations in the given format
pub fn run(path: &Path, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    match format {
        OutputFormat::Text => render_text(name, &contract.equations),
        OutputFormat::Latex => render_latex(name, &contract.equations),
        OutputFormat::Ptx => render_ptx(name, &contract),
        OutputFormat::Asm => render_asm(name, &contract),
    }

    Ok(())
}

/// Render equations as human-readable plain text
fn render_text(name: &str, equations: &std::collections::BTreeMap<String, Equation>) {
    println!("Equations for {name}");
    println!("{}", "=".repeat(40 + name.len()));
    println!();

    for (id, eq) in equations {
        println!("  {id}");
        println!("    formula:  {}", eq.formula);
        if let Some(ref dom) = eq.domain {
            println!("    domain:   {dom}");
        }
        if let Some(ref cod) = eq.codomain {
            println!("    codomain: {cod}");
        }
        if !eq.invariants.is_empty() {
            println!("    invariants:");
            for inv in &eq.invariants {
                println!("      - {inv}");
            }
        }
        println!();
    }
}

/// Render equations as LaTeX sections with math environments
fn render_latex(name: &str, equations: &std::collections::BTreeMap<String, Equation>) {
    let escaped_name = latex_escape(name);
    println!("% Equations for {name}");
    println!("\\section{{Equations: {escaped_name}}}");
    println!();

    for (id, eq) in equations {
        let escaped_id = latex_escape(id);
        let latex_formula = math_to_latex(&eq.formula);

        println!("\\subsection{{{escaped_id}}}");
        println!("\\label{{eq:{id}}}");
        println!();
        println!("\\begin{{equation}}");
        println!("  {latex_formula}");
        println!("\\end{{equation}}");
        println!();

        if let Some(ref dom) = eq.domain {
            println!("\\textbf{{Domain:}} ${}$", math_to_latex(dom));
            println!();
        }

        if let Some(ref cod) = eq.codomain {
            println!("\\textbf{{Codomain:}} ${}$", math_to_latex(cod));
            println!();
        }

        if !eq.invariants.is_empty() {
            println!("\\textbf{{Invariants:}}");
            println!("\\begin{{itemize}}");
            for inv in &eq.invariants {
                println!("  \\item ${}$", math_to_latex(inv));
            }
            println!("\\end{{itemize}}");
            println!();
        }
    }
}

/// Render a PTX kernel stub with equation header comments
fn render_ptx(name: &str, contract: &Contract) {
    let kernel = kernel_name(name);
    render_header_comment("PTX kernel stub", name, contract);
    println!(".version 8.5");
    println!(".target sm_90");
    println!(".address_size 64");
    println!();
    println!(".visible .entry {kernel}(");
    println!("    .param .u64 input,");
    println!("    .param .u64 output,");
    println!("    .param .u32 n");
    println!(")");
    println!("{{");
    println!("    .reg .u32 %tid, %n;");
    println!("    .reg .u64 %in_ptr, %out_ptr;");
    println!("    .reg .f32 %val, %acc;");
    println!();
    println!("    ld.param.u64 %in_ptr, [input];");
    println!("    ld.param.u64 %out_ptr, [output];");
    println!("    ld.param.u32 %n, [n];");
    println!();
    println!("    mov.u32 %tid, %ctaid.x;");
    println!("    mad.lo.u32 %tid, %tid, %ntid.x, %tid.x;");
    println!();
    render_body_comments(contract);
    println!("    ret;");
    println!("}}");
}

/// Render an x86-64 assembly stub with SIMD register initialization
fn render_asm(name: &str, contract: &Contract) {
    let kernel = kernel_name(name);
    let isa = detect_simd_isa(contract);
    render_header_comment(&format!("x86-64 {} stub", isa.label), name, contract);
    println!(".intel_syntax noprefix");
    println!(".text");
    println!(".globl {kernel}_{}", isa.suffix);
    println!(".p2align 4");
    println!();
    println!("{kernel}_{}:", isa.suffix);
    println!("    push rbp");
    println!("    mov rbp, rsp");
    println!("    // rdi = input ptr, rsi = output ptr, edx = n");
    println!();
    if isa.width > 0 {
        let r = isa.reg_prefix;
        println!("    // {} registers: {r} x {}", isa.label, isa.reg_count);
        for i in 0..std::cmp::min(isa.reg_count, 4) {
            println!("    vxorps {r}{i}, {r}{i}, {r}{i}");
        }
        println!();
    }
    render_body_comments(contract);
    println!("    pop rbp");
    println!("    ret");
}

/// Emit a block comment with kernel description and equation summaries
fn render_header_comment(label: &str, name: &str, contract: &Contract) {
    println!("//");
    println!("// {label}: {name}");
    println!("// {}", contract.metadata.description);
    for (id, eq) in &contract.equations {
        println!("// Equation {id}: {}", eq.formula);
    }
    println!("//");
    println!();
}

/// Emit inline comments for kernel phases or equation formulas
fn render_body_comments(contract: &Contract) {
    if let Some(ref ks) = contract.kernel_structure {
        for (i, phase) in ks.phases.iter().enumerate() {
            println!("    // Phase {}: {}", i + 1, phase.name);
            println!("    // {}", phase.description);
            if let Some(ref inv) = phase.invariant {
                println!("    // Invariant: {inv}");
            }
            println!();
        }
    } else {
        for (id, eq) in &contract.equations {
            println!("    // Equation: {id}");
            println!("    // {}", eq.formula);
            println!();
        }
    }
    if !contract.proof_obligations.is_empty() {
        println!("    // Proof obligations:");
        for ob in &contract.proof_obligations {
            println!("    //   [{}] {}", ob.obligation_type, ob.property);
        }
        println!();
    }
}

/// SIMD instruction set descriptor for assembly generation
struct SimdIsa {
    /// Human-readable ISA name (e.g. "AVX-512")
    label: &'static str,
    /// Function name suffix (e.g. "avx512")
    suffix: &'static str,
    /// Register prefix for the ISA (e.g. "zmm")
    reg_prefix: &'static str,
    /// Number of available SIMD registers
    reg_count: u32,
    /// Bit width of SIMD registers (0 for scalar)
    width: u32,
}

/// Detect the highest SIMD ISA level from the contract's `simd_dispatch` map
fn detect_simd_isa(contract: &Contract) -> SimdIsa {
    let has = |pat: &str| {
        contract
            .simd_dispatch
            .values()
            .any(|m| m.keys().any(|k| k.contains(pat)))
    };
    if has("avx512") || has("512") {
        SimdIsa {
            label: "AVX-512",
            suffix: "avx512",
            reg_prefix: "zmm",
            reg_count: 32,
            width: 512,
        }
    } else if has("avx2") {
        SimdIsa {
            label: "AVX2",
            suffix: "avx2",
            reg_prefix: "ymm",
            reg_count: 16,
            width: 256,
        }
    } else if !contract.simd_dispatch.is_empty() {
        SimdIsa {
            label: "SSE4.1",
            suffix: "sse41",
            reg_prefix: "xmm",
            reg_count: 16,
            width: 128,
        }
    } else {
        SimdIsa {
            label: "scalar",
            suffix: "scalar",
            reg_prefix: "xmm",
            reg_count: 0,
            width: 0,
        }
    }
}

/// Derive a kernel function name from the contract stem.
/// Strips version suffix and `-kernel`, converts hyphens to underscores.
fn kernel_name(contract_name: &str) -> String {
    let mut s = contract_name.to_string();
    if let Some(pos) = s.rfind("-v") {
        if s[pos + 2..].chars().all(|c| c.is_ascii_digit()) {
            s.truncate(pos);
        }
    }
    if let Some(stripped) = s.strip_suffix("-kernel") {
        s = stripped.to_string();
    }
    s.replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(OutputFormat::from_str("text").unwrap(), OutputFormat::Text);
        assert_eq!(
            OutputFormat::from_str("latex").unwrap(),
            OutputFormat::Latex
        );
        assert_eq!(OutputFormat::from_str("ptx").unwrap(), OutputFormat::Ptx);
        assert_eq!(OutputFormat::from_str("asm").unwrap(), OutputFormat::Asm);
        assert!(OutputFormat::from_str("json").is_err());
    }

    #[test]
    fn test_from_str_other_format_returns_descriptive_error() {
        // Exercises the `other =>` catch-all arm in from_str
        let other = "csv";
        let err = OutputFormat::from_str(other).unwrap_err();
        assert!(
            err.contains(other),
            "error should include the unrecognized format"
        );
        assert!(err.contains("unknown format"));
    }

    #[test]
    fn test_kernel_name() {
        assert_eq!(kernel_name("softmax-kernel-v1"), "softmax");
        assert_eq!(kernel_name("rmsnorm-kernel-v1"), "rmsnorm");
        assert_eq!(kernel_name("flash-attention-v1"), "flash_attention");
        assert_eq!(
            kernel_name("model-config-algebra-v1"),
            "model_config_algebra"
        );
        assert_eq!(kernel_name("silu-kernel-v2"), "silu");
    }

    #[test]
    fn test_detect_simd_isa_avx2() {
        use provable_contracts::schema::parse_contract_str;
        let yaml = "metadata:\n  version: '1.0'\n  description: test\n\
                     equations:\n  eq1:\n    formula: 'y = x'\n\
                     simd_dispatch:\n  k:\n    scalar: s\n    avx2: a\n";
        let contract = parse_contract_str(yaml).unwrap();
        let isa = detect_simd_isa(&contract);
        assert_eq!(isa.suffix, "avx2");
        assert_eq!(isa.width, 256);
    }
}
