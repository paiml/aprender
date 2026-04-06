use super::super::*;

// =========================================================================
// PARITY-114: Barrier Safety Tests for Quantized Kernels
// =========================================================================

#[test]
fn test_q4k_ggml_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = QuantizeKernel::ggml(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);

    // Debug output: show what's detected
    if !result.is_safe {
        println!("Q4K GGML barrier_count: {}", result.barrier_count);
        println!("Q4K GGML exit_count: {}", result.exit_count);
        for v in &result.violations {
            println!(
                "Violation at line {}: {:?} - {}",
                v.line, v.kind, v.instruction
            );
        }
        // Print PTX around the violation
        for (i, line) in ptx.lines().enumerate() {
            let lineno = i + 1;
            if result
                .violations
                .iter()
                .any(|v| v.line.saturating_sub(5) <= lineno && lineno <= v.line.saturating_add(5))
            {
                println!("{:4}: {}", lineno, line);
            }
        }
    }

    assert!(
        result.is_safe,
        "Q4K GGML should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q5k_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q5KKernel::new(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q5K should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q6k_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q6KKernel::new(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q6K should be barrier-safe: {:?}",
        result.violations
    );
}
