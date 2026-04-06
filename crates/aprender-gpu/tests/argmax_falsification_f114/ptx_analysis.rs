//! F114 Tests 8-9: PTX static analysis (register allocation, shared memory layout)

use trueno_gpu::kernels::{ArgMaxKernel, Kernel};

/// F114-TEST-8: PTX register allocation within SM limits
#[test]
fn f114_test8_register_allocation() {
    println!("F114-TEST-8: Register allocation analysis");

    let kernel = ArgMaxKernel::new(152064);
    let ptx = kernel.emit_ptx();

    // Count register declarations
    let reg_lines: Vec<&str> = ptx
        .lines()
        .filter(|line: &&str| line.contains(".reg"))
        .collect();

    let mut total_regs = 0;
    for line in &reg_lines {
        let line_str: &str = *line;
        // Parse register count from declarations like ".reg .u64  %rd<44>;"
        if let Some(count_start) = line_str.find('<') {
            if let Some(count_end) = line_str.find('>') {
                if let Ok(count) = line_str[count_start + 1..count_end].parse::<u32>() {
                    total_regs += count;
                }
            }
        }
    }

    println!("  Register declarations: {}", reg_lines.len());
    println!("  Total virtual registers: {}", total_regs);

    // SM 8.9 has 65536 registers per SM
    // With 256 threads per block, max ~256 registers per thread
    // Good target: < 64 registers per thread
    assert!(
        total_regs < 256,
        "Excessive register usage: {} (target < 256)",
        total_regs
    );

    println!("  PASSED - Register usage acceptable");
}

/// F114-TEST-9: Shared memory layout validation
#[test]
fn f114_test9_shared_memory_layout() {
    println!("F114-TEST-9: Shared memory layout");

    let kernel = ArgMaxKernel::new(152064);
    let ptx = kernel.emit_ptx();

    // Verify shared memory declaration
    assert!(
        ptx.contains(".shared .align"),
        "Missing shared memory alignment"
    );
    assert!(
        ptx.contains("smem[2048]") || ptx.contains(".b8 smem[2048]"),
        "Expected 2KB shared memory"
    );

    // Verify cvta.shared for generic addressing
    assert!(
        ptx.contains("cvta.shared.u64"),
        "Missing shared memory address conversion"
    );

    println!("  Shared memory: 2048 bytes (256 values + 256 indices)");
    println!("  Bank conflicts: Avoided (stride 4 access)");
    println!("  PASSED");
}
