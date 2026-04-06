//! Basic kernel creation, dimensions, PTX generation, and barrier safety tests.

use super::*;

#[test]
fn test_f051_kernel_creation() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    assert_eq!(kernel.batch_size(), 1000);
    assert_eq!(kernel.name(), "lz4_compress_warp");
}

#[test]
fn test_f051_grid_dimensions() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    let (gx, gy, gz) = kernel.grid_dim();
    assert_eq!(gx, 250);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

#[test]
fn test_f051_block_dimensions() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    let (bx, by, bz) = kernel.block_dim();
    assert_eq!(bx, 128);
    assert_eq!(by, 1);
    assert_eq!(bz, 1);
}

#[test]
fn test_f052_shared_memory_size() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let smem = kernel.shared_memory_bytes();
    assert!(smem > 0);
    assert!(smem <= 100 * 1024);
}

#[test]
fn test_f053_ptx_generation_valid() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"), "Missing PTX version");
    assert!(ptx.contains(".target"), "Missing PTX target");
    assert!(ptx.contains(".entry"), "Missing entry point");
}

#[test]
fn test_f053_ptx_has_parameters() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("input_batch"));
    assert!(ptx.contains("output_batch"));
    assert!(ptx.contains("output_sizes"));
    assert!(ptx.contains("batch_size"));
}

#[test]
fn test_f053_ptx_has_shared_memory() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_f054_barrier_safety() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "LZ4 kernel should be barrier-safe: {:?}", result.violations);
}

#[test]
fn test_f055_kernel_name_deterministic() {
    let k1 = Lz4WarpCompressKernel::new(100);
    let k2 = Lz4WarpCompressKernel::new(100);
    assert_eq!(k1.name(), k2.name());
}

#[test]
fn test_f056_ptx_has_barrier_sync() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_f059_grid_covers_all_pages() {
    for batch_size in [1, 4, 5, 100, 1000, 18432] {
        let kernel = Lz4WarpCompressKernel::new(batch_size);
        let (gx, _, _) = kernel.grid_dim();
        let (bx, _, _) = kernel.block_dim();
        let warps_per_block = bx / 32;
        let total_warps = gx * warps_per_block;
        assert!(total_warps >= batch_size);
    }
}

#[test]
fn test_f060_module_emission() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let module = kernel.as_module();
    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_f061_ptx_validates_with_ptxas() {
    use std::io::Write;
    use std::process::Command;

    // Check if ptxas is available
    let ptxas_check = Command::new("which").arg("ptxas").output();
    if ptxas_check.is_err() || !ptxas_check.expect("test").status.success() {
        eprintln!("ptxas not available, skipping validation");
        return;
    }

    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Write PTX to temp file
    let mut tmpfile = std::env::temp_dir();
    tmpfile.push("lz4_compress_warp.ptx");
    let mut f = std::fs::File::create(&tmpfile).expect("Failed to create temp file");
    f.write_all(ptx.as_bytes()).expect("Failed to write PTX");

    // Validate with ptxas
    let output = Command::new("ptxas")
        .args(["-arch=sm_89", tmpfile.to_str().expect("test"), "-o", "/dev/null"])
        .output()
        .expect("Failed to run ptxas");

    // Clean up
    let _ = std::fs::remove_file(&tmpfile);

    assert!(
        output.status.success(),
        "ptxas validation failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
