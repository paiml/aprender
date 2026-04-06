//! Solver demonstration: LU, QR, SVD, Cholesky, TRSM, BLAS Level-3.
//!
//! ```sh
//! cargo run --example solver_demo -p trueno-solve
//! ```

use trueno_solve::{
    cholesky, f32_to_f16, gemm_ex, gemm_ex_epilogue, gemm_strided_batched, lu_factorize,
    qr_factorize, svd, symm, syr2k, syrk, trmm, trsm, Epilogue, Solver,
};
use trueno_solve::{DiagonalType, TriangularSide};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== trueno-solve: Full Solver Demo ===\n");

    // ── LU ─────────────────────────────────────────────────
    println!("--- LU Factorization ---");
    let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0_f32];
    let lu = lu_factorize(&a, 3)?;
    let x = lu.solve(&[4.0, 7.0, 10.0])?;
    println!("Ax=b: x = [{:.4}, {:.4}, {:.4}]", x[0], x[1], x[2]);

    // ── QR ─────────────────────────────────────────────────
    println!("\n--- QR Factorization ---");
    let qr = qr_factorize(&[1.0, 1.0, 1.0, 2.0, 1.0, 3.0_f32], 3, 2)?;
    let x_ls = qr.solve(&[1.0, 2.0, 3.0])?;
    println!("Least-squares: x = [{:.4}, {:.4}]", x_ls[0], x_ls[1]);

    // ── SVD ────────────────────────────────────────────────
    println!("\n--- SVD ---");
    let result = svd(&[3.0, 2.0, 2.0, 3.0_f32], 2, 2)?;
    println!("Singular values: [{:.4}, {:.4}]", result.sigma[0], result.sigma[1]);

    // ── Cholesky ───────────────────────────────────────────
    println!("\n--- Cholesky ---");
    let chol = cholesky(&[4.0, 2.0, 2.0, 3.0_f32], 2)?;
    let x_chol = chol.solve(&[8.0, 7.0])?;
    println!("x = [{:.4}, {:.4}]", x_chol[0], x_chol[1]);

    // ── TRSM ───────────────────────────────────────────────
    println!("\n--- TRSM (triangular solve) ---");
    let tri = [2.0, 0.0, 3.0, 4.0_f32]; // lower triangular
    let result = trsm(&tri, &[2.0, 11.0], 2, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;
    println!("Lower triangular solve: x = [{:.4}, {:.4}]", result.x[0], result.x[1]);

    // ── BLAS Level-3 ───────────────────────────────────────
    println!("\n--- BLAS Level-3 ---");

    // syrk: C = A·Aᵀ
    let a_syrk = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let mut c_syrk = [0.0_f32; 4];
    syrk(&a_syrk, &mut c_syrk, 2, 2, 1.0, 0.0)?;
    println!(
        "syrk(A·Aᵀ): [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_syrk[0], c_syrk[1], c_syrk[2], c_syrk[3]
    );

    // syr2k: C = A·Bᵀ + B·Aᵀ
    let b_syr2k = [5.0, 6.0, 7.0, 8.0_f32];
    let mut c_syr2k = [0.0_f32; 4];
    syr2k(&a_syrk, &b_syr2k, &mut c_syr2k, 2, 2, 1.0, 0.0)?;
    println!("syr2k: [{:.1}, {:.1}; {:.1}, {:.1}]", c_syr2k[0], c_syr2k[1], c_syr2k[2], c_syr2k[3]);

    // trmm: B = A·B (lower triangular)
    let a_tri = [2.0, 0.0, 3.0, 4.0_f32];
    let mut b_trmm = [1.0, 1.0_f32]; // 2×1
    trmm(&a_tri, &mut b_trmm, 2, 1, 1.0)?;
    println!("trmm(lower·[1,1]ᵀ): [{:.1}, {:.1}]", b_trmm[0], b_trmm[1]);

    // symm: C = A·B (symmetric A)
    let a_sym = [1.0, 2.0, 2.0, 3.0_f32];
    let b_sym = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c_sym = [0.0_f32; 4];
    symm(&a_sym, &b_sym, &mut c_sym, 2, 2, 1.0, 0.0)?;
    println!("symm: [{:.1}, {:.1}; {:.1}, {:.1}]", c_sym[0], c_sym[1], c_sym[2], c_sym[3]);

    // ── gemmEx (mixed-precision f16→f32) ───────────────────
    println!("\n--- gemmEx (mixed-precision) ---");
    let a_f16: Vec<u16> = [1.0, 2.0, 3.0, 4.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let b_f16: Vec<u16> = [5.0, 6.0, 7.0, 8.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let mut c_ex = [0.0_f32; 4];
    gemm_ex(&a_f16, &b_f16, &mut c_ex, 2, 2, 2, 1.0, 0.0)?;
    println!("f16 matmul: [{:.1}, {:.1}; {:.1}, {:.1}]", c_ex[0], c_ex[1], c_ex[2], c_ex[3]);

    // ── gemmStridedBatched ───────────────────────────────
    println!("\n--- gemmStridedBatched ---");
    let a_batch = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0_f32]; // 2 batches of 2×2
    let b_batch = [3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0_f32];
    let mut c_batch = [0.0_f32; 8];
    gemm_strided_batched(&a_batch, 4, &b_batch, 4, &mut c_batch, 4, 2, 2, 2, 2, 1.0, 0.0)?;
    println!(
        "Batch 0: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_batch[0], c_batch[1], c_batch[2], c_batch[3]
    );
    println!(
        "Batch 1: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_batch[4], c_batch[5], c_batch[6], c_batch[7]
    );

    // ── Solver trait (dynamic dispatch) ────────────────────
    println!("\n--- Solver trait (unified interface) ---");
    let lu2 = lu_factorize(&[4.0, 1.0, 1.0, 3.0_f32], 2)?;
    let solver: &dyn Solver = &lu2;
    let x_dyn = solver.solve(&[5.0, 7.0])?;
    println!("dyn Solver (LU): dim={}, x=[{:.4}, {:.4}]", solver.dimension(), x_dyn[0], x_dyn[1]);

    let chol2 = cholesky(&[4.0, 2.0, 2.0, 3.0_f32], 2)?;
    let solver2: &dyn Solver = &chol2;
    let x_dyn2 = solver2.solve(&[8.0, 7.0])?;
    println!(
        "dyn Solver (Cholesky): dim={}, x=[{:.4}, {:.4}]",
        solver2.dimension(),
        x_dyn2[0],
        x_dyn2[1]
    );

    // ── gemmEx with epilogue fusion ─────────────────────────
    println!("\n--- gemmEx with epilogue fusion ---");
    let a_ep: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let b_ep: Vec<u16> = [-3.0, 4.0, 5.0, -6.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let mut c_relu = [0.0_f32; 4];
    gemm_ex_epilogue(&a_ep, &b_ep, &mut c_relu, 2, 2, 2, 1.0, 0.0, Epilogue::Relu, None)?;
    println!(
        "ReLU epilogue: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_relu[0], c_relu[1], c_relu[2], c_relu[3]
    );

    let mut c_bias = [0.0_f32; 4];
    let bias = [100.0, 200.0_f32];
    gemm_ex_epilogue(&a_ep, &b_ep, &mut c_bias, 2, 2, 2, 1.0, 0.0, Epilogue::Bias, Some(&bias))?;
    println!(
        "Bias epilogue: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_bias[0], c_bias[1], c_bias[2], c_bias[3]
    );

    println!("\n=== All solver demos passed ===");
    Ok(())
}
