//! Sparse matrix demonstration — all formats and operations.
//!
//! ```bash
//! cargo run -p trueno-sparse --example sparse_spmv
//! ```

use trueno_sparse::{BsrMatrix, CooMatrix, CsrMatrix, SellMatrix, SparseOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Trueno Sparse: Full Demo ===\n");

    // ── COO → CSR ──────────────────────────────────────────
    let coo = CooMatrix::new(
        4,
        4,
        vec![0, 0, 1, 2, 2, 2, 3],
        vec![0, 2, 1, 0, 2, 3, 3],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )?;
    let csr = CsrMatrix::from_coo(&coo);
    println!(
        "CSR: {}×{}, {} nnz, avg {:.2} nnz/row",
        csr.rows(),
        csr.cols(),
        csr.nnz(),
        csr.avg_nnz_per_row()
    );

    // ── SpMV ───────────────────────────────────────────────
    let x = vec![1.0, 2.0, 3.0, 4.0_f32];
    let mut y = vec![0.0_f32; 4];
    csr.spmv(1.0, &x, 0.0, &mut y)?;
    println!("SpMV y = A*x: {y:?}");

    // ── SpMM ───────────────────────────────────────────────
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0_f32]; // 4×2 identity-ish
    let mut c = vec![0.0_f32; 8]; // 4×2
    csr.spmm(1.0, &b, 2, 0.0, &mut c)?;
    println!("SpMM C = A*B (4×2): {:?}", &c[..8]);

    // ── SpGEMM ─────────────────────────────────────────────
    println!("\n--- SpGEMM (sparse × sparse) ---");
    let diag_a = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 4.0_f32])?;
    let diag_b = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![5.0, 6.0, 7.0_f32])?;
    let product = trueno_sparse::spgemm(&diag_a, &diag_b)?;
    println!("diag([2,3,4]) × diag([5,6,7]) = diag({:?})", product.values());

    // ── BSR ────────────────────────────────────────────────
    println!("\n--- BSR (Block Sparse Row) ---");
    #[rustfmt::skip]
    let dense = [
        1.0, 2.0, 0.0, 0.0,
        3.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 6.0,
        0.0, 0.0, 7.0, 8.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
    let mut y_bsr = vec![0.0_f32; 4];
    bsr.spmv(1.0, &[1.0, 1.0, 1.0, 1.0], 0.0, &mut y_bsr)?;
    println!("BSR SpMV (block_size=2): {y_bsr:?}");

    // ── SELL ───────────────────────────────────────────────
    println!("\n--- SELL (Sliced ELLPACK) ---");
    let sell = SellMatrix::from_csr(&csr, 2);
    let mut y_sell = vec![0.0_f32; 4];
    sell.spmv(1.0, &x, 0.0, &mut y_sell)?;
    println!("SELL SpMV (slice_size=2): {y_sell:?}");
    println!("SELL storage: {} elements (padded)", sell.storage_size());

    // Verify SELL matches CSR
    let max_diff: f32 = y.iter().zip(y_sell.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("SELL vs CSR max diff: {max_diff:.2e}");

    println!("\n=== Done ===");
    Ok(())
}
