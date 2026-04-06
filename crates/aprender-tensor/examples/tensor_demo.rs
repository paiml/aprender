//! Tensor contraction demo — Einstein summation via TTGT.

use trueno_tensor::{einsum, einsum_nary, matmul, outer, trace, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== trueno-tensor: Einstein Summation Demo ===\n");

    // Matrix multiply: ij,jk->ik
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let c = matmul(&a, &b)?;
    println!("Matrix multiply (2x3 * 3x2):");
    println!(
        "  C[0,0]={}, C[0,1]={}, C[1,0]={}, C[1,1]={}\n",
        c.get(&[0, 0]),
        c.get(&[0, 1]),
        c.get(&[1, 0]),
        c.get(&[1, 1])
    );

    // Outer product: i,j->ij
    let u = Tensor::new(vec![3], vec![1.0, 2.0, 3.0])?;
    let v = Tensor::new(vec![2], vec![4.0, 5.0])?;
    let op = outer(&u, &v)?;
    println!("Outer product (3 x 2):");
    println!("  shape={:?}, data={:?}\n", op.shape(), op.data());

    // Trace
    let eye = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;
    let tr = trace(&eye)?;
    println!("Trace of diag(1,2,3) = {tr}\n");

    // 3D contraction: ijk,jkl->il
    let t1 = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let t2 = Tensor::new(vec![2, 2, 3], (1..=12).map(|i| i as f32).collect())?;
    let t3 = einsum("ijk,jkl->il", &t1, &t2)?;
    println!("3D contraction ijk,jkl->il:");
    println!("  shape={:?}, data={:?}\n", t3.shape(), t3.data());

    // Transpose
    let m = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let mt = m.transpose(&[1, 0]);
    println!("Transpose (2x3 -> 3x2):");
    println!("  shape={:?}, data={:?}\n", mt.shape(), mt.data());

    // N-ary einsum: chain of 3 matmuls
    let ma = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let mb =
        Tensor::new(vec![3, 4], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])?;
    let mc = Tensor::new(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])?;
    let chain = einsum_nary("ij,jk,kl->il", &[&ma, &mb, &mc])?;
    println!("N-ary einsum (3 matmuls, 2×3 × 3×4 × 4×2 → 2×2):");
    println!("  shape={:?}, data={:?}", chain.shape(), chain.data());

    Ok(())
}
