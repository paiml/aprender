// Passing a rank-3 tensor where a matrix is required is a type error.
use trueno_tensor::{Matrix, RankedTensor};

fn main() {
    let a = Matrix::zeros([2, 2]);
    let b = RankedTensor::<3>::zeros([2, 2, 1]);
    let _ = a.matmul(&b);
}
