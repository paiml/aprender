// A GGUF (column-major) right-hand side cannot reach the row-major kernel.
use trueno_tensor::{ColMajor, Matrix, RankedTensor};

fn main() {
    let a = Matrix::zeros([2, 2]);
    let b = RankedTensor::<2, ColMajor>::zeros([2, 2]);
    let _ = a.matmul(&b);
}
