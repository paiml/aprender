// A column-major (GGUF) matrix has no `matmul`: it must cross the import
// boundary (`into_apr` / `to_row_major`) first.
use trueno_tensor::{ColMajor, RankedTensor};

fn main() {
    let a = RankedTensor::<2, ColMajor>::zeros([2, 2]);
    let _ = a.matmul(&a);
}
