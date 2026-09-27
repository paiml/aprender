// A row-major and a column-major tensor cannot be added: the layout is a type.
use trueno_tensor::{ColMajor, RankedTensor};

fn main() {
    let a = RankedTensor::<2>::zeros([2, 2]);
    let b = RankedTensor::<2, ColMajor>::zeros([2, 2]);
    let _ = a.add(&b);
}
