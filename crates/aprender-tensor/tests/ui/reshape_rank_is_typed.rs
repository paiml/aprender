// `reshape` returns the rank of the shape it was given; binding a rank-3
// reshape to a matrix does not compile.
use trueno_tensor::{Matrix, RankedTensor};

fn main() {
    let v = RankedTensor::<1>::zeros([6]);
    let _m: Matrix = v.reshape([1, 2, 3]).expect("same element count");
}
