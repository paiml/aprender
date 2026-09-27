// Adding tensors of different rank is a type error, not a runtime check.
use trueno_tensor::RankedTensor;

fn main() {
    let a = RankedTensor::<2>::zeros([2, 2]);
    let b = RankedTensor::<3>::zeros([2, 2, 1]);
    let _ = a.add(&b);
}
