// A rank-3 tensor has no `matmul`: the method exists only on rank 2.
use trueno_tensor::RankedTensor;

fn main() {
    let a = RankedTensor::<3>::zeros([2, 2, 2]);
    let _ = a.matmul(&a);
}
