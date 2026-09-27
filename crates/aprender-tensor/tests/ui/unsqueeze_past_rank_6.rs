// Rank 6 is the ceiling: there is no rank-7 tensor to unsqueeze into.
use trueno_tensor::RankedTensor;

fn main() {
    let a = RankedTensor::<6>::zeros([1, 1, 1, 1, 1, 1]);
    let _ = a.unsqueeze(0);
}
