// A rank-0 tensor has no axis to squeeze.
use trueno_tensor::RankedTensor;

fn main() {
    let s = RankedTensor::<0>::zeros([]);
    let _ = s.squeeze(0);
}
