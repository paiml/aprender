// A dynamic `Tensor` is row-major; it cannot be read in as column-major.
use trueno_tensor::{ColMajor, RankedTensor, Tensor};

fn main() {
    let t = Tensor::new(vec![2, 2], vec![0.0; 4]).expect("t");
    let _ = RankedTensor::<2, ColMajor>::from_dynamic(&t);
}
