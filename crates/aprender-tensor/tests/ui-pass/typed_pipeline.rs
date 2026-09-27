// The positive half: the programs the types are meant to allow still compile.
use trueno_tensor::{ColMajor, Matrix, RankedTensor, Tensor};

fn main() {
    let w = RankedTensor::<2, ColMajor>::from_gguf([2, 3], vec![0.0; 6]).expect("gguf");
    let w: Matrix = w.into_apr();
    let x = Matrix::zeros([3, 3]);
    let y = x.matmul(&w).expect("matmul").add(&Matrix::zeros([3, 2])).expect("add");
    let v: RankedTensor<3> = y.unsqueeze(0).expect("unsqueeze");
    let back: Matrix = v.squeeze(0).expect("squeeze");
    let flat: RankedTensor<1> = back.t().reshape([6]).expect("reshape");
    let dynamic: Tensor = flat.into_dynamic();
    let _ = RankedTensor::<1>::from_dynamic(&dynamic).expect("round trip");
}
