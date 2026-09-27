// `into_apr` is the GGUF import boundary: a row-major tensor never crosses it.
use trueno_tensor::Matrix;

fn main() {
    let a = Matrix::zeros([2, 2]);
    let _ = a.into_apr();
}
