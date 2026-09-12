//! Classification witness: `needs-hardware`. The wording is the CUDA runtime's
//! own, so the classifier's pattern is tested against what a driver-less host
//! really prints — a skip, never a pass.
fn main() {
    eprintln!("Error: no CUDA-capable device is detected (CUDA_ERROR_NO_DEVICE)");
    std::process::exit(1);
}
