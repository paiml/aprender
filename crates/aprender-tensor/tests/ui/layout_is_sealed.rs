// Layout is sealed: a third layout cannot be invented downstream.
use trueno_tensor::Layout;

#[derive(Debug, Clone, Copy, Default)]
struct Diagonal;

impl Layout for Diagonal {
    const NAME: &'static str = "diagonal";
    const LAST_AXIS_CONTIGUOUS: bool = true;
}

fn main() {}
