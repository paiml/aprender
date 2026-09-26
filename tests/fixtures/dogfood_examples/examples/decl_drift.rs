//! Classification witness: `fail`. Declared needs-hardware with
//! expect = "CUDA init failed", but it panics on an assertion instead. A
//! declaration whose line is absent must go red, not stay skipped.
fn main() {
    assert_eq!(1 + 1, 3, "dogfood-examples fixture: drifted failure");
}
