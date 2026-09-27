//! #4219: stamp `APR_GIT_SHA` so `--version` prints `<name> <version> (<sha>)`,
//! the shape `apr --version` has carried since #597. Resolution order and
//! rerun triggers live in the shared `aprender-build-sha` crate.
fn main() {
    build_sha::emit();
}
