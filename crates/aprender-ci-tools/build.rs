//! #4219: stamp `APR_GIT_SHA` so `--version` prints `<name> <version> (<sha>)`.
fn main() {
    build_sha::emit();
}
