fn main() {
    // #4219: stamp APR_GIT_SHA for `--version` before anything can return early.
    build_sha::emit();
}
