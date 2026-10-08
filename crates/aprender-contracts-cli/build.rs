// #4219: stamp APR_GIT_SHA so `--version` prints `<name> <version> (<sha>)`,
// the shape `apr --version` has carried since #597. Resolution order and
// rerun triggers live in the shared `aprender-build-sha` crate.
#[cfg(feature = "build-sha")]
fn main() {
    build_sha::emit();
}

// #4604 C2: without `build-sha` (the crates.io default) the stamp is the
// release override when one is set, else the same `v<ver>+no-git` fallback
// `build_sha` ends on.
#[cfg(not(feature = "build-sha"))]
fn main() {
    let sha = std::env::var("APR_GIT_SHA_OVERRIDE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("v{}+no-git", env!("CARGO_PKG_VERSION")));
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");
    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
}
