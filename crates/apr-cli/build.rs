// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. .cargo_vcs_info.json `git.sha1` (crates.io / packaged builds, #4110)
//   3. git rev-parse --short HEAD, retried with the checkout marked safe.directory (#4110)
//   4. committed .git-sha file
//   5. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
//
// The resolution and its rerun-if-changed triggers live in the shared
// `aprender-build-sha` crate (#4219) so every workspace [[bin]] stamps the same SHA.
fn main() {
    build_sha::emit();
}
