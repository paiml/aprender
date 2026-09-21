//! #3545 — `apr devices` must describe the binary it is part of.
//!
//! **The defect.** The shipped `-cuda` asset ran CUDA correctly and told every new user
//! `cuda unavailable reason=NotCompiled` on the first screen they saw. Two flags decided one fact:
//! `apr-cli`'s `cuda` feature enabled `realizar/cuda` (so inference ran on the GPU) but **not**
//! `trueno/cuda` — and `trueno`'s backend registry is what `devices` reads. With no `CudaFactory`
//! in `default_factories()`, the registry emitted `missing_entry(Cuda)` = `Source::NotCompiled`.
//!
//! **This test is the one that RED-turns.** It asserts the two flags are the same flag: if a build
//! has `apr-cli/cuda`, the registry it reports from must not answer `not-compiled`. Remove
//! `trueno/cuda` from the feature in `crates/apr-cli/Cargo.toml` and this file fails — which is the
//! whole point, because nothing else would have noticed for another release.
//!
//! **It is not "always say yes".** The `#[cfg(not(feature = "cuda"))]` arm requires the opposite
//! answer on the same host: a build without the feature must still report `not-compiled` **even
//! where a GPU is physically present**. A fix that made `devices` optimistic would fail that arm.
//!
//! Measured 2026-09-20 on lambda (RTX 4090 present for all three):
//!
//! | build | cuda status | cuda source |
//! |---|---|---|
//! | `--features cuda`, before the fix | unavailable | `not-compiled` ← the defect |
//! | `--features cuda`, after | ready | `dlopen` |
//! | `--no-default-features --features cli,inference` | unavailable | `not-compiled` ← honest |
//!
//! **`dlopen`, not `compiled-in`** — and #3545's acceptance query asks for `compiled-in`, which
//! would be false even after a correct fix. `Source::CompiledIn` means *linked into the binary*;
//! `Source::Dlopen(path)` means *loaded at run time from this library path*, which is exactly how
//! the CUDA driver is reached. The factory is linked in; the driver is not. Asserting
//! `compiled-in` would require the registry to lie in the other direction, so this file asserts
//! **`!= not-compiled`** and the issue's query needs correcting rather than satisfying.

#[test]
#[cfg(feature = "cuda")]
fn a_cuda_build_does_not_report_its_own_cuda_as_not_compiled() {
    let reg = trueno::registry::BackendRegistry::discover();
    let cuda = reg
        .entries
        .iter()
        .find(|e| e.kind.as_str() == "cuda")
        .expect("the registry always emits a cuda line, even when absent");

    assert_ne!(
        cuda.source,
        trueno::registry::Source::NotCompiled,
        "built with apr-cli/cuda, but the registry `devices` reads says not-compiled — \
         `trueno/cuda` is missing from the feature again (#3545). Inference would still run on \
         the GPU, and the first screen a user sees would still be wrong."
    );
    assert!(
        !matches!(
            cuda.status,
            trueno::registry::Status::Unavailable(trueno::registry::Reason::NotCompiled)
        ),
        "the cuda entry reports Reason::NotCompiled on a cuda build"
    );
}

#[test]
#[cfg(not(feature = "cuda"))]
fn a_non_cuda_build_still_says_not_compiled_even_where_a_gpu_exists() {
    // The falsifier for the fix: "always say yes" passes the arm above and fails this one. This
    // runs on hosts with and without a GPU and must answer the same way on both — the question is
    // about the BINARY, not about the hardware.
    let reg = trueno::registry::BackendRegistry::discover();
    let cuda = reg
        .entries
        .iter()
        .find(|e| e.kind.as_str() == "cuda")
        .expect("the registry always emits a cuda line, even when absent");

    assert_eq!(
        cuda.source,
        trueno::registry::Source::NotCompiled,
        "built WITHOUT apr-cli/cuda, but the registry claims cuda is present — `devices` is \
         describing the host instead of the binary"
    );
}
