//! Two independent build-time jobs, one env var each.
//!
//! **1. `CHRONOS_EMBED_DIR` — embedding (D-13).** Set at BUILD time ⇒ its
//! `model.safetensors` + `config.json` are copied into `OUT_DIR` and compiled into the
//! binary by `include_bytes!` (the `aprender-mcp-setfit-lambda` pattern). Unset ⇒ EMPTY
//! markers are written so the crate still builds — CI has no weights to embed — and the
//! runtime falls back to reading `CHRONOS_MODEL_DIR` as a path instead. Never a
//! download: weights come only from the pinned, sha256-verified `just fetch-chronos-tiny`
//! (T-06-06).
//!
//! **2. `CHRONOS_MODEL_DIR` — arming (D-18).** Set at BUILD time and holding a
//! `model.safetensors` ⇒ `cfg(chronos_weights)`, which turns the weight-dependent e2e
//! tests from COUNTED SKIPS into real tests. Repeated verbatim from
//! `crates/aprender-forecast/build.rs`: a cfg is per-crate, so each crate emits its own.
//! The `println!("SKIP"); return;` style reports `0 ignored` — a silent green — which
//! D-18 forbids.
//!
//! The two are deliberately independent: an embedded build is not automatically an armed
//! one (the oracle comparison needs the f32 weights DIRECTORY), and an armed build need
//! not embed anything.
//!
//! `cargo::rustc-check-cfg` declares the cfg so `unexpected_cfgs` (a workspace lint)
//! stays quiet without listing it in the root manifest.

use std::path::PathBuf;

fn main() {
    // ---- 1. embedding ----------------------------------------------------------------
    println!("cargo:rerun-if-env-changed=CHRONOS_EMBED_DIR");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let default_f16 = manifest_dir.join("../../models/chronos-bolt-tiny/f16");
    let embed_dir = std::env::var_os("CHRONOS_EMBED_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            if default_f16.join("model.safetensors").exists() {
                Some(default_f16)
            } else {
                None
            }
        });

    match embed_dir {
        Some(dir) => {
            for name in ["model.safetensors", "config.json"] {
                let src = dir.join(name);
                println!("cargo:rerun-if-changed={}", src.display());
                std::fs::copy(&src, out.join(name)).unwrap_or_else(|e| {
                    panic!(
                        "CHRONOS_EMBED_DIR={} could not be staged for embedding: cannot copy \
                         {name}: {e}",
                        dir.display()
                    )
                });
            }
        }
        None => {
            for name in ["model.safetensors", "config.json"] {
                std::fs::write(out.join(name), []).expect("write empty embed marker");
            }
        }
    }

    // ---- 2. arming -------------------------------------------------------------------
    println!("cargo::rustc-check-cfg=cfg(chronos_weights)");
    println!("cargo:rerun-if-env-changed=CHRONOS_MODEL_DIR");
    // Emitting ANY `rerun-if-*` replaces cargo's default "rerun when the package changed",
    // so naming the artifact narrows — but does not close — the case where weights appear
    // or vanish at an unchanged path with an unchanged mtime (restored from a cache, `tar
    // -p`, a docker layer). The same gap is documented at length in
    // `crates/aprender-forecast/build.rs`; closing it needs a CONTENT marker the fetch
    // recipe sets, tracked for 06-08 alongside the CI arming decision.
    let weights = std::env::var_os("CHRONOS_MODEL_DIR")
        .map(|dir| std::path::Path::new(&dir).join("model.safetensors"));
    if let Some(w) = &weights {
        println!("cargo:rerun-if-changed={}", w.display());
    }
    if weights.is_some_and(|w| w.is_file()) {
        println!("cargo:rustc-cfg=chronos_weights");
    }
}
