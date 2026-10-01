//! Turn Chronos weight PRESENCE into a compile-time cfg, so the weight-dependent parity
//! tests are COUNTED skips when unarmed and real tests when armed (D-18).
//!
//! `CHRONOS_MODEL_DIR` set at BUILD time and holding a `model.safetensors` => `cfg(chronos_weights)`.
//! Unset (or pointing at a directory with no weights) => the cfg is absent, and every gated test
//! carries `#[cfg_attr(not(chronos_weights), ignore = "…run `just fetch-chronos-tiny` to arm")]`,
//! which libtest reports as `N ignored` with the reason printed. That is the whole point: the
//! `println!("SKIP"); return;` style reports `0 ignored`, i.e. a silent green, which D-18 forbids.
//!
//! `cargo::rustc-check-cfg` declares the cfg so `unexpected_cfgs` (workspace lint) stays quiet
//! without listing it in the root manifest.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(chronos_weights)");
    println!("cargo:rerun-if-env-changed=CHRONOS_MODEL_DIR");

    // Emitting ANY `rerun-if-*` replaces cargo's default "rerun when the package changed",
    // so the env-var line alone leaves this script blind to the weights APPEARING or
    // VANISHING at an unchanged path. Naming the artifact narrows that, but does NOT close
    // it, and the gap is proven, not theoretical:
    //
    //   `rerun-if-changed` compares MTIME. Weights restored with an old mtime — actions/cache,
    //   `tar -p`, a docker layer — leave the cfg unarmed, and `just fetch-chronos-tiny`
    //   deliberately never overwrites a present file (justfile ~line 390), so a re-fetch does
    //   not bump it either. The seven gated tests then report `ignored` on a green run: the
    //   silent green D-18 exists to close, arriving through the cache instead of the skip.
    //
    // Do not read this line as a guarantee. Closing it needs a CONTENT marker the fetch
    // recipe sets (e.g. CHRONOS_WEIGHTS_SHA) rather than a filesystem probe, plus a gate
    // that asserts the gated tests actually RAN — today nothing reads that count.
    // Tracked for plan 06-08 alongside the CI arming decision.
    let weights = std::env::var_os("CHRONOS_MODEL_DIR")
        .map(|dir| std::path::Path::new(&dir).join("model.safetensors"));
    if let Some(w) = &weights {
        println!("cargo:rerun-if-changed={}", w.display());
    }
    let armed = weights.is_some_and(|w| w.is_file());
    if armed {
        println!("cargo:rustc-cfg=chronos_weights");
    }
}
