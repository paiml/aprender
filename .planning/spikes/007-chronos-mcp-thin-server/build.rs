//! Stage the Chronos-Bolt weights for `include_bytes!` — the `aprender-mcp-setfit-lambda` pattern.
//! `CHRONOS_EMBED_DIR` set at BUILD time ⇒ its `model.safetensors` + `config.json` are copied into
//! OUT_DIR and compiled into the binary. Unset ⇒ empty markers so the crate still builds, and the
//! runtime reads `CHRONOS_MODEL_DIR` as a path instead.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CHRONOS_EMBED_DIR");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    match std::env::var_os("CHRONOS_EMBED_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            for f in ["model.safetensors", "config.json"] {
                let src = dir.join(f);
                println!("cargo:rerun-if-changed={}", src.display());
                std::fs::copy(&src, out.join(f)).unwrap_or_else(|e| panic!("CHRONOS_EMBED_DIR={}: cannot stage {f}: {e}", dir.display()));
            }
        }
        None => {
            for f in ["model.safetensors", "config.json"] { std::fs::write(out.join(f), []).expect("write empty embed marker"); }
        }
    }
}
