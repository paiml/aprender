//! Where `apr model publish` gets its HF token (EXT-14). Operator ruling, 2026-09-26:
//! "EXT-14 should be able to use local hugging face token".
//!
//! `--token-file` still wins and must be mode 0600 (R-7). Without it the token is
//! resolved the way `huggingface_hub` resolves it on this host: `HF_TOKEN`, then
//! `HUGGING_FACE_HUB_TOKEN`, then the file at `HF_TOKEN_PATH`, `$HF_HOME/token`,
//! `$XDG_CACHE_HOME/huggingface/token` or `~/.cache/huggingface/token`.
//!
//! Only the SOURCE is ever reported (`env:HF_TOKEN`, `file:<path>`), in the receipt
//! and on stderr; the value goes into the `Authorization` header and nowhere else.
//! The publisher modules themselves still read no environment: this module is the
//! one place that does.

use super::hf_publish::Token;
use crate::error::{CliError, Result};
use std::path::{Path, PathBuf};

/// The env vars that may carry the token, in `huggingface_hub`'s order.
pub(crate) const TOKEN_VARS: [&str; 2] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// The standard HF token file for this environment.
fn standard_file(env: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let set = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(p) = set("HF_TOKEN_PATH") {
        return Some(PathBuf::from(p));
    }
    if let Some(h) = set("HF_HOME") {
        return Some(PathBuf::from(h).join("token"));
    }
    let cache = set("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| set("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(cache.join("huggingface").join("token"))
}

/// The token and where it came from. `env` is the environment lookup, injected so the
/// order is testable without touching the process environment.
///
/// # Errors
///
/// `ValidationFailed` when `explicit` fails its 0600 check or holds no single token,
/// or when no source yields one; the message names the sources, never a value.
pub(crate) fn resolve(
    explicit: Option<&Path>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<(Token, String)> {
    if let Some(p) = explicit {
        return Ok((Token::from_file(p)?, format!("file:{}", p.display())));
    }
    for k in TOKEN_VARS {
        if let Some(v) = env(k).filter(|v| !v.trim().is_empty()) {
            return Ok((Token::parse(&v, k)?, format!("env:{k}")));
        }
    }
    let file = standard_file(env);
    if let Some(p) = file.as_deref().filter(|p| p.is_file()) {
        let body = std::fs::read_to_string(p)
            .map_err(|e| CliError::ValidationFailed(format!("{}: {e}", p.display())))?;
        return Ok((
            Token::parse(&body, &p.display().to_string())?,
            format!("file:{}", p.display()),
        ));
    }
    Err(CliError::ValidationFailed(format!(
        "no HF token: pass --token-file, set {}, or log in so {} exists",
        TOKEN_VARS.join(" or "),
        file.map_or_else(
            || "~/.cache/huggingface/token".to_string(),
            |p| p.display().to_string()
        )
    )))
}

/// [`resolve`] against this process's environment.
pub(crate) fn resolve_from_process(explicit: Option<&Path>) -> Result<(Token, String)> {
    resolve(explicit, &|k| std::env::var(k).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    const SECRET: &str = "hf_LOCALtokenVALUE0123";

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn falsify_ext_014_local_hf_token_resolves_in_huggingface_hub_order() {
        let t = TempDir::new().expect("tmp");
        let home = t.path().to_str().expect("utf8");
        let cache = t.path().join(".cache/huggingface");
        std::fs::create_dir_all(&cache).expect("mkdir");
        std::fs::write(cache.join("token"), format!("{SECRET}-cache\n")).expect("w");
        let hf_home = t.path().join("hfhome");
        std::fs::create_dir_all(&hf_home).expect("mkdir");
        std::fs::write(hf_home.join("token"), format!("{SECRET}-hfhome")).expect("w");
        let hf_home = hf_home.to_str().expect("utf8");

        let cases: [(&[(&str, &str)], &str, &str); 5] = [
            (
                &[
                    ("HF_TOKEN", "a"),
                    ("HUGGING_FACE_HUB_TOKEN", "b"),
                    ("HOME", home),
                ],
                "env:HF_TOKEN",
                "a",
            ),
            (
                &[
                    ("HF_TOKEN", " "),
                    ("HUGGING_FACE_HUB_TOKEN", "b"),
                    ("HOME", home),
                ],
                "env:HUGGING_FACE_HUB_TOKEN",
                "b",
            ),
            (
                &[("HF_HOME", hf_home), ("HOME", home)],
                "hfhome/token",
                "-hfhome",
            ),
            (&[("HOME", home)], ".cache/huggingface/token", "-cache"),
            (
                &[
                    ("XDG_CACHE_HOME", &format!("{home}/.cache")),
                    ("HOME", "/nonexistent"),
                ],
                ".cache/huggingface/token",
                "-cache",
            ),
        ];
        for (env, source, value) in cases {
            let (tok, src) = resolve(None, &env_of(env)).expect(source);
            assert!(src.ends_with(source), "{src} vs {source}");
            assert!(tok.bearer().ends_with(value), "{env:?}");
            assert!(
                !src.contains(SECRET),
                "the source names a place, not a value"
            );
        }
    }

    #[test]
    fn an_explicit_token_file_wins_and_keeps_its_0600_check() {
        use std::os::unix::fs::PermissionsExt;
        let t = TempDir::new().expect("tmp");
        let p = t.path().join("tok");
        std::fs::write(&p, "hf_fromfile").expect("w");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        let env = env_of(&[("HF_TOKEN", SECRET)]);
        let (tok, src) = resolve(Some(&p), &env).expect("explicit");
        assert_eq!(tok.bearer(), "Bearer hf_fromfile");
        assert_eq!(src, format!("file:{}", p.display()));

        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let err = resolve(Some(&p), &env).expect_err("0644").to_string();
        assert!(err.contains("0600"), "{err}");
    }

    #[test]
    fn no_token_anywhere_names_the_sources_and_no_value() {
        let t = TempDir::new().expect("tmp");
        let env = env_of(&[("HOME", t.path().to_str().expect("utf8"))]);
        let err = resolve(None, &env).expect_err("none").to_string();
        assert!(
            err.contains("HF_TOKEN") && err.contains("--token-file"),
            "{err}"
        );
        assert!(err.contains("huggingface/token"), "{err}");

        let env = env_of(&[("HF_TOKEN", "two words")]);
        let err = resolve(None, &env).expect_err("malformed").to_string();
        assert!(
            err.contains("HF_TOKEN") && !err.contains("two words"),
            "{err}"
        );
    }
}
