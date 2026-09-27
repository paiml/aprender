//! `~/.cache/<bin>/update-check.json`: what the last check found, and when.

use crate::decide::{Candidate, Source};
use crate::policy::Installs;
use crate::Product;
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "sovereign-update-check/v1";
/// Re-check at most once per this many seconds.
pub const TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cache {
    pub schema: String,
    /// Unix seconds of the last attempt, successful or not.
    pub checked_at: u64,
    /// The binary the result is about. A different binary ignores it.
    pub current_version: String,
    pub current_sha: Option<String>,
    pub available: Option<Candidate>,
    /// Last failure, kept for `update --check`; never printed at startup.
    pub error: Option<String>,
}

/// `$XDG_CACHE_HOME/<bin>/update-check.json`, else `$HOME/.cache/...`.
#[must_use]
pub fn path(env: &dyn Fn(&str) -> Option<String>, bin: &str) -> Option<PathBuf> {
    let base = env("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env("HOME")
                .filter(|v| !v.is_empty())
                .map(|h| Path::new(&h).join(".cache"))
        })?;
    Some(base.join(bin).join("update-check.json"))
}

#[must_use]
pub fn load(path: &Path) -> Option<Cache> {
    let c: Cache = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    (c.schema == SCHEMA).then_some(c)
}

/// Write via a temp file and rename, so a reader never sees half a file.
pub fn store(path: &Path, c: &Cache) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".update-check.{}.tmp", std::process::id()));
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(c).map_err(std::io::Error::other)?,
    )?;
    std::fs::rename(&tmp, path)
}

fn same_binary(p: &Product, c: &Cache) -> bool {
    c.current_version == p.version && c.current_sha.as_deref() == p.build_sha
}

/// True when there is no usable cache, it is older than the TTL, it is about
/// another binary, or its timestamp is in the future (clock moved back).
#[must_use]
pub fn needs_refresh(p: &Product, c: Option<&Cache>, now: u64) -> bool {
    c.is_none_or(|c| !same_binary(p, c) || c.checked_at > now || now - c.checked_at >= TTL_SECS)
}

/// The record the startup check writes before spawning its refresh: stamped
/// now, so no other run starts one, and keeping this binary's known update.
#[must_use]
pub fn claim(p: &Product, old: Option<&Cache>, now: u64) -> Cache {
    Cache {
        schema: SCHEMA.into(),
        checked_at: now,
        current_version: p.version.into(),
        current_sha: p.build_sha.map(Into::into),
        available: old
            .filter(|c| same_binary(p, c))
            .and_then(|c| c.available.clone()),
        error: Some("refresh in progress".into()),
    }
}

/// The one stderr line, when the cache says a newer build exists for THIS binary.
#[must_use]
pub fn notice(p: &Product, c: &Cache, installs: &Installs) -> Option<String> {
    let a = c.available.as_ref().filter(|_| same_binary(p, c))?;
    let src = match a.source {
        Source::Release => "release",
        Source::Nightly => "nightly",
    };
    let bin = p.bin;
    let tail = match installs {
        Installs::User => format!("run `{bin} update`"),
        Installs::Arbiter(_) => "fleet host: the arbiter installs it".to_string(),
    };
    Some(format!(
        "{bin} {} -> {} available ({src}) — {tail}",
        p.version, a.version
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Product = Product {
        bin: "apr",
        repo: "paiml/aprender",
        version: "0.69.0",
        build_sha: Some("abc"),
        release_asset: None,
        nightly: true,
    };

    fn cache(at: u64, ver: &str, avail: bool) -> Cache {
        Cache {
            schema: SCHEMA.into(),
            checked_at: at,
            current_version: ver.into(),
            current_sha: Some("abc".into()),
            available: avail.then(|| Candidate {
                source: Source::Release,
                version: "0.69.1".into(),
                git_ref: "v0.69.1".into(),
                asset_url: String::new(),
                sha256: None,
                bin_sha256: None,
            }),
            error: None,
        }
    }

    #[test]
    fn refresh_rows() {
        let now = 10 * TTL_SECS;
        assert!(needs_refresh(&P, None, now), "no cache");
        assert!(
            !needs_refresh(&P, Some(&cache(now - 60, "0.69.0", false)), now),
            "fresh"
        );
        assert!(
            needs_refresh(&P, Some(&cache(now - TTL_SECS, "0.69.0", false)), now),
            "exactly 24 h"
        );
        assert!(
            needs_refresh(&P, Some(&cache(now + 60, "0.69.0", false)), now),
            "future timestamp"
        );
        assert!(
            needs_refresh(&P, Some(&cache(now, "0.68.0", false)), now),
            "another binary's cache"
        );
    }

    #[test]
    fn claim_rows() {
        let now = 10 * TTL_SECS;
        let c = claim(&P, Some(&cache(0, "0.69.0", true)), now);
        assert!(
            !needs_refresh(&P, Some(&c), now),
            "a claim blocks a second refresh"
        );
        assert!(
            c.available.is_some(),
            "a claim keeps this binary's known update"
        );
        assert!(
            claim(&P, Some(&cache(0, "0.68.0", true)), now)
                .available
                .is_none(),
            "a claim drops another binary's update"
        );
        assert!(claim(&P, None, now).available.is_none());
    }

    #[test]
    fn notice_rows() {
        let n = notice(&P, &cache(0, "0.69.0", true), &Installs::User).expect("newer cached");
        assert_eq!(
            n,
            "apr 0.69.0 -> 0.69.1 available (release) — run `apr update`"
        );
        let f = notice(
            &P,
            &cache(0, "0.69.0", true),
            &Installs::Arbiter("x".into()),
        )
        .expect("fleet");
        assert!(f.ends_with("fleet host: the arbiter installs it"), "{f}");
        assert!(
            notice(&P, &cache(0, "0.69.0", false), &Installs::User).is_none(),
            "up to date"
        );
        assert!(
            notice(&P, &cache(0, "0.68.0", true), &Installs::User).is_none(),
            "a stale binary's cache is ignored"
        );
    }

    #[test]
    fn store_load_roundtrip_and_paths() {
        let d = tempfile::tempdir().expect("tempdir");
        let p = d.path().join("apr").join("update-check.json");
        let c = cache(5, "0.69.0", true);
        store(&p, &c).expect("store");
        assert_eq!(load(&p), Some(c));
        std::fs::write(&p, b"{\"schema\":\"other\"}").expect("write");
        assert_eq!(load(&p), None, "a foreign schema is no cache");
        let env = |k: &str| (k == "HOME").then(|| "/h".to_string());
        assert_eq!(
            path(&env, "apr"),
            Some(PathBuf::from("/h/.cache/apr/update-check.json"))
        );
        let xdg = |k: &str| (k == "XDG_CACHE_HOME").then(|| "/x".to_string());
        assert_eq!(
            path(&xdg, "pv"),
            Some(PathBuf::from("/x/pv/update-check.json"))
        );
        assert_eq!(path(&|_: &str| None, "pv"), None);
    }
}
