//! `<bin> update`: download, verify (fail closed), smoke, swap atomically.

use crate::decide::Candidate;
use crate::fetch::Net;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// First field of a `shasum -a 256` line; must be 64 hex chars.
fn parse_sha_file(body: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(body)
        .ok()?
        .split_whitespace()
        .next()?
        .to_ascii_lowercase();
    is_hex64(&s).then_some(s)
}

/// The tarball's expected sha256: inline from the source, else the published
/// `<asset>.sha256`. No digest means no install.
fn expected_sha(net: &dyn Net, c: &Candidate) -> Result<String, String> {
    if let Some(s) = &c.sha256 {
        let s = s.to_ascii_lowercase();
        return if is_hex64(&s) {
            Ok(s)
        } else {
            Err(format!("malformed sha256 {s:?}"))
        };
    }
    let url = format!("{}.sha256", c.asset_url);
    let body = net
        .get(&url)?
        .ok_or_else(|| format!("no published digest at {url}; refusing"))?;
    parse_sha_file(&body).ok_or_else(|| format!("{url} holds no sha256; refusing"))
}

/// The one regular file named `bin` in the tarball.
fn extract_bin(tarball: &[u8], bin: &str) -> Result<Vec<u8>, String> {
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(tarball));
    let mut found = None;
    for e in ar.entries().map_err(|e| format!("tarball: {e}"))? {
        let mut e = e.map_err(|e| format!("tarball: {e}"))?;
        let name = e
            .path()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_os_string()));
        if e.header().entry_type().is_file() && name.as_deref() == Some(std::ffi::OsStr::new(bin)) {
            if found.is_some() {
                return Err(format!("tarball holds more than one `{bin}`; refusing"));
            }
            let mut buf = Vec::new();
            e.read_to_end(&mut buf)
                .map_err(|e| format!("tarball: {e}"))?;
            found = Some(buf);
        }
    }
    found.ok_or_else(|| format!("tarball holds no `{bin}`"))
}

/// Whole-token match, so `0.69.1` is not found inside `10.69.1`.
fn prints_version(out: &str, version: &str) -> bool {
    out.split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',')
        .any(|t| t.trim_start_matches('v') == version)
}

#[cfg(unix)]
fn write_executable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o755)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

#[cfg(not(unix))]
fn write_executable(_: &Path, _: &[u8]) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "self-update is supported on unix only",
    ))
}

/// The new binary must run and print the version it claims.
fn smoke(new: &Path, version: &str) -> Result<(), String> {
    // ETXTBSY (26): a concurrent fork elsewhere in the process can still hold
    // the write fd for a moment after it closed here. Retry briefly.
    let mut tries = 0;
    let o = loop {
        match std::process::Command::new(new).arg("--version").output() {
            Err(e) if e.raw_os_error() == Some(26) && tries < 20 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            r => break r.map_err(|e| format!("new binary does not run: {e}"))?,
        }
    };
    let out = String::from_utf8_lossy(&o.stdout);
    if o.status.success() && prints_version(&out, version) {
        Ok(())
    } else {
        Err(format!(
            "new binary failed its --version smoke (want {version}): {}",
            out.trim()
        ))
    }
}

/// Keep the running binary as `<exe>.prev`, then rename the new one over it.
fn swap(new: &Path, exe: &Path) -> Result<PathBuf, String> {
    let mut prev = exe.as_os_str().to_os_string();
    prev.push(".prev");
    let prev = PathBuf::from(prev);
    let _ = std::fs::remove_file(&prev);
    if std::fs::hard_link(exe, &prev).is_err() {
        std::fs::copy(exe, &prev).map_err(|e| format!("keep {}: {e}", prev.display()))?;
    }
    std::fs::rename(new, exe).map_err(|e| format!("install {}: {e}", exe.display()))?;
    Ok(prev)
}

/// Install `c` over `exe`. Returns where the previous binary was kept.
pub fn install(net: &dyn Net, bin: &str, c: &Candidate, exe: &Path) -> Result<PathBuf, String> {
    let want = expected_sha(net, c)?;
    let tarball = net
        .get(&c.asset_url)?
        .ok_or_else(|| format!("{} is gone (404)", c.asset_url))?;
    let got = sha256_hex(&tarball);
    if got != want {
        return Err(format!(
            "sha256 mismatch for {}: published {want}, downloaded {got}; refusing",
            c.asset_url
        ));
    }
    let bytes = extract_bin(&tarball, bin)?;
    if let Some(b) = &c.bin_sha256 {
        let g = sha256_hex(&bytes);
        if !g.eq_ignore_ascii_case(b) {
            return Err(format!(
                "executable sha256 {g} != manifest bin_sha256 {b}; refusing"
            ));
        }
    }
    let dir = exe.parent().ok_or("executable has no directory")?;
    let new = dir.join(format!(".{bin}.new.{}", std::process::id()));
    let _ = std::fs::remove_file(&new);
    write_executable(&new, &bytes).map_err(|e| format!("write {}: {e}", new.display()))?;
    let r = smoke(&new, &c.version).and_then(|()| swap(&new, exe));
    if r.is_err() {
        let _ = std::fs::remove_file(&new);
    }
    r
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::decide::Source;
    use std::collections::HashMap;

    struct Fake(HashMap<String, Vec<u8>>);
    impl Net for Fake {
        fn get(&self, url: &str) -> Result<Option<Vec<u8>>, String> {
            Ok(self.0.get(url).cloned())
        }
    }

    fn tarball(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut b = tar::Builder::new(flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast(),
        ));
        for (path, data) in files {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            b.append_data(&mut h, path, *data).expect("append");
        }
        b.into_inner().expect("tar").finish().expect("gz")
    }

    const URL: &str = "https://dl/apr.tar.gz";
    fn script(v: &str) -> Vec<u8> {
        format!("#!/bin/sh\necho \"apr {v} (abc1234)\"\n").into_bytes()
    }
    fn cand(sha: Option<String>, bin_sha: Option<String>) -> Candidate {
        Candidate {
            source: Source::Nightly,
            version: "0.69.1".into(),
            git_ref: "abc".into(),
            asset_url: URL.into(),
            sha256: sha,
            bin_sha256: bin_sha,
        }
    }
    /// A running "old" apr in a temp dir.
    fn old_exe() -> (tempfile::TempDir, PathBuf) {
        let d = tempfile::tempdir().expect("tempdir");
        let exe = d.path().join("apr");
        write_executable(&exe, &script("0.69.0")).expect("old");
        (d, exe)
    }
    fn contents(p: &Path) -> Vec<u8> {
        std::fs::read(p).expect("read")
    }

    #[test]
    fn install_verified_nightly_swaps_and_keeps_prev() {
        let bin = script("0.69.1");
        let tb = tarball(&[
            ("apr-x86_64-unknown-linux-gnu/apr", &bin),
            ("apr-x86_64-unknown-linux-gnu/README.md", b"r"),
        ]);
        let c = cand(Some(sha256_hex(&tb)), Some(sha256_hex(&bin)));
        let net = Fake(HashMap::from([(URL.to_string(), tb)]));
        let (_d, exe) = old_exe();
        let prev = install(&net, "apr", &c, &exe).expect("installs");
        assert_eq!(contents(&exe), bin);
        assert_eq!(contents(&prev), script("0.69.0"));
    }

    #[test]
    fn release_digest_comes_from_the_published_sha256_file() {
        let bin = script("0.69.1");
        let tb = tarball(&[("apr", &bin)]);
        let line = format!("{}  apr.tar.gz\n", sha256_hex(&tb)).into_bytes();
        let net = Fake(HashMap::from([
            (URL.to_string(), tb),
            (format!("{URL}.sha256"), line),
        ]));
        let (_d, exe) = old_exe();
        install(&net, "apr", &cand(None, None), &exe).expect("installs");
        assert_eq!(contents(&exe), bin);
    }

    /// Every refusal leaves the running binary untouched and no temp file behind.
    #[test]
    fn refusals_fail_closed() {
        let good = script("0.69.1");
        let tb = tarball(&[("apr", &good)]);
        let ok_sha = sha256_hex(&tb);
        let rows: Vec<(&str, Vec<(String, Vec<u8>)>, Candidate, &str)> = vec![
            (
                "tarball sha mismatch",
                vec![(URL.into(), tb.clone())],
                cand(Some("0".repeat(64)), None),
                "sha256 mismatch",
            ),
            (
                "malformed inline sha",
                vec![(URL.into(), tb.clone())],
                cand(Some("xyz".into()), None),
                "malformed",
            ),
            (
                "no published digest",
                vec![(URL.into(), tb.clone())],
                cand(None, None),
                "no published digest",
            ),
            (
                "digest file is junk",
                vec![
                    (URL.into(), tb.clone()),
                    (format!("{URL}.sha256"), b"nope".to_vec()),
                ],
                cand(None, None),
                "holds no sha256",
            ),
            (
                "bin_sha256 mismatch",
                vec![(URL.into(), tb.clone())],
                cand(Some(ok_sha.clone()), Some("1".repeat(64))),
                "bin_sha256",
            ),
            (
                "asset gone",
                vec![],
                cand(Some(ok_sha.clone()), None),
                "404",
            ),
            (
                "no bin in tarball",
                vec![(URL.into(), tarball(&[("other", b"x")]))],
                cand(Some(sha256_hex(&tarball(&[("other", b"x")]))), None),
                "holds no `apr`",
            ),
            (
                "two bins in tarball",
                vec![(URL.into(), tarball(&[("a/apr", &good), ("b/apr", &good)]))],
                cand(
                    Some(sha256_hex(&tarball(&[("a/apr", &good), ("b/apr", &good)]))),
                    None,
                ),
                "more than one",
            ),
            (
                "new binary prints another version",
                vec![(URL.into(), tarball(&[("apr", &script("10.69.1"))]))],
                cand(
                    Some(sha256_hex(&tarball(&[("apr", &script("10.69.1"))]))),
                    None,
                ),
                "smoke",
            ),
        ];
        let mut bad = Vec::new();
        for (name, files, c, want) in rows {
            let (d, exe) = old_exe();
            let r = install(&Fake(files.into_iter().collect()), "apr", &c, &exe);
            let left: Vec<_> = std::fs::read_dir(d.path())
                .expect("dir")
                .map(|e| e.expect("e").file_name())
                .collect();
            match r {
                Err(e)
                    if e.contains(want)
                        && contents(&exe) == script("0.69.0")
                        && left.len() == 1 => {}
                other => bad.push(format!("{name}: {other:?}, dir {left:?}")),
            }
        }
        assert!(bad.is_empty(), "refusal rows failed:\n{}", bad.join("\n"));
    }

    #[test]
    fn helpers() {
        assert!(prints_version("apr 0.69.1 (abc)", "0.69.1"));
        assert!(prints_version("pv v0.69.1", "0.69.1"));
        assert!(!prints_version("apr 10.69.1", "0.69.1"));
        assert_eq!(
            parse_sha_file(format!("{}  x\n", "A".repeat(64)).as_bytes()),
            Some("a".repeat(64))
        );
    }
}
