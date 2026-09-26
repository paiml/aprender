//! EXT-17 (aprender#4399): the GHCR mirror is idempotent, resumable, never moves a
//! version tag, and keeps the token out of the env, the receipt and every error.

use super::*;
use crate::commands::model_confirm::compare;
use serde_json::json;
use std::collections::BTreeSet;
use tempfile::TempDir;

const FILES: [(&str, &str); 4] = [
    ("LICENSE", "Apache-2.0 text"),
    ("NOTICE", "upstream notice"),
    ("README.md", "# card"),
    ("tuned.apr", "tuned-weights"),
];

const TOKEN: &str = "ghp_EXT17testTOKENvalue0123456789";

/// A release dir as `apr model pack` leaves it, on `channel`.
fn release(dir: &Path, channel: &str) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let files: Vec<_> = FILES
        .iter()
        .map(|(n, b)| {
            std::fs::write(dir.join(n), b).expect("write");
            json!({"name": n, "format": "file", "quant": null, "bytes": b.len(), "sha256": sha256_hex(b.as_bytes())})
        })
        .collect();
    let m = json!({
        "schema": "model-release-v1",
        "line": "paiml/qwen3.5-4b-apr",
        "version": if channel == "rc" { "0.1.0-rc.1" } else { "0.1.0" },
        "channel": channel,
        "files": files,
        "base": {"hf_id": "Qwen/Qwen3.5-4B", "revision": "main@0123abc", "sha256": sha256_hex(b"b")},
        "lineage": ["01RUN"],
        "datasets": [],
        "engine": {"apr_version": "0.71.0", "crate_tarball_sha256": sha256_hex(b"t")},
        "gates": {},
        "license": {"spdx_or_name": "Apache-2.0", "upstream_notice_sha256": sha256_hex(b"upstream notice")}
    });
    let body = serde_json::to_string_pretty(&m).expect("json");
    std::fs::write(dir.join(MANIFEST), format!("{body}\n")).expect("write");
}

/// An in-memory OCI registry. It checks every blob against its digest, as a real
/// registry must, counts what was sent, and can fail after `fail_after` blob uploads.
#[derive(Default)]
struct Fake {
    blobs: BTreeMap<String, Vec<u8>>,
    manifests: BTreeMap<String, Vec<u8>>,
    tags: BTreeMap<String, String>,
    uploads: usize,
    manifest_puts: usize,
    fail_after: Option<usize>,
}

impl Registry for Fake {
    fn blob_exists(&mut self, digest: &str) -> Result<bool> {
        Ok(self.blobs.contains_key(digest))
    }

    fn put_blob(&mut self, digest: &str, size: u64, blob: Blob<'_>) -> Result<()> {
        if self.fail_after == Some(self.uploads) {
            return Err(CliError::NetworkError("connection reset".into()));
        }
        let bytes = match blob {
            Blob::Bytes(b) => b.to_vec(),
            Blob::File(p) => std::fs::read(p)?,
        };
        if format!("sha256:{}", sha256_hex(&bytes)) != digest || bytes.len() as u64 != size {
            return Err(CliError::NetworkError("DIGEST_INVALID".into()));
        }
        self.uploads += 1;
        self.blobs.insert(digest.into(), bytes);
        Ok(())
    }

    fn tag_digest(&mut self, tag: &str) -> Result<Option<String>> {
        Ok(self.tags.get(tag).cloned())
    }

    fn put_manifest(&mut self, tag: &str, body: &[u8]) -> Result<()> {
        let m: OciManifest = serde_json::from_slice(body).expect("manifest json");
        for d in m.layers.iter().chain([&m.config]) {
            if !self.blobs.contains_key(&d.digest) {
                return Err(CliError::NetworkError("MANIFEST_BLOB_UNKNOWN".into()));
            }
        }
        let digest = format!("sha256:{}", sha256_hex(body));
        self.manifest_puts += 1;
        self.manifests.insert(digest.clone(), body.to_vec());
        self.tags.insert(tag.into(), digest);
        Ok(())
    }

    fn get_manifest(&mut self, reference: &str) -> Result<Vec<u8>> {
        let digest = self.tags.get(reference).map_or(reference, String::as_str);
        self.manifests
            .get(digest)
            .cloned()
            .ok_or_else(|| CliError::NetworkError("HTTP 404".into()))
    }

    fn get_blob(&mut self, digest: &str, to: &Path) -> Result<()> {
        let b = self
            .blobs
            .get(digest)
            .ok_or_else(|| CliError::NetworkError("HTTP 404".into()))?;
        std::fs::write(to, b)?;
        Ok(())
    }
}

const REPO: &str = "ghcr.io/paiml/qwen3.5-4b-apr";

#[test]
fn falsify_ext_017_ghcr_rerun_of_a_completed_push_is_a_no_op() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    let mut reg = Fake::default();
    let first = push(&mut reg, &rel, REPO).expect("first push");
    assert!(!first.no_op);
    assert_eq!(
        reg.uploads,
        FILES.len() + 2,
        "config + manifest layer + files"
    );
    assert_eq!(
        first.tags,
        vec![TagRecord {
            tag: "0.1.0-rc.1".into(),
            action: Action::Created
        }]
    );

    let (uploads, puts) = (reg.uploads, reg.manifest_puts);
    let again = push(&mut reg, &rel, REPO).expect("re-run");
    assert!(
        again.no_op,
        "a re-run of a completed push must send nothing"
    );
    assert_eq!((reg.uploads, reg.manifest_puts), (uploads, puts));
    assert_eq!(again.manifest_digest, first.manifest_digest);
    assert!(again.layers.iter().all(|l| l.action == Action::Present));
}

#[test]
fn falsify_ext_017_ghcr_interrupted_push_resumes_to_the_same_digest() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    let want = push(&mut Fake::default(), &rel, REPO)
        .expect("clean push")
        .manifest_digest;

    let mut reg = Fake {
        fail_after: Some(3),
        ..Fake::default()
    };
    push(&mut reg, &rel, REPO).expect_err("interrupted");
    assert!(reg.tags.is_empty(), "an interrupted push must leave no tag");
    assert_eq!(reg.uploads, 3);

    reg.fail_after = None;
    let r = push(&mut reg, &rel, REPO).expect("resume");
    assert_eq!(
        r.manifest_digest, want,
        "resume must land on the uninterrupted digest"
    );
    assert_eq!(
        reg.uploads,
        FILES.len() + 2,
        "each blob sent exactly once across both runs"
    );
    let resent: Vec<_> = r
        .layers
        .iter()
        .filter(|l| l.action == Action::Uploaded)
        .collect();
    assert_eq!(
        resent.len(),
        FILES.len() + 1 - 2,
        "the resume sends only what was missing"
    );
    assert_eq!(reg.tags.get("0.1.0-rc.1"), Some(&want));
}

#[test]
fn falsify_ext_017_ghcr_version_tag_is_never_moved() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    let mut reg = Fake::default();
    push(&mut reg, &rel, REPO).expect("push");
    let before = reg.tags.clone();

    std::fs::write(rel.join("tuned.apr"), "retrained-weights").expect("write");
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(rel.join(MANIFEST)).expect("read")).expect("json");
    m["files"][3]["sha256"] = json!(sha256_hex(b"retrained-weights"));
    m["files"][3]["bytes"] = json!("retrained-weights".len());
    std::fs::write(
        rel.join(MANIFEST),
        serde_json::to_string_pretty(&m).expect("json"),
    )
    .expect("write");

    let err = push(&mut reg, &rel, REPO).expect_err("same version, new bytes");
    assert!(err.to_string().contains("I-8"), "{err}");
    assert_eq!(reg.tags, before, "the refused push must not touch any tag");
}

#[test]
fn ghcr_released_moves_latest_but_rc_does_not() {
    let t = TempDir::new().expect("tmp");
    let (rc, rel) = (t.path().join("rc"), t.path().join("rel"));
    release(&rc, "rc");
    release(&rel, "released");
    let mut reg = Fake::default();
    push(&mut reg, &rc, REPO).expect("rc");
    assert!(!reg.tags.contains_key(LATEST), "an rc must not move latest");

    let r = push(&mut reg, &rel, REPO).expect("released");
    assert_eq!(reg.tags.get(LATEST), Some(&r.manifest_digest));
    assert_eq!(
        r.tags[1],
        TagRecord {
            tag: LATEST.into(),
            action: Action::Created
        }
    );
    let again = push(&mut reg, &rel, REPO).expect("re-run");
    assert!(again.no_op);
    assert_eq!(again.tags[1].action, Action::Present);
}

#[test]
fn ghcr_release_dir_disagreeing_with_its_manifest_is_refused() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    std::fs::write(rel.join("LICENSE"), "tampered").expect("write");
    let mut reg = Fake::default();
    let err = push(&mut reg, &rel, REPO).expect_err("tampered");
    assert!(err.to_string().contains("LICENSE"), "{err}");
    assert_eq!(
        reg.uploads, 0,
        "nothing is sent for a release the gates did not see"
    );
}

#[test]
fn ghcr_build_is_deterministic() {
    let t = TempDir::new().expect("tmp");
    let (a, b) = (t.path().join("a"), t.path().join("b"));
    release(&a, "rc");
    release(&b, "rc");
    let (x, y) = (build(&a).expect("a"), build(&b).expect("b"));
    assert_eq!(x.manifest, y.manifest);
    assert_eq!(x.digest, format!("sha256:{}", sha256_hex(&x.manifest)));
    let names: Vec<_> = x.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(
        names,
        [MANIFEST, "LICENSE", "NOTICE", "README.md", "tuned.apr"]
    );
}

#[test]
fn falsify_ext_017_ghcr_fetch_feeds_m7_green_and_a_corrupt_blob_is_caught() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    std::fs::write(rel.join(RECEIPT), "{\"gate\":\"green\"}\n").expect("receipt");
    let mut reg = Fake::default();
    push(&mut reg, &rel, REPO).expect("push");

    let got = t.path().join("got");
    fetch(&mut reg, "0.1.0-rc.1", &got).expect("fetch");
    let m = read_manifest(&rel).expect("manifest");
    let checks =
        compare(&m, &std::fs::read(rel.join(MANIFEST)).expect("read"), &got).expect("compare");
    assert!(checks.iter().all(|c| c.ok), "{checks:?}");
    let on_disk: BTreeSet<_> = std::fs::read_dir(&got)
        .expect("dir")
        .map(|e| e.expect("e").file_name().to_string_lossy().into_owned())
        .collect();
    assert!(on_disk.contains(RECEIPT), "the gate receipt rides along");

    let digest = format!("sha256:{}", sha256_hex(b"tuned-weights"));
    reg.blobs.insert(digest.clone(), b"bitflipped-w".to_vec());
    let got2 = t.path().join("got2");
    let err = fetch(&mut reg, "0.1.0-rc.1", &got2).expect_err("corrupt");
    assert!(err.to_string().contains("tuned.apr"), "{err}");
    assert!(
        !got2.exists(),
        "a failed fetch must not leave half a release in `to`"
    );

    reg.blobs.insert(digest, b"tuned-weights".to_vec());
    fetch(&mut reg, "0.1.0-rc.1", &got2).expect("a re-run after the failure starts clean");
    assert!(!t.path().join("got2.partial").exists());
}

/// A registry on a loopback socket that answers like GHCR: 401 with a challenge, then
/// only the credential it expects. Records every Authorization header it was sent.
fn auth_registry(
    challenge: &'static str,
    accept: &'static str,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(8) {
            let mut stream = stream.expect("conn");
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut line = String::new();
            reader.read_line(&mut line).expect("request line");
            let mut auth = String::new();
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).expect("header");
                if h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("authorization") {
                        auth = v.trim().to_string();
                    }
                }
            }
            log.lock().expect("lock").push(auth.clone());
            let reply = if line.starts_with("GET /token") {
                let expected = format!(
                    "Basic {}",
                    base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        format!("apr:{TOKEN}")
                    )
                );
                if auth == expected && line.contains("scope=repository%3Apaiml%2Fm%3Apull%2Cpush") {
                    let body = r#"{"token":"EXCHANGED-registry-token"}"#;
                    format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                } else {
                    "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_string()
                }
            } else if auth == accept {
                "HTTP/1.1 200 OK\r\nDocker-Content-Digest: sha256:abc\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
            } else {
                let ch = challenge.replace("{addr}", &addr.to_string());
                // The body echoes the credential, as a careless registry might.
                let body = format!("denied for {auth}");
                format!("HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: {ch}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
            };
            stream.write_all(reply.as_bytes()).expect("reply");
        }
    });
    (format!("http://{addr}"), seen)
}

#[test]
fn falsify_ext_017_ghcr_bearer_challenge_exchanges_the_file_token() {
    let (base, seen) = auth_registry(
        r#"Bearer realm="http://{addr}/token",service="ghcr.io",scope="repository:paiml/m:pull""#,
        "Bearer EXCHANGED-registry-token",
    );
    let mut reg = HttpRegistry::new(
        base,
        "paiml/m".into(),
        "apr".into(),
        Some(Token(TOKEN.into())),
    );
    assert_eq!(
        reg.tag_digest("0.1.0").expect("authenticated"),
        Some("sha256:abc".into())
    );
    // The exchanged token is reused, not re-exchanged, on the next call.
    assert_eq!(
        reg.tag_digest("0.1.0").expect("reuse"),
        Some("sha256:abc".into())
    );
    let seen = seen.lock().expect("lock").clone();
    assert_eq!(seen.len(), 4, "401, token exchange, retry, reuse: {seen:?}");
    assert_eq!(seen[0], "", "the first request goes out anonymous");
    assert!(
        seen[1].starts_with("Basic "),
        "the file token is only sent to the realm"
    );
    assert_eq!(seen[2], "Bearer EXCHANGED-registry-token");
    assert_eq!(seen[3], "Bearer EXCHANGED-registry-token");
}

#[test]
fn ghcr_basic_challenge_gets_the_file_token_and_a_wrong_one_leaks_nothing() {
    let good = format!(
        "Basic {}",
        base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            format!("apr:{TOKEN}")
        )
    );
    let good: &'static str = Box::leak(good.into_boxed_str());
    let (base, _) = auth_registry(r#"Basic realm="registry""#, good);
    let mut reg = HttpRegistry::new(
        base,
        "paiml/m".into(),
        "apr".into(),
        Some(Token(TOKEN.into())),
    );
    assert_eq!(
        reg.tag_digest("0.1.0").expect("basic"),
        Some("sha256:abc".into())
    );

    let (base, _) = auth_registry(r#"Basic realm="registry""#, "Basic nobody");
    let mut reg = HttpRegistry::new(
        base,
        "paiml/m".into(),
        "apr".into(),
        Some(Token(TOKEN.into())),
    );
    let err = reg.tag_digest("0.1.0").expect_err("rejected").to_string();
    let b64 = good.trim_start_matches("Basic ");
    assert!(err.contains("401"), "{err}");
    assert!(
        !err.contains(TOKEN) && !err.contains(b64),
        "credential leaked: {err}"
    );
}

#[test]
fn ghcr_fetch_refuses_a_path_in_a_layer_title() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    let mut reg = Fake::default();
    push(&mut reg, &rel, REPO).expect("push");
    let digest = reg.tags["0.1.0-rc.1"].clone();
    let mut m: OciManifest = serde_json::from_slice(&reg.manifests[&digest]).expect("m");
    m.layers[1]
        .annotations
        .insert(TITLE.into(), "../escape".into());
    let body = serde_json::to_vec(&m).expect("json");
    reg.put_manifest("evil", &body).expect("put");
    let err = fetch(&mut reg, "evil", &t.path().join("got")).expect_err("traversal");
    assert!(err.to_string().contains("plain file name"), "{err}");
    assert!(!t.path().join("escape").exists());
}

fn token_file(dir: &Path, mode: u32) -> PathBuf {
    let p = dir.join("ghcr-token");
    std::fs::write(&p, format!("{TOKEN}\n")).expect("write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).expect("chmod");
    }
    p
}

#[test]
fn falsify_ext_017_ghcr_token_in_env_is_refused_without_printing_it() {
    let t = TempDir::new().expect("tmp");
    let p = token_file(t.path(), 0o600);
    let ok = read_token(&p, [("PATH".to_string(), "/usr/bin".to_string())]).expect("clean env");
    assert_eq!(ok.expose(), TOKEN);
    assert!(!format!("{ok:?}").contains(TOKEN));

    let env = [("GHCR_TOKEN".to_string(), format!("prefix-{TOKEN}"))];
    let err = read_token(&p, env).expect_err("token in env").to_string();
    assert!(err.contains("R-7") && err.contains("GHCR_TOKEN"), "{err}");
    assert!(!err.contains(TOKEN), "the refusal must not print the token");
}

#[cfg(unix)]
#[test]
fn ghcr_group_readable_token_file_is_refused() {
    let t = TempDir::new().expect("tmp");
    let p = token_file(t.path(), 0o640);
    let err = read_token(&p, std::iter::empty())
        .expect_err("0640")
        .to_string();
    assert!(err.contains("owner-only"), "{err}");
}

#[test]
fn falsify_ext_017_ghcr_receipt_and_errors_carry_no_token() {
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "rc");
    let mut reg = Fake::default();
    let r = push(&mut reg, &rel, REPO).expect("push");
    let path = write_receipt(&t.path().join("state"), &r).expect("receipt");
    let body = std::fs::read_to_string(path).expect("read");
    assert_eq!(body.matches(TOKEN).count(), 0);
    assert_eq!(
        r.max_layer_bytes,
        FILES
            .iter()
            .map(|(_, b)| b.len() as u64)
            .max()
            .unwrap_or(0)
            .max(std::fs::metadata(rel.join(MANIFEST)).expect("meta").len())
    );
    assert_eq!(r.t13_layer_limit_bytes, T13_LAYER_LIMIT_BYTES);

    let tok = read_token(&token_file(t.path(), 0o600), std::iter::empty()).expect("token");
    let http = HttpRegistry::new(
        "https://ghcr.io".into(),
        "paiml/x".into(),
        "apr".into(),
        Some(tok),
    );
    let e = http.fail(
        "blob upload",
        format!("HTTP 400: echoed Authorization: Basic {TOKEN}"),
    );
    assert!(!e.to_string().contains(TOKEN), "{e}");
}

#[test]
fn ghcr_reference_parsing() {
    let (base, repo, tag) = parse_reference("ghcr.io/paiml/qwen3.5-4b-apr:0.1.0-rc.1").expect("ok");
    assert_eq!(
        (base.as_str(), repo.as_str(), tag.as_deref()),
        (
            "https://ghcr.io",
            "paiml/qwen3.5-4b-apr",
            Some("0.1.0-rc.1")
        )
    );
    let (base, _, tag) = parse_reference("localhost:5000/paiml/x").expect("ok");
    assert_eq!((base.as_str(), tag), ("http://localhost:5000", None));
    for bad in [
        "ghcr.io",
        "ghcr.io/Paiml/X",
        "ghcr.io/paiml//x",
        "ghcr.io/paiml/x:-bad",
        "/paiml/x",
    ] {
        assert!(parse_reference(bad).is_err(), "{bad} must be refused");
    }
    for (tag, ok) in [
        ("0.1.0", true),
        ("0.1.0-rc.1", true),
        ("_x", true),
        ("-x", false),
        ("a+b", false),
        ("", false),
    ] {
        assert_eq!(valid_tag(tag), ok, "{tag}");
    }
}

/// The HTTP transport against a real OCI distribution registry (e.g. a local
/// `docker run -p 127.0.0.1:5017:5000 registry:2`). Ignored by default: it needs the
/// registry named in `APR_EXT17_OCI_REGISTRY` (e.g. `localhost:5017`). GHCR itself is
/// the same protocol behind a Bearer token exchange, which needs the driver-host token.
#[test]
#[ignore = "needs a live OCI registry in APR_EXT17_OCI_REGISTRY"]
fn ghcr_http_roundtrip_against_a_live_registry() {
    let host = std::env::var("APR_EXT17_OCI_REGISTRY").expect("APR_EXT17_OCI_REGISTRY");
    let t = TempDir::new().expect("tmp");
    let rel = t.path().join("rel");
    release(&rel, "released");
    // A fresh repository per run: the registry outlives the test.
    let run = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let reference = format!("{host}/paiml/ext17-roundtrip-{run}");
    let (base, repo, _) = parse_reference(&reference).expect("reference");
    let mut reg = HttpRegistry::new(base, repo, "apr".into(), None);

    let first = push(&mut reg, &rel, &reference).expect("first push");
    assert!(!first.no_op);
    let again = push(&mut reg, &rel, &reference).expect("re-run");
    assert!(again.no_op, "{again:?}");
    assert_eq!(again.manifest_digest, first.manifest_digest);

    let got = t.path().join("got");
    fetch(&mut reg, "0.1.0", &got).expect("fetch");
    let m = read_manifest(&rel).expect("manifest");
    let checks =
        compare(&m, &std::fs::read(rel.join(MANIFEST)).expect("read"), &got).expect("compare");
    assert!(checks.iter().all(|c| c.ok), "{checks:?}");
    assert_eq!(
        reg.tag_digest(LATEST).expect("latest"),
        Some(first.manifest_digest.clone())
    );

    std::fs::write(rel.join("README.md"), "# card v2").expect("write");
    let mut mj: serde_json::Value =
        serde_json::from_slice(&std::fs::read(rel.join(MANIFEST)).expect("read")).expect("json");
    mj["files"][2]["sha256"] = json!(sha256_hex(b"# card v2"));
    mj["files"][2]["bytes"] = json!("# card v2".len());
    std::fs::write(
        rel.join(MANIFEST),
        serde_json::to_string_pretty(&mj).expect("json"),
    )
    .expect("write");
    let err = push(&mut reg, &rel, &reference).expect_err("same version, new bytes");
    assert!(err.to_string().contains("I-8"), "{err}");
    assert_eq!(
        reg.tag_digest("0.1.0").expect("tag"),
        Some(first.manifest_digest)
    );
    eprintln!(
        "LIVE OCI roundtrip ok: {reference}:0.1.0 = {}",
        again.manifest_digest
    );
}
