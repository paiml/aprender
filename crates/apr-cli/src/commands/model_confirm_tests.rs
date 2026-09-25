//! EXT-15 (aprender#4397): FALSIFY-EXT-017 — a post-publish mismatch yanks and STOPs,
//! and a yanked version stays fetchable.

use super::*;
use serde_json::json;
use tempfile::TempDir;

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

const FILES: [(&str, &str); 4] = [
    ("LICENSE", "Apache-2.0 text"),
    ("NOTICE", "upstream notice"),
    ("README.md", "# card"),
    ("tuned.apr", "tuned-weights"),
];

/// A release dir as `apr model pack` leaves it.
fn release(dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let files: Vec<_> = FILES
        .iter()
        .map(|(n, b)| {
            std::fs::write(dir.join(n), b).expect("write");
            json!({"name": n, "format": "file", "quant": null, "bytes": b.len(), "sha256": sha(b.as_bytes())})
        })
        .collect();
    let m = json!({
        "schema": "model-release-v1",
        "line": "paiml/qwen3.5-4b-apr",
        "version": "0.1.0-rc.1",
        "channel": "rc",
        "files": files,
        "base": {"hf_id": "Qwen/Qwen3.5-4B", "revision": "main@0123abc", "sha256": sha(b"b")},
        "lineage": ["01RUN"],
        "datasets": [],
        "engine": {"apr_version": "0.71.0", "crate_tarball_sha256": sha(b"t")},
        "gates": {},
        "license": {"spdx_or_name": "Apache-2.0", "upstream_notice_sha256": sha(b"upstream notice")}
    });
    let body = serde_json::to_string_pretty(&m).expect("json");
    std::fs::write(dir.join(MANIFEST), format!("{body}\n")).expect("write");
}

/// A fetch of the published revision: a copy of the release dir.
fn fetch(release: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("mkdir");
    for e in std::fs::read_dir(release).expect("read_dir") {
        let e = e.expect("entry");
        std::fs::copy(e.path(), to.join(e.file_name())).expect("copy");
    }
}

fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| {
            let e = e.expect("entry");
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("read"),
            )
        })
        .collect()
}

struct W {
    _t: TempDir,
    rel: PathBuf,
    state: PathBuf,
    root: PathBuf,
}

fn world() -> W {
    let t = TempDir::new().expect("tempdir");
    let root = t.path().to_path_buf();
    let rel = root.join("rel");
    release(&rel);
    W {
        rel,
        state: root.join("state"),
        root,
        _t: t,
    }
}

const SRC: &str = "hf:paiml/qwen3.5-4b-apr@v0.1.0-rc.1";

/// FALSIFY-EXT-017: each planted post-publish mismatch yanks and STOPs; the release,
/// and the fetched tag, keep every file.
#[test]
fn falsify_ext_017_post_publish_mismatch_yanks() {
    let plants: [(&str, fn(&Path)); 5] = [
        ("tuned.apr", |d| {
            std::fs::write(d.join("tuned.apr"), "tuned-weightz").expect("w")
        }),
        ("LICENSE", |d| {
            std::fs::remove_file(d.join("LICENSE")).expect("rm")
        }),
        ("extra.bin", |d| {
            std::fs::write(d.join("extra.bin"), "x").expect("w")
        }),
        (MANIFEST, |d| {
            let p = d.join(MANIFEST);
            let s = std::fs::read_to_string(&p).expect("r");
            std::fs::write(p, s.replace("0.71.0", "0.71.1")).expect("w");
        }),
        (MANIFEST, |d| {
            std::fs::remove_file(d.join(MANIFEST)).expect("rm")
        }),
    ];
    for (bad, plant) in plants {
        let w = world();
        let before = snapshot(&w.rel);
        let got = w.root.join("got");
        fetch(&w.rel, &got);
        plant(&got);
        let fetched_before = snapshot(&got);

        let err = confirm(&w.rel, &got, SRC, &w.state)
            .expect_err(bad)
            .to_string();
        assert!(
            err.contains("STOP (S-7)") && err.contains(bad),
            "{bad}: {err}"
        );
        assert!(err.contains("; yanked"), "{bad}: {err}");

        let vdir = w.state.join("0.1.0-rc.1");
        let rec: YankRecord =
            serde_json::from_slice(&std::fs::read(vdir.join(YANKED)).expect("yanked.json"))
                .expect("record");
        assert!(rec.reason.contains(bad), "{bad}: {}", rec.reason);
        let receipt = std::fs::read(vdir.join(CONFIRM_RECEIPT)).expect("receipt");
        assert_eq!(rec.receipt_id, format!("sha256:{}", sha(&receipt)));
        let banner_text = std::fs::read_to_string(vdir.join(BANNER)).expect("banner");
        assert!(banner_text.starts_with("> **YANKED:"), "{banner_text}");

        assert_eq!(snapshot(&w.rel), before, "{bad}: a yank edited the release");
        assert_eq!(
            snapshot(&got),
            fetched_before,
            "{bad}: a yank touched the tag"
        );
    }
}

/// A clean fetch is green, writes a receipt, and yanks nothing. Gate files may ride
/// along at the tag.
#[test]
fn a_clean_fetch_is_green() {
    let w = world();
    let got = w.root.join("got");
    fetch(&w.rel, &got);
    std::fs::write(got.join("model-gate-receipt-v1.json"), "{}").expect("w");
    let r = confirm(&w.rel, &got, SRC, &w.state).expect("green");
    assert!(r.green);
    assert_eq!(r.files.len(), FILES.len() + 1);
    let vdir = w.state.join("0.1.0-rc.1");
    assert!(vdir.join(CONFIRM_RECEIPT).exists());
    assert!(!vdir.join(YANKED).exists());

    // Deterministic: a second confirm writes the same receipt.
    let first = std::fs::read(vdir.join(CONFIRM_RECEIPT)).expect("r");
    confirm(&w.rel, &got, SRC, &w.state).expect("green again");
    assert_eq!(std::fs::read(vdir.join(CONFIRM_RECEIPT)).expect("r"), first);
}

/// A yanked version is still fetchable: its tag confirms green after the yank.
#[test]
fn a_yanked_version_stays_fetchable() {
    let w = world();
    let m = read_manifest(&w.rel).expect("manifest");
    yank(&w.state, &m, "0.1.0-rc.1", "bad eval", "sha256:abc").expect("yank");
    let got = w.root.join("got");
    fetch(&w.rel, &got);
    assert!(confirm(&w.rel, &got, SRC, &w.state).expect("green").green);
    assert!(w.state.join("0.1.0-rc.1").join(YANKED).exists());
}

/// A yank is written once, for the manifest's version, with a reason and a receipt.
#[test]
fn a_yank_is_written_once_with_a_reason() {
    let w = world();
    let m = read_manifest(&w.rel).expect("manifest");
    for (version, reason, receipt, want) in [
        ("0.1.0-rc.1", " ", "sha256:abc", "reason and a receipt"),
        ("0.1.0-rc.1", "why", "", "reason and a receipt"),
        ("0.1.0", "why", "sha256:abc", "not the manifest's"),
    ] {
        let err = yank(&w.state, &m, version, reason, receipt)
            .expect_err(want)
            .to_string();
        assert!(err.contains(want), "{err}");
    }
    assert!(!w.state.exists(), "a refused yank wrote state");

    let r = yank(&w.state, &m, "0.1.0-rc.1", " bad eval ", "sha256:abc").expect("yank");
    assert_eq!(r.reason, "bad eval");
    let path = w.state.join("0.1.0-rc.1").join(YANKED);
    let first = std::fs::read(&path).expect("r");
    let err = yank(&w.state, &m, "0.1.0-rc.1", "other", "sha256:def")
        .expect_err("twice")
        .to_string();
    assert!(err.contains("already yanked"), "{err}");
    assert_eq!(
        std::fs::read(&path).expect("r"),
        first,
        "the record was rewritten"
    );
}
