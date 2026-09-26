//! EXT-14 (aprender#4396): the HF publisher is idempotent and resumable, an rc is
//! immutable, a release is promoted from an rc, and the token reaches nothing but
//! the HTTP header. Driven against an in-memory hub; the live hub is NotRun
//! {NoDeclaredExecutor} until a token is provisioned on the driver host (R-7).

use super::super::hf_http::{commit_body, next_link, parse_refs, parse_tree, redact, rev_segment};
use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use tempfile::TempDir;

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

type Tree = BTreeMap<String, RemoteFile>;

/// An in-memory HF repo: `*.apr` goes to LFS, everything else is a regular file.
#[derive(Default)]
struct FakeHub {
    branches: RefCell<BTreeMap<String, String>>,
    tags: RefCell<BTreeMap<String, String>>,
    commits: RefCell<BTreeMap<String, Tree>>,
    lfs: RefCell<BTreeSet<String>>,
    uploads: Cell<usize>,
    /// Fail the upload with this 0-based index.
    fail_upload_at: Cell<Option<usize>>,
    fail_tag_once: Cell<bool>,
}

impl FakeHub {
    fn new() -> Self {
        let h = Self::default();
        let mut t = Tree::new();
        t.insert(KEEP.into(), remote_file(KEEP, b"*.apr filter=lfs\n", false));
        h.commits.borrow_mut().insert("c0".into(), t);
        h.branches.borrow_mut().insert("main".into(), "c0".into());
        h
    }

    fn resolve(&self, rev: &str) -> HubResult<String> {
        if let Some(c) = self.branches.borrow().get(rev) {
            return Ok(c.clone());
        }
        if let Some(c) = self.tags.borrow().get(rev) {
            return Ok(c.clone());
        }
        if self.commits.borrow().contains_key(rev) {
            return Ok(rev.to_string());
        }
        Err(format!("no rev {rev}"))
    }

    fn n_commits(&self) -> usize {
        self.commits.borrow().len() - 1
    }

    fn files_at(&self, rev: &str) -> BTreeMap<String, String> {
        self.tree(rev)
            .expect("rev")
            .into_iter()
            .map(|f| (f.path, f.lfs_sha256.unwrap_or(f.git_oid)))
            .collect()
    }
}

fn remote_file(path: &str, bytes: &[u8], lfs: bool) -> RemoteFile {
    RemoteFile {
        path: path.into(),
        size: bytes.len() as u64,
        git_oid: if lfs {
            git_blob_oid(b"pointer")
        } else {
            git_blob_oid(bytes)
        },
        lfs_sha256: lfs.then(|| sha(bytes)),
    }
}

impl Hub for FakeHub {
    fn refs(&self) -> HubResult<Refs> {
        Ok(Refs {
            branches: self.branches.borrow().clone(),
            tags: self.tags.borrow().clone(),
        })
    }

    fn tree(&self, rev: &str) -> HubResult<Vec<RemoteFile>> {
        let c = self.resolve(rev)?;
        Ok(self.commits.borrow()[&c].values().cloned().collect())
    }

    fn preupload(&self, _rev: &str, files: &[Upload<'_>]) -> HubResult<Vec<bool>> {
        Ok(files.iter().map(|f| f.path.ends_with(".apr")).collect())
    }

    fn upload_lfs(&self, sha256: &str, size: u64, file: &Path) -> HubResult<bool> {
        if self.lfs.borrow().contains(sha256) {
            return Ok(false);
        }
        let i = self.uploads.get();
        self.uploads.set(i + 1);
        if self.fail_upload_at.get() == Some(i) {
            return Err("connection reset".into());
        }
        let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
        assert_eq!(
            (sha(&bytes), bytes.len() as u64),
            (sha256.to_string(), size)
        );
        self.lfs.borrow_mut().insert(sha256.into());
        Ok(true)
    }

    fn create_branch(&self, branch: &str, from: &str) -> HubResult<()> {
        let c = self.resolve(from)?;
        let mut b = self.branches.borrow_mut();
        if b.contains_key(branch) {
            return Err(format!("{branch} exists"));
        }
        b.insert(branch.into(), c);
        Ok(())
    }

    fn commit(&self, rev: &str, _summary: &str, ops: &[Op]) -> HubResult<String> {
        let head = self
            .branches
            .borrow()
            .get(rev)
            .cloned()
            .ok_or_else(|| format!("{rev} is not a branch"))?;
        let mut tree = self.commits.borrow()[&head].clone();
        for op in ops {
            match op {
                Op::Lfs { path, sha256, size } => {
                    if !self.lfs.borrow().contains(sha256) {
                        return Err(format!("{path}: LFS object not uploaded"));
                    }
                    tree.insert(
                        path.clone(),
                        RemoteFile {
                            path: path.clone(),
                            size: *size,
                            git_oid: git_blob_oid(b"pointer"),
                            lfs_sha256: Some(sha256.clone()),
                        },
                    );
                }
                Op::File { path, bytes } => {
                    tree.insert(path.clone(), remote_file(path, bytes, false));
                }
                Op::Delete { path } => {
                    tree.remove(path)
                        .ok_or_else(|| format!("{path}: not there"))?;
                }
            }
        }
        let id = format!("c{}", self.commits.borrow().len());
        self.commits.borrow_mut().insert(id.clone(), tree);
        self.branches.borrow_mut().insert(rev.into(), id.clone());
        Ok(id)
    }

    fn create_tag(&self, rev: &str, tag: &str) -> HubResult<()> {
        if self.fail_tag_once.replace(false) {
            return Err("502".into());
        }
        let c = self.resolve(rev)?;
        let mut t = self.tags.borrow_mut();
        if t.contains_key(tag) {
            return Err(format!("{tag} exists"));
        }
        t.insert(tag.into(), c);
        Ok(())
    }
}

/// A release dir as `apr model pack` leaves it.
fn release(dir: &Path, version: &str, channel: &str, weights: &str) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let files: Vec<(&str, &str)> = vec![
        ("LICENSE", "Apache-2.0 text"),
        ("NOTICE", "upstream notice"),
        ("README.md", "# card"),
        ("tokenizer.apr", "tok"),
        ("tuned.apr", weights),
    ];
    let listed: Vec<_> = files
        .iter()
        .map(|(n, b)| {
            std::fs::write(dir.join(n), b).expect("write");
            json!({"name": n, "format": "file", "quant": null, "bytes": b.len(), "sha256": sha(b.as_bytes())})
        })
        .collect();
    let m = json!({
        "schema": "model-release-v1",
        "line": "paiml/qwen3.5-4b-apr",
        "version": version,
        "channel": channel,
        "files": listed,
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

const REPO: &str = "paiml/qwen3.5-4b-apr";

fn go(hub: &FakeHub, dir: &Path) -> Result<PublishReceipt> {
    publish(hub, REPO, dir, &mut Vec::new())
}

fn rc(t: &TempDir, name: &str, weights: &str) -> PathBuf {
    let d = t.path().join(name);
    release(&d, "0.1.0-rc.1", "rc", weights);
    d
}

/// An rc lands in one commit on its branch; the rerun is a no-op; main is untouched
/// and `.gitattributes` survives.
#[test]
fn an_rc_publish_reruns_as_a_noop() {
    let t = TempDir::new().expect("tmp");
    let d = rc(&t, "rel", "w1");
    let hub = FakeHub::new();
    let r = go(&hub, &d).expect("publish");
    assert_eq!(
        (r.target.as_str(), r.action.as_str()),
        ("rc/v0.1.0-rc.1", "committed")
    );
    assert_eq!(r.uploaded, ["tokenizer.apr", "tuned.apr"]);
    assert_eq!(r.files.len(), 6, "5 files + manifest");
    assert_eq!(hub.n_commits(), 1);
    let at = hub.files_at("rc/v0.1.0-rc.1");
    assert_eq!(at["tuned.apr"], sha(b"w1"));
    assert!(at.contains_key(KEEP));
    assert_eq!(hub.files_at("main").len(), 1, "an rc moved main");

    let again = go(&hub, &d).expect("rerun");
    assert_eq!((again.action.as_str(), again.commit), ("noop", None));
    assert!(again.uploaded.is_empty());
    assert_eq!((hub.n_commits(), hub.uploads.get()), (1, 2));
}

/// An upload interrupted mid-release resumes: nothing is committed until every LFS
/// object is stored, stored objects are not re-sent, and the release is ONE commit.
#[test]
fn an_interrupted_upload_resumes_to_one_commit() {
    let t = TempDir::new().expect("tmp");
    let d = rc(&t, "rel", "w1");
    let hub = FakeHub::new();
    hub.fail_upload_at.set(Some(1));
    let err = go(&hub, &d).expect_err("interrupted").to_string();
    assert!(err.contains("connection reset"), "{err}");
    assert_eq!(hub.n_commits(), 0, "a partial revision was committed");

    let mut log = Vec::new();
    let r = publish(&hub, REPO, &d, &mut log).expect("resume");
    assert_eq!(r.action, "committed");
    assert_eq!(r.uploaded, ["tuned.apr"], "only the missing object is sent");
    assert!(
        log.iter().any(|l| l == "tokenizer.apr: already stored"),
        "{log:?}"
    );
    assert_eq!(hub.n_commits(), 1);
    assert_eq!(go(&hub, &d).expect("rerun").action, "noop");
}

/// An rc is immutable: other bytes under the same version are refused, and a branch
/// whose files drifted from its manifest is refused (M7's job), never overwritten.
#[test]
fn an_rc_is_immutable() {
    let t = TempDir::new().expect("tmp");
    let hub = FakeHub::new();
    go(&hub, &rc(&t, "a", "w1")).expect("publish");
    let before = hub.files_at("rc/v0.1.0-rc.1");

    let err = go(&hub, &rc(&t, "b", "w2")).expect_err("same version, other bytes");
    assert!(err.to_string().contains("immutable"), "{err}");

    let head = hub.branches.borrow()["rc/v0.1.0-rc.1"].clone();
    hub.commits
        .borrow_mut()
        .get_mut(&head)
        .expect("head")
        .insert(
            "README.md".into(),
            remote_file("README.md", b"# edited", false),
        );
    let err = go(&hub, &rc(&t, "c", "w1")).expect_err("tampered");
    assert!(err.to_string().contains("apr model confirm"), "{err}");
    assert_eq!(hub.n_commits(), 1);
    assert_ne!(hub.files_at("rc/v0.1.0-rc.1"), before, "the plant took");
}

/// A release is promoted from an rc: main carries it, the tag points at main's head,
/// and the rerun is a no-op. With no rc, nothing is promoted.
#[test]
fn a_release_is_promoted_from_an_rc() {
    let t = TempDir::new().expect("tmp");
    let hub = FakeHub::new();
    let released = t.path().join("rel");
    release(&released, "0.1.0", "released", "w1");
    let err = go(&hub, &released).expect_err("no rc");
    assert!(err.to_string().contains("promoted from an rc"), "{err}");
    assert_eq!(hub.n_commits(), 0);

    go(&hub, &rc(&t, "rc", "w1")).expect("rc");
    let r = go(&hub, &released).expect("promote");
    assert_eq!(
        (r.target.as_str(), r.action.as_str()),
        ("v0.1.0", "committed+tagged")
    );
    assert!(
        r.uploaded.is_empty(),
        "the rc already stored every LFS object"
    );
    let main = hub.branches.borrow()["main"].clone();
    assert_eq!(hub.tags.borrow()["v0.1.0"], main);
    assert_eq!(r.commit.as_deref(), Some(main.as_str()));
    assert_eq!(hub.files_at("v0.1.0"), hub.files_at("main"));
    assert_eq!(hub.files_at("main")["tuned.apr"], sha(b"w1"));

    let again = go(&hub, &released).expect("rerun");
    assert_eq!(again.action, "noop");
    assert_eq!(hub.n_commits(), 2);
}

/// A promote interrupted between its commit and its tag resumes by tagging only.
#[test]
fn a_promote_interrupted_before_the_tag_resumes() {
    let t = TempDir::new().expect("tmp");
    let hub = FakeHub::new();
    go(&hub, &rc(&t, "rc", "w1")).expect("rc");
    let released = t.path().join("rel");
    release(&released, "0.1.0", "released", "w1");
    hub.fail_tag_once.set(true);
    go(&hub, &released).expect_err("tag failed");
    assert!(hub.tags.borrow().is_empty());
    let r = go(&hub, &released).expect("resume");
    assert_eq!(r.action, "tagged");
    assert_eq!(hub.n_commits(), 2, "the resume committed again");
    assert_eq!(hub.tags.borrow()["v0.1.0"], hub.branches.borrow()["main"]);
}

/// main is never moved back under a newer release, and a tag is never moved.
#[test]
fn main_is_not_moved_back_and_tags_do_not_move() {
    let t = TempDir::new().expect("tmp");
    let hub = FakeHub::new();
    go(&hub, &rc(&t, "rc", "w1")).expect("rc");
    hub.tags.borrow_mut().insert("v0.10.0".into(), "c0".into());
    let released = t.path().join("rel");
    release(&released, "0.1.0", "released", "w1");
    let err = go(&hub, &released).expect_err("older");
    assert!(
        err.to_string().contains("v0.10.0 is released and newer"),
        "{err}"
    );

    hub.tags.borrow_mut().clear();
    hub.tags.borrow_mut().insert("v0.1.0".into(), "c0".into());
    let err = go(&hub, &released).expect_err("tag holds other bytes");
    assert!(err.to_string().contains("immutable"), "{err}");
    assert_eq!(hub.n_commits(), 1);
}

/// Only what the manifest vouches for is published.
#[test]
fn an_unvouched_release_dir_is_refused() {
    let t = TempDir::new().expect("tmp");
    for (what, plant) in [
        (
            "not listed",
            (|d: &Path| std::fs::write(d.join("x.bin"), "x").expect("w")) as fn(&Path),
        ),
        ("bytes differ", |d| {
            std::fs::write(d.join("tuned.apr"), "w9").expect("w")
        }),
        ("missing", |d| {
            std::fs::remove_file(d.join("LICENSE")).expect("rm")
        }),
    ] {
        let d = rc(&t, what, "w1");
        plant(&d);
        let hub = FakeHub::new();
        let err = go(&hub, &d).expect_err(what).to_string();
        assert!(err.contains(what), "{what}: {err}");
        assert_eq!((hub.n_commits(), hub.uploads.get()), (0, 0), "{what}");
        assert_eq!(
            hub.branches.borrow().len(),
            1,
            "{what}: a branch was created"
        );
    }
}

/// R-7: the token comes from a 0600 file only, prints redacted, and neither module
/// reads the environment.
#[test]
fn the_token_stays_in_its_file() {
    use std::os::unix::fs::PermissionsExt;
    let t = TempDir::new().expect("tmp");
    let p = t.path().join("tok");
    std::fs::write(&p, "hf_SECRETvalue\n").expect("w");
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).expect("chmod");
    let err = Token::from_file(&p).expect_err("0644").to_string();
    assert!(err.contains("0600") && !err.contains("SECRET"), "{err}");

    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let tok = Token::from_file(&p).expect("0600");
    assert_eq!(tok.bearer(), "Bearer hf_SECRETvalue");
    assert!(!format!("{tok:?}").contains("SECRET"));

    for body in ["", "  \n", "two tokens"] {
        std::fs::write(&p, body).expect("w");
        let err = Token::from_file(&p).expect_err(body).to_string();
        assert!(err.contains("one token"), "{err}");
    }
    for src in [include_str!("hf_publish.rs"), include_str!("hf_http.rs")] {
        assert!(
            !src.contains(concat!("env", "::var")),
            "a publisher module reads env"
        );
    }
}

#[test]
fn wire_formats() {
    assert_eq!(
        git_blob_oid(b""),
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    );
    assert_eq!(
        git_blob_oid(b"hello\n"),
        "ce013625030ba8dba906f756967f9e9ca394464a"
    );
    assert_eq!(
        redact("https://s3.x/obj?X-Amz-Signature=abc#f"),
        "https://s3.x/obj"
    );
    assert_eq!(rev_segment("rc/v0.1.0-rc.1"), "rc%2Fv0.1.0-rc.1");
    assert_eq!(
        next_link(Some("<https://h/api?cursor=2>; rel=\"next\"")).as_deref(),
        Some("https://h/api?cursor=2")
    );
    assert_eq!(next_link(Some("<https://h/p1>; rel=\"prev\"")), None);

    let tree = parse_tree(&json!([
        {"type": "file", "path": "a.apr", "size": 3, "oid": "p", "lfs": {"oid": "s", "size": 3}},
        {"type": "file", "path": "README.md", "size": 6, "oid": "g"},
        {"type": "directory", "path": "d", "oid": "t"}
    ]))
    .expect("tree");
    assert_eq!(tree.len(), 2);
    assert_eq!(tree[0].lfs_sha256.as_deref(), Some("s"));
    assert_eq!(tree[1].lfs_sha256, None);

    let refs = parse_refs(
        &json!({"branches": [{"name": "main", "targetCommit": "c1"}],
        "tags": [{"name": "v1.0.0", "targetCommit": "c2"}], "converts": []}),
    );
    assert_eq!(refs.branches["main"], "c1");
    assert_eq!(refs.tags["v1.0.0"], "c2");

    let body = commit_body(
        "s",
        &[
            Op::Lfs {
                path: "a.apr".into(),
                sha256: "ab".into(),
                size: 3,
            },
            Op::File {
                path: "R".into(),
                bytes: b"hi".to_vec(),
            },
            Op::Delete { path: "old".into() },
        ],
    );
    let lines: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).expect("ndjson line"))
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["key"], "header");
    assert_eq!(lines[1]["value"]["oid"], "ab");
    assert_eq!(lines[2]["value"]["content"], "aGk=");
    assert_eq!(lines[3]["key"], "deletedFile");
}
