//! `apr model publish`: the Rust HF publisher (EXT-001 §3.5/§3.8, row EXT-14,
//! aprender#4396 collapsed into #4393). No `huggingface_hub`, no `hf` CLI, no Python (R-1).
//!
//! - `rc` releases go to the HF branch `rc/vX.Y.Z-rc.N`, created from `main`. The
//!   branch is immutable once it carries the release: pushing different bytes under
//!   the same version is refused (a fix is `rc.N+1`).
//! - `released` releases are promoted: a commit makes `main` carry the release, then
//!   the tag `vX.Y.Z` is created at `main`'s head. A tag is never moved (I-8).
//!
//! Idempotent and resumable: every step first diffs the remote tree against the
//! release dir (LFS files by sha256, regular files by git blob id), so a completed
//! publish re-runs as a no-op. LFS objects are content-addressed and skipped when
//! already stored, and the release lands in ONE commit, so an interrupted run
//! resumes to the same tree with no partial revision in between.
//!
//! The token is read from a file only — mode 0600, on the driver host (R-7) — never
//! from the environment, and it has no `Display`: it can reach an HTTP header and
//! nothing else (receipt, log, error).

use super::model_confirm::{read_manifest, version_dir};
use super::model_gate::{sha256_file, ReleaseManifest, MANIFEST, RECEIPT};
use super::model_pack::check_version;
use crate::error::{CliError, Result};
use serde::Serialize;
use sha1::{Digest as _, Sha1};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The receipt a publish that changed something writes under `state/<version>/`.
pub(crate) const PUBLISH_RECEIPT: &str = "model-publish-receipt-v1.json";
/// Created by HF with every repo; never compared, never deleted.
const KEEP: &str = ".gitattributes";

/// What a hub call returns; the message never carries the token.
pub(crate) type HubResult<T> = std::result::Result<T, String>;

/// A publish token. It prints as `<redacted>` and has no `Display`.
pub(crate) struct Token(String);

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

impl Token {
    /// Read the token from `path`, which only its owner may read.
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let shown = path.display();
        let meta = std::fs::metadata(path).map_err(|e| invalid(format!("{shown}: {e}")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o077 != 0 {
                return Err(invalid(format!(
                    "{shown}: readable by group or other; the token file must be mode 0600 (R-7)"
                )));
            }
        }
        #[cfg(not(unix))]
        let _ = meta;
        let body = std::fs::read_to_string(path).map_err(|e| invalid(format!("{shown}: {e}")))?;
        let t = body.trim();
        if t.is_empty() || t.contains(char::is_whitespace) {
            return Err(invalid(format!("{shown}: does not hold one token")));
        }
        Ok(Self(t.to_string()))
    }

    /// The `Authorization` header value.
    pub(crate) fn bearer(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

/// Branch and tag heads, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Refs {
    pub branches: BTreeMap<String, String>,
    pub tags: BTreeMap<String, String>,
}

/// One file of a remote tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemoteFile {
    pub path: String,
    pub size: u64,
    /// The git blob id (of the LFS pointer, for an LFS file).
    pub git_oid: String,
    /// The LFS object's sha256, for an LFS file.
    pub lfs_sha256: Option<String>,
}

/// A file offered to the hub's preupload check.
#[derive(Debug, Clone)]
pub(crate) struct Upload<'a> {
    pub path: &'a str,
    pub size: u64,
    /// The first 512 bytes.
    pub sample: Vec<u8>,
}

/// One operation of a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Op {
    Lfs {
        path: String,
        sha256: String,
        size: u64,
    },
    File {
        path: String,
        bytes: Vec<u8>,
    },
    Delete {
        path: String,
    },
}

/// The HF Hub API, as the publisher uses it.
pub(crate) trait Hub {
    fn refs(&self) -> HubResult<Refs>;
    /// Every file at `rev` (branch, tag or commit).
    fn tree(&self, rev: &str) -> HubResult<Vec<RemoteFile>>;
    /// For each file, `true` when it goes to LFS.
    fn preupload(&self, rev: &str, files: &[Upload<'_>]) -> HubResult<Vec<bool>>;
    /// Store one LFS object; `false` when it was already stored.
    fn upload_lfs(&self, sha256: &str, size: u64, file: &Path) -> HubResult<bool>;
    fn create_branch(&self, branch: &str, from: &str) -> HubResult<()>;
    /// One atomic commit on branch `rev`; returns the commit id.
    fn commit(&self, rev: &str, summary: &str, ops: &[Op]) -> HubResult<String>;
    fn create_tag(&self, rev: &str, tag: &str) -> HubResult<()>;
}

fn invalid(msg: impl Into<String>) -> CliError {
    CliError::ValidationFailed(msg.into())
}

fn hub_err(what: &str) -> impl Fn(String) -> CliError + '_ {
    move |e| invalid(format!("hub: {what}: {e}"))
}

/// The git blob id of `bytes`: sha1 of `blob <len>\0<bytes>`.
pub(crate) fn git_blob_oid(bytes: &[u8]) -> String {
    let mut h = Sha1::new();
    h.update(format!("blob {}\0", bytes.len()).as_bytes());
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn git_blob_oid_file(path: &Path, size: u64) -> std::io::Result<String> {
    let mut h = Sha1::new();
    h.update(format!("blob {size}\0").as_bytes());
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// One file of the release dir, hashed.
#[derive(Debug, Clone)]
pub(crate) struct Local {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub sha256: String,
    pub git_oid: String,
}

/// The files to publish: exactly the manifest's files, each re-hashed against it,
/// plus the manifest and (when present) the gate receipt.
pub(crate) fn local_files(dir: &Path, m: &ReleaseManifest) -> Result<Vec<Local>> {
    let listed: BTreeMap<&str, _> = m.files.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| invalid(format!("{}: {e}", dir.display())))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        let (size, sha256) = sha256_file(&path)?;
        match listed.get(name.as_str()) {
            Some(f) if f.sha256 != sha256 || f.bytes != size => {
                return Err(invalid(format!(
                    "{name}: bytes differ from the manifest; only what the manifest vouches for is published"
                )))
            }
            Some(_) => {}
            None if name == MANIFEST || name == RECEIPT => {}
            None => {
                return Err(invalid(format!(
                    "{name}: not listed in {MANIFEST}; the release dir is not publishable as is"
                )))
            }
        }
        out.push(Local {
            git_oid: git_blob_oid_file(&path, size)?,
            name,
            path,
            size,
            sha256,
        });
    }
    for f in &m.files {
        if !out.iter().any(|l| l.name == f.name) {
            return Err(invalid(format!(
                "{}: listed in the manifest but missing",
                f.name
            )));
        }
    }
    if !out.iter().any(|l| l.name == MANIFEST) {
        return Err(invalid(format!("{MANIFEST}: missing")));
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn same(r: &RemoteFile, l: &Local) -> bool {
    match &r.lfs_sha256 {
        Some(s) => *s == l.sha256 && r.size == l.size,
        None => r.git_oid == l.git_oid,
    }
}

/// What a sync of one ref would change.
struct Plan<'a> {
    put: Vec<&'a Local>,
    delete: Vec<String>,
}

impl Plan<'_> {
    fn is_empty(&self) -> bool {
        self.put.is_empty() && self.delete.is_empty()
    }
}

fn plan<'a>(remote: &[RemoteFile], local: &'a [Local]) -> Plan<'a> {
    let by_path: BTreeMap<&str, &RemoteFile> =
        remote.iter().map(|r| (r.path.as_str(), r)).collect();
    Plan {
        put: local
            .iter()
            .filter(|l| by_path.get(l.name.as_str()).is_none_or(|r| !same(r, l)))
            .collect(),
        delete: remote
            .iter()
            .filter(|r| r.path != KEEP && !local.iter().any(|l| l.name == r.path))
            .map(|r| r.path.clone())
            .collect(),
    }
}

fn sample(path: &Path) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(512);
    std::fs::File::open(path)?.take(512).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Upload what `p` puts and commit it, with `p`'s deletes, to branch `rev`.
fn apply(
    hub: &dyn Hub,
    rev: &str,
    p: &Plan<'_>,
    summary: &str,
    log: &mut Vec<String>,
    uploaded: &mut Vec<String>,
) -> Result<String> {
    let uploads = p
        .put
        .iter()
        .map(|l| {
            Ok(Upload {
                path: &l.name,
                size: l.size,
                sample: sample(&l.path)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let modes = hub.preupload(rev, &uploads).map_err(hub_err("preupload"))?;
    if modes.len() != p.put.len() {
        return Err(invalid("hub: preupload answered for a different file set"));
    }
    let mut ops = Vec::new();
    for (l, lfs) in p.put.iter().zip(modes) {
        if lfs {
            if hub
                .upload_lfs(&l.sha256, l.size, &l.path)
                .map_err(hub_err(&l.name))?
            {
                log.push(format!("uploaded {} ({} bytes)", l.name, l.size));
                uploaded.push(l.name.clone());
            } else {
                log.push(format!("{}: already stored", l.name));
            }
            ops.push(Op::Lfs {
                path: l.name.clone(),
                sha256: l.sha256.clone(),
                size: l.size,
            });
        } else {
            ops.push(Op::File {
                path: l.name.clone(),
                bytes: std::fs::read(&l.path)?,
            });
        }
    }
    ops.extend(
        p.delete
            .iter()
            .map(|path| Op::Delete { path: path.clone() }),
    );
    let commit = hub.commit(rev, summary, &ops).map_err(hub_err("commit"))?;
    log.push(format!("committed {commit} on {rev}"));
    Ok(commit)
}

/// One published file, as the receipt lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct PublishedFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// `model-publish-receipt-v1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct PublishReceipt {
    pub schema: &'static str,
    pub repo: String,
    pub line: String,
    pub version: String,
    pub channel: String,
    /// The branch (rc) or tag (released) that carries the release.
    pub target: String,
    /// `noop`, `committed`, `tagged` or `committed+tagged`.
    pub action: String,
    pub commit: Option<String>,
    /// LFS objects this run stored (a resumed run lists only what it added).
    pub uploaded: Vec<String>,
    pub files: Vec<PublishedFile>,
}

fn remote_manifest<'a>(tree: &'a [RemoteFile]) -> Option<&'a RemoteFile> {
    tree.iter().find(|r| r.path == MANIFEST)
}

fn semver(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.split('.').map(|p| p.parse::<u64>().ok());
    let t = (it.next()??, it.next()??, it.next()??);
    it.next().is_none().then_some(t)
}

/// Publish the release in `dir` to `repo` through `hub`: an rc to its branch, a
/// released version to `main` plus its tag.
///
/// # Errors
///
/// `ValidationFailed` on a release dir the manifest does not vouch for, an immutable
/// ref that already carries different bytes, a promote with no rc, or a hub error.
pub(crate) fn publish(
    hub: &dyn Hub,
    repo: &str,
    dir: &Path,
    log: &mut Vec<String>,
) -> Result<PublishReceipt> {
    let m = read_manifest(dir)?;
    check_version(&m.version, &m.channel)?;
    let local = local_files(dir, &m)?;
    let refs = hub.refs().map_err(hub_err("refs"))?;
    if !refs.branches.contains_key("main") {
        return Err(invalid(format!(
            "{repo}: no main branch; create the repo first"
        )));
    }
    let manifest = local
        .iter()
        .find(|l| l.name == MANIFEST)
        .ok_or_else(|| invalid("no manifest"))?;
    let mut uploaded = Vec::new();
    let (target, action, commit) = if m.channel == "rc" {
        let branch = format!("rc/v{}", m.version);
        let summary = format!("{} v{} (rc)", m.line, m.version);
        if refs.branches.contains_key(&branch) {
            let tree = hub.tree(&branch).map_err(hub_err(&branch))?;
            let p = plan(&tree, &local);
            match remote_manifest(&tree) {
                Some(r) if !same(r, manifest) => {
                    return Err(invalid(format!(
                        "{branch} already carries a different {MANIFEST}; an rc is immutable once pushed, a fix is the next -rc.N"
                    )))
                }
                Some(_) if !p.is_empty() => {
                    return Err(invalid(format!(
                        "{branch} carries this manifest but other bytes; run `apr model confirm` (M7) and yank"
                    )))
                }
                Some(_) => (branch, "noop", None),
                None => {
                    let c = apply(hub, &branch, &p, &summary, log, &mut uploaded)?;
                    (branch, "committed", Some(c))
                }
            }
        } else {
            hub.create_branch(&branch, "main")
                .map_err(hub_err(&branch))?;
            log.push(format!("created branch {branch} from main"));
            let tree = hub.tree(&branch).map_err(hub_err(&branch))?;
            let c = apply(
                hub,
                &branch,
                &plan(&tree, &local),
                &summary,
                log,
                &mut uploaded,
            )?;
            (branch, "committed", Some(c))
        }
    } else {
        let tag = format!("v{}", m.version);
        let rc = format!("rc/v{}-rc.", m.version);
        if !refs.branches.keys().any(|b| b.starts_with(&rc)) {
            return Err(invalid(format!(
                "no {rc}N branch: a release is promoted from an rc"
            )));
        }
        if refs.tags.contains_key(&tag) {
            let tree = hub.tree(&tag).map_err(hub_err(&tag))?;
            if !plan(&tree, &local).is_empty() {
                return Err(invalid(format!(
                    "tag {tag} exists with other bytes; a released tag is immutable (I-8)"
                )));
            }
            (tag, "noop", None)
        } else {
            let ours = semver(&m.version);
            if let Some(newer) = refs
                .tags
                .keys()
                .filter_map(|t| t.strip_prefix('v'))
                .filter(|v| semver(v) > ours)
                .max_by_key(|v| semver(v))
            {
                return Err(invalid(format!(
                    "v{newer} is released and newer; main is not moved back to v{}",
                    m.version
                )));
            }
            let tree = hub.tree("main").map_err(hub_err("main"))?;
            let p = plan(&tree, &local);
            let summary = format!("{} v{}", m.line, m.version);
            let commit = if p.is_empty() {
                None
            } else {
                Some(apply(hub, "main", &p, &summary, log, &mut uploaded)?)
            };
            let head = hub
                .refs()
                .map_err(hub_err("refs"))?
                .branches
                .get("main")
                .cloned()
                .ok_or_else(|| invalid("main vanished"))?;
            hub.create_tag(&head, &tag).map_err(hub_err(&tag))?;
            log.push(format!("tagged {tag} at {head}"));
            let action = if commit.is_some() {
                "committed+tagged"
            } else {
                "tagged"
            };
            (tag, action, Some(commit.unwrap_or(head)))
        }
    };
    Ok(PublishReceipt {
        schema: "model-publish-receipt-v1",
        repo: repo.into(),
        line: m.line.clone(),
        version: m.version.clone(),
        channel: m.channel.clone(),
        target,
        action: action.into(),
        commit,
        uploaded,
        files: local
            .iter()
            .map(|l| PublishedFile {
                name: l.name.clone(),
                bytes: l.size,
                sha256: l.sha256.clone(),
            })
            .collect(),
    })
}

/// `owner/name`, each `[A-Za-z0-9._-]+`.
fn check_repo(repo: &str) -> Result<()> {
    let ok = |s: &str| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    match repo.split_once('/') {
        Some((o, n)) if ok(o) && ok(n) => Ok(()),
        _ => Err(invalid(format!("repo {repo:?}: want owner/name"))),
    }
}

/// `apr model publish`.
pub(crate) fn run_publish(
    dir: &Path,
    repo: &str,
    token_file: &Path,
    state: &Path,
    endpoint: &str,
    json: bool,
) -> Result<()> {
    check_repo(repo)?;
    let token = Token::from_file(token_file)?;
    let hub = super::hf_http::HfHttp::new(endpoint, repo, token);
    let mut log = Vec::new();
    let result = publish(&hub, repo, dir, &mut log);
    for line in &log {
        eprintln!("{line}");
    }
    let r = result?;
    if r.action != "noop" {
        let vdir = version_dir(state, &r.version)?;
        std::fs::create_dir_all(&vdir)?;
        let body = serde_json::to_string_pretty(&r).map_err(|e| invalid(e.to_string()))?;
        std::fs::write(vdir.join(PUBLISH_RECEIPT), format!("{body}\n"))?;
    }
    if json {
        let s = serde_json::to_string_pretty(&r).map_err(|e| invalid(e.to_string()))?;
        println!("{s}");
    } else {
        println!(
            "{} {} -> {}:{} ({})",
            r.line, r.version, repo, r.target, r.action
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "hf_publish_tests.rs"]
mod tests;
