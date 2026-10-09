//! The HF Hub write endpoints of one model repo, at a named revision (#4961).
//!
//! This is the one HF upload code path behind the `apr` binary: `apr publish`
//! (through [`HfHubClient`](super::HfHubClient)) and `apr model publish` (EXT-14)
//! both call it. It covers refs, the recursive tree, preupload, the LFS batch
//! upload, branch and tag creation, and the NDJSON commit, each at a revision
//! the caller names (`main`, or an rc branch such as `rc/v0.1.0-rc.1`).
//!
//! Errors name the method and the URL path with its query stripped, so a
//! presigned upload URL's signature is never printed, and never the token.
//! `RepoApi`'s `Debug` prints the token as `<redacted>`.

use super::{base64_encode, HfHubError, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

const LFS_JSON: &str = "application/vnd.git-lfs+json";

/// Branch and tag names of a repo, each mapped to its target commit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Refs {
    /// Branch name -> commit id.
    pub branches: BTreeMap<String, String>,
    /// Tag name -> commit id.
    pub tags: BTreeMap<String, String>,
}

/// One file of a remote tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    /// Path in the repo.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// The git blob id (for an LFS file, the id of its pointer).
    pub git_oid: String,
    /// The LFS object's sha256, for an LFS file.
    pub lfs_sha256: Option<String>,
}

/// A file offered to the preupload check: its path, size and first bytes.
#[derive(Debug, Clone, Copy)]
pub struct PreuploadFile<'a> {
    /// Path in the repo.
    pub path: &'a str,
    /// Size in bytes.
    pub size: u64,
    /// The first bytes of the file; at most 512 are sent.
    pub sample: &'a [u8],
}

/// One operation of a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitOp {
    /// Point `path` at an LFS object already uploaded.
    Lfs {
        /// Path in the repo.
        path: String,
        /// The object's sha256, lowercase hex.
        sha256: String,
        /// The object's size in bytes.
        size: u64,
    },
    /// Write `bytes` to `path` as a regular git file.
    File {
        /// Path in the repo.
        path: String,
        /// The file's content.
        bytes: Vec<u8>,
    },
    /// Delete `path`.
    Delete {
        /// Path in the repo.
        path: String,
    },
}

/// The bytes of an LFS object: in memory, or a file streamed from disk.
#[derive(Debug, Clone, Copy)]
pub enum LfsBody<'a> {
    /// Bytes already in memory.
    Bytes(&'a [u8]),
    /// A file, streamed and never read into memory whole.
    File(&'a Path),
}

/// One model repo on the Hub at `api_base`, with a write token.
pub struct RepoApi {
    api_base: String,
    repo_id: String,
    token: String,
    agent: ureq::Agent,
}

impl std::fmt::Debug for RepoApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepoApi")
            .field("api_base", &self.api_base)
            .field("repo_id", &self.repo_id)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// `url` without its query or fragment.
pub(crate) fn redact(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

/// A git rev as one URL path segment.
pub(crate) fn rev_segment(rev: &str) -> String {
    rev.replace('%', "%25").replace('/', "%2F")
}

fn fail(method: &str, url: &str, e: ureq::Error) -> HfHubError {
    HfHubError::NetworkError(match e {
        ureq::Error::Status(code, resp) => {
            let body: String = resp
                .into_string()
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect();
            format!("{method} {}: HTTP {code}: {body}", redact(url))
        }
        ureq::Error::Transport(t) => format!("{method} {}: {:?}", redact(url), t.kind()),
    })
}

fn json_of(method: &str, url: &str, r: ureq::Response) -> Result<Value> {
    let bad = |e: &dyn std::fmt::Display| {
        HfHubError::NetworkError(format!("{method} {}: {e}", redact(url)))
    };
    let s = r.into_string().map_err(|e| bad(&e))?;
    serde_json::from_str(&s).map_err(|e| bad(&e))
}

fn bad_reply(what: &str) -> HfHubError {
    HfHubError::NetworkError(what.to_string())
}

/// The next page from a `Link: <url>; rel="next"` header.
pub(crate) fn next_link(link: Option<&str>) -> Option<String> {
    link?.split(',').find_map(|part| {
        let (url, rel) = part.split_once(';')?;
        rel.contains("rel=\"next\"").then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

/// Parse one page of `GET /api/models/{repo}/tree/{rev}`.
pub(crate) fn parse_tree(v: &Value) -> Result<Vec<RemoteFile>> {
    let items = v.as_array().ok_or_else(|| bad_reply("tree: not a list"))?;
    Ok(items
        .iter()
        .filter(|i| i["type"] == "file")
        .map(|i| RemoteFile {
            path: i["path"].as_str().unwrap_or_default().to_string(),
            size: i["size"].as_u64().unwrap_or(0),
            git_oid: i["oid"].as_str().unwrap_or_default().to_string(),
            lfs_sha256: i["lfs"]["oid"].as_str().map(str::to_string),
        })
        .collect())
}

/// Parse `GET /api/models/{repo}/refs`.
pub(crate) fn parse_refs(v: &Value) -> Refs {
    let names = |key: &str| -> BTreeMap<String, String> {
        v[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|r| {
                        Some((
                            r["name"].as_str()?.into(),
                            r["targetCommit"].as_str()?.into(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    Refs {
        branches: names("branches"),
        tags: names("tags"),
    }
}

/// The NDJSON body of `POST /api/models/{repo}/commit/{rev}`.
///
/// `feedback_hf_commit_ndjson_load_bearing.md` (PMAT-690 defect 5): the commit
/// endpoint needs `application/x-ndjson` with `lfsFile` lines for LFS files and
/// `file` lines for regular files. A JSON `addOrUpdate` body returns 200 and
/// silently drops the files (paiml/albor-370m-v1: 9 commits, only
/// `.gitattributes` in the tree).
#[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
pub(crate) fn commit_body(summary: &str, ops: &[CommitOp]) -> String {
    let mut lines =
        vec![json!({"key": "header", "value": {"summary": summary, "description": ""}})];
    for op in ops {
        lines.push(match op {
            CommitOp::Lfs { path, sha256, size } => json!({"key": "lfsFile", "value": {
                "path": path, "algo": "sha256", "oid": sha256, "size": size}}),
            CommitOp::File { path, bytes } => json!({"key": "file", "value": {
                "path": path, "encoding": "base64", "content": base64_encode(bytes)}}),
            CommitOp::Delete { path } => {
                json!({"key": "deletedFile", "value": {"path": path}})
            }
        });
    }
    lines.iter().map(|l| format!("{l}\n")).collect()
}

/// The LFS batch request for one object. Only the `basic` transfer is offered.
#[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
fn lfs_batch_body(sha256: &str, size: u64) -> Value {
    json!({"operation": "upload", "transfers": ["basic"],
        "objects": [{"oid": sha256, "size": size}]})
}

impl RepoApi {
    /// The repo `repo_id` (`org/name`) on the Hub at `api_base`
    /// (`https://huggingface.co`), acting with `token`.
    #[must_use]
    pub fn new(api_base: &str, repo_id: &str, token: impl Into<String>) -> Self {
        Self {
            api_base: api_base.trim_end_matches('/').to_string(),
            repo_id: repo_id.to_string(),
            token: token.into(),
            agent: ureq::AgentBuilder::new()
                .user_agent(concat!("aprender/", env!("CARGO_PKG_VERSION")))
                .build(),
        }
    }

    fn api(&self, tail: &str) -> String {
        format!("{}/api/models/{}/{tail}", self.api_base, self.repo_id)
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    fn get_json(&self, url: &str) -> Result<(Value, Option<String>)> {
        let r = self
            .agent
            .get(url)
            .set("Authorization", &self.bearer())
            .call()
            .map_err(|e| fail("GET", url, e))?;
        let next = next_link(r.header("Link"));
        Ok((json_of("GET", url, r)?, next))
    }

    fn post_json(&self, url: &str, content_type: &str, body: &Value) -> Result<Value> {
        let r = self
            .agent
            .post(url)
            .set("Authorization", &self.bearer())
            .set("Accept", content_type)
            .set("Content-Type", content_type)
            .send_string(&body.to_string())
            .map_err(|e| fail("POST", url, e))?;
        json_of("POST", url, r)
    }

    /// Branches and tags, each with its target commit.
    ///
    /// # Errors
    /// Network or HTTP failure, or a reply that is not JSON.
    pub fn refs(&self) -> Result<Refs> {
        Ok(parse_refs(&self.get_json(&self.api("refs"))?.0))
    }

    /// Every file at `rev`, following the `Link` pages.
    ///
    /// # Errors
    /// Network or HTTP failure, or a page that is not a JSON list.
    pub fn tree(&self, rev: &str) -> Result<Vec<RemoteFile>> {
        let mut url = Some(self.api(&format!("tree/{}?recursive=true", rev_segment(rev))));
        let mut out = Vec::new();
        while let Some(u) = url {
            let (v, next) = self.get_json(&u)?;
            out.extend(parse_tree(&v)?);
            url = next;
        }
        Ok(out)
    }

    /// The Hub's per-file answer to a preupload of `files` at `rev`: one JSON
    /// object per file, each with `path` and `uploadMode` (`lfs` or `regular`).
    ///
    /// # Errors
    /// Network or HTTP failure, or a reply without a `files` list.
    #[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
    pub fn preupload(&self, rev: &str, files: &[PreuploadFile<'_>]) -> Result<Vec<Value>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({"files": files.iter().map(|f| json!({
            "path": f.path,
            "size": f.size,
            "sample": base64_encode(&f.sample[..f.sample.len().min(512)]),
        })).collect::<Vec<_>>()});
        let url = self.api(&format!("preupload/{}", rev_segment(rev)));
        let v = self.post_json(&url, "application/json", &body)?;
        v["files"]
            .as_array()
            .cloned()
            .ok_or_else(|| bad_reply("preupload: reply has no files list"))
    }

    fn put(&self, url: &str, body: LfsBody<'_>, size: u64, header: &Value) -> Result<()> {
        let mut req = self
            .agent
            .put(url)
            .set("Content-Type", "application/octet-stream")
            .set("Content-Length", &size.to_string())
            .timeout(Duration::from_secs(2 * 3600));
        for (k, v) in header.as_object().into_iter().flatten() {
            if let Some(v) = v.as_str() {
                req = req.set(k, v);
            }
        }
        let sent = match body {
            LfsBody::Bytes(b) => req.send_bytes(b),
            LfsBody::File(p) => req.send(std::fs::File::open(p)?.take(size)),
        };
        sent.map(drop).map_err(|e| fail("PUT", url, e))
    }

    /// Upload one LFS object through the LFS batch API (`basic` transfer: one
    /// presigned PUT). Returns `false` when the Hub already holds the object,
    /// so nothing was sent. The caller commits the pointer.
    ///
    /// # Errors
    /// Network or HTTP failure, an object error from the batch API, a reply
    /// that asks for a transfer other than `basic`, or an unreadable file.
    #[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
    pub fn upload_lfs(&self, sha256: &str, size: u64, body: LfsBody<'_>) -> Result<bool> {
        let url = format!(
            "{}/{}.git/info/lfs/objects/batch",
            self.api_base, self.repo_id
        );
        let v = self.post_json(&url, LFS_JSON, &lfs_batch_body(sha256, size))?;
        let obj = &v["objects"][0];
        if obj.is_null() {
            return Err(bad_reply("lfs batch: reply has no objects"));
        }
        if let Some(e) = obj.get("error") {
            return Err(HfHubError::NetworkError(format!("lfs batch: {e}")));
        }
        let Some(actions) = obj.get("actions") else {
            return Ok(false);
        };
        let upload = &actions["upload"];
        let href = upload["href"]
            .as_str()
            .ok_or_else(|| bad_reply("lfs batch: no upload href"))?;
        if upload["header"].get("chunk_size").is_some() {
            return Err(bad_reply(
                "lfs batch: the Hub answered a multipart transfer, but only basic was offered",
            ));
        }
        self.put(href, body, size, &upload["header"])?;
        // Some LFS servers expect a verify POST; its failure surfaces at commit.
        if let Some(verify) = actions["verify"]["href"].as_str() {
            let _ = self.post_json(verify, LFS_JSON, &json!({"oid": sha256, "size": size}));
        }
        Ok(true)
    }

    /// Create branch `branch` at `from` (a branch, tag or commit).
    ///
    /// # Errors
    /// Network or HTTP failure.
    #[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
    pub fn create_branch(&self, branch: &str, from: &str) -> Result<()> {
        let url = self.api(&format!("branch/{}", rev_segment(branch)));
        self.post_json(&url, "application/json", &json!({"startingPoint": from}))
            .map(drop)
    }

    /// Commit `ops` to branch `rev` in one commit; returns the new commit id.
    ///
    /// # Errors
    /// Network or HTTP failure, or a reply without `commitOid`.
    pub fn commit(&self, rev: &str, summary: &str, ops: &[CommitOp]) -> Result<String> {
        let url = self.api(&format!("commit/{}", rev_segment(rev)));
        let r = self
            .agent
            .post(&url)
            .set("Authorization", &self.bearer())
            .set("Content-Type", "application/x-ndjson")
            .timeout(Duration::from_secs(600))
            .send_string(&commit_body(summary, ops))
            .map_err(|e| fail("POST", &url, e))?;
        json_of("POST", &url, r)?["commitOid"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| bad_reply("commit: reply has no commitOid"))
    }

    /// Create tag `tag` at `rev`. The Hub refuses an existing tag.
    ///
    /// # Errors
    /// Network or HTTP failure.
    #[allow(clippy::disallowed_methods)] // serde_json::json! uses unwrap() internally
    pub fn create_tag(&self, rev: &str, tag: &str) -> Result<()> {
        let url = self.api(&format!("tag/{}", rev_segment(rev)));
        self.post_json(&url, "application/json", &json!({"tag": tag}))
            .map(drop)
    }
}

#[cfg(test)]
#[path = "repo_api_tests.rs"]
mod tests;
