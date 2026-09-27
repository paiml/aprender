//! The HF Hub over HTTP, for `apr model publish` (EXT-14). ureq only; no Python.
//!
//! Errors name the method and the URL path with its query stripped, so a presigned
//! upload URL's signature is never printed, and never the token.

use super::hf_publish::{Hub, HubResult, Op, Refs, RemoteFile, Token, Upload};
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const LFS_JSON: &str = "application/vnd.git-lfs+json";

/// The HF Hub at `endpoint`, acting on the model repo `repo`.
pub(crate) struct HfHttp {
    endpoint: String,
    repo: String,
    token: Token,
    agent: ureq::Agent,
}

/// `url` without its query or fragment.
pub(crate) fn redact(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

/// A git rev as one URL path segment.
pub(crate) fn rev_segment(rev: &str) -> String {
    rev.replace('%', "%25").replace('/', "%2F")
}

fn fail(method: &str, url: &str, e: ureq::Error) -> String {
    match e {
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
    }
}

fn json_of(method: &str, url: &str, r: ureq::Response) -> HubResult<Value> {
    let s = r
        .into_string()
        .map_err(|e| format!("{method} {}: {e}", redact(url)))?;
    serde_json::from_str(&s).map_err(|e| format!("{method} {}: {e}", redact(url)))
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
pub(crate) fn parse_tree(v: &Value) -> HubResult<Vec<RemoteFile>> {
    let items = v.as_array().ok_or("tree: not a list")?;
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
pub(crate) fn commit_body(summary: &str, ops: &[Op]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut lines =
        vec![json!({"key": "header", "value": {"summary": summary, "description": ""}})];
    for op in ops {
        lines.push(match op {
            Op::Lfs { path, sha256, size } => json!({"key": "lfsFile", "value": {
                "path": path, "algo": "sha256", "oid": sha256, "size": size}}),
            Op::File { path, bytes } => json!({"key": "file", "value": {
                "path": path, "encoding": "base64", "content": b64.encode(bytes)}}),
            Op::Delete { path } => json!({"key": "deletedFile", "value": {"path": path}}),
        });
    }
    lines.iter().map(|l| format!("{l}\n")).collect()
}

impl HfHttp {
    pub(crate) fn new(endpoint: &str, repo: &str, token: Token) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            repo: repo.to_string(),
            token,
            agent: ureq::AgentBuilder::new()
                .user_agent(concat!("apr/", env!("CARGO_PKG_VERSION")))
                .build(),
        }
    }

    fn api(&self, tail: &str) -> String {
        format!("{}/api/models/{}/{tail}", self.endpoint, self.repo)
    }

    fn post_json(&self, url: &str, body: &Value) -> HubResult<Value> {
        let r = self
            .agent
            .post(url)
            .set("Authorization", &self.token.bearer())
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| fail("POST", url, e))?;
        json_of("POST", url, r)
    }

    fn put_part(&self, url: &str, file: &Path, offset: u64, len: u64) -> HubResult<ureq::Response> {
        let mut f = std::fs::File::open(file).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        self.agent
            .put(url)
            .set("Content-Length", &len.to_string())
            .send(f.take(len))
            .map_err(|e| fail("PUT", url, e))
    }
}

impl Hub for HfHttp {
    fn refs(&self) -> HubResult<Refs> {
        let url = self.api("refs");
        let r = self
            .agent
            .get(&url)
            .set("Authorization", &self.token.bearer())
            .call()
            .map_err(|e| fail("GET", &url, e))?;
        Ok(parse_refs(&json_of("GET", &url, r)?))
    }

    fn tree(&self, rev: &str) -> HubResult<Vec<RemoteFile>> {
        let mut url = Some(self.api(&format!("tree/{}?recursive=true", rev_segment(rev))));
        let mut out = Vec::new();
        while let Some(u) = url {
            let r = self
                .agent
                .get(&u)
                .set("Authorization", &self.token.bearer())
                .call()
                .map_err(|e| fail("GET", &u, e))?;
            url = next_link(r.header("Link"));
            out.extend(parse_tree(&json_of("GET", &u, r)?)?);
        }
        Ok(out)
    }

    fn preupload(&self, rev: &str, files: &[Upload<'_>]) -> HubResult<Vec<bool>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let b64 = base64::engine::general_purpose::STANDARD;
        let body = json!({"files": files.iter().map(|f| json!({
            "path": f.path, "size": f.size, "sample": b64.encode(&f.sample)})).collect::<Vec<_>>()});
        let url = self.api(&format!("preupload/{}", rev_segment(rev)));
        let v = self.post_json(&url, &body)?;
        let modes: BTreeMap<&str, bool> = v["files"]
            .as_array()
            .ok_or("preupload: no files")?
            .iter()
            .filter_map(|f| Some((f["path"].as_str()?, f["uploadMode"] == "lfs")))
            .collect();
        files
            .iter()
            .map(|f| {
                modes
                    .get(f.path)
                    .copied()
                    .ok_or_else(|| format!("preupload: no mode for {}", f.path))
            })
            .collect()
    }

    fn upload_lfs(&self, sha256: &str, size: u64, file: &Path) -> HubResult<bool> {
        let url = format!("{}/{}.git/info/lfs/objects/batch", self.endpoint, self.repo);
        let body = json!({"operation": "upload", "transfers": ["basic", "multipart"],
            "hash_algo": "sha256", "objects": [{"oid": sha256, "size": size}]});
        let r = self
            .agent
            .post(&url)
            .set("Authorization", &self.token.bearer())
            .set("Accept", LFS_JSON)
            .set("Content-Type", LFS_JSON)
            .send_string(&body.to_string())
            .map_err(|e| fail("POST", &url, e))?;
        let v = json_of("POST", &url, r)?;
        let obj = &v["objects"][0];
        if let Some(e) = obj.get("error") {
            return Err(format!("lfs batch: {e}"));
        }
        let Some(actions) = obj.get("actions") else {
            return Ok(false);
        };
        let upload = &actions["upload"];
        let href = upload["href"].as_str().ok_or("lfs batch: no upload href")?;
        let header = &upload["header"];
        if let Some(chunk) = header["chunk_size"]
            .as_str()
            .and_then(|c| c.parse::<u64>().ok())
            .or_else(|| header["chunk_size"].as_u64())
        {
            let mut parts: Vec<(u64, &str)> = header
                .as_object()
                .ok_or("lfs batch: bad header")?
                .iter()
                .filter_map(|(k, v)| Some((k.parse::<u64>().ok()?, v.as_str()?)))
                .collect();
            parts.sort_unstable();
            let mut done = Vec::new();
            for (n, part_url) in parts {
                let offset = (n - 1) * chunk;
                let len = chunk.min(size.saturating_sub(offset));
                let r = self.put_part(part_url, file, offset, len)?;
                let etag = r.header("ETag").ok_or("lfs part: no ETag")?.to_string();
                done.push(json!({"partNumber": n, "etag": etag}));
            }
            let r = self
                .agent
                .post(href)
                .set("Accept", LFS_JSON)
                .set("Content-Type", LFS_JSON)
                .send_string(&json!({"oid": sha256, "parts": done}).to_string())
                .map_err(|e| fail("POST", href, e))?;
            drop(r);
        } else {
            self.put_part(href, file, 0, size)?;
        }
        if let Some(verify) = actions["verify"]["href"].as_str() {
            let r = self
                .agent
                .post(verify)
                .set("Authorization", &self.token.bearer())
                .set("Accept", LFS_JSON)
                .set("Content-Type", LFS_JSON)
                .send_string(&json!({"oid": sha256, "size": size}).to_string())
                .map_err(|e| fail("POST", verify, e))?;
            drop(r);
        }
        Ok(true)
    }

    fn create_branch(&self, branch: &str, from: &str) -> HubResult<()> {
        let url = self.api(&format!("branch/{}", rev_segment(branch)));
        self.post_json(&url, &json!({"startingPoint": from}))
            .map(drop)
    }

    fn commit(&self, rev: &str, summary: &str, ops: &[Op]) -> HubResult<String> {
        let url = self.api(&format!("commit/{}", rev_segment(rev)));
        let r = self
            .agent
            .post(&url)
            .set("Authorization", &self.token.bearer())
            .set("Content-Type", "application/x-ndjson")
            .send_string(&commit_body(summary, ops))
            .map_err(|e| fail("POST", &url, e))?;
        let v = json_of("POST", &url, r)?;
        v["commitOid"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "commit: no commitOid".to_string())
    }

    fn create_tag(&self, rev: &str, tag: &str) -> HubResult<()> {
        let url = self.api(&format!("tag/{}", rev_segment(rev)));
        self.post_json(&url, &json!({"tag": tag})).map(drop)
    }
}
