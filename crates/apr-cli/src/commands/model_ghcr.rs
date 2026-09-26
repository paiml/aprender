//! GHCR mirror: `apr model ghcr-push` and `apr model ghcr-fetch` (EXT-001 D-2, row EXT-17,
//! aprender#4399 collapsed into #4393).
//!
//! A release dir is pushed as ONE OCI artifact: an image manifest (OCI 1.1,
//! `artifactType` [`ARTIFACT_TYPE`]) whose config is the empty descriptor and whose
//! layers are `model-release-v1.json`, the files it lists, and the gate receipt when
//! present. Each layer carries its file name in `org.opencontainers.image.title`, so a
//! fetch writes back the same directory M7 (`apr model confirm`) compares.
//!
//! - **Idempotent.** The manifest bytes are a pure function of the release dir (no
//!   timestamp, fixed layer order), so one release has one digest. A blob the registry
//!   already holds is not sent again, and a tag already on that digest is left alone:
//!   re-running a completed push uploads nothing and writes no tag.
//! - **Resumable.** Blobs are content-addressed and the manifest is put last, so an
//!   interrupted push leaves no tag. The re-run sends only the blobs still missing and
//!   ends on the same digest.
//! - **Immutable tags (I-8).** The version tag is never moved. If it already points at
//!   another digest, the push refuses. Only `latest` moves, and only for a `released`
//!   manifest (the GHCR counterpart of moving HF `main` on promote).
//! - **Token (R-7).** The token is read only from `--token-file`, which must be
//!   owner-only (mode `0600`), on the driver host. The push refuses to run if the token
//!   value appears in any environment variable. The token is never written to the
//!   receipt, and the transport strips it from every error it returns.
//! - **T13 `[U]`.** The per-layer limit is recorded, not assumed. The receipt carries
//!   the largest layer actually pushed next to the reported 10 GB limit.

use super::model_confirm::read_manifest;
use super::model_gate::{sha256_file, MANIFEST, RECEIPT};
use crate::error::{CliError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The receipt a push writes under `state/<version>/`.
pub(crate) const GHCR_RECEIPT: &str = "model-ghcr-receipt-v1.json";
/// `artifactType` of the OCI manifest.
pub(crate) const ARTIFACT_TYPE: &str = "application/vnd.paiml.model-release.v1";
/// OCI image manifest media type.
pub(crate) const MANIFEST_MEDIA: &str = "application/vnd.oci.image.manifest.v1+json";
/// The empty config descriptor's media type (OCI 1.1 artifact guidance).
pub(crate) const EMPTY_MEDIA: &str = "application/vnd.oci.empty.v1+json";
/// The empty config blob.
pub(crate) const EMPTY_CONFIG: &[u8] = b"{}";
/// Media type of `model-release-v1.json` as a layer.
pub(crate) const RELEASE_MEDIA: &str = "application/vnd.paiml.model-release.v1+json";
/// Media type of every other release file.
pub(crate) const FILE_MEDIA: &str = "application/vnd.paiml.model-release.file.v1";
/// The annotation carrying a layer's file name.
pub(crate) const TITLE: &str = "org.opencontainers.image.title";
/// T13 `[U]`: the per-layer limit GHCR is reported to enforce (10 GB).
pub(crate) const T13_LAYER_LIMIT_BYTES: u64 = 10_000_000_000;
/// The one tag that moves.
pub(crate) const LATEST: &str = "latest";

fn invalid(msg: impl Into<String>) -> CliError {
    CliError::ValidationFailed(msg.into())
}

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// An OCI content descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Descriptor {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
}

/// An OCI 1.1 image manifest used as an artifact manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OciManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "mediaType")]
    pub media_type: String,
    #[serde(rename = "artifactType")]
    pub artifact_type: String,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
}

/// What a blob upload reads from.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Blob<'a> {
    Bytes(&'a [u8]),
    File(&'a Path),
}

/// The registry operations the publisher needs, over one repository. The HTTP
/// implementation is [`HttpRegistry`]; tests use an in-memory registry.
pub(crate) trait Registry {
    /// Whether the registry holds the blob `digest`.
    fn blob_exists(&mut self, digest: &str) -> Result<bool>;
    /// Upload one blob. The registry must reject bytes that do not hash to `digest`.
    fn put_blob(&mut self, digest: &str, size: u64, blob: Blob<'_>) -> Result<()>;
    /// The digest `tag` points at, if the tag exists.
    fn tag_digest(&mut self, tag: &str) -> Result<Option<String>>;
    /// Put `body` as the manifest under `tag`.
    fn put_manifest(&mut self, tag: &str, body: &[u8]) -> Result<()>;
    /// The manifest bytes under `reference` (a tag or a digest).
    fn get_manifest(&mut self, reference: &str) -> Result<Vec<u8>>;
    /// Write the blob `digest` to `to`.
    fn get_blob(&mut self, digest: &str, to: &Path) -> Result<()>;
}

/// One layer of the artifact, with the file it is read from.
#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub name: String,
    pub path: PathBuf,
    pub descriptor: Descriptor,
}

/// The artifact a release dir becomes: manifest bytes and their layers.
#[derive(Debug, Clone)]
pub(crate) struct Artifact {
    pub line: String,
    pub version: String,
    pub channel: String,
    pub manifest: Vec<u8>,
    pub digest: String,
    pub layers: Vec<Layer>,
}

fn layer(dir: &Path, name: &str, media: &str) -> Result<Layer> {
    let path = dir.join(name);
    let (size, sha) =
        sha256_file(&path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    Ok(Layer {
        name: name.to_string(),
        descriptor: Descriptor {
            media_type: media.to_string(),
            digest: format!("sha256:{sha}"),
            size,
            annotations: BTreeMap::from([(TITLE.to_string(), name.to_string())]),
        },
        path,
    })
}

/// An OCI tag: `[A-Za-z0-9_][A-Za-z0-9._-]{0,127}`.
pub(crate) fn valid_tag(tag: &str) -> bool {
    let mut chars = tag.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphanumeric() || c == '_')
        && tag.len() <= 128
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Build the artifact for the release in `dir`. Every listed file must match its
/// manifest sha and size, so the mirror can only carry what the gates saw.
pub(crate) fn build(dir: &Path) -> Result<Artifact> {
    let m = read_manifest(dir)?;
    if !valid_tag(&m.version) {
        return Err(invalid(format!(
            "version {:?} is not a valid OCI tag",
            m.version
        )));
    }
    let mut layers = vec![layer(dir, MANIFEST, RELEASE_MEDIA)?];
    for f in &m.files {
        if f.name.contains(['/', '\\']) || f.name.starts_with('.') {
            return Err(invalid(format!("{:?} is not a plain file name", f.name)));
        }
        let l = layer(dir, &f.name, FILE_MEDIA)?;
        if l.descriptor.digest != format!("sha256:{}", f.sha256) || l.descriptor.size != f.bytes {
            return Err(invalid(format!(
                "{}: the release dir disagrees with {MANIFEST} (sha or size); re-pack, do not mirror",
                f.name
            )));
        }
        layers.push(l);
    }
    if dir.join(RECEIPT).is_file() {
        layers.push(layer(dir, RECEIPT, FILE_MEDIA)?);
    }
    let manifest = OciManifest {
        schema_version: 2,
        media_type: MANIFEST_MEDIA.into(),
        artifact_type: ARTIFACT_TYPE.into(),
        config: Descriptor {
            media_type: EMPTY_MEDIA.into(),
            digest: format!("sha256:{}", sha256_hex(EMPTY_CONFIG)),
            size: EMPTY_CONFIG.len() as u64,
            annotations: BTreeMap::new(),
        },
        layers: layers.iter().map(|l| l.descriptor.clone()).collect(),
        annotations: BTreeMap::from([
            (
                "org.opencontainers.image.version".to_string(),
                m.version.clone(),
            ),
            ("dev.paiml.model-release.line".to_string(), m.line.clone()),
            (
                "dev.paiml.model-release.channel".to_string(),
                m.channel.clone(),
            ),
        ]),
    };
    let body = serde_json::to_vec(&manifest).map_err(|e| invalid(e.to_string()))?;
    Ok(Artifact {
        line: m.line,
        version: m.version,
        channel: m.channel,
        digest: format!("sha256:{}", sha256_hex(&body)),
        manifest: body,
        layers,
    })
}

/// What a push did to one blob or tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Action {
    Uploaded,
    Present,
    Created,
    Moved,
}

/// One layer in the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LayerRecord {
    pub name: String,
    pub digest: String,
    pub bytes: u64,
    pub action: Action,
}

/// One tag in the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TagRecord {
    pub tag: String,
    pub action: Action,
}

/// `model-ghcr-receipt-v1`. No timestamp and no token, so one push of one release
/// state writes one receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct GhcrReceipt {
    pub schema: &'static str,
    pub line: String,
    pub version: String,
    /// `<registry>/<repo>`, without a tag.
    pub repository: String,
    pub manifest_digest: String,
    pub config: Action,
    pub layers: Vec<LayerRecord>,
    pub tags: Vec<TagRecord>,
    /// True when the push sent nothing: every blob and tag was already in place.
    pub no_op: bool,
    /// T13: the largest layer this push carried, against the reported limit.
    pub max_layer_bytes: u64,
    pub t13_layer_limit_bytes: u64,
}

fn ensure_blob(reg: &mut dyn Registry, d: &Descriptor, blob: Blob<'_>) -> Result<Action> {
    if reg.blob_exists(&d.digest)? {
        return Ok(Action::Present);
    }
    reg.put_blob(&d.digest, d.size, blob)?;
    Ok(Action::Uploaded)
}

/// Push the release in `dir` to `reg` as `<repository>:<version>`, moving `latest` for
/// a `released` manifest.
///
/// # Errors
///
/// `ValidationFailed` when the release dir disagrees with its manifest, or when the
/// version tag already points at another digest (I-8: a tag is never moved). Transport
/// errors pass through unchanged.
pub(crate) fn push(reg: &mut dyn Registry, dir: &Path, repository: &str) -> Result<GhcrReceipt> {
    let a = build(dir)?;
    let existing = reg.tag_digest(&a.version)?;
    if let Some(d) = &existing {
        if *d != a.digest {
            return Err(invalid(format!(
                "STOP (I-8): {repository}:{} already points at {d}, not this release ({}); a published tag is never moved",
                a.version, a.digest
            )));
        }
    }
    let config = OciManifest::empty_config();
    let config_action = ensure_blob(reg, &config, Blob::Bytes(EMPTY_CONFIG))?;
    let mut layers = Vec::with_capacity(a.layers.len());
    for l in &a.layers {
        let action = ensure_blob(reg, &l.descriptor, Blob::File(&l.path))?;
        layers.push(LayerRecord {
            name: l.name.clone(),
            digest: l.descriptor.digest.clone(),
            bytes: l.descriptor.size,
            action,
        });
    }
    let mut tags = Vec::new();
    if existing.is_some() {
        tags.push(TagRecord {
            tag: a.version.clone(),
            action: Action::Present,
        });
    } else {
        reg.put_manifest(&a.version, &a.manifest)?;
        let landed = reg.tag_digest(&a.version)?;
        if landed.as_deref() != Some(a.digest.as_str()) {
            return Err(invalid(format!(
                "{repository}:{} reads back as {landed:?}, not the pushed {}",
                a.version, a.digest
            )));
        }
        tags.push(TagRecord {
            tag: a.version.clone(),
            action: Action::Created,
        });
    }
    if a.channel == "released" {
        let action = match reg.tag_digest(LATEST)? {
            Some(d) if d == a.digest => Action::Present,
            Some(_) => Action::Moved,
            None => Action::Created,
        };
        if action != Action::Present {
            reg.put_manifest(LATEST, &a.manifest)?;
        }
        tags.push(TagRecord {
            tag: LATEST.into(),
            action,
        });
    }
    let no_op = config_action == Action::Present
        && layers.iter().all(|l| l.action == Action::Present)
        && tags.iter().all(|t| t.action == Action::Present);
    Ok(GhcrReceipt {
        schema: "model-ghcr-receipt-v1",
        line: a.line,
        version: a.version,
        repository: repository.into(),
        manifest_digest: a.digest,
        config: config_action,
        max_layer_bytes: layers.iter().map(|l| l.bytes).max().unwrap_or(0),
        t13_layer_limit_bytes: T13_LAYER_LIMIT_BYTES,
        layers,
        tags,
        no_op,
    })
}

impl OciManifest {
    fn empty_config() -> Descriptor {
        Descriptor {
            media_type: EMPTY_MEDIA.into(),
            digest: format!("sha256:{}", sha256_hex(EMPTY_CONFIG)),
            size: EMPTY_CONFIG.len() as u64,
            annotations: BTreeMap::new(),
        }
    }
}

/// Fetch `reference` (a tag or digest) into the empty or absent directory `to`, the
/// layout `apr model confirm --fetched` reads. Every blob is re-hashed on arrival.
///
/// # Errors
///
/// `ValidationFailed` when the manifest is not a model-release artifact, a layer title
/// is not a plain file name, a blob does not hash to its digest, or `to` is not empty.
pub(crate) fn fetch(reg: &mut dyn Registry, reference: &str, to: &Path) -> Result<OciManifest> {
    if to.exists() && std::fs::read_dir(to)?.next().is_some() {
        return Err(invalid(format!("{}: not empty", to.display())));
    }
    let body = reg.get_manifest(reference)?;
    if reference.starts_with("sha256:") && format!("sha256:{}", sha256_hex(&body)) != reference {
        return Err(invalid(format!("manifest does not hash to {reference}")));
    }
    let m: OciManifest =
        serde_json::from_slice(&body).map_err(|e| invalid(format!("manifest: {e}")))?;
    if m.artifact_type != ARTIFACT_TYPE {
        return Err(invalid(format!(
            "{reference} is a {:?}, not a {ARTIFACT_TYPE}",
            m.artifact_type
        )));
    }
    // Stage beside `to` and rename on success, so a failed fetch leaves `to` as it was
    // and a re-run starts clean instead of tripping on half a release.
    let mut staged = to.as_os_str().to_owned();
    staged.push(".partial");
    let staged = PathBuf::from(staged);
    if staged.exists() {
        std::fs::remove_dir_all(&staged)?;
    }
    std::fs::create_dir_all(&staged)?;
    for l in &m.layers {
        let name = l
            .annotations
            .get(TITLE)
            .ok_or_else(|| invalid(format!("layer {} has no {TITLE}", l.digest)))?;
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(invalid(format!(
                "layer title {name:?} is not a plain file name"
            )));
        }
        let path = staged.join(name);
        reg.get_blob(&l.digest, &path)?;
        let (size, sha) = sha256_file(&path)?;
        if format!("sha256:{sha}") != l.digest || size != l.size {
            return Err(invalid(format!(
                "{name}: fetched bytes hash to sha256:{sha} ({size} B), manifest says {} ({} B)",
                l.digest, l.size
            )));
        }
    }
    if to.exists() {
        std::fs::remove_dir(to)?;
    }
    std::fs::rename(&staged, to)?;
    Ok(m)
}

/// A publish token. `Debug` never prints it.
pub(crate) struct Token(String);

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

impl Token {
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    /// `msg` with every occurrence of the token replaced.
    pub(crate) fn redact(&self, msg: &str) -> String {
        if self.0.is_empty() {
            msg.to_string()
        } else {
            msg.replace(&self.0, "<redacted>")
        }
    }
}

/// Read the publish token from `path` (R-7). The file must be owner-only, and the token
/// must appear in no environment variable in `env`.
///
/// # Errors
///
/// `ValidationFailed` naming the rule, never the token.
pub(crate) fn read_token<I>(path: &Path, env: I) -> Result<Token>
where
    I: IntoIterator<Item = (String, String)>,
{
    let meta = std::fs::metadata(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(invalid(format!(
                "R-7: {} is mode {mode:o}; a publish token file must be owner-only (0600)",
                path.display()
            )));
        }
    }
    #[cfg(not(unix))]
    let _ = meta;
    let raw =
        std::fs::read_to_string(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    let token = raw.trim().to_string();
    if token.is_empty() {
        return Err(invalid(format!("{}: empty token file", path.display())));
    }
    for (k, v) in env {
        if v.contains(&token) {
            return Err(invalid(format!(
                "R-7: the publish token is in the environment ({k}); it is read from the token file only — unset {k}"
            )));
        }
    }
    Ok(Token(token))
}

/// `<host>/<repo>[:<tag>]` split into (base URL, repo, tag). `localhost` and
/// `127.0.0.1` registries are plain HTTP (a local `registry:2`); every other host is HTTPS.
pub(crate) fn parse_reference(r: &str) -> Result<(String, String, Option<String>)> {
    let (host, rest) = r
        .split_once('/')
        .ok_or_else(|| invalid(format!("{r:?}: expected <registry>/<owner>/<name>[:<tag>]")))?;
    let (repo, tag) = match rest.rsplit_once(':') {
        Some((repo, tag)) => (repo, Some(tag.to_string())),
        None => (rest, None),
    };
    let repo_ok = !repo.is_empty()
        && repo.split('/').all(|p| {
            !p.is_empty()
                && p.chars().all(|c| {
                    c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-')
                })
        });
    if host.is_empty() || !repo_ok {
        return Err(invalid(format!(
            "{r:?}: not a <registry>/<owner>/<name> reference"
        )));
    }
    if let Some(t) = &tag {
        if !valid_tag(t) {
            return Err(invalid(format!("{t:?} is not a valid OCI tag")));
        }
    }
    let local = host.starts_with("localhost") || host.starts_with("127.0.0.1");
    let scheme = if local { "http" } else { "https" };
    Ok((format!("{scheme}://{host}"), repo.to_string(), tag))
}

/// The OCI distribution API over HTTP (ureq). Authenticates on the first 401 using
/// the registry's `WWW-Authenticate` challenge: a Bearer realm gets a token exchanged
/// for the file token (GHCR), and a Basic challenge gets the file token directly.
pub(crate) struct HttpRegistry {
    base: String,
    repo: String,
    user: String,
    token: Option<Token>,
    auth: Option<String>,
}

impl HttpRegistry {
    pub(crate) fn new(base: String, repo: String, user: String, token: Option<Token>) -> Self {
        Self {
            base,
            repo,
            user,
            token,
            auth: None,
        }
    }

    fn url(&self, tail: &str) -> String {
        format!("{}/v2/{}/{tail}", self.base, self.repo)
    }

    /// A network error with every credential this registry holds stripped from it:
    /// the file token, its Basic encoding, and a token exchanged for it.
    fn fail(&self, what: &str, e: impl std::fmt::Display) -> CliError {
        let mut msg = format!("{what}: {e}");
        if let Some(t) = &self.token {
            msg = t.redact(&msg);
        }
        for header in [self.basic(), self.auth.clone()].into_iter().flatten() {
            if let Some((_, cred)) = header.split_once(' ') {
                if !cred.is_empty() {
                    msg = msg.replace(cred, "<redacted>");
                }
            }
        }
        CliError::NetworkError(msg)
    }

    fn basic(&self) -> Option<String> {
        use base64::Engine as _;
        self.token.as_ref().map(|t| {
            let pair = format!("{}:{}", self.user, t.expose());
            format!(
                "Basic {}",
                base64::engine::general_purpose::STANDARD.encode(pair)
            )
        })
    }

    /// Answer a 401 challenge; `Ok(false)` when there is nothing to answer with. A
    /// Bearer realm is asked for a token even without a file token: that is how a
    /// public package is pulled anonymously.
    fn authenticate(&mut self, challenge: &str) -> Result<bool> {
        let basic = self.basic();
        let lower = challenge.to_ascii_lowercase();
        if lower.starts_with("basic") {
            let Some(basic) = basic else {
                return Ok(false);
            };
            self.auth = Some(basic);
            return Ok(true);
        }
        if !lower.starts_with("bearer") {
            return Ok(false);
        }
        let param = |key: &str| -> Option<String> {
            let at = challenge.find(&format!("{key}=\""))? + key.len() + 2;
            let end = challenge[at..].find('"')? + at;
            Some(challenge[at..end].to_string())
        };
        let realm =
            param("realm").ok_or_else(|| self.fail("auth", "Bearer challenge without realm"))?;
        let scope = if basic.is_some() { "pull,push" } else { "pull" };
        let mut req =
            ureq::get(&realm).query("scope", &format!("repository:{}:{scope}", self.repo));
        if let Some(basic) = &basic {
            req = req.set("Authorization", basic);
        }
        if let Some(service) = param("service") {
            req = req.query("service", &service);
        }
        let v: serde_json::Value = req
            .call()
            .map_err(|e| self.fail("token exchange", e))?
            .into_json()
            .map_err(|e| self.fail("token exchange", e))?;
        let t = v
            .get("token")
            .or_else(|| v.get("access_token"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| self.fail("token exchange", "no token in the response"))?;
        self.auth = Some(format!("Bearer {t}"));
        Ok(true)
    }

    /// Send a request, authenticating once on a 401. A non-2xx status is returned as
    /// `Ok((status, response))` so callers can read 404 as "absent".
    fn send(
        &mut self,
        what: &str,
        make: &dyn Fn() -> ureq::Request,
        body: &dyn Fn(ureq::Request) -> std::result::Result<ureq::Response, ureq::Error>,
    ) -> Result<ureq::Response> {
        let mut answered = false;
        loop {
            let mut req = make();
            if let Some(a) = &self.auth {
                req = req.set("Authorization", a);
            }
            match body(req) {
                Ok(r) => return Ok(r),
                Err(ureq::Error::Status(401, r)) if !answered => {
                    answered = true;
                    let challenge = r.header("WWW-Authenticate").unwrap_or("").to_string();
                    if !self.authenticate(&challenge)? {
                        return Err(self.fail(what, "401 and no usable credential"));
                    }
                }
                Err(ureq::Error::Status(code, r)) => {
                    let text = r.into_string().unwrap_or_default();
                    return Err(self.fail(what, format!("HTTP {code}: {}", text.trim())));
                }
                Err(e) => return Err(self.fail(what, e)),
            }
        }
    }

    /// Like [`Self::send`] but a 404 is `Ok(None)`.
    fn send_opt(
        &mut self,
        what: &str,
        make: &dyn Fn() -> ureq::Request,
    ) -> Result<Option<ureq::Response>> {
        match self.send(what, make, &|r| r.call()) {
            Ok(r) => Ok(Some(r)),
            Err(CliError::NetworkError(m)) if m.contains("HTTP 404") => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Registry for HttpRegistry {
    fn blob_exists(&mut self, digest: &str) -> Result<bool> {
        let url = self.url(&format!("blobs/{digest}"));
        Ok(self.send_opt("blob HEAD", &|| ureq::head(&url))?.is_some())
    }

    fn put_blob(&mut self, digest: &str, size: u64, blob: Blob<'_>) -> Result<()> {
        let start = self.url("blobs/uploads/");
        let r = self.send("blob upload start", &|| ureq::post(&start), &|r| r.call())?;
        let loc = r
            .header("Location")
            .ok_or_else(|| self.fail("blob upload start", "no Location header"))?
            .to_string();
        let loc = if loc.starts_with("http") {
            loc
        } else {
            format!("{}{loc}", self.base)
        };
        let sep = if loc.contains('?') { '&' } else { '?' };
        let put = format!("{loc}{sep}digest={digest}");
        let len = size.to_string();
        self.send(
            "blob upload",
            &|| {
                ureq::put(&put)
                    .set("Content-Type", "application/octet-stream")
                    .set("Content-Length", &len)
            },
            &|r| match blob {
                Blob::Bytes(b) => r.send_bytes(b),
                Blob::File(p) => match std::fs::File::open(p) {
                    Ok(f) => r.send(f),
                    Err(e) => Err(ureq::Error::from(e)),
                },
            },
        )?;
        Ok(())
    }

    fn tag_digest(&mut self, tag: &str) -> Result<Option<String>> {
        let url = self.url(&format!("manifests/{tag}"));
        let Some(r) = self.send_opt("manifest HEAD", &|| {
            ureq::head(&url).set("Accept", MANIFEST_MEDIA)
        })?
        else {
            return Ok(None);
        };
        if let Some(d) = r.header("Docker-Content-Digest") {
            return Ok(Some(d.to_string()));
        }
        let body = self.get_manifest(tag)?;
        Ok(Some(format!("sha256:{}", sha256_hex(&body))))
    }

    fn put_manifest(&mut self, tag: &str, body: &[u8]) -> Result<()> {
        let url = self.url(&format!("manifests/{tag}"));
        self.send(
            "manifest PUT",
            &|| ureq::put(&url).set("Content-Type", MANIFEST_MEDIA),
            &|r| r.send_bytes(body),
        )?;
        Ok(())
    }

    fn get_manifest(&mut self, reference: &str) -> Result<Vec<u8>> {
        let url = self.url(&format!("manifests/{reference}"));
        let r = self.send(
            "manifest GET",
            &|| ureq::get(&url).set("Accept", MANIFEST_MEDIA),
            &|r| r.call(),
        )?;
        let mut body = Vec::new();
        std::io::Read::read_to_end(&mut r.into_reader(), &mut body)
            .map_err(|e| self.fail("manifest GET", e))?;
        Ok(body)
    }

    fn get_blob(&mut self, digest: &str, to: &Path) -> Result<()> {
        let url = self.url(&format!("blobs/{digest}"));
        let r = self.send("blob GET", &|| ureq::get(&url), &|r| r.call())?;
        let mut f = std::fs::File::create(to)?;
        std::io::copy(&mut r.into_reader(), &mut f).map_err(|e| self.fail("blob GET", e))?;
        Ok(())
    }
}

fn version_dir(state: &Path, version: &str) -> Result<PathBuf> {
    if !valid_tag(version) || version.starts_with('.') {
        return Err(invalid(format!(
            "version {version:?} is not a directory name"
        )));
    }
    Ok(state.join(version))
}

/// Write the receipt under `state/<version>/`.
pub(crate) fn write_receipt(state: &Path, r: &GhcrReceipt) -> Result<PathBuf> {
    let dir = version_dir(state, &r.version)?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(GHCR_RECEIPT);
    let body = serde_json::to_string_pretty(r).map_err(|e| invalid(e.to_string()))?;
    std::fs::write(&path, format!("{body}\n"))?;
    Ok(path)
}

/// Arguments of `apr model ghcr-push`.
pub(crate) struct PushArgs<'a> {
    pub dir: &'a Path,
    pub repository: &'a str,
    pub token_file: &'a Path,
    pub user: &'a str,
    pub state: &'a Path,
    pub json: bool,
}

/// `apr model ghcr-push`.
pub(crate) fn run_push(a: &PushArgs<'_>) -> Result<()> {
    let (base, repo, tag) = parse_reference(a.repository)?;
    if tag.is_some() {
        return Err(invalid(
            "give the repository without a tag; the tag is the manifest's version",
        ));
    }
    let token = read_token(a.token_file, std::env::vars())?;
    let mut reg = HttpRegistry::new(base, repo, a.user.to_string(), Some(token));
    let r = push(&mut reg, a.dir, a.repository)?;
    let path = write_receipt(a.state, &r)?;
    if a.json {
        let s = serde_json::to_string_pretty(&r).map_err(|e| invalid(e.to_string()))?;
        println!("{s}");
    } else {
        let sent = r
            .layers
            .iter()
            .filter(|l| l.action == Action::Uploaded)
            .count();
        println!(
            "{}:{} = {} — {} ({sent} of {} layers sent; largest {} B); receipt {}",
            r.repository,
            r.version,
            r.manifest_digest,
            if r.no_op {
                "no-op, already published"
            } else {
                "pushed"
            },
            r.layers.len(),
            r.max_layer_bytes,
            path.display()
        );
    }
    Ok(())
}

/// `apr model ghcr-fetch`. The token file is optional: a public package needs none.
pub(crate) fn run_fetch(
    reference: &str,
    to: &Path,
    token_file: Option<&Path>,
    user: &str,
    json: bool,
) -> Result<()> {
    let (base, repo, tag) = parse_reference(reference)?;
    let tag = tag.ok_or_else(|| invalid("give <registry>/<owner>/<name>:<tag>"))?;
    let token = token_file
        .map(|p| read_token(p, std::env::vars()))
        .transpose()?;
    let mut reg = HttpRegistry::new(base, repo, user.to_string(), token);
    let m = fetch(&mut reg, &tag, to)?;
    if json {
        let s = serde_json::to_string_pretty(&m).map_err(|e| invalid(e.to_string()))?;
        println!("{s}");
    } else {
        println!(
            "fetched {reference}: {} layers into {} (next: apr model confirm --fetched {} --source ghcr:{reference})",
            m.layers.len(),
            to.display(),
            to.display()
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_ghcr_tests.rs"]
mod tests;
