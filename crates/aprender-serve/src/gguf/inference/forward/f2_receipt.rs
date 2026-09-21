//! The F2 hybrid guard's receipt: validate once per (model, executable,
//! device), and let later runs read the answer instead of re-deriving it.
//!
//! # The key is the EXECUTABLE, not its version (#3748)
//!
//! #3604 keyed a receipt by `CARGO_PKG_VERSION`, so every 0.69.0 build shared
//! one: a receipt written by build A let build B skip the CPU reference, and a
//! regressed GPU kernel could serve behind another build's green receipt
//! (aprender-37's Qwen3.5 cells logged `source=receipt` and measured nothing).
//! Version + git sha would still let a dirty tree, or a different feature set,
//! ride a clean build's receipt. So the key is the sha256 of the RUNNING
//! executable (`/proc/self/exe`): a different binary can never skip. The
//! version (and git sha, where the binary carries one) is recorded beside the
//! key for humans, never compared. One file per (model, executable, device),
//! so builds A, B, A never overwrite each other; both hashes are cached by the
//! file's (canonical path, size, mtime, inode), so a warm run hashes nothing.
//!
//! # Why (#3604, operator ruling 2026-09-20)
//!
//! `f2_validate_qwen35` proves the CUDA hybrid forward against a CPU reference
//! before it will serve a token. Measured on lambda 4090 with Qwen3.5-4B-Q4_K_M
//! (#3598 row 1): that guard is **67 % of a 14 s time-to-first-token**, and
//! 90–93 % of the guard is the CPU reference forward. It re-derives the same
//! answer for the same three inputs on every `apr run`.
//!
//! The guard is right to exist and wrong to run per call. So: the first run of
//! a (model sha256, apr version, device) triple validates and writes a receipt;
//! a later run whose triple matches reads the receipt and skips the forward;
//! `--revalidate` forces a fresh run and rewrites it.
//!
//! # The receipt IS the validation — which is why it is strict
//!
//! Every path that is not "a receipt whose three keys all match" validates:
//!
//! - no receipt → validate. Absence is never consent (`done_when` 4).
//! - unreadable or malformed receipt → validate, and say why.
//! - any one of the three keys differs → validate, naming the key
//!   (`done_when` 3 — the three planted-receipt falsifiers live in
//!   `f2_receipt_tests.rs`).
//! - `--revalidate` → validate, even on a perfect match (`done_when` 2).
//!
//! And a receipt is only ever WRITTEN after a validation that actually judged
//! something. The guard has three early-`true` exits that judge nothing —
//! `SKIP_PARITY_GATE=1`, a probe shorter than two tokens, a CPU reference that
//! failed to run — and none of them may launder itself into a receipt, or a
//! one-token prompt would "validate" the triple for every prompt after it.
//!
//! # What this module is not
//!
//! It knows nothing about CUDA and compiles without the `cuda` feature, so its
//! decision table is unit-tested on every build. The GPU-facing wrapper that
//! calls it lives beside `f2_validate_qwen35` in `forward_qwen35.rs`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The three things a validation is a statement about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2ReceiptKey {
    /// sha256 of the model file's bytes, lower-case hex. The whole file: a
    /// prefix or a size+mtime fingerprint would let a planted receipt with the
    /// wrong hash pass, which is the first falsifier. (The hash itself is
    /// cached by file identity — [`cached_sha256`] — but what is compared is
    /// always the content hash.)
    pub model_sha256: String,
    /// sha256 of the executable that ran the guard (`/proc/self/exe`) — the
    /// build identity (#3748). Strictly finer than its version or git sha.
    pub exe_sha256: String,
    /// The device the GPU half ran on, as the driver names it.
    pub device: String,
}

/// What a passed validation leaves behind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2Receipt {
    /// Format version of this file, so a future shape change re-validates
    /// instead of misreading.
    pub schema: u32,
    /// The triple this receipt vouches for.
    #[serde(flatten)]
    pub key: F2ReceiptKey,
    /// For humans only, never compared: the build that wrote the receipt
    /// ([`build_label`] — the apr version, and its git sha where the binary
    /// set one) and the executable's path.
    pub build: String,
    /// The executable's path when it wrote the receipt; for humans only.
    pub exe_path: String,
    /// Unix seconds when the validation passed.
    pub validated_at: u64,
    /// How many probe positions the passing validation actually compared.
    pub positions_judged: usize,
}

/// The current receipt schema. Bump it and every old receipt re-validates.
///
/// 2 (#3748): keyed by the executable's sha256 instead of the apr version, so
/// every version-keyed schema-1 receipt revalidates once.
pub const F2_RECEIPT_SCHEMA: u32 = 2;

/// Why a run is validating instead of reading the receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F2ValidateReason {
    /// No receipt exists for this (model, executable, device).
    NoReceipt,
    /// A file exists but could not be read or parsed.
    Unreadable(String),
    /// The receipt is for a different schema version.
    SchemaMismatch {
        /// What the file says.
        found: u32,
        /// What this build writes.
        expected: u32,
    },
    /// The receipt is for a different model.
    ModelSha256Mismatch {
        /// The hash in the file.
        found: String,
        /// The hash of the model being loaded.
        expected: String,
    },
    /// The receipt was written by a different executable — whatever its
    /// version says (#3748).
    ExeSha256Mismatch {
        /// The executable hash in the file.
        found: String,
        /// This executable's hash.
        expected: String,
    },
    /// The receipt was written for a different device.
    DeviceMismatch {
        /// The device in the file.
        found: String,
        /// The device this run is on.
        expected: String,
    },
    /// The user asked for a fresh run.
    Revalidate,
}

impl std::fmt::Display for F2ValidateReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoReceipt => write!(f, "no receipt for this (model, executable, device)"),
            Self::Unreadable(e) => write!(f, "receipt unreadable ({e})"),
            Self::SchemaMismatch { found, expected } => {
                write!(f, "receipt schema {found}, this build writes {expected}")
            },
            Self::ModelSha256Mismatch { found, expected } => write!(
                f,
                "receipt is for model {}…, this file is {}…",
                &found[..found.len().min(12)],
                &expected[..expected.len().min(12)]
            ),
            Self::ExeSha256Mismatch { found, expected } => write!(
                f,
                "receipt written by executable {}…, this executable is {}…",
                &found[..found.len().min(12)],
                &expected[..expected.len().min(12)]
            ),
            Self::DeviceMismatch { found, expected } => {
                write!(f, "receipt written for {found}, this device is {expected}")
            },
            Self::Revalidate => write!(f, "--revalidate"),
        }
    }
}

/// The decision, and its evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F2Decision {
    /// Read the receipt; skip the forward.
    Skip {
        /// The receipt that matched.
        receipt: F2Receipt,
    },
    /// Run the forward.
    Validate(F2ValidateReason),
}

/// THE DECISION TABLE. Pure: no filesystem, no clock, no env, so the three
/// planted-receipt falsifiers and the absence case are ordinary unit tests.
///
/// `found` is what the reader returned: `Ok(None)` for no file, `Err` for a
/// file that would not read or parse, `Ok(Some)` for a parsed receipt.
#[must_use]
pub fn decide(
    found: Result<Option<F2Receipt>, String>,
    expected: &F2ReceiptKey,
    revalidate: bool,
) -> F2Decision {
    // The user's word comes first: a perfect receipt does not survive
    // --revalidate, and the reason names the flag rather than the receipt.
    if revalidate {
        return F2Decision::Validate(F2ValidateReason::Revalidate);
    }
    let receipt = match found {
        Err(e) => return F2Decision::Validate(F2ValidateReason::Unreadable(e)),
        Ok(None) => return F2Decision::Validate(F2ValidateReason::NoReceipt),
        Ok(Some(r)) => r,
    };
    if receipt.schema != F2_RECEIPT_SCHEMA {
        return F2Decision::Validate(F2ValidateReason::SchemaMismatch {
            found: receipt.schema,
            expected: F2_RECEIPT_SCHEMA,
        });
    }
    // Each key is compared and NAMED on its own. A combined "keys differ" would
    // hide which of the three moved, and the three falsifiers are one per key.
    if receipt.key.model_sha256 != expected.model_sha256 {
        return F2Decision::Validate(F2ValidateReason::ModelSha256Mismatch {
            found: receipt.key.model_sha256,
            expected: expected.model_sha256.clone(),
        });
    }
    if receipt.key.exe_sha256 != expected.exe_sha256 {
        return F2Decision::Validate(F2ValidateReason::ExeSha256Mismatch {
            found: receipt.key.exe_sha256,
            expected: expected.exe_sha256.clone(),
        });
    }
    if receipt.key.device != expected.device {
        return F2Decision::Validate(F2ValidateReason::DeviceMismatch {
            found: receipt.key.device,
            expected: expected.device.clone(),
        });
    }
    F2Decision::Skip { receipt }
}

/// Where receipts live. `APR_F2_RECEIPT_DIR` wins (tests and operators pin it),
/// then `$XDG_CACHE_HOME/apr/f2-receipts`, then `$HOME/.cache/apr/f2-receipts`.
/// `None` when no home can be found — the caller then validates every time and
/// says so; it never errors.
#[must_use]
pub fn receipt_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("APR_F2_RECEIPT_DIR") {
        return Some(PathBuf::from(d));
    }
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        if !x.is_empty() {
            return Some(PathBuf::from(x).join("apr").join("f2-receipts"));
        }
    }
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".cache")
            .join("apr")
            .join("f2-receipts")
    })
}

/// One file per (model, executable, device) — #3748. A per-model file let
/// builds A and B overwrite each other, so A→B→A re-validated every time and
/// the receipt answered for whichever build wrote last. The whole key is also
/// INSIDE the file and compared by [`decide`], so a file under the wrong name
/// still cannot skip.
#[must_use]
pub fn receipt_path(dir: &Path, key: &F2ReceiptKey) -> PathBuf {
    let device: String = key
        .device
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .take(48)
        .collect();
    dir.join(format!(
        "{}-{}-{device}.json",
        key.model_sha256,
        &key.exe_sha256[..key.exe_sha256.len().min(16)]
    ))
}

/// Read a receipt. `Ok(None)` when the file does not exist; `Err` for anything
/// else, because "it was there and I could not read it" must not look like
/// "it was not there" — both validate, but the operator should see which.
pub fn read_receipt(path: &Path) -> Result<Option<F2Receipt>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    serde_json::from_str::<F2Receipt>(&text)
        .map(Some)
        .map_err(|e| format!("{}: not a receipt: {e}", path.display()))
}

/// Write a receipt atomically: a temp file PRIVATE TO THIS WRITER, then a
/// rename. A crash mid-write leaves the old receipt or none, and two `apr run`
/// processes validating the same model at once each rename their own complete
/// file — the last rename wins whole, never half of the other's.
///
/// The first version of this used one shared `<sha>.json.tmp`. The AD-04
/// quorum on #3634 (lane 1) read that against the docstring's "never a
/// truncated one" and was right: with a shared name, writer B truncates the
/// file writer A is about to rename, and A renames B's partial into place. The
/// reader would classify it `Unreadable` and validate — the safe direction —
/// but the atomicity claim was false. The temp name now carries the pid and a
/// per-process counter, so no two writers share one.
pub fn write_receipt(path: &Path, receipt: &F2Receipt) -> Result<(), String> {
    write_json_atomically(path, receipt)
}

/// [`write_receipt`]'s atomic write, for any serializable value.
fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let dir = path
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", path.display()))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stem = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("receipt");
    let tmp = dir.join(format!(
        ".{stem}.{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let body = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    if let Err(e) = std::fs::write(&tmp, body) {
        return Err(format!("{}: {e}", tmp.display()));
    }
    // If the rename fails, do not leave the private temp behind to be mistaken
    // for anything: it carries no meaning once it is not the receipt.
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// sha256 of the model bytes, lower-case hex.
///
/// The WHOLE file, deliberately. On a 2.5 GB Q4_K_M this is measurable (the
/// receipt for this ticket reports it), but it is the only identity under
/// which the first falsifier holds: a planted receipt with the wrong hash must
/// re-validate, and a cheaper identity that the planted receipt could still
/// satisfy would pass it.
#[must_use]
pub fn model_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// The version of this crate — the one that ran the guard.
#[must_use]
pub fn apr_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

static BUILD_LABEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Record the human-readable build label the binary knows and this library
/// does not — `apr` sets `"<version> (<git sha>)"` at startup. First call wins.
pub fn set_build_label(label: String) {
    let _ = BUILD_LABEL.set(label);
}

/// The build label recorded in a receipt for humans; `apr <version>` when the
/// binary set none. Never part of the key.
#[must_use]
pub fn build_label() -> String {
    BUILD_LABEL
        .get()
        .cloned()
        .unwrap_or_else(|| format!("apr {}", apr_version()))
}

/// What identifies a file's CONTENT without reading it: if all four are
/// unchanged, the bytes are taken to be unchanged. `mtime` is in nanoseconds;
/// the inode catches a replace-by-rename that kept size and mtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileIdentity {
    /// The canonical path (symlinks resolved).
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// Modification time, nanoseconds since the epoch.
    pub mtime_ns: i128,
    /// Inode number.
    pub inode: u64,
}

/// The identity of the file at `path`, or why it has none. Unix only: without
/// an inode there is no identity, and callers hash instead of caching.
pub fn file_identity(path: &Path) -> Result<FileIdentity, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let canonical =
            std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let md =
            std::fs::metadata(&canonical).map_err(|e| format!("{}: {e}", canonical.display()))?;
        Ok(FileIdentity {
            path: canonical.to_string_lossy().into_owned(),
            size: md.size(),
            mtime_ns: i128::from(md.mtime()) * 1_000_000_000 + i128::from(md.mtime_nsec()),
            inode: md.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        Err(format!(
            "{}: no file identity on this platform",
            path.display()
        ))
    }
}

/// One entry of the content-hash cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HashCacheEntry {
    identity: FileIdentity,
    sha256: String,
}

/// Where a hash came from: the cache, or a fresh read of the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashSource {
    /// The file's identity matched a cached entry; nothing was read.
    Cache,
    /// The file was hashed on this run.
    Hashed,
}

impl HashSource {
    /// How it reads in a log line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cached",
            Self::Hashed => "hashed",
        }
    }
}

/// The sha256 of the file at `path`, from `cache_dir` when the file's
/// (canonical path, size, mtime, inode) all match the cached entry, else from
/// `compute` — which is then cached, but only if the identity is the same
/// after hashing as before (a file rewritten mid-hash is never cached).
///
/// Every failure of the cache (no dir, no identity, unreadable or unwritable
/// entry) falls back to `compute`: the cache may only ever save work, never
/// supply an answer it cannot vouch for.
///
/// # Errors
/// Only `compute`'s own error.
pub fn cached_sha256(
    cache_dir: Option<&Path>,
    path: &Path,
    compute: impl FnOnce() -> Result<String, String>,
) -> Result<(String, HashSource), String> {
    let before = file_identity(path).ok();
    let entry_path = match (cache_dir, &before) {
        (Some(dir), Some(id)) => Some(
            dir.join("sha256-cache")
                .join(format!("{}.json", &model_sha256(id.path.as_bytes())[..32])),
        ),
        _ => None,
    };
    if let (Some(ep), Some(id)) = (&entry_path, &before) {
        let cached = std::fs::read_to_string(ep)
            .ok()
            .and_then(|t| serde_json::from_str::<HashCacheEntry>(&t).ok());
        if let Some(entry) = cached {
            if &entry.identity == id {
                return Ok((entry.sha256, HashSource::Cache));
            }
        }
    }
    let sha256 = compute()?;
    if let (Some(ep), Some(id)) = (entry_path, before) {
        if file_identity(path).ok().as_ref() == Some(&id) {
            let _ = write_json_atomically(
                &ep,
                &HashCacheEntry {
                    identity: id,
                    sha256: sha256.clone(),
                },
            );
        }
    }
    Ok((sha256, HashSource::Hashed))
}

/// sha256 of the file at `path`, streamed.
///
/// # Errors
/// The file cannot be read.
pub fn file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    Ok(s)
}

/// The running executable's sha256 and path (`/proc/self/exe`), cached by
/// file identity under `cache_dir`.
///
/// # Errors
/// The executable cannot be located or read — the caller then cannot key a
/// receipt, validates, and writes none.
pub fn exe_sha256(cache_dir: Option<&Path>) -> Result<(String, HashSource, PathBuf), String> {
    let exe = std::env::current_exe().map_err(|e| format!("the running executable: {e}"))?;
    let (sha, source) = cached_sha256(cache_dir, &exe, || file_sha256(&exe))?;
    Ok((sha, source, exe))
}

/// Now, in unix seconds; 0 if the clock is before the epoch.
#[must_use]
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Did the user ask for a fresh run? `apr run --revalidate` sets this before
/// inference starts — the same env seam the guard already uses for
/// `SKIP_PARITY_GATE`, chosen over threading a bool through six signatures that
/// another open PR (#3606) is changing at the same time.
#[must_use]
pub fn revalidate_requested() -> bool {
    std::env::var("APR_F2_REVALIDATE").is_ok_and(|v| v == "1")
}

#[cfg(test)]
#[path = "f2_receipt_tests.rs"]
mod f2_receipt_tests;
