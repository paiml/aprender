//! #4087: a cache for the `tensor_contract` qa gate, keyed so it can never mask a checker change.
//!
//! The gate reads every tensor of the model (15–40 s measured on yoga, 19–46% of `apr qa`), and a ladder runs
//! `apr qa` on the same file once per backend with the same binary, so all but the first run repeat work.
//!
//! THE KEY is (sha256 of the model file, sha256 of the running `apr` executable). The executable stands in
//! for "the checker's source": every line the gate executes is in it, so ANY change to the checker, its
//! helpers, the rules it embeds, or the compiler that built it is a different key. A narrower hand-kept list of
//! "checker source files" would go stale the first time the checker called a new helper, and a stale key is a
//! stale green. The checker reads no file other than the model (verified: `rosetta::validate*` reads only its
//! path), except for sharded models, which are BYPASSED (run, never cached): hashing one path would not cover
//! the other shards. A key that cannot be computed is a miss, never a hit.
//!
//! AN ENTRY carries its schema, its key and the result, plus `integrity` = sha256 of those three. On read, a parse
//! failure, a schema or key mismatch, an integrity mismatch or a result for another gate is REFUSED. The gate
//! then runs and the entry is rewritten; a refused entry is never trusted.
//!
//! Default on; `APR_QA_NO_CONTRACT_CACHE=1` turns it off; `APR_QA_CACHE_DIR` moves it (tests always set this,
//! so no test ever reads a real `~/.cache`).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::qa::{GateCache, GateResult};

pub(crate) const SCHEMA: &str = "apr-qa-tensor-contract-cache/v1";
const GATE: &str = "tensor_contract";

/// What the key is made of. Both parts are required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CacheKey {
    pub(crate) model_sha256: String,
    pub(crate) checker_sha256: String,
}

impl CacheKey {
    /// The file name an entry is stored under: a digest of BOTH parts. Dropping either part from this digest is
    /// the #4087 mutant; `a_checker_change_is_a_miss` goes RED on it.
    pub(crate) fn file_name(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.model_sha256.as_bytes());
        h.update(b"\0");
        h.update(self.checker_sha256.as_bytes());
        format!("{}.json", hex(&h.finalize()))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    schema: String,
    key: CacheKey,
    result: GateResult,
    integrity: String,
}

/// The outcome of a lookup.
#[derive(Debug)]
pub(crate) enum Lookup {
    Hit(GateResult),
    Miss,
    /// An entry exists and was not trusted; the string says why.
    Refused(String),
}

/// Why the cache did not apply to this model at all (the gate runs, nothing is stored).
pub(crate) fn bypass_reason(model: &Path) -> Option<String> {
    let name = model.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with(".index.json") {
        return Some(
            "sharded safetensors index: other shards are not covered by one file's hash".into(),
        );
    }
    if !model.is_file() {
        // A missing single-file path may resolve to a sharded index (GH-346) — also multi-file.
        return Some("not a single regular file".into());
    }
    let stem = name.trim_end_matches(".gguf");
    if name.ends_with(".gguf") && is_split_gguf(stem) {
        return Some(
            "split GGUF (-NNNNN-of-NNNNN): other parts are not covered by one file's hash".into(),
        );
    }
    None
}

fn is_split_gguf(stem: &str) -> bool {
    // "<name>-00001-of-00003"
    let parts: Vec<&str> = stem.rsplitn(3, '-').collect();
    parts.len() == 3
        && parts[1] == "of"
        && [parts[0], parts[2].rsplit('-').next().unwrap_or("")]
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// sha256 of a file, streamed.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// sha256 of the running `apr` executable, computed once per process. `None` when it cannot be read: then there
/// is no key, and the gate runs uncached.
fn checker_sha256() -> Option<String> {
    static CHECKER: OnceLock<Option<String>> = OnceLock::new();
    CHECKER
        .get_or_init(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| sha256_file(&p).ok())
        })
        .clone()
}

/// The key for `model`, or `None` when either part cannot be computed.
pub(crate) fn key_for(model: &Path) -> Option<CacheKey> {
    Some(CacheKey {
        model_sha256: sha256_file(model).ok()?,
        checker_sha256: checker_sha256()?,
    })
}

/// The cache directory, or `None` when the cache is turned off or no location can be found.
pub(crate) fn cache_dir() -> Option<PathBuf> {
    if std::env::var_os("APR_QA_NO_CONTRACT_CACHE").is_some_and(|v| v == "1") {
        return None;
    }
    if let Some(d) = std::env::var_os("APR_QA_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("apr").join("qa-cache"))
}

fn integrity(schema: &str, key: &CacheKey, result: &GateResult) -> String {
    let body = serde_json::to_string(&(schema, key, result)).unwrap_or_default();
    hex(&Sha256::digest(body.as_bytes()))
}

/// Look `key` up in `dir`.
pub(crate) fn lookup(dir: &Path, key: &CacheKey) -> Lookup {
    let path = dir.join(key.file_name());
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Lookup::Miss,
        Err(e) => return Lookup::Refused(format!("unreadable entry {}: {e}", path.display())),
    };
    let entry: Entry = match serde_json::from_str(&text) {
        Ok(e) => e,
        Err(e) => return Lookup::Refused(format!("entry does not parse: {e}")),
    };
    if entry.schema != SCHEMA {
        return Lookup::Refused(format!("schema {:?}, want {SCHEMA:?}", entry.schema));
    }
    if &entry.key != key {
        return Lookup::Refused("the entry's key is not the key it is stored under".into());
    }
    if entry.result.name != GATE {
        return Lookup::Refused(format!(
            "entry holds gate {:?}, not {GATE}",
            entry.result.name
        ));
    }
    if integrity(&entry.schema, &entry.key, &entry.result) != entry.integrity {
        return Lookup::Refused("integrity hash does not match the entry's content".into());
    }
    Lookup::Hit(entry.result)
}

/// Store a COMPLETED gate result under `key` (atomically: a temp file, then a rename).
pub(crate) fn store(dir: &Path, key: &CacheKey, result: &GateResult) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut result = result.clone();
    result.cache = None;
    let entry = Entry {
        schema: SCHEMA.into(),
        integrity: integrity(SCHEMA, key, &result),
        key: key.clone(),
        result,
    };
    let path = dir.join(key.file_name());
    let tmp = dir.join(format!(".{}.{}.tmp", key.file_name(), std::process::id()));
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(&entry).map_err(std::io::Error::other)?,
    )?;
    std::fs::rename(&tmp, &path)
}

/// Run `gate` through the cache. `completed` from the closure says whether the result is a real verdict
/// (stored) or an error such as an unreadable file (never stored: it may be transient).
pub(crate) fn cached_gate(
    model: &Path,
    dir: Option<PathBuf>,
    gate: impl FnOnce() -> (GateResult, bool),
) -> GateResult {
    let Some(dir) = dir else {
        return gate().0;
    };
    if let Some(why) = bypass_reason(model) {
        let (mut r, _) = gate();
        r.cache = Some(GateCache::bypassed(&why));
        return r;
    }
    let Some(key) = key_for(model) else {
        let (mut r, _) = gate();
        r.cache = Some(GateCache::bypassed("the cache key could not be computed"));
        return r;
    };
    let refused = match lookup(&dir, &key) {
        Lookup::Hit(mut r) => {
            r.cache = Some(GateCache::hit(&key));
            return r;
        }
        Lookup::Miss => None,
        Lookup::Refused(why) => Some(why),
    };
    let (mut r, completed) = gate();
    if completed {
        let _ = store(&dir, &key, &r);
    }
    r.cache = Some(match refused {
        Some(why) => GateCache::refused(&key, &why),
        None => GateCache::miss(&key),
    });
    r
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[path = "qa_contract_cache_tests.rs"]
mod tests;
