//! `apr cp SRC DST` copy-by-manifest classifier (CRUX-A-11).
//!
//! Contract: `contracts/crux-A-11-v1.yaml`.
//!
//! Pure classifier — models what `apr cp` does at the manifest layer
//! without touching the filesystem. Given a source manifest (list of
//! blob shas) and a destination tag, returns the destination manifest
//! and a plan of what filesystem ops are expected (0 blob bytes
//! copied; one new manifest file; N hard-link-or-noop operations).
//!
//! Formula (from contract):
//!   `manifest(DST).blobs == manifest(SRC).blobs  (identical sha256 list)`
//!   `stat(blob_path).st_ino == stat(blob_path_after_cp).st_ino`
//!   `disk_usage_delta ≈ sizeof(manifest_json)`
//!
//! The integration-level claims
//!   * `du -b` delta ≤ 4 KiB,
//!   * `stat -c %i` equality across SRC/DST blob paths,
//! are discharged by a separate filesystem-gated harness. This module
//! proves the algorithm-level precondition: the destination manifest
//! references the same blob-sha set, and the planned op stream contains
//! zero "copy bytes" ops.

/// Reason the classifier rejects a copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyError {
    /// Source tag not found in the local registry.
    SourceNotFound(String),
    /// Destination tag already exists — `apr cp` refuses to overwrite.
    DestinationExists(String),
    /// Tag string is syntactically invalid (empty or contains a NUL /
    /// path separator that would escape the registry directory).
    InvalidTag(String),
    /// Source manifest has zero blobs — nothing to copy.
    EmptyManifest(String),
}

impl std::fmt::Display for CopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyError::SourceNotFound(t) => write!(f, "source tag not found: {t:?}"),
            CopyError::DestinationExists(t) => {
                write!(f, "destination tag already exists: {t:?}")
            }
            CopyError::InvalidTag(t) => write!(f, "invalid tag: {t:?}"),
            CopyError::EmptyManifest(t) => {
                write!(f, "source manifest has no blobs: {t:?}")
            }
        }
    }
}

impl std::error::Error for CopyError {}

/// Minimal view of a manifest the classifier needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestView {
    pub tag: String,
    /// Sha256 digests (lowercase hex) of each blob referenced by the
    /// manifest, in manifest order.
    pub blob_shas: Vec<String>,
}

/// One step in the planned op stream for `apr cp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyOp {
    /// Attempt to hard-link the blob identified by `sha` from the
    /// registry's blob dir to itself at the same path. Real hard-link
    /// in the filesystem harness; in the classifier this is a no-op
    /// that records the sha so the caller can assert "no byte-copy".
    HardLink { sha: String },
    /// Write a new manifest JSON file for the destination tag. The
    /// `bytes` field is the serialized manifest length in the harness;
    /// the classifier only records that exactly one such op exists.
    WriteManifest { tag: String },
}

/// Plan returned by `plan_copy`. `dst` is the destination manifest;
/// `ops` is the ordered op stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyPlan {
    pub dst: ManifestView,
    pub ops: Vec<CopyOp>,
}

/// Tags are `name:tag` or `name` forms. Reject empty strings, strings
/// containing path separators, and NUL. This is the algorithm-level
/// analogue of the filesystem-level "tag path must stay inside
/// ~/.apr/models" invariant.
fn tag_is_valid(tag: &str) -> bool {
    !tag.is_empty()
        && !tag.contains('/')
        && !tag.contains('\\')
        && !tag.contains('\0')
}

/// Look up a manifest by tag in a registry slice.
fn find_manifest<'a>(
    registry: &'a [ManifestView],
    tag: &str,
) -> Option<&'a ManifestView> {
    registry.iter().find(|m| m.tag == tag)
}

/// Build the destination manifest + op stream for `apr cp SRC DST`.
///
/// Algorithm-level precondition for FALSIFY-CRUX-A-11-001/002:
/// * the destination manifest has the EXACT same blob-sha list as SRC,
///   so no new bytes need to land on disk (disk-usage delta is the
///   manifest file only);
/// * every blob op is a `HardLink`, not a byte-copy;
/// * exactly one `WriteManifest { tag: dst }` op is emitted.
pub fn plan_copy(
    registry: &[ManifestView],
    src: &str,
    dst: &str,
) -> Result<CopyPlan, CopyError> {
    if !tag_is_valid(src) {
        return Err(CopyError::InvalidTag(src.to_string()));
    }
    if !tag_is_valid(dst) {
        return Err(CopyError::InvalidTag(dst.to_string()));
    }

    let src_manifest = find_manifest(registry, src)
        .ok_or_else(|| CopyError::SourceNotFound(src.to_string()))?;

    if find_manifest(registry, dst).is_some() {
        return Err(CopyError::DestinationExists(dst.to_string()));
    }

    if src_manifest.blob_shas.is_empty() {
        return Err(CopyError::EmptyManifest(src.to_string()));
    }

    let dst_manifest = ManifestView {
        tag: dst.to_string(),
        blob_shas: src_manifest.blob_shas.clone(),
    };

    let mut ops: Vec<CopyOp> = src_manifest
        .blob_shas
        .iter()
        .map(|sha| CopyOp::HardLink { sha: sha.clone() })
        .collect();
    ops.push(CopyOp::WriteManifest {
        tag: dst.to_string(),
    });

    Ok(CopyPlan {
        dst: dst_manifest,
        ops,
    })
}

/// Return true iff `plan` contains zero byte-copy operations. Used by
/// the FALSIFY-001 algorithm-level proof that `apr cp` never allocates
/// new blob bytes.
pub fn plan_has_no_byte_copy(plan: &CopyPlan) -> bool {
    plan.ops
        .iter()
        .all(|op| matches!(op, CopyOp::HardLink { .. } | CopyOp::WriteManifest { .. }))
}

/// Return the number of blob ops in the plan. All of them must be
/// `HardLink`; asserted by `plan_has_no_byte_copy`.
pub fn plan_blob_op_count(plan: &CopyPlan) -> usize {
    plan.ops
        .iter()
        .filter(|op| matches!(op, CopyOp::HardLink { .. }))
        .count()
}

/// Count the number of manifest-write ops. Must be exactly 1 for a
/// well-formed copy.
pub fn plan_manifest_write_count(plan: &CopyPlan) -> usize {
    plan.ops
        .iter()
        .filter(|op| matches!(op, CopyOp::WriteManifest { .. }))
        .count()
}

/// Return true iff src and dst manifests reference the SAME blob shas
/// in the SAME order — the algorithm-level precondition for the
/// hard-link-inode-equality FALSIFY-002 check.
pub fn dst_blob_shas_match_src(
    src: &ManifestView,
    dst: &ManifestView,
) -> bool {
    src.blob_shas == dst.blob_shas
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> Vec<ManifestView> {
        vec![
            ManifestView {
                tag: "qwen2.5-0.5b:latest".to_string(),
                blob_shas: vec![
                    "a".repeat(64),
                    "b".repeat(64),
                    "c".repeat(64),
                ],
            },
            ManifestView {
                tag: "llama3:latest".to_string(),
                blob_shas: vec!["d".repeat(64)],
            },
        ]
    }

    #[test]
    fn copy_produces_identical_blob_sha_list() {
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:mycopy")
            .unwrap();
        assert_eq!(plan.dst.tag, "qwen2.5-0.5b:mycopy");
        assert_eq!(plan.dst.blob_shas, reg[0].blob_shas);
        assert!(dst_blob_shas_match_src(&reg[0], &plan.dst));
    }

    #[test]
    fn falsify_001_sub_claim_zero_byte_copy_ops() {
        // CRUX-A-11 ALGO-001 sub-claim of FALSIFY-001: the planned op
        // stream contains zero byte-copy operations — every blob op
        // is a HardLink. Algorithm-level analogue of the `du -b`
        // delta ≤ 4 KiB check (if no byte-copy, then delta == sizeof
        // manifest file, which is well under 4 KiB).
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:mycopy")
            .unwrap();
        assert!(plan_has_no_byte_copy(&plan));
        assert_eq!(plan_blob_op_count(&plan), reg[0].blob_shas.len());
        assert_eq!(plan_manifest_write_count(&plan), 1);
    }

    #[test]
    fn falsify_002_sub_claim_blob_shas_equal() {
        // CRUX-A-11 ALGO-002 sub-claim of FALSIFY-002: if SRC and DST
        // manifests reference the same sha256, then (assuming a
        // content-addressed blob store) `stat -c %i` on the resolved
        // blob path is equal — there is only one path per sha.
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:mycopy")
            .unwrap();
        for (src_sha, dst_sha) in
            reg[0].blob_shas.iter().zip(plan.dst.blob_shas.iter())
        {
            assert_eq!(src_sha, dst_sha, "blob-sha divergence breaks hard-link claim");
        }
    }

    #[test]
    fn source_not_found_is_error() {
        let reg = sample_registry();
        let err = plan_copy(&reg, "does-not-exist:latest", "x:y").unwrap_err();
        assert_eq!(
            err,
            CopyError::SourceNotFound("does-not-exist:latest".to_string())
        );
    }

    #[test]
    fn destination_exists_is_error() {
        let reg = sample_registry();
        let err =
            plan_copy(&reg, "qwen2.5-0.5b:latest", "llama3:latest").unwrap_err();
        assert_eq!(
            err,
            CopyError::DestinationExists("llama3:latest".to_string())
        );
    }

    #[test]
    fn empty_source_tag_is_invalid() {
        let reg = sample_registry();
        let err = plan_copy(&reg, "", "x:y").unwrap_err();
        assert!(matches!(err, CopyError::InvalidTag(_)));
    }

    #[test]
    fn empty_destination_tag_is_invalid() {
        let reg = sample_registry();
        let err = plan_copy(&reg, "qwen2.5-0.5b:latest", "").unwrap_err();
        assert!(matches!(err, CopyError::InvalidTag(_)));
    }

    #[test]
    fn tag_with_path_separator_is_invalid() {
        let reg = sample_registry();
        // Forward slash would escape ~/.apr/models via path traversal.
        let err =
            plan_copy(&reg, "qwen2.5-0.5b:latest", "../evil").unwrap_err();
        assert!(matches!(err, CopyError::InvalidTag(_)));
        // Backslash, same reason on Windows hosts.
        let err =
            plan_copy(&reg, "qwen2.5-0.5b:latest", r"a\b").unwrap_err();
        assert!(matches!(err, CopyError::InvalidTag(_)));
    }

    #[test]
    fn tag_with_nul_is_invalid() {
        let reg = sample_registry();
        let err =
            plan_copy(&reg, "qwen2.5-0.5b:latest", "bad\0tag").unwrap_err();
        assert!(matches!(err, CopyError::InvalidTag(_)));
    }

    #[test]
    fn empty_manifest_source_is_error() {
        let reg = vec![ManifestView {
            tag: "empty:latest".to_string(),
            blob_shas: vec![],
        }];
        let err = plan_copy(&reg, "empty:latest", "empty:copy").unwrap_err();
        assert_eq!(err, CopyError::EmptyManifest("empty:latest".to_string()));
    }

    #[test]
    fn single_blob_source_works() {
        let reg = sample_registry();
        let plan = plan_copy(&reg, "llama3:latest", "llama3:pinned").unwrap();
        assert_eq!(plan.dst.blob_shas.len(), 1);
        assert_eq!(plan_blob_op_count(&plan), 1);
        assert_eq!(plan_manifest_write_count(&plan), 1);
    }

    #[test]
    fn plan_is_deterministic() {
        // Same inputs → byte-identical plan across invocations.
        let reg = sample_registry();
        let a = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:a")
            .unwrap();
        let b = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:a")
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ops_preserve_source_manifest_order() {
        // Blob order must match SRC manifest so content-addressed
        // lookup is stable downstream.
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:a")
            .unwrap();
        let mut seen = vec![];
        for op in &plan.ops {
            if let CopyOp::HardLink { sha } = op {
                seen.push(sha.clone());
            }
        }
        assert_eq!(seen, reg[0].blob_shas);
    }

    #[test]
    fn write_manifest_op_is_last() {
        // The manifest file must be written AFTER all hard-links are
        // in place; crash-safety invariant (a partial state with a
        // manifest pointing at a missing blob is forbidden).
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:a")
            .unwrap();
        match plan.ops.last() {
            Some(CopyOp::WriteManifest { tag }) => {
                assert_eq!(tag, "qwen2.5-0.5b:a");
            }
            other => panic!("expected WriteManifest last, got {other:?}"),
        }
    }

    #[test]
    fn no_byte_copy_holds_for_all_plans() {
        // Stronger: FALSIFY-001 sub-claim holds for every manifest in
        // the sample registry.
        let reg = sample_registry();
        for src in &reg {
            let plan =
                plan_copy(&reg, &src.tag, &format!("{}-copy", src.tag)).unwrap();
            assert!(plan_has_no_byte_copy(&plan));
        }
    }

    #[test]
    fn copying_to_self_is_destination_exists() {
        // SRC == DST ⇒ DestinationExists (SRC already occupies that
        // tag). Prevents accidental self-clobber.
        let reg = sample_registry();
        let err =
            plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:latest")
                .unwrap_err();
        assert_eq!(
            err,
            CopyError::DestinationExists("qwen2.5-0.5b:latest".to_string())
        );
    }

    #[test]
    fn blob_count_matches_src_exactly() {
        // FALSIFY-001 invariant: blob count unchanged across registry.
        // The DST manifest references the SAME number of shas as SRC
        // (and, since shas are content-addressed, refers to the same
        // physical blobs).
        let reg = sample_registry();
        let plan = plan_copy(&reg, "qwen2.5-0.5b:latest", "qwen2.5-0.5b:a")
            .unwrap();
        assert_eq!(plan.dst.blob_shas.len(), reg[0].blob_shas.len());
    }
}
