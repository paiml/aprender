//! FALSIFY-DECIDE-APR-001/-008/-009/-011/-013: every ladder rung refuses its own induced
//! negative, naming the rung, and every manifest leaf is bound to its sha-bound source. Negatives are induced by editing the PACKED bytes (or by
//! packing a mutated in-memory [`PackInputs`]) — never by weakening a rung.

use super::tests::pack_tiny;
use super::{
    check_index_extent, check_size, expected_bytes, load_verified, load_verified_within,
    read_bounded_within, read_decide_apr_bytes_bounded, write_decide_apr_within, ArtifactError,
    ArtifactLimits, CUSTOM_METADATA_KEY, MAX_ARTIFACT_BYTES, MAX_METADATA_BYTES, MAX_TENSOR_COUNT,
    MIN_INDEX_ENTRY_BYTES, PROBE_MAX_ROW_TOKENS,
};
use crate::pack::PackInputs;
use crate::test_support::fixture_dir;
use aprender::format::v2::{
    AprV2Header, AprV2Metadata, AprV2ReaderRef, AprV2Writer, TensorDType, HEADER_SIZE_V2,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

type Tensors = Vec<(String, TensorDType, Vec<usize>, Vec<u8>)>;

/// The packed tiny fixture, packed once per test binary.
fn packed() -> Vec<u8> {
    static PACKED: OnceLock<Vec<u8>> = OnceLock::new();
    PACKED.get_or_init(pack_tiny).clone()
}

fn inputs() -> PackInputs {
    let dir = fixture_dir();
    PackInputs::from_run_dir(&dir, &dir.join("data")).expect("the tiny run dir reads")
}

/// The ladder's refusal of `bytes` (panics if the ladder ACCEPTS an induced negative).
fn refuse(bytes: &[u8]) -> ArtifactError {
    load_verified(bytes).expect_err("the ladder accepted an induced negative")
}

/// Re-write `bytes` through the container writer after `edit`.
fn repack(bytes: &[u8], edit: impl FnOnce(&mut AprV2Metadata, &mut Tensors)) -> Vec<u8> {
    let r = AprV2ReaderRef::from_bytes(bytes).expect("open the packed artifact");
    let mut meta = r.metadata().clone();
    let mut tensors: Tensors = r
        .tensor_index()
        .iter()
        .map(|e| {
            let data = r.get_tensor_data(&e.name).expect("tensor data");
            (e.name.clone(), e.dtype, e.shape.clone(), data.to_vec())
        })
        .collect();
    edit(&mut meta, &mut tensors);
    let mut w = AprV2Writer::new(meta);
    for (name, dtype, shape, data) in tensors {
        w.add_tensor(name, dtype, shape, data);
    }
    w.write().expect("repack")
}

/// Edit the manifest JSON and re-write the container.
fn edit_manifest(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    repack(bytes, |meta, _| {
        let text = meta.custom[CUSTOM_METADATA_KEY]
            .as_str()
            .expect("manifest string")
            .to_string();
        let mut v: serde_json::Value = serde_json::from_str(&text).expect("manifest JSON");
        edit(&mut v);
        let text = serde_json::to_string(&v).expect("manifest re-serializes");
        meta.custom.insert(
            CUSTOM_METADATA_KEY.to_string(),
            serde_json::Value::String(text),
        );
    })
}

/// Edit one tensor's bytes in place.
fn edit_tensor(bytes: &[u8], name: &str, edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut edit = Some(edit);
    repack(bytes, |_, tensors| {
        let t = tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("tensor {name}"));
        if let Some(f) = edit.take() {
            f(&mut t.3);
        }
    })
}

/// Rewrite the header with `edit` applied and a VALID CRC (a forged header).
fn forge_header(bytes: &mut [u8], edit: impl FnOnce(&mut AprV2Header)) {
    let mut h = AprV2Header::from_bytes(bytes).expect("header parses");
    edit(&mut h);
    h.update_checksum();
    bytes[..HEADER_SIZE_V2].copy_from_slice(&h.to_bytes());
}

fn header(bytes: &[u8]) -> AprV2Header {
    AprV2Header::from_bytes(bytes).expect("header parses")
}

/// A reader that must never be touched.
struct Untouchable;

impl std::io::Read for Untouchable {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("the reader was touched before the declared length was checked")
    }
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// The repack helper itself changes nothing: the unedited re-write is byte-identical
/// and loads, so every negative below is caused by its edit alone.
#[test]
fn repack_control_loads() {
    let b = packed();
    let r = repack(&b, |_, _| {});
    assert_eq!(r, b, "repack without an edit is byte-identical");
    load_verified(&r).expect("the control loads");
}

// ---------------------------------------------------------------------------
// Rung 1: bounded read
// ---------------------------------------------------------------------------

#[test]
fn declared_length_over_cap() {
    let e = read_decide_apr_bytes_bounded(Untouchable, Some(MAX_ARTIFACT_BYTES + 1))
        .expect_err("over-cap declared length");
    assert_eq!(
        e,
        ArtifactError::ArtifactTooLarge {
            what: "declared_length",
            observed: MAX_ARTIFACT_BYTES + 1,
            cap: MAX_ARTIFACT_BYTES,
        }
    );
    assert_eq!(e.rung(), "1 bounded_read");
}

#[test]
fn read_over_cap() {
    let b = packed();
    let n = b.len() as u64;
    // Exactly the cap is a legal artifact.
    let at_cap = ArtifactLimits::tiny(n, PROBE_MAX_ROW_TOKENS);
    let ok = read_bounded_within(&b[..], Some(n), &at_cap).expect("exactly the cap reads");
    assert_eq!(ok, b);
    // A stream whose length lies (declared 0) is cut at cap + 1 and refused.
    let under = ArtifactLimits::tiny(n - 1, PROBE_MAX_ROW_TOKENS);
    let e = read_bounded_within(&b[..], Some(0), &under).expect_err("one byte over");
    assert_eq!(
        e,
        ArtifactError::ArtifactTooLarge {
            what: "stream",
            observed: n,
            cap: n - 1,
        }
    );
    // An endless stream reads at most cap + 1 bytes.
    let e = read_bounded_within(std::io::repeat(7), None, &under).expect_err("endless");
    assert_eq!(
        e,
        ArtifactError::ArtifactTooLarge {
            what: "stream",
            observed: n,
            cap: n - 1,
        }
    );
}

#[test]
fn in_memory_over_cap() {
    let b = packed();
    let n = b.len() as u64;
    let e = load_verified_within(&b, &ArtifactLimits::tiny(n - 1, PROBE_MAX_ROW_TOKENS))
        .expect_err("in-memory bytes over the cap");
    assert_eq!(
        e,
        ArtifactError::ArtifactTooLarge {
            what: "input_bytes",
            observed: n,
            cap: n - 1,
        }
    );
}

// ---------------------------------------------------------------------------
// Rung 2: header and index extent, before the reader
// ---------------------------------------------------------------------------

#[test]
fn header_crc_flip() {
    let mut b = packed();
    b[20] ^= 0x01; // metadata_size, inside the CRC-covered header
    let e = refuse(&b);
    assert_eq!(e, ArtifactError::HeaderChecksum);
    assert_eq!(e.rung(), "2 header_and_index_extent");
}

/// A forged header (valid CRC) declaring 4097 tensors is refused at rung 2. The
/// container reader, handed the same bytes, fails differently — so the refusal
/// provably came from rung 2, before any index parser ran.
#[test]
fn tensor_count_over_cap() {
    let mut b = packed();
    forge_header(&mut b, |h| h.tensor_count = MAX_TENSOR_COUNT + 1);
    assert_eq!(
        refuse(&b),
        ArtifactError::TensorCountOverCap {
            declared: MAX_TENSOR_COUNT + 1,
            cap: MAX_TENSOR_COUNT,
        }
    );
    assert!(
        AprV2ReaderRef::from_bytes(&b).is_err(),
        "the reader would have refused differently"
    );
}

#[test]
fn index_extent_too_small() {
    let mut b = packed();
    let h = header(&b);
    let extent = h.data_offset - h.tensor_index_offset;
    let declared = u32::try_from(extent / MIN_INDEX_ENTRY_BYTES + 1).expect("fits u32");
    assert!(declared <= MAX_TENSOR_COUNT, "the count is under the cap");
    forge_header(&mut b, |h| h.tensor_count = declared);
    assert_eq!(
        refuse(&b),
        ArtifactError::IndexExtentTooSmall { declared, extent }
    );
}

#[test]
fn index_past_end() {
    let mut b = packed();
    let len = b.len() as u64;
    forge_header(&mut b, |h| h.data_offset = len + 1);
    assert_eq!(
        refuse(&b),
        ArtifactError::IndexPastEnd {
            data_offset: len + 1,
            file_len: len,
        }
    );
}

/// V9-d / T-08-26-06: a forged header (valid CRC) whose version is not APR v2's is refused
/// at rung 2 naming the version. The container reader alone accepts the same bytes, so the
/// refusal is provably rung 2's own.
#[test]
fn header_version_refused() {
    for version in [(3u8, 0u8), (2, 1), (1, 0)] {
        let mut b = packed();
        forge_header(&mut b, |h| h.version = version);
        let e = refuse(&b);
        let named = format!("{}.{}", version.0, version.1);
        assert!(
            matches!(&e, ArtifactError::Header { reason } if reason.contains(&named)),
            "{named}: {e}"
        );
        assert_eq!(e.rung(), "2 header_and_index_extent", "{named}");
        assert!(
            AprV2ReaderRef::from_bytes(&b).is_ok(),
            "{named}: the container reader does not check the version itself"
        );
    }
}

/// Replace blob `name` with `new` AND re-pin its manifest sha256 — the artifact's author
/// controls both, so the blob hash alone is not a bound on what the blob declares.
fn swap_blob(bytes: &[u8], name: &str, new: Vec<u8>) -> Vec<u8> {
    let sha = crate::digest::sha256_hex(&new);
    let swapped = edit_tensor(bytes, name, |data| *data = new);
    edit_manifest(&swapped, |m| {
        for blob in m["blobs"].as_array_mut().expect("manifest blobs") {
            if blob["name"] == name {
                blob["sha256"] = serde_json::Value::String(sha.clone());
            }
        }
    })
}

/// A self-consistent artifact whose config blobs declare astronomically many layers is a
/// typed rung-4 refusal, never an allocation sized by the declared count (a 2^62-layer
/// encoder config used to abort the process inside the config parse).
#[test]
fn untrusted_layer_counts_are_bounded_before_derivation() {
    let b = packed();
    let r = AprV2ReaderRef::from_bytes(&b).expect("open the packed artifact");
    let blob_json = |name: &str| -> serde_json::Value {
        serde_json::from_slice(r.get_tensor_data(name).expect("blob")).expect("blob JSON")
    };
    let mut enc = blob_json(super::ENCODER_CONFIG_BLOB);
    enc.as_object_mut()
        .expect("encoder config object")
        .remove("layer_types");
    enc["num_hidden_layers"] = serde_json::Value::from(1u64 << 62);
    let mut agent = blob_json(super::AGENT_CONFIG_BLOB);
    agent["head_layers"] = serde_json::Value::from(1u64 << 40);
    for (blob, value) in [
        (super::ENCODER_CONFIG_BLOB, enc),
        (super::AGENT_CONFIG_BLOB, agent),
    ] {
        let forged = swap_blob(&b, blob, serde_json::to_vec(&value).expect("serialize"));
        let e = refuse(&forged);
        assert!(
            matches!(&e, ArtifactError::ConfigBlob { blob: got, .. } if *got == blob),
            "{blob}: {e}"
        );
        assert_eq!(e.rung(), "4 structural", "{blob}");
    }
}

/// A forged header (valid CRC) declaring a metadata section over the cap is refused at
/// rung 2, before the container reader would parse that many bytes into a JSON tree.
#[test]
fn metadata_over_cap() {
    let mut b = packed();
    forge_header(&mut b, |h| h.metadata_size = MAX_METADATA_BYTES + 1);
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::MetadataOverCap {
            declared: MAX_METADATA_BYTES + 1,
            cap: MAX_METADATA_BYTES,
        }
    );
    assert_eq!(e.rung(), "2 header_and_index_extent");
}

/// KANI-DECIDE-APR-001's evidence: the rung-2 predicate over every combination of edge
/// values, including overflow edges, against a u128 reference.
#[test]
fn rung2_predicate_exhaustive() {
    let counts = [0u32, 1, 2, 63, 4095, 4096, 4097, u32::MAX / 20, u32::MAX];
    let offsets = [
        0u64,
        1,
        19,
        20,
        64,
        4096,
        81_920,
        u64::MAX / 20,
        u64::MAX - 1,
        u64::MAX,
    ];
    let mut checked = 0usize;
    for &count in &counts {
        for &tio in &offsets {
            for &dof in &offsets {
                for &len in &offsets {
                    let want = count <= MAX_TENSOR_COUNT
                        && dof <= len
                        && dof >= tio
                        && u128::from(count) * u128::from(MIN_INDEX_ENTRY_BYTES)
                            <= u128::from(dof - tio.min(dof));
                    let got = check_index_extent(count, tio, dof, len).is_ok();
                    assert_eq!(got, want, "count {count} tio {tio} dof {dof} len {len}");
                    checked += 1;
                }
            }
        }
    }
    println!("rung2_predicate_exhaustive: {checked} cases");
}

// ---------------------------------------------------------------------------
// Rung 3: manifest
// ---------------------------------------------------------------------------

#[test]
fn wrong_model_type() {
    let b = repack(&packed(), |meta, _| meta.model_type = "setfit".to_string());
    assert_eq!(
        refuse(&b),
        ArtifactError::WrongModelType {
            observed: "setfit".to_string(),
        }
    );
}

#[test]
fn two_custom_keys() {
    let b = repack(&packed(), |meta, _| {
        meta.custom
            .insert("extra".to_string(), serde_json::Value::from("x"));
    });
    assert_eq!(
        refuse(&b),
        ArtifactError::CustomKeys {
            observed: vec!["decide".to_string(), "extra".to_string()],
        }
    );
}

#[test]
fn unknown_manifest_key() {
    let b = edit_manifest(&packed(), |v| v["extra"] = serde_json::Value::from(1));
    let e = refuse(&b);
    assert!(
        matches!(&e, ArtifactError::ManifestParse { reason } if reason.contains("extra")),
        "{e}"
    );
    assert_eq!(e.rung(), "3 manifest");
}

#[test]
fn schema_version_2() {
    let b = edit_manifest(&packed(), |v| {
        v["schema_version"] = serde_json::Value::from(2);
    });
    assert_eq!(refuse(&b), ArtifactError::SchemaVersion { observed: 2 });
}

#[test]
fn unknown_method() {
    let b = edit_manifest(&packed(), |v| v["method"] = serde_json::Value::from("kev"));
    assert_eq!(
        refuse(&b),
        ArtifactError::UnknownMethod {
            observed: "kev".to_string(),
        }
    );
}

// ---------------------------------------------------------------------------
// Rung 4: structure
// ---------------------------------------------------------------------------

#[test]
fn missing_tensor() {
    let b = repack(&packed(), |_, t| t.retain(|t| t.0 != "scorer.3.bias"));
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::MissingTensor {
            name: "scorer.3.bias".to_string(),
        }
    );
    assert_eq!(e.rung(), "4 structural");
}

#[test]
fn extra_tensor() {
    let b = repack(&packed(), |_, t| {
        t.push((
            "rogue.weight".to_string(),
            TensorDType::F16,
            vec![1],
            vec![0, 0],
        ));
    });
    assert_eq!(
        refuse(&b),
        ArtifactError::UnexpectedTensor {
            name: "rogue.weight".to_string(),
        }
    );
}

#[test]
fn size_mismatch() {
    let b = repack(&packed(), |_, t| {
        let e = t
            .iter_mut()
            .find(|t| t.0 == "type_emb.weight")
            .expect("type_emb");
        e.2 = vec![3, 31];
    });
    assert_eq!(
        refuse(&b),
        ArtifactError::SizeMismatch {
            name: "type_emb.weight".to_string(),
            expected: 3 * 31 * 2,
            observed: 3 * 32 * 2,
        }
    );
}

/// KANI-DECIDE-APR-003's evidence: the rung-4 size rule over bounded shapes of length
/// <= 4 for every carried dtype, against a u128 reference, including overflow.
#[test]
fn size_rule_exhaustive() {
    let dims = [0usize, 1, 2, 3, 7, 1 << 32, usize::MAX];
    let dtypes = [
        (TensorDType::F16, 2u128),
        (TensorDType::F32, 4),
        (TensorDType::U8, 1),
    ];
    let mut shapes: Vec<Vec<usize>> = vec![vec![]];
    for rank in 1..=4usize {
        let mut next = Vec::new();
        for s in shapes.iter().filter(|s| s.len() == rank - 1) {
            for &d in &dims {
                let mut t = s.clone();
                t.push(d);
                next.push(t);
            }
        }
        shapes.extend(next);
    }
    let mut checked = 0usize;
    for shape in &shapes {
        for &(dtype, width) in &dtypes {
            // The exact product: 0 whenever any dimension is 0, else the u128 product,
            // which exceeds u64 (-> None) whenever it overflows u128.
            let reference = if shape.contains(&0) {
                Some(0)
            } else {
                shape
                    .iter()
                    .try_fold(width, |a, &d| a.checked_mul(d as u128))
                    .and_then(|v| u64::try_from(v).ok())
            };
            assert_eq!(
                expected_bytes(shape, dtype),
                reference,
                "{shape:?} {dtype:?}"
            );
            if let Some(size) = reference {
                assert!(check_size("t", shape, dtype, size).is_ok());
                assert!(check_size("t", shape, dtype, size.wrapping_add(1)).is_err());
            } else {
                assert!(check_size("t", shape, dtype, 0).is_err());
                assert!(check_size("t", shape, dtype, u64::MAX).is_err());
            }
            checked += 1;
        }
    }
    assert_eq!(
        expected_bytes(&[2], TensorDType::BF16),
        None,
        "no other dtype"
    );
    println!("size_rule_exhaustive: {checked} cases");
}

#[test]
fn tokenizer_blob_hash() {
    let b = edit_tensor(&packed(), "tokenizer.blob", |d| d[0] ^= 0x01);
    assert_eq!(
        refuse(&b),
        ArtifactError::BlobHashMismatch {
            blob: "tokenizer.blob".to_string(),
        }
    );
}

#[test]
fn task_blob_hash() {
    let b = edit_tensor(&packed(), "decide.task_json", |d| d[0] ^= 0x01);
    assert_eq!(
        refuse(&b),
        ArtifactError::BlobHashMismatch {
            blob: "decide.task_json".to_string(),
        }
    );
}

/// The contract's qa_gate falsification: one flipped byte of the recipe blob is refused
/// at rung 4 naming the blob (so recipe_id can never describe other bytes).
#[test]
fn recipe_blob_hash() {
    let b = edit_tensor(&packed(), "decide.recipe_json", |d| {
        let last = d.len() - 1;
        d[last] ^= 0x01;
    });
    assert_eq!(
        refuse(&b),
        ArtifactError::BlobHashMismatch {
            blob: "decide.recipe_json".to_string(),
        }
    );
}

/// WR-01, decide side: the rung-4 repeat walk names the first repeated name whatever the
/// order of the index — an adjacent repeat, a non-adjacent one — and `None` when every
/// name is unique. This is the red-side proof of the rung-4 logic: while plan 08-20's
/// reader refuses a repeated name at rung 3, no real artifact reaches the rung-4 call.
#[test]
fn repeated_name_walk_names_the_first_repeat() {
    assert_eq!(
        super::first_repeated_tensor_name(["a", "b", "b", "c"]),
        Some("b"),
        "adjacent repeat"
    );
    assert_eq!(
        super::first_repeated_tensor_name(["a", "b", "c", "a"]),
        Some("a"),
        "non-adjacent repeat (not visible to a sorted-neighbour check)"
    );
    assert_eq!(
        super::first_repeated_tensor_name(["b", "a", "b", "a"]),
        Some("b"),
        "the FIRST repeat in index order"
    );
    assert_eq!(super::first_repeated_tensor_name(["a", "b", "c"]), None);
    assert_eq!(super::first_repeated_tensor_name(Vec::<&str>::new()), None);
}

/// WR-01, decide side: an artifact whose index names one encoder weight twice (the writer
/// wrote both entries) is refused at load through the stdio door, and the refusal names the
/// tensor. While plan 08-20's reader change stands the refusal is the container's (rung 3,
/// `duplicate tensor name ...`); rung 4's `DuplicateTensor` is the decide-side backstop.
#[test]
fn duplicate_tensor_name_is_refused_at_load() {
    let b = packed();
    let name = AprV2ReaderRef::from_bytes(&b)
        .expect("open the packed artifact")
        .tensor_index()
        .iter()
        .map(|e| e.name.clone())
        .find(|n| n.starts_with("encoder."))
        .expect("an encoder weight");
    let dup = repack(&b, |_, tensors| {
        let copy = tensors
            .iter()
            .find(|t| t.0 == name)
            .cloned()
            .expect("the weight to duplicate");
        tensors.push(copy);
    });
    assert_eq!(
        header(&dup).tensor_count,
        header(&b).tensor_count + 1,
        "the writer wrote both entries"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("duplicate-name.apr");
    std::fs::write(&path, &dup).expect("write the forged artifact");
    let e = crate::Decider::load_path(&path).expect_err("a repeated tensor name loaded");
    match &e {
        ArtifactError::DuplicateTensor { name: got } => {
            assert_eq!(got, &name, "{e}");
            assert_eq!(e.rung(), "4 structural");
        }
        ArtifactError::Container { reason } => {
            assert!(
                reason.contains("duplicate") && reason.contains(&name),
                "the container refusal must name the repeated tensor: {e}"
            );
            assert_eq!(e.rung(), "3 manifest");
        }
        other => panic!("refused for another reason: {other}"),
    }
    println!(
        "duplicate_tensor_name_is_refused_at_load: {name} refused at rung {}: {e}",
        e.rung()
    );
}

/// `inspect` prints identity only after rungs 1-4, so a manifest whose base its own recipe
/// blob contradicts is refused by `inspect_manifest` too (IN-02, A2-3); the packed control
/// inspects.
#[test]
fn inspect_refuses_a_manifest_its_blobs_contradict() {
    super::inspect_manifest(&packed()).expect("control: the packed artifact inspects");
    let b = edit_manifest(&packed(), |v| {
        v["base"]["revision"] = serde_json::Value::from("0123456789abcdef0123456789abcdef01234567");
    });
    let e = super::inspect_manifest(&b).expect_err("inspect printed a contradicted identity");
    assert_eq!(
        e,
        ArtifactError::ManifestDisagreesWithBlob { field: "base" }
    );
    assert_eq!(e.rung(), "4 structural");
}

#[test]
fn labels_disagree_with_task() {
    let b = edit_manifest(&packed(), |v| {
        let labels = v["labels"].as_array_mut().expect("labels");
        labels.reverse();
    });
    assert_eq!(
        refuse(&b),
        ArtifactError::LabelsDisagreeWithTask {
            manifest: vec!["account".into(), "billing".into(), "shipping".into()],
            task: vec!["shipping".into(), "billing".into(), "account".into()],
        }
    );
}

// ---------------------------------------------------------------------------
// Rung 4 (e): manifest bindings (decide-apr-v1 manifest.bindings, plan 08-19)
// ---------------------------------------------------------------------------

/// `manifest.base` is minted into `ModelIdentity.base` (served as `model.base` in every
/// classify response): a revision that is not the embedded recipe blob's is refused at
/// rung 4, and so is a variant the recipe does not declare.
#[test]
fn manifest_base_disagrees_with_recipe_blob() {
    let b = edit_manifest(&packed(), |v| {
        v["base"]["revision"] = serde_json::Value::from("0123456789abcdef0123456789abcdef01234567");
    });
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::ManifestDisagreesWithBlob { field: "base" }
    );
    assert_eq!(e.rung(), "4 structural");

    let b = edit_manifest(&packed(), |v| {
        v["variant"] = serde_json::Value::from("production");
    });
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::ManifestDisagreesWithBlob { field: "variant" }
    );
    assert_eq!(e.rung(), "4 structural");
}

/// The same crafted bytes are refused through every door a server loads by: the in-memory
/// door, the pre-hashed door the Lambda cold start uses, and the path door the stdio server
/// uses — none of them needs `verify` to see it.
#[test]
fn manifest_bound_at_every_load_door() {
    let want = ArtifactError::ManifestDisagreesWithBlob { field: "base" };
    let b = edit_manifest(&packed(), |v| {
        v["base"]["checkpoint"] = serde_json::Value::from("en-root");
    });
    let bytes_door = crate::Decider::load_bytes(&b).expect_err("load_bytes accepted a forged base");
    assert_eq!(bytes_door, want, "load_bytes");
    let hashed_door = crate::Decider::load_hashed(&super::HashedArtifact::new(&b))
        .expect_err("load_hashed accepted a forged base");
    assert_eq!(hashed_door, want, "load_hashed (the Lambda door)");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("forged-base.apr");
    std::fs::write(&path, &b).expect("write the forged artifact");
    let path_door = crate::Decider::load_path(&path).expect_err("load_path accepted a forged base");
    assert_eq!(path_door, want, "load_path (the stdio door)");
    // Control: the unforged bytes load through the same three doors.
    let ok = packed();
    crate::Decider::load_bytes(&ok).expect("control: load_bytes");
    crate::Decider::load_hashed(&super::HashedArtifact::new(&ok)).expect("control: load_hashed");
    let ok_path = dir.path().join("control.apr");
    std::fs::write(&ok_path, &ok).expect("write the control artifact");
    crate::Decider::load_path(&ok_path).expect("control: load_path");
}

// ---------------------------------------------------------------------------
// Rung 4 (e): every manifest leaf, bound (plan 08-19 Task 2)
// ---------------------------------------------------------------------------

/// JSON-pointer edits: set each pointer to its value.
type Edits = Vec<(String, Value)>;

/// A COHERENT forgery. The artifact's author controls every blob and every hash, so a
/// forger who edits the recipe or the gate report also re-pins what hashes them: the
/// recipe's digest into `manifest.blobs`, `manifest.recipe_id` and the report's `recipe_id`;
/// the report's digest into `manifest.blobs` and `manifest.gate.report_sha256`. Explicit
/// report edits land after the recipe re-pin and explicit manifest edits land last, so an
/// edit always wins over a re-pin. With no edits the bytes are unchanged.
fn forge(
    bytes: &[u8],
    recipe: &[(String, Value)],
    report: &[(String, Value)],
    manifest: &[(String, Value)],
) -> Vec<u8> {
    let set = |v: &mut Value, edits: &[(String, Value)]| {
        for (ptr, new) in edits {
            *v.pointer_mut(ptr)
                .unwrap_or_else(|| panic!("forge: no field {ptr}")) = new.clone();
        }
    };
    let r = AprV2ReaderRef::from_bytes(bytes).expect("open the packed artifact");
    let blob = |name: &str| r.get_tensor_data(name).expect("blob").to_vec();
    let mut recipe_bytes = blob(super::RECIPE_BLOB);
    let mut report_bytes = blob(super::GATE_REPORT_BLOB);
    let mut pins = Edits::new();
    let mut report_edits = Edits::new();
    if !recipe.is_empty() {
        let mut v: Value = serde_json::from_slice(&recipe_bytes).expect("recipe JSON");
        set(&mut v, recipe);
        recipe_bytes = serde_json::to_vec(&v).expect("recipe re-serializes");
        let sha = Value::from(crate::digest::sha256_hex(&recipe_bytes));
        pins.push(("/blobs/4/sha256".into(), sha.clone()));
        pins.push(("/recipe_id".into(), sha.clone()));
        report_edits.push(("/recipe_id".into(), sha));
    }
    report_edits.extend(report.iter().cloned());
    if !report_edits.is_empty() {
        let mut v: Value = serde_json::from_slice(&report_bytes).expect("report JSON");
        set(&mut v, &report_edits);
        report_bytes = serde_json::to_vec(&v).expect("report re-serializes");
        let sha = Value::from(crate::digest::sha256_hex(&report_bytes));
        pins.push(("/blobs/5/sha256".into(), sha.clone()));
        pins.push(("/gate/report_sha256".into(), sha));
    }
    let swapped = repack(bytes, |_, tensors| {
        for t in tensors.iter_mut() {
            if t.0 == super::RECIPE_BLOB {
                t.2 = vec![recipe_bytes.len()];
                t.3 = recipe_bytes.clone();
            } else if t.0 == super::GATE_REPORT_BLOB {
                t.2 = vec![report_bytes.len()];
                t.3 = report_bytes.clone();
            }
        }
    });
    let all: Edits = pins.into_iter().chain(manifest.iter().cloned()).collect();
    if all.is_empty() {
        return swapped;
    }
    edit_manifest(&swapped, |v| set(v, &all))
}

fn edits(pairs: &[(&str, Value)]) -> Edits {
    pairs
        .iter()
        .map(|(p, v)| ((*p).to_string(), v.clone()))
        .collect()
}

/// The packed tiny manifest as JSON.
fn manifest_json(bytes: &[u8]) -> Value {
    let m = super::inspect_manifest(bytes).expect("the packed manifest");
    serde_json::to_value(m).expect("manifest to JSON")
}

/// A hex digest with its first digit changed (same length, still hex).
fn flip_hex(h: &str) -> String {
    let first = if h.starts_with('0') { '1' } else { '0' };
    std::iter::once(first).chain(h.chars().skip(1)).collect()
}

fn is_hex_digest(s: &str) -> bool {
    s.len() >= 8 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The sweep's deterministic mutation of one leaf. A stored probe probability moves at
/// least 0.25, so rung 7 cannot accept it within probe_probabilities_abs.
fn mutate(pattern: &str, v: &Value) -> Value {
    if pattern.ends_with("/probabilities_f32_hex/[*]") {
        let h = v.as_str().expect("probe hex");
        let p = f32::from_bits(u32::from_str_radix(h, 16).expect("probe hex parses"));
        let moved = if p >= 0.5 { p - 0.3 } else { p + 0.3 };
        return Value::from(format!("{:08x}", moved.to_bits()));
    }
    match v {
        Value::String(s) if is_hex_digest(s) => Value::from(flip_hex(s)),
        Value::String(s) => Value::from(format!("{s}x")),
        Value::Bool(b) => Value::from(!b),
        Value::Number(n) if n.is_u64() => Value::from(n.as_u64().expect("u64") + 1),
        Value::Number(n) if n.is_i64() => Value::from(n.as_i64().expect("i64") + 1),
        Value::Number(n) => Value::from(n.as_f64().expect("f64") * 2.0 + 1.0),
        other => panic!("mutate: {pattern} is not a leaf: {other}"),
    }
}

/// Every leaf of `v` as `(pointer, pattern)`, the pattern with array indices as `[*]`.
fn leaves(v: &Value, ptr: &str, pattern: &str, out: &mut Vec<(String, String)>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                leaves(child, &format!("{ptr}/{k}"), &format!("{pattern}/{k}"), out);
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                leaves(child, &format!("{ptr}/{i}"), &format!("{pattern}/[*]"), out);
            }
        }
        _ => out.push((ptr.to_string(), pattern.to_string())),
    }
}

/// One `manifest.bindings` row.
struct Binding {
    rung: u64,
    sources: Vec<String>,
}

/// decide-apr-v1 `manifest.bindings`, read from the contract.
fn bindings_table() -> BTreeMap<String, Binding> {
    let c = crate::test_support::contract_yaml("decide-apr-v1.yaml");
    c["manifest"]["bindings"]
        .as_mapping()
        .expect("decide-apr-v1 manifest.bindings")
        .iter()
        .map(|(k, row)| {
            let leaf = k.as_str().expect("binding key").to_string();
            let rung = row["rung"].as_u64().expect("binding rung");
            let sources = row["sources"]
                .as_sequence()
                .unwrap_or_else(|| panic!("{leaf}: sources"))
                .iter()
                .map(|s| s.as_str().expect("source string").to_string())
                .collect();
            (leaf, Binding { rung, sources })
        })
        .collect()
}

/// A settable node of the binding graph: a manifest leaf, or a field of the recipe or
/// gate-report blob. Every other source (`sha256:`, `derived:`) is a FIXED value.
fn settable(node: &str) -> bool {
    ["manifest:", "recipe:", "report:"]
        .iter()
        .any(|p| node.starts_with(p))
}

/// The binding graph over concrete nodes: an edge per (manifest leaf, source).
fn binding_edges(
    leaf_ptrs: &[(String, String)],
    table: &BTreeMap<String, Binding>,
) -> Vec<(String, String)> {
    let mut edges = Vec::new();
    for (ptr, pattern) in leaf_ptrs {
        if let Some(row) = table.get(pattern) {
            for s in &row.sources {
                edges.push((format!("manifest:{ptr}"), s.clone()));
            }
        }
    }
    edges
}

/// The settable nodes connected to `start` without crossing `skip`, and whether any of them
/// is also tied to a fixed source (then the component cannot move as one).
fn component(start: &str, edges: &[(String, String)], skip: usize) -> (BTreeSet<String>, bool) {
    let mut seen = BTreeSet::from([start.to_string()]);
    let mut todo = vec![start.to_string()];
    let mut pinned = false;
    while let Some(n) = todo.pop() {
        for (i, (a, b)) in edges.iter().enumerate() {
            if i == skip {
                continue;
            }
            let other = if *a == n {
                b
            } else if *b == n {
                a
            } else {
                continue;
            };
            if !settable(other) {
                pinned = true;
            } else if seen.insert(other.clone()) {
                todo.push(other.clone());
            }
        }
    }
    (seen, pinned)
}

/// A forgery that breaks EXACTLY the one binding `edges[i]`: every other binding still
/// holds. The component of the leaf (without edge i) moves to the mutated value as one; if
/// that component is tied to a fixed source, the source's component moves instead.
fn break_only(
    bytes: &[u8],
    edges: &[(String, String)],
    i: usize,
    pattern: &str,
    old: &Value,
) -> Vec<u8> {
    let (leaf, source) = &edges[i];
    let (comp, pinned) = component(leaf, edges, i);
    let comp = if pinned {
        assert!(
            settable(source),
            "{leaf} <-> {source}: both ends are fixed; no forgery breaks only this binding"
        );
        let (c, p) = component(source, edges, i);
        assert!(
            !p,
            "{leaf} <-> {source}: both components are tied to a fixed source"
        );
        c
    } else {
        comp
    };
    let new = mutate(pattern, old);
    let (mut recipe, mut report, mut manifest) = (Edits::new(), Edits::new(), Edits::new());
    for node in comp {
        let (kind, ptr) = node.split_once(':').expect("node kind");
        let target = match kind {
            "manifest" => &mut manifest,
            "recipe" => &mut recipe,
            "report" => &mut report,
            other => panic!("{node}: {other} is not settable"),
        };
        target.push((ptr.to_string(), new.clone()));
    }
    forge(bytes, &recipe, &report, &manifest)
}

/// `/calibration/t_applied` -> `calibration.t_applied`.
fn dotted(ptr: &str) -> String {
    ptr.trim_start_matches('/').replace('/', ".")
}

fn rung_number(e: &ArtifactError) -> u64 {
    e.rung()
        .split(' ')
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{e}: not a load rung"))
}

/// The forge itself changes nothing: with no edits the bytes are byte-identical, and a
/// recipe or report re-serialized with every hash re-pinned still loads.
#[test]
fn manifest_forge_control_loads() {
    let b = packed();
    assert_eq!(forge(&b, &[], &[], &[]), b, "no edits: byte-identical");
    let r = forge(
        &b,
        &edits(&[("/seed", Value::from(20_260_925))]),
        &edits(&[("/device_used", Value::from("cpu"))]),
        &[],
    );
    assert_ne!(r, b, "the blobs were re-serialized and re-pinned");
    load_verified(&r).expect("a coherent re-pin of unchanged values loads");
}

fn assert_field(bytes: &[u8], field: &'static str) {
    let e = refuse(bytes);
    assert_eq!(e, ArtifactError::ManifestDisagreesWithBlob { field }, "{e}");
    assert_eq!(e.rung(), "4 structural");
}

/// `manifest.gate.report_sha256` must be the digest of the EMBEDDED gate report.
#[test]
fn manifest_gate_report_sha_disagrees_with_blob() {
    let b = packed();
    let m = manifest_json(&b);
    let flipped = flip_hex(m["gate"]["report_sha256"].as_str().expect("sha"));
    let f = forge(
        &b,
        &[],
        &[],
        &edits(&[("/gate/report_sha256", Value::from(flipped))]),
    );
    assert_field(&f, "gate.report_sha256");
}

/// `inputs_sha256.task_json` / `.tokenizer_json` must be the embedded blobs' digests — even
/// when the embedded report agrees with the manifest's forged value.
#[test]
fn manifest_inputs_disagree_with_blobs() {
    let b = packed();
    let m = manifest_json(&b);
    for (ptr, field) in [
        ("/inputs_sha256/task_json", "inputs_sha256.task_json"),
        (
            "/inputs_sha256/tokenizer_json",
            "inputs_sha256.tokenizer_json",
        ),
    ] {
        let flipped = Value::from(flip_hex(m.pointer(ptr).and_then(Value::as_str).expect(ptr)));
        let f = forge(
            &b,
            &[],
            &edits(&[(ptr, flipped.clone())]),
            &edits(&[(ptr, flipped)]),
        );
        assert_field(&f, field);
    }
}

/// Every `inputs_sha256` field must equal the embedded report's.
#[test]
fn manifest_inputs_disagree_with_report() {
    let b = packed();
    let m = manifest_json(&b);
    for (key, field) in [
        ("task_json", "inputs_sha256.task_json"),
        ("train_jsonl", "inputs_sha256.train_jsonl"),
        ("eval_jsonl", "inputs_sha256.eval_jsonl"),
        ("base_model", "inputs_sha256.base_model"),
        ("tokenizer_json", "inputs_sha256.tokenizer_json"),
    ] {
        let ptr = format!("/inputs_sha256/{key}");
        let flipped = Value::from(flip_hex(
            m.pointer(&ptr).and_then(Value::as_str).expect("sha"),
        ));
        // The report side moves, so the blob-digest bindings still hold.
        let f = forge(&b, &[], &edits(&[(&ptr, flipped)]), &[]);
        assert_field(&f, field);
    }
}

/// `base.sha256` must equal the report's `inputs_sha256.base_model`, even when the recipe
/// (and so `manifest.base`) was re-pinned to agree with the forged value.
#[test]
fn manifest_base_sha_disagrees_with_report() {
    let b = packed();
    let m = manifest_json(&b);
    let flipped = Value::from(flip_hex(m["base"]["sha256"].as_str().expect("sha")));
    let f = forge(
        &b,
        &edits(&[("/base/sha256", flipped.clone())]),
        &[],
        &edits(&[("/base/sha256", flipped)]),
    );
    assert_field(&f, "base.sha256");
}

/// `recipe_id` must equal the embedded report's `recipe_id` (the recipe blob digest is
/// already rung 4's RecipeIdMismatch).
#[test]
fn manifest_recipe_id_disagrees_with_report() {
    let b = packed();
    let m = manifest_json(&b);
    let flipped = Value::from(flip_hex(m["recipe_id"].as_str().expect("recipe_id")));
    let f = forge(&b, &[], &edits(&[("/recipe_id", flipped)]), &[]);
    assert_field(&f, "recipe_id");
}

/// `gate.pass` / `gate.margin` / `gate.ece_post` must equal the embedded report's (f64 bits).
#[test]
fn manifest_gate_summary_disagrees_with_report() {
    let b = packed();
    for (ptr, new, field) in [
        ("/gate/pass", Value::from(true), "gate.pass"),
        ("/gate/margin", Value::from(0.25), "gate.margin"),
        ("/gate/ece_post", Value::from(0.01), "gate.ece_post"),
    ] {
        let f = forge(&b, &[], &[], &edits(&[(ptr, new)]));
        assert_field(&f, field);
    }
}

/// Every calibration field must equal the embedded report's calibration block.
#[test]
fn manifest_calibration_disagrees_with_report() {
    let b = packed();
    for (ptr, report_ptr, new, field) in [
        (
            "/calibration/t_fitted",
            "",
            Value::from(1.5),
            "calibration.t_fitted",
        ),
        (
            "/calibration/clamp_hit",
            "",
            Value::from(true),
            "calibration.clamp_hit",
        ),
        (
            "/calibration/slice_ids_sha256",
            "",
            Value::from("0".repeat(64)),
            "calibration.slice_ids_sha256",
        ),
        // bucket and t_applied are also bound to the task / agent config, so the REPORT
        // moves and the manifest keeps the value the model actually runs with.
        (
            "",
            "/calibration/bucket",
            Value::from("choice:2"),
            "calibration.bucket",
        ),
        (
            "",
            "/calibration/t_applied",
            Value::from(1.5),
            "calibration.t_applied",
        ),
    ] {
        let report = if report_ptr.is_empty() {
            Edits::new()
        } else {
            edits(&[(report_ptr, new.clone())])
        };
        let manifest = if ptr.is_empty() {
            Edits::new()
        } else {
            edits(&[(ptr, new.clone())])
        };
        let f = forge(&b, &[], &report, &manifest);
        assert_field(&f, field);
    }
}

/// `calibration.bucket` must be the task's bucket (`bucket_key(choice, K)`), even when the
/// embedded report agrees with the forged value.
#[test]
fn manifest_calibration_bucket_disagrees_with_task() {
    let b = packed();
    let wrong = edits(&[("/calibration/bucket", Value::from("choice:2"))]);
    let f = forge(&b, &[], &wrong, &wrong);
    assert_field(&f, "calibration.bucket");
}

/// `calibration.t_applied` must be, bit for bit, the temperature the loaded model applies
/// to the task (the agent config's clamped bucket lookup), even when the report agrees.
#[test]
fn manifest_t_applied_disagrees_with_agent() {
    let b = packed();
    // 1.5 is a real temperature of the tiny agent config, for another bucket (choice:2).
    let wrong = edits(&[("/calibration/t_applied", Value::from(1.5))]);
    let f = forge(&b, &[], &wrong, &wrong);
    assert_field(&f, "calibration.t_applied");
    // One ULP away is still a different temperature.
    let m = manifest_json(&b);
    let t = m["calibration"]["t_applied"].as_f64().expect("t_applied");
    let ulp = edits(&[(
        "/calibration/t_applied",
        Value::from(f64::from_bits(t.to_bits() + 1)),
    )]);
    assert_field(&forge(&b, &[], &ulp, &ulp), "calibration.t_applied");
}

/// `device_used` must equal the embedded report's.
#[test]
fn manifest_device_used_disagrees_with_report() {
    let b = packed();
    let f = forge(
        &b,
        &[],
        &[],
        &edits(&[("/device_used", Value::from("mps:0"))]),
    );
    assert_field(&f, "device_used");
}

/// FALSIFY-DECIDE-APR-013: EVERY leaf of the packed manifest, for EVERY binding the
/// contract table names for it, is forged so that exactly that one binding breaks — and
/// the ladder must refuse it at the table's rung. A binding whose check is disabled lets
/// its forgery load, which this test names.
#[test]
fn every_manifest_leaf_is_bound() {
    let b = packed();
    let m = manifest_json(&b);
    let mut ptrs = Vec::new();
    leaves(&m, "", "", &mut ptrs);
    let table = bindings_table();
    let edges = binding_edges(&ptrs, &table);
    let pattern_of: BTreeMap<&str, &str> = ptrs
        .iter()
        .map(|(p, pat)| (p.as_str(), pat.as_str()))
        .collect();
    let mut failures = Vec::new();
    for (i, (leaf, source)) in edges.iter().enumerate() {
        let ptr = leaf.trim_start_matches("manifest:");
        let pattern = pattern_of[ptr];
        let row = &table[pattern];
        let old = m.pointer(ptr).expect("leaf value");
        let forged = break_only(&b, &edges, i, pattern, old);
        match load_verified(&forged) {
            Ok(_) => failures.push(format!(
                "{ptr} <-> {source}: LOADED after the binding was broken"
            )),
            Err(e) => {
                let rung = rung_number(&e);
                if rung != row.rung {
                    failures.push(format!(
                        "{ptr} <-> {source}: refused at rung {rung}, the table names {}: {e}",
                        row.rung
                    ));
                }
                if let ArtifactError::ManifestDisagreesWithBlob { field } = &e {
                    let d = dotted(ptr);
                    if !(d == *field || d.starts_with(&format!("{field}."))) {
                        failures.push(format!(
                            "{ptr} <-> {source}: refused naming another field: {e}"
                        ));
                    }
                }
            }
        }
    }
    println!(
        "every_manifest_leaf_is_bound: {} bindings over {} leaves",
        edges.len(),
        ptrs.len()
    );
    assert!(
        failures.is_empty(),
        "unbound manifest leaves:\n{}",
        failures.join("\n")
    );
}

/// The contract table and the manifest agree leaf for leaf: a stale row, or a manifest leaf
/// added without a binding, fails here naming it. Every source is a known kind and every
/// `recipe:` / `report:` pointer resolves in the embedded blobs.
#[test]
fn manifest_bindings_table_matches_manifest_leaves() {
    let b = packed();
    let mut ptrs = Vec::new();
    leaves(&manifest_json(&b), "", "", &mut ptrs);
    let observed: BTreeSet<String> = ptrs.into_iter().map(|(_, p)| p).collect();
    let table = bindings_table();
    let listed: BTreeSet<String> = table.keys().cloned().collect();
    let unbound: Vec<_> = observed.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&observed).collect();
    assert!(
        unbound.is_empty(),
        "manifest leaves with no manifest.bindings row: {unbound:?}"
    );
    assert!(
        stale.is_empty(),
        "manifest.bindings rows for no manifest leaf: {stale:?}"
    );

    let r = AprV2ReaderRef::from_bytes(&b).expect("open the packed artifact");
    let json = |name: &str| -> Value {
        serde_json::from_slice(r.get_tensor_data(name).expect("blob")).expect("blob JSON")
    };
    let (recipe, report) = (json(super::RECIPE_BLOB), json(super::GATE_REPORT_BLOB));
    for (leaf, row) in &table {
        assert!([3, 4, 7].contains(&row.rung), "{leaf}: rung {}", row.rung);
        assert!(!row.sources.is_empty(), "{leaf}: no source");
        for s in &row.sources {
            let (kind, rest) = s
                .split_once(':')
                .unwrap_or_else(|| panic!("{leaf}: source {s} has no kind"));
            match kind {
                "recipe" => assert!(
                    recipe.pointer(rest).is_some(),
                    "{leaf}: {s} is not in the recipe blob"
                ),
                "report" => assert!(
                    report.pointer(rest).is_some(),
                    "{leaf}: {s} is not in the gate report"
                ),
                "sha256" => assert!(
                    super::BLOB_TENSORS.contains(&rest),
                    "{leaf}: {s} names no blob"
                ),
                "derived" => assert!(!rest.trim().is_empty(), "{leaf}: {s} says nothing"),
                other => panic!("{leaf}: unknown source kind {other}"),
            }
            assert!(
                !leaf.contains("[*]") || !settable(s),
                "{leaf}: an array leaf may only bind to a fixed source, not {s}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rung 5: non-finite scan
// ---------------------------------------------------------------------------

/// One F16 weight set to 0x7E00 (a NaN) AFTER packing — weights carry no hash, so only
/// rung 5 can see it.
#[test]
fn nan_weight() {
    let b = edit_tensor(&packed(), "scorer.1.weight", |d| {
        d[..2].copy_from_slice(&0x7E00u16.to_le_bytes());
    });
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::NonFiniteWeight {
            name: "scorer.1.weight".to_string(),
        }
    );
    assert_eq!(e.rung(), "5 non_finite_scan");
}

// ---------------------------------------------------------------------------
// Rung 7: probe replay
// ---------------------------------------------------------------------------

/// One stored probe probability moved by 2e-5 (over probe_probabilities_abs) is
/// refused at rung 7.
#[test]
fn probe_mismatch() {
    let b = edit_manifest(&packed(), |v| {
        let h = &mut v["probes"][0]["probabilities_f32_hex"][0];
        let bits = u32::from_str_radix(h.as_str().expect("hex"), 16).expect("hex parses");
        let moved = f32::from_bits(bits) + 2e-5;
        *h = serde_json::Value::from(format!("{:08x}", moved.to_bits()));
    });
    let e = refuse(&b);
    assert_eq!(
        e,
        ArtifactError::ProbeMismatch {
            index: 0,
            component: "probabilities",
        }
    );
    assert_eq!(e.rung(), "7 probe_replay");
}

/// Swap blob `name` for `new` (data AND its `[len]` shape) and re-pin its manifest sha256.
fn swap_blob_sized(bytes: &[u8], name: &str, new: &[u8]) -> Vec<u8> {
    let sha = crate::digest::sha256_hex(new);
    let swapped = repack(bytes, |_, tensors| {
        let t = tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("blob {name}"));
        t.2 = vec![new.len()];
        t.3 = new.to_vec();
    });
    edit_manifest(&swapped, |m| {
        for blob in m["blobs"].as_array_mut().expect("manifest blobs") {
            if blob["name"] == name {
                blob["sha256"] = Value::String(sha.clone());
            }
        }
    })
}

/// Blob `name` of `bytes` as JSON.
fn blob_json(bytes: &[u8], name: &str) -> Value {
    let r = AprV2ReaderRef::from_bytes(bytes).expect("open the artifact");
    serde_json::from_slice(r.get_tensor_data(name).expect("blob")).expect("blob JSON")
}

/// A COHERENT forgery replacing the tokenizer blob with `doc`: the blob's manifest digest and
/// both `inputs_sha256.tokenizer_json` pins (manifest and gate report) are re-pinned, so the
/// only thing the ladder can object to is what the tokenizer DOES.
fn with_tokenizer(bytes: &[u8], doc: &Value) -> Vec<u8> {
    let new = serde_json::to_vec(doc).expect("tokenizer re-serializes");
    let sha = Value::from(crate::digest::sha256_hex(&new));
    let swapped = swap_blob_sized(bytes, super::TOKENIZER_BLOB, &new);
    let pin = edits(&[("/inputs_sha256/tokenizer_json", sha)]);
    forge(&swapped, &[], &pin, &pin)
}

/// A COHERENT forgery of one agent-config field: the blob swapped and re-pinned, and
/// `manifest.agent.<field>` set to the same value.
fn with_agent_field(bytes: &[u8], field: &str, value: Value) -> Vec<u8> {
    let mut agent = blob_json(bytes, super::AGENT_CONFIG_BLOB);
    agent[field] = value.clone();
    let new = serde_json::to_vec(&agent).expect("agent re-serializes");
    let swapped = swap_blob_sized(bytes, super::AGENT_CONFIG_BLOB, &new);
    forge(
        &swapped,
        &[],
        &[],
        &edits(&[(&format!("/agent/{field}"), value)]),
    )
}

/// Blob `name` rewritten by `edit` and re-pinned (its digest only): enough for a refusal the
/// blob's own parse gives, which rung 4 (b) / (d) reach before the manifest bindings (e).
fn with_blob_json(bytes: &[u8], name: &str, edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut doc = blob_json(bytes, name);
    edit(&mut doc);
    swap_blob_sized(
        bytes,
        name,
        &serde_json::to_vec(&doc).expect("blob re-serializes"),
    )
}

/// The packed artifact with its tokenizer's merges removed (coherently re-pinned): every
/// character is a token, so probe input 0's row (43 state tokens) is cut at the tiny
/// max_len 64 — over probe_max_row_tokens — while the served task still keeps its markers.
fn over_budget_probe_artifact() -> Vec<u8> {
    let b = packed();
    let mut doc = blob_json(&b, super::TOKENIZER_BLOB);
    doc["model"]["merges"] = Value::Array(Vec::new());
    with_tokenizer(&b, &doc)
}

/// The load-time replay checks every probe row against probe_max_row_tokens BEFORE any
/// forward pass: an artifact whose tokenizer builds an over-budget probe row is refused at
/// rung 7 naming the row, and `forward_row` was never entered. (The refusal alone is the same
/// before or after a forward, so the forward counter is the observation.)
#[test]
fn probe_row_budget_checked_before_replay_forward() {
    let forged = over_budget_probe_artifact();
    crate::laya::FORWARD_ROWS.with(|n| n.set(0));
    let e = refuse(&forged);
    let forwards = crate::laya::FORWARD_ROWS.with(std::cell::Cell::get);
    assert_eq!(
        e,
        ArtifactError::ProbeMismatch {
            index: 0,
            component: "tokens",
        }
    );
    assert_eq!(e.rung(), "7 probe_replay");
    assert_eq!(
        forwards, 0,
        "the over-budget probe row reached {forwards} forward pass(es) before the budget refused it"
    );
    // Control: the same counter moves on a load that replays (so it observes the forward).
    crate::laya::FORWARD_ROWS.with(|n| n.set(0));
    load_verified(&packed()).expect("control loads");
    assert!(
        crate::laya::FORWARD_ROWS.with(std::cell::Cell::get) >= super::PROBE_INPUTS.len(),
        "the control's replay was counted"
    );
}

/// V9-b: a classify failure during LOAD-time probe replay is a rung-7 refusal, not a
/// rung-6 one. The forged tokenizer blob maps one token of probe input 0 to an id past the
/// encoder's vocabulary: rung 6 only tokenizes the served task's (empty-state) prefix, so it
/// rebuilds; the first forward is the replay, where core's embedding lookup refuses the id
/// (`OutOfVocab`). Every digest bound to the tokenizer is re-pinned, and the control — the
/// same coherent re-serialization with the id left alone — loads.
#[test]
fn probe_replay_failure_is_rung_7() {
    let b = packed();
    let r = AprV2ReaderRef::from_bytes(&b).expect("open the packed artifact");
    let tok_bytes = r
        .get_tensor_data(super::TOKENIZER_BLOB)
        .expect("tokenizer blob")
        .to_vec();
    let encoder: Value = serde_json::from_slice(
        r.get_tensor_data(super::ENCODER_CONFIG_BLOB)
            .expect("encoder blob"),
    )
    .expect("encoder JSON");
    let vocab_size = encoder["vocab_size"].as_u64().expect("vocab_size");
    let tok = tokenizers::Tokenizer::from_bytes(&tok_bytes).expect("tokenizer");
    let probe = tok
        .encode(super::PROBE_INPUTS[0], false)
        .expect("probe input 0 encodes");
    let doc: Value = serde_json::from_slice(&tok_bytes).expect("tokenizer JSON");
    let vocab = doc["model"]["vocab"].as_object().expect("BPE vocab");
    let victim = probe
        .get_tokens()
        .iter()
        .find(|t| vocab.contains_key(t.as_str()))
        .expect("a vocab token in probe input 0")
        .clone();
    let far = vocab_size + 1000;
    assert!(
        vocab.values().all(|id| id.as_u64() != Some(far)),
        "the forged id is unused"
    );
    let forged_tokenizer = |id: Option<u64>| -> Vec<u8> {
        let mut d = doc.clone();
        if let Some(id) = id {
            d["model"]["vocab"][victim.as_str()] = Value::from(id);
        }
        with_tokenizer(&b, &d)
    };
    load_verified(&forged_tokenizer(None))
        .expect("control: the coherent re-serialized tokenizer loads");
    let e = refuse(&forged_tokenizer(Some(far)));
    assert!(
        matches!(&e, ArtifactError::ProbeReplay { reason } if reason.contains("out of vocabulary")),
        "{victim} -> {far}: {e}"
    );
    assert_eq!(e.rung(), "7 probe_replay");
}

// ---------------------------------------------------------------------------
// CLASS B, artifact half: decide-apr-v1 untrusted_input_bounds (plan 08-26)
// ---------------------------------------------------------------------------

/// A row value as text (a YAML string or number).
fn row_text(row: &serde_yaml::Value, key: &str) -> String {
    match &row[key] {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

/// Every `.rs` file under `dir`, concatenated.
fn rust_source(dir: &std::path::Path, out: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_source(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
        }
    }
}

/// Every `&&`-joined `cargo test -p <crate> ... <path>::<name>` command names a `fn <name>(`
/// in that crate's `src/` or `tests/`; the first one that does not is returned.
fn missing_named_test(test: &str) -> Option<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let commands: Vec<&str> = test.split("&&").map(str::trim).collect();
    if commands.iter().any(|c| c.is_empty()) {
        return Some(format!("an empty command in {test:?}"));
    }
    commands.into_iter().find_map(|cmd| {
        let words: Vec<&str> = cmd.split_whitespace().collect();
        let krate = words
            .iter()
            .position(|w| *w == "-p")
            .and_then(|i| words.get(i + 1))?;
        let name = words.last()?.rsplit("::").next()?;
        let mut source = String::new();
        for dir in ["src", "tests"] {
            rust_source(&root.join("crates").join(krate).join(dir), &mut source);
        }
        (!source.contains(&format!("fn {name}("))).then(|| format!("{krate}: no fn {name}"))
    })
}

/// The hostile case for an ENFORCED row owned by this crate: every refusal it provoked (an
/// empty list for a row whose bound overrides the value instead of refusing it, after the case
/// asserted the override). `None` for an id this test has no case for.
#[allow(clippy::too_many_lines)] // one arm per contract row; the dispatch IS the table
fn hostile_case(id: &str) -> Option<Vec<ArtifactError>> {
    let b = packed();
    let errors = match id {
        "file_length" => {
            let n = b.len() as u64;
            vec![
                read_decide_apr_bytes_bounded(Untouchable, Some(MAX_ARTIFACT_BYTES + 1))
                    .expect_err("declared length over the cap"),
                load_verified_within(&b, &ArtifactLimits::tiny(n - 1, PROBE_MAX_ROW_TOKENS))
                    .expect_err("in-memory bytes over the cap"),
            ]
        }
        "header_version" => {
            let mut f = b.clone();
            forge_header(&mut f, |h| h.version = (3, 0));
            vec![refuse(&f)]
        }
        "metadata_size" => {
            let mut f = b.clone();
            forge_header(&mut f, |h| h.metadata_size = MAX_METADATA_BYTES + 1);
            vec![refuse(&f)]
        }
        "tensor_count" => {
            let mut f = b.clone();
            forge_header(&mut f, |h| h.tensor_count = MAX_TENSOR_COUNT + 1);
            vec![refuse(&f)]
        }
        "index_extent" => {
            let h = header(&b);
            let extent = h.data_offset - h.tensor_index_offset;
            let declared = u32::try_from(extent / MIN_INDEX_ENTRY_BYTES + 1).expect("fits u32");
            let (mut small, mut past) = (b.clone(), b.clone());
            forge_header(&mut small, |h| h.tensor_count = declared);
            let len = b.len() as u64;
            forge_header(&mut past, |h| h.data_offset = len + 1);
            vec![refuse(&small), refuse(&past)]
        }
        "duplicate_names_ladder" => {
            let r = AprV2ReaderRef::from_bytes(&b).expect("open the packed artifact");
            let mut names: Vec<&str> = r.tensor_index().iter().map(|e| e.name.as_str()).collect();
            assert_eq!(
                super::first_repeated_tensor_name(names.iter().copied()),
                None,
                "the packed index is unique"
            );
            names.push(names[0]);
            let name = super::first_repeated_tensor_name(names.iter().copied())
                .expect("the appended repeat is found");
            // reachable_via_load: false. A real duplicate is refused at rung 3 by the reader;
            // if that ever stops, this row must load its hostile artifact instead.
            let first = names[0].to_string();
            let dup = repack(&b, |_, t| {
                let copy = t.iter().find(|t| t.0 == first).cloned().expect("tensor");
                t.push(copy);
            });
            assert_eq!(
                refuse(&dup).rung(),
                "3 manifest",
                "duplicate_names_reader no longer refuses first: set reachable_via_load true and load the artifact here"
            );
            vec![ArtifactError::DuplicateTensor {
                name: name.to_string(),
            }]
        }
        "num_hidden_layers" => {
            let over = aprender::models::modernbert::MAX_NUM_HIDDEN_LAYERS + 1;
            let f = with_blob_json(&b, super::ENCODER_CONFIG_BLOB, |v| {
                v.as_object_mut().expect("config").remove("layer_types");
                v["num_hidden_layers"] = Value::from(over);
            });
            let e = refuse(&f);
            assert!(e.to_string().contains(&(over - 1).to_string()), "{e}");
            vec![e]
        }
        "head_layers" => {
            let over = crate::laya::MAX_HEAD_LAYERS + 1;
            let f = with_blob_json(&b, super::AGENT_CONFIG_BLOB, |v| {
                v["head_layers"] = Value::from(over);
            });
            let e = refuse(&f);
            assert!(e.to_string().contains(&(over - 1).to_string()), "{e}");
            vec![e]
        }
        "activation_and_rope" => {
            let relu = with_blob_json(&b, super::ENCODER_CONFIG_BLOB, |v| {
                v["hidden_activation"] = Value::from("relu");
            });
            let scaled = with_blob_json(&b, super::ENCODER_CONFIG_BLOB, |v| {
                v["rope_scaling"] = serde_json::from_str(r#"{"type": "linear", "factor": 2.0}"#)
                    .expect("rope_scaling");
            });
            vec![refuse(&relu), refuse(&scaled)]
        }
        "tensor_dtype" => {
            let f = repack(&b, |_, t| {
                let e = t
                    .iter_mut()
                    .find(|t| t.0 == "scorer.1.bias")
                    .expect("scorer.1.bias");
                e.1 = TensorDType::F32;
                e.3 = vec![0; e.3.len() * 2];
            });
            vec![refuse(&f)]
        }
        "tensor_entry_size" => {
            let f = repack(&b, |_, t| {
                let e = t
                    .iter_mut()
                    .find(|t| t.0 == "type_emb.weight")
                    .expect("type_emb");
                e.2 = vec![3, 31];
            });
            vec![refuse(&f)]
        }
        "tensor_data_range" => {
            // Point the tokenizer blob's index entry past the end of the file (the header CRC
            // covers only the header, so the index can be edited in place).
            let mut f = b.clone();
            let h = header(&f);
            let (lo, hi) = (
                usize::try_from(h.tensor_index_offset).expect("offset"),
                usize::try_from(h.data_offset).expect("offset"),
            );
            let name = super::TOKENIZER_BLOB.as_bytes();
            let mut pattern = u16::try_from(name.len())
                .expect("short name")
                .to_le_bytes()
                .to_vec();
            pattern.extend_from_slice(name);
            let at = lo
                + f[lo..hi]
                    .windows(pattern.len())
                    .position(|w| w == pattern)
                    .expect("the tokenizer blob's index entry");
            let ndim_at = at + pattern.len() + 1;
            let offset_at = ndim_at + 1 + 8 * usize::from(f[ndim_at]);
            let len = f.len() as u64;
            f[offset_at..offset_at + 8].copy_from_slice(&len.to_le_bytes());
            vec![refuse(&f)]
        }
        "tensor_name_set" => vec![
            refuse(&repack(&b, |_, t| t.retain(|t| t.0 != "scorer.3.bias"))),
            refuse(&repack(&b, |_, t| {
                t.push(("rogue.weight".into(), TensorDType::F16, vec![1], vec![0, 0]));
            })),
        ],
        "non_finite" => vec![refuse(&edit_tensor(&b, "scorer.1.weight", |d| {
            d[..2].copy_from_slice(&0x7E00u16.to_le_bytes());
        }))],
        "criteria_count" => {
            let n = crate::task::MAX_CRITERIA + 1;
            let criteria: Vec<String> = (0..n).map(|i| format!(r#""c{i}":null"#)).collect();
            let task = format!(
                r#"{{"type":"choice","instructions":"q","criteria":{{{}}}}}"#,
                criteria.join(",")
            );
            vec![refuse(&swap_blob_sized(
                &b,
                super::TASK_BLOB,
                task.as_bytes(),
            ))]
        }
        "probe_row_tokens" => {
            let forged = over_budget_probe_artifact();
            crate::laya::FORWARD_ROWS.with(|n| n.set(0));
            let e = refuse(&forged);
            assert_eq!(
                crate::laya::FORWARD_ROWS.with(std::cell::Cell::get),
                0,
                "refused before any forward"
            );
            vec![e]
        }
        "tokenizer_truncation_padding" => {
            let mut doc = blob_json(&b, super::TOKENIZER_BLOB);
            doc["truncation"] = serde_json::from_str(
                r#"{"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0}"#,
            )
            .expect("truncation block");
            doc["padding"] = serde_json::from_str(
                r#"{"strategy": {"Fixed": 40}, "direction": "Right", "pad_to_multiple_of": null,
                    "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}"#,
            )
            .expect("padding block");
            let clean = load_verified(&b).expect("the packed artifact loads");
            let forged = load_verified(&with_tokenizer(&b, &doc))
                .expect("a tokenizer declaring truncation / padding loads, its blocks disabled");
            let texts = vec!["a short text".to_string(), "word ".repeat(200)];
            let want = clean.classify(&texts).expect("clean classify");
            let got = forged.classify(&texts).expect("forged classify");
            for (w, g) in want.iter().zip(&got) {
                assert_eq!(
                    (w.tokens, w.truncated, &w.probabilities),
                    (g.tokens, g.truncated, &g.probabilities),
                    "the tokenizer file's blocks changed a served decision"
                );
            }
            Vec::new()
        }
        "inspect_read" => {
            let src = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/pack_laya.rs"),
            )
            .expect("read pack_laya.rs");
            let body = src
                .split("fn cmd_inspect")
                .nth(1)
                .and_then(|rest| rest.split("\nfn ").next())
                .expect("cmd_inspect");
            assert!(
                body.contains("read_decide_apr_bytes_bounded(")
                    && body.contains("inspect_manifest("),
                "inspect reads through the bounded rung-1 reader, then rungs 1-4"
            );
            assert!(
                !src.contains("fs::read("),
                "no unbounded whole-file read in pack_laya"
            );
            vec![
                read_decide_apr_bytes_bounded(Untouchable, Some(MAX_ARTIFACT_BYTES + 1))
                    .expect_err("inspect's reader refuses an over-cap file unread"),
            ]
        }
        _ => return None,
    };
    Some(errors)
}

/// The documented behaviour of an ACCEPTED row owned by this crate, asserted; `false` for an
/// id this test has no case for.
fn accepted_case(id: &str) -> bool {
    let b = packed();
    match id {
        "agent_max_len" => {
            // A max_len far past Laya's loads: the replay rows are bounded by the probe budget,
            // and a served row is as long as its text (the request half bounds it).
            let d = load_verified(&with_agent_field(&b, "max_len", Value::from(1u64 << 40)))
                .expect("a coherent artifact with max_len 2^40 loads");
            let long = d.classify(&["word ".repeat(200)]).expect("classify");
            assert!(
                long[0].tokens > 64 && !long[0].truncated,
                "the row is no longer cut at Laya's 64"
            );
        }
        "agent_head_max_len" => {
            load_verified(&with_agent_field(
                &b,
                "head_max_len",
                Value::from(1u64 << 40),
            ))
            .expect("a coherent artifact with head_max_len 2^40 loads");
        }
        "encoder_row_length" => {
            let laya = crate::test_support::load_laya(
                &crate::test_support::fixture_apr(),
                crate::test_support::fixture_task(),
            )
            .expect("the fixture loads");
            let builder = laya.builder();
            let row = builder
                .build(
                    &"word ".repeat(200),
                    "choice",
                    "q",
                    &["a".into(), "b".into()],
                )
                .expect("row");
            assert_eq!(
                row.tokens,
                builder.max_len(),
                "every built row is cut at max_len"
            );
            assert!(row.truncated);
        }
        "config_blob_parse" => {
            // A 200-deep value (past serde_json's recursion limit of 128) in the encoder config:
            // under a key the config IGNORES it is skipped iteratively (no recursion, so no
            // stack to overflow) and the artifact loads; inside a field parsed as a JSON value
            // (rope_scaling) it is a typed ConfigBlob refusal naming the recursion limit.
            let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
            let nest = |key: &str| -> Vec<u8> {
                let f = with_blob_json(&b, super::ENCODER_CONFIG_BLOB, |v| {
                    v[key] = Value::from("SEED");
                });
                let text = String::from_utf8(
                    blob_json(&f, super::ENCODER_CONFIG_BLOB)
                        .to_string()
                        .into_bytes(),
                )
                .expect("utf8")
                .replace(r#""SEED""#, &deep);
                swap_blob_sized(&b, super::ENCODER_CONFIG_BLOB, text.as_bytes())
            };
            load_verified(&nest("x_ignored")).expect(
                "an ignored deep value is skipped without recursion and the artifact loads",
            );
            let e = refuse(&nest("rope_scaling"));
            assert!(
                matches!(&e, ArtifactError::ConfigBlob { reason, .. } if reason.contains("recursion limit")),
                "nesting past serde_json's limit is an error, not a stack overflow: {e}"
            );
        }
        "tokenizer_pipeline" => {
            let gate = crate::test_support::contract_yaml("laya-finetune-gate-v1.yaml");
            let pin = gate["base"]["tokenizer_json_sha256"]
                .as_str()
                .unwrap_or_default();
            assert!(
                pin.len() == 64 && pin.bytes().all(|c| c.is_ascii_hexdigit()),
                "the gate contract pins the base tokenizer"
            );
            let table = bindings_table();
            assert!(
                table["/inputs_sha256/tokenizer_json"]
                    .sources
                    .iter()
                    .any(|s| s == "sha256:tokenizer.blob"),
                "rung 4 binds inputs_sha256.tokenizer_json to the tokenizer blob digest"
            );
        }
        "run_dir_files" => {
            let e = write_decide_apr_within(
                &inputs(),
                &ArtifactLimits::tiny(1000, PROBE_MAX_ROW_TOKENS),
            )
            .expect_err("a packed product over the cap");
            assert!(
                matches!(e, ArtifactError::ArtifactTooLarge { what: "packed", .. }),
                "{e}"
            );
        }
        _ => return false,
    }
    true
}

/// decide-apr-v1 `untrusted_input_bounds` (CLASS B, artifact half): every row owned by this
/// crate is dispatched by id — an enforced row to a hostile case built from the packed tiny
/// artifact, whose every refusal must be a variant the row names at the rung it names; an
/// accepted row to its documented behaviour. Rows owned by another crate must name a test
/// that exists there. An unknown id, an owned row without a case, a duplicate id or a missing
/// field fails.
#[test]
fn artifact_bounds_table_is_swept() {
    let c = crate::test_support::contract_yaml("decide-apr-v1.yaml");
    let table = &c["untrusted_input_bounds"];
    assert!(
        table["see_also"]
            .as_str()
            .is_some_and(|s| s.contains("decide-tool-boundary-v1 untrusted_input_bounds")),
        "see_also points at the request half"
    );
    let rows = table["rows"]
        .as_sequence()
        .expect("untrusted_input_bounds.rows");
    let (mut swept, mut accepted, mut external) = (0usize, 0usize, 0usize);
    let mut ids = BTreeSet::new();
    let mut failures = Vec::new();
    for row in rows {
        let id = row_text(row, "id");
        assert!(ids.insert(id.clone()), "duplicate row {id}");
        for key in [
            "bound",
            "checked_by",
            "owner_crate",
            "test",
            "disposition",
            "rung",
        ] {
            if row_text(row, key).is_empty() {
                failures.push(format!("{id}: no {key}"));
            }
        }
        if let Some(missing) = missing_named_test(&row_text(row, "test")) {
            failures.push(format!("{id}: {missing}"));
        }
        if row_text(row, "owner_crate") != "aprender-decide" {
            external += 1;
            continue;
        }
        match row_text(row, "disposition").as_str() {
            "accepted" => {
                if row_text(row, "reason").is_empty() {
                    failures.push(format!("{id}: accepted without a reason"));
                }
                if !accepted_case(&id) {
                    failures.push(format!(
                        "{id}: accepted row without a documented-behaviour case"
                    ));
                }
                accepted += 1;
            }
            "enforced" => {
                let Some(errors) = hostile_case(&id) else {
                    failures.push(format!("{id}: owned row without a hostile case"));
                    continue;
                };
                let refusals: Vec<String> = match &row["refusal"] {
                    serde_yaml::Value::Sequence(v) => v
                        .iter()
                        .filter_map(|r| r.as_str().map(str::to_string))
                        .collect(),
                    _ => Vec::new(),
                };
                if row_text(row, "refusal") == "none" {
                    if !errors.is_empty() {
                        failures.push(format!("{id}: an overriding bound refused: {errors:?}"));
                    }
                } else if errors.is_empty() || refusals.is_empty() {
                    failures.push(format!("{id}: no refusal provoked or named"));
                }
                let rung = row["rung"].as_u64();
                for e in &errors {
                    let debug = format!("{e:?}");
                    if !refusals.iter().any(|r| debug.starts_with(r.as_str())) {
                        failures.push(format!(
                            "{id}: refused as {debug}, the row names {refusals:?}"
                        ));
                    }
                    if let Some(rung) = rung {
                        if rung_number(e) != rung {
                            failures.push(format!(
                                "{id}: refused at rung {} ({e}), the row names {rung}",
                                e.rung()
                            ));
                        }
                    }
                }
                swept += 1;
            }
            other => failures.push(format!("{id}: unknown disposition {other}")),
        }
    }
    println!("ARTIFACT BOUNDS swept={swept} accepted={accepted} external={external}");
    assert!(
        failures.is_empty(),
        "untrusted_input_bounds:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Pack-time probe refusals
// ---------------------------------------------------------------------------

/// A probe row over the cap is refused at pack (the cap shrunk to 20 so the 21-token
/// probe 0 exceeds it; the contracted value is asserted in `contract_mirror`).
#[test]
fn probe_row_over_budget() {
    let e = write_decide_apr_within(&inputs(), &ArtifactLimits::tiny(MAX_ARTIFACT_BYTES, 20))
        .expect_err("probe row over the cap");
    assert_eq!(
        e,
        ArtifactError::ProbeRowOverBudget {
            index: 0,
            tokens: 21,
            cap: 20,
        }
    );
}

/// A probes.json expectation shifted by 1e-3 is refused at pack.
#[test]
fn probe_disagrees_with_oracle() {
    let mut i = inputs();
    let h = &mut i.probes[0].probabilities_f32_hex[0];
    let bits = u32::from_str_radix(h, 16).expect("hex parses");
    *h = format!("{:08x}", (f32::from_bits(bits) + 1e-3).to_bits());
    let e = write_decide_apr_within(&i, &ArtifactLimits::CONTRACTED)
        .expect_err("probe disagrees with the oracle");
    assert_eq!(
        e,
        ArtifactError::ProbeDisagreesWithOracle {
            index: 0,
            component: "probabilities",
        }
    );
}
