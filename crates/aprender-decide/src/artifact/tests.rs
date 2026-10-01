//! The decide-apr-v1 tracer on the tiny fixture: pack -> bytes -> ladder -> Decider ->
//! classify, with the identity the classify response will carry.

use super::{artifact_sha256_hex, ProbeRecord};
use crate::digest::sha256_hex;
use crate::pack::{pack_run_dir, ProbesFile};
use crate::test_support::{f32_list, fixture_dir, max_abs, oracle, read, tolerance, within};
use crate::{Decider, Task};

/// The committed tiny run dir (plan 08-02) packed with the shipped door.
pub(crate) fn pack_tiny() -> Vec<u8> {
    let dir = fixture_dir();
    pack_run_dir(&dir, &dir.join("data")).expect("the tiny run dir packs")
}

/// The run dir's `probes.json`, parsed with the same typed schema the packer uses.
pub(crate) fn fixture_probes() -> Vec<ProbeRecord> {
    serde_json::from_slice::<ProbesFile>(&read("probes.json"))
        .expect("probes.json parses")
        .probes
}

/// FALSIFY-DECIDE-APR-004/-005 on the tracer path: the packed tiny artifact loads only
/// through the ladder, classifies the oracle's task rows within laya-parity-v1
/// `probs_abs`, and reports the identity of the served bytes (D-11).
#[test]
fn tiny_roundtrip() {
    let bytes = pack_tiny();
    let decider = Decider::load_bytes(&bytes).expect("the ladder accepts the packed fixture");

    // Classification equals Laya's own oracle on the task rows.
    let bar = tolerance("probs_abs");
    let o = oracle();
    let rows: Vec<&serde_json::Value> = o["rows"]
        .as_array()
        .expect("oracle rows")
        .iter()
        .filter(|r| r["qid"] == "team")
        .collect();
    assert!(rows.len() >= 2, "the oracle has task rows");
    let texts: Vec<String> = rows
        .iter()
        .map(|r| r["state"].as_str().expect("state").to_string())
        .collect();
    let decisions = decider.classify(&texts).expect("classify");
    assert_eq!(decisions.len(), rows.len());
    let mut worst = 0.0f64;
    for (i, (r, d)) in rows.iter().zip(&decisions).enumerate() {
        let dp = max_abs(&d.probabilities, &f32_list(&r["probabilities"]));
        assert!(within(dp, bar), "task row {i}: probs max|d| {dp} > {bar}");
        let want = usize::try_from(r["argmax"].as_u64().expect("argmax")).expect("fits");
        assert_eq!(d.label_index, want, "task row {i}: argmax exact");
        worst = worst.max(dp);
    }

    // Identity (D-11): the whole-file sha256 and the recipe blob's sha256.
    let id = decider.identity();
    assert_eq!(id.artifact_sha256, artifact_sha256_hex(&bytes));
    assert_eq!(id.artifact_sha256, sha256_hex(&bytes));
    assert_eq!(id.artifact_sha256.len(), 64);
    assert_eq!(id.recipe_id, sha256_hex(&read("recipe.json")));
    assert_eq!(id.method, "laya");
    assert_eq!(id.base, "laya-tiny-synthetic@fixtures");

    // Labels in data/task.json document order (deliberately not sorted).
    let task = Task::from_slice(&read("data/task.json")).expect("data task parses");
    assert_eq!(decider.labels(), task.labels());
    assert_eq!(decider.labels(), ["shipping", "billing", "account"]);

    // The declared base and variant ride in the manifest (D-04).
    let m = decider.manifest();
    assert_eq!(m.base.checkpoint, "tiny-synthetic");
    assert_eq!(m.variant, "synthetic-fixture");
    assert_eq!(m.base, id.base_decl);

    // Probe expectations are the PYTHON values from probes.json, stored verbatim.
    assert_eq!(m.probes, fixture_probes());

    println!(
        "tiny_roundtrip: {} bytes, sha256 {}, {} task rows max|d| {worst:.3e} (bar {bar:e}), ARCH={}",
        bytes.len(),
        id.artifact_sha256,
        rows.len(),
        std::env::consts::ARCH
    );
}

/// The hashed door mints exactly the identity the plain door mints: the digest a caller
/// pinned IS the rung-8 identity (one sha256 pass), and a refusing rung still refuses.
#[test]
fn load_hashed_mints_the_same_identity() {
    let bytes = pack_tiny();
    let hashed = super::HashedArtifact::new(&bytes);
    assert_eq!(hashed.sha256(), artifact_sha256_hex(&bytes));
    assert!(std::ptr::eq(hashed.bytes(), bytes.as_slice()));
    let via_hash = Decider::load_hashed(&hashed).expect("the ladder accepts the packed fixture");
    let via_bytes = Decider::load_bytes(&bytes).expect("the ladder accepts the packed fixture");
    assert_eq!(via_hash.identity(), via_bytes.identity());
    assert_eq!(via_hash.identity().artifact_sha256, hashed.sha256());

    let mut corrupt = bytes.clone();
    corrupt[20] ^= 0x01; // inside the CRC-covered header
    let e = Decider::load_hashed(&super::HashedArtifact::new(&corrupt))
        .expect_err("a corrupt header is refused on the hashed door too");
    assert_eq!(e, super::ArtifactError::HeaderChecksum);
}

/// FALSIFY-DECIDE-APR-004: every checkpoint tensor's raw bytes (F16 weights and the
/// F32 temperature), dtype and shape, and every blob, round-trip byte-identical
/// against the run dir — nothing re-rounded through f32, nothing re-serialized.
#[test]
fn pack_unpack_closure() {
    use aprender::format::v2::AprV2ReaderRef;
    let bytes = pack_tiny();
    let r = AprV2ReaderRef::from_bytes(&bytes).expect("open packed");
    let checkpoint = crate::test_support::checkpoint_tensors();
    for (name, dtype, shape, raw) in &checkpoint {
        let e = r
            .get_tensor(name)
            .unwrap_or_else(|| panic!("{name} packed"));
        assert_eq!(&e.dtype, dtype, "{name}: dtype");
        assert_eq!(&e.shape, shape, "{name}: shape");
        assert_eq!(
            r.get_tensor_data(name),
            Some(raw.as_slice()),
            "{name}: bytes"
        );
    }
    let blobs = [
        (super::TOKENIZER_BLOB, "checkpoint/tokenizer/tokenizer.json"),
        (super::TASK_BLOB, "task.json"),
        (super::ENCODER_CONFIG_BLOB, "checkpoint/encoder/config.json"),
        (super::AGENT_CONFIG_BLOB, "checkpoint/rl_agent_config.json"),
        (super::RECIPE_BLOB, "recipe.json"),
        (super::GATE_REPORT_BLOB, "gate-report.json"),
    ];
    for (blob, file) in blobs {
        let e = r
            .get_tensor(blob)
            .unwrap_or_else(|| panic!("{blob} packed"));
        assert_eq!(e.dtype, aprender::format::v2::TensorDType::U8, "{blob}: U8");
        assert_eq!(
            r.get_tensor_data(blob),
            Some(read(file).as_slice()),
            "{blob} == {file}"
        );
    }
    assert_eq!(
        r.tensor_names().len(),
        checkpoint.len() + blobs.len(),
        "the index is exactly the checkpoint plus the six blobs"
    );
    let meta = r.metadata();
    assert_eq!(meta.model_type, super::MODEL_TYPE_TAG);
    assert_eq!(meta.custom.len(), 1, "exactly one custom key");
    assert!(meta.custom.contains_key(super::CUSTOM_METADATA_KEY));
    assert_eq!(meta.created_at, None, "no timestamp");
}

/// The file door: bounded read by metadata length, then the ladder; the identity is
/// the sha256 of the file's bytes.
#[test]
fn load_path_door() {
    let bytes = pack_tiny();
    let dir = std::env::temp_dir().join(format!("decide-0805-door-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("tiny.apr");
    std::fs::write(&path, &bytes).expect("write the artifact");
    let d = Decider::load_path(&path).expect("load_path");
    assert_eq!(d.identity().artifact_sha256, sha256_hex(&bytes));
    let missing = Decider::load_path(dir.join("absent.apr")).expect_err("absent file");
    assert!(
        matches!(missing, super::ArtifactError::Read { .. }),
        "{missing}"
    );
    std::fs::remove_dir_all(&dir).expect("clean up");
}

/// Every constant the artifact restates equals `contracts/decide-apr-v1.yaml`.
#[test]
fn contract_mirror() {
    use super::{
        ArtifactLimits, ARTIFACT_SCHEMA, ARTIFACT_SCHEMA_VERSION, BLOB_TENSORS,
        CUSTOM_METADATA_KEY, MAX_ARTIFACT_BYTES, MAX_METADATA_BYTES, MAX_TENSOR_COUNT, METHOD_LAYA,
        MIN_INDEX_ENTRY_BYTES, MODEL_TYPE_TAG, PROBE_INPUTS, PROBE_MAX_ROW_TOKENS,
        PROBE_PROBABILITIES_ABS, PROBE_TASK,
    };
    let c = crate::test_support::contract_yaml("decide-apr-v1.yaml");
    let k = &c["constants"];
    assert_eq!(k["max_artifact_bytes"].as_u64(), Some(MAX_ARTIFACT_BYTES));
    assert_eq!(
        k["max_tensor_count"].as_u64(),
        Some(u64::from(MAX_TENSOR_COUNT))
    );
    assert_eq!(
        k["max_metadata_bytes"].as_u64(),
        Some(u64::from(MAX_METADATA_BYTES))
    );
    assert_eq!(
        k["probe_max_row_tokens"].as_u64(),
        Some(PROBE_MAX_ROW_TOKENS as u64)
    );
    assert_eq!(
        k["probe_probabilities_abs"].as_f64(),
        Some(PROBE_PROBABILITIES_ABS)
    );
    assert_eq!(
        k["min_index_entry_bytes"].as_u64(),
        Some(MIN_INDEX_ENTRY_BYTES)
    );
    assert_eq!(
        k["schema_version"].as_u64(),
        Some(u64::from(ARTIFACT_SCHEMA_VERSION))
    );
    let limits = ArtifactLimits::CONTRACTED;
    assert_eq!(limits.max_artifact_bytes(), MAX_ARTIFACT_BYTES);
    assert_eq!(limits.probe_max_row_tokens(), PROBE_MAX_ROW_TOKENS);

    // Probe inputs verbatim, and the synthetic probe task field by field.
    let inputs: Vec<&str> = c["probe_policy"]["inputs"]
        .as_sequence()
        .expect("probe inputs")
        .iter()
        .map(|v| v.as_str().expect("probe input string"))
        .collect();
    assert_eq!(inputs, PROBE_INPUTS);
    let pt = &c["probe_policy"]["probe_task"];
    let task = Task::from_slice(PROBE_TASK.as_bytes()).expect("PROBE_TASK parses");
    assert_eq!(pt["type"].as_str(), Some("choice"));
    assert_eq!(pt["instructions"].as_str(), Some(task.instructions()));
    let criteria = pt["criteria"].as_mapping().expect("probe criteria");
    let names: Vec<&str> = criteria
        .iter()
        .map(|(key, v)| {
            assert!(v.is_null(), "probe criteria carry no description");
            key.as_str().expect("criterion name")
        })
        .collect();
    assert_eq!(names, task.labels(), "criteria in document order");
    assert!(task.criteria().iter().all(|c| c.description.is_none()));
    assert_eq!(pt["k"].as_u64(), Some(task.labels().len() as u64));

    // Blob set, model type, the single key, schema and method.
    let blobs: Vec<&str> = c["blob_tensors"]
        .as_sequence()
        .expect("blob_tensors")
        .iter()
        .map(|v| v.as_str().expect("blob name"))
        .collect();
    assert_eq!(blobs, BLOB_TENSORS);
    let m = &c["manifest"];
    assert_eq!(m["model_type"].as_str(), Some(MODEL_TYPE_TAG));
    assert!(m["storage"]
        .as_str()
        .expect("storage")
        .contains(&format!("custom metadata key `{CUSTOM_METADATA_KEY}`")));
    let fields = &m["fields"];
    assert!(fields["schema"]
        .as_str()
        .expect("schema")
        .contains(&format!("\"{ARTIFACT_SCHEMA}\"")));
    assert!(fields["method"]
        .as_str()
        .expect("method")
        .contains(&format!("\"{METHOD_LAYA}\"")));

    // The manifest's field SET equals the contract's.
    let mut want: Vec<&str> = fields
        .as_mapping()
        .expect("manifest.fields")
        .keys()
        .map(|key| key.as_str().expect("field name"))
        .collect();
    want.sort_unstable();
    let packed = super::inspect_manifest(&pack_tiny()).expect("manifest");
    let value = serde_json::to_value(&packed).expect("manifest to value");
    let mut got: Vec<&str> = value
        .as_object()
        .expect("manifest object")
        .keys()
        .map(String::as_str)
        .collect();
    got.sort_unstable();
    assert_eq!(
        got, want,
        "manifest fields == decide-apr-v1 manifest.fields"
    );
}
