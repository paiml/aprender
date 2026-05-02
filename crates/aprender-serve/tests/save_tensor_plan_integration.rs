//! Integration tests for `SaveTensorPlan` end-to-end through the public
//! `save_tensor*` I/O helpers.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! ## Why this complements `save_tensor_integration.rs`
//!
//! `save_tensor_integration.rs` exercises the writer (`write_tensor_file`)
//! and path builder (`output_path`) in isolation. This file exercises the
//! plan-builder ([`SaveTensorPlan`] from PR-B) driving those helpers as
//! they will be driven from inside the forward pass once PR-C lands the
//! `forward_traced` wiring.
//!
//! Concretely: we build a plan from CLI argument strings (the same form
//! the future dispatch site will pass), iterate a fixed sequence of
//! `(stage, layer, data)` tuples that mimic what the forward pass produces,
//! and assert:
//!
//! 1. Files appear ONLY for `(stage, layer)` pairs where the plan returns
//!    `should_save == true`.
//! 2. Each file's path matches `plan.stage_path(stage, layer)` exactly.
//! 3. Each file's bytes match what `write_tensor_file` would emit for the
//!    given layer + values.
//! 4. Byte-for-byte determinism: the same plan + same data → identical
//!    files across two runs.
//!
//! ## Discharge map
//!
//! | Falsifier | Discharge level | Test |
//! |-----------|-----------------|------|
//! | FALSIFY-APR-TRACE-SAVE-005 (multi-stage in one run) | partial | `plan_three_stages_layer_zero_writes_three_files` |
//! | FALSIFY-APR-TRACE-SAVE-LAYER-FILTER (range honoured) | partial | `plan_layer_range_filter_excludes_out_of_range` |
//! | FALSIFY-APR-TRACE-SAVE-WHOLE-MODEL-PATH (no layer-N segment) | partial | `plan_whole_model_stage_writes_to_root_not_layer_dir` |
//! | FALSIFY-APR-TRACE-SAVE-002 (determinism) | partial | `plan_byte_determinism_across_two_runs` |
//!
//! "Partial" because full discharge requires PR-C/D's live CLI integration
//! invoking the plan from inside `forward_traced`. These tests pin the
//! plan ↔ writer contract so the wiring PR has a stable target.

use std::path::Path;

use realizar::inference_trace::save_tensor::{self, MAGIC};
use realizar::inference_trace::save_tensor_paths::ensure_layer_dir;
use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;
use realizar::inference_trace::save_tensor_stage::SaveTensorStage;

/// Helper: drive a plan against a fixed `(stage, layer, data)` sequence,
/// writing only the entries the plan selects. Returns the list of paths
/// the plan caused to be written.
///
/// Mirrors the future dispatch flow:
///   for (stage, layer, data) in forward_pass_stages:
///       if plan.should_save(stage, layer):
///           ensure_layer_dir(plan.output_dir, layer)
///           write_tensor_file(plan.stage_path(stage, layer), layer, data)
fn execute_plan_against_sequence(
    plan: &SaveTensorPlan,
    sequence: &[(SaveTensorStage, u32, Vec<f32>)],
) -> Vec<std::path::PathBuf> {
    use std::io::Write;
    let mut written = Vec::new();
    for (stage, layer, data) in sequence {
        if !plan.should_save(*stage, *layer) {
            continue;
        }
        // Whole-model stages route to plan.output_dir directly; per-layer
        // stages route to <output_dir>/layer-<N>/. plan.stage_path() encodes
        // this; ensure_layer_dir mirrors the same fork.
        let layer_for_dir = if stage.is_per_layer() {
            *layer
        } else {
            save_tensor::WHOLE_MODEL_LAYER
        };
        ensure_layer_dir(&plan.output_dir, layer_for_dir).expect("ensure_layer_dir");
        let path = plan.stage_path(*stage, *layer);
        let file = std::fs::File::create(&path).expect("create file");
        let mut writer = std::io::BufWriter::new(file);
        // For whole-model stages, the layer field stored in the file header
        // is the WHOLE_MODEL_LAYER sentinel — that's what the writer expects
        // and what readers will use to round-trip.
        let layer_for_header = if stage.is_per_layer() {
            *layer
        } else {
            save_tensor::WHOLE_MODEL_LAYER
        };
        save_tensor::write_tensor_file(&mut writer, layer_for_header, data).expect("write");
        writer.flush().expect("flush");
        written.push(path);
    }
    written
}

#[test]
fn plan_three_stages_layer_zero_writes_three_files() {
    // Realistic SHIP-007 layer-0 capture: three stages on layer 0 only.
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan = SaveTensorPlan::from_cli(
        "embedding,qkv_matmul,attention",
        "0..1",
        tmp.path().to_path_buf(),
    )
    .expect("plan parses");

    // Mimic forward pass producing all 3 stages on layer 0, plus an
    // unrelated stage on layer 0 that the user did NOT select.
    let sequence = vec![
        (SaveTensorStage::Embedding, 0u32, vec![1.0_f32, 2.0, 3.0]),
        (
            SaveTensorStage::QkvMatmul,
            0,
            vec![10.0_f32, 20.0, 30.0, 40.0],
        ),
        (SaveTensorStage::Attention, 0, vec![100.0_f32, 200.0, 300.0]),
        // NOT in plan — must be skipped.
        (SaveTensorStage::FfnGate, 0, vec![999.0_f32]),
    ];

    let written = execute_plan_against_sequence(&plan, &sequence);
    assert_eq!(
        written.len(),
        3,
        "plan should write exactly 3 files (one per selected stage)"
    );

    // Every written path is what plan.stage_path() predicted:
    assert_eq!(written[0], plan.stage_path(SaveTensorStage::Embedding, 0));
    assert_eq!(written[1], plan.stage_path(SaveTensorStage::QkvMatmul, 0));
    assert_eq!(written[2], plan.stage_path(SaveTensorStage::Attention, 0));

    // Every written file exists and has the expected MAGIC bytes.
    for path in &written {
        let bytes = std::fs::read(path).expect("read");
        assert_eq!(
            &bytes[..4],
            MAGIC,
            "header MAGIC must be APRT for {}",
            path.display()
        );
    }

    // The unselected stage's file MUST NOT exist.
    let unselected = plan.stage_path(SaveTensorStage::FfnGate, 0);
    assert!(
        !unselected.exists(),
        "FfnGate file must NOT be written when the plan does not select it"
    );
}

#[test]
fn plan_layer_range_filter_excludes_out_of_range() {
    // Plan saves layer 0..3 only. Stages produced for layers 0,1,2 must
    // appear; layer 3,4 must not.
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan = SaveTensorPlan::from_cli("ffn_gate", "0..3", tmp.path().to_path_buf())
        .expect("plan parses");

    let sequence: Vec<_> = (0..5u32)
        .map(|l| (SaveTensorStage::FfnGate, l, vec![l as f32, (l * 2) as f32]))
        .collect();

    let written = execute_plan_against_sequence(&plan, &sequence);
    assert_eq!(
        written.len(),
        3,
        "layer range 0..3 must write 3 files (layers 0,1,2 — END exclusive)"
    );

    // Layer 0,1,2 files exist; layer 3,4 files do not.
    for layer_in in [0, 1, 2] {
        let p = plan.stage_path(SaveTensorStage::FfnGate, layer_in);
        assert!(p.exists(), "layer-{layer_in} should be written");
    }
    for layer_out in [3, 4] {
        let p = plan.stage_path(SaveTensorStage::FfnGate, layer_out);
        assert!(
            !p.exists(),
            "layer-{layer_out} must be excluded by range 0..3"
        );
    }
}

#[test]
fn plan_whole_model_stage_writes_to_root_not_layer_dir() {
    // Whole-model stages (final_norm, lm_head) bypass the layer-N
    // directory segment — the plan.stage_path() fork pins this.
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan =
        SaveTensorPlan::from_cli("lm_head", "0..1", tmp.path().to_path_buf()).expect("plan parses");

    // Even if the dispatch site passes layer=0, lm_head should ignore it.
    let sequence = vec![(SaveTensorStage::LmHead, 0u32, vec![0.1_f32, 0.2, 0.3])];
    let written = execute_plan_against_sequence(&plan, &sequence);
    assert_eq!(written.len(), 1);

    // Path must be <output_dir>/lm_head.bin (no layer-0/ segment).
    let expected = tmp.path().join("lm_head.bin");
    assert_eq!(written[0], expected);
    assert!(written[0].exists());

    // The "layer-0/lm_head.bin" mistake-path must NOT exist.
    let mistake = tmp.path().join("layer-0").join("lm_head.bin");
    assert!(
        !mistake.exists(),
        "lm_head must not land in layer-0/ subdirectory"
    );
}

#[test]
fn plan_unselected_stage_produces_no_file() {
    // Even if forward pass produces a stage's data, the plan acts as a
    // strict allow-list: should_save == false → zero I/O.
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan = SaveTensorPlan::from_cli("embedding", "0..1", tmp.path().to_path_buf())
        .expect("plan parses");

    let sequence = vec![
        (SaveTensorStage::Embedding, 0u32, vec![1.0_f32]),
        (SaveTensorStage::Attention, 0, vec![2.0_f32]),
        (SaveTensorStage::FfnGate, 0, vec![3.0_f32]),
        (SaveTensorStage::LmHead, 0, vec![4.0_f32]),
    ];
    let written = execute_plan_against_sequence(&plan, &sequence);
    assert_eq!(
        written.len(),
        1,
        "only embedding should be written; 3 unselected stages must produce no I/O"
    );

    // Confirm by listing the tmp dir.
    let entries: Vec<_> = walk_files(tmp.path()).collect();
    assert_eq!(
        entries.len(),
        1,
        "filesystem state: exactly one file under output_dir, got {entries:?}"
    );
}

#[test]
fn plan_byte_determinism_across_two_runs() {
    // The same plan + same data must produce byte-identical files
    // (FALSIFY-APR-TRACE-SAVE-002 at the plan integration boundary).
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir_a = tmp.path().join("run_a");
    let dir_b = tmp.path().join("run_b");

    let plan_a = SaveTensorPlan::from_cli("embedding,ffn_gate", "0..2", dir_a.clone()).unwrap();
    let plan_b = SaveTensorPlan::from_cli("embedding,ffn_gate", "0..2", dir_b.clone()).unwrap();

    let sequence: Vec<_> = (0..2u32)
        .flat_map(|l| {
            vec![
                (
                    SaveTensorStage::Embedding,
                    l,
                    (0..32).map(|i| (i + l as i32) as f32 * 0.5).collect(),
                ),
                (
                    SaveTensorStage::FfnGate,
                    l,
                    (0..64).map(|i| (i - 32 + l as i32) as f32 * 0.25).collect(),
                ),
            ]
        })
        .collect();

    let written_a = execute_plan_against_sequence(&plan_a, &sequence);
    let written_b = execute_plan_against_sequence(&plan_b, &sequence);
    assert_eq!(
        written_a.len(),
        written_b.len(),
        "same plan → same file count"
    );
    for (path_a, path_b) in written_a.iter().zip(written_b.iter()) {
        let bytes_a = std::fs::read(path_a).expect("read A");
        let bytes_b = std::fs::read(path_b).expect("read B");
        assert_eq!(
            bytes_a,
            bytes_b,
            "FALSIFIED determinism: plan {} vs {} produced different bytes",
            path_a.display(),
            path_b.display(),
        );
    }
}

#[test]
fn plan_all_keyword_writes_18_per_layer_files_for_one_layer() {
    // The `all` keyword expands to 18 stages. With layer range `0..1` and
    // a single layer, we expect 16 per-layer files + 2 whole-model files.
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan =
        SaveTensorPlan::from_cli("all", "0..1", tmp.path().to_path_buf()).expect("plan parses");
    assert_eq!(plan.stages.len(), 18);

    // Build a sequence with one entry per stage, all on layer 0.
    let sequence: Vec<_> = SaveTensorStage::ALL
        .iter()
        .map(|s| (*s, 0u32, vec![1.0_f32, 2.0, 3.0]))
        .collect();

    let written = execute_plan_against_sequence(&plan, &sequence);
    assert_eq!(
        written.len(),
        18,
        "`all` + layer 0..1 should produce all 18 stage files"
    );

    // Per-layer stages live under layer-0/; whole-model stages live at root.
    let per_layer_count = SaveTensorStage::ALL
        .iter()
        .filter(|s| s.is_per_layer())
        .count();
    let whole_model_count = SaveTensorStage::ALL
        .iter()
        .filter(|s| !s.is_per_layer())
        .count();
    assert_eq!(per_layer_count + whole_model_count, 18);

    // Sanity-check: layer-0/ contains the per-layer files; root contains
    // the whole-model files.
    for stage in SaveTensorStage::ALL {
        let p = plan.stage_path(stage, 0);
        assert!(
            p.exists(),
            "{} file must exist at {}",
            stage.canonical_name(),
            p.display()
        );
        let in_layer_dir = p
            .parent()
            .and_then(|d| d.file_name())
            .and_then(|n| n.to_str())
            .map(|s| s == "layer-0")
            .unwrap_or(false);
        assert_eq!(
            in_layer_dir,
            stage.is_per_layer(),
            "{}: per-layer expected layer-0/ dir, whole-model expected root",
            stage.canonical_name()
        );
    }
}

/// Small helper: recursively walk `dir` and yield file paths.
fn walk_files(dir: &Path) -> impl Iterator<Item = std::path::PathBuf> {
    fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    collect(&p, out);
                } else {
                    out.push(p);
                }
            }
        }
    }
    let mut v = Vec::new();
    collect(dir, &mut v);
    v.into_iter()
}
