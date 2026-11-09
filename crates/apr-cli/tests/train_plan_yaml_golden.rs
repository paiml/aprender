//! `apr train plan --format yaml` text: the bytes serde_yaml_ng wrote for a plan manifest,
//! pinned so the swap to serde-saphyr (#5209) is judged on a measurement.
//!
//! The manifest is a `serde_json::Value` shaped as `pretrain_plan_manifest` builds it (keys
//! inserted in sorted order), the same input the old `serde_yaml::to_string` took.

use serde_json::json;

const SERDE_YAML_NG_GOLDEN: &str = "\
config: cfg.yaml
data:
  batch_size: 8
  seq_len: 1024
  train: data/train.parquet
model:
  mode: Pretrain
  path: model/
optimizer:
  lr: 0.0003
  name: adamw
task: pretrain
training:
  epochs: 3
  gradient_accumulation: 1
  mixed_precision: bf16
  mode: Causal
  output_dir: out
  warmup_steps: 100
verdict: ready
";

#[test]
fn plan_yaml_text_is_what_serde_yaml_ng_wrote() {
    // Keys are inserted in sorted order, so the text does not depend on whether a
    // transitive dependency switches serde_json's `preserve_order` on in this build.
    let plan = json!({
        "config": "cfg.yaml",
        "data": {"batch_size": 8, "seq_len": 1024, "train": "data/train.parquet"},
        "model": {"mode": "Pretrain", "path": "model/"},
        "optimizer": {"lr": 0.0003, "name": "adamw"},
        "task": "pretrain",
        "training": {
            "epochs": 3,
            "gradient_accumulation": 1,
            "mixed_precision": "bf16",
            "mode": "Causal",
            "output_dir": "out",
            "warmup_steps": 100,
        },
        "verdict": "ready",
    });
    let rendered = serde_saphyr::to_string(&plan).expect("plan renders as YAML");
    assert_eq!(rendered, SERDE_YAML_NG_GOLDEN);
}
