//! `apr finetune --recipe` (CRUX-001, PMAT-5316; contracts/apr-recipe-v1.yaml).

use super::*;

const VALID: &str = "\
recipe: apr-recipe-v1
name: qwen-lora-demo
base: models/base.apr
method: qlora
data:
  train: data/train.jsonl
eval:
  heldout: data/heldout.jsonl
seed: 42
training:
  rank: 8
  epochs: 1
";

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-ft-recipe-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn write_recipe(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("r.yaml");
    std::fs::write(&path, body).expect("write recipe");
    path
}

fn flags(plan_only: bool) -> RecipeRunFlags<'static> {
    RecipeRunFlags {
        vram_gb: 16.0,
        plan_only,
        output_path: None,
        model_size: Some("1B"),
        checkpoint_format: "apr,safetensors",
        gpus: None,
        gpu_backend: "auto",
        wait_gpu: 0,
        json_output: true,
        experimental_mps: false,
        gpu_share: 50,
    }
}

/// The positive control: the same recipe WITH its eval block plans cleanly.
#[test]
fn valid_recipe_plans() {
    let dir = scratch("plan");
    let path = write_recipe(&dir, VALID);
    run_from_recipe(&path, &flags(true)).expect("a valid recipe must plan");
    std::fs::remove_dir_all(&dir).ok();
}
