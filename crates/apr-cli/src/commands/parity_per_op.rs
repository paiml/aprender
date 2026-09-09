//! `apr parity --per-op` (L0-1b, #2971, PP-066): the per-op CPU-vs-GPU table.
//!
//! For every position of the prompt, runs the CPU reference forward
//! (`forward_single_with_cache`, the same function the load-time gate compares
//! against) with the per-op tap armed, and the GPU-resident forward with the
//! executor's stage dump armed, both writing APRT files per (stage, layer).
//! The two trees are then compared file by file; the table is printed in
//! forward order and the FIRST op under the threshold is named. Exit 0 with
//! the table even when an op is red — the table IS the verdict — and the
//! load-time admission gate is bypassed INTERNALLY (never `SKIP_PARITY_GATE`).
use std::path::Path;

use crate::error::Result;

#[cfg(feature = "cuda")]
pub fn run(
    file: &Path,
    prompt: &str,
    out: Option<&Path>,
    threshold: f32,
    json: bool,
) -> Result<()> {
    use super::parity_per_op_table::{aggregate, first_divergence, render};
    use crate::error::CliError;
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };
    use realizar::inference_trace::gpu_stage_dump::per_op_tap;
    use realizar::inference_trace::gpu_stage_dump::GpuStageDumpConfig;
    use realizar::inference_trace::save_tensor_compose::read_stage_file;
    use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;

    // The graphed decode path replays a captured graph; host read-backs inside it are
    // impossible, so the instrument forces the non-graphed path. Printed: an override.
    std::env::set_var("SKIP_CUDA_GRAPH", "1");
    eprintln!("override: SKIP_CUDA_GRAPH=1 (per-op instrument runs the non-graphed GPU path; arms A0/A1 measured identical cosine)");

    let mapped = MappedGGUFModel::from_path(file)
        .map_err(|e| CliError::ValidationFailed(format!("GGUF load failed: {e}")))?;
    let tokens = mapped.model.encode(prompt).unwrap_or_else(|| vec![1u32]);
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("model load failed: {e}")))?;
    let num_layers = model.config().num_layers;
    let kv_dim = model.config().kv_dim();

    per_op_tap::arm_gate_bypass();
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
    eprintln!(
        "selected: cuda ({}) parity: {} — {}",
        cuda_model.device_name(),
        cuda_model.parity.status,
        cuda_model.parity.basis
    );

    let root = match out {
        Some(p) => p.to_path_buf(),
        None => std::env::temp_dir().join(format!("apr-parity-per-op-{}", std::process::id())),
    };
    std::fs::create_dir_all(&root)
        .map_err(|e| CliError::ValidationFailed(format!("{}: {e}", root.display())))?;
    eprintln!(
        "per-op dump root: {} ({} positions, {} layers)",
        root.display(),
        tokens.len(),
        num_layers
    );

    let max_seq = tokens.len() + 1;
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    cuda_model.executor_mut().reset_kv_cache_gpu();

    for (pos, &token_id) in tokens.iter().enumerate() {
        let cpu_dir = root.join("cpu").join(format!("pos-{pos:04}"));
        let gpu_dir = root.join("gpu").join(format!("pos-{pos:04}"));
        let plan = SaveTensorPlan::from_cli("all", &format!("0..{num_layers}"), cpu_dir)
            .map_err(|e| CliError::ValidationFailed(format!("plan: {e:?}")))?;
        per_op_tap::set_plan(Some(plan));
        let cpu = cuda_model
            .model()
            .forward_single_with_cache(token_id, &mut cpu_cache, pos);
        per_op_tap::set_plan(None);
        cpu.map_err(|e| {
            CliError::InferenceFailed(format!("CPU forward failed at pos {pos}: {e}"))
        })?;

        per_op_tap::set_gpu_dump(Some(GpuStageDumpConfig::with_output_dir(gpu_dir)));
        let gpu = cuda_model.forward_gpu_resident(token_id, &mut gpu_cache, pos);
        per_op_tap::set_gpu_dump(None);
        gpu.map_err(|e| {
            CliError::InferenceFailed(format!("GPU forward failed at pos {pos}: {e}"))
        })?;
    }

    // Pair every CPU file with its GPU twin; a stage one side never emits is skipped
    // (and listed), a stage both emit with different lengths scores cosine 0.
    let mut samples = Vec::new();
    let mut cpu_only = std::collections::BTreeSet::new();
    for pos in 0..tokens.len() {
        let cpu_dir = root.join("cpu").join(format!("pos-{pos:04}"));
        let gpu_dir = root.join("gpu").join(format!("pos-{pos:04}"));
        for (rel, layer) in stage_files(&cpu_dir) {
            let gpu_path = gpu_dir.join(&rel);
            if !gpu_path.exists() {
                cpu_only.insert(rel.clone());
                continue;
            }
            let (_, cpu) = read_stage_file(&cpu_dir.join(&rel))
                .map_err(|e| CliError::ValidationFailed(format!("{}: {e:?}", rel)))?;
            let (_, gpu) = read_stage_file(&gpu_path)
                .map_err(|e| CliError::ValidationFailed(format!("{}: {e:?}", rel)))?;
            let stage = Path::new(&rel)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            samples.push((stage, layer, pos, cpu, gpu));
        }
    }
    let rows = aggregate(&samples);
    let first = first_divergence(&rows, threshold);
    if json {
        let table: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| serde_json::json!({"stage": r.stage, "layer": r.layer, "min_cosine": r.min_cosine, "min_position": r.min_position, "max_abs": r.max_abs, "positions": r.positions}))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "model": file.display().to_string(),
                "positions": tokens.len(),
                "threshold": threshold,
                "threshold_basis": "evidence/parity/thresholds.yaml min_cosine (0.98, driver [U] until the gx10 pair)",
                "first_divergence": first.map(|r| serde_json::json!({"stage": r.stage, "layer": r.layer, "min_cosine": r.min_cosine, "min_position": r.min_position})),
                "cpu_only_stages": cpu_only.iter().collect::<Vec<_>>(),
                "rows": table,
            })
        );
    } else {
        print!("{}", render(&rows, threshold));
        if !cpu_only.is_empty() {
            println!(
                "(cpu-only stages, not compared: {})",
                cpu_only.iter().cloned().collect::<Vec<_>>().join(", ")
            );
        }
        match first {
            Some(r) => println!(
                "FIRST DIVERGING OP: {} layer {} — min cosine {:.6} at position {} (threshold {threshold})",
                r.stage,
                r.layer.map_or_else(|| "-".to_string(), |l| l.to_string()),
                r.min_cosine,
                r.min_position
            ),
            None => println!("every op >= {threshold} over {} positions: no diverging op", tokens.len()),
        }
    }
    Ok(())
}

/// `(relative path, layer)` for every `layer-N/<stage>.bin` and `<stage>.bin` under `dir`.
#[cfg(feature = "cuda")]
fn stage_files(dir: &Path) -> Vec<(String, Option<u32>)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(n) = name
            .strip_prefix("layer-")
            .and_then(|n| n.parse::<u32>().ok())
        {
            if let Ok(files) = std::fs::read_dir(e.path()) {
                for f in files.flatten() {
                    let fname = f.file_name().to_string_lossy().to_string();
                    if fname.ends_with(".bin") {
                        out.push((format!("{name}/{fname}"), Some(n)));
                    }
                }
            }
        } else if name.ends_with(".bin") {
            out.push((name, None));
        }
    }
    out.sort();
    out
}

#[cfg(not(feature = "cuda"))]
pub fn run(
    _file: &Path,
    _prompt: &str,
    _out: Option<&Path>,
    _threshold: f32,
    _json: bool,
) -> Result<()> {
    Err(crate::error::CliError::FeatureDisabled(
        "cuda feature required for apr parity --per-op".to_string(),
    ))
}
