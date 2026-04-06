//! WGPU LoRA checkpoint save/load (7-module LoRA)
//!
//! # Contract: C-WGPU-CKPT-001, C-WGPU-CKPT-002

/// All 7 LoRA adapters for one transformer layer (Q/K/V/O/gate/up/down)
#[cfg(feature = "gpu")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraLayerSet {
    pub q: super::wgpu_nf4::LoraAdapter,
    pub k: super::wgpu_nf4::LoraAdapter,
    pub v: super::wgpu_nf4::LoraAdapter,
    pub o: super::wgpu_nf4::LoraAdapter,
    pub gate: super::wgpu_nf4::LoraAdapter,
    pub up: super::wgpu_nf4::LoraAdapter,
    pub down: super::wgpu_nf4::LoraAdapter,
}

#[cfg(feature = "gpu")]
impl LoraLayerSet {
    pub fn new(rank: u32, h: u32, q_dim: u32, kv_dim: u32, i_size: u32) -> Self {
        use super::wgpu_nf4::LoraAdapter;
        Self {
            q: LoraAdapter::new(rank, h, q_dim),
            k: LoraAdapter::new(rank, h, kv_dim),
            v: LoraAdapter::new(rank, h, kv_dim),
            o: LoraAdapter::new(rank, q_dim, h),
            gate: LoraAdapter::new(rank, h, i_size),
            up: LoraAdapter::new(rank, h, i_size),
            down: LoraAdapter::new(rank, i_size, h),
        }
    }
    pub fn num_params(&self) -> usize {
        self.q.num_params()
            + self.k.num_params()
            + self.v.num_params()
            + self.o.num_params()
            + self.gate.num_params()
            + self.up.num_params()
            + self.down.num_params()
    }
}

/// Checkpoint format: all 7 LoRA adapters per layer + metadata
#[cfg(feature = "gpu")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LoraCheckpointV2 {
    pub step: u32,
    pub rank: u32,
    pub alpha: f32,
    pub loss: f32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub layers: Vec<LoraLayerSet>,
}

#[cfg(feature = "gpu")]
pub fn save_lora_checkpoint(
    lora: &[LoraLayerSet],
    hidden_size: usize,
    output_dir: &std::path::Path,
    step: u32,
    loss: f32,
    rank: u32,
    alpha: f32,
) -> Result<std::path::PathBuf, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Cannot create output dir: {e}"))?;
    let ckpt = LoraCheckpointV2 {
        step,
        rank,
        alpha,
        loss,
        hidden_size: hidden_size as u32,
        num_layers: lora.len() as u32,
        layers: lora.to_vec(),
    };
    let filename = format!("lora-checkpoint-step{step}.json");
    let path = output_dir.join(&filename);
    let json = serde_json::to_string(&ckpt).map_err(|e| format!("Serialize: {e}"))?;
    std::fs::write(&path, &json).map_err(|e| format!("Write: {e}"))?;
    let mb = json.len() as f64 / 1024.0 / 1024.0;
    eprintln!("  Saved checkpoint: {} ({mb:.1} MB)", path.display());
    Ok(path)
}

#[cfg(feature = "gpu")]
pub fn load_lora_checkpoint(
    lora: &mut [LoraLayerSet],
    num_layers: usize,
    hidden_size: usize,
    checkpoint_path: &std::path::Path,
) -> Result<(u32, f32), String> {
    let json = std::fs::read_to_string(checkpoint_path).map_err(|e| format!("Read: {e}"))?;
    let ckpt: LoraCheckpointV2 = serde_json::from_str(&json).map_err(|e| format!("Parse: {e}"))?;
    if ckpt.layers.len() != num_layers {
        return Err(format!("Checkpoint {} layers, model {}", ckpt.layers.len(), num_layers));
    }
    if ckpt.hidden_size != hidden_size as u32 {
        return Err(format!("Checkpoint h={}, model h={hidden_size}", ckpt.hidden_size));
    }
    for (i, layer) in ckpt.layers.into_iter().enumerate() {
        lora[i] = layer;
    }
    eprintln!(
        "  Loaded checkpoint: step={}, loss={:.3}, {} layers",
        ckpt.step, ckpt.loss, num_layers
    );
    Ok((ckpt.step, ckpt.loss))
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use LoraLayerSet;

    #[test]
    fn test_checkpoint_round_trip() {
        let lora: Vec<LoraLayerSet> = (0..2).map(|_| LoraLayerSet::new(4, 8, 8, 8, 16)).collect();
        let tmpdir = std::env::temp_dir().join("entrenar-ckpt-v2");
        let path = save_lora_checkpoint(&lora, 8, &tmpdir, 42, 3.14, 4, 8.0).expect("save");
        assert!(path.exists());
        let mut lora2: Vec<LoraLayerSet> =
            (0..2).map(|_| LoraLayerSet::new(4, 8, 8, 8, 16)).collect();
        let (step, loss) = load_lora_checkpoint(&mut lora2, 2, 8, &path).expect("load");
        assert_eq!(step, 42);
        assert!((loss - 3.14).abs() < 1e-5);
        assert_eq!(lora[0].q.a, lora2[0].q.a);
        assert_eq!(lora[0].gate.b, lora2[0].gate.b);
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_checkpoint_dimension_mismatch() {
        let lora = vec![LoraLayerSet::new(4, 8, 8, 8, 16)];
        let tmpdir = std::env::temp_dir().join("entrenar-ckpt-v2-mm");
        let path = save_lora_checkpoint(&lora, 8, &tmpdir, 1, 5.0, 4, 8.0).expect("save");
        let mut lora2 =
            vec![LoraLayerSet::new(4, 16, 16, 16, 32), LoraLayerSet::new(4, 16, 16, 16, 32)];
        let result = load_lora_checkpoint(&mut lora2, 2, 16, &path);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
