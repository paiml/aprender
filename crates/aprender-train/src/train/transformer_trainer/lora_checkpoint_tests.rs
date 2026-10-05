//! FALSIFY-LORA_TARGET_SELECTION_V1_010 (R15a C5a): the CUDA trainer's LoRA checkpoint
//! names, lays out and reads back every adapter by its target, and keeps the q/v names,
//! shapes and values it wrote before. No device: these drive the pure helpers the
//! trainer's APR save, resume and PEFT save call.

use aprender::serialization::apr::{AprReader, AprWriter};

use super::lora_checkpoint::{
    apr_tensor_names, device_to_peft, held_module_names, peft_adapter, peft_module_path,
    read_apr_adapters, write_apr_adapters, DeviceAdapters,
};
use crate::lora::LoraTarget;
use crate::transformer::TransformerConfig;

const RANK: usize = 4;
const ALPHA: f32 = 8.0;

fn model_config() -> TransformerConfig {
    TransformerConfig { num_kv_heads: 1, head_dim_override: Some(16), ..TransformerConfig::tiny() }
}

/// A device-layout adapter for `target` with values distinct per layer, target and index.
fn device_adapter(
    config: &TransformerConfig,
    layer: usize,
    target: LoraTarget,
) -> (Vec<f32>, Vec<f32>) {
    let (d_out, d_in) = target.dims(config);
    let tag = (layer * 7 + target as usize) as f32 * 1000.0;
    let a = (0..d_in * RANK).map(|i| tag + i as f32 * 0.5).collect();
    let b = (0..RANK * d_out).map(|i| -tag - i as f32 * 0.25).collect();
    (a, b)
}

fn all_adapters(
    config: &TransformerConfig,
    layer: usize,
    targets: &[LoraTarget],
) -> DeviceAdapters {
    targets
        .iter()
        .map(|&t| {
            let (a, b) = device_adapter(config, layer, t);
            (t, a, b)
        })
        .collect()
}

fn round_trip(writer: &AprWriter) -> AprReader {
    let bytes = writer.to_bytes().expect("APR to_bytes");
    AprReader::from_bytes(bytes).expect("APR from_bytes")
}

/// The q/v conversion `save_cuda_lora_adapter` did before C5a, kept as the oracle.
fn old_qv_conversion(
    a: &[f32],
    b_scaled: &[f32],
    d_out: usize,
    hidden: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut a_transposed = vec![0.0f32; RANK * hidden];
    for r in 0..hidden {
        for c in 0..RANK {
            a_transposed[c * hidden + r] = a[r * RANK + c];
        }
    }
    let inv_scale = if scale.abs() > 1e-10 { 1.0 / scale } else { 1.0 };
    let mut b_transposed = vec![0.0f32; d_out * RANK];
    for r in 0..RANK {
        for c in 0..d_out {
            b_transposed[c * RANK + r] = b_scaled[r * d_out + c] * inv_scale;
        }
    }
    (a_transposed, b_transposed)
}

#[test]
fn falsify_lora_target_selection_v1_010_qv_names_are_the_old_names() {
    for l in 0..3 {
        assert_eq!(
            apr_tensor_names(l, LoraTarget::Q),
            (format!("lora.{l}.q_proj.lora_a"), format!("lora.{l}.q_proj.lora_b"))
        );
        assert_eq!(
            apr_tensor_names(l, LoraTarget::V),
            (format!("lora.{l}.v_proj.lora_a"), format!("lora.{l}.v_proj.lora_b"))
        );
        assert_eq!(
            peft_module_path(l, LoraTarget::Q),
            format!("model.layers.{l}.self_attn.q_proj")
        );
        assert_eq!(
            peft_module_path(l, LoraTarget::V),
            format!("model.layers.{l}.self_attn.v_proj")
        );
    }
    let qv = vec![all_adapters(&model_config(), 0, &[LoraTarget::Q, LoraTarget::V])];
    assert_eq!(held_module_names(&qv), vec!["q_proj", "v_proj"]);
}

#[test]
fn falsify_lora_target_selection_v1_010_names_follow_each_target() {
    // TransformerTrainer::save_lora_adapter's module paths (CPU trainer).
    let cpu_paths = [
        ("q_proj", "self_attn.q_proj"),
        ("k_proj", "self_attn.k_proj"),
        ("v_proj", "self_attn.v_proj"),
        ("o_proj", "self_attn.o_proj"),
        ("gate_proj", "mlp.gate_proj"),
        ("up_proj", "mlp.up_proj"),
        ("down_proj", "mlp.down_proj"),
    ];
    let mut apr_names = std::collections::BTreeSet::new();
    for l in 0..2 {
        for (t, (module, path)) in LoraTarget::ALL.into_iter().zip(cpu_paths) {
            assert_eq!(t.module_name(), module);
            assert_eq!(peft_module_path(l, t), format!("model.layers.{l}.{path}"), "{t:?}");
            let (a, b) = apr_tensor_names(l, t);
            assert_eq!(a, format!("lora.{l}.{module}.lora_a"));
            assert_eq!(b, format!("lora.{l}.{module}.lora_b"));
            assert!(apr_names.insert(a) && apr_names.insert(b));
        }
    }
    let mixed = vec![
        all_adapters(&model_config(), 0, &[LoraTarget::Down, LoraTarget::K]),
        all_adapters(&model_config(), 1, &[LoraTarget::Q, LoraTarget::Down]),
    ];
    assert_eq!(held_module_names(&mixed), vec!["q_proj", "k_proj", "down_proj"]);
}

#[test]
fn falsify_lora_target_selection_v1_010_peft_layout_per_target() {
    let config = model_config();
    let scale = ALPHA / RANK as f32;
    let (hidden, q_dim, kv) =
        (config.hidden_size, config.q_dim(), config.num_kv_heads * config.head_dim());
    assert!(q_dim != kv && kv != hidden, "the test needs non-square q, k, v");

    // q and v: bit for bit what the old code wrote.
    for (t, d_out) in [(LoraTarget::Q, q_dim), (LoraTarget::V, kv)] {
        let (a, b) = device_adapter(&config, 0, t);
        let (name, lora) = peft_adapter(0, (t, &a, &b), &config, RANK, ALPHA);
        let (old_a, old_b) = old_qv_conversion(&a, &b, d_out, hidden, scale);
        assert_eq!(name, peft_module_path(0, t));
        assert_eq!(lora.lora_a().data().to_vec(), old_a, "{t:?} A");
        assert_eq!(lora.lora_b().data().to_vec(), old_b, "{t:?} B");
    }

    // every target: its own (d_out, d_in), A transposed, B transposed and unscaled.
    for t in LoraTarget::ALL {
        let (d_out, d_in) = t.dims(&config);
        let (a, b) = device_adapter(&config, 1, t);
        let (_, lora) = peft_adapter(1, (t, &a, &b), &config, RANK, ALPHA);
        assert_eq!((lora.d_out(), lora.d_in()), (d_out, d_in), "{t:?} dims");
        let (pa, pb) = (lora.lora_a().data().to_vec(), lora.lora_b().data().to_vec());
        assert_eq!((pa.len(), pb.len()), (RANK * d_in, d_out * RANK), "{t:?} lengths");
        for i in 0..d_in {
            for r in 0..RANK {
                assert_eq!(pa[r * d_in + i], a[i * RANK + r], "{t:?} A[{r},{i}]");
            }
        }
        for o in 0..d_out {
            for r in 0..RANK {
                assert_eq!(pb[o * RANK + r], b[r * d_out + o] / scale, "{t:?} B[{o},{r}]");
            }
        }
    }

    // σ about 0 leaves B as it is.
    let (a, b) = device_adapter(&config, 0, LoraTarget::Up);
    let (_, pb) = device_to_peft(&a, &b, LoraTarget::Up.dims(&config), RANK, 0.0);
    let (d_out, _) = LoraTarget::Up.dims(&config);
    assert_eq!(pb[RANK], b[1]);
    assert_eq!(pb.len(), d_out * RANK);
}

#[test]
fn falsify_lora_target_selection_v1_010_apr_round_trip_every_target() {
    let config = model_config();
    let layers: Vec<DeviceAdapters> =
        (0..2).map(|l| all_adapters(&config, l, &LoraTarget::ALL)).collect();
    let mut writer = AprWriter::new();
    for (l, adapters) in layers.iter().enumerate() {
        write_apr_adapters(&mut writer, l, adapters);
    }
    // an adapter with an empty A is not written
    write_apr_adapters(&mut writer, 2, &vec![(LoraTarget::K, Vec::new(), vec![1.0])]);
    let reader = round_trip(&writer);

    for (l, adapters) in layers.iter().enumerate() {
        assert_eq!(&read_apr_adapters(&reader, l, &LoraTarget::ALL), adapters, "layer {l}");
        let qv = read_apr_adapters(&reader, l, &[LoraTarget::Q, LoraTarget::V]);
        assert_eq!(qv, vec![adapters[0].clone(), adapters[2].clone()], "layer {l} q/v");
        let down_k = read_apr_adapters(&reader, l, &[LoraTarget::Down, LoraTarget::K]);
        assert_eq!(down_k, vec![adapters[6].clone(), adapters[1].clone()], "layer {l} order");
    }
    assert!(read_apr_adapters(&reader, 2, &LoraTarget::ALL).is_empty());
    assert!(reader.read_tensor_f32("lora.2.k_proj.lora_b").is_err());
}

#[test]
fn falsify_lora_target_selection_v1_010_old_qv_checkpoint_still_restores() {
    let config = model_config();
    let (a_q, b_q) = device_adapter(&config, 0, LoraTarget::Q);
    let (a_v, b_v) = device_adapter(&config, 0, LoraTarget::V);
    // the names and flat shapes the pre-C5a APR save wrote
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("lora.0.q_proj.lora_a", vec![a_q.len()], &a_q);
    writer.add_tensor_f32("lora.0.q_proj.lora_b", vec![b_q.len()], &b_q);
    writer.add_tensor_f32("lora.0.v_proj.lora_a", vec![a_v.len()], &a_v);
    writer.add_tensor_f32("lora.0.v_proj.lora_b", vec![b_v.len()], &b_v);
    let reader = round_trip(&writer);

    let want = vec![(LoraTarget::Q, a_q, b_q), (LoraTarget::V, a_v, b_v)];
    assert_eq!(read_apr_adapters(&reader, 0, &[LoraTarget::Q, LoraTarget::V]), want);
    // a block holding more targets restores the two the file has and keeps the rest
    assert_eq!(read_apr_adapters(&reader, 0, &LoraTarget::ALL), want);
    assert!(read_apr_adapters(&reader, 1, &[LoraTarget::Q, LoraTarget::V]).is_empty());

    // and the new save writes exactly those names and values for q/v
    let mut new_writer = AprWriter::new();
    write_apr_adapters(&mut new_writer, 0, &want);
    let new_reader = round_trip(&new_writer);
    for name in [
        "lora.0.q_proj.lora_a",
        "lora.0.q_proj.lora_b",
        "lora.0.v_proj.lora_a",
        "lora.0.v_proj.lora_b",
    ] {
        assert_eq!(new_reader.read_tensor_f32(name), reader.read_tensor_f32(name), "{name}");
    }
}
