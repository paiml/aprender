//! FALSIFY-LORA_TARGET_SELECTION_V1_011 (R15a C5b): the CUDA trainer's NF4 blocks get
//! one adapter per selected target, each initialised by target, and q_proj and v_proj
//! keep the init they had. No device: these drive the pure helpers `upload_blocks` and
//! `with_model` call; the block calls themselves are row 005's #[ignore] tests.

use super::lora_init::{adapter_pair, added_adapters, nf4_init_adapters, nf4_targets};
use super::trainer::trainer_targets;
use crate::lora::LoraTarget;
use crate::transformer::TransformerConfig;

const RANK: usize = 4;

fn model_config() -> TransformerConfig {
    TransformerConfig { num_kv_heads: 1, head_dim_override: Some(16), ..TransformerConfig::tiny() }
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(ToString::to_string).collect()
}

/// The q_proj and v_proj init `upload_blocks` built before C5b, kept as the oracle.
fn old_qv_init(
    config: &TransformerConfig,
    i: usize,
) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let hidden_size = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim();
    let kv_hidden = config.num_kv_heads * config.head_dim();
    let lora_a_q: Vec<f32> = (0..hidden_size * RANK)
        .map(|j| ((j as f32 + i as f32 * 1000.0) * 0.1).sin() * 0.01)
        .collect();
    let lora_b_q = vec![0.0f32; RANK * q_dim];
    let lora_a_v: Vec<f32> = (0..hidden_size * RANK)
        .map(|j| ((j as f32 + i as f32 * 2000.0 + 500.0) * 0.1).sin() * 0.01)
        .collect();
    let lora_b_v = vec![0.0f32; RANK * kv_hidden];
    ((lora_a_q, lora_b_q), (lora_a_v, lora_b_v))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn falsify_lora_target_selection_v1_011_qv_init_is_the_old_init() {
    let config = model_config();
    for i in [0, 1, 2, 27] {
        let ((a_q, b_q), (a_v, b_v)) = old_qv_init(&config, i);
        for targets in [&[LoraTarget::Q, LoraTarget::V][..], &LoraTarget::ALL[..]] {
            let adapters = nf4_init_adapters(&config, i, targets, RANK);
            let (q_a, q_b) = adapter_pair(&adapters, LoraTarget::Q).expect("q_proj adapter");
            let (v_a, v_b) = adapter_pair(&adapters, LoraTarget::V).expect("v_proj adapter");
            assert_eq!(bits(q_a), bits(&a_q), "layer {i} q_proj A");
            assert_eq!(bits(q_b), bits(&b_q), "layer {i} q_proj B");
            assert_eq!(bits(v_a), bits(&a_v), "layer {i} v_proj A");
            assert_eq!(bits(v_b), bits(&b_v), "layer {i} v_proj B");
        }
    }
}

#[test]
fn falsify_lora_target_selection_v1_011_every_target_has_its_shape_and_own_init() {
    let config = model_config();
    let (hidden, q_dim, kv) =
        (config.hidden_size, config.q_dim(), config.num_kv_heads * config.head_dim());
    assert!(q_dim != kv && kv != hidden, "the test needs non-square q, k, v");

    let mut seen: Vec<(usize, LoraTarget, Vec<u32>)> = Vec::new();
    for l in 0..2 {
        let adapters = nf4_init_adapters(&config, l, &LoraTarget::ALL, RANK);
        let order: Vec<LoraTarget> = adapters.iter().map(|(t, _, _)| *t).collect();
        assert_eq!(order, LoraTarget::ALL.to_vec(), "layer {l} slot order");
        for (t, a, b) in &adapters {
            let (d_out, d_in) = t.dims(&config);
            assert_eq!(a.len(), d_in * RANK, "layer {l} {t:?} A length");
            assert_eq!(b.len(), RANK * d_out, "layer {l} {t:?} B length");
            assert!(b.iter().all(|&x| x == 0.0), "layer {l} {t:?} B is zero");
            assert!(a.iter().any(|&x| x != 0.0), "layer {l} {t:?} A is not zero");
            assert!(a.iter().all(|x| x.abs() <= 0.01), "layer {l} {t:?} A is small");
            let a_bits = bits(a);
            for (l2, t2, other) in &seen {
                assert_ne!(&a_bits, other, "layer {l} {t:?} A equals layer {l2} {t2:?} A");
            }
            seen.push((l, *t, a_bits));
        }
    }
    assert_eq!(seen.len(), 14);
}

#[test]
fn falsify_lora_target_selection_v1_011_targets_are_the_cpu_trainers() {
    let cases: [(Option<Vec<String>>, Vec<LoraTarget>); 6] = [
        (None, vec![LoraTarget::Q, LoraTarget::V]),
        (Some(names(&["all_linear"])), LoraTarget::ALL.to_vec()),
        (Some(names(&["mlp"])), vec![LoraTarget::Gate, LoraTarget::Up, LoraTarget::Down]),
        (Some(names(&["down_proj", "k_proj", "k_proj"])), vec![LoraTarget::K, LoraTarget::Down]),
        (Some(names(&["q_proj", "lm_head"])), vec![LoraTarget::Q]),
        (Some(names(&["lm_head"])), Vec::new()),
    ];
    let config = model_config();
    for (modules, want) in cases {
        let got = nf4_targets(modules.as_deref());
        assert_eq!(got, want, "{modules:?}");
        let cpu = trainer_targets(modules.as_deref()).map(|t| t.as_slice().to_vec());
        assert_eq!(cpu.unwrap_or_default(), got, "{modules:?} CPU trainer");
        // the blocks get exactly these targets, in this order
        let built: Vec<LoraTarget> =
            nf4_init_adapters(&config, 1, &got, RANK).iter().map(|(t, _, _)| *t).collect();
        assert_eq!(built, want, "{modules:?} built");
    }
}

#[test]
fn falsify_lora_target_selection_v1_011_new_gets_q_and_v_and_add_gets_the_rest() {
    let config = model_config();
    let all = nf4_init_adapters(&config, 0, &LoraTarget::ALL, RANK);
    let added: Vec<LoraTarget> = added_adapters(&all).map(|(t, _, _)| *t).collect();
    assert_eq!(
        added,
        vec![LoraTarget::K, LoraTarget::O, LoraTarget::Gate, LoraTarget::Up, LoraTarget::Down]
    );
    for (t, a, b) in added_adapters(&all) {
        let (pa, pb) = adapter_pair(&all, *t).expect("held target");
        assert_eq!((pa, pb), (a.as_slice(), b.as_slice()), "{t:?} pair");
    }

    let qv = nf4_init_adapters(&config, 0, &[LoraTarget::Q, LoraTarget::V], RANK);
    assert_eq!(added_adapters(&qv).count(), 0);
    let (qa, _) = adapter_pair(&qv, LoraTarget::Q).expect("q_proj");
    assert_eq!(qa, qv[0].1.as_slice());
    let (va, _) = adapter_pair(&qv, LoraTarget::V).expect("v_proj");
    assert_eq!(va, qv[1].1.as_slice());

    let k_down = nf4_init_adapters(&config, 0, &[LoraTarget::K, LoraTarget::Down], RANK);
    assert!(adapter_pair(&k_down, LoraTarget::Q).is_none());
    assert!(adapter_pair(&k_down, LoraTarget::V).is_none());
    assert_eq!(added_adapters(&k_down).count(), 2);
    assert!(adapter_pair(&k_down, LoraTarget::O).is_none());
}
