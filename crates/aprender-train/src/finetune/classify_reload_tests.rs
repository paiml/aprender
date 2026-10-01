//! Phase 5 tracer preflight: train -> write -> FRESH-PROCESS reload -> ordered
//! probability vector.
//!
//! # Why a whole test file for one round trip
//!
//! EVAL-05 and every LoRA `QualityBlock` in the benchmark depend on a route that, before
//! plan 05-06, did not exist and that the code did not obviously lack:
//!
//! - `ClassifyPipeline::from_apr` loads the base transformer and then calls
//!   `build_lora_layers`, producing FRESH adapters. The trained ones are never read.
//! - `forward_only_tokenized` returns `(loss, predicted_class)`. A benchmark row needs the
//!   full ordered distribution, and it must be obtainable without handing the model a
//!   ground-truth label.
//!
//! Scheduling 40 remote 9B cells against an assumed API is how a matrix of confident,
//! wrong numbers gets produced. This runs the thinnest real version of that route on a toy
//! config first, on CPU, in under a minute.
//!
//! # Why a FRESH PROCESS
//!
//! An in-process "reload" inherits warm state — the very tensors it claims to be reading
//! from disk are still resident and still referenced. It can pass while the adapter file
//! is never opened. The child leg re-execs this test binary with
//! `APRENDER_RELOAD_CHILD=1`, so the only things crossing the boundary are the bytes on
//! disk and the numbers the child prints.
//!
//! # Why a TWO-SIDED control
//!
//! "Fresh-process vectors match in-process vectors" is satisfiable by an adapter that is
//! never applied, if the base alone happens to produce those numbers. So the child also
//! evaluates the base WITHOUT the adapter and the parent asserts the two differ. Together
//! the assertions say: the reload reproduces the trained model, AND the trained model is
//! not the base.

use super::classification::SafetySample;
use super::classify_pipeline::{ClassifyConfig, ClassifyPipeline};
use super::classify_trainer::{ClassifyTrainer, TrainingConfig};
use crate::transformer::{Transformer, TransformerConfig};
use std::path::{Path, PathBuf};

/// Marker that puts this binary into the child leg.
const CHILD_ENV: &str = "APRENDER_RELOAD_CHILD";
/// Directory the child reloads from.
const DIR_ENV: &str = "APRENDER_RELOAD_DIR";

const NUM_CLASSES: usize = 3;
const MAX_SEQ_LEN: usize = 24;
const LORA_RANK: usize = 4;
const EPOCHS: usize = 3;

/// The inputs both legs classify. Fixed text so the byte-level tokenization is identical
/// in parent and child without shipping a tokenizer.
const PROBES: [&str; 3] = ["stance probe alpha", "stance probe bravo", "stance probe charlie"];

fn model_config() -> TransformerConfig {
    TransformerConfig::tiny()
}

fn classify_config() -> ClassifyConfig {
    ClassifyConfig {
        num_classes: NUM_CLASSES,
        lora_rank: LORA_RANK,
        lora_alpha: 8.0,
        // Large enough that three epochs on six samples visibly move the head away from
        // its initialization — the two-sided control has to clear 1e-3.
        learning_rate: 5e-2,
        epochs: EPOCHS,
        max_seq_len: MAX_SEQ_LEN,
        log_interval: 100,
        batch_size: 2,
        accumulation_steps: 1,
        gradient_clip_norm: None,
        class_weights: None,
        quantize_nf4: false,
    }
}

/// Two deterministic samples per class.
fn corpus() -> Vec<SafetySample> {
    let mut rows = Vec::new();
    for label in 0..NUM_CLASSES {
        for k in 0..2 {
            rows.push(SafetySample { input: format!("class {label} example {k}"), label });
        }
    }
    rows
}

/// The byte-level tokenization `ClassifyPipeline::pre_tokenize` falls back to when no BPE
/// tokenizer is loaded. Reproduced here so both legs feed the model identical ids.
fn probe_tokens(text: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();
    ids.truncate(MAX_SEQ_LEN);
    if ids.is_empty() {
        ids.push(0);
    }
    ids
}

fn probability_vectors(pipeline: &mut ClassifyPipeline) -> Vec<Vec<f32>> {
    PROBES.iter().map(|text| pipeline.predict_proba_tokenized(&probe_tokens(text))).collect()
}

/// Largest elementwise gap between two equally-shaped vector sets.
fn max_elementwise_difference(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "probe count must match");
    let mut worst = 0.0f32;
    for (left, right) in a.iter().zip(b.iter()) {
        assert_eq!(left.len(), right.len(), "class count must match");
        for (&l, &r) in left.iter().zip(right.iter()) {
            worst = worst.max((l - r).abs());
        }
    }
    worst
}

// ======================================================================================
// The child leg
// ======================================================================================

/// Reload base + adapter from `dir` and print both probability vector sets.
///
/// Runs in a process that has never seen the trainer: everything it knows comes from the
/// two files it opens.
fn run_child(dir: &Path) {
    let mcfg = model_config();

    // (a) WITH the adapter — the route under test.
    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg)
        .expect("child reloads the written base APR");
    let mut with_adapter = ClassifyPipeline::from_model(base, &mcfg, classify_config());
    with_adapter
        .load_adapter(&dir.join("model.adapter.apr"))
        .expect("child installs the written adapter");
    for (i, probs) in probability_vectors(&mut with_adapter).iter().enumerate() {
        let joined: Vec<String> = probs.iter().map(|p| format!("{p:.9}")).collect();
        println!("RELOAD_CHILD_WITH_ADAPTER {i} {}", joined.join(" "));
    }

    // (b) WITHOUT the adapter — the control. Same base file, fresh head and fresh LoRA.
    let base_only = Transformer::from_apr(dir.join("base.apr"), &mcfg)
        .expect("child reloads the written base APR a second time");
    let mut no_adapter = ClassifyPipeline::from_model(base_only, &mcfg, classify_config());
    for (i, probs) in probability_vectors(&mut no_adapter).iter().enumerate() {
        let joined: Vec<String> = probs.iter().map(|p| format!("{p:.9}")).collect();
        println!("RELOAD_CHILD_NO_ADAPTER {i} {}", joined.join(" "));
    }
}

/// Parse the child's marker lines back into vectors.
fn parse_child_vectors(stdout: &str, marker: &str) -> Vec<Vec<f32>> {
    let mut rows: Vec<(usize, Vec<f32>)> = stdout
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(marker)?.trim();
            let mut parts = rest.split_whitespace();
            let index: usize = parts.next()?.parse().ok()?;
            let probs: Vec<f32> = parts.filter_map(|p| p.parse::<f32>().ok()).collect();
            Some((index, probs))
        })
        .collect();
    rows.sort_by_key(|(index, _)| *index);
    rows.into_iter().map(|(_, probs)| probs).collect()
}

// ======================================================================================
// The preflight
// ======================================================================================

#[test]
fn lora_adapter_reload_yields_ordered_probability_vector() {
    // The child leg re-enters this same test by name. When the marker is set, do the
    // child's job and return — the parent reads what we print.
    if std::env::var(CHILD_ENV).is_ok() {
        let dir = PathBuf::from(
            std::env::var(DIR_ENV).expect("the parent always sets the reload directory"),
        );
        run_child(&dir);
        return;
    }

    let dir = std::env::temp_dir().join(format!(
        "aprender-reload-preflight-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("preflight directory is creatable");

    // ── 1. Train the toy classifier ────────────────────────────────────────────────
    let mcfg = model_config();
    let pipeline = ClassifyPipeline::new(&mcfg, classify_config());
    let training = TrainingConfig {
        epochs: EPOCHS,
        // The A6 configuration this plan made real: no held-out rows, no early stop.
        val_split: 0.0,
        save_every: EPOCHS,
        early_stopping_patience: 0,
        checkpoint_dir: dir.join("ckpt"),
        seed: 13,
        log_interval: 1,
        ..TrainingConfig::default()
    };
    let mut trainer = ClassifyTrainer::new(pipeline, corpus(), training)
        .expect("the toy trainer constructs under val_split 0.0");
    let result = trainer.train();
    assert_eq!(
        result.epochs_completed, EPOCHS,
        "the preflight must train the epochs it asked for, or the artifact it writes is \
         not the artifact it thinks it wrote"
    );

    // ── 1b. Make the LoRA half of the artifact LOAD-BEARING ────────────────────────
    // CPU training leaves the adapters at their initialization (see
    // `cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing`), and LoRA B
    // is zero-initialized — so a round trip over an untouched adapter would be satisfied
    // by a loader that read only the classifier head. Stamping small non-zero,
    // position-dependent weights makes the probability vectors DEPEND on the LoRA
    // tensors, so the fidelity assertion below can only pass if they were reloaded too.
    {
        let pipeline = trainer.pipeline_mut();
        for (idx, lora) in pipeline.lora_layers.iter_mut().enumerate() {
            let a: Vec<f32> = (0..lora.lora_a().data().len())
                .map(|i| 0.01 + (idx as f32).mul_add(1e-3, i as f32 * 1e-5))
                .collect();
            let b: Vec<f32> = (0..lora.lora_b().data().len())
                .map(|i| -0.01 - (idx as f32).mul_add(1e-3, i as f32 * 1e-5))
                .collect();
            lora.lora_a_mut()
                .data_mut()
                .as_slice_mut()
                .expect("contiguous lora_a")
                .copy_from_slice(&a);
            lora.lora_b_mut()
                .data_mut()
                .as_slice_mut()
                .expect("contiguous lora_b")
                .copy_from_slice(&b);
        }
    }
    // Re-write the checkpoint so the file on disk carries the stamped adapters.
    trainer
        .save_checkpoint(
            &dir.join("ckpt").join(format!("epoch-{}", EPOCHS - 1)),
            EPOCHS - 1,
            result.epoch_metrics.last().expect("the run recorded per-epoch metrics"),
        )
        .expect("the checkpoint re-writes with the stamped adapters");

    // ── 2. The in-process answer, from the model that just trained ─────────────────
    let in_process = probability_vectors(trainer.pipeline_mut());
    for probs in &in_process {
        assert_eq!(
            probs.len(),
            NUM_CLASSES,
            "a probability vector must have one entry per class, in class-index order"
        );
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "probabilities must sum to 1.0 within 1e-5, got {total}"
        );
        assert!(
            probs.iter().all(|p| p.is_finite() && *p >= 0.0),
            "probabilities must be finite and non-negative: {probs:?}"
        );
    }

    // ── 3. Write the artifacts the child will reload ───────────────────────────────
    let base_path = dir.join("base.apr");
    trainer.pipeline_mut().model.save_apr(&base_path).expect("the base transformer writes to APR");
    // The adapter is what `ClassifyTrainer::save_checkpoint` already wrote at the final
    // epoch — this preflight reloads the PRODUCTION artifact, not a special one.
    let adapter_src =
        dir.join("ckpt").join(format!("epoch-{}", EPOCHS - 1)).join("model.adapter.apr");
    assert!(
        adapter_src.exists(),
        "the trainer must have written {} — without it there is no adapter to reload",
        adapter_src.display()
    );
    let adapter_path = dir.join("model.adapter.apr");
    std::fs::copy(&adapter_src, &adapter_path).expect("adapter is copyable");

    // The exact artifact set, enumerated. These byte counts become `base_model_bytes` /
    // `artifact_bytes` / `deployable_total_bytes` in 05-09.
    let base_bytes = std::fs::metadata(&base_path).expect("base.apr exists").len();
    let adapter_bytes = std::fs::metadata(&adapter_path).expect("model.adapter.apr exists").len();
    assert!(base_bytes > 0 && adapter_bytes > 0);
    println!(
        "RELOAD_ARTIFACTS base.apr={base_bytes} model.adapter.apr={adapter_bytes} \
         deployable_total={}",
        base_bytes + adapter_bytes
    );

    // ── 4. The fresh process ───────────────────────────────────────────────────────
    // `--exact` matches libtest's FULL path, which does NOT include the crate name.
    // Passing the bare function name selects nothing, and the child then exits 0 having
    // run no tests — a green result proving nothing. Derived from `module_path!` rather
    // than written out, so renaming this module cannot re-introduce that.
    let module = module_path!().split_once("::").map_or(module_path!(), |(_crate_name, rest)| rest);
    let child_test = format!("{module}::lora_adapter_reload_yields_ordered_probability_vector");

    let exe = std::env::current_exe().expect("the test binary knows its own path");
    let output = std::process::Command::new(exe)
        .arg(&child_test)
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DIR_ENV, &dir)
        .output()
        .expect("the child test process spawns");

    // The ExitStatus is read from the reaped child, never through a pipe.
    let status = output.status;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        status.success(),
        "the fresh-process reload leg failed ({status})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    // A child that selected NO test also exits 0. Refuse that explicitly: it is the same
    // vacuity CR-02 names, and it is easy to reach by mistyping the test filter.
    assert!(
        stdout.contains("1 passed"),
        "the child must have RUN the reload test, not merely exited 0 with an empty \
         filter\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );

    let reloaded = parse_child_vectors(&stdout, "RELOAD_CHILD_WITH_ADAPTER");
    let base_only = parse_child_vectors(&stdout, "RELOAD_CHILD_NO_ADAPTER");
    assert_eq!(
        reloaded.len(),
        PROBES.len(),
        "the child must report one vector per probe\n--- stdout ---\n{stdout}"
    );
    assert_eq!(base_only.len(), PROBES.len());

    // ── 5. Reload fidelity ─────────────────────────────────────────────────────────
    let reload_gap = max_elementwise_difference(&in_process, &reloaded);
    println!("RELOAD_FIDELITY max_elementwise_difference={reload_gap:.9}");
    assert!(
        reload_gap < 1e-4,
        "fresh-process probabilities must reproduce the in-process ones within 1e-4; \
         max elementwise difference was {reload_gap}. A reload that silently dropped the \
         adapter would land here."
    );

    // ── 6. Two-sided control ───────────────────────────────────────────────────────
    let control_gap = max_elementwise_difference(&in_process, &base_only);
    println!("RELOAD_CONTROL max_elementwise_difference={control_gap:.9}");
    assert!(
        control_gap > 1e-3,
        "the base WITHOUT the adapter must differ from the trained model by more than \
         1e-3 on at least one element; it differed by {control_gap}. Without this the \
         fidelity assertion above is tautological — it would also pass if the adapter \
         were never applied."
    );

    // ── 7. Persist the measurements ────────────────────────────────────────────────
    // stdout from a `cargo test` run is easy to lose and easy to misquote. These four
    // numbers are the preflight's actual finding — the SUMMARY quotes THIS file, not a
    // remembered console line.
    let metrics = format!(
        "{{\n  \"base_model_bytes\": {base_bytes},\n  \"artifact_bytes\": {adapter_bytes},\n  \
         \"deployable_total_bytes\": {},\n  \"reload_fidelity_max_elementwise_difference\": {reload_gap:.9},\n  \
         \"two_sided_control_max_elementwise_difference\": {control_gap:.9},\n  \
         \"num_classes\": {NUM_CLASSES},\n  \"epochs_completed\": {}\n}}\n",
        base_bytes + adapter_bytes,
        result.epochs_completed,
    );
    // `APRENDER_RELOAD_METRICS_OUT` lets a caller pin the destination; otherwise the
    // system temp directory, which under a sandboxed runner may be private to the test
    // process.
    let metrics_path = std::env::var("APRENDER_RELOAD_METRICS_OUT").map_or_else(
        |_| std::env::temp_dir().join("aprender-reload-preflight-metrics.json"),
        PathBuf::from,
    );
    let _ = std::fs::write(&metrics_path, &metrics);

    let _ = std::fs::remove_dir_all(&dir);
}

// ======================================================================================
// The refusals that make a silent partial load impossible (T-05-06-05)
// ======================================================================================

/// Build and train the smallest artifact set, returning the directory holding
/// `base.apr` + `model.adapter.apr`.
fn written_artifacts(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "aprender-reload-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("directory is creatable");

    let mcfg = model_config();
    let pipeline = ClassifyPipeline::new(&mcfg, classify_config());
    let training = TrainingConfig {
        epochs: 1,
        val_split: 0.0,
        save_every: 1,
        early_stopping_patience: 0,
        checkpoint_dir: dir.join("ckpt"),
        seed: 13,
        log_interval: 1,
        ..TrainingConfig::default()
    };
    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus(), training).expect("toy trainer constructs");
    trainer.train();
    trainer.pipeline_mut().model.save_apr(dir.join("base.apr")).expect("base writes");
    std::fs::copy(
        dir.join("ckpt").join("epoch-0").join("model.adapter.apr"),
        dir.join("model.adapter.apr"),
    )
    .expect("adapter is copyable");
    dir
}

/// FINDING (plan 05-06, pre-existing defect — NOT introduced here, NOT fixed here).
///
/// On the CPU path, `ClassifyTrainer` moves the classification head and NOTHING else. The
/// LoRA adapters are applied in the forward pass — `forward_hidden_with_lora` really does
/// use them — but they receive no gradient, so after training they are byte-identical to
/// their initialization, and since LoRA B is zero-initialized their forward contribution
/// is exactly zero as well.
///
/// The mechanism is a graph cut in `ClassificationHead::mean_pool`
/// (`finetune/classification.rs`), which returns `Tensor::from_vec(pooled, requires_grad)`
/// — a NEW tensor carrying no backward op back to `hidden_states`. The backward started at
/// `logits` therefore terminates at the pooled vector and never enters the transformer.
///
/// Why this test exists rather than a fix: making the CPU path train adapters means
/// implementing a differentiable pooling plus a CPU transformer backward. That is an
/// architectural change, several times the size of this plan, and it belongs to whoever
/// owns the CPU training path. Encoding the CURRENT behaviour here means the day someone
/// does fix it, this test fails and says exactly what changed — which is strictly better
/// than a comment nobody runs.
///
/// Scope for Phase 5: the 9B LoRA cells run on the lambda-vector GPU (D-09), where
/// `gpu_training` is `Some` and `backward_gpu_blocks` / `backward_nf4_gpu_blocks` run an
/// explicit transformer backward. Whether THAT path trains the adapters is a separate
/// question this CPU test cannot answer and must not be read as answering.
#[test]
fn cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing() {
    let mcfg = model_config();
    let fresh = ClassifyPipeline::new(&mcfg, classify_config());
    let fresh_a: Vec<Vec<f32>> =
        fresh.lora_layers.iter().map(|l| l.lora_a().data().iter().copied().collect()).collect();
    let fresh_b: Vec<Vec<f32>> =
        fresh.lora_layers.iter().map(|l| l.lora_b().data().iter().copied().collect()).collect();
    let fresh_head: Vec<f32> = fresh.classifier.weight.data().iter().copied().collect();
    assert!(
        !fresh.lora_layers.is_empty(),
        "a config with lora_rank {LORA_RANK} must build LoRA layers, or this test is \
         comparing two empty vectors"
    );

    let dir = std::env::temp_dir().join(format!(
        "aprender-lora-movement-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let training = TrainingConfig {
        epochs: EPOCHS,
        val_split: 0.0,
        save_every: EPOCHS,
        early_stopping_patience: 0,
        checkpoint_dir: dir.clone(),
        seed: 13,
        log_interval: 1,
        ..TrainingConfig::default()
    };
    let mut trainer =
        ClassifyTrainer::new(ClassifyPipeline::new(&mcfg, classify_config()), corpus(), training)
            .expect("toy trainer constructs");
    trainer.train();

    let trained = trainer.pipeline_mut();
    let mut worst_a = 0.0f32;
    let mut worst_b = 0.0f32;
    for (i, l) in trained.lora_layers.iter().enumerate() {
        for (&x, &y) in fresh_a[i].iter().zip(l.lora_a().data().iter()) {
            worst_a = worst_a.max((x - y).abs());
        }
        for (&x, &y) in fresh_b[i].iter().zip(l.lora_b().data().iter()) {
            worst_b = worst_b.max((x - y).abs());
        }
    }
    let mut worst_head = 0.0f32;
    for (&x, &y) in fresh_head.iter().zip(trained.classifier.weight.data().iter()) {
        worst_head = worst_head.max((x - y).abs());
    }

    // The positive half: something DID train, so a "nothing moved" reading is excluded.
    assert!(
        worst_head > 1e-6,
        "the classification head must move, or this test proves nothing about WHICH \
         parameters train (head moved {worst_head})"
    );
    // The finding.
    assert_eq!(
        (worst_a, worst_b),
        (0.0, 0.0),
        "CPU training moved the LoRA adapters (A by {worst_a}, B by {worst_b}). If this \
         is now intentional, the graph cut in ClassificationHead::mean_pool was repaired \
         — update this test and re-read the Phase 5 LoRA-cell assumptions, which were \
         written against adapters that did NOT train on CPU."
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Isolate the LOADER from the TRAINER.
///
/// Because CPU training leaves the adapters at their initialization (see above), the
/// preflight's two-sided control cannot distinguish "the LoRA tensors were installed" from
/// "only the head was installed" — both halves of the file happen to agree with a fresh
/// pipeline on the LoRA side. So this test PERTURBS the LoRA weights by hand before the
/// checkpoint is written, then requires the reloaded pipeline to carry exactly those
/// perturbed values. Nothing about training is involved; this is purely: does
/// `load_adapter` install the LoRA half of the file?
#[test]
fn classify_reload_installs_the_lora_tensors_not_only_the_head() {
    use super::classify_trainer::EpochMetrics;

    let dir = std::env::temp_dir().join(format!(
        "aprender-reload-loader-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("directory is creatable");

    let mcfg = model_config();
    let training = TrainingConfig {
        epochs: 1,
        val_split: 0.0,
        save_every: 1,
        early_stopping_patience: 0,
        checkpoint_dir: dir.join("ckpt"),
        seed: 13,
        log_interval: 1,
        ..TrainingConfig::default()
    };
    let mut trainer =
        ClassifyTrainer::new(ClassifyPipeline::new(&mcfg, classify_config()), corpus(), training)
            .expect("toy trainer constructs");
    trainer.train();

    // Stamp distinctive, position-dependent values into every LoRA tensor. Position
    // dependence matters: a loader that filled every element with one constant, or that
    // installed layer 0's tensors into every layer, would pass a uniform-value check.
    let mut expected_a: Vec<Vec<f32>> = Vec::new();
    let mut expected_b: Vec<Vec<f32>> = Vec::new();
    {
        let pipeline = trainer.pipeline_mut();
        for (idx, lora) in pipeline.lora_layers.iter_mut().enumerate() {
            let a: Vec<f32> = lora
                .lora_a()
                .data()
                .iter()
                .enumerate()
                .map(|(i, _)| 0.5 + idx as f32 + i as f32 * 1e-3)
                .collect();
            let b: Vec<f32> = lora
                .lora_b()
                .data()
                .iter()
                .enumerate()
                .map(|(i, _)| -0.25 - idx as f32 - i as f32 * 1e-3)
                .collect();
            lora.lora_a_mut()
                .data_mut()
                .as_slice_mut()
                .expect("contiguous lora_a")
                .copy_from_slice(&a);
            lora.lora_b_mut()
                .data_mut()
                .as_slice_mut()
                .expect("contiguous lora_b")
                .copy_from_slice(&b);
            expected_a.push(a);
            expected_b.push(b);
        }
    }

    let stamped = dir.join("stamped");
    trainer
        .save_checkpoint(
            &stamped,
            0,
            &EpochMetrics {
                epoch: 0,
                train_loss: 0.0,
                train_accuracy: 0.0,
                val_loss: 0.0,
                val_accuracy: 0.0,
                learning_rate: 0.0,
                epoch_time_ms: 0,
                samples_per_sec: 0.0,
            },
        )
        .expect("the stamped checkpoint writes");

    trainer.pipeline_mut().model.save_apr(dir.join("base.apr")).expect("base writes");

    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg).expect("base reloads");
    let mut loaded = ClassifyPipeline::from_model(base, &mcfg, classify_config());
    loaded.load_adapter(&stamped.join("model.adapter.apr")).expect("the stamped adapter installs");

    let mut worst = 0.0f32;
    for (idx, lora) in loaded.lora_layers.iter().enumerate() {
        for (&want, &got) in expected_a[idx].iter().zip(lora.lora_a().data().iter()) {
            worst = worst.max((want - got).abs());
        }
        for (&want, &got) in expected_b[idx].iter().zip(lora.lora_b().data().iter()) {
            worst = worst.max((want - got).abs());
        }
    }
    assert!(
        worst == 0.0,
        "the reloaded LoRA tensors must be exactly the ones written; largest \
         disagreement was {worst}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn classify_reload_refuses_an_adapter_with_a_different_class_count() {
    let dir = written_artifacts("classes");
    let mcfg = model_config();
    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg).expect("base reloads");

    let mut wrong = ClassifyConfig { num_classes: NUM_CLASSES + 1, ..classify_config() };
    wrong.num_classes = NUM_CLASSES + 1;
    let mut pipeline = ClassifyPipeline::from_model(base, &mcfg, wrong);

    let error = pipeline
        .load_adapter(&dir.join("model.adapter.apr"))
        .expect_err("an adapter trained for a different class count must be refused");
    let text = error.to_string();
    assert!(text.contains("classes"), "the refusal must say what disagreed: {text}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn classify_reload_refuses_an_adapter_with_a_different_rank() {
    let dir = written_artifacts("rank");
    let mcfg = model_config();
    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg).expect("base reloads");

    let wrong = ClassifyConfig { lora_rank: LORA_RANK * 2, ..classify_config() };
    let mut pipeline = ClassifyPipeline::from_model(base, &mcfg, wrong);

    let error = pipeline
        .load_adapter(&dir.join("model.adapter.apr"))
        .expect_err("an adapter of a different rank must be refused");
    let text = error.to_string();
    assert!(text.contains("rank"), "the refusal must say what disagreed: {text}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn classify_reload_refuses_a_missing_adapter_file() {
    let dir = written_artifacts("missing");
    let mcfg = model_config();
    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg).expect("base reloads");
    let mut pipeline = ClassifyPipeline::from_model(base, &mcfg, classify_config());

    let error = pipeline
        .load_adapter(&dir.join("no-such-adapter.apr"))
        .expect_err("an absent adapter is a refusal, not a silent no-op");
    assert!(error.to_string().contains("no-such-adapter.apr"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// The load leaves the pipeline UNTOUCHED when it refuses. A loader that installed the
/// classifier head and only then discovered a bad LoRA tensor would leave a model that is
/// neither the base nor the trained one, and nothing downstream could tell.
#[test]
fn classify_reload_leaves_the_pipeline_untouched_when_it_refuses() {
    let dir = written_artifacts("atomic");
    let mcfg = model_config();
    let base = Transformer::from_apr(dir.join("base.apr"), &mcfg).expect("base reloads");
    let wrong = ClassifyConfig { lora_rank: LORA_RANK * 2, ..classify_config() };
    let mut pipeline = ClassifyPipeline::from_model(base, &mcfg, wrong);

    let before = probability_vectors(&mut pipeline);
    let _ = pipeline
        .load_adapter(&dir.join("model.adapter.apr"))
        .expect_err("rank disagreement is refused");
    let after = probability_vectors(&mut pipeline);

    let moved = max_elementwise_difference(&before, &after);
    assert!(moved == 0.0, "a refused load must change nothing; the probabilities moved by {moved}");

    let _ = std::fs::remove_dir_all(&dir);
}
