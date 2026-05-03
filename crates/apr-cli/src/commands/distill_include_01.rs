#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_distill_strategy_parse() {
        assert!(matches!(
            "standard".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Standard)
        ));
        assert!(matches!(
            "kl".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Standard)
        ));
        assert!(matches!(
            "progressive".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Progressive)
        ));
        assert!(matches!(
            "ensemble".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Ensemble)
        ));
        assert!("unknown".parse::<DistillStrategy>().is_err());
    }

    #[test]
    fn test_run_teacher_not_found() {
        let result = run(
            Some(Path::new("/nonexistent.apr")),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            0.7,
            3,
            false,
            None,
            None,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_invalid_temperature() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            Some(input.path()),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            0.0,
            0.7,
            3,
            false,
            None,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Temperature")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_invalid_alpha() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            Some(input.path()),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            1.5,
            3,
            false,
            None,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Alpha")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_student() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            Some(input.path()),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            0.7,
            3,
            false,
            None,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Student")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_output() {
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 512]).expect("write");
        let mut student = NamedTempFile::with_suffix(".apr").expect("create student");
        student.write_all(&[0u8; 256]).expect("write");
        let result = run(
            Some(teacher.path()),
            Some(student.path()),
            None,
            None,
            "standard",
            3.0,
            0.7,
            3,
            false,
            None,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Output")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    /// Create a valid APR test model with some tensors
    fn make_test_model() -> NamedTempFile {
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let w0: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight", vec![8, 8], &w0);
        let w1: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        writer.add_tensor_f32("model.layers.1.self_attn.q_proj.weight", vec![8, 8], &w1);
        writer.add_tensor_f32("model.norm.weight", vec![8], &vec![1.0; 8]);
        writer.add_tensor_f32("model.embed_tokens.weight", vec![10, 8], &vec![0.1; 80]);

        let file = NamedTempFile::with_suffix(".apr").expect("create model");
        let bytes = writer.to_bytes().expect("serialize");
        std::fs::write(file.path(), bytes).expect("write");
        file
    }

    #[test]
    fn test_run_valid() {
        let teacher = make_test_model();
        let student = make_test_model();
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            Some(teacher.path()),
            Some(student.path()),
            None,
            Some(output.path()),
            "standard",
            3.0,
            0.7,
            3,
            false,
            None,
            None,
            true,
        );
        assert!(result.is_ok(), "Distill should succeed: {result:?}");

        // Verify output is a valid APR file
        let reader = aprender::serialization::apr::AprReader::open(output.path())
            .expect("output should be valid APR");
        assert!(!reader.tensors.is_empty(), "Output should have tensors");
        assert!(reader.get_metadata("distillation_teacher").is_some());
    }

    #[test]
    fn test_plan_mode() {
        let teacher = make_test_model();
        let result = run(
            Some(teacher.path()),
            None,
            None,
            None,
            "standard",
            3.0,
            0.7,
            3,
            true,
            None,
            None,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_json() {
        let teacher = make_test_model();
        let result = run(
            Some(teacher.path()),
            None,
            None,
            None,
            "progressive",
            4.0,
            0.5,
            5,
            true,
            None,
            None,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_progressive_no_student() {
        // Progressive distillation creates student from teacher (drops every other layer)
        let teacher = make_test_model();
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            Some(teacher.path()),
            None,
            None,
            Some(output.path()),
            "progressive",
            3.0,
            0.7,
            3,
            false,
            None,
            None,
            true,
        );
        assert!(result.is_ok(), "Progressive should succeed: {result:?}");

        // Verify student has fewer layers than teacher
        let reader = aprender::serialization::apr::AprReader::open(output.path())
            .expect("output should be valid APR");
        // Teacher has layers 0 and 1, progressive keeps only even (layer 0)
        let layer_names: Vec<_> = reader
            .tensors
            .iter()
            .filter(|t| t.name.contains("layers.1."))
            .collect();
        assert!(
            layer_names.is_empty(),
            "Layer 1 should be dropped by progressive distillation"
        );

        let layer0_names: Vec<_> = reader
            .tensors
            .iter()
            .filter(|t| t.name.contains("layers.0."))
            .collect();
        assert!(!layer0_names.is_empty(), "Layer 0 should be kept");
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            extract_layer_number("model.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
        assert_eq!(extract_layer_number("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_number("model.norm.weight"), None);
        assert_eq!(extract_layer_number("lm_head.weight"), None);
    }

    #[test]
    fn test_create_student_progressive() {
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            (vec![1.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.1.weight".to_string(),
            (vec![2.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.2.weight".to_string(),
            (vec![3.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.3.weight".to_string(),
            (vec![4.0; 4], vec![2, 2]),
        );
        tensors.insert("model.norm.weight".to_string(), (vec![1.0; 2], vec![2]));

        let student = create_student_from_teacher(&tensors, DistillStrategy::Progressive);
        // Even layers (0, 2) + non-layer tensors (norm) = 3
        assert_eq!(student.len(), 3);
        assert!(student.contains_key("model.layers.0.weight"));
        assert!(!student.contains_key("model.layers.1.weight"));
        assert!(student.contains_key("model.layers.2.weight"));
        assert!(!student.contains_key("model.layers.3.weight"));
        assert!(student.contains_key("model.norm.weight"));
    }

    #[test]
    fn test_create_student_standard() {
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("a".to_string(), (vec![1.0], vec![1]));
        tensors.insert("b".to_string(), (vec![2.0], vec![1]));

        let student = create_student_from_teacher(&tensors, DistillStrategy::Standard);
        assert_eq!(student.len(), 2, "Standard copies all tensors");
    }

    /// FALSIFY-APR-DISTILL-TRAIN-005: precompute is byte-deterministic.
    ///
    /// Contract `apr-cli-distill-train-v1.yaml` predicts: two runs of
    /// `apr distill --stage precompute` with the same inputs produce
    /// byte-identical `teacher_logits/manifest.json` output.
    ///
    /// Uses a deterministic fake teacher dir (two model-suffix files of
    /// fixed size + content) and asserts manifest equality across two
    /// invocations. Fails if the precompute manifest gains any
    /// non-deterministic field (timestamp, UUID, RNG draw, atomic-add
    /// reduction noise) — caught at the algorithm layer before MODEL-2
    /// can call into a real teacher forward.
    #[test]
    fn falsify_apr_distill_train_005_precompute_is_byte_deterministic() {
        use std::fs;
        let workdir = tempfile::tempdir().expect("create tempdir");
        let teacher_dir = workdir.path().join("teacher");
        fs::create_dir_all(&teacher_dir).expect("create teacher dir");
        let mut t1 = fs::File::create(teacher_dir.join("part1.bin")).expect("create part1");
        t1.write_all(&[0xABu8; 1024]).expect("write part1");
        let mut t2 = fs::File::create(teacher_dir.join("part2.bin")).expect("create part2");
        t2.write_all(&[0xCDu8; 2048]).expect("write part2");

        let dataset_path = workdir.path().join("dataset.bin");
        fs::write(&dataset_path, b"fake-dataset-shard").expect("write dataset");

        let make_config = |output_dir: &std::path::Path| -> String {
            format!(
                "teacher:\n  model_id: {teacher}\nstudent:\n  model_id: dummy-student\ndataset:\n  path: {dataset}\noutput:\n  dir: {out}\n",
                teacher = teacher_dir.display(),
                dataset = dataset_path.display(),
                out = output_dir.display()
            )
        };

        let out1 = workdir.path().join("run1");
        let out2 = workdir.path().join("run2");
        let cfg1_path = workdir.path().join("cfg1.yaml");
        let cfg2_path = workdir.path().join("cfg2.yaml");
        fs::write(&cfg1_path, make_config(&out1)).expect("write cfg1");
        fs::write(&cfg2_path, make_config(&out2)).expect("write cfg2");

        let cfg1 = DistillYamlConfig::load(&cfg1_path).expect("load cfg1");
        let cfg2 = DistillYamlConfig::load(&cfg2_path).expect("load cfg2");

        run_config_precompute(&cfg1, &cfg1_path, true).expect("precompute run1");
        run_config_precompute(&cfg2, &cfg2_path, true).expect("precompute run2");

        let manifest1 = fs::read(out1.join("logits/manifest.json")).expect("read manifest1");
        let manifest2 = fs::read(out2.join("logits/manifest.json")).expect("read manifest2");

        assert_eq!(
            manifest1, manifest2,
            "FALSIFY-APR-DISTILL-TRAIN-005: precompute manifest bytes diverged across runs — non-determinism in stage 1"
        );
    }

    /// FALSIFY-APR-DISTILL-TRAIN-005 (HF teacher branch): when the teacher
    /// `model_id` does not resolve to a local path, the manifest stub is
    /// also byte-deterministic across runs.
    #[test]
    fn falsify_apr_distill_train_005_precompute_remote_teacher_stub_is_deterministic() {
        use std::fs;
        let workdir = tempfile::tempdir().expect("create tempdir");
        let dataset_path = workdir.path().join("dataset.bin");
        fs::write(&dataset_path, b"fake-dataset-shard").expect("write dataset");

        let make_config = |output_dir: &std::path::Path| -> String {
            format!(
                "teacher:\n  model_id: paiml/qwen2.5-coder-7b-instruct\nstudent:\n  model_id: dummy-student\ndataset:\n  path: {dataset}\noutput:\n  dir: {out}\n",
                dataset = dataset_path.display(),
                out = output_dir.display()
            )
        };

        let out1 = workdir.path().join("run1");
        let out2 = workdir.path().join("run2");
        let cfg1_path = workdir.path().join("cfg1.yaml");
        let cfg2_path = workdir.path().join("cfg2.yaml");
        fs::write(&cfg1_path, make_config(&out1)).expect("write cfg1");
        fs::write(&cfg2_path, make_config(&out2)).expect("write cfg2");

        let cfg1 = DistillYamlConfig::load(&cfg1_path).expect("load cfg1");
        let cfg2 = DistillYamlConfig::load(&cfg2_path).expect("load cfg2");

        run_config_precompute(&cfg1, &cfg1_path, true).expect("precompute run1");
        run_config_precompute(&cfg2, &cfg2_path, true).expect("precompute run2");

        let manifest1 = fs::read(out1.join("logits/manifest.json")).expect("read manifest1");
        let manifest2 = fs::read(out2.join("logits/manifest.json")).expect("read manifest2");

        assert_eq!(
            manifest1, manifest2,
            "FALSIFY-APR-DISTILL-TRAIN-005 (remote stub): precompute manifest diverged across runs"
        );
    }

    /// FALSIFY-APR-DISTILL-TRAIN-006: stage train can resume from precompute cache.
    ///
    /// Contract `apr-cli-distill-train-v1.yaml` predicts: if `teacher_logits/`
    /// cache exists (i.e. precompute completed), stage train MUST proceed —
    /// it MUST NOT silently re-run teacher forward, and it MUST NOT error.
    /// If the cache is absent, stage train MUST error with a clear "run
    /// precompute first" message (the inverse half of the idempotency
    /// invariant — proves train ACTUALLY reads the cache).
    #[test]
    fn falsify_apr_distill_train_006_train_errors_without_precompute_cache() {
        use std::fs;
        let workdir = tempfile::tempdir().expect("create tempdir");
        let dataset_path = workdir.path().join("dataset.bin");
        fs::write(&dataset_path, b"fake-dataset-shard").expect("write dataset");

        let out_dir = workdir.path().join("run");
        let cfg_path = workdir.path().join("cfg.yaml");
        fs::write(
            &cfg_path,
            format!(
                "teacher:\n  model_id: paiml/some-teacher\nstudent:\n  model_id: dummy-student\ndataset:\n  path: {dataset}\noutput:\n  dir: {out}\n",
                dataset = dataset_path.display(),
                out = out_dir.display()
            ),
        )
        .expect("write cfg");

        let cfg = DistillYamlConfig::load(&cfg_path).expect("load cfg");
        let result = run_config_train(&cfg, &cfg_path, true);
        assert!(
            result.is_err(),
            "FALSIFY-APR-DISTILL-TRAIN-006: stage train without precompute cache MUST error — instead it succeeded"
        );
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(
                    msg.contains("Precompute") || msg.contains("precompute"),
                    "FALSIFY-APR-DISTILL-TRAIN-006: error must mention 'precompute' so user knows what to run, got: {msg}"
                );
            }
            other => panic!(
                "FALSIFY-APR-DISTILL-TRAIN-006: expected ValidationFailed, got {other:?}"
            ),
        }
    }

    /// FALSIFY-APR-DISTILL-TRAIN-006 (positive half): with the precompute
    /// cache present, stage train MUST NOT error on the cache-missing
    /// branch. Proves the manifest is actually consulted.
    #[test]
    fn falsify_apr_distill_train_006_train_does_not_error_when_cache_present() {
        use std::fs;
        let workdir = tempfile::tempdir().expect("create tempdir");
        let teacher_dir = workdir.path().join("teacher");
        fs::create_dir_all(&teacher_dir).expect("create teacher");
        let mut t1 = fs::File::create(teacher_dir.join("part1.bin")).expect("create part1");
        t1.write_all(&[0xABu8; 1024]).expect("write part1");

        let dataset_path = workdir.path().join("dataset.bin");
        fs::write(&dataset_path, b"fake-dataset-shard").expect("write dataset");

        let out_dir = workdir.path().join("run");
        let cfg_path = workdir.path().join("cfg.yaml");
        fs::write(
            &cfg_path,
            format!(
                "teacher:\n  model_id: {teacher}\nstudent:\n  model_id: paiml/some-student\ndataset:\n  path: {dataset}\noutput:\n  dir: {out}\n",
                teacher = teacher_dir.display(),
                dataset = dataset_path.display(),
                out = out_dir.display()
            ),
        )
        .expect("write cfg");

        let cfg = DistillYamlConfig::load(&cfg_path).expect("load cfg");

        run_config_precompute(&cfg, &cfg_path, true).expect("precompute");
        assert!(
            out_dir.join("logits/manifest.json").exists(),
            "precompute must drop manifest as a precondition for the cache-resume test"
        );

        let train_result = run_config_train(&cfg, &cfg_path, true);
        if let Err(CliError::ValidationFailed(msg)) = &train_result {
            assert!(
                !(msg.contains("Precompute") && msg.contains("not completed")),
                "FALSIFY-APR-DISTILL-TRAIN-006: train errored with 'Precompute stage not completed' even though manifest.json exists — cache-resume is broken: {msg}"
            );
        }
    }
}
