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

    /// Regression: MODEL-1 v2 5th failure mode — `samples_per_prompt` was
    /// silently ignored because `ResumeState` keyed only on prompt text, so
    /// the second sample of the same prompt was always counted as already
    /// done. The key is now `(prompt, sample_idx)`.
    ///
    /// Drop a JSONL where prompt "P" has only sample_idx=0; check that the
    /// resume set contains (P, 0) but NOT (P, 1), so a run with
    /// samples_per_prompt=2 would correctly re-request sample_idx=1.
    #[test]
    fn test_resume_state_keys_on_prompt_and_sample_idx() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output_path = dir.path().join("out.jsonl");
        {
            let mut f = std::fs::File::create(&output_path).expect("create");
            writeln!(
                f,
                r#"{{"prompt":"P","completion":"c0","tokens":10,"sample_idx":0,"source":"","kind":""}}"#
            )
            .unwrap();
        }
        let state = load_resume_state(&output_path).expect("load");
        assert!(state.existing_samples.contains(&("P".to_string(), 0)));
        assert!(!state.existing_samples.contains(&("P".to_string(), 1)));
        assert_eq!(state.total_tokens, 10);
        assert_eq!(state.generated_count, 1);
    }

    /// Back-compat: records written by the pre-patch binary had no
    /// `sample_idx` field; they must be treated as sample_idx=0 so existing
    /// teacher-completions.jsonl files resume correctly without being
    /// re-generated.
    #[test]
    fn test_resume_state_legacy_missing_sample_idx_is_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output_path = dir.path().join("legacy.jsonl");
        {
            let mut f = std::fs::File::create(&output_path).expect("create");
            writeln!(
                f,
                r#"{{"prompt":"P","completion":"c","tokens":7,"source":"","kind":""}}"#
            )
            .unwrap();
        }
        let state = load_resume_state(&output_path).expect("load");
        assert!(state.existing_samples.contains(&("P".to_string(), 0)));
    }
}
