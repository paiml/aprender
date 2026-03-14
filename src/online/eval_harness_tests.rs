use super::*;

// ============================================================================
// Scoring tests
// ============================================================================

#[test]
fn test_score_multiple_choice_basic() {
    let scores = vec![-1.0, -2.0, -3.0, -4.0];
    let (idx, score) = score_multiple_choice(&scores, false);
    assert_eq!(idx, 0);
    assert!((score - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_score_multiple_choice_second_best() {
    let scores = vec![-5.0, -1.0, -3.0, -4.0];
    let (idx, _) = score_multiple_choice(&scores, false);
    assert_eq!(idx, 1);
}

#[test]
fn test_score_multiple_choice_empty() {
    let (idx, score) = score_multiple_choice(&[], false);
    assert_eq!(idx, 0);
    assert!(score == f64::NEG_INFINITY);
}

#[test]
fn test_score_multiple_choice_single() {
    let scores = vec![-2.5];
    let (idx, score) = score_multiple_choice(&scores, false);
    assert_eq!(idx, 0);
    assert!((score - (-2.5)).abs() < 1e-10);
}

#[test]
fn test_score_multiple_choice_equal() {
    let scores = vec![-2.0, -2.0, -2.0];
    let (idx, _) = score_multiple_choice(&scores, false);
    // Any index is valid when all equal; just check it's in range
    assert!(idx < 3);
}

// ============================================================================
// Perplexity tests
// ============================================================================

#[test]
fn test_compute_perplexity_basic() {
    // PPL = exp(-LL/N)
    let ppl = compute_perplexity(-10.0, 10);
    assert!((ppl - 1.0_f64.exp()).abs() < 1e-10);
}

#[test]
fn test_compute_perplexity_zero_tokens() {
    let ppl = compute_perplexity(-5.0, 0);
    assert!(ppl == f64::INFINITY);
}

#[test]
fn test_compute_perplexity_zero_ll() {
    // exp(0) = 1.0
    let ppl = compute_perplexity(0.0, 10);
    assert!((ppl - 1.0).abs() < 1e-10);
}

#[test]
fn test_compute_perplexity_large_negative() {
    let ppl = compute_perplexity(-100.0, 10);
    // exp(10) ≈ 22026
    assert!(ppl > 20000.0);
    assert!(ppl.is_finite());
}

// ============================================================================
// Accuracy tests
// ============================================================================

#[test]
fn test_compute_accuracy_perfect() {
    let preds = vec![0, 1, 2, 3];
    let gold = vec![0, 1, 2, 3];
    assert!((compute_accuracy(&preds, &gold) - 1.0).abs() < 1e-10);
}

#[test]
fn test_compute_accuracy_zero() {
    let preds = vec![1, 2, 3, 0];
    let gold = vec![0, 1, 2, 3];
    assert!((compute_accuracy(&preds, &gold) - 0.0).abs() < 1e-10);
}

#[test]
fn test_compute_accuracy_half() {
    let preds = vec![0, 1, 0, 0];
    let gold = vec![0, 1, 2, 3];
    assert!((compute_accuracy(&preds, &gold) - 0.5).abs() < 1e-10);
}

#[test]
fn test_compute_accuracy_empty() {
    assert!((compute_accuracy(&[], &[]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_compute_accuracy_mismatched_len() {
    let preds = vec![0, 1];
    let gold = vec![0, 1, 2];
    assert!((compute_accuracy(&preds, &gold) - 0.0).abs() < 1e-10);
}

// ============================================================================
// EvalTask tests
// ============================================================================

#[test]
fn test_eval_task_new() {
    let task = EvalTask::new("test_task", TaskType::MultipleChoice);
    assert_eq!(task.name, "test_task");
    assert_eq!(task.task_type, TaskType::MultipleChoice);
    assert!(task.is_empty());
    assert_eq!(task.len(), 0);
    assert_eq!(task.num_fewshot, 0);
}

#[test]
fn test_eval_task_add_example() {
    let mut task = EvalTask::new("t", TaskType::Classification);
    task.add_example(EvalExample {
        context: "ctx".to_string(),
        choices: vec!["a".to_string(), "b".to_string()],
        gold_idx: Some(0),
        reference: None,
    });
    assert_eq!(task.len(), 1);
    assert!(!task.is_empty());
}

#[test]
fn test_eval_task_with_fewshot() {
    let task = EvalTask::new("t", TaskType::MultipleChoice).with_fewshot(5);
    assert_eq!(task.num_fewshot, 5);
}

// ============================================================================
// Mock data tests
// ============================================================================

#[test]
fn test_mock_hellaswag() {
    let task = mock_hellaswag();
    assert_eq!(task.name, "hellaswag");
    assert_eq!(task.task_type, TaskType::MultipleChoice);
    assert_eq!(task.len(), 2);
    assert_eq!(task.examples[0].choices.len(), 4);
    assert_eq!(task.examples[0].gold_idx, Some(0));
    assert_eq!(task.examples[1].gold_idx, Some(1));
}

// ============================================================================
// EvalReport tests
// ============================================================================

#[test]
fn test_eval_report_from_tasks() {
    let metrics = vec![
        TaskMetrics {
            task_name: "task_a".to_string(),
            task_type: TaskType::MultipleChoice,
            num_examples: 10,
            accuracy: Some(0.8),
            perplexity: None,
            avg_log_likelihood: None,
            predictions: vec![],
        },
        TaskMetrics {
            task_name: "task_b".to_string(),
            task_type: TaskType::MultipleChoice,
            num_examples: 20,
            accuracy: Some(0.6),
            perplexity: None,
            avg_log_likelihood: None,
            predictions: vec![],
        },
    ];
    let report = EvalReport::from_tasks(metrics);
    assert_eq!(report.num_tasks, 2);
    assert_eq!(report.total_examples, 30);
    assert!((report.macro_accuracy - 0.7).abs() < 1e-10);
}

#[test]
fn test_eval_report_empty() {
    let report = EvalReport::from_tasks(vec![]);
    assert_eq!(report.num_tasks, 0);
    assert_eq!(report.total_examples, 0);
    assert!((report.macro_accuracy - 0.0).abs() < 1e-10);
}

#[test]
fn test_eval_report_perplexity_only() {
    let metrics = vec![TaskMetrics {
        task_name: "ppl_task".to_string(),
        task_type: TaskType::Perplexity,
        num_examples: 5,
        accuracy: None,
        perplexity: Some(15.3),
        avg_log_likelihood: Some(-2.5),
        predictions: vec![],
    }];
    let report = EvalReport::from_tasks(metrics);
    assert_eq!(report.num_tasks, 1);
    // No accuracy tasks → macro_accuracy = 0.0
    assert!((report.macro_accuracy - 0.0).abs() < 1e-10);
}

// ============================================================================
// Harness config tests
// ============================================================================

#[test]
fn test_harness_config_default() {
    let cfg = HarnessConfig::default();
    assert!(cfg.tasks.is_empty());
    assert!(!cfg.length_normalize);
    assert_eq!(cfg.max_examples, 0);
}

// ============================================================================
// run_harness tests
// ============================================================================

#[test]
fn test_run_harness_empty_tasks() {
    let config = HarnessConfig::default();
    let result = run_harness(&config, |_: &str, _: &str| 0.0);
    assert!(result.is_err());
}

#[test]
fn test_run_harness_multiple_choice() {
    let task = mock_hellaswag();
    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 0,
    };

    // Score function: longer completions get higher scores (sensible choices are longer)
    let report = run_harness(&config, |_ctx: &str, completion: &str| {
        -(completion.len() as f64)
    })
    .unwrap();

    assert_eq!(report.num_tasks, 1);
    assert_eq!(report.total_examples, 2);
    assert!(report.tasks[0].accuracy.is_some());
}

#[test]
fn test_run_harness_perfect_scorer() {
    let task = mock_hellaswag();
    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 0,
    };

    // Perfect scorer: gives highest score to the correct answer
    let report = run_harness(&config, |ctx: &str, completion: &str| {
        if (ctx.contains("sandwich") && completion.contains("butter"))
            || (ctx.contains("cat") && completion.contains("purred"))
        {
            0.0 // Highest (least negative)
        } else {
            -10.0
        }
    })
    .unwrap();

    assert!((report.tasks[0].accuracy.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_run_harness_max_examples() {
    let mut task = mock_hellaswag();
    // Add more examples
    for i in 0..10 {
        task.add_example(EvalExample {
            context: format!("Context {}", i),
            choices: vec!["a".to_string(), "b".to_string()],
            gold_idx: Some(0),
            reference: None,
        });
    }

    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 3,
    };

    let report = run_harness(&config, |_: &str, _: &str| -1.0).unwrap();
    assert_eq!(report.tasks[0].num_examples, 3);
}

#[test]
fn test_run_harness_perplexity() {
    let mut task = EvalTask::new("ppl_test", TaskType::Perplexity);
    task.add_example(EvalExample {
        context: "The quick brown fox".to_string(),
        choices: vec![],
        gold_idx: None,
        reference: Some("The quick brown fox jumps over the lazy dog".to_string()),
    });
    task.add_example(EvalExample {
        context: "Hello world".to_string(),
        choices: vec![],
        gold_idx: None,
        reference: None, // Falls back to context
    });

    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 0,
    };

    let report = run_harness(&config, |_: &str, text: &str| {
        // Fake log-likelihood: -0.5 per whitespace-delimited token
        -(text.split_whitespace().count() as f64) * 0.5
    })
    .unwrap();

    assert!(report.tasks[0].perplexity.is_some());
    let ppl = report.tasks[0].perplexity.unwrap();
    assert!(ppl > 0.0);
    assert!(ppl.is_finite());
    assert!(report.tasks[0].avg_log_likelihood.is_some());
}

#[test]
fn test_run_harness_generation() {
    let mut task = EvalTask::new("gen_test", TaskType::Generation);
    task.add_example(EvalExample {
        context: "Translate: hello".to_string(),
        choices: vec![],
        gold_idx: None,
        reference: Some("hola".to_string()),
    });

    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 0,
    };

    let report = run_harness(&config, |_: &str, _: &str| 0.0).unwrap();
    // Generation returns placeholder metrics
    assert!(report.tasks[0].accuracy.is_none());
    assert!(report.tasks[0].perplexity.is_none());
}

#[test]
fn test_run_harness_multi_task() {
    let mc_task = mock_hellaswag();
    let mut ppl_task = EvalTask::new("ppl", TaskType::Perplexity);
    ppl_task.add_example(EvalExample {
        context: "test text".to_string(),
        choices: vec![],
        gold_idx: None,
        reference: None,
    });

    let config = HarnessConfig {
        tasks: vec![mc_task, ppl_task],
        length_normalize: false,
        max_examples: 0,
    };

    let report = run_harness(&config, |_: &str, _: &str| -1.0).unwrap();
    assert_eq!(report.num_tasks, 2);
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-EVAL-001: Perplexity is always positive.
#[test]
fn falsify_eval_001_perplexity_positive() {
    for ll in [-100.0, -10.0, -1.0, 0.0, 1.0] {
        for tokens in [1, 5, 10, 100, 1000] {
            let ppl = compute_perplexity(ll, tokens);
            assert!(
                ppl > 0.0,
                "PPL must be > 0, got {} for ll={}, tokens={}",
                ppl,
                ll,
                tokens
            );
        }
    }
}

/// FALSIFY-EVAL-002: Accuracy is bounded in [0.0, 1.0].
#[test]
fn falsify_eval_002_accuracy_bounded() {
    let test_cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 0, 0, 0], vec![0, 0, 0, 0]),
        (vec![0, 1, 2, 3], vec![3, 2, 1, 0]),
        (vec![0, 1, 0, 1], vec![0, 0, 1, 1]),
        (vec![0], vec![0]),
    ];

    for (preds, gold) in &test_cases {
        let acc = compute_accuracy(preds, gold);
        assert!(
            (0.0..=1.0).contains(&acc),
            "Accuracy must be in [0,1], got {} for {:?} vs {:?}",
            acc,
            preds,
            gold
        );
    }
}

/// FALSIFY-EVAL-003: Harness report aggregation is consistent.
#[test]
fn falsify_eval_003_report_consistency() {
    let metrics = vec![
        TaskMetrics {
            task_name: "a".to_string(),
            task_type: TaskType::MultipleChoice,
            num_examples: 100,
            accuracy: Some(0.9),
            perplexity: None,
            avg_log_likelihood: None,
            predictions: vec![],
        },
        TaskMetrics {
            task_name: "b".to_string(),
            task_type: TaskType::Classification,
            num_examples: 50,
            accuracy: Some(0.7),
            perplexity: None,
            avg_log_likelihood: None,
            predictions: vec![],
        },
        TaskMetrics {
            task_name: "c".to_string(),
            task_type: TaskType::Perplexity,
            num_examples: 25,
            accuracy: None,
            perplexity: Some(10.0),
            avg_log_likelihood: Some(-2.3),
            predictions: vec![],
        },
    ];

    let report = EvalReport::from_tasks(metrics);

    // Total examples = sum of all tasks
    assert_eq!(report.total_examples, 175);
    // num_tasks = number of tasks
    assert_eq!(report.num_tasks, 3);
    // macro_accuracy = mean of non-None accuracies only
    let expected_macro = (0.9 + 0.7) / 2.0;
    assert!(
        (report.macro_accuracy - expected_macro).abs() < 1e-10,
        "Macro accuracy should be mean of accuracy tasks only, got {}",
        report.macro_accuracy
    );
}
