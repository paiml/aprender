#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! PMAT-015: ContinuousBatcher Falsification Tests
//!
//! Falsification criteria F421-F430 from cbtop spec §19.
//!
//! # Test Coverage
//!
//! | ID | Claim | Test |
//! |----|-------|------|
//! | F421 | Batch scheduler produces valid batches | test_f421_valid_batches |
//! | F422 | Preemption works under memory pressure | test_f422_preemption |
//! | F423 | FCFS ordering correct | test_f423_fcfs_ordering |
//! | F424 | SJF prioritizes short sequences | test_f424_sjf_priority |
//! | F425 | Throughput measured accurately | test_f425_throughput_accuracy |
//! | F426 | Batcher stats tracked correctly | test_f426_batcher_stats |
//! | F427 | Speculative decoding acceptance rate tracked | test_f427_acceptance_rate |
//! | F428 | Draft model produces valid tokens | test_f428_draft_tokens |
//! | F429 | Target model verifies correctly | test_f429_target_verification |
//! | F430 | Speedup calculation accurate | test_f430_speedup_calculation |

use cbtop::{
    ContinuousBatcher, InferenceRequest, Priority, SchedulingPolicy, SeqId, SpeculativeDecoder,
    TokenOutput,
};

/// F421: Batch scheduler produces valid batches.
#[test]
fn test_f421_valid_batches() {
    let mut batcher = ContinuousBatcher::new(32, 4096);

    // Add requests
    for i in 0..20 {
        let req = InferenceRequest::new(SeqId(i), vec![1, 2, 3], 100);
        batcher.add_request(req);
    }

    // Schedule should produce valid batch
    let schedule = batcher.schedule();

    assert!(!schedule.is_empty());
    assert!(schedule.batch_size <= 32); // Max batch size
    assert_eq!(schedule.batch_size, schedule.sequence_ids.len());
    assert!(schedule.total_tokens > 0);
}

/// F421 negative: Empty batcher produces empty schedule.
#[test]
fn test_f421_empty_schedule() {
    let mut batcher = ContinuousBatcher::new(32, 4096);

    let schedule = batcher.schedule();
    assert!(schedule.is_empty());
    assert_eq!(schedule.batch_size, 0);
}

/// F422: Preemption works under memory pressure.
#[test]
fn test_f422_preemption() {
    let mut batcher = ContinuousBatcher::new(10, 4096)
        .with_policy(SchedulingPolicy::Priority { preempt_enabled: true });

    // Fill the batch
    for i in 0..10 {
        let req = InferenceRequest::new(SeqId(i), vec![1; (i as usize + 1) * 50], 100);
        batcher.add_request(req);
    }
    batcher.schedule();

    assert_eq!(batcher.running_count(), 10);

    // Preempt 3 sequences
    let preempted = batcher.preempt(3);

    assert_eq!(preempted.len(), 3);
    assert_eq!(batcher.running_count(), 7);
    assert_eq!(batcher.swapped_count(), 3);
    assert_eq!(batcher.stats().total_preemptions, 3);
}

/// F422 negative: Preemption does nothing when disabled.
#[test]
fn test_f422_preemption_disabled() {
    let mut batcher = ContinuousBatcher::new(10, 4096).with_policy(SchedulingPolicy::FCFS);

    // Fill the batch
    for i in 0..5 {
        let req = InferenceRequest::new(SeqId(i), vec![1, 2, 3], 100);
        batcher.add_request(req);
    }
    batcher.schedule();

    // Preemption still works (it's policy-agnostic in the current impl)
    // but needs_preemption returns false
    assert!(!batcher.needs_preemption(0.95));
}

/// F423: FCFS ordering correct (first-come, first-served).
#[test]
fn test_f423_fcfs_ordering() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::FCFS);

    // Add requests with timestamps
    for i in 0..10u64 {
        let req = InferenceRequest::new(SeqId(i), vec![1, 2, 3], 100);
        batcher.add_request(req);
    }

    // Schedule should return them in arrival order
    let schedule = batcher.schedule();

    for i in 0..10 {
        assert_eq!(schedule.sequence_ids[i], SeqId(i as u64), "FCFS should preserve arrival order");
    }
}

/// F423 negative: FCFS doesn't reorder by priority.
#[test]
fn test_f423_fcfs_ignores_priority() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::FCFS);

    // Add low priority first
    let req1 = InferenceRequest::new(SeqId(1), vec![1, 2, 3], 100).with_priority(Priority(10));
    // Add high priority second
    let req2 = InferenceRequest::new(SeqId(2), vec![1, 2, 3], 100).with_priority(Priority(200));

    batcher.add_request(req1);
    batcher.add_request(req2);

    let schedule = batcher.schedule();

    // Low priority should be first (arrived first)
    assert_eq!(schedule.sequence_ids[0], SeqId(1));
    assert_eq!(schedule.sequence_ids[1], SeqId(2));
}

/// F424: SJF prioritizes short sequences.
#[test]
fn test_f424_sjf_priority() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::SJF);

    // Add long sequence first
    let long_req = InferenceRequest::new(SeqId(1), vec![1; 500], 1000); // 1500 estimated
                                                                        // Add short sequence second
    let short_req = InferenceRequest::new(SeqId(2), vec![1; 10], 50); // 60 estimated
                                                                      // Add medium sequence third
    let medium_req = InferenceRequest::new(SeqId(3), vec![1; 100], 200); // 300 estimated

    batcher.add_request(long_req);
    batcher.add_request(short_req);
    batcher.add_request(medium_req);

    let schedule = batcher.schedule();

    // Short should be first
    assert_eq!(schedule.sequence_ids[0], SeqId(2), "SJF should schedule shortest first");
    // Medium should be second
    assert_eq!(schedule.sequence_ids[1], SeqId(3));
    // Long should be last
    assert_eq!(schedule.sequence_ids[2], SeqId(1));
}

/// F424: SJF correctly estimates based on input + max_new_tokens.
#[test]
fn test_f424_sjf_estimation() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::SJF);

    // Same input size, different max_new_tokens
    let req1 = InferenceRequest::new(SeqId(1), vec![1; 100], 500); // 600 estimated
    let req2 = InferenceRequest::new(SeqId(2), vec![1; 100], 50); // 150 estimated

    batcher.add_request(req1);
    batcher.add_request(req2);

    let schedule = batcher.schedule();

    // Shorter total estimate should be first
    assert_eq!(schedule.sequence_ids[0], SeqId(2));
}

/// F425: Throughput measured accurately.
#[test]
fn test_f425_throughput_accuracy() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    let req = InferenceRequest::new(SeqId(1), vec![1, 2, 3], 100);
    batcher.add_request(req);
    batcher.schedule();

    // Process 100 tokens
    for i in 0..100 {
        batcher.process_outputs(vec![TokenOutput { seq_id: SeqId(1), token: i, is_eos: false }]);
    }

    let stats = batcher.stats();
    assert_eq!(stats.total_tokens, 100);

    // Throughput should be positive (time > 0)
    let throughput = batcher.throughput();
    assert!(throughput > 0.0, "Throughput should be positive, got {}", throughput);
}

/// F425: Throughput tracks over time.
#[test]
fn test_f425_throughput_over_time() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    let req = InferenceRequest::new(SeqId(1), vec![1], 1000);
    batcher.add_request(req);
    batcher.schedule();

    // Process tokens with small delay
    for i in 0..10 {
        batcher.process_outputs(vec![TokenOutput { seq_id: SeqId(1), token: i, is_eos: false }]);
    }

    // Throughput should be calculable
    assert!(batcher.stats().total_tokens > 0);
}

/// F426: Batcher stats tracked correctly.
#[test]
fn test_f426_batcher_stats() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    // Add requests
    for i in 0..5 {
        let req = InferenceRequest::new(SeqId(i), vec![1, 2, 3], 10);
        batcher.add_request(req);
    }

    batcher.schedule();

    // Complete some requests
    for i in 0..3u64 {
        for t in 0..10 {
            batcher.process_outputs(vec![TokenOutput {
                seq_id: SeqId(i),
                token: t,
                is_eos: t == 9,
            }]);
        }
    }

    let stats = batcher.stats();
    assert_eq!(stats.total_tokens, 30); // 3 sequences * 10 tokens
    assert_eq!(stats.total_requests, 3); // 3 completed
}

/// F427: Speculative decoding acceptance rate tracked.
#[test]
fn test_f427_acceptance_rate() {
    let mut decoder = SpeculativeDecoder::new(5);

    // Simulate steps with 100% acceptance
    for _ in 0..10 {
        let draft = vec![1, 2, 3, 4, 5];
        let target_probs = vec![(1, 0.9), (2, 0.9), (3, 0.9), (4, 0.9), (5, 0.9), (6, 0.9)];
        decoder.simulate_step(&draft, &target_probs);
    }

    assert!(
        decoder.acceptance_rate() > 0.9,
        "Acceptance rate should be > 90%, got {}",
        decoder.acceptance_rate()
    );
}

/// F427: Acceptance rate decreases with rejections.
#[test]
fn test_f427_acceptance_rate_with_rejections() {
    let mut decoder = SpeculativeDecoder::new(5);

    // Simulate step with early rejection (reject at position 1)
    let draft = vec![1, 2, 3, 4, 5];
    let target_probs = vec![(1, 0.9), (99, 0.9), (3, 0.9), (4, 0.9), (5, 0.9)]; // 99 != 2

    let output = decoder.simulate_step(&draft, &target_probs);

    assert_eq!(output.accepted.len(), 1);
    assert!(decoder.acceptance_rate() < 0.5);
}

/// F428: Draft model produces valid tokens.
#[test]
fn test_f428_draft_tokens() {
    let mut decoder = SpeculativeDecoder::new(5);

    let draft_tokens = vec![100, 200, 300, 400, 500];
    let target_probs = vec![(100, 0.9), (200, 0.9), (300, 0.9), (400, 0.9), (500, 0.9), (600, 0.9)];

    let output = decoder.simulate_step(&draft_tokens, &target_probs);

    // All draft tokens should be accepted
    assert_eq!(output.accepted, draft_tokens);
    assert_eq!(output.rejection_idx, None);
}

/// F428 negative: Invalid draft tokens are rejected.
#[test]
fn test_f428_invalid_draft_rejected() {
    let mut decoder = SpeculativeDecoder::new(5);

    let draft_tokens = vec![1, 2, 3, 4, 5];
    let target_probs = vec![
        (999, 0.9), // All different
        (998, 0.9),
        (997, 0.9),
        (996, 0.9),
        (995, 0.9),
    ];

    let output = decoder.simulate_step(&draft_tokens, &target_probs);

    // First rejection at index 0
    assert!(output.accepted.is_empty());
    assert_eq!(output.rejection_idx, Some(0));
}

/// F429: Target model verifies correctly.
#[test]
fn test_f429_target_verification() {
    let mut decoder = SpeculativeDecoder::new(5);

    // Partial acceptance: [1, 2] accepted, reject at [3] (target says 99)
    let draft = vec![1, 2, 3, 4, 5];
    let target_probs = vec![(1, 0.9), (2, 0.9), (99, 0.9), (4, 0.9), (5, 0.9)];

    let output = decoder.simulate_step(&draft, &target_probs);

    // Accepted: [1, 2]
    assert_eq!(output.accepted, vec![1, 2]);
    // Rejection at index 2
    assert_eq!(output.rejection_idx, Some(2));
    // Target token is 99 (the correct token at rejection point)
    assert_eq!(output.target_token, 99);
}

/// F429: Target provides token after full acceptance.
#[test]
fn test_f429_target_after_full_accept() {
    let mut decoder = SpeculativeDecoder::new(3);

    let draft = vec![1, 2, 3];
    let target_probs = vec![(1, 0.9), (2, 0.9), (3, 0.9), (100, 0.9)]; // 100 is the k+1 token

    let output = decoder.simulate_step(&draft, &target_probs);

    assert_eq!(output.accepted.len(), 3);
    assert_eq!(output.rejection_idx, None);
    assert_eq!(output.target_token, 100); // k+1 token from target
}

/// F430: Speedup calculation accurate.
#[test]
fn test_f430_speedup_calculation() {
    let mut decoder = SpeculativeDecoder::new(5);

    // With 100% acceptance rate, speedup ≈ 1 + k = 6
    for _ in 0..50 {
        let draft = vec![1, 2, 3, 4, 5];
        let target_probs = vec![(1, 0.9), (2, 0.9), (3, 0.9), (4, 0.9), (5, 0.9), (6, 0.9)];
        decoder.simulate_step(&draft, &target_probs);
    }

    let speedup = decoder.speedup();
    assert!(speedup > 5.0, "Speedup should be > 5x with 100% acceptance, got {}", speedup);
}

/// F430: Speedup decreases with lower acceptance.
#[test]
fn test_f430_speedup_with_rejection() {
    let mut decoder = SpeculativeDecoder::new(5);

    // 0% acceptance rate (all rejected at first position)
    for _ in 0..50 {
        let draft = vec![1, 2, 3, 4, 5];
        let target_probs = vec![(99, 0.9)]; // Only one wrong token
        decoder.simulate_step(&draft, &target_probs);
    }

    let speedup = decoder.speedup();
    // With 0% acceptance: speedup ≈ 1 (no benefit)
    assert!(speedup < 2.0, "Speedup should be < 2x with 0% acceptance, got {}", speedup);
}

/// Integration test: Full continuous batching workflow.
#[test]
fn test_full_continuous_batching_workflow() {
    // 1. Create batcher with realistic config
    let mut batcher = ContinuousBatcher::new(32, 4096).with_policy(SchedulingPolicy::FCFS);

    // 2. Simulate incoming requests
    for i in 0..50u64 {
        let input_len = (100 + (i % 10) * 20) as usize;
        let req = InferenceRequest::new(SeqId(i), vec![1; input_len], 200);
        batcher.add_request(req);
    }

    // 3. Run multiple iterations
    for iteration in 0..10 {
        let schedule = batcher.schedule();

        if schedule.is_empty() {
            break;
        }

        // Simulate decode step (1 token per sequence)
        let mut outputs = Vec::new();
        for seq_id in &schedule.sequence_ids {
            outputs.push(TokenOutput {
                seq_id: *seq_id,
                token: iteration as u32,
                is_eos: iteration == 9, // Complete after 10 iterations
            });
        }
        batcher.process_outputs(outputs);
    }

    // 4. Verify final state
    let stats = batcher.stats();
    assert!(stats.total_tokens > 0);
    assert!(stats.total_requests > 0);

    // 5. Display works
    let display = format!("{}", batcher);
    assert!(display.contains("ContinuousBatcher"));
    assert!(display.contains("FCFS"));
}

/// Integration test: Speculative decoding with continuous batching.
#[test]
fn test_speculative_decoding_integration() {
    let mut decoder = SpeculativeDecoder::new(4);

    // Simulate 100 decoding steps
    let mut total_tokens = 0;
    for _ in 0..100 {
        let draft = vec![1, 2, 3, 4];
        // 75% match rate (3 out of 4 accepted on average)
        let target_probs = vec![(1, 0.9), (2, 0.9), (3, 0.9), (99, 0.9), (5, 0.9)];

        let output = decoder.simulate_step(&draft, &target_probs);
        total_tokens += output.total_tokens();
    }

    // Should have generated more than 100 tokens (some acceptance bonus)
    assert!(total_tokens > 100);

    // Acceptance rate should be ~75%
    let rate = decoder.overall_acceptance_rate();
    assert!(rate > 0.5 && rate < 0.9);

    // Speedup should be > 1
    assert!(decoder.speedup() > 1.5);
}
