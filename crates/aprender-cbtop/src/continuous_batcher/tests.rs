use super::*;

#[test]
fn test_create_batcher() {
    let batcher = ContinuousBatcher::new(64, 4096);
    assert_eq!(batcher.max_batch_size(), 64);
    assert_eq!(batcher.max_seq_len(), 4096);
    assert_eq!(batcher.running_count(), 0);
    assert_eq!(batcher.waiting_count(), 0);
}

#[test]
fn test_add_request() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    let req = InferenceRequest::new(SeqId(1), vec![1, 2, 3], 100);
    batcher.add_request(req);

    assert_eq!(batcher.waiting_count(), 1);
}

#[test]
fn test_fcfs_scheduling() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::FCFS);

    // Add requests in order
    for i in 0..5 {
        let req = InferenceRequest::new(SeqId(i), vec![1, 2, 3], 100);
        batcher.add_request(req);
    }

    // Schedule should return them in FCFS order
    let schedule = batcher.schedule();
    assert_eq!(schedule.batch_size, 5);
    assert_eq!(schedule.sequence_ids[0], SeqId(0));
    assert_eq!(schedule.sequence_ids[4], SeqId(4));
}

#[test]
fn test_sjf_scheduling() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::SJF);

    // Add requests with different sizes
    let req1 = InferenceRequest::new(SeqId(1), vec![1; 100], 500); // 600 estimated
    let req2 = InferenceRequest::new(SeqId(2), vec![1; 10], 50); // 60 estimated (shortest)
    let req3 = InferenceRequest::new(SeqId(3), vec![1; 50], 200); // 250 estimated

    batcher.add_request(req1);
    batcher.add_request(req2);
    batcher.add_request(req3);

    // Schedule should return shortest first
    let schedule = batcher.schedule();
    assert_eq!(schedule.sequence_ids[0], SeqId(2)); // Shortest
}

#[test]
fn test_process_outputs() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    let req = InferenceRequest::new(SeqId(1), vec![1, 2, 3], 5);
    batcher.add_request(req);
    batcher.schedule(); // Move to running

    // Process some tokens
    for token in [10, 11, 12] {
        batcher.process_outputs(vec![TokenOutput {
            seq_id: SeqId(1),
            token,
            is_eos: false,
        }]);
    }

    let seq = batcher.get_sequence(SeqId(1)).unwrap();
    assert_eq!(seq.output_tokens.len(), 3);
    assert_eq!(seq.output_tokens, vec![10, 11, 12]);
}

#[test]
fn test_preemption() {
    let mut batcher = ContinuousBatcher::new(64, 4096).with_policy(SchedulingPolicy::Priority {
        preempt_enabled: true,
    });

    // Add and schedule requests
    for i in 0..5 {
        let req = InferenceRequest::new(SeqId(i), vec![1; (i as usize + 1) * 100], 100);
        batcher.add_request(req);
    }
    batcher.schedule();

    // Preempt 2 sequences
    let preempted = batcher.preempt(2);
    assert_eq!(preempted.len(), 2);
    assert_eq!(batcher.running_count(), 3);
    assert_eq!(batcher.swapped_count(), 2);
}

#[test]
fn test_ema() {
    let mut ema = ExponentialMovingAverage::new(0.5);

    ema.update(1.0);
    assert!((ema.value() - 1.0).abs() < 0.01);

    ema.update(0.0);
    assert!((ema.value() - 0.5).abs() < 0.01);

    ema.update(0.0);
    assert!((ema.value() - 0.25).abs() < 0.01);
}

#[test]
fn test_speculative_output() {
    let output = SpeculativeOutput {
        accepted: vec![1, 2, 3],
        rejection_idx: Some(3),
        target_token: 100,
        draft_count: 5,
    };

    assert!((output.acceptance_rate() - 0.6).abs() < 0.01);
    assert_eq!(output.total_tokens(), 4);
}

#[test]
fn test_speculative_decoder() {
    let mut decoder = SpeculativeDecoder::new(5);

    // Simulate steps where all are accepted
    for _ in 0..10 {
        let draft = vec![1, 2, 3, 4, 5];
        let target_probs = vec![(1, 0.9), (2, 0.9), (3, 0.9), (4, 0.9), (5, 0.9), (6, 0.9)];
        decoder.simulate_step(&draft, &target_probs);
    }

    // High acceptance rate expected
    assert!(decoder.acceptance_rate() > 0.9);
    assert!(decoder.speedup() > 4.0);
}

#[test]
fn test_speculative_decoder_partial_accept() {
    let mut decoder = SpeculativeDecoder::new(5);

    // Draft: [1, 2, 3, 4, 5], Target rejects at position 2
    let draft = vec![1, 2, 3, 4, 5];
    let target_probs = vec![(1, 0.9), (2, 0.9), (99, 0.9), (4, 0.9), (5, 0.9)]; // 99 != 3

    let output = decoder.simulate_step(&draft, &target_probs);

    assert_eq!(output.accepted.len(), 2); // Only [1, 2] accepted
    assert_eq!(output.rejection_idx, Some(2));
    assert_eq!(output.target_token, 99);
}

#[test]
fn test_batch_schedule() {
    let schedule = BatchSchedule::empty();
    assert!(schedule.is_empty());
    assert_eq!(schedule.batch_size, 0);
}

#[test]
fn test_throughput_tracking() {
    let mut batcher = ContinuousBatcher::new(64, 4096);

    let req = InferenceRequest::new(SeqId(1), vec![1, 2, 3], 100);
    batcher.add_request(req);
    batcher.schedule();

    // Process tokens
    for i in 0..10 {
        batcher.process_outputs(vec![TokenOutput {
            seq_id: SeqId(1),
            token: i,
            is_eos: false,
        }]);
    }

    // Throughput should be > 0
    assert!(batcher.stats().total_tokens == 10);
}
