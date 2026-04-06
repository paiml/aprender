
/// IMP-058: Token batch accumulator
/// Target: Accumulate tokens for batched processing
#[test]
#[cfg(feature = "gpu")]
fn test_imp_058_token_batch() {
    use crate::gpu::TokenBatch;

    // Test 1: Create token batch with capacity
    let mut batch = TokenBatch::new(4);
    assert_eq!(batch.capacity(), 4, "IMP-058: Batch should have capacity 4");
    assert_eq!(batch.len(), 0, "IMP-058: New batch should be empty");
    assert!(!batch.is_full(), "IMP-058: New batch should not be full");

    // Test 2: Push tokens and check state
    assert!(
        batch.push(100).is_none(),
        "IMP-058: First push should not return batch"
    );
    assert_eq!(batch.len(), 1, "IMP-058: Batch should have 1 token");

    assert!(
        batch.push(101).is_none(),
        "IMP-058: Second push should not return batch"
    );
    assert!(
        batch.push(102).is_none(),
        "IMP-058: Third push should not return batch"
    );
    assert_eq!(batch.len(), 3, "IMP-058: Batch should have 3 tokens");

    // Test 3: Push final token returns full batch
    let full_batch = batch.push(103);
    assert!(
        full_batch.is_some(),
        "IMP-058: Fourth push should return full batch"
    );
    let tokens = full_batch.expect("test");
    assert_eq!(
        tokens,
        vec![100, 101, 102, 103],
        "IMP-058: Batch should contain all tokens"
    );
    assert_eq!(
        batch.len(),
        0,
        "IMP-058: After returning, batch should be empty"
    );

    // Test 4: Flush partial batch
    batch.push(200);
    batch.push(201);
    let partial = batch.flush();
    assert_eq!(
        partial,
        vec![200, 201],
        "IMP-058: Flush should return partial batch"
    );
    assert_eq!(
        batch.len(),
        0,
        "IMP-058: After flush, batch should be empty"
    );

    // Test 5: Flush empty batch returns empty vec
    let empty = batch.flush();
    assert!(
        empty.is_empty(),
        "IMP-058: Flush empty batch should return empty vec"
    );
}

/// IMP-059: Speculative token buffer
/// Target: Buffer for speculative decoding candidates
#[test]
#[cfg(feature = "gpu")]
fn test_imp_059_speculative_buffer() {
    use crate::gpu::SpeculativeBuffer;

    // Test 1: Create speculative buffer with capacity
    let mut buffer = SpeculativeBuffer::new(8);
    assert_eq!(
        buffer.capacity(),
        8,
        "IMP-059: Buffer should have capacity 8"
    );
    assert_eq!(buffer.len(), 0, "IMP-059: New buffer should be empty");

    // Test 2: Add candidates with confidence scores
    buffer.add_candidate(100, 0.95);
    buffer.add_candidate(101, 0.85);
    buffer.add_candidate(102, 0.75);
    assert_eq!(buffer.len(), 3, "IMP-059: Buffer should have 3 candidates");

    // Test 3: Verify candidates against actual tokens (all match)
    let actual_tokens = vec![100, 101, 102];
    let (accepted, rejected_at) = buffer.verify(&actual_tokens);
    assert_eq!(accepted, 3, "IMP-059: All 3 candidates should be accepted");
    assert!(
        rejected_at.is_none(),
        "IMP-059: No rejection point when all match"
    );

    // Test 4: Verify with mismatch (clear buffer first)
    buffer.reject(); // Clear previous candidates
    buffer.add_candidate(200, 0.90);
    buffer.add_candidate(201, 0.80);
    buffer.add_candidate(202, 0.70);
    let actual_with_mismatch = vec![200, 201, 999]; // 999 doesn't match 202
    let (accepted2, rejected_at2) = buffer.verify(&actual_with_mismatch);
    assert_eq!(accepted2, 2, "IMP-059: Only first 2 should be accepted");
    assert_eq!(rejected_at2, Some(2), "IMP-059: Rejection at index 2");

    // Test 5: Accept/reject resolution (clear buffer first)
    buffer.reject();
    buffer.add_candidate(300, 0.95);
    buffer.add_candidate(301, 0.85);
    buffer.accept(1); // Accept first candidate
    assert_eq!(
        buffer.len(),
        1,
        "IMP-059: After accept(1), 1 candidate remains"
    );

    buffer.reject(); // Reject remaining
    assert_eq!(
        buffer.len(),
        0,
        "IMP-059: After reject, buffer should be empty"
    );
}

/// IMP-060: Batch scheduling coordinator
/// Target: Coordinate batched inference scheduling
#[test]
#[cfg(feature = "gpu")]
fn test_imp_060_batch_scheduler() {
    use crate::gpu::InferenceBatchScheduler;

    // Test 1: Create batch scheduler
    let mut scheduler = InferenceBatchScheduler::new();
    assert_eq!(
        scheduler.pending_count(),
        0,
        "IMP-060: New scheduler has no pending"
    );
    assert_eq!(
        scheduler.completed_count(),
        0,
        "IMP-060: New scheduler has no completed"
    );

    // Test 2: Submit batches
    let batch_id_1 = scheduler.submit(vec![100, 101, 102]);
    let batch_id_2 = scheduler.submit(vec![200, 201]);
    assert_eq!(
        scheduler.pending_count(),
        2,
        "IMP-060: Should have 2 pending batches"
    );
    assert!(
        batch_id_1 != batch_id_2,
        "IMP-060: Batch IDs should be unique"
    );

    // Test 3: Poll for completed (none yet since we need to mark complete)
    assert!(
        scheduler.poll().is_none(),
        "IMP-060: No batches completed yet"
    );

    // Test 4: Mark batch as complete with results
    scheduler.complete(batch_id_1, vec![1000, 1001, 1002]);
    assert_eq!(
        scheduler.completed_count(),
        1,
        "IMP-060: Should have 1 completed"
    );
    assert_eq!(
        scheduler.pending_count(),
        1,
        "IMP-060: Should have 1 pending"
    );

    // Test 5: Poll returns completed batch
    let completed = scheduler.poll();
    assert!(completed.is_some(), "IMP-060: Should get completed batch");
    let (id, results) = completed.expect("test");
    assert_eq!(id, batch_id_1, "IMP-060: Should get batch_id_1");
    assert_eq!(
        results,
        vec![1000, 1001, 1002],
        "IMP-060: Should get correct results"
    );

    // Test 6: Drain all completed
    scheduler.complete(batch_id_2, vec![2000, 2001]);
    let all_completed = scheduler.drain();
    assert_eq!(
        all_completed.len(),
        1,
        "IMP-060: Drain should return 1 batch"
    );
    assert_eq!(
        scheduler.completed_count(),
        0,
        "IMP-060: After drain, no completed"
    );
}

// =========================================================================
// M26: Async I/O & Event-Driven Processing Tests (Phase 17)
// =========================================================================

/// IMP-061: Async request queue
/// Tests non-blocking request submission and retrieval with backpressure.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_061_async_request_queue() {
    use crate::gpu::AsyncRequestQueue;

    // Test 1: Create queue with capacity
    let mut queue: AsyncRequestQueue<String> = AsyncRequestQueue::new(3);
    assert_eq!(queue.capacity(), 3, "IMP-061: Queue capacity should be 3");
    assert!(queue.is_empty(), "IMP-061: New queue should be empty");
    assert!(!queue.is_full(), "IMP-061: New queue should not be full");
    assert_eq!(queue.len(), 0, "IMP-061: New queue length should be 0");

    // Test 2: Push items
    assert!(
        queue.try_push("request1".to_string()),
        "IMP-061: Should push first item"
    );
    assert!(
        queue.try_push("request2".to_string()),
        "IMP-061: Should push second item"
    );
    assert_eq!(queue.len(), 2, "IMP-061: Queue should have 2 items");
    assert!(!queue.is_empty(), "IMP-061: Queue should not be empty");

    // Test 3: Fill to capacity
    assert!(
        queue.try_push("request3".to_string()),
        "IMP-061: Should push third item"
    );
    assert!(queue.is_full(), "IMP-061: Queue should be full");
    assert!(
        !queue.try_push("request4".to_string()),
        "IMP-061: Should reject when full"
    );

    // Test 4: Pop items (FIFO order)
    let item = queue.try_pop();
    assert!(item.is_some(), "IMP-061: Should pop item");
    assert_eq!(
        item.expect("test"),
        "request1",
        "IMP-061: Should pop in FIFO order"
    );
    assert!(
        !queue.is_full(),
        "IMP-061: Queue should not be full after pop"
    );

    // Test 5: Pop remaining
    assert_eq!(
        queue.try_pop(),
        Some("request2".to_string()),
        "IMP-061: Pop second"
    );
    assert_eq!(
        queue.try_pop(),
        Some("request3".to_string()),
        "IMP-061: Pop third"
    );
    assert!(queue.is_empty(), "IMP-061: Queue should be empty");
    assert!(
        queue.try_pop().is_none(),
        "IMP-061: Pop from empty returns None"
    );
}

/// IMP-062: Event notifier for completion
/// Tests callback-based notification of inference completion.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_062_event_notifier() {
    use crate::gpu::InferenceEventNotifier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Test 1: Create notifier
    let mut notifier = InferenceEventNotifier::new();
    assert_eq!(
        notifier.handler_count(),
        0,
        "IMP-062: New notifier has no handlers"
    );

    // Test 2: Register handlers
    let counter1 = Arc::new(AtomicUsize::new(0));
    let counter1_clone = counter1.clone();
    notifier.register(Box::new(move |_request_id, _tokens| {
        counter1_clone.fetch_add(1, Ordering::SeqCst);
    }));
    assert_eq!(
        notifier.handler_count(),
        1,
        "IMP-062: Should have 1 handler"
    );

    let counter2 = Arc::new(AtomicUsize::new(0));
    let counter2_clone = counter2.clone();
    notifier.register(Box::new(move |_request_id, _tokens| {
        counter2_clone.fetch_add(10, Ordering::SeqCst);
    }));
    assert_eq!(
        notifier.handler_count(),
        2,
        "IMP-062: Should have 2 handlers"
    );

    // Test 3: Notify triggers all handlers
    notifier.notify(1, &[100, 101, 102]);
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        1,
        "IMP-062: Handler 1 should be called"
    );
    assert_eq!(
        counter2.load(Ordering::SeqCst),
        10,
        "IMP-062: Handler 2 should be called"
    );

    // Test 4: Multiple notifications
    notifier.notify(2, &[200]);
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        2,
        "IMP-062: Handler 1 called twice"
    );
    assert_eq!(
        counter2.load(Ordering::SeqCst),
        20,
        "IMP-062: Handler 2 called twice"
    );

    // Test 5: Clear handlers
    notifier.clear();
    assert_eq!(
        notifier.handler_count(),
        0,
        "IMP-062: After clear, no handlers"
    );
    notifier.notify(3, &[300]); // Should not crash, just no-op
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        2,
        "IMP-062: Counter unchanged after clear"
    );
}

/// IMP-063: Timeout manager for requests
/// Tests deadline-based request timeout handling.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_063_timeout_manager() {
    use crate::gpu::TimeoutManager;
    use std::time::{Duration, Instant};

    // Test 1: Create timeout manager
    let mut manager = TimeoutManager::new();
    assert_eq!(
        manager.active_count(),
        0,
        "IMP-063: New manager has no active timeouts"
    );

    // Test 2: Register timeouts with different deadlines
    let now = Instant::now();
    let short_deadline = now + Duration::from_millis(10);
    let long_deadline = now + Duration::from_millis(1000);

    manager.register(1, short_deadline);
    manager.register(2, long_deadline);
    assert_eq!(
        manager.active_count(),
        2,
        "IMP-063: Should have 2 active timeouts"
    );

    // Test 3: Check for expired (wait for short timeout to expire)
    std::thread::sleep(Duration::from_millis(20));
    let expired = manager.check_expired();
    assert_eq!(expired.len(), 1, "IMP-063: Should have 1 expired timeout");
    assert_eq!(expired[0], 1, "IMP-063: Request 1 should be expired");
    assert_eq!(
        manager.active_count(),
        1,
        "IMP-063: Should have 1 active after check"
    );

    // Test 4: Remove timeout manually
    manager.remove(2);
    assert_eq!(manager.active_count(), 0, "IMP-063: No active after remove");

    // Test 5: Check expired on empty returns empty vec
    let expired = manager.check_expired();
    assert!(expired.is_empty(), "IMP-063: No expired when empty");
}
