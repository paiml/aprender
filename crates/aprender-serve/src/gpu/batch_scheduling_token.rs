
    // ==================== TokenBatch Tests ====================

    #[test]
    fn test_token_batch_new() {
        let batch = TokenBatch::new(8);
        assert_eq!(batch.capacity(), 8);
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_token_batch_push_partial() {
        let mut batch = TokenBatch::new(4);
        assert!(batch.push(1).is_none());
        assert!(batch.push(2).is_none());
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_token_batch_push_full() {
        let mut batch = TokenBatch::new(3);
        batch.push(1);
        batch.push(2);
        let result = batch.push(3);
        assert!(result.is_some());
        assert_eq!(result.expect("expected value"), vec![1, 2, 3]);
        assert!(batch.is_empty()); // Flushed
    }

    #[test]
    fn test_token_batch_flush() {
        let mut batch = TokenBatch::new(10);
        batch.push(10);
        batch.push(20);
        let flushed = batch.flush();
        assert_eq!(flushed, vec![10, 20]);
        assert!(batch.is_empty());
    }

    // ==================== SpeculativeBuffer Tests ====================

    #[test]
    fn test_speculative_buffer_new() {
        let buf = SpeculativeBuffer::new(5);
        assert_eq!(buf.capacity(), 5);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_speculative_buffer_add_candidate() {
        let mut buf = SpeculativeBuffer::new(3);
        buf.add_candidate(100, 0.9);
        buf.add_candidate(101, 0.8);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_speculative_buffer_capacity_limit() {
        let mut buf = SpeculativeBuffer::new(2);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);
        buf.add_candidate(3, 0.7); // Should be ignored
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_speculative_buffer_verify_all_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);
        buf.add_candidate(20, 0.8);
        buf.add_candidate(30, 0.7);

        let (accepted, rejection) = buf.verify(&[10, 20, 30]);
        assert_eq!(accepted, 3);
        assert!(rejection.is_none());
    }

    #[test]
    fn test_speculative_buffer_verify_partial_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);
        buf.add_candidate(20, 0.8);
        buf.add_candidate(30, 0.7);

        let (accepted, rejection) = buf.verify(&[10, 20, 99]);
        assert_eq!(accepted, 2);
        assert_eq!(rejection, Some(2));
    }

    #[test]
    fn test_speculative_buffer_verify_no_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);

        let (accepted, rejection) = buf.verify(&[99]);
        assert_eq!(accepted, 0);
        assert_eq!(rejection, Some(0));
    }

    #[test]
    fn test_speculative_buffer_accept() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);
        buf.add_candidate(3, 0.7);

        buf.accept(2);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_speculative_buffer_accept_all() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);

        buf.accept(10); // More than available
        assert!(buf.is_empty());
    }

    #[test]
    fn test_speculative_buffer_reject() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);

        buf.reject();
        assert!(buf.is_empty());
    }

    // ==================== InferenceBatchScheduler Tests ====================

    #[test]
    fn test_inference_batch_scheduler_new() {
        let scheduler = InferenceBatchScheduler::new();
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.completed_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_default() {
        let scheduler = InferenceBatchScheduler::default();
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_submit() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id1 = scheduler.submit(vec![1, 2, 3]);
        let id2 = scheduler.submit(vec![4, 5]);

        assert_eq!(scheduler.pending_count(), 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_inference_batch_scheduler_complete() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id = scheduler.submit(vec![1, 2, 3]);
        assert_eq!(scheduler.pending_count(), 1);

        scheduler.complete(id, vec![100, 200]);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.completed_count(), 1);
    }

    #[test]
    fn test_inference_batch_scheduler_poll() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id = scheduler.submit(vec![1]);
        scheduler.complete(id, vec![99]);

        let result = scheduler.poll();
        assert!(result.is_some());
        let (batch_id, tokens) = result.expect("expected value");
        assert_eq!(batch_id, id);
        assert_eq!(tokens, vec![99]);
        assert_eq!(scheduler.completed_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_drain() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id1 = scheduler.submit(vec![1]);
        let id2 = scheduler.submit(vec![2]);
        scheduler.complete(id1, vec![10]);
        scheduler.complete(id2, vec![20]);

        let drained = scheduler.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(scheduler.completed_count(), 0);
    }

    // ==================== AsyncRequestQueue Tests ====================

    #[test]
    fn test_async_request_queue_new() {
        let queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(5);
        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
    }

    #[test]
    fn test_async_request_queue_try_push() {
        let mut queue = AsyncRequestQueue::new(3);
        assert!(queue.try_push(1));
        assert!(queue.try_push(2));
        assert!(queue.try_push(3));
        assert!(!queue.try_push(4)); // Queue is full
        assert!(queue.is_full());
    }

    #[test]
    fn test_async_request_queue_try_pop() {
        let mut queue = AsyncRequestQueue::new(3);
        queue.try_push(10);
        queue.try_push(20);

        assert_eq!(queue.try_pop(), Some(10));
        assert_eq!(queue.try_pop(), Some(20));
        assert_eq!(queue.try_pop(), None);
    }

    #[test]
    fn test_async_request_queue_fifo_order() {
        let mut queue = AsyncRequestQueue::new(5);
        for i in 0..5 {
            queue.try_push(i);
        }
        for i in 0..5 {
            assert_eq!(queue.try_pop(), Some(i));
        }
    }

    // ==================== InferenceEventNotifier Tests ====================

    #[test]
    fn test_inference_event_notifier_new() {
        let notifier = InferenceEventNotifier::new();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_default() {
        let notifier = InferenceEventNotifier::default();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_register() {
        let mut notifier = InferenceEventNotifier::new();
        notifier.register(Box::new(|_id, _tokens| {}));
        notifier.register(Box::new(|_id, _tokens| {}));
        assert_eq!(notifier.handler_count(), 2);
    }

    #[test]
    fn test_inference_event_notifier_notify() {
        let mut notifier = InferenceEventNotifier::new();
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        notifier.register(Box::new(move |id, _tokens| {
            counter_clone.fetch_add(id, Ordering::SeqCst);
        }));

        notifier.notify(42, &[1, 2, 3]);
        assert_eq!(counter.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_inference_event_notifier_clear() {
        let mut notifier = InferenceEventNotifier::new();
        notifier.register(Box::new(|_id, _tokens| {}));
        notifier.clear();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_debug() {
        let notifier = InferenceEventNotifier::new();
        let debug_str = format!("{:?}", notifier);
        assert!(debug_str.contains("handler_count"));
    }

    // ==================== TimeoutManager Tests ====================

    #[test]
    fn test_timeout_manager_new() {
        let manager = TimeoutManager::new();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_default() {
        let manager = TimeoutManager::default();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_register() {
        let mut manager = TimeoutManager::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        manager.register(1, deadline);
        manager.register(2, deadline);
        assert_eq!(manager.active_count(), 2);
    }

    #[test]
    fn test_timeout_manager_remove() {
        let mut manager = TimeoutManager::new();
        manager.register(1, Instant::now() + Duration::from_secs(10));
        manager.remove(1);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_check_expired() {
        let mut manager = TimeoutManager::new();
        // Already expired
        manager.register(
            1,
            Instant::now().checked_sub(Duration::from_secs(1)).expect("expected value"),
        );
        // Not expired
        manager.register(2, Instant::now() + Duration::from_secs(60));

        let expired = manager.check_expired();
        assert_eq!(expired, vec![1]);
        assert_eq!(manager.active_count(), 1); // Only non-expired remains
    }

    // ==================== PriorityRequest Tests ====================

    #[test]
    fn test_priority_request_new() {
        let req = PriorityRequest::new(5, "data");
        assert_eq!(req.priority(), 5);
        assert_eq!(req.data(), &"data");
    }

    #[test]
    fn test_priority_request_into_data() {
        let req = PriorityRequest::new(10, vec![1, 2, 3]);
        let data = req.into_data();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_priority_request_clone() {
        let req = PriorityRequest::new(3, "test");
        let cloned = req.clone();
        assert_eq!(cloned.priority(), 3);
    }

    // ==================== PriorityRequestQueue Tests ====================

    #[test]
    fn test_priority_request_queue_new() {
        let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_request_queue_default() {
        let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::default();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_request_queue_enqueue() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(1, "low"));
        queue.enqueue(PriorityRequest::new(5, "high"));
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_priority_request_queue_dequeue_highest() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(1, "low"));
        queue.enqueue(PriorityRequest::new(10, "high"));
        queue.enqueue(PriorityRequest::new(5, "medium"));

        let top = queue.dequeue_highest().expect("top");
        assert_eq!(top.priority(), 10);
        assert_eq!(top.data(), &"high");
    }

    #[test]
    fn test_priority_request_queue_fifo_same_priority() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(5, "first"));
        queue.enqueue(PriorityRequest::new(5, "second"));
        queue.enqueue(PriorityRequest::new(5, "third"));

        // Should return in FIFO order for same priority
        assert_eq!(queue.dequeue_highest().expect("dequeue_highest").data(), &"first");
        assert_eq!(queue.dequeue_highest().expect("dequeue_highest").data(), &"second");
        assert_eq!(queue.dequeue_highest().expect("dequeue_highest").data(), &"third");
    }

    #[test]
    fn test_priority_request_queue_empty() {
        let mut queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
        assert!(queue.dequeue_highest().is_none());
    }

    // ==================== TokenRateLimiter Tests ====================

    #[test]
    fn test_token_rate_limiter_new() {
        let limiter = TokenRateLimiter::new(10.0, 100);
        assert_eq!(limiter.tokens_available(), 100); // Starts full
    }

    #[test]
    fn test_token_rate_limiter_try_acquire() {
        let mut limiter = TokenRateLimiter::new(10.0, 50);
        assert!(limiter.try_acquire(30));
        assert_eq!(limiter.tokens_available(), 20);
        assert!(limiter.try_acquire(20));
        assert_eq!(limiter.tokens_available(), 0);
        assert!(!limiter.try_acquire(1));
    }

    #[test]
    fn test_token_rate_limiter_refill() {
        let mut limiter = TokenRateLimiter::new(1000.0, 100);
        limiter.try_acquire(100);
        assert_eq!(limiter.tokens_available(), 0);

        // Sleep briefly to allow some refill
        std::thread::sleep(Duration::from_millis(50));
        limiter.refill();
        assert!(limiter.tokens_available() > 0);
    }
