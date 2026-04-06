    use super::*;

    struct TestBrick {
        name: &'static str,
    }

    impl Brick for TestBrick {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[BrickAssertion::TextVisible]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![BrickAssertion::TextVisible],
                failed: vec![],
                verification_time: Duration::from_micros(100),
            }
        }

        fn to_html(&self) -> String {
            format!("<div>{}</div>", self.name)
        }

        fn to_css(&self) -> String {
            ".test { }".into()
        }
    }

    #[test]
    fn test_worker_id() {
        let id = WorkerId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(format!("{id}"), "worker-42");
    }

    #[test]
    fn test_backend_availability() {
        assert!(Backend::Cpu.is_available());
        assert!(Backend::Simd.is_available());
        // GPU/Remote depend on feature flags
    }

    #[test]
    fn test_backend_performance() {
        assert!(Backend::Gpu.performance_estimate() > Backend::Simd.performance_estimate());
        assert!(Backend::Simd.performance_estimate() > Backend::Cpu.performance_estimate());
    }

    #[test]
    fn test_distributed_brick_creation() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner)
            .with_backend(Backend::Gpu)
            .with_data_dependencies(vec!["weights".into(), "biases".into()])
            .with_preferred_worker(WorkerId::new(1));

        assert_eq!(distributed.backend(), Backend::Gpu);
        assert_eq!(distributed.data_dependencies().len(), 2);
        assert_eq!(distributed.preferred_worker(), Some(WorkerId::new(1)));
        assert_eq!(distributed.brick_name(), "Test");
    }

    #[test]
    fn test_distributed_brick_implements_brick() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);

        // Verify it implements Brick trait
        assert!(distributed.verify().is_valid());
        assert_eq!(distributed.budget().total_ms, 16);
    }

    #[test]
    fn test_task_spec() {
        let inner = TestBrick { name: "TestTask" };
        let distributed = DistributedBrick::new(inner)
            .with_backend(Backend::Simd)
            .with_data_dependencies(vec!["model".into()]);

        let spec = distributed.to_task_spec();
        assert_eq!(spec.brick_name, "TestTask");
        assert_eq!(spec.backend, Backend::Simd);
        assert_eq!(spec.data_dependencies, vec!["model"]);
    }

    #[test]
    fn test_brick_input_output() {
        let input = BrickInput::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(input.element_count(), 4);
        assert_eq!(input.size_bytes(), 16);

        let output = BrickOutput::new(vec![5.0, 6.0], vec![2]);
        assert_eq!(output.size_bytes(), 8);
    }

    #[test]
    fn test_data_tracker() {
        let tracker = BrickDataTracker::new();

        // Track some data
        tracker.track_data("model_weights", WorkerId::new(1), 1024);
        tracker.track_data("model_weights", WorkerId::new(2), 1024);
        tracker.track_data("biases", WorkerId::new(1), 256);

        // Check workers
        let workers = tracker.get_workers_for_data("model_weights");
        assert_eq!(workers.len(), 2);

        // Calculate affinity
        let affinity = tracker.calculate_affinity(&["model_weights".into(), "biases".into()]);
        assert!(affinity.get(&WorkerId::new(1)).unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_data_tracker_find_best_worker() {
        let tracker = BrickDataTracker::new();

        let brick = TestBrick { name: "MelBrick" };
        tracker.track_weights("MelBrick", WorkerId::new(5));

        let best = tracker.find_best_worker(&brick);
        assert_eq!(best, Some(WorkerId::new(5)));
    }

    #[test]
    fn test_backend_selector() {
        let selector = BackendSelector::new()
            .with_gpu_threshold(1000)
            .with_simd_threshold(100);

        // Small input -> CPU
        assert_eq!(selector.select(50, true), Backend::Cpu);

        // Medium input -> SIMD
        assert_eq!(selector.select(500, true), Backend::Simd);

        // Large input with GPU -> GPU
        assert_eq!(selector.select(5000, true), Backend::Gpu);

        // Large input without GPU -> SIMD
        assert_eq!(selector.select(5000, false), Backend::Simd);
    }

    #[test]
    fn test_multi_executor() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = executor.execute(&brick, input);
        assert!(result.is_ok());

        let output = result.expect("execution should succeed");
        assert_eq!(output.data.len(), 3);
        assert!(output.metrics.execution_time >= Duration::ZERO);
    }

    #[test]
    fn test_brick_coordinator() {
        let coordinator = BrickCoordinator::new();

        // Subscribe to events
        let sub = coordinator.subscribe_brick("MyBrick");

        // Broadcast event
        coordinator.broadcast_state_change("MyBrick", "loaded");

        // Check subscription received message
        assert!(sub.has_messages());
        let messages = sub.drain();
        assert_eq!(messages.len(), 1);
        matches!(&messages[0], BrickMessage::StateChange { brick_name, .. } if brick_name == "MyBrick");
    }

    #[test]
    fn test_coordinator_weight_broadcast() {
        let coordinator = BrickCoordinator::new();

        let sub = coordinator.subscribe("brick/Encoder/weights");
        coordinator.broadcast_weights("Encoder", vec![1, 2, 3, 4]);

        let messages = sub.drain();
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            BrickMessage::WeightUpdate {
                brick_name,
                weights,
                version,
            } => {
                assert_eq!(brick_name, "Encoder");
                assert_eq!(weights, &vec![1, 2, 3, 4]);
                assert_eq!(*version, 0);
            }
            _ => panic!("Expected WeightUpdate message"),
        }
    }

    #[test]
    fn test_subscription_topic() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("my/topic");
        assert_eq!(sub.topic(), "my/topic");
    }

    #[test]
    fn test_execution_metrics() {
        let metrics = ExecutionMetrics::new(Duration::from_millis(50), Backend::Gpu);
        assert_eq!(metrics.execution_time, Duration::from_millis(50));
        assert_eq!(metrics.backend, Backend::Gpu);
        assert!(metrics.worker_id.is_none());
    }

    // ========================================================================
    // Work-Stealing Scheduler Tests (Phase 10e)
    // ========================================================================

    #[test]
    fn test_work_stealing_task() {
        let spec = TaskSpec {
            brick_name: "TestBrick".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "input_key".into()).with_priority(10);

        assert_eq!(task.id, 1);
        assert_eq!(task.priority, 10);
        assert_eq!(task.input_key, "input_key");
        assert!(task.age() >= Duration::ZERO);
    }

    #[test]
    fn test_worker_queue_basic() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "key".into());
        queue.push(task);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let popped = queue.pop();
        assert!(popped.is_some());
        assert!(queue.is_empty());
    }

    #[test]
    fn test_worker_queue_priority_ordering() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        // Push tasks with different priorities
        for i in 0..5 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: None,
            };
            let task = WorkStealingTask::new(i as u64, spec, "key".into()).with_priority(i);
            queue.push(task);
        }

        // Pop should return highest priority first
        let task = queue.pop().unwrap();
        assert_eq!(task.priority, 4);

        let task = queue.pop().unwrap();
        assert_eq!(task.priority, 3);
    }

    #[test]
    fn test_worker_queue_steal() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        // Push 3 tasks with priorities 0, 1, 2
        for i in 0..3 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: None,
            };
            let task = WorkStealingTask::new(i as u64, spec, "key".into()).with_priority(i);
            queue.push(task);
        }

        // Steal takes from front (lowest priority after sort)
        let stolen = queue.steal().unwrap();
        assert_eq!(stolen.priority, 0);
        assert_eq!(queue.stolen_count(), 1);

        // Queue still has 2 tasks
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_work_stealing_scheduler_basic() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        // Register workers
        let _q1 = scheduler.register_worker(WorkerId::new(1));
        let _q2 = scheduler.register_worker(WorkerId::new(2));

        let stats = scheduler.stats();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 0);
    }

    #[test]
    fn test_work_stealing_scheduler_submit() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit(spec, "input".into());
        assert_eq!(task_id, 0);

        let stats = scheduler.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_pending, 1);
    }

    #[test]
    fn test_work_stealing_scheduler_get_work() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit task preferring worker 1
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: Some(WorkerId::new(1)),
        };
        scheduler.submit(spec, "input".into());

        // Worker 1 should get the task
        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());

        // Worker 2 has nothing to get (or steal since queue is now empty)
        let task = scheduler.get_work(WorkerId::new(2));
        assert!(task.is_none());
    }

    #[test]
    fn test_work_stealing_scheduler_steal() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit 3 tasks to worker 1
        for i in 0..3 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: Some(WorkerId::new(1)),
            };
            scheduler.submit(spec, format!("input{}", i));
        }

        // Worker 2 should be able to steal a task
        let stolen = scheduler.get_work(WorkerId::new(2));
        assert!(stolen.is_some());

        let stats = scheduler.stats();
        assert_eq!(stats.total_stolen, 1);
        assert_eq!(stats.total_pending, 2); // 3 submitted - 1 stolen
    }

    #[test]
    fn test_work_stealing_scheduler_locality() {
        let tracker = Arc::new(BrickDataTracker::new());

        // Track data on worker 1
        tracker.track_data("model_weights", WorkerId::new(1), 1024);

        let scheduler = WorkStealingScheduler::new(Arc::clone(&tracker));
        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit task with data dependency - should go to worker 1
        let spec = TaskSpec {
            brick_name: "MelBrick".into(),
            backend: Backend::Cpu,
            data_dependencies: vec!["model_weights".into()],
            preferred_worker: None,
        };
        scheduler.submit(spec, "audio_input".into());

        // Worker 1 should have the task
        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());
        assert_eq!(task.unwrap().spec.brick_name, "MelBrick");
    }

    #[test]
    fn test_scheduler_stats() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit some tasks
        for i in 0..5 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: if i % 2 == 0 {
                    Some(WorkerId::new(1))
                } else {
                    Some(WorkerId::new(2))
                },
            };
            scheduler.submit(spec, format!("input{}", i));
        }

        let stats = scheduler.stats();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 5);
        assert_eq!(stats.total_pending, 5);
        assert_eq!(stats.workers.len(), 2);
    }

    // ========================================================================
    // Additional comprehensive tests for 95%+ coverage
    // ========================================================================

    #[test]
    fn test_worker_id_copy_clone() {
        let id = WorkerId::new(123);
        let cloned = id;
        assert_eq!(id, cloned);
        assert_eq!(id.0, 123);
    }

    #[test]
    fn test_worker_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WorkerId::new(1));
        set.insert(WorkerId::new(2));
        set.insert(WorkerId::new(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_backend_default() {
        let backend = Backend::default();
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_backend_remote_not_available() {
        assert!(!Backend::Remote.is_available());
    }

    #[test]
    fn test_backend_performance_remote() {
        assert_eq!(Backend::Remote.performance_estimate(), 5);
        assert_eq!(Backend::Cpu.performance_estimate(), 10);
    }

    #[test]
    fn test_brick_input_default() {
        let input = BrickInput::default();
        assert!(input.data.is_empty());
        assert!(input.shape.is_empty());
        assert!(input.metadata.is_empty());
    }

    #[test]
    fn test_brick_input_with_metadata() {
        let input = BrickInput::new(vec![1.0], vec![1])
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");
        assert_eq!(input.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(input.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_brick_output_default() {
        let output = BrickOutput::default();
        assert!(output.data.is_empty());
        assert!(output.shape.is_empty());
    }

    #[test]
    fn test_execution_metrics_default() {
        let metrics = ExecutionMetrics::default();
        assert_eq!(metrics.execution_time, Duration::ZERO);
        assert_eq!(metrics.backend, Backend::Cpu);
        assert!(metrics.worker_id.is_none());
        assert!(metrics.transfer_time.is_none());
    }

    #[test]
    fn test_distributed_brick_inner() {
        let inner = TestBrick { name: "Inner" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.inner().brick_name(), "Inner");
    }

    #[test]
    fn test_distributed_brick_inner_mut() {
        let inner = TestBrick { name: "Inner" };
        let mut distributed = DistributedBrick::new(inner);
        let _ = distributed.inner_mut();
        // Just verify we can get mutable reference
    }

    #[test]
    fn test_distributed_brick_to_html() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.to_html(), "<div>Test</div>");
    }

    #[test]
    fn test_distributed_brick_to_css() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.to_css(), ".test { }");
    }

    #[test]
    fn test_distributed_brick_assertions() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.assertions().len(), 1);
    }

    #[test]
    fn test_task_spec_clone() {
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Gpu,
            data_dependencies: vec!["dep1".into()],
            preferred_worker: Some(WorkerId::new(5)),
        };
        let cloned = spec.clone();
        assert_eq!(spec.brick_name, cloned.brick_name);
        assert_eq!(spec.backend, cloned.backend);
    }

    #[test]
    fn test_brick_data_tracker_default() {
        let tracker = BrickDataTracker::default();
        assert_eq!(tracker.total_data_size(), 0);
    }

    #[test]
    fn test_brick_data_tracker_remove_data() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("data1", WorkerId::new(1), 100);
        tracker.track_data("data1", WorkerId::new(2), 100);

        let workers = tracker.get_workers_for_data("data1");
        assert_eq!(workers.len(), 2);

        tracker.remove_data("data1", WorkerId::new(1));
        let workers = tracker.get_workers_for_data("data1");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0], WorkerId::new(2));
    }

    #[test]
    fn test_brick_data_tracker_total_size() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("data1", WorkerId::new(1), 100);
        tracker.track_data("data2", WorkerId::new(1), 200);
        assert_eq!(tracker.total_data_size(), 300);
    }

    #[test]
    fn test_brick_data_tracker_get_nonexistent() {
        let tracker = BrickDataTracker::new();
        let workers = tracker.get_workers_for_data("nonexistent");
        assert!(workers.is_empty());
    }

    #[test]
    fn test_brick_data_tracker_calculate_affinity_empty() {
        let tracker = BrickDataTracker::new();
        let affinity = tracker.calculate_affinity(&["nonexistent".into()]);
        assert!(affinity.is_empty());
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_no_weights() {
        let tracker = BrickDataTracker::new();
        let brick = TestBrick { name: "NoBrick" };
        let best = tracker.find_best_worker(&brick);
        assert!(best.is_none());
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_distributed_preferred() {
        let tracker = BrickDataTracker::new();
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_preferred_worker(WorkerId::new(42));

        let best = tracker.find_best_worker_for_distributed(&distributed);
        assert_eq!(best, Some(WorkerId::new(42)));
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_distributed_affinity() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("dep1", WorkerId::new(5), 100);

        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_data_dependencies(vec!["dep1".into()]);

        let best = tracker.find_best_worker_for_distributed(&distributed);
        assert_eq!(best, Some(WorkerId::new(5)));
    }

    #[test]
    fn test_backend_selector_default() {
        let selector = BackendSelector::default();
        // Default thresholds
        assert_eq!(selector.select(50, true), Backend::Cpu);
    }

    #[test]
    fn test_backend_selector_cpu_max_threshold() {
        let selector = BackendSelector::new()
            .with_cpu_max_threshold(100)
            .with_simd_threshold(50);
        // Over cpu_max_threshold but Remote not available, so falls through to GPU/SIMD/CPU selection
        // Since 200 >= simd_threshold (50), returns SIMD
        let backend = selector.select(200, false);
        assert_eq!(backend, Backend::Simd);

        // Below simd_threshold returns CPU
        let backend = selector.select(10, false);
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_backend_selector_select_for_brick() {
        let selector = BackendSelector::new();
        let backend = selector.select_for_brick(50, 100, true);
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_multi_executor_with_selector() {
        let tracker = Arc::new(BrickDataTracker::new());
        let selector = BackendSelector::new().with_simd_threshold(1);
        let executor = MultiBrickExecutor::new(tracker).with_selector(selector);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0], vec![2]);
        let result = executor.execute(&brick, input);
        assert!(result.is_ok());
        // With threshold 1, should use SIMD
        assert_eq!(result.unwrap().metrics.backend, Backend::Simd);
    }

    #[test]
    fn test_multi_executor_with_gpu_available() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker).with_gpu_available(true);
        let _ = executor.data_tracker();
    }

    #[test]
    fn test_multi_executor_execute_distributed() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker);

        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Cpu);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_executor_execute_simd() {
        let tracker = Arc::new(BrickDataTracker::new());
        let selector = BackendSelector::new().with_simd_threshold(1);
        let executor = MultiBrickExecutor::new(tracker).with_selector(selector);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0], vec![2]);

        let result = executor.execute(&brick, input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().metrics.backend, Backend::Simd);
    }

    #[test]
    fn test_multi_executor_execute_gpu_unavailable() {
        let tracker = Arc::new(BrickDataTracker::new());
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Gpu);
        let executor = MultiBrickExecutor::new(tracker).with_gpu_available(false);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_executor_execute_remote_unavailable() {
        let tracker = Arc::new(BrickDataTracker::new());
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Remote);
        let executor = MultiBrickExecutor::new(tracker);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_err());
    }

    #[test]
    fn test_subscription_drain_empty() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("test/topic");
        let messages = sub.drain();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_subscription_has_messages_false() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("test/topic");
        assert!(!sub.has_messages());
    }

    #[test]
    fn test_brick_coordinator_default() {
        let coordinator = BrickCoordinator::default();
        let id = coordinator.next_request_id();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_brick_coordinator_next_request_id() {
        let coordinator = BrickCoordinator::new();
        assert_eq!(coordinator.next_request_id(), 0);
        assert_eq!(coordinator.next_request_id(), 1);
        assert_eq!(coordinator.next_request_id(), 2);
    }

    #[test]
    fn test_brick_coordinator_publish_no_subscribers() {
        let coordinator = BrickCoordinator::new();
        // Should not panic even with no subscribers
        coordinator.publish(
            "nonexistent/topic",
            BrickMessage::StateChange {
                brick_name: "Test".into(),
                event: "test".into(),
            },
        );
    }

    #[test]
    fn test_brick_message_execution_request() {
        let msg = BrickMessage::ExecutionRequest {
            brick_name: "Test".into(),
            input_key: "key".into(),
            request_id: 42,
        };
        match msg {
            BrickMessage::ExecutionRequest {
                brick_name,
                input_key,
                request_id,
            } => {
                assert_eq!(brick_name, "Test");
                assert_eq!(input_key, "key");
                assert_eq!(request_id, 42);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_brick_message_execution_result() {
        let msg = BrickMessage::ExecutionResult {
            request_id: 42,
            output_key: "out".into(),
            success: true,
        };
        match msg {
            BrickMessage::ExecutionResult {
                request_id,
                output_key,
                success,
            } => {
                assert_eq!(request_id, 42);
                assert_eq!(output_key, "out");
                assert!(success);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_work_stealing_task_clone() {
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "key".into());
        let cloned = task.clone();
        assert_eq!(task.id, cloned.id);
    }

    #[test]
    fn test_worker_queue_worker_id() {
        let queue = WorkerQueue::new(WorkerId::new(42));
        assert_eq!(queue.worker_id(), WorkerId::new(42));
    }

    #[test]
    fn test_worker_queue_completed_count() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert_eq!(queue.completed_count(), 0);
        queue.mark_completed();
        assert_eq!(queue.completed_count(), 1);
        queue.mark_completed();
        assert_eq!(queue.completed_count(), 2);
    }

    #[test]
    fn test_worker_queue_pop_empty() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_worker_queue_steal_empty() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert!(queue.steal().is_none());
    }

    #[test]
    fn test_scheduler_unregister_worker() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        assert_eq!(scheduler.stats().worker_count, 1);

        scheduler.unregister_worker(WorkerId::new(1));
        assert_eq!(scheduler.stats().worker_count, 0);
    }

    #[test]
    fn test_scheduler_submit_no_workers() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit(spec, "input".into());
        assert_eq!(task_id, 0);
        // Task submitted but no workers to receive it
        assert_eq!(scheduler.stats().total_submitted, 1);
    }

    #[test]
    fn test_scheduler_submit_priority() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit_priority(spec, "input".into(), 100);
        assert_eq!(task_id, 0);

        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());
        assert_eq!(task.unwrap().priority, 100);
    }

    #[test]
    fn test_scheduler_get_work_unregistered_worker() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        // Try to get work for worker that doesn't exist
        let task = scheduler.get_work(WorkerId::new(999));
        assert!(task.is_none());
    }

    #[test]
    fn test_scheduler_data_tracker_accessor() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(Arc::clone(&tracker));

        let _ = scheduler.data_tracker();
    }

    #[test]
    fn test_worker_stats_fields() {
        let stats = WorkerStats {
            worker_id: WorkerId::new(1),
            queue_length: 5,
            completed: 10,
            stolen_from: 2,
        };
        assert_eq!(stats.worker_id, WorkerId::new(1));
        assert_eq!(stats.queue_length, 5);
        assert_eq!(stats.completed, 10);
        assert_eq!(stats.stolen_from, 2);
    }

    #[test]
    fn test_scheduler_stats_fields() {
        let stats = SchedulerStats {
            worker_count: 2,
            total_submitted: 10,
            total_pending: 5,
            total_completed: 4,
            total_stolen: 1,
            workers: vec![],
        };
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 10);
        assert_eq!(stats.total_pending, 5);
        assert_eq!(stats.total_completed, 4);
        assert_eq!(stats.total_stolen, 1);
    }

    #[test]
    fn test_data_location_clone() {
        let loc = DataLocation {
            key: "test".into(),
            workers: vec![WorkerId::new(1)],
            size_bytes: 100,
            last_access: Instant::now(),
        };
        let cloned = loc.clone();
        assert_eq!(loc.key, cloned.key);
    }

    #[test]
    fn test_track_data_updates_existing() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("key", WorkerId::new(1), 100);
        tracker.track_data("key", WorkerId::new(1), 200); // Same worker again

        let workers = tracker.get_workers_for_data("key");
        assert_eq!(workers.len(), 1); // Should not duplicate
    }
