#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::uninlined_format_args,
    clippy::collection_is_never_read
)]
//! Property-based tests for repartir (Certeza Tier 2)
//!
//! These tests use proptest to verify invariants hold across
//! arbitrary inputs.

use proptest::prelude::*;
use repartir::scheduler::Scheduler;
use repartir::task::{Backend, Priority, Task};

// Property: All submitted tasks can be retrieved
proptest! {
    #[test]
    fn prop_scheduler_submit_retrieve(
        num_tasks in 1..100usize
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let scheduler = Scheduler::with_capacity(1000);
            let mut task_ids = Vec::new();

            // Submit tasks
            for i in 0..num_tasks {
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg(format!("task_{}", i))
                    .backend(Backend::Cpu)
                    .build()
                    .unwrap();

                let task_id = scheduler.submit(task).await.unwrap();
                task_ids.push(task_id);
            }

            // Verify pending count
            assert_eq!(scheduler.pending_count().await, num_tasks);

            // Retrieve all tasks
            for _ in 0..num_tasks {
                let task = scheduler.next_task().await;
                assert!(task.is_some());
            }

            // Queue should be empty
            assert_eq!(scheduler.pending_count().await, 0);
        });
    }
}

// Property: Capacity invariant (queue never exceeds max)
proptest! {
    #[test]
    fn prop_scheduler_capacity_invariant(
        capacity in 1..50usize,
        attempts in 1..100usize
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let scheduler = Scheduler::with_capacity(capacity);
            let mut submitted = 0;

            for i in 0..attempts {
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg(format!("task_{}", i))
                    .backend(Backend::Cpu)
                    .build()
                    .unwrap();

                if scheduler.submit(task).await.is_ok() {
                    submitted += 1;
                }

                // Invariant: pending count never exceeds capacity
                assert!(scheduler.pending_count().await <= capacity);
            }

            // Should have submitted at least capacity tasks
            assert!(submitted >= capacity.min(attempts));
        });
    }
}

// Property: Priority ordering (high priority before low)
proptest! {
    #[test]
    fn prop_scheduler_priority_ordering(
        num_high in 1..20usize,
        num_normal in 1..20usize,
        num_low in 1..20usize
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let scheduler = Scheduler::with_capacity(1000);

            // Submit mixed priorities
            for i in 0..num_low {
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg(format!("low_{}", i))
                    .backend(Backend::Cpu)
                    .priority(Priority::Low)
                    .build()
                    .unwrap();
                scheduler.submit(task).await.unwrap();
            }

            for i in 0..num_high {
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg(format!("high_{}", i))
                    .backend(Backend::Cpu)
                    .priority(Priority::High)
                    .build()
                    .unwrap();
                scheduler.submit(task).await.unwrap();
            }

            for i in 0..num_normal {
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg(format!("normal_{}", i))
                    .backend(Backend::Cpu)
                    .priority(Priority::Normal)
                    .build()
                    .unwrap();
                scheduler.submit(task).await.unwrap();
            }

            // First num_high tasks should be high priority
            for _ in 0..num_high {
                let task = scheduler.next_task().await.unwrap();
                assert_eq!(task.priority(), Priority::High);
            }

            // Next num_normal tasks should be normal priority
            for _ in 0..num_normal {
                let task = scheduler.next_task().await.unwrap();
                assert_eq!(task.priority(), Priority::Normal);
            }

            // Remaining should be low priority
            for _ in 0..num_low {
                let task = scheduler.next_task().await.unwrap();
                assert_eq!(task.priority(), Priority::Low);
            }
        });
    }
}

// Property: Task IDs are unique
proptest! {
    #[test]
    fn prop_task_ids_unique(num_tasks in 2..100usize) {
        use std::collections::HashSet;

        let mut ids = HashSet::new();

        for i in 0..num_tasks {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg(format!("task_{}", i))
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            // Each task ID should be unique
            assert!(ids.insert(task.id()));
        }

        assert_eq!(ids.len(), num_tasks);
    }
}
