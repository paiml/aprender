#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Integration Tests for certeza
//!
//! These tests verify the public API of the certeza crate works correctly
//! when used as a library. Integration tests are in the middle tier of the
//! testing pyramid (~10% of tests) and verify system-level properties.
//!
//! This file contains both:
//! 1. Basic integration tests organized by functionality
//! 2. Real-world scenario-based tests demonstrating practical usage

use certeza::TruenoVec;

// ============================================================================
// Basic Integration Tests
// ============================================================================

#[test]
fn test_trueno_vec_basic_operations() {
    let mut vec = TruenoVec::new();

    // Test empty vector
    assert!(vec.is_empty());
    assert_eq!(vec.len(), 0);

    // Test push and length
    vec.push(10);
    vec.push(20);
    vec.push(30);
    assert_eq!(vec.len(), 3);
    assert!(!vec.is_empty());

    // Test get
    assert_eq!(vec.get(0), Some(&10));
    assert_eq!(vec.get(1), Some(&20));
    assert_eq!(vec.get(2), Some(&30));
    assert_eq!(vec.get(3), None);

    // Test pop
    assert_eq!(vec.pop(), Some(30));
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.pop(), Some(20));
    assert_eq!(vec.pop(), Some(10));
    assert_eq!(vec.pop(), None);
    assert!(vec.is_empty());
}

#[test]
fn test_trueno_vec_with_capacity() {
    let mut vec = TruenoVec::with_capacity(100);
    assert_eq!(vec.capacity(), 100);
    assert_eq!(vec.len(), 0);

    // Should not reallocate for first 100 elements
    for i in 0..100 {
        vec.push(i);
    }
    assert_eq!(vec.capacity(), 100);
    assert_eq!(vec.len(), 100);

    // Should reallocate after 100 elements
    vec.push(100);
    assert_eq!(vec.capacity(), 200);
    assert_eq!(vec.len(), 101);
}

#[test]
fn test_trueno_vec_stack_semantics() {
    // Verify TruenoVec behaves like a stack (LIFO)
    let mut stack = TruenoVec::new();

    // Push sequence
    for i in 1..=5 {
        stack.push(i);
    }

    // Pop should return in reverse order
    assert_eq!(stack.pop(), Some(5));
    assert_eq!(stack.pop(), Some(4));
    assert_eq!(stack.pop(), Some(3));
    assert_eq!(stack.pop(), Some(2));
    assert_eq!(stack.pop(), Some(1));
    assert_eq!(stack.pop(), None);
}

// ============================================================================
// Integration with Standard Rust Types
// ============================================================================

#[test]
fn test_trueno_vec_with_strings() {
    let mut vec = TruenoVec::new();

    vec.push(String::from("hello"));
    vec.push(String::from("world"));
    vec.push(String::from("rust"));

    assert_eq!(vec.len(), 3);
    assert_eq!(vec.get(0).map(String::as_str), Some("hello"));
    assert_eq!(vec.get(1).map(String::as_str), Some("world"));
    assert_eq!(vec.get(2).map(String::as_str), Some("rust"));

    // Test pop with owned strings
    assert_eq!(vec.pop(), Some(String::from("rust")));
    assert_eq!(vec.len(), 2);
}

#[test]
fn test_trueno_vec_with_complex_types() {
    #[derive(Debug, PartialEq, Clone)]
    struct Person {
        name: String,
        age: u32,
    }

    let mut vec = TruenoVec::new();

    vec.push(Person { name: String::from("Alice"), age: 30 });
    vec.push(Person { name: String::from("Bob"), age: 25 });

    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(0), Some(&Person { name: String::from("Alice"), age: 30 }));

    // Test mutation
    if let Some(person) = vec.get_mut(1) {
        person.age = 26;
    }

    assert_eq!(vec.get(1), Some(&Person { name: String::from("Bob"), age: 26 }));
}

// ============================================================================
// Real-World Usage Scenarios
// ============================================================================

#[test]
fn test_trueno_vec_as_buffer() {
    // Simulate using TruenoVec as a buffer for collecting data
    let mut buffer = TruenoVec::with_capacity(256);

    // Collect some data (0..256 to avoid u8 overflow)
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    for i in 0..256 {
        buffer.push(i as u8);
    }

    assert_eq!(buffer.len(), 256);
    assert_eq!(buffer.capacity(), 256);

    // Process and remove data
    let mut sum = 0u64;
    while let Some(value) = buffer.pop() {
        sum += u64::from(value);
    }

    // Sum of 0..256 = (n * (n-1)) / 2 = (256 * 255) / 2
    assert_eq!(sum, (256 * 255) / 2);
    assert!(buffer.is_empty());
}

#[test]
fn test_trueno_vec_interleaved_operations() {
    // Test realistic pattern of interleaved push/pop operations
    let mut vec = TruenoVec::new();

    // Build up
    for i in 0..10 {
        vec.push(i);
    }
    assert_eq!(vec.len(), 10);

    // Remove some
    for _ in 0..5 {
        vec.pop();
    }
    assert_eq!(vec.len(), 5);

    // Add more
    for i in 10..15 {
        vec.push(i);
    }
    assert_eq!(vec.len(), 10);

    // Verify contents
    assert_eq!(vec.get(0), Some(&0));
    assert_eq!(vec.get(4), Some(&4));
    assert_eq!(vec.get(5), Some(&10));
    assert_eq!(vec.get(9), Some(&14));
}

// ============================================================================
// Error Cases and Edge Conditions
// ============================================================================

#[test]
fn test_trueno_vec_out_of_bounds_access() {
    let mut vec = TruenoVec::new();
    vec.push(1);
    vec.push(2);

    // Out of bounds should return None
    assert_eq!(vec.get(2), None);
    assert_eq!(vec.get(100), None);
    assert_eq!(vec.get(usize::MAX), None);
}

#[test]
fn test_trueno_vec_pop_until_empty() {
    let mut vec = TruenoVec::new();
    vec.push(1);
    vec.push(2);

    assert_eq!(vec.pop(), Some(2));
    assert_eq!(vec.pop(), Some(1));
    assert_eq!(vec.pop(), None);
    assert_eq!(vec.pop(), None); // Multiple pops on empty
    assert_eq!(vec.pop(), None);
}

#[test]
fn test_trueno_vec_zero_capacity_initialization() {
    let vec: TruenoVec<i32> = TruenoVec::with_capacity(0);
    assert_eq!(vec.capacity(), 0);
    assert_eq!(vec.len(), 0);
    assert!(vec.is_empty());
}

// ============================================================================
// Library API Tests
// ============================================================================

#[test]
fn test_certeza_math_functions() {
    // Test the example math functions from lib.rs
    use certeza::{add, multiply};

    assert_eq!(add(2, 3), 5);
    assert_eq!(add(0, 0), 0);
    assert_eq!(add(100, 200), 300);

    assert_eq!(multiply(2, 3), 6);
    assert_eq!(multiply(0, 100), 0);
    assert_eq!(multiply(10, 10), 100);
}

#[test]
fn test_default_trait_integration() {
    let vec: TruenoVec<i32> = TruenoVec::default();
    assert!(vec.is_empty());
    assert_eq!(vec.len(), 0);
    assert_eq!(vec.capacity(), 0);
}

// ============================================================================
// Thread Safety Integration Tests
// ============================================================================

#[test]
fn test_trueno_vec_send_across_threads() {
    // Verify TruenoVec can be sent across threads
    let mut vec = TruenoVec::new();
    for i in 0..100 {
        vec.push(i);
    }

    let handle = std::thread::spawn(move || {
        assert_eq!(vec.len(), 100);
        vec.pop()
    });

    assert_eq!(handle.join().unwrap(), Some(99));
}

#[test]
fn test_trueno_vec_with_arc() {
    use std::sync::Arc;

    // Test that TruenoVec works with Arc<T> where T: Sync
    let mut vec = TruenoVec::new();
    let data = Arc::new(42);

    vec.push(Arc::clone(&data));
    vec.push(Arc::clone(&data));
    vec.push(Arc::clone(&data));

    assert_eq!(Arc::strong_count(&data), 4); // 1 original + 3 in vec

    vec.pop();
    assert_eq!(Arc::strong_count(&data), 3);
}

// ============================================================================
// Advanced Scenario-Based Tests
// ============================================================================

/// Test scenario: Building a dynamic list of user records
#[test]
fn test_user_records_scenario() {
    #[derive(Debug, Clone, PartialEq)]
    struct UserRecord {
        id: u64,
        name: String,
        score: i32,
    }

    let mut users = TruenoVec::new();

    // Add users dynamically
    users.push(UserRecord { id: 1, name: "Alice".to_string(), score: 100 });
    users.push(UserRecord { id: 2, name: "Bob".to_string(), score: 85 });
    users.push(UserRecord { id: 3, name: "Charlie".to_string(), score: 92 });

    assert_eq!(users.len(), 3);

    // Access and verify
    if let Some(user) = users.get(1) {
        assert_eq!(user.name, "Bob");
        assert_eq!(user.score, 85);
    }

    // Update a record
    if let Some(user) = users.get_mut(2) {
        user.score = 95;
    }

    assert_eq!(users.get(2).unwrap().score, 95);

    // Remove last user
    let removed = users.pop();
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().name, "Charlie");
    assert_eq!(users.len(), 2);
}

/// Test scenario: Stack-based expression evaluation
#[test]
fn test_stack_evaluation_scenario() {
    let mut stack = TruenoVec::new();

    // Simulate evaluating: (2 + 3) * 4
    // In postfix: 2 3 + 4 *
    stack.push(2);
    stack.push(3);

    let b = stack.pop().unwrap();
    let a = stack.pop().unwrap();
    stack.push(a + b); // 5

    stack.push(4);

    let b = stack.pop().unwrap();
    let a = stack.pop().unwrap();
    stack.push(a * b); // 20

    assert_eq!(stack.pop(), Some(20));
    assert!(stack.is_empty());
}

/// Test scenario: Batch processing with pre-allocated capacity
#[test]
fn test_batch_processing_scenario() {
    const BATCH_SIZE: usize = 1000;

    // Pre-allocate for efficiency
    let mut batch = TruenoVec::with_capacity(BATCH_SIZE);

    // Verify no allocation during batch fill
    let initial_capacity = batch.capacity();

    for i in 0..BATCH_SIZE {
        batch.push(i);
    }

    assert_eq!(batch.len(), BATCH_SIZE);
    assert_eq!(batch.capacity(), initial_capacity);

    // Verify data integrity
    for i in 0..BATCH_SIZE {
        assert_eq!(batch.get(i), Some(&i));
    }
}

/// Test scenario: Undo/Redo system
#[test]
fn test_undo_redo_scenario() {
    #[derive(Debug, Clone, PartialEq)]
    enum Action {
        Insert(char),
        _Delete,
    }

    let mut undo_stack = TruenoVec::new();
    let mut redo_stack = TruenoVec::new();

    // Perform actions
    undo_stack.push(Action::Insert('H'));
    undo_stack.push(Action::Insert('e'));
    undo_stack.push(Action::Insert('l'));
    undo_stack.push(Action::Insert('l'));
    undo_stack.push(Action::Insert('o'));

    assert_eq!(undo_stack.len(), 5);

    // Undo last two actions
    if let Some(action) = undo_stack.pop() {
        redo_stack.push(action);
    }
    if let Some(action) = undo_stack.pop() {
        redo_stack.push(action);
    }

    assert_eq!(undo_stack.len(), 3);
    assert_eq!(redo_stack.len(), 2);

    // Redo one action
    if let Some(action) = redo_stack.pop() {
        undo_stack.push(action);
    }

    assert_eq!(undo_stack.len(), 4);
    assert_eq!(redo_stack.len(), 1);
}

/// Test scenario: Dynamic graph adjacency list
#[test]
fn test_graph_adjacency_scenario() {
    type NodeId = usize;

    struct Graph {
        adjacency: Vec<TruenoVec<NodeId>>,
    }

    impl Graph {
        fn new(num_nodes: usize) -> Self {
            let mut adjacency = Vec::new();
            for _ in 0..num_nodes {
                adjacency.push(TruenoVec::new());
            }
            Self { adjacency }
        }

        fn add_edge(&mut self, from: NodeId, to: NodeId) {
            if from < self.adjacency.len() {
                self.adjacency[from].push(to);
            }
        }

        fn neighbors(&self, node: NodeId) -> Option<&TruenoVec<NodeId>> {
            self.adjacency.get(node)
        }
    }

    let mut graph = Graph::new(5);

    // Build a small graph: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 3);
    graph.add_edge(2, 3);
    graph.add_edge(3, 4);

    // Verify adjacency lists
    let node0_neighbors = graph.neighbors(0).unwrap();
    assert_eq!(node0_neighbors.len(), 2);
    assert_eq!(node0_neighbors.get(0), Some(&1));
    assert_eq!(node0_neighbors.get(1), Some(&2));

    let node3_neighbors = graph.neighbors(3).unwrap();
    assert_eq!(node3_neighbors.len(), 1);
    assert_eq!(node3_neighbors.get(0), Some(&4));
}

/// Test scenario: Ring buffer implementation
#[test]
fn test_ring_buffer_scenario() {
    struct RingBuffer<T> {
        buffer: TruenoVec<T>,
        _head: usize,
        tail: usize,
        capacity: usize,
    }

    impl<T: Clone> RingBuffer<T> {
        fn new(capacity: usize) -> Self {
            Self { buffer: TruenoVec::with_capacity(capacity), _head: 0, tail: 0, capacity }
        }

        fn push(&mut self, item: T) {
            if self.buffer.len() < self.capacity {
                self.buffer.push(item);
                self.tail = self.buffer.len();
            }
        }

        fn len(&self) -> usize {
            self.buffer.len()
        }
    }

    let mut ring = RingBuffer::new(5);
    ring.push(1);
    ring.push(2);
    ring.push(3);

    assert_eq!(ring.len(), 3);
}

/// Test scenario: Large dataset processing
#[test]
fn test_large_dataset_scenario() {
    const LARGE_SIZE: usize = 10_000;

    let mut data = TruenoVec::new();

    // Fill with sequential data
    for i in 0..LARGE_SIZE {
        data.push(i * 2);
    }

    assert_eq!(data.len(), LARGE_SIZE);

    // Verify exponential growth kept reallocations minimal
    // With 2x growth, we expect approximately log2(10000) ≈ 14 reallocations
    // Final capacity should be >= 16384 (2^14)
    assert!(data.capacity() >= 16384);

    // Verify data integrity
    for i in 0..LARGE_SIZE {
        assert_eq!(data.get(i), Some(&(i * 2)));
    }

    // Test pop operations
    for _ in 0..1000 {
        data.pop();
    }

    assert_eq!(data.len(), LARGE_SIZE - 1000);
}

/// Test scenario: Type safety with different element types
#[test]
fn test_type_safety_scenario() {
    // String vector
    let mut strings = TruenoVec::new();
    strings.push("hello".to_string());
    strings.push("world".to_string());
    assert_eq!(strings.get(0).map(String::as_str), Some("hello"));

    // Option vector
    let mut options: TruenoVec<Option<i32>> = TruenoVec::new();
    options.push(Some(42));
    options.push(None);
    options.push(Some(100));
    assert_eq!(options.get(1), Some(&None));

    // Nested vector
    let mut nested = TruenoVec::new();
    let mut inner1 = TruenoVec::new();
    inner1.push(1);
    inner1.push(2);
    nested.push(inner1);
    assert_eq!(nested.len(), 1);
}

/// Test scenario: Concurrent safety verification (compile-time)
/// This test verifies that `TruenoVec` is Send + Sync where appropriate
#[test]
fn test_thread_safety_compile_time() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    // TruenoVec<i32> should be Send and Sync
    assert_send::<TruenoVec<i32>>();
    assert_sync::<TruenoVec<i32>>();

    // TruenoVec<String> should be Send and Sync
    assert_send::<TruenoVec<String>>();
    assert_sync::<TruenoVec<String>>();
}

/// Test scenario: Memory efficiency verification
#[test]
fn test_memory_efficiency_scenario() {
    use std::mem;

    let vec = TruenoVec::<i32>::new();

    // TruenoVec should have minimal overhead
    // Expected: pointer (8 bytes) + len (8 bytes) + capacity (8 bytes) + PhantomData (0 bytes)
    let size = mem::size_of_val(&vec);
    assert_eq!(size, 24); // 3 * 8 bytes (no padding needed)

    // Verify zero-cost abstraction for empty vec
    let empty = TruenoVec::<u8>::new();
    assert_eq!(empty.capacity(), 0);
    assert_eq!(empty.len(), 0);
}

/// Test scenario: Error handling and edge cases
#[test]
fn test_edge_cases_scenario() {
    let mut vec = TruenoVec::new();

    // Pop from empty
    assert_eq!(vec.pop(), None);

    // Get out of bounds
    assert_eq!(vec.get(0), None);
    assert_eq!(vec.get(100), None);

    // Single element operations
    vec.push(42);
    assert_eq!(vec.len(), 1);
    assert_eq!(vec.get(0), Some(&42));
    assert_eq!(vec.pop(), Some(42));
    assert!(vec.is_empty());

    // Capacity preservation after pops
    let mut vec2 = TruenoVec::with_capacity(100);
    for i in 0..50 {
        vec2.push(i);
    }
    let capacity_before = vec2.capacity();
    for _ in 0..25 {
        vec2.pop();
    }
    assert_eq!(vec2.capacity(), capacity_before); // Capacity doesn't shrink
}

/// Test scenario: Comparison with standard Vec behavior
#[test]
fn test_std_vec_equivalence() {
    let mut trueno = TruenoVec::new();
    let mut std_vec = Vec::new();

    // Parallel operations
    for i in 0..100 {
        trueno.push(i);
        std_vec.push(i);
    }

    assert_eq!(trueno.len(), std_vec.len());

    for i in 0..100 {
        assert_eq!(trueno.get(i), std_vec.get(i));
    }

    for _ in 0..50 {
        assert_eq!(trueno.pop(), std_vec.pop());
    }

    assert_eq!(trueno.len(), std_vec.len());
}
