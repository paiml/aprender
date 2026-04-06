#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::super::*;

    #[test]
    fn test_new() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let vec: TruenoVec<i32> = TruenoVec::with_capacity(10);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity_zero() {
        let vec: TruenoVec<i32> = TruenoVec::with_capacity(0);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
    }

    #[test]
    fn test_push_single() {
        let mut vec = TruenoVec::new();
        vec.push(42);
        assert_eq!(vec.len(), 1);
        assert!(!vec.is_empty());
        assert_eq!(vec.get(0), Some(&42));
    }

    #[test]
    fn test_push_multiple() {
        let mut vec = TruenoVec::new();
        for i in 0..10 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 10);
        for i in 0..10 {
            assert_eq!(vec.get(i), Some(&i));
        }
    }

    #[test]
    fn test_pop_empty() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_pop_single() {
        let mut vec = TruenoVec::new();
        vec.push(42);
        assert_eq!(vec.pop(), Some(42));
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_push_pop_sequence() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        vec.push(4);
        assert_eq!(vec.pop(), Some(4));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_get_valid_index() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);
        assert_eq!(vec.get(0), Some(&10));
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(2), Some(&30));
    }

    #[test]
    fn test_get_invalid_index() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(100), None);
    }

    #[test]
    fn test_get_mut() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        if let Some(elem) = vec.get_mut(1) {
            *elem = 25;
        }
        assert_eq!(vec.get(1), Some(&25));
    }

    #[test]
    fn test_get_mut_boundary() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Valid indices should work
        assert!(vec.get_mut(0).is_some());
        assert!(vec.get_mut(1).is_some());
        assert!(vec.get_mut(2).is_some());

        // Index equal to len should return None
        assert_eq!(vec.get_mut(3), None);

        // Higher indices should return None
        assert_eq!(vec.get_mut(4), None);
        assert_eq!(vec.get_mut(100), None);
    }

    #[test]
    fn test_growth() {
        let mut vec = TruenoVec::new();
        assert_eq!(vec.capacity(), 0);

        vec.push(1);
        assert_eq!(vec.capacity(), 1);

        vec.push(2);
        assert_eq!(vec.capacity(), 2);

        vec.push(3);
        assert_eq!(vec.capacity(), 4);

        vec.push(4);
        assert_eq!(vec.capacity(), 4);

        vec.push(5);
        assert_eq!(vec.capacity(), 8);
    }

    #[test]
    fn test_capacity_maintained() {
        let mut vec = TruenoVec::with_capacity(10);
        for i in 0..10 {
            vec.push(i);
        }
        assert_eq!(vec.capacity(), 10);

        vec.push(10);
        assert_eq!(vec.capacity(), 20);
    }

    #[test]
    fn test_drop_calls_destructors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let mut vec = TruenoVec::new();
            for _ in 0..5 {
                vec.push(DropCounter(Arc::clone(&counter)));
            }
        } // vec dropped here

        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_default() {
        let vec: TruenoVec<i32> = TruenoVec::default();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
    }

    #[test]
    fn test_drop_empty_vector() {
        // Test that dropping an empty vector (capacity=0) doesn't crash
        {
            let vec: TruenoVec<i32> = TruenoVec::new();
            assert_eq!(vec.capacity(), 0);
        } // vec dropped here - should not attempt deallocation

        // Test with capacity > 0 but no elements
        {
            let vec: TruenoVec<i32> = TruenoVec::with_capacity(10);
            assert_eq!(vec.capacity(), 10);
            assert_eq!(vec.len(), 0);
        } // vec dropped here - should deallocate

        // Test after popping all elements
        {
            let mut vec = TruenoVec::new();
            vec.push(1);
            vec.push(2);
            vec.pop();
            vec.pop();
            assert_eq!(vec.len(), 0);
            assert!(vec.capacity() > 0);
        } // vec dropped here - should deallocate despite len=0
    }

    #[test]
    fn test_clear_empty() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_clear_with_elements() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let capacity = vec.capacity();
        vec.clear();

        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.capacity(), capacity); // Capacity should be unchanged
    }

    #[test]
    fn test_clear_with_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let mut vec = TruenoVec::new();
            for _ in 0..5 {
                vec.push(DropCounter(Arc::clone(&counter)));
            }
            vec.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 5); // All elements dropped
            assert_eq!(vec.len(), 0);
        }
    }

    #[test]
    fn test_pop_decrements_length() {
        // Explicitly test that pop decrements length correctly
        // This catches mutations where -= is replaced with += or other operators
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        let initial_len = vec.len();
        assert_eq!(initial_len, 3);

        // Each pop must decrement by exactly 1
        let val1 = vec.pop();
        assert_eq!(val1, Some(30));
        assert_eq!(vec.len(), initial_len - 1);
        assert_eq!(vec.len(), 2); // Must be exactly 2, not 4 (if += used)

        let val2 = vec.pop();
        assert_eq!(val2, Some(20));
        assert_eq!(vec.len(), 1); // Must be exactly 1, not 5

        let val3 = vec.pop();
        assert_eq!(val3, Some(10));
        assert_eq!(vec.len(), 0); // Must be exactly 0, not 6

        // Further pops should return None and keep length at 0
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.len(), 0); // Must stay 0, not become 1
    }

    #[test]
    fn test_drop_with_capacity_deallocates() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Test that drop properly deallocates memory for vector with capacity > 0
        // We use DropCounter to verify all cleanup happens correctly
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            // Create vector with capacity and add elements
            let mut vec = TruenoVec::with_capacity(5);
            let capacity = vec.capacity();
            assert!(capacity > 0); // Capacity must be > 0 for this test
            assert_eq!(capacity, 5);

            vec.push(DropCounter(Arc::clone(&counter)));
            vec.push(DropCounter(Arc::clone(&counter)));
            vec.push(DropCounter(Arc::clone(&counter)));
            assert_eq!(vec.len(), 3);
            assert_eq!(vec.capacity(), 5); // Capacity unchanged
        } // vec dropped here - must deallocate because capacity > 0

        // All 3 elements must be dropped
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_drop_deallocates_empty_with_capacity() {
        // Specifically test the capacity > 0 check in drop
        // This catches mutations like: if self.capacity > 0 -> if self.capacity < 0
        let vec: TruenoVec<i32> = TruenoVec::with_capacity(100);

        // Verify we have capacity but no elements
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        // When vec is dropped, it MUST deallocate because capacity > 0
        // If the condition is mutated to capacity < 0, this would leak memory
        // Since usize can't be < 0, the deallocation would never happen
        drop(vec);

        // We can't directly test for memory leaks in safe Rust, but we can
        // ensure the drop path is exercised for vectors with capacity > 0
        // The mutation test will catch if the deallocation block is skipped
    }

    #[test]
    fn test_insert_at_start() {
        let mut vec = TruenoVec::new();
        vec.push(2);
        vec.push(3);
        vec.insert(0, 1);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
    }

    #[test]
    fn test_insert_at_middle() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(3);
        vec.insert(1, 2);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
    }

    #[test]
    fn test_insert_at_end() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.insert(2, 3); // Insert at end (equivalent to push)

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(2), Some(&3));
    }

    #[test]
    fn test_insert_empty_vector() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        vec.insert(0, 42);

        assert_eq!(vec.len(), 1);
        assert_eq!(vec.get(0), Some(&42));
    }

    #[test]
    #[should_panic(expected = "insertion index out of bounds")]
    fn test_insert_out_of_bounds() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.insert(5, 2); // index > len
    }

    #[test]
    fn test_remove_at_start() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let removed = vec.remove(0);
        assert_eq!(removed, 1);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(&2));
        assert_eq!(vec.get(1), Some(&3));
    }

    #[test]
    fn test_remove_at_middle() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let removed = vec.remove(1);
        assert_eq!(removed, 2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&3));
    }

    #[test]
    fn test_remove_at_end() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let removed = vec.remove(2);
        assert_eq!(removed, 3);
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_remove_all_elements() {
        let mut vec = TruenoVec::new();
        vec.push(1);

        let removed = vec.remove(0);
        assert_eq!(removed, 1);
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    #[should_panic(expected = "removal index out of bounds")]
    fn test_remove_empty_vector() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        vec.remove(0);
    }

    #[test]
    #[should_panic(expected = "removal index out of bounds")]
    fn test_remove_out_of_bounds() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.remove(5); // index >= len
    }

    #[test]
    fn test_iter() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let mut iter = vec.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let mut iter = vec.iter();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_collect() {
        let mut vec = TruenoVec::new();
        for i in 0..10 {
            vec.push(i);
        }

        let collected: Vec<_> = vec.iter().copied().collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_rev() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let collected: Vec<_> = vec.iter().copied().rev().collect();
        assert_eq!(collected, vec![3, 2, 1]);
    }

    #[test]
    fn test_iter_mut() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        for elem in &mut vec {
            *elem *= 2;
        }

        assert_eq!(vec.get(0), Some(&2));
        assert_eq!(vec.get(1), Some(&4));
        assert_eq!(vec.get(2), Some(&6));
    }

    #[test]
    fn test_iter_mut_empty() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        let mut iter = vec.iter_mut();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let mut iter = vec.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_sum() {
        let mut vec = TruenoVec::new();
        for i in 1..=10 {
            vec.push(i);
        }

        let sum: i32 = vec.into_iter().sum();
        assert_eq!(sum, 55); // 1+2+3+...+10 = 55
    }

    #[test]
    fn test_for_loop_ref() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let mut sum = 0;
        for &elem in &vec {
            sum += elem;
        }
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_for_loop_mut() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        for elem in &mut vec {
            *elem += 10;
        }

        assert_eq!(vec.get(0), Some(&11));
        assert_eq!(vec.get(1), Some(&12));
        assert_eq!(vec.get(2), Some(&13));
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut vec = TruenoVec::new();
        for i in 0..10 {
            vec.push(i);
        }

        let iter = vec.iter();
        assert_eq!(iter.len(), 10);

        let iter_mut = vec.iter_mut();
        assert_eq!(iter_mut.len(), 10);
    }

    #[test]
    fn test_into_iter_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let mut vec = TruenoVec::new();
            for _ in 0..5 {
                vec.push(DropCounter(Arc::clone(&counter)));
            }

            // Consume only 2 elements
            let mut iter = vec.into_iter();
            iter.next();
            iter.next();
            // iter dropped here, should drop remaining 3 + consumed 2 = 5 total
        }

        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    // ========================================================================
    // Index Trait Tests
    // ========================================================================

    #[test]
    fn test_index_read() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        assert_eq!(vec[2], 30);
    }

    #[test]
    fn test_index_mut_write() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        vec[1] = 25;
        assert_eq!(vec[1], 25);
        assert_eq!(vec.get(1), Some(&25));
    }

    #[test]
    fn test_index_mut_compound_assignment() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        vec[0] += 5;
        vec[1] *= 2;
        vec[2] -= 10;

        assert_eq!(vec[0], 15);
        assert_eq!(vec[1], 40);
        assert_eq!(vec[2], 20);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        let _ = vec[5]; // Should panic
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec[5] = 10; // Should panic
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_empty_vector() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let _ = vec[0]; // Should panic
    }

    // ========================================================================
    // Clone Trait Tests
    // ========================================================================

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_clone_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let cloned = vec.clone();

        // Both should be empty
        assert_eq!(vec.len(), 0);
        assert_eq!(cloned.len(), 0);
        assert_eq!(vec.capacity(), cloned.capacity());
        assert!(vec.is_empty());
        assert!(cloned.is_empty());
    }

    #[test]
    fn test_clone_with_elements() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let cloned = vec.clone();

        assert_eq!(vec.len(), cloned.len());
        for i in 0..vec.len() {
            assert_eq!(vec.get(i), cloned.get(i));
        }
    }

    #[test]
    fn test_clone_independence() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = vec1.clone();

        // Modify vec2
        vec2.push(4);
        vec2[0] = 10;

        // vec1 should be unchanged
        assert_eq!(vec1.len(), 3);
        assert_eq!(vec1[0], 1);
        assert_eq!(vec1[1], 2);
        assert_eq!(vec1[2], 3);

        // vec2 has the changes
        assert_eq!(vec2.len(), 4);
        assert_eq!(vec2[0], 10);
        assert_eq!(vec2[1], 2);
        assert_eq!(vec2[2], 3);
        assert_eq!(vec2[3], 4);
    }

    #[test]
    fn test_clone_with_strings() {
        let mut vec1 = TruenoVec::new();
        vec1.push(String::from("hello"));
        vec1.push(String::from("world"));

        let vec2 = vec1.clone();

        assert_eq!(vec1.len(), vec2.len());
        assert_eq!(vec1[0], "hello");
        assert_eq!(vec2[0], "hello");
        assert_eq!(vec1[1], "world");
        assert_eq!(vec2[1], "world");
    }

    #[test]
    fn test_clone_capacity() {
        let mut vec = TruenoVec::with_capacity(100);
        vec.push(1);
        vec.push(2);

        let cloned = vec.clone();

        // Clone should have capacity equal to length (optimized allocation)
        assert_eq!(cloned.capacity(), 2);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn test_clone_drop_independence() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        impl Clone for DropCounter {
            fn clone(&self) -> Self {
                Self(Arc::clone(&self.0))
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let mut vec1 = TruenoVec::new();
            vec1.push(DropCounter(Arc::clone(&counter)));
            vec1.push(DropCounter(Arc::clone(&counter)));

            {
                let vec2 = vec1.clone();
                assert_eq!(vec2.len(), 2);
            } // vec2 dropped here - should drop 2 elements

            assert_eq!(counter.load(Ordering::SeqCst), 2);
        } // vec1 dropped here - should drop 2 more elements

        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    // ========================================================================
    // PartialEq and Eq Trait Tests
    // ========================================================================

    #[test]
    fn test_equality_same_elements() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);
        vec2.push(3);

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_equality_empty_vectors() {
        let vec1: TruenoVec<i32> = TruenoVec::new();
        let vec2: TruenoVec<i32> = TruenoVec::new();

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_inequality_different_length() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);
        vec2.push(3);

        assert_ne!(vec1, vec2);
    }

    #[test]
    fn test_inequality_different_elements() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(5);
        vec2.push(3);

        assert_ne!(vec1, vec2);
    }

    #[test]
    fn test_equality_with_strings() {
        let mut vec1 = TruenoVec::new();
        vec1.push(String::from("hello"));
        vec1.push(String::from("world"));

        let mut vec2 = TruenoVec::new();
        vec2.push(String::from("hello"));
        vec2.push(String::from("world"));

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_equality_reflexive() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);

        assert_eq!(vec, vec);
    }

    #[test]
    fn test_equality_symmetric() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);

        assert_eq!(vec1, vec2);
        assert_eq!(vec2, vec1);
    }

    #[test]
    fn test_equality_transitive() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);

        let mut vec3 = TruenoVec::new();
        vec3.push(1);
        vec3.push(2);

        assert_eq!(vec1, vec2);
        assert_eq!(vec2, vec3);
        assert_eq!(vec1, vec3);
    }

    // ========================================================================
    // Debug Trait Tests
    // ========================================================================

    #[test]
    fn test_debug_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        assert_eq!(format!("{vec:?}"), "[]");
    }

    #[test]
    fn test_debug_single_element() {
        let mut vec = TruenoVec::new();
        vec.push(42);
        assert_eq!(format!("{vec:?}"), "[42]");
    }

    #[test]
    fn test_debug_multiple_elements() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(format!("{vec:?}"), "[1, 2, 3]");
    }

    #[test]
    fn test_debug_with_strings() {
        let mut vec = TruenoVec::new();
        vec.push(String::from("hello"));
        vec.push(String::from("world"));
        assert_eq!(format!("{vec:?}"), "[\"hello\", \"world\"]");
    }

    #[test]
    fn test_debug_nested() {
        let mut inner = TruenoVec::new();
        inner.push(1);
        inner.push(2);

        let mut outer = TruenoVec::new();
        outer.push(inner.clone());
        outer.push(inner);

        assert_eq!(format!("{outer:?}"), "[[1, 2], [1, 2]]");
    }

    // ========================================================================
    // Conversion Trait Tests (From, FromIterator, Extend)
    // ========================================================================

    #[test]
    fn test_from_vec_basic() {
        let std_vec = vec![1, 2, 3, 4, 5];
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(std_vec);

        assert_eq!(trueno_vec.len(), 5);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(2), Some(&3));
        assert_eq!(trueno_vec.get(4), Some(&5));
    }

    #[test]
    fn test_from_vec_empty() {
        let std_vec: Vec<i32> = vec![];
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(std_vec);

        assert_eq!(trueno_vec.len(), 0);
        assert!(trueno_vec.is_empty());
    }

    #[test]
    fn test_from_vec_large() {
        let std_vec: Vec<i32> = (0..100).collect();
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(std_vec);

        assert_eq!(trueno_vec.len(), 100);
        for i in 0..100 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    #[test]
    fn test_from_vec_drops_properly() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let mut std_vec = Vec::new();
            for _ in 0..5 {
                std_vec.push(DropCounter(Arc::clone(&counter)));
            }

            let trueno_vec = TruenoVec::from(std_vec);
            assert_eq!(trueno_vec.len(), 5);
        } // trueno_vec dropped here

        // All 5 elements should be dropped exactly once
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_from_slice_basic() {
        let array = [1, 2, 3, 4, 5];
        let slice: &[i32] = &array;
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(slice);

        assert_eq!(trueno_vec.len(), 5);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(2), Some(&3));
        assert_eq!(trueno_vec.get(4), Some(&5));
    }

    #[test]
    fn test_from_slice_empty() {
        let slice: &[i32] = &[];
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(slice);

        assert_eq!(trueno_vec.len(), 0);
        assert!(trueno_vec.is_empty());
    }

    #[test]
    fn test_from_slice_clones_elements() {
        // Test that elements are cloned, not moved
        let original = vec![String::from("hello"), String::from("world")];
        let slice = original.as_slice();

        let trueno_vec: TruenoVec<String> = TruenoVec::from(slice);

        // Original should still be valid
        assert_eq!(original.len(), 2);
        assert_eq!(original[0], "hello");
        assert_eq!(original[1], "world");

        // TruenoVec should have cloned values
        assert_eq!(trueno_vec.len(), 2);
        assert_eq!(trueno_vec.get(0), Some(&String::from("hello")));
        assert_eq!(trueno_vec.get(1), Some(&String::from("world")));
    }

    #[test]
    fn test_from_slice_large() {
        let slice: Vec<i32> = (0..100).collect();
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(slice.as_slice());

        assert_eq!(trueno_vec.len(), 100);
        for i in 0..100 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    #[test]
    fn test_from_iterator_range() {
        let trueno_vec: TruenoVec<i32> = (0..10).collect();

        assert_eq!(trueno_vec.len(), 10);
        for i in 0..10 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    #[test]
    fn test_from_iterator_vec() {
        let std_vec = vec![1, 2, 3, 4, 5];
        let trueno_vec: TruenoVec<i32> = std_vec.into_iter().collect();

        assert_eq!(trueno_vec.len(), 5);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(4), Some(&5));
    }

    #[test]
    fn test_from_iterator_empty() {
        let empty: Vec<i32> = vec![];
        let trueno_vec: TruenoVec<i32> = empty.into_iter().collect();

        assert_eq!(trueno_vec.len(), 0);
        assert!(trueno_vec.is_empty());
    }

    #[test]
    fn test_from_iterator_map() {
        let numbers = [1, 2, 3, 4, 5];
        let doubled: TruenoVec<i32> = numbers.iter().map(|x| x * 2).collect();

        assert_eq!(doubled.len(), 5);
        assert_eq!(doubled.get(0), Some(&2));
        assert_eq!(doubled.get(2), Some(&6));
        assert_eq!(doubled.get(4), Some(&10));
    }

    #[test]
    fn test_from_iterator_filter() {
        let trueno_vec: TruenoVec<i32> = (0..20).filter(|x| x % 2 == 0).collect();

        assert_eq!(trueno_vec.len(), 10); // 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
        assert_eq!(trueno_vec.get(0), Some(&0));
        assert_eq!(trueno_vec.get(5), Some(&10));
        assert_eq!(trueno_vec.get(9), Some(&18));
    }

    #[test]
    fn test_from_iterator_large() {
        let trueno_vec: TruenoVec<i32> = (0..1000).collect();

        assert_eq!(trueno_vec.len(), 1000);
        assert_eq!(trueno_vec.get(0), Some(&0));
        assert_eq!(trueno_vec.get(500), Some(&500));
        assert_eq!(trueno_vec.get(999), Some(&999));
    }

    #[test]
    fn test_extend_empty_vec() {
        let mut trueno_vec: TruenoVec<i32> = TruenoVec::new();
        trueno_vec.extend(vec![1, 2, 3, 4, 5]);

        assert_eq!(trueno_vec.len(), 5);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(4), Some(&5));
    }

    #[test]
    fn test_extend_existing_vec() {
        let mut trueno_vec = TruenoVec::new();
        trueno_vec.push(1);
        trueno_vec.push(2);

        trueno_vec.extend(vec![3, 4, 5]);

        assert_eq!(trueno_vec.len(), 5);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(2), Some(&3));
        assert_eq!(trueno_vec.get(4), Some(&5));
    }

    #[test]
    fn test_extend_multiple_times() {
        let mut trueno_vec = TruenoVec::new();

        trueno_vec.extend(vec![1, 2]);
        assert_eq!(trueno_vec.len(), 2);

        trueno_vec.extend(vec![3, 4]);
        assert_eq!(trueno_vec.len(), 4);

        trueno_vec.extend(vec![5, 6]);
        assert_eq!(trueno_vec.len(), 6);

        for i in 0..6 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i + 1).unwrap()));
        }
    }

    #[test]
    fn test_extend_from_range() {
        let mut trueno_vec = TruenoVec::new();
        trueno_vec.extend(0..10);

        assert_eq!(trueno_vec.len(), 10);
        for i in 0..10 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    #[test]
    fn test_extend_from_iterator() {
        let mut trueno_vec = TruenoVec::new();
        trueno_vec.push(1);

        let numbers = vec![2, 3, 4];
        trueno_vec.extend(numbers.into_iter().map(|x| x * 2));

        assert_eq!(trueno_vec.len(), 4);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(1), Some(&4));
        assert_eq!(trueno_vec.get(2), Some(&6));
        assert_eq!(trueno_vec.get(3), Some(&8));
    }

    #[test]
    fn test_extend_empty_iterator() {
        let mut trueno_vec = TruenoVec::new();
        trueno_vec.push(1);
        trueno_vec.push(2);

        let empty: Vec<i32> = vec![];
        trueno_vec.extend(empty);

        assert_eq!(trueno_vec.len(), 2);
        assert_eq!(trueno_vec.get(0), Some(&1));
        assert_eq!(trueno_vec.get(1), Some(&2));
    }

    #[test]
    fn test_extend_large() {
        let mut trueno_vec = TruenoVec::new();
        trueno_vec.extend(0..500);

        assert_eq!(trueno_vec.len(), 500);

        trueno_vec.extend(500..1000);

        assert_eq!(trueno_vec.len(), 1000);
        for i in 0..1000 {
            assert_eq!(trueno_vec.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    #[test]
    fn test_extend_preserves_capacity() {
        let mut trueno_vec = TruenoVec::with_capacity(100);
        trueno_vec.extend(0..50);

        assert_eq!(trueno_vec.len(), 50);
        assert!(trueno_vec.capacity() >= 100);
    }

    #[test]
    fn test_conversion_chain() {
        // Test that conversions can be chained
        let std_vec = vec![1, 2, 3, 4, 5];
        let trueno_vec: TruenoVec<i32> = TruenoVec::from(std_vec);

        // Extend it
        let mut extended = trueno_vec;
        extended.extend(6..10);

        assert_eq!(extended.len(), 9);
        for i in 0..9 {
            assert_eq!(extended.get(i), Some(&i32::try_from(i + 1).unwrap()));
        }
    }

    #[test]
    fn test_collect_then_extend() {
        let trueno_vec: TruenoVec<i32> = (0..5).collect();
        let mut extended = trueno_vec;
        extended.extend(5..10);

        assert_eq!(extended.len(), 10);
        for i in 0..10 {
            assert_eq!(extended.get(i), Some(&i32::try_from(i).unwrap()));
        }
    }

    // ========================================================================
    // Phase 3.1: Deref/DerefMut Trait Tests
    // ========================================================================

    #[test]
    fn test_deref_to_slice() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Deref coercion to slice
        let slice: &[i32] = &vec;
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
    }

    #[test]
    fn test_deref_slice_methods() {
        let mut vec = TruenoVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);

        // Use slice methods through deref
        assert_eq!(vec.first(), Some(&10));
        assert_eq!(vec.last(), Some(&30));
        assert!(!vec.is_empty());
        assert_eq!(vec.len(), 3);
    }

    #[test]
    fn test_deref_empty_vec() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let slice: &[i32] = &vec;
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
        assert_eq!(vec.first(), None);
        assert_eq!(vec.last(), None);
    }

    #[test]
    fn test_deref_with_functions() {
        fn sum_slice(s: &[i32]) -> i32 {
            s.iter().sum()
        }

        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);

        // Deref coercion allows passing TruenoVec to functions expecting &[T]
        assert_eq!(sum_slice(&vec), 10);
    }

    #[test]
    fn test_deref_iter() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Use slice's iter method through deref
        let sum: i32 = vec.iter().sum();
        assert_eq!(sum, 6);

        let doubled: Vec<i32> = vec.iter().map(|x| x * 2).collect();
        assert_eq!(doubled, vec![2, 4, 6]);
    }

    #[test]
    fn test_deref_contains() {
        let mut vec = TruenoVec::new();
        vec.push(5);
        vec.push(10);
        vec.push(15);

        // Use slice's contains method through deref
        assert!(vec.contains(&10));
        assert!(!vec.contains(&20));
    }

    #[test]
    fn test_deref_starts_with_ends_with() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);

        // Use slice methods through deref
        assert!(vec.starts_with(&[1, 2]));
        assert!(vec.ends_with(&[3, 4]));
        assert!(!vec.starts_with(&[2, 3]));
    }

    #[test]
    fn test_deref_mut_to_slice() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // DerefMut coercion to mutable slice
        let slice: &mut [i32] = &mut vec;
        slice[0] = 10;
        slice[1] = 20;
        slice[2] = 30;

        assert_eq!(vec.get(0), Some(&10));
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(2), Some(&30));
    }

    #[test]
    fn test_deref_mut_sort() {
        let mut vec = TruenoVec::new();
        vec.push(3);
        vec.push(1);
        vec.push(4);
        vec.push(2);

        // Use mutable slice methods through DerefMut
        vec.sort_unstable();
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_deref_mut_reverse() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Use reverse method through DerefMut
        vec.reverse();
        assert_eq!(vec.as_slice(), &[3, 2, 1]);
    }

    #[test]
    fn test_deref_mut_fill() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Use fill method through DerefMut
        vec.fill(42);
        assert_eq!(vec.as_slice(), &[42, 42, 42]);
    }

    #[test]
    fn test_deref_mut_swap() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Use swap method through DerefMut
        vec.swap(0, 2);
        assert_eq!(vec.as_slice(), &[3, 2, 1]);
    }

    #[test]
    fn test_deref_mut_iter_mut() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Use iter_mut through DerefMut
        for elem in &mut vec {
            *elem *= 2;
        }

        assert_eq!(vec.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_deref_mut_split_at_mut() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);

        // Use split_at_mut through DerefMut
        let (left, right) = vec.split_at_mut(2);
        left[0] = 10;
        right[0] = 30;

        assert_eq!(vec.as_slice(), &[10, 2, 30, 4]);
    }

    // ========================================================================
    // Phase 3.1: AsRef Trait Tests
    // ========================================================================

    #[test]
    fn test_as_ref_slice() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let slice: &[i32] = vec.as_ref();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_as_ref_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let slice: &[i32] = vec.as_ref();
        assert_eq!(slice, &[] as &[i32]);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_as_ref_generic_function() {
        fn process<T: AsRef<[i32]>>(data: T) -> i32 {
            data.as_ref().iter().sum()
        }

        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(process(&vec), 6);
        assert_eq!(process(vec![1, 2, 3]), 6);
        assert_eq!(process([1, 2, 3]), 6);
    }

    #[test]
    fn test_as_ref_with_strings() {
        let mut vec = TruenoVec::new();
        vec.push(String::from("hello"));
        vec.push(String::from("world"));

        let slice: &[String] = vec.as_ref();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], "hello");
        assert_eq!(slice[1], "world");
    }

    #[test]
    fn test_as_ref_trueno_vec() {
        let vec = TruenoVec::from(vec![1, 2, 3]);
        let vec_ref: &TruenoVec<i32> = vec.as_ref();
        assert_eq!(vec_ref.len(), 3);
        assert_eq!(vec_ref.get(0), Some(&1));
    }

    #[test]
    fn test_as_ref_multiple_calls() {
        let mut vec = TruenoVec::new();
        vec.push(42);

        let ref1: &[i32] = vec.as_ref();
        let ref2: &[i32] = vec.as_ref();

        // Multiple AsRef calls should return equivalent slices
        assert_eq!(ref1, ref2);
        assert_eq!(ref1.as_ptr(), ref2.as_ptr());
    }

    // ========================================================================
    // Phase 3.1: AsMut Trait Tests
    // ========================================================================

    #[test]
    fn test_as_mut_slice() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let slice: &mut [i32] = vec.as_mut();
        slice[0] = 10;
        slice[2] = 30;

        assert_eq!(vec.as_slice(), &[10, 2, 30]);
    }

    #[test]
    fn test_as_mut_empty() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        let slice: &mut [i32] = vec.as_mut();
        assert_eq!(slice, &mut [] as &mut [i32]);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_as_mut_generic_function() {
        fn double_in_place<T: AsMut<[i32]>>(mut data: T) {
            for x in data.as_mut() {
                *x *= 2;
            }
        }

        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        double_in_place(&mut vec);
        assert_eq!(vec.as_slice(), &[2, 4, 6]);

        let mut std_vec = vec![1, 2, 3];
        double_in_place(&mut std_vec);
        assert_eq!(std_vec, vec![2, 4, 6]);
    }

    #[test]
    fn test_as_mut_fill_via_generic() {
        fn fill_with_value<T: AsMut<[i32]>>(mut data: T, value: i32) {
            for x in data.as_mut() {
                *x = value;
            }
        }

        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        fill_with_value(&mut vec, 99);
        assert_eq!(vec.as_slice(), &[99, 99, 99]);
    }

    #[test]
    fn test_as_mut_trueno_vec() {
        let mut vec = TruenoVec::from(vec![1, 2, 3]);
        let vec_mut: &mut TruenoVec<i32> = vec.as_mut();
        vec_mut.push(4);
        vec_mut.push(5);

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_as_mut_sorting() {
        fn sort_slice<T: AsMut<[i32]>>(mut data: T) {
            data.as_mut().sort_unstable();
        }

        let mut vec = TruenoVec::new();
        vec.push(3);
        vec.push(1);
        vec.push(4);
        vec.push(2);

        sort_slice(&mut vec);
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
    }

    // ========================================================================
    // Phase 3.1: as_slice and as_slice_mut Tests
    // ========================================================================

    #[test]
    fn test_as_slice_basic() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let slice = vec.as_slice();
        assert_eq!(slice, &[1, 2, 3]);
        assert_eq!(slice.len(), 3);
    }

    #[test]
    fn test_as_slice_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        let slice = vec.as_slice();
        assert_eq!(slice, &[] as &[i32]);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_as_slice_after_push() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        assert_eq!(vec.as_slice(), &[1]);

        vec.push(2);
        assert_eq!(vec.as_slice(), &[1, 2]);

        vec.push(3);
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_as_slice_mut_basic() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let slice = vec.as_slice_mut();
        slice[1] = 20;

        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.as_slice(), &[1, 20, 3]);
    }

    #[test]
    fn test_as_slice_mut_empty() {
        let mut vec: TruenoVec<i32> = TruenoVec::new();
        let slice = vec.as_slice_mut();
        assert_eq!(slice, &mut [] as &mut [i32]);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_as_slice_mut_modify_all() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        for elem in vec.as_slice_mut() {
            *elem *= 10;
        }

        assert_eq!(vec.as_slice(), &[10, 20, 30]);
    }

    // ========================================================================
    // Phase 3.1: Integration Tests
    // ========================================================================

    #[test]
    fn test_deref_and_as_ref_consistency() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let via_deref: &[i32] = &vec;
        let via_as_ref: &[i32] = vec.as_ref();
        let via_as_slice = vec.as_slice();

        assert_eq!(via_deref, via_as_ref);
        assert_eq!(via_deref, via_as_slice);
        assert_eq!(via_deref.as_ptr(), via_as_ref.as_ptr());
        assert_eq!(via_deref.as_ptr(), via_as_slice.as_ptr());
    }

    #[test]
    fn test_deref_mut_and_as_mut_consistency() {
        let mut vec1 = TruenoVec::from(vec![1, 2, 3]);
        let mut vec2 = TruenoVec::from(vec![1, 2, 3]);

        let slice1: &mut [i32] = &mut vec1;
        let slice2: &mut [i32] = vec2.as_mut();

        slice1[0] = 10;
        slice2[0] = 10;

        assert_eq!(vec1.as_slice(), vec2.as_slice());
    }

    #[test]
    fn test_slice_methods_comprehensive() {
        let mut vec = TruenoVec::new();
        for i in 0..10 {
            vec.push(i);
        }

        // Test various slice methods through Deref
        assert_eq!(vec.first(), Some(&0));
        assert_eq!(vec.last(), Some(&9));
        assert!(vec.contains(&5));
        assert!(!vec.contains(&20));
        let slice = vec.as_slice();
        assert_eq!(&slice[2..5], &[2, 3, 4]);
        assert_eq!(slice.get(5..8), Some(&[5, 6, 7][..]));
    }

    #[test]
    fn test_mutable_slice_methods_comprehensive() {
        let mut vec = TruenoVec::from(vec![5, 2, 8, 1, 9, 3]);

        // Sort through DerefMut
        vec.sort_unstable();
        assert_eq!(vec.as_slice(), &[1, 2, 3, 5, 8, 9]);

        // Reverse
        vec.reverse();
        assert_eq!(vec.as_slice(), &[9, 8, 5, 3, 2, 1]);

        // Rotate
        vec.rotate_left(2);
        assert_eq!(vec.as_slice(), &[5, 3, 2, 1, 9, 8]);
    }

    #[test]
    fn test_generic_function_interoperability() {
        fn count_evens<T: AsRef<[i32]>>(data: T) -> usize {
            data.as_ref().iter().filter(|&&x| x % 2 == 0).count()
        }

        let trueno_vec = TruenoVec::from(vec![1, 2, 3, 4, 5, 6]);
        let std_vec = vec![1, 2, 3, 4, 5, 6];

        assert_eq!(count_evens(&trueno_vec), count_evens(&std_vec));
        assert_eq!(count_evens(&trueno_vec), 3);
    }

    // ========================================================================
    // Phase 3.2: PartialOrd Trait Tests
    // ========================================================================

    #[test]
    fn test_partial_ord_less_than() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(3);

        assert!(vec1 < vec2);
        assert!((vec2 >= vec1));
    }

    #[test]
    fn test_partial_ord_greater_than() {
        let mut vec1 = TruenoVec::new();
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(4);

        assert!(vec1 > vec2);
        assert!((vec2 <= vec1));
    }

    #[test]
    fn test_partial_ord_less_equal() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);

        let mut vec3 = TruenoVec::new();
        vec3.push(1);
        vec3.push(3);

        assert!(vec1 <= vec2); // Equal
        assert!(vec1 <= vec3); // Less
        assert!((vec3 > vec1));
    }

    #[test]
    fn test_partial_ord_greater_equal() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);

        let mut vec3 = TruenoVec::new();
        vec3.push(1);
        vec3.push(1);

        assert!(vec1 >= vec2); // Equal
        assert!(vec1 >= vec3); // Greater
        assert!((vec3 < vec1));
    }

    #[test]
    fn test_partial_ord_different_lengths() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);

        // Shorter vector is less
        assert!(vec2 < vec1);
        assert!(vec1 > vec2);
        assert!((vec1 >= vec2));
    }

    #[test]
    fn test_partial_ord_empty_vectors() {
        let vec1: TruenoVec<i32> = TruenoVec::new();
        let vec2: TruenoVec<i32> = TruenoVec::new();

        assert!((vec1 >= vec2));
        assert!((vec1 <= vec2));
        assert!(vec1 <= vec2);
        assert!(vec1 >= vec2);
    }

    #[test]
    fn test_partial_ord_empty_vs_non_empty() {
        let vec1: TruenoVec<i32> = TruenoVec::new();
        let mut vec2 = TruenoVec::new();
        vec2.push(1);

        assert!(vec1 < vec2);
        assert!((vec2 >= vec1));
    }

    #[test]
    fn test_partial_ord_lexicographic() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(2);
        vec2.push(4);

        assert!(vec1 < vec2);
    }

    #[test]
    fn test_partial_ord_with_floats() {
        let mut vec1 = TruenoVec::new();
        vec1.push(1.5);
        vec1.push(2.5);

        let mut vec2 = TruenoVec::new();
        vec2.push(1.5);
        vec2.push(3.5);

        assert!(vec1 < vec2);
    }

    // ========================================================================
    // Phase 3.2: Ord Trait Tests
    // ========================================================================

    #[test]
    fn test_ord_cmp() {
        use std::cmp::Ordering;

        let mut vec1 = TruenoVec::new();
        vec1.push(1);
        vec1.push(2);

        let mut vec2 = TruenoVec::new();
        vec2.push(1);
        vec2.push(3);

        assert_eq!(vec1.cmp(&vec2), Ordering::Less);
        assert_eq!(vec2.cmp(&vec1), Ordering::Greater);
        assert_eq!(vec1.cmp(&vec1), Ordering::Equal);
    }

    #[test]
    fn test_ord_sorting() {
        let vec1 = TruenoVec::from(vec![3, 2, 1]);
        let vec2 = TruenoVec::from(vec![1, 2, 3]);
        let vec3 = TruenoVec::from(vec![2, 1, 3]);

        let mut vectors = [vec1.clone(), vec2.clone(), vec3.clone()];
        vectors.sort();

        assert_eq!(vectors[0], vec2);
        assert_eq!(vectors[1], vec3);
        assert_eq!(vectors[2], vec1);
    }

    #[test]
    fn test_ord_btree_map() {
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();

        let mut key1 = TruenoVec::new();
        key1.push(1);
        key1.push(2);

        let mut key2 = TruenoVec::new();
        key2.push(1);
        key2.push(3);

        map.insert(key1.clone(), "first");
        map.insert(key2.clone(), "second");

        assert_eq!(map.get(&key1), Some(&"first"));
        assert_eq!(map.get(&key2), Some(&"second"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_ord_btree_set() {
        use std::collections::BTreeSet;

        let mut set = BTreeSet::new();

        let vec1 = TruenoVec::from(vec![1, 2]);
        let vec2 = TruenoVec::from(vec![1, 3]);
        let vec3 = TruenoVec::from(vec![1, 2]); // Duplicate

        set.insert(vec1.clone());
        set.insert(vec2.clone());
        set.insert(vec3);

        assert_eq!(set.len(), 2); // vec3 is duplicate, not inserted
        assert!(set.contains(&vec1));
        assert!(set.contains(&vec2));
    }

    #[test]
    fn test_ord_min_max() {
        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![1, 2, 4]);
        let vec3 = TruenoVec::from(vec![1, 2, 2]);

        assert_eq!(vec1.clone().min(vec2.clone()), vec1);
        assert_eq!(vec1.clone().max(vec3), vec1);
        assert_eq!(vec2.clone().max(vec1), vec2);
    }

    #[test]
    fn test_ord_different_lengths() {
        use std::cmp::Ordering;

        let vec1 = TruenoVec::from(vec![1, 2]);
        let vec2 = TruenoVec::from(vec![1, 2, 3]);

        assert_eq!(vec1.cmp(&vec2), Ordering::Less);
        assert_eq!(vec2.cmp(&vec1), Ordering::Greater);
    }

    // ========================================================================
    // Phase 3.2: Hash Trait Tests
    // ========================================================================

    #[test]
    fn test_hash_basic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![1, 2, 3]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        vec1.hash(&mut hasher1);
        vec2.hash(&mut hasher2);

        // Same content should have same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_different_content() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![1, 2, 4]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        vec1.hash(&mut hasher1);
        vec2.hash(&mut hasher2);

        // Different content should (likely) have different hash
        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_different_lengths() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vec1 = TruenoVec::from(vec![1, 2]);
        let vec2 = TruenoVec::from(vec![1, 2, 3]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        vec1.hash(&mut hasher1);
        vec2.hash(&mut hasher2);

        // Different lengths should have different hash
        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_hashmap_insert() {
        use std::collections::HashMap;

        let mut map = HashMap::new();

        let key = TruenoVec::from(vec![1, 2, 3]);
        map.insert(key.clone(), "value");

        assert_eq!(map.get(&key), Some(&"value"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_hash_hashmap_multiple_keys() {
        use std::collections::HashMap;

        let mut map = HashMap::new();

        let key1 = TruenoVec::from(vec![1, 2, 3]);
        let key2 = TruenoVec::from(vec![4, 5, 6]);
        let key3 = TruenoVec::from(vec![1, 2]); // Different from key1

        map.insert(key1.clone(), "first");
        map.insert(key2.clone(), "second");
        map.insert(key3.clone(), "third");

        assert_eq!(map.get(&key1), Some(&"first"));
        assert_eq!(map.get(&key2), Some(&"second"));
        assert_eq!(map.get(&key3), Some(&"third"));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_hash_hashmap_update() {
        use std::collections::HashMap;

        let mut map = HashMap::new();

        let key = TruenoVec::from(vec![1, 2, 3]);
        map.insert(key.clone(), "first");
        map.insert(key.clone(), "second"); // Update

        assert_eq!(map.get(&key), Some(&"second"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_hash_hashset() {
        use std::collections::HashSet;

        let mut set = HashSet::new();

        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![4, 5, 6]);
        let vec3 = TruenoVec::from(vec![1, 2, 3]); // Duplicate

        set.insert(vec1.clone());
        set.insert(vec2.clone());
        set.insert(vec3);

        assert_eq!(set.len(), 2); // vec3 is duplicate
        assert!(set.contains(&vec1));
        assert!(set.contains(&vec2));
    }

    #[test]
    fn test_hash_empty_vector() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vec1: TruenoVec<i32> = TruenoVec::new();
        let vec2: TruenoVec<i32> = TruenoVec::new();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        vec1.hash(&mut hasher1);
        vec2.hash(&mut hasher2);

        // Empty vectors should have same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_with_strings() {
        use std::collections::HashMap;

        let mut map = HashMap::new();

        let mut key = TruenoVec::new();
        key.push(String::from("hello"));
        key.push(String::from("world"));

        map.insert(key.clone(), 42);

        assert_eq!(map.get(&key), Some(&42));
    }

    // ========================================================================
    // Phase 3.2: Integration Tests
    // ========================================================================

    #[test]
    fn test_comparison_and_hash_together() {
        use std::collections::{BTreeMap, HashMap};

        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![1, 2, 4]);

        // Can use in HashMap
        let mut hash_map = HashMap::new();
        hash_map.insert(vec1.clone(), "hash");
        assert_eq!(hash_map.get(&vec1), Some(&"hash"));

        // Can use in BTreeMap
        let mut btree_map = BTreeMap::new();
        btree_map.insert(vec1.clone(), "btree");
        assert_eq!(btree_map.get(&vec1), Some(&"btree"));

        // Can compare
        assert!(vec1 < vec2);
    }

    #[test]
    fn test_sorting_collection_of_vectors() {
        let mut vecs = [
            TruenoVec::from(vec![3, 2, 1]),
            TruenoVec::from(vec![1, 2, 3]),
            TruenoVec::from(vec![2, 1, 3]),
            TruenoVec::from(vec![1]),
        ];

        vecs.sort_unstable();

        assert_eq!(vecs[0].as_slice(), &[1]);
        assert_eq!(vecs[1].as_slice(), &[1, 2, 3]);
        assert_eq!(vecs[2].as_slice(), &[2, 1, 3]);
        assert_eq!(vecs[3].as_slice(), &[3, 2, 1]);
    }

    #[test]
    fn test_comparison_operators_comprehensive() {
        let vec1 = TruenoVec::from(vec![1, 2, 3]);
        let vec2 = TruenoVec::from(vec![1, 2, 3]);
        let vec3 = TruenoVec::from(vec![1, 2, 4]);
        let vec4 = TruenoVec::from(vec![1, 2]);

        // Equality
        assert!(vec1 == vec2);
        assert!(vec1 != vec3);

        // Less than
        assert!(vec1 < vec3);
        assert!(vec4 < vec1);

        // Greater than
        assert!(vec3 > vec1);
        assert!(vec1 > vec4);

        // Less than or equal
        assert!(vec1 <= vec2);
        assert!(vec1 <= vec3);

        // Greater than or equal
        assert!(vec1 >= vec2);
        assert!(vec3 >= vec1);
    }

    // Phase 3.3: Display Trait Tests

    #[test]
    fn test_display_empty() {
        let vec: TruenoVec<i32> = TruenoVec::new();
        assert_eq!(format!("{vec}"), "[]");
    }

    #[test]
    fn test_display_single_element() {
        let vec = TruenoVec::from(vec![42]);
        assert_eq!(format!("{vec}"), "[42]");
    }

    #[test]
    fn test_display_multiple_elements() {
        let vec = TruenoVec::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(format!("{vec}"), "[1, 2, 3, 4, 5]");
    }

    #[test]
    fn test_display_strings() {
        let vec = TruenoVec::from(vec!["hello", "world"]);
        assert_eq!(format!("{vec}"), "[hello, world]");
    }

    #[test]
    fn test_display_negative_numbers() {
        let vec = TruenoVec::from(vec![-1, -2, -3]);
        assert_eq!(format!("{vec}"), "[-1, -2, -3]");
    }

    #[test]
    fn test_display_chars() {
        let vec = TruenoVec::from(vec!['a', 'b', 'c']);
        assert_eq!(format!("{vec}"), "[a, b, c]");
    }

    #[test]
    fn test_display_after_push() {
        let mut vec = TruenoVec::new();
        vec.push(1);
        vec.push(2);
        assert_eq!(format!("{vec}"), "[1, 2]");
    }

    #[test]
    fn test_display_after_pop() {
        let mut vec = TruenoVec::from(vec![1, 2, 3]);
        vec.pop();
        assert_eq!(format!("{vec}"), "[1, 2]");
    }

    #[test]
    fn test_display_floating_point() {
        let vec = TruenoVec::from(vec![1.5, 2.7, 3.9]);
        assert_eq!(format!("{vec}"), "[1.5, 2.7, 3.9]");
    }

    #[test]
    fn test_display_with_println() {
        // Verify Display works with println! and format! macros
        let vec = TruenoVec::from(vec![1, 2, 3]);
        let display_output = format!("{vec}");
        let debug_output = format!("{vec:?}");

        // Display should produce clean output
        assert_eq!(display_output, "[1, 2, 3]");
        // Both Display and Debug should be available
        assert!(!debug_output.is_empty());
    }

    #[test]
    fn test_display_long_vector() {
        let vec: TruenoVec<i32> = (1..=20).collect();
        assert_eq!(
            format!("{vec}"),
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]"
        );
    }

    #[test]
    fn test_display_consistency_with_std_vec() {
        let trueno_vec = TruenoVec::from(vec![10, 20, 30]);
        let std_vec = vec![10, 20, 30];

        // Both should format the same way
        assert_eq!(format!("{trueno_vec}"), format!("{:?}", std_vec));
    }

    // Phase 3.3: Borrow Trait Tests

    #[test]
    fn test_borrow_to_slice() {
        use std::borrow::Borrow;

        let vec = TruenoVec::from(vec![1, 2, 3]);
        let slice: &[i32] = vec.borrow();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_borrow_empty() {
        use std::borrow::Borrow;

        let vec: TruenoVec<i32> = TruenoVec::new();
        let slice: &[i32] = vec.borrow();
        assert_eq!(slice, &[] as &[i32]);
    }

    #[test]
    fn test_borrow_mut_to_slice() {
        use std::borrow::BorrowMut;

        let mut vec = TruenoVec::from(vec![1, 2, 3]);
        let slice: &mut [i32] = vec.borrow_mut();
        slice[0] = 10;
        assert_eq!(slice, &[10, 2, 3]);
    }

    #[test]
    fn test_borrow_mut_modify() {
        use std::borrow::BorrowMut;

        let mut vec = TruenoVec::from(vec![5, 10, 15]);
        {
            let slice: &mut [i32] = vec.borrow_mut();
            for elem in slice.iter_mut() {
                *elem *= 2;
            }
        }
        assert_eq!(vec.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_borrow_with_generic_function() {
        use std::borrow::Borrow;

        fn sum_slice<T: Borrow<[i32]>>(data: &T) -> i32 {
            data.borrow().iter().sum()
        }

        let vec = TruenoVec::from(vec![1, 2, 3, 4]);
        assert_eq!(sum_slice(&vec), 10);
    }

    #[test]
    fn test_borrow_mut_with_generic_function() {
        use std::borrow::BorrowMut;

        fn double_all<T: BorrowMut<[i32]>>(data: &mut T) {
            for elem in data.borrow_mut().iter_mut() {
                *elem *= 2;
            }
        }

        let mut vec = TruenoVec::from(vec![1, 2, 3]);
        double_all(&mut vec);
        assert_eq!(vec.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_borrow_consistency_with_as_ref() {
        use std::borrow::Borrow;

        let vec = TruenoVec::from(vec![7, 8, 9]);
        let borrowed: &[i32] = vec.borrow();
        let as_ref: &[i32] = vec.as_ref();

        assert_eq!(borrowed, as_ref);
        assert_eq!(borrowed.as_ptr(), as_ref.as_ptr());
    }

    #[test]
    fn test_borrow_mut_consistency_with_as_mut() {
        use std::borrow::BorrowMut;

        let mut vec1 = TruenoVec::from(vec![1, 2, 3]);
        let mut vec2 = TruenoVec::from(vec![1, 2, 3]);

        let borrowed: &mut [i32] = vec1.borrow_mut();
        let as_mut: &mut [i32] = vec2.as_mut();

        borrowed[0] = 99;
        as_mut[0] = 99;

        assert_eq!(vec1.as_slice(), vec2.as_slice());
    }

    #[test]
    fn test_borrow_hashmap_lookup() {
        use std::borrow::Borrow;
        use std::collections::HashMap;

        let mut map: HashMap<Vec<i32>, &str> = HashMap::new();
        map.insert(vec![1, 2, 3], "found");

        let key = TruenoVec::from(vec![1, 2, 3]);
        // This works because TruenoVec implements Borrow<[T]>
        // and Vec's Borrow<[T]> matches
        let borrowed: &[i32] = key.borrow();

        // Verify the map has the value and borrow works
        assert_eq!(map.get(&vec![1, 2, 3]), Some(&"found"));
        assert_eq!(borrowed, &[1, 2, 3]);
    }

    #[test]
    fn test_borrow_slice_methods() {
        use std::borrow::Borrow;

        let vec = TruenoVec::from(vec![1, 2, 3, 4, 5]);
        let slice: &[i32] = vec.borrow();

        assert_eq!(slice.len(), 5);
        assert_eq!(slice.first(), Some(&1));
        assert_eq!(slice.last(), Some(&5));
        assert_eq!(&slice[1..3], &[2, 3]);
    }
}

// ============================================================================
// Property-Based Tests (Tier 2: 1-5 minutes)
// ============================================================================
