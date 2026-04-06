#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod property_tests {
    use super::super::*;
    use proptest::prelude::*;

    // Property 1: Length Invariant
    // After pushing n elements, len() == n
    proptest! {
        #[test]
        fn prop_length_after_pushes(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut v = TruenoVec::new();
            let expected_len = elements.len();

            for elem in elements {
                v.push(elem);
            }

            prop_assert_eq!(v.len(), expected_len);
        }
    }

    // Property 2: Capacity Bound
    // capacity() >= len() at all times
    proptest! {
        #[test]
        fn prop_capacity_bound(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut v = TruenoVec::new();
            prop_assert!(v.capacity() >= v.len());

            for elem in elements {
                v.push(elem);
                prop_assert!(v.capacity() >= v.len());
            }
        }
    }

    // Property 3: Push/Pop Symmetry
    // pop() returns exactly what was pushed
    proptest! {
        #[test]
        fn prop_push_pop_inverse(
            initial in prop::collection::vec(any::<i32>(), 0..50),
            x in any::<i32>()
        ) {
            let mut v = TruenoVec::new();
            for val in &initial {
                v.push(*val);
            }

            let len_before = v.len();
            v.push(x);
            let popped = v.pop();

            prop_assert_eq!(popped, Some(x));
            prop_assert_eq!(v.len(), len_before);
        }
    }

    // Property 4: Index Access Correctness
    // get(i) returns the i-th pushed element
    proptest! {
        #[test]
        fn prop_index_access(elements in prop::collection::vec(any::<i32>(), 1..100)) {
            let mut v = TruenoVec::new();
            for elem in &elements {
                v.push(*elem);
            }

            for (i, expected) in elements.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // Property 5: Out of Bounds Access
    // get(i) returns None when i >= len()
    proptest! {
        #[test]
        fn prop_out_of_bounds(
            elements in prop::collection::vec(any::<i32>(), 0..50),
            invalid_index in 100usize..200
        ) {
            let mut v = TruenoVec::new();
            for elem in elements {
                v.push(elem);
            }

            prop_assert_eq!(v.get(invalid_index), None);
        }
    }

    // Property 6: Growth Factor
    // Capacity doubles on reallocation (exponential growth)
    proptest! {
        #[test]
        fn prop_exponential_growth(count in 1usize..20) {
            let mut v = TruenoVec::<i32>::new();
            let mut prev_capacity = 0;

            for i in 0..count {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                v.push(i as i32);
                if v.capacity() > prev_capacity && prev_capacity > 0 {
                    // Capacity should double (or be the initial 1)
                    prop_assert!(v.capacity() == prev_capacity * 2 || prev_capacity == 0);
                }
                prev_capacity = v.capacity();
            }
        }
    }

    // Property 7: Comparison with std::vec::Vec
    // TruenoVec behavior matches std::vec::Vec
    proptest! {
        #[test]
        fn prop_matches_std_vec(operations in prop::collection::vec(0i32..100, 0..50)) {
            let mut trueno = TruenoVec::new();
            let mut std_vec = Vec::new();

            for op in operations {
                if op % 3 == 0 {
                    // Push operation
                    trueno.push(op);
                    std_vec.push(op);
                } else {
                    // Pop operation
                    prop_assert_eq!(trueno.pop(), std_vec.pop());
                }

                prop_assert_eq!(trueno.len(), std_vec.len());
                prop_assert_eq!(trueno.is_empty(), std_vec.is_empty());
            }
        }
    }

    // Property 8: Empty Vector Properties
    proptest! {
        #[test]
        fn prop_empty_invariants(capacity in 0usize..100) {
            let v: TruenoVec<i32> = if capacity == 0 {
                TruenoVec::new()
            } else {
                TruenoVec::with_capacity(capacity)
            };

            prop_assert!(v.is_empty());
            prop_assert_eq!(v.len(), 0);
            prop_assert_eq!(v.get(0), None);
        }
    }

    // Property 9: Mutable Access Correctness
    proptest! {
        #[test]
        fn prop_get_mut_modifies(
            mut elements in prop::collection::vec(any::<i32>(), 1..50),
            index in 0usize..50,
            new_value in any::<i32>()
        ) {
            if index >= elements.len() {
                return Ok(());
            }

            let mut v = TruenoVec::new();
            for elem in &elements {
                v.push(*elem);
            }

            if let Some(elem) = v.get_mut(index) {
                *elem = new_value;
            }

            prop_assert_eq!(v.get(index), Some(&new_value));
            elements[index] = new_value;

            for (i, expected) in elements.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // Property 10: Repeated Push/Pop Maintains Invariants
    proptest! {
        #[test]
        fn prop_repeated_operations(ops in prop::collection::vec((any::<bool>(), any::<i32>()), 0..100)) {
            let mut v = TruenoVec::new();
            let mut shadow_vec = Vec::new();

            for (is_push, value) in ops {
                if is_push {
                    v.push(value);
                    shadow_vec.push(value);
                } else if !shadow_vec.is_empty() {
                    prop_assert_eq!(v.pop(), shadow_vec.pop());
                }

                prop_assert_eq!(v.len(), shadow_vec.len());
                prop_assert!(v.capacity() >= v.len());
            }
        }
    }

    // Property 11: Clear Empties Vector
    proptest! {
        #[test]
        fn prop_clear_empties(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut v = TruenoVec::new();
            for elem in elements {
                v.push(elem);
            }

            let capacity_before = v.capacity();
            v.clear();

            prop_assert!(v.is_empty());
            prop_assert_eq!(v.len(), 0);
            prop_assert_eq!(v.capacity(), capacity_before); // Capacity unchanged
        }
    }

    // Property 12: Insert Maintains Order
    proptest! {
        #[test]
        fn prop_insert_maintains_order(
            mut elements in prop::collection::vec(any::<i32>(), 1..50),
            index in 0usize..50,
            value in any::<i32>()
        ) {
            let len = elements.len();
            if index > len {
                return Ok(());
            }

            let mut v = TruenoVec::new();
            for elem in &elements {
                v.push(*elem);
            }

            v.insert(index, value);
            elements.insert(index, value);

            prop_assert_eq!(v.len(), elements.len());
            for (i, expected) in elements.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // Property 13: Remove Maintains Order
    proptest! {
        #[test]
        fn prop_remove_maintains_order(
            mut elements in prop::collection::vec(any::<i32>(), 1..50),
            index in 0usize..50
        ) {
            if elements.is_empty() || index >= elements.len() {
                return Ok(());
            }

            let mut v = TruenoVec::new();
            for elem in &elements {
                v.push(*elem);
            }

            let removed_trueno = v.remove(index);
            let removed_std = elements.remove(index);

            prop_assert_eq!(removed_trueno, removed_std);
            prop_assert_eq!(v.len(), elements.len());

            for (i, expected) in elements.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // Property 14: Insert Then Remove Inverse
    proptest! {
        #[test]
        fn prop_insert_remove_inverse(
            elements in prop::collection::vec(any::<i32>(), 0..50),
            index in 0usize..50,
            value in any::<i32>()
        ) {
            if index > elements.len() {
                return Ok(());
            }

            let mut v = TruenoVec::new();
            for elem in &elements {
                v.push(*elem);
            }

            let len_before = v.len();
            v.insert(index, value);
            let removed = v.remove(index);

            prop_assert_eq!(removed, value);
            prop_assert_eq!(v.len(), len_before);

            // Verify original elements are intact
            for (i, expected) in elements.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // Property 15: Clear Then Reuse
    proptest! {
        #[test]
        fn prop_clear_then_reuse(
            first_batch in prop::collection::vec(any::<i32>(), 0..50),
            second_batch in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let mut v = TruenoVec::new();

            // Fill with first batch
            for elem in &first_batch {
                v.push(*elem);
            }

            // Clear
            v.clear();
            prop_assert_eq!(v.len(), 0);

            // Fill with second batch
            for elem in &second_batch {
                v.push(*elem);
            }

            // Verify second batch is correct
            prop_assert_eq!(v.len(), second_batch.len());
            for (i, expected) in second_batch.iter().enumerate() {
                prop_assert_eq!(v.get(i), Some(expected));
            }
        }
    }

    // ========================================================================
    // Property-Based Tests for Conversion Traits
    // ========================================================================

    // Property 16: From<Vec<T>> Preserves Elements
    proptest! {
        #[test]
        fn prop_from_vec_preserves_elements(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let expected = elements.clone();
            let trueno_vec: TruenoVec<i32> = TruenoVec::from(elements);

            prop_assert_eq!(trueno_vec.len(), expected.len());
            for (i, &expected_val) in expected.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 17: From<&[T]> Preserves Elements
    proptest! {
        #[test]
        fn prop_from_slice_preserves_elements(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let slice = elements.as_slice();
            let trueno_vec: TruenoVec<i32> = TruenoVec::from(slice);

            prop_assert_eq!(trueno_vec.len(), elements.len());
            for (i, &expected_val) in elements.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 18: FromIterator Matches Vec::from_iter
    proptest! {
        #[test]
        fn prop_from_iterator_matches_vec(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let trueno_vec: TruenoVec<i32> = elements.iter().copied().collect();
            let std_vec: Vec<i32> = elements;

            prop_assert_eq!(trueno_vec.len(), std_vec.len());
            for (i, &expected_val) in std_vec.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 19: Extend Matches Vec::extend
    proptest! {
        #[test]
        fn prop_extend_matches_vec(
            initial in prop::collection::vec(any::<i32>(), 0..50),
            extension in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let mut trueno_vec = TruenoVec::new();
            let mut std_vec = Vec::new();

            for &elem in &initial {
                trueno_vec.push(elem);
                std_vec.push(elem);
            }

            trueno_vec.extend(extension.iter().copied());
            std_vec.extend(extension.iter().copied());

            prop_assert_eq!(trueno_vec.len(), std_vec.len());
            for (i, &expected_val) in std_vec.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 20: Extend Maintains Length Invariant
    proptest! {
        #[test]
        fn prop_extend_length_invariant(
            initial in prop::collection::vec(any::<i32>(), 0..50),
            extension in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let mut trueno_vec = TruenoVec::new();

            for elem in &initial {
                trueno_vec.push(*elem);
            }

            let len_before = trueno_vec.len();
            trueno_vec.extend(extension.iter().copied());

            prop_assert_eq!(trueno_vec.len(), len_before + extension.len());
        }
    }

    // Property 21: Multiple Extends Preserve Order
    proptest! {
        #[test]
        fn prop_multiple_extends_preserve_order(
            batch1 in prop::collection::vec(any::<i32>(), 0..30),
            batch2 in prop::collection::vec(any::<i32>(), 0..30),
            batch3 in prop::collection::vec(any::<i32>(), 0..30)
        ) {
            let mut trueno_vec = TruenoVec::new();
            trueno_vec.extend(batch1.iter().copied());
            trueno_vec.extend(batch2.iter().copied());
            trueno_vec.extend(batch3.iter().copied());

            let mut expected = Vec::new();
            expected.extend(&batch1);
            expected.extend(&batch2);
            expected.extend(&batch3);

            prop_assert_eq!(trueno_vec.len(), expected.len());
            for (i, &expected_val) in expected.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 22: FromIterator with Range
    proptest! {
        #[test]
        fn prop_from_iterator_range(start in 0i32..100, count in 0usize..100) {
            let end = start.saturating_add(i32::try_from(count).unwrap_or(i32::MAX));
            let trueno_vec: TruenoVec<i32> = (start..end).collect();

            prop_assert_eq!(trueno_vec.len(), count);
            for (i, expected_val) in (start..end).enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 23: Conversion Roundtrip Vec -> TruenoVec -> Iter -> Vec
    proptest! {
        #[test]
        fn prop_conversion_roundtrip(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let original = elements.clone();
            let trueno_vec: TruenoVec<i32> = TruenoVec::from(elements);
            let back_to_vec: Vec<i32> = trueno_vec.iter().copied().collect();

            prop_assert_eq!(original, back_to_vec);
        }
    }

    // Property 24: Extend with Empty Iterator is No-op
    proptest! {
        #[test]
        fn prop_extend_empty_noop(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut trueno_vec = TruenoVec::new();
            for elem in &elements {
                trueno_vec.push(*elem);
            }

            let len_before = trueno_vec.len();
            let empty: Vec<i32> = vec![];
            trueno_vec.extend(empty);

            prop_assert_eq!(trueno_vec.len(), len_before);
            for (i, &expected_val) in elements.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 25: FromIterator Filter and Map Composition
    proptest! {
        #[test]
        fn prop_from_iterator_filter_map(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let trueno_vec: TruenoVec<i32> = elements
                .iter()
                .filter(|&&x| x % 2 == 0)
                .map(|&x| x.wrapping_mul(2))
                .collect();

            let expected: Vec<i32> = elements
                .iter()
                .filter(|&&x| x % 2 == 0)
                .map(|&x| x.wrapping_mul(2))
                .collect();

            prop_assert_eq!(trueno_vec.len(), expected.len());
            for (i, &expected_val) in expected.iter().enumerate() {
                prop_assert_eq!(trueno_vec.get(i), Some(&expected_val));
            }
        }
    }

    // Property 26: Extend Maintains Capacity Bound
    proptest! {
        #[test]
        fn prop_extend_capacity_bound(
            initial in prop::collection::vec(any::<i32>(), 0..50),
            extension in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let mut trueno_vec = TruenoVec::new();
            for elem in &initial {
                trueno_vec.push(*elem);
            }

            trueno_vec.extend(extension.iter().copied());

            // Capacity must always be >= len
            prop_assert!(trueno_vec.capacity() >= trueno_vec.len());
        }
    }

    // Property 27: From<Vec<T>> and From<&[T]> Produce Same Result
    proptest! {
        #[test]
        fn prop_from_vec_and_slice_equivalent(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let from_vec: TruenoVec<i32> = TruenoVec::from(elements.clone());
            let from_slice: TruenoVec<i32> = TruenoVec::from(elements.as_slice());

            prop_assert_eq!(from_vec.len(), from_slice.len());
            for i in 0..from_vec.len() {
                prop_assert_eq!(from_vec.get(i), from_slice.get(i));
            }
        }
    }

    // Property 28: Collect and Extend are Equivalent for Same Data
    proptest! {
        #[test]
        fn prop_collect_extend_equivalent(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let collected: TruenoVec<i32> = elements.iter().copied().collect();

            let mut extended = TruenoVec::new();
            extended.extend(elements.iter().copied());

            prop_assert_eq!(collected.len(), extended.len());
            for i in 0..collected.len() {
                prop_assert_eq!(collected.get(i), extended.get(i));
            }
        }
    }

    // ========================================================================
    // Phase 3.1: Property-Based Tests for Deref/AsRef Traits
    // ========================================================================

    // Property 29: Deref Produces Same Slice as as_slice
    proptest! {
        #[test]
        fn prop_deref_equals_as_slice(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let vec = TruenoVec::from(elements);
            let via_deref: &[i32] = &vec;
            let via_as_slice = vec.as_slice();

            prop_assert_eq!(via_deref, via_as_slice);
            prop_assert_eq!(via_deref.len(), via_as_slice.len());
            prop_assert_eq!(via_deref.as_ptr(), via_as_slice.as_ptr());
        }
    }

    // Property 30: AsRef<[T]> Produces Same Slice as as_slice
    proptest! {
        #[test]
        fn prop_as_ref_equals_as_slice(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let vec = TruenoVec::from(elements);
            let via_as_ref: &[i32] = vec.as_ref();
            let via_as_slice = vec.as_slice();

            prop_assert_eq!(via_as_ref, via_as_slice);
            prop_assert_eq!(via_as_ref.len(), vec.len());
        }
    }

    // Property 31: Deref Enables Slice Methods
    proptest! {
        #[test]
        fn prop_deref_slice_methods(elements in prop::collection::vec(any::<i32>(), 1..100)) {
            let expected_first = elements.first().copied();
            let expected_last = elements.last().copied();
            let expected_len = elements.len();
            let expected_empty = elements.is_empty();
            let vec = TruenoVec::from(elements);

            // Test various slice methods through Deref
            prop_assert_eq!(vec.first().copied(), expected_first);
            prop_assert_eq!(vec.last().copied(), expected_last);
            prop_assert_eq!(vec.len(), expected_len);
            prop_assert_eq!(vec.is_empty(), expected_empty);

            for elem in &vec {
                prop_assert!(vec.contains(elem));
            }
        }
    }

    // Property 32: DerefMut Enables Mutable Slice Methods
    proptest! {
        #[test]
        fn prop_deref_mut_sorting(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut vec = TruenoVec::from(elements.clone());
            let mut std_vec = elements;

            // Sort using DerefMut
            vec.sort_unstable();
            std_vec.sort_unstable();

            prop_assert_eq!(vec.len(), std_vec.len());
            for i in 0..vec.len() {
                prop_assert_eq!(vec.get(i), std_vec.get(i));
            }
        }
    }

    // Property 33: DerefMut Reverse Correctness
    proptest! {
        #[test]
        fn prop_deref_mut_reverse(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let mut vec = TruenoVec::from(elements.clone());
            let mut std_vec = elements;

            vec.reverse();
            std_vec.reverse();

            prop_assert_eq!(vec.len(), std_vec.len());
            for i in 0..vec.len() {
                prop_assert_eq!(vec.get(i), std_vec.get(i));
            }
        }
    }

    // Property 34: AsRef Generic Function Interoperability
    proptest! {
        #[test]
        fn prop_as_ref_generic_interop(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            fn sum_via_as_ref<T: AsRef<[i32]>>(data: T) -> i32 {
                data.as_ref().iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
            }

            let vec = TruenoVec::from(elements.clone());
            let sum_trueno = sum_via_as_ref(&vec);
            let sum_std = sum_via_as_ref(&elements);

            prop_assert_eq!(sum_trueno, sum_std);
        }
    }

    // Property 35: AsMut Generic Function Interoperability
    proptest! {
        #[test]
        fn prop_as_mut_generic_interop(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            fn double_via_as_mut<T: AsMut<[i32]>>(mut data: T) {
                for x in data.as_mut() {
                    *x = x.wrapping_mul(2);
                }
            }

            let mut vec = TruenoVec::from(elements.clone());
            let mut std_vec = elements;

            double_via_as_mut(&mut vec);
            double_via_as_mut(&mut std_vec);

            prop_assert_eq!(vec.len(), std_vec.len());
            for i in 0..vec.len() {
                prop_assert_eq!(vec.get(i), std_vec.get(i));
            }
        }
    }

    // Property 36: Deref Coercion with Function Parameters
    proptest! {
        #[test]
        fn prop_deref_function_param(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            fn process_slice(s: &[i32]) -> i32 {
                s.iter().map(|x| x.wrapping_mul(2)).fold(0i32, i32::wrapping_add)
            }

            let vec = TruenoVec::from(elements.clone());
            let result_trueno = process_slice(&vec);
            let result_std = process_slice(&elements);

            prop_assert_eq!(result_trueno, result_std);
        }
    }

    // Property 37: as_slice Length Invariant
    proptest! {
        #[test]
        fn prop_as_slice_length_invariant(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let vec = TruenoVec::from(elements.clone());
            let slice = vec.as_slice();

            prop_assert_eq!(slice.len(), vec.len());
            prop_assert_eq!(slice.len(), elements.len());
        }
    }

    // Property 38: Slice Methods Equivalence with std::Vec
    proptest! {
        #[test]
        fn prop_slice_methods_std_vec_equivalence(elements in prop::collection::vec(any::<i32>(), 1..100)) {
            let vec = TruenoVec::from(elements.clone());

            prop_assert_eq!(vec.first(), elements.first());
            prop_assert_eq!(vec.last(), elements.last());
            prop_assert_eq!(vec.is_empty(), elements.is_empty());

            for (i, elem) in elements.iter().enumerate() {
                prop_assert_eq!(vec.get(i), Some(elem));
            }
        }
    }

    // Property 39: DerefMut Fill Correctness
    proptest! {
        #[test]
        fn prop_deref_mut_fill(elements in prop::collection::vec(any::<i32>(), 0..100), fill_value in any::<i32>()) {
            let mut vec = TruenoVec::from(elements.clone());
            vec.fill(fill_value);

            prop_assert_eq!(vec.len(), elements.len());
            for i in 0..vec.len() {
                prop_assert_eq!(vec.get(i), Some(&fill_value));
            }
        }
    }

    // Property 40: Multiple Deref Calls Return Same Pointer
    proptest! {
        #[test]
        fn prop_multiple_deref_same_pointer(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let vec = TruenoVec::from(elements);
            let slice1: &[i32] = &vec;
            let slice2: &[i32] = &vec;

            prop_assert_eq!(slice1.as_ptr(), slice2.as_ptr());
            prop_assert_eq!(slice1.len(), slice2.len());
        }
    }

    // Property 41: AsRef and AsRef<TruenoVec> Consistency
    proptest! {
        #[test]
        fn prop_as_ref_self_consistency(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            let vec = TruenoVec::from(elements);
            let vec_ref: &TruenoVec<i32> = vec.as_ref();

            prop_assert_eq!(vec.len(), vec_ref.len());
            prop_assert_eq!(vec.capacity(), vec_ref.capacity());
            prop_assert_eq!(vec.is_empty(), vec_ref.is_empty());
        }
    }

    // Property 42: Deref and Index Consistency
    proptest! {
        #[test]
        fn prop_deref_index_consistency(elements in prop::collection::vec(any::<i32>(), 1..100)) {
            let vec = TruenoVec::from(elements.clone());

            for (i, elem) in elements.iter().enumerate() {
                prop_assert_eq!(&vec[i], elem);
                prop_assert_eq!(vec.get(i), Some(elem));
            }
        }
    }

    // Property 43: Slice Range Operations via Deref
    proptest! {
        #[test]
        fn prop_slice_range_operations(elements in prop::collection::vec(any::<i32>(), 5..100)) {
            let vec = TruenoVec::from(elements.clone());
            let len = vec.len();

            // Test various range operations
            if len > 2 {
                let range = 1..(len - 1);
                let vec_slice = vec.as_slice();
                prop_assert_eq!(&vec_slice[range.clone()], &elements[range]);
            }

            // Test starts_with
            if len >= 3 {
                prop_assert!(vec.starts_with(&elements[0..3]));
            }

            // Test ends_with
            if len >= 3 {
                prop_assert!(vec.ends_with(&elements[(len - 3)..len]));
            }
        }
    }

    // ========================================================================
    // Phase 3.2: Property-Based Tests for PartialOrd/Ord/Hash
    // ========================================================================

    // Property 44: PartialOrd Matches std::Vec
    proptest! {
        #[test]
        fn prop_partial_ord_matches_vec(
            elements1 in prop::collection::vec(any::<i32>(), 0..50),
            elements2 in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let trueno1 = TruenoVec::from(elements1.clone());
            let trueno2 = TruenoVec::from(elements2.clone());

            // Compare TruenoVec with same comparison as std::Vec
            prop_assert_eq!(trueno1 < trueno2, elements1 < elements2);
            prop_assert_eq!(trueno1 > trueno2, elements1 > elements2);
            prop_assert_eq!(trueno1 <= trueno2, elements1 <= elements2);
            prop_assert_eq!(trueno1 >= trueno2, elements1 >= elements2);
        }
    }

    // Property 45: Ord Transitivity
    proptest! {
        #[test]
        fn prop_ord_transitivity(
            elements1 in prop::collection::vec(any::<i32>(), 0..20),
            elements2 in prop::collection::vec(any::<i32>(), 0..20),
            elements3 in prop::collection::vec(any::<i32>(), 0..20)
        ) {
            let vec1 = TruenoVec::from(elements1);
            let vec2 = TruenoVec::from(elements2);
            let vec3 = TruenoVec::from(elements3);

            // If a < b and b < c, then a < c
            if vec1 < vec2 && vec2 < vec3 {
                prop_assert!(vec1 < vec3);
            }
        }
    }

    // Property 46: Ord Antisymmetry
    proptest! {
        #[test]
        fn prop_ord_antisymmetry(
            elements1 in prop::collection::vec(any::<i32>(), 0..50),
            elements2 in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            let vec1 = TruenoVec::from(elements1);
            let vec2 = TruenoVec::from(elements2);

            // If a <= b and b <= a, then a == b
            if vec1 <= vec2 && vec2 <= vec1 {
                prop_assert_eq!(vec1, vec2);
            }
        }
    }

    // Property 47: Ord cmp Consistency with Operators
    proptest! {
        #[test]
        fn prop_ord_cmp_consistency(
            elements1 in prop::collection::vec(any::<i32>(), 0..50),
            elements2 in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            use std::cmp::Ordering;

            let vec1 = TruenoVec::from(elements1);
            let vec2 = TruenoVec::from(elements2);

            let ordering = vec1.cmp(&vec2);

            match ordering {
                Ordering::Less => prop_assert!(vec1 < vec2),
                Ordering::Greater => prop_assert!(vec1 > vec2),
                Ordering::Equal => prop_assert_eq!(vec1, vec2),
            }
        }
    }

    // Property 48: Sorting Preserves Elements
    proptest! {
        #[test]
        fn prop_sorting_preserves_elements(vecs in prop::collection::vec(
            prop::collection::vec(any::<i32>(), 0..10),
            0..20
        )) {
            let mut trueno_vecs: Vec<TruenoVec<i32>> = vecs.iter()
                .map(|v| TruenoVec::from(v.clone()))
                .collect();

            let original_count = trueno_vecs.len();
            trueno_vecs.sort();

            // Sorting preserves number of elements
            prop_assert_eq!(trueno_vecs.len(), original_count);

            // Sorting produces sorted order
            for i in 0..trueno_vecs.len().saturating_sub(1) {
                prop_assert!(trueno_vecs[i] <= trueno_vecs[i + 1]);
            }
        }
    }

    // Property 49: Hash Equality Implies Same Hash
    proptest! {
        #[test]
        fn prop_hash_equality_same_hash(elements in prop::collection::vec(any::<i32>(), 0..100)) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let vec1 = TruenoVec::from(elements.clone());
            let vec2 = TruenoVec::from(elements);

            // Equal vectors must have equal hashes
            prop_assert_eq!(&vec1, &vec2);

            let mut hasher1 = DefaultHasher::new();
            let mut hasher2 = DefaultHasher::new();

            vec1.hash(&mut hasher1);
            vec2.hash(&mut hasher2);

            prop_assert_eq!(hasher1.finish(), hasher2.finish());
        }
    }

    // Property 50: Hash Consistent with Eq
    proptest! {
        #[test]
        fn prop_hash_consistent_with_eq(
            elements1 in prop::collection::vec(any::<i32>(), 0..50),
            elements2 in prop::collection::vec(any::<i32>(), 0..50)
        ) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let vec1 = TruenoVec::from(elements1);
            let vec2 = TruenoVec::from(elements2);

            if vec1 == vec2 {
                let mut hasher1 = DefaultHasher::new();
                let mut hasher2 = DefaultHasher::new();

                vec1.hash(&mut hasher1);
                vec2.hash(&mut hasher2);

                // If equal, hashes must be equal
                prop_assert_eq!(hasher1.finish(), hasher2.finish());
            }
        }
    }

    // Property 51: HashMap Operations Correctness
    proptest! {
        #[test]
        fn prop_hashmap_operations(
            keys in prop::collection::vec(prop::collection::vec(any::<i32>(), 0..10), 0..20)
        ) {
            use std::collections::HashMap;

            let mut map = HashMap::new();

            // Insert all keys (later duplicates will overwrite earlier ones)
            for (i, key_vec) in keys.iter().enumerate() {
                let key = TruenoVec::from(key_vec.clone());
                map.insert(key, i);
            }

            // Verify that we can retrieve values
            // For duplicates, the last inserted value should be present
            for key_vec in &keys {
                let key = TruenoVec::from(key_vec.clone());

                // Find the last occurrence of this key in the original list
                let last_index = keys.iter()
                    .enumerate()
                    .rev()
                    .find(|(_, k)| k == &key_vec)
                    .map(|(idx, _)| idx)
                    .unwrap();

                if let Some(&value) = map.get(&key) {
                    // Value should be the last inserted one for this key
                    prop_assert_eq!(value, last_index);
                }
            }
        }
    }

    // Property 52: BTreeMap Ordering
    proptest! {
        #[test]
        fn prop_btreemap_ordering(
            keys in prop::collection::vec(prop::collection::vec(any::<i32>(), 0..10), 0..20)
        ) {
            use std::collections::BTreeMap;

            let mut map = BTreeMap::new();

            for (i, key_vec) in keys.iter().enumerate() {
                let key = TruenoVec::from(key_vec.clone());
                map.insert(key, i);
            }

            // Keys should be in sorted order
            let keys_in_order: Vec<_> = map.keys().collect();
            for i in 0..keys_in_order.len().saturating_sub(1) {
                prop_assert!(keys_in_order[i] <= keys_in_order[i + 1]);
            }
        }
    }

    // Property 53: Lexicographic Ordering Matches std::Vec
    proptest! {
        #[test]
        fn prop_lexicographic_ordering(
            elements1 in prop::collection::vec(any::<i32>(), 0..50),
            elements2 in prop::collection::vec(any::<i32>(), 0..50)
        ) {


            let trueno1 = TruenoVec::from(elements1.clone());
            let trueno2 = TruenoVec::from(elements2.clone());

            // Ordering should match std::Vec
            prop_assert_eq!(trueno1.cmp(&trueno2), elements1.cmp(&elements2));
        }
    }

    // ========================================================================
    // Phase 3.3: Property-Based Tests for Display and Borrow
    // ========================================================================

    // Property 54: Display Format Consistency
    proptest! {
        #[test]
        fn prop_display_format_consistency(elements in prop::collection::vec(any::<i32>(), 0..20)) {
            let vec = TruenoVec::from(elements.clone());
            let display_output = format!("{vec}");
            let expected_output = format!("{elements:?}");

            // Display should match std::Vec's Debug output
            prop_assert_eq!(display_output, expected_output);
        }
    }

    // Property 55: Display Empty Vector
    proptest! {
        #[test]
        fn prop_display_empty(_n in 0..100usize) {
            let vec: TruenoVec<i32> = TruenoVec::new();
            prop_assert_eq!(format!("{}", vec), "[]");
        }
    }

    // Property 56: Display After Operations
    proptest! {
        #[test]
        fn prop_display_after_push(
            initial in prop::collection::vec(any::<i32>(), 0..10),
            x in any::<i32>()
        ) {
            let mut vec = TruenoVec::from(initial.clone());
            vec.push(x);

            let mut expected = initial;
            expected.push(x);

            prop_assert_eq!(format!("{}", vec), format!("{:?}", expected));
        }
    }

    // Property 57: Borrow Equivalence with AsRef
    proptest! {
        #[test]
        fn prop_borrow_equals_as_ref(elements in prop::collection::vec(any::<i32>(), 0..50)) {
            use std::borrow::Borrow;

            let vec = TruenoVec::from(elements);
            let borrowed: &[i32] = vec.borrow();
            let as_ref: &[i32] = vec.as_ref();

            prop_assert_eq!(borrowed, as_ref);
            prop_assert_eq!(borrowed.as_ptr(), as_ref.as_ptr());
        }
    }

    // Property 58: BorrowMut Equivalence with AsMut
    proptest! {
        #[test]
        fn prop_borrow_mut_equals_as_mut(elements in prop::collection::vec(any::<i32>(), 0..50)) {
            use std::borrow::BorrowMut;

            let mut vec1 = TruenoVec::from(elements.clone());
            let mut vec2 = TruenoVec::from(elements);

            let borrowed_slice: &mut [i32] = vec1.borrow_mut();
            let as_mut_slice: &mut [i32] = vec2.as_mut();

            let borrowed_ptr = borrowed_slice.as_ptr();
            let as_mut_ptr = as_mut_slice.as_ptr();

            // Both should return the same underlying pointer
            prop_assert_eq!(borrowed_ptr, vec1.as_slice().as_ptr());
            prop_assert_eq!(as_mut_ptr, vec2.as_slice().as_ptr());
        }
    }

    // Property 59: Borrow Slice Operations
    proptest! {
        #[test]
        fn prop_borrow_slice_operations(elements in prop::collection::vec(any::<i32>(), 1..50)) {
            use std::borrow::Borrow;

            let vec = TruenoVec::from(elements.clone());
            let slice: &[i32] = vec.borrow();

            // All slice operations should work correctly
            prop_assert_eq!(slice.len(), elements.len());
            prop_assert_eq!(slice.first(), elements.first());
            prop_assert_eq!(slice.last(), elements.last());
            prop_assert_eq!(slice, elements.as_slice());
        }
    }

    // Property 60: Display Respects Element Display Impl
    proptest! {
        #[test]
        fn prop_display_respects_element_display(elements in prop::collection::vec(any::<u32>(), 0..15)) {
            let vec = TruenoVec::from(elements.clone());
            let display_str = format!("{vec}");

            // Display should contain each element's string representation
            for elem in &elements {
                let elem_str = format!("{elem}");
                prop_assert!(display_str.contains(&elem_str));
            }
        }
    }
}

// ============================================================================
// Formal Verification (Tier 3: ON-MERGE/NIGHTLY)
// ============================================================================

#[cfg(kani)]
#[cfg_attr(test, mutants::skip)] // Skip mutation testing for Kani verification code
mod verification {
    use super::*;

    /// Verifies that capacity >= len invariant holds for all execution paths
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_capacity_invariant() {
        let mut v: TruenoVec<u8> = TruenoVec::new();

        for _ in 0..10 {
            let x: u8 = kani::any();
            v.push(x);
            assert!(v.capacity() >= v.len());
        }
    }

    /// Verifies that push followed by pop returns the same value
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_push_pop_correctness() {
        let mut v: TruenoVec<u32> = TruenoVec::new();
        let x: u32 = kani::any();

        let len_before = v.len();
        v.push(x);
        assert_eq!(v.len(), len_before + 1);

        let popped = v.pop();
        assert_eq!(popped, Some(x));
        assert_eq!(v.len(), len_before);
    }

    /// Verifies that get returns None for out-of-bounds indices
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_bounds_checking() {
        let mut v: TruenoVec<u8> = TruenoVec::new();

        // Push some elements
        for i in 0..5 {
            v.push(i);
        }

        // Valid access should succeed
        for i in 0..5 {
            assert!(v.get(i).is_some());
        }

        // Out of bounds should return None
        assert_eq!(v.get(5), None);
        assert_eq!(v.get(100), None);
    }
}
