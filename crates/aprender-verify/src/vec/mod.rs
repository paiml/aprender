#![allow(unsafe_code)]
//! # `TruenoVec` - A Growable Array Type with Verified Correctness
//!
//! This module provides `TruenoVec<T>`, a custom growable vector implementation
//! that demonstrates the certeza framework's tiered testing approach:
//!
//! - **Tier 1**: Unit tests for basic functionality (sub-second feedback)
//! - **Tier 2**: Property-based tests verifying algebraic properties (1-5 minutes)
//! - **Tier 3**: Formal verification of critical invariants with Kani (hours)
//!
//! ## Safety Invariants
//!
//! The following invariants are maintained and formally verified:
//!
//! 1. `ptr` is either null or points to valid memory for `capacity` elements
//! 2. `len <= capacity` at all times
//! 3. Elements `[0..len)` are initialized
//! 4. Elements `[len..capacity)` are uninitialized
//!
//! ## Risk Classification
//!
//! **Very High Risk** - Contains `unsafe` code for manual memory management.
//! Full verification framework applied:
//! - Property-based testing
//! - 95%+ coverage requirement
//! - >85% mutation score target
//! - Formal verification of invariants
//!
//! ## Example
//!
//! ```rust
//! use certeza::TruenoVec;
//!
//! let mut vec = TruenoVec::new();
//! vec.push(1);
//! vec.push(2);
//! vec.push(3);
//!
//! assert_eq!(vec.len(), 3);
//! assert_eq!(vec.get(1), Some(&2));
//! assert_eq!(vec.pop(), Some(3));
//! ```

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

/// A growable array type with verified correctness properties.
///
/// This implementation demonstrates asymptotic test effectiveness through:
/// - 95%+ line coverage
/// - >85% mutation score
/// - Comprehensive property-based test suite
/// - Formal verification of critical invariants
///
/// # Type Parameters
///
/// * `T` - The type of elements stored in the vector
///
/// # Safety
///
/// This type uses `unsafe` code for manual memory management. All safety
/// invariants are formally verified using Kani and tested using property-based
/// testing with proptest.
///
/// # Examples
///
/// ```rust
/// use certeza::TruenoVec;
///
/// let mut v = TruenoVec::new();
/// v.push(1);
/// v.push(2);
/// assert_eq!(v.len(), 2);
/// assert_eq!(v.pop(), Some(2));
/// ```
pub struct TruenoVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    _marker: PhantomData<T>,
}

// ============================================================================
// Core Implementation
// ============================================================================

impl<T> TruenoVec<T> {
    /// Creates a new, empty `TruenoVec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Complexity
    ///
    /// - Time: O(1)
    /// - Space: O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec: TruenoVec<i32> = TruenoVec::new();
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self { ptr: NonNull::dangling(), len: 0, capacity: 0, _marker: PhantomData }
    }

    /// Creates a new, empty `TruenoVec<T>` with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the vector will not allocate.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the capacity
    /// - Space: O(n)
    ///
    /// # Panics
    ///
    /// Panics if the capacity exceeds `isize::MAX` bytes or if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec: TruenoVec<i32> = TruenoVec::with_capacity(10);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 10);
    ///
    /// // These pushes won't reallocate
    /// for i in 0..10 {
    ///     vec.push(i);
    /// }
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            Self::new()
        } else {
            let layout = Layout::array::<T>(capacity).expect("capacity overflow");
            // SAFETY: layout has non-zero size
            let ptr = unsafe {
                let ptr = alloc::alloc(layout).cast::<T>();
                NonNull::new(ptr).expect("allocation failed")
            };
            Self { ptr, len: 0, capacity, _marker: PhantomData }
        }
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// assert_eq!(vec.len(), 0);
    /// vec.push(1);
    /// assert_eq!(vec.len(), 1);
    /// ```
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// assert!(vec.is_empty());
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without reallocating.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let vec: TruenoVec<i32> = TruenoVec::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Complexity
    ///
    /// - Amortized O(1) time
    /// - Worst-case O(n) when reallocation is needed
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec.get(0), Some(&1));
    /// assert_eq!(vec.get(1), Some(&2));
    /// ```
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }
        // SAFETY: len < capacity after grow, ptr is valid
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
    }

    /// Removes the last element from the vector and returns it, or `None` if empty.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(vec.pop(), Some(1));
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[allow(clippy::missing_const_for_fn)] // Cannot be const due to ptr::read
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            // SAFETY: len was > 0, so len-1 is a valid initialized index
            unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.len))) }
        }
    }

    /// Returns a reference to an element at the given index, or `None` if out of bounds.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// vec.push(30);
    /// assert_eq!(vec.get(1), Some(&20));
    /// assert_eq!(vec.get(3), None);
    /// ```
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: index < len, so it's a valid initialized element
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to an element at the given index, or `None` if out of bounds.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// if let Some(elem) = vec.get_mut(1) {
    ///     *elem = 25;
    /// }
    /// assert_eq!(vec.get(1), Some(&25));
    /// ```
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: index < len, so it's a valid initialized element
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// This method removes all elements from the vector but does not change
    /// its capacity. The allocated memory is retained for reuse.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let capacity = vec.capacity();
    /// vec.clear();
    ///
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), capacity); // Capacity unchanged
    /// assert!(vec.is_empty());
    /// ```
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
    }

    /// Inserts an element at position `index`, shifting all elements after it
    /// to the right.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the number of elements after the insertion point
    /// - Space: O(1) amortized (may reallocate if capacity is exceeded)
    ///
    /// # Panics
    ///
    /// Panics if `index > len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(3);
    /// vec.insert(1, 2); // Insert 2 at index 1
    ///
    /// assert_eq!(vec.get(0), Some(&1));
    /// assert_eq!(vec.get(1), Some(&2));
    /// assert_eq!(vec.get(2), Some(&3));
    /// ```
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index <= self.len, "insertion index out of bounds");

        if self.len == self.capacity {
            self.grow();
        }

        // SAFETY: len < capacity after grow, index <= len
        unsafe {
            let ptr = self.ptr.as_ptr();
            // Shift elements [index..len) one position to the right
            if index < self.len {
                ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);
            }
            // Write the new value at index
            ptr::write(ptr.add(index), value);
        }

        self.len += 1;
    }

    /// Removes and returns the element at position `index`, shifting all
    /// elements after it to the left.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements after the removal point
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let removed = vec.remove(1);
    /// assert_eq!(removed, 2);
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec.get(0), Some(&1));
    /// assert_eq!(vec.get(1), Some(&3));
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "removal index out of bounds");

        // SAFETY: index < len, so it's a valid initialized element
        unsafe {
            let ptr = self.ptr.as_ptr();
            // Read the value at index
            let value = ptr::read(ptr.add(index));
            // Shift elements [index+1..len) one position to the left
            if index < self.len - 1 {
                ptr::copy(ptr.add(index + 1), ptr.add(index), self.len - index - 1);
            }
            self.len -= 1;
            value
        }
    }

    /// Returns an immutable slice containing all elements of the vector.
    ///
    /// This provides a safe view into the vector's contents without copying.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let slice = vec.as_slice();
    /// assert_eq!(slice, &[1, 2, 3]);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: ptr points to len initialized elements
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice containing all elements of the vector.
    ///
    /// This provides a safe mutable view into the vector's contents without copying.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let slice = vec.as_slice_mut();
    /// slice[1] = 10;
    /// assert_eq!(vec.get(1), Some(&10));
    /// ```
    #[must_use]
    pub const fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: ptr points to len initialized elements
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Doubles the capacity of the vector.
    ///
    /// Uses a growth factor of 2x for exponential growth, ensuring amortized O(1) push.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 {
            1
        } else {
            self.capacity.checked_mul(2).expect("capacity overflow")
        };

        let new_layout = Layout::array::<T>(new_capacity).expect("capacity overflow");

        let new_ptr = if self.capacity == 0 {
            // SAFETY: new_layout has non-zero size
            unsafe {
                let ptr = alloc::alloc(new_layout).cast::<T>();
                NonNull::new(ptr).expect("allocation failed")
            }
        } else {
            let old_layout = Layout::array::<T>(self.capacity).expect("capacity overflow");
            // SAFETY: ptr is valid, old_layout matches the current allocation
            unsafe {
                let ptr =
                    alloc::realloc(self.ptr.as_ptr().cast::<u8>(), old_layout, new_layout.size())
                        .cast::<T>();
                NonNull::new(ptr).expect("allocation failed")
            }
        };

        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl<T> Default for TruenoVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for TruenoVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        while self.pop().is_some() {}

        // Deallocate memory if capacity > 0
        if self.capacity > 0 {
            let layout = Layout::array::<T>(self.capacity).expect("capacity overflow");
            // SAFETY: ptr was allocated with this layout, all elements dropped
            unsafe {
                alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

// Thread safety: TruenoVec<T> is Send if T is Send
unsafe impl<T: Send> Send for TruenoVec<T> {}

// Thread safety: TruenoVec<T> is Sync if T is Sync
unsafe impl<T: Sync> Sync for TruenoVec<T> {}

// ============================================================================
// Conversion Trait Implementations
// ============================================================================

impl<T> From<Vec<T>> for TruenoVec<T> {
    /// Converts a `Vec<T>` into a `TruenoVec<T>`.
    ///
    /// This conversion takes ownership of the `Vec` and efficiently reuses
    /// its underlying allocation by transferring elements one by one.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the number of elements
    /// - Space: O(n) for the new allocation (original Vec allocation is dropped)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let std_vec = vec![1, 2, 3, 4, 5];
    /// let trueno_vec: TruenoVec<i32> = TruenoVec::from(std_vec);
    ///
    /// assert_eq!(trueno_vec.len(), 5);
    /// assert_eq!(trueno_vec.get(0), Some(&1));
    /// assert_eq!(trueno_vec.get(4), Some(&5));
    /// ```
    fn from(vec: Vec<T>) -> Self {
        let mut trueno = Self::with_capacity(vec.len());
        for item in vec {
            trueno.push(item);
        }
        trueno
    }
}

impl<T: Clone> From<&[T]> for TruenoVec<T> {
    /// Converts a slice `&[T]` into a `TruenoVec<T>`.
    ///
    /// This conversion clones all elements from the slice into a new `TruenoVec`.
    /// Requires `T: Clone` since we're creating owned copies.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the number of elements
    /// - Space: O(n)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let array = [1, 2, 3, 4, 5];
    /// let slice: &[i32] = &array;
    /// let trueno_vec: TruenoVec<i32> = TruenoVec::from(slice);
    ///
    /// assert_eq!(trueno_vec.len(), 5);
    /// assert_eq!(trueno_vec.get(2), Some(&3));
    /// ```
    fn from(slice: &[T]) -> Self {
        let mut trueno = Self::with_capacity(slice.len());
        for item in slice {
            trueno.push(item.clone());
        }
        trueno
    }
}

impl<T> FromIterator<T> for TruenoVec<T> {
    /// Creates a `TruenoVec<T>` from an iterator.
    ///
    /// This enables the `.collect()` method on iterators to produce a `TruenoVec`.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the number of elements in the iterator
    /// - Space: O(n)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let trueno_vec: TruenoVec<i32> = (0..10).collect();
    ///
    /// assert_eq!(trueno_vec.len(), 10);
    /// assert_eq!(trueno_vec.get(5), Some(&5));
    /// ```
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let numbers = vec![1, 2, 3, 4, 5];
    /// let doubled: TruenoVec<i32> = numbers.iter().map(|x| x * 2).collect();
    ///
    /// assert_eq!(doubled.len(), 5);
    /// assert_eq!(doubled.get(0), Some(&2));
    /// assert_eq!(doubled.get(4), Some(&10));
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut trueno = Self::with_capacity(lower_bound);

        for item in iter {
            trueno.push(item);
        }

        trueno
    }
}

impl<T> Extend<T> for TruenoVec<T> {
    /// Extends a `TruenoVec<T>` with the contents of an iterator.
    ///
    /// This method appends all elements from the iterator to the end of the vector.
    ///
    /// # Complexity
    ///
    /// - Time: O(n) where n is the number of elements in the iterator
    /// - Space: O(n) amortized
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// vec.extend(vec![3, 4, 5]);
    ///
    /// assert_eq!(vec.len(), 5);
    /// assert_eq!(vec.get(0), Some(&1));
    /// assert_eq!(vec.get(4), Some(&5));
    /// ```
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.extend(0..10);
    ///
    /// assert_eq!(vec.len(), 10);
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        // Reserve space if we know the size hint
        if lower_bound > 0 {
            let needed_capacity = self.len.saturating_add(lower_bound);
            while self.capacity < needed_capacity {
                self.grow();
            }
        }

        for item in iter {
            self.push(item);
        }
    }
}

// ============================================================================
// Iterator Implementations
// ============================================================================

/// Immutable iterator over `TruenoVec<T>`.
///
/// This struct is created by the `iter` method on `TruenoVec`.
pub struct Iter<'a, T> {
    ptr: *const T,
    end: *const T,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: ptr is valid and within bounds
            unsafe {
                let item = &*self.ptr;
                self.ptr = self.ptr.add(1);
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = std::mem::size_of::<T>();
        let len = if size == 0 { 0 } else { (self.end as usize - self.ptr as usize) / size };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: end-1 is valid and within bounds
            unsafe {
                self.end = self.end.sub(1);
                Some(&*self.end)
            }
        }
    }
}

/// Mutable iterator over `TruenoVec<T>`.
///
/// This struct is created by the `iter_mut` method on `TruenoVec`.
pub struct IterMut<'a, T> {
    ptr: *mut T,
    end: *mut T,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: ptr is valid and within bounds
            unsafe {
                let item = &mut *self.ptr;
                self.ptr = self.ptr.add(1);
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = std::mem::size_of::<T>();
        let len = if size == 0 { 0 } else { (self.end as usize - self.ptr as usize) / size };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: end-1 is valid and within bounds
            unsafe {
                self.end = self.end.sub(1);
                Some(&mut *self.end)
            }
        }
    }
}

/// Consuming iterator over `TruenoVec<T>`.
///
/// This struct is created by the `into_iter` method on `TruenoVec`
/// (provided by the `IntoIterator` trait).
pub struct IntoIter<T> {
    vec: TruenoVec<T>,
    current: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.vec.len {
            let index = self.current;
            self.current += 1;
            // SAFETY: index < len, so it's a valid initialized element
            // We take ownership by reading, so the Drop impl won't try to drop it again
            unsafe { Some(ptr::read(self.vec.ptr.as_ptr().add(index))) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len - self.current;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop all remaining unconsumed elements
        while self.current < self.vec.len {
            // SAFETY: current is bounded by vec.len, ptr is valid for vec.len elements
            unsafe {
                ptr::drop_in_place(self.vec.ptr.as_ptr().add(self.current));
            }
            self.current += 1;
        }

        // Now deallocate the memory (all elements have been dropped)
        if self.vec.capacity > 0 {
            let layout = Layout::array::<T>(self.vec.capacity).expect("capacity overflow");
            // SAFETY: ptr was allocated with this layout and capacity > 0 guarantees valid alloc
            unsafe {
                alloc::dealloc(self.vec.ptr.as_ptr().cast::<u8>(), layout);
            }
        }

        // Prevent the vec's Drop from running by setting capacity to 0
        // This is safe because we've already handled deallocation
        self.vec.capacity = 0;
        self.vec.len = 0;
    }
}

impl<T> TruenoVec<T> {
    /// Returns an iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let mut iter = vec.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[must_use]
    pub const fn iter(&self) -> Iter<'_, T> {
        // SAFETY: ptr and ptr+len are both valid
        unsafe {
            Iter {
                ptr: self.ptr.as_ptr(),
                end: self.ptr.as_ptr().add(self.len),
                _marker: PhantomData,
            }
        }
    }

    /// Returns a mutable iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// for elem in vec.iter_mut() {
    ///     *elem *= 2;
    /// }
    ///
    /// assert_eq!(vec.get(0), Some(&2));
    /// assert_eq!(vec.get(1), Some(&4));
    /// assert_eq!(vec.get(2), Some(&6));
    /// ```
    #[must_use]
    pub const fn iter_mut(&mut self) -> IterMut<'_, T> {
        // SAFETY: ptr and ptr+len are both valid
        unsafe {
            IterMut {
                ptr: self.ptr.as_ptr(),
                end: self.ptr.as_ptr().add(self.len),
                _marker: PhantomData,
            }
        }
    }
}

impl<T> IntoIterator for TruenoVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Consumes the vector and returns an iterator over its elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let sum: i32 = vec.into_iter().sum();
    /// assert_eq!(sum, 6);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { vec: self, current: 0 }
    }
}

impl<'a, T> IntoIterator for &'a TruenoVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut TruenoVec<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ============================================================================
// Index Trait Implementations
// ============================================================================

impl<T> std::ops::Index<usize> for TruenoVec<T> {
    type Output = T;

    /// Returns a reference to an element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// vec.push(30);
    ///
    /// assert_eq!(vec[0], 10);
    /// assert_eq!(vec[1], 20);
    /// assert_eq!(vec[2], 30);
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        let len = self.len;
        self.get(index).unwrap_or_else(|| {
            panic!("index out of bounds: the len is {len} but the index is {index}")
        })
    }
}

impl<T> std::ops::IndexMut<usize> for TruenoVec<T> {
    /// Returns a mutable reference to an element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// vec.push(30);
    ///
    /// vec[1] = 25;
    /// assert_eq!(vec[1], 25);
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let len = self.len;
        self.get_mut(index).unwrap_or_else(|| {
            panic!("index out of bounds: the len is {len} but the index is {index}")
        })
    }
}

// ============================================================================
// Clone Trait Implementation
// ============================================================================

impl<T: Clone> Clone for TruenoVec<T> {
    /// Creates a deep copy of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec1 = TruenoVec::new();
    /// vec1.push(1);
    /// vec1.push(2);
    /// vec1.push(3);
    ///
    /// let mut vec2 = vec1.clone();
    ///
    /// // Modify vec2
    /// vec2.push(4);
    ///
    /// // vec1 is unchanged
    /// assert_eq!(vec1.len(), 3);
    /// assert_eq!(vec2.len(), 4);
    /// ```
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len);
        for elem in self {
            new_vec.push(elem.clone());
        }
        new_vec
    }
}

// ============================================================================
// PartialEq and Eq Trait Implementations
// ============================================================================

impl<T: PartialEq> PartialEq for TruenoVec<T> {
    /// Compares two vectors for equality.
    ///
    /// Two vectors are equal if they have the same length and all elements
    /// at corresponding indices are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec1 = TruenoVec::new();
    /// vec1.push(1);
    /// vec1.push(2);
    ///
    /// let mut vec2 = TruenoVec::new();
    /// vec2.push(1);
    /// vec2.push(2);
    ///
    /// assert_eq!(vec1, vec2);
    ///
    /// vec2.push(3);
    /// assert_ne!(vec1, vec2);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        // First check lengths - fast path
        if self.len != other.len {
            return false;
        }

        // Compare elements
        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                return false;
            }
        }

        true
    }
}

impl<T: Eq> Eq for TruenoVec<T> {}

// ============================================================================
// Debug Trait Implementation
// ============================================================================

impl<T: std::fmt::Debug> std::fmt::Debug for TruenoVec<T> {
    /// Formats the vector for debugging output.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// assert_eq!(format!("{:?}", vec), "[1, 2, 3]");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// ============================================================================
// Phase 3.1 Ergonomic Trait Implementations (Deref/AsRef)
// ============================================================================

impl<T> std::ops::Deref for TruenoVec<T> {
    type Target = [T];

    /// Dereferences the `TruenoVec<T>` to a slice `&[T]`.
    ///
    /// This allows `TruenoVec` to be used seamlessly with any function
    /// that accepts a slice, enabling automatic coercion and access to
    /// all slice methods.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// // Deref coercion allows slice methods
    /// assert_eq!(vec.len(), 3);
    /// assert_eq!(vec.first(), Some(&1));
    /// assert_eq!(vec.last(), Some(&3));
    ///
    /// // Works with functions expecting slices
    /// fn sum_slice(s: &[i32]) -> i32 {
    ///     s.iter().sum()
    /// }
    /// assert_eq!(sum_slice(&vec), 6);
    /// ```
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for TruenoVec<T> {
    /// Mutably dereferences the `TruenoVec<T>` to a mutable slice `&mut [T]`.
    ///
    /// This allows mutable access to slice methods through deref coercion.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(3);
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// // DerefMut coercion allows mutable slice methods
    /// vec.sort();
    /// assert_eq!(vec.as_slice(), &[1, 2, 3]);
    ///
    /// vec.reverse();
    /// assert_eq!(vec.as_slice(), &[3, 2, 1]);
    /// ```
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T> AsRef<[T]> for TruenoVec<T> {
    /// Returns a reference to the vector's contents as a slice.
    ///
    /// This enables `TruenoVec` to be used with generic functions that
    /// accept `AsRef<[T]>` bounds, a common pattern in the Rust standard library.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// // Works with generic functions using AsRef<[T]>
    /// fn process<T: AsRef<[i32]>>(data: T) -> i32 {
    ///     data.as_ref().iter().sum()
    /// }
    ///
    /// assert_eq!(process(&vec), 6);
    /// assert_eq!(process(vec![1, 2, 3]), 6);
    /// ```
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for TruenoVec<T> {
    /// Returns a mutable reference to the vector's contents as a slice.
    ///
    /// This enables `TruenoVec` to be used with generic functions that
    /// accept `AsMut<[T]>` bounds, enabling mutable slice operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// // Works with generic functions using AsMut<[T]>
    /// fn double_in_place<T: AsMut<[i32]>>(mut data: T) {
    ///     for x in data.as_mut() {
    ///         *x *= 2;
    ///     }
    /// }
    ///
    /// double_in_place(&mut vec);
    /// assert_eq!(vec.as_slice(), &[2, 4, 6]);
    /// ```
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T> AsRef<Self> for TruenoVec<T> {
    /// Returns a reference to self.
    ///
    /// This implementation allows `TruenoVec` to work with APIs that
    /// require `AsRef<TruenoVec<T>>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let vec = TruenoVec::from(vec![1, 2, 3]);
    /// let vec_ref: &TruenoVec<i32> = vec.as_ref();
    /// assert_eq!(vec_ref.len(), 3);
    /// ```
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T> AsMut<Self> for TruenoVec<T> {
    /// Returns a mutable reference to self.
    ///
    /// This implementation allows `TruenoVec` to work with APIs that
    /// require `AsMut<TruenoVec<T>>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec = TruenoVec::from(vec![1, 2, 3]);
    /// let vec_mut: &mut TruenoVec<i32> = vec.as_mut();
    /// vec_mut.push(4);
    /// assert_eq!(vec_mut.len(), 4);
    /// ```
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

// ============================================================================
// Phase 3.2 Comparison and Hash Trait Implementations
// ============================================================================

impl<T: PartialOrd> PartialOrd for TruenoVec<T> {
    /// Compares two vectors lexicographically.
    ///
    /// Vectors are compared element-by-element from left to right.
    /// The first mismatching element determines the ordering.
    /// If all elements match but one vector is shorter, the shorter vector is less.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    ///
    /// let mut vec1 = TruenoVec::new();
    /// vec1.push(1);
    /// vec1.push(2);
    ///
    /// let mut vec2 = TruenoVec::new();
    /// vec2.push(1);
    /// vec2.push(3);
    ///
    /// assert!(vec1 < vec2);
    /// assert!(vec2 > vec1);
    ///
    /// let mut vec3 = TruenoVec::new();
    /// vec3.push(1);
    /// assert!(vec3 < vec1); // Shorter vector is less
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }

    fn lt(&self, other: &Self) -> bool {
        self.as_slice() < other.as_slice()
    }

    fn le(&self, other: &Self) -> bool {
        self.as_slice() <= other.as_slice()
    }

    fn gt(&self, other: &Self) -> bool {
        self.as_slice() > other.as_slice()
    }

    fn ge(&self, other: &Self) -> bool {
        self.as_slice() >= other.as_slice()
    }
}

impl<T: Ord> Ord for TruenoVec<T> {
    /// Returns the total ordering between two vectors.
    ///
    /// This provides a total ordering for vectors where `T: Ord`,
    /// enabling use in `BTreeMap` and `BTreeSet` as keys, and
    /// allowing sorting of vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    /// use std::cmp::Ordering;
    ///
    /// let mut vec1 = TruenoVec::new();
    /// vec1.push(1);
    /// vec1.push(2);
    ///
    /// let mut vec2 = TruenoVec::new();
    /// vec2.push(1);
    /// vec2.push(3);
    ///
    /// assert_eq!(vec1.cmp(&vec2), Ordering::Less);
    ///
    /// // Can be used for sorting
    /// let mut vecs = vec![vec2.clone(), vec1.clone()];
    /// vecs.sort();
    /// assert_eq!(vecs[0], vec1);
    /// assert_eq!(vecs[1], vec2);
    /// ```
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: std::hash::Hash> std::hash::Hash for TruenoVec<T> {
    /// Hashes the vector by hashing all its elements in order.
    ///
    /// This enables `TruenoVec` to be used as keys in `HashMap` and
    /// values in `HashSet`, provided `T: Hash`.
    ///
    /// # Examples
    ///
    /// ```
    /// use certeza::TruenoVec;
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    ///
    /// let mut key = TruenoVec::new();
    /// key.push(1);
    /// key.push(2);
    /// key.push(3);
    ///
    /// map.insert(key.clone(), "value");
    /// assert_eq!(map.get(&key), Some(&"value"));
    ///
    /// // Different vectors have different hashes
    /// let mut key2 = TruenoVec::new();
    /// key2.push(1);
    /// key2.push(2);
    /// assert_eq!(map.get(&key2), None);
    /// ```
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the length first (same as std::Vec)
        self.len.hash(state);
        // Hash each element
        for elem in self {
            elem.hash(state);
        }
    }
}

// ============================================================================
// Phase 3.3: Display and Borrow Traits (Polish & Advanced Use Cases)
// ============================================================================

/// Display trait for user-friendly formatting
///
/// Formats the vector as `[elem1, elem2, ...]` similar to `std::Vec`.
///
/// # Examples
///
/// ```
/// use certeza::TruenoVec;
///
/// let vec = TruenoVec::from(vec![1, 2, 3]);
/// assert_eq!(format!("{}", vec), "[1, 2, 3]");
/// ```
impl<T: std::fmt::Display> std::fmt::Display for TruenoVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, elem) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{elem}")?;
        }
        write!(f, "]")
    }
}

/// Borrow<[T]> trait for advanced borrowing patterns
///
/// Enables using `TruenoVec` with APIs that accept `Borrow<[T]>` bounds,
/// particularly useful for `HashMap` and `BTreeMap` lookups.
///
/// # Examples
///
/// ```
/// use certeza::TruenoVec;
/// use std::borrow::Borrow;
///
/// let vec = TruenoVec::from(vec![1, 2, 3]);
/// let slice: &[i32] = vec.borrow();
/// assert_eq!(slice, &[1, 2, 3]);
/// ```
impl<T> std::borrow::Borrow<[T]> for TruenoVec<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

/// `BorrowMut<[T]>` trait for advanced mutable borrowing patterns
///
/// Enables using `TruenoVec` with APIs that accept `BorrowMut<[T]>` bounds.
///
/// # Examples
///
/// ```
/// use certeza::TruenoVec;
/// use std::borrow::BorrowMut;
///
/// let mut vec = TruenoVec::from(vec![1, 2, 3]);
/// let slice: &mut [i32] = vec.borrow_mut();
/// slice[0] = 10;
/// assert_eq!(vec.get(0), Some(&10));
/// ```
impl<T> std::borrow::BorrowMut<[T]> for TruenoVec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

// ============================================================================
// Unit Tests (Tier 1: Sub-second feedback)
// ============================================================================

#[cfg(test)]
#[path = "tests.rs"]
mod vec_tests;

#[cfg(test)]
#[path = "property_tests.rs"]
mod vec_property_tests;
