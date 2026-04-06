//! AWP-11: Bounded Message Queue

use std::collections::VecDeque;

/// Bounded message queue with back-pressure.
///
/// # Example
/// ```rust
/// use trueno::brick::BoundedQueue;
///
/// let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);
///
/// assert!(queue.try_push(1).is_ok());
/// assert!(queue.try_push(2).is_ok());
/// assert!(queue.try_push(3).is_ok());
/// assert!(queue.try_push(4).is_err()); // Queue full
///
/// assert_eq!(queue.pop(), Some(1));
/// assert!(queue.try_push(4).is_ok()); // Space available
/// ```
#[derive(Debug)]
pub struct BoundedQueue<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedQueue<T> {
    /// Create a new bounded queue.
    pub fn new(capacity: usize) -> Self {
        Self { items: VecDeque::with_capacity(capacity), capacity }
    }

    /// Try to push an item. Returns error if queue is full.
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        if self.items.len() >= self.capacity {
            Err(item)
        } else {
            self.items.push_back(item);
            Ok(())
        }
    }

    /// Pop an item from the front.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    /// Peek at the front item.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.items.front()
    }

    /// Get the number of items in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Check if the queue is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Get the capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get remaining capacity.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.items.len())
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl<T> Default for BoundedQueue<T> {
    fn default() -> Self {
        Self::new(16)
    }
}
