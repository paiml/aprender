//! Event scheduler with deterministic ordering.
//!
//! Implements a priority queue that ensures:
//! - Events are processed in time order
//! - Ties are broken by insertion order (sequence number)
//! - Reproducible across runs

use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::engine::state::SimEvent;
use crate::engine::SimTime;

/// A scheduled event with time and sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// Scheduled time.
    pub time: SimTime,
    /// Sequence number for deterministic tie-breaking.
    pub sequence: u64,
    /// The event to execute.
    pub event: SimEvent,
}

impl ScheduledEvent {
    /// Create a new scheduled event.
    #[must_use]
    pub const fn new(time: SimTime, sequence: u64, event: SimEvent) -> Self {
        Self {
            time,
            sequence,
            event,
        }
    }
}

// Custom ordering for BinaryHeap (min-heap by time, then sequence)
impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.sequence == other.sequence
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First by time, then by sequence
        match self.time.cmp(&other.time) {
            std::cmp::Ordering::Equal => self.sequence.cmp(&other.sequence),
            ord => ord,
        }
    }
}

/// Priority-ordered event queue.
///
/// Ensures deterministic processing order:
/// 1. Events are sorted by time
/// 2. Ties are broken by sequence number (insertion order)
///
/// # Example
///
/// ```rust
/// use simular::engine::scheduler::EventScheduler;
/// use simular::engine::SimTime;
/// use simular::engine::state::{SimEvent, Vec3};
///
/// let mut scheduler = EventScheduler::new();
///
/// scheduler.schedule(
///     SimTime::from_secs(1.0),
///     SimEvent::AddBody {
///         mass: 1.0,
///         position: Vec3::zero(),
///         velocity: Vec3::zero(),
///     },
/// );
/// ```
#[derive(Debug, Default)]
pub struct EventScheduler {
    /// Min-heap ordered by (time, sequence).
    queue: BinaryHeap<Reverse<ScheduledEvent>>,
    /// Monotonic sequence counter for tie-breaking.
    sequence: u64,
}

impl EventScheduler {
    /// Create a new event scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule an event at the given time.
    pub fn schedule(&mut self, time: SimTime, event: SimEvent) {
        let seq = self.sequence;
        self.sequence += 1;

        self.queue
            .push(Reverse(ScheduledEvent::new(time, seq, event)));
    }

    /// Schedule multiple events at the same time.
    ///
    /// Events will be processed in the order they appear in the slice.
    pub fn schedule_all(&mut self, time: SimTime, events: &[SimEvent]) {
        for event in events {
            self.schedule(time, event.clone());
        }
    }

    /// Get the next event (removes from queue).
    #[must_use]
    #[allow(clippy::should_implement_trait)] // Not an Iterator, different semantics
    pub fn next(&mut self) -> Option<ScheduledEvent> {
        self.queue.pop().map(|Reverse(e)| e)
    }

    /// Peek at the next event without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&ScheduledEvent> {
        self.queue.peek().map(|Reverse(e)| e)
    }

    /// Get the next event if its time is before or at the given time.
    #[must_use]
    pub fn next_before(&mut self, time: SimTime) -> Option<ScheduledEvent> {
        if let Some(Reverse(e)) = self.queue.peek() {
            if e.time <= time {
                return self.next();
            }
        }
        None
    }

    /// Get all events up to and including the given time.
    #[must_use]
    pub fn drain_until(&mut self, time: SimTime) -> Vec<ScheduledEvent> {
        let mut events = Vec::new();

        while let Some(event) = self.next_before(time) {
            events.push(event);
        }

        events
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get the number of pending events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Clear all pending events.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Get the time of the next event, if any.
    #[must_use]
    pub fn next_event_time(&self) -> Option<SimTime> {
        self.peek().map(|e| e.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::state::Vec3;

    fn make_add_body_event(mass: f64) -> SimEvent {
        SimEvent::AddBody {
            mass,
            position: Vec3::zero(),
            velocity: Vec3::zero(),
        }
    }

    #[test]
    fn test_scheduler_time_ordering() {
        let mut scheduler = EventScheduler::new();

        // Schedule events out of order
        scheduler.schedule(SimTime::from_secs(3.0), make_add_body_event(3.0));
        scheduler.schedule(SimTime::from_secs(1.0), make_add_body_event(1.0));
        scheduler.schedule(SimTime::from_secs(2.0), make_add_body_event(2.0));

        // Should come out in time order
        let e1 = scheduler.next();
        assert!(e1.is_some());
        assert!(
            (e1.as_ref().map(|e| e.time.as_secs_f64()).unwrap_or(0.0) - 1.0).abs() < f64::EPSILON
        );

        let e2 = scheduler.next();
        assert!(
            (e2.as_ref().map(|e| e.time.as_secs_f64()).unwrap_or(0.0) - 2.0).abs() < f64::EPSILON
        );

        let e3 = scheduler.next();
        assert!(
            (e3.as_ref().map(|e| e.time.as_secs_f64()).unwrap_or(0.0) - 3.0).abs() < f64::EPSILON
        );

        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_scheduler_sequence_ordering() {
        let mut scheduler = EventScheduler::new();

        // Schedule multiple events at same time
        let time = SimTime::from_secs(1.0);
        scheduler.schedule(time, make_add_body_event(1.0));
        scheduler.schedule(time, make_add_body_event(2.0));
        scheduler.schedule(time, make_add_body_event(3.0));

        // Should come out in insertion order (sequence)
        if let Some(e) = scheduler.next() {
            if let SimEvent::AddBody { mass, .. } = e.event {
                assert!((mass - 1.0).abs() < f64::EPSILON);
            }
        }

        if let Some(e) = scheduler.next() {
            if let SimEvent::AddBody { mass, .. } = e.event {
                assert!((mass - 2.0).abs() < f64::EPSILON);
            }
        }

        if let Some(e) = scheduler.next() {
            if let SimEvent::AddBody { mass, .. } = e.event {
                assert!((mass - 3.0).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_scheduler_next_before() {
        let mut scheduler = EventScheduler::new();

        scheduler.schedule(SimTime::from_secs(1.0), make_add_body_event(1.0));
        scheduler.schedule(SimTime::from_secs(2.0), make_add_body_event(2.0));
        scheduler.schedule(SimTime::from_secs(3.0), make_add_body_event(3.0));

        // Get events up to time 1.5
        let e1 = scheduler.next_before(SimTime::from_secs(1.5));
        assert!(e1.is_some());
        assert!((e1.map(|e| e.time.as_secs_f64()).unwrap_or(0.0) - 1.0).abs() < f64::EPSILON);

        // No more events before 1.5
        let e2 = scheduler.next_before(SimTime::from_secs(1.5));
        assert!(e2.is_none());

        // But there are events at 2.0
        let e3 = scheduler.next_before(SimTime::from_secs(2.0));
        assert!(e3.is_some());
    }

    #[test]
    fn test_scheduler_drain_until() {
        let mut scheduler = EventScheduler::new();

        for i in 1..=5 {
            scheduler.schedule(SimTime::from_secs(i as f64), make_add_body_event(i as f64));
        }

        let events = scheduler.drain_until(SimTime::from_secs(3.0));
        assert_eq!(events.len(), 3);
        assert_eq!(scheduler.len(), 2);
    }

    #[test]
    fn test_scheduler_peek() {
        let mut scheduler = EventScheduler::new();

        assert!(scheduler.peek().is_none());

        scheduler.schedule(SimTime::from_secs(1.0), make_add_body_event(1.0));

        // Peek doesn't remove
        assert!(scheduler.peek().is_some());
        assert!(scheduler.peek().is_some());
        assert_eq!(scheduler.len(), 1);

        // Next removes
        let _ = scheduler.next();
        assert!(scheduler.peek().is_none());
    }

    #[test]
    fn test_scheduler_clear() {
        let mut scheduler = EventScheduler::new();

        for i in 1..=10 {
            scheduler.schedule(SimTime::from_secs(i as f64), make_add_body_event(i as f64));
        }

        assert_eq!(scheduler.len(), 10);

        scheduler.clear();

        assert!(scheduler.is_empty());
        assert_eq!(scheduler.len(), 0);
    }

    #[test]
    fn test_scheduler_schedule_all() {
        let mut scheduler = EventScheduler::new();
        let events = vec![
            make_add_body_event(1.0),
            make_add_body_event(2.0),
            make_add_body_event(3.0),
        ];

        scheduler.schedule_all(SimTime::from_secs(1.0), &events);
        assert_eq!(scheduler.len(), 3);

        // All events should be at the same time, but in insertion order
        let mut masses = Vec::new();
        while let Some(e) = scheduler.next() {
            if let SimEvent::AddBody { mass, .. } = e.event {
                masses.push(mass);
            }
        }
        assert_eq!(masses, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scheduler_next_event_time() {
        let mut scheduler = EventScheduler::new();

        assert!(scheduler.next_event_time().is_none());

        scheduler.schedule(SimTime::from_secs(2.5), make_add_body_event(1.0));
        scheduler.schedule(SimTime::from_secs(1.0), make_add_body_event(2.0));

        // Should return earliest event time
        let next_time = scheduler.next_event_time();
        assert!(next_time.is_some());
        assert!((next_time.map_or(0.0, |t| t.as_secs_f64()) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduled_event_new() {
        let event = ScheduledEvent::new(SimTime::from_secs(1.0), 42, make_add_body_event(5.0));

        assert!((event.time.as_secs_f64() - 1.0).abs() < f64::EPSILON);
        assert_eq!(event.sequence, 42);
    }

    #[test]
    fn test_scheduled_event_eq() {
        let e1 = ScheduledEvent::new(SimTime::from_secs(1.0), 1, make_add_body_event(1.0));
        let e2 = ScheduledEvent::new(SimTime::from_secs(1.0), 1, make_add_body_event(2.0));
        let e3 = ScheduledEvent::new(SimTime::from_secs(1.0), 2, make_add_body_event(1.0));
        let e4 = ScheduledEvent::new(SimTime::from_secs(2.0), 1, make_add_body_event(1.0));

        // Same time and sequence = equal (event content ignored)
        assert_eq!(e1, e2);
        // Different sequence = not equal
        assert_ne!(e1, e3);
        // Different time = not equal
        assert_ne!(e1, e4);
    }

    #[test]
    fn test_scheduled_event_ord() {
        let earlier = ScheduledEvent::new(SimTime::from_secs(1.0), 1, make_add_body_event(1.0));
        let later = ScheduledEvent::new(SimTime::from_secs(2.0), 1, make_add_body_event(1.0));
        let same_time_seq1 =
            ScheduledEvent::new(SimTime::from_secs(1.0), 1, make_add_body_event(1.0));
        let same_time_seq2 =
            ScheduledEvent::new(SimTime::from_secs(1.0), 2, make_add_body_event(1.0));

        assert!(earlier < later);
        assert!(same_time_seq1 < same_time_seq2);
    }

    #[test]
    fn test_scheduled_event_partial_ord() {
        let e1 = ScheduledEvent::new(SimTime::from_secs(1.0), 1, make_add_body_event(1.0));
        let e2 = ScheduledEvent::new(SimTime::from_secs(2.0), 1, make_add_body_event(1.0));

        assert!(e1.partial_cmp(&e2).is_some());
        assert!(e1 < e2);
    }

    #[test]
    fn test_scheduled_event_clone() {
        let event = ScheduledEvent::new(SimTime::from_secs(1.0), 5, make_add_body_event(3.0));
        let cloned = event.clone();

        assert_eq!(event.time, cloned.time);
        assert_eq!(event.sequence, cloned.sequence);
    }

    #[test]
    fn test_scheduled_event_debug() {
        let event = ScheduledEvent::new(SimTime::from_secs(1.0), 5, make_add_body_event(3.0));
        let debug = format!("{:?}", event);
        assert!(debug.contains("ScheduledEvent"));
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler: EventScheduler = Default::default();
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.len(), 0);
    }

    #[test]
    fn test_scheduler_debug() {
        let scheduler = EventScheduler::new();
        let debug = format!("{:?}", scheduler);
        assert!(debug.contains("EventScheduler"));
    }

    #[test]
    fn test_scheduler_next_before_empty() {
        let mut scheduler = EventScheduler::new();
        assert!(scheduler.next_before(SimTime::from_secs(1.0)).is_none());
    }

    #[test]
    fn test_scheduler_drain_until_empty() {
        let mut scheduler = EventScheduler::new();
        let events = scheduler.drain_until(SimTime::from_secs(1.0));
        assert!(events.is_empty());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::engine::state::Vec3;
    use proptest::prelude::*;

    fn make_event(mass: f64) -> SimEvent {
        SimEvent::AddBody {
            mass,
            position: Vec3::zero(),
            velocity: Vec3::zero(),
        }
    }

    proptest! {
        /// Falsification: events always come out in time order.
        #[test]
        fn prop_time_ordering(times in prop::collection::vec(0.0f64..1000.0, 1..100)) {
            let mut scheduler = EventScheduler::new();

            for (i, &t) in times.iter().enumerate() {
                scheduler.schedule(SimTime::from_secs(t), make_event(i as f64));
            }

            let mut last_time = 0.0;
            while let Some(event) = scheduler.next() {
                let current_time = event.time.as_secs_f64();
                prop_assert!(current_time >= last_time, "Events not in time order");
                last_time = current_time;
            }
        }

        /// Falsification: drain_until gets correct count.
        #[test]
        fn prop_drain_count(
            times in prop::collection::vec(0.0f64..100.0, 1..50),
            threshold in 0.0f64..100.0,
        ) {
            let mut scheduler = EventScheduler::new();

            for (i, &t) in times.iter().enumerate() {
                scheduler.schedule(SimTime::from_secs(t), make_event(i as f64));
            }

            let expected_count = times.iter().filter(|&&t| t <= threshold).count();
            let events = scheduler.drain_until(SimTime::from_secs(threshold));

            prop_assert_eq!(events.len(), expected_count);
        }
    }
}
