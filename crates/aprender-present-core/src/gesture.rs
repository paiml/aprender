#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::many_single_char_names
)]
//! Gesture recognition from touch/pointer events.
//!
//! This module provides gesture recognizers that process raw touch and pointer
//! events and emit high-level gesture events like pinch, rotate, pan, tap, etc.

use crate::event::{Event, GestureState, PointerId, PointerType, TouchId};
use crate::geometry::Point;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for gesture recognition.
#[derive(Debug, Clone)]
pub struct GestureConfig {
    /// Minimum distance to start a pan gesture (in pixels).
    pub pan_threshold: f32,
    /// Maximum time for a tap (in milliseconds).
    pub tap_timeout_ms: u64,
    /// Maximum movement for a tap to still be valid.
    pub tap_slop: f32,
    /// Time required for a long press (in milliseconds).
    pub long_press_ms: u64,
    /// Maximum time between taps for a double tap.
    pub double_tap_ms: u64,
    /// Minimum scale change to start pinch.
    pub pinch_threshold: f32,
    /// Minimum rotation to start rotate gesture (radians).
    pub rotate_threshold: f32,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            pan_threshold: 10.0,
            tap_timeout_ms: 300,
            tap_slop: 10.0,
            long_press_ms: 500,
            double_tap_ms: 300,
            pinch_threshold: 0.05,
            rotate_threshold: 0.05,
        }
    }
}

/// Active touch point being tracked.
#[derive(Debug, Clone)]
pub struct TouchPoint {
    /// Touch ID.
    pub id: TouchId,
    /// Starting position.
    pub start_position: Point,
    /// Current position.
    pub current_position: Point,
    /// Previous position.
    pub previous_position: Point,
    /// When the touch started.
    pub start_time: Instant,
    /// Pressure (0.0-1.0).
    pub pressure: f32,
}

impl TouchPoint {
    /// Create a new touch point.
    pub fn new(id: TouchId, position: Point, pressure: f32) -> Self {
        let now = Instant::now();
        Self {
            id,
            start_position: position,
            current_position: position,
            previous_position: position,
            start_time: now,
            pressure,
        }
    }

    /// Update the touch point position.
    pub fn update(&mut self, position: Point, pressure: f32) {
        self.previous_position = self.current_position;
        self.current_position = position;
        self.pressure = pressure;
    }

    /// Get the total distance moved from start.
    pub fn total_distance(&self) -> f32 {
        self.start_position.distance(&self.current_position)
    }

    /// Get the delta from previous position.
    pub fn delta(&self) -> Point {
        self.current_position - self.previous_position
    }

    /// Get duration since touch started.
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// State of a recognized gesture.
#[derive(Debug, Clone, Default)]
pub enum RecognizedGesture {
    /// No gesture recognized yet.
    #[default]
    None,
    /// Tap gesture (with tap count).
    Tap { position: Point, count: u8 },
    /// Long press gesture.
    LongPress { position: Point },
    /// Pan/drag gesture.
    Pan {
        delta: Point,
        velocity: Point,
        state: GestureState,
    },
    /// Pinch gesture.
    Pinch {
        scale: f32,
        center: Point,
        state: GestureState,
    },
    /// Rotate gesture.
    Rotate {
        angle: f32,
        center: Point,
        state: GestureState,
    },
}

/// Tracks last tap for double-tap detection.
#[derive(Debug, Clone)]
struct LastTap {
    position: Point,
    time: Instant,
    count: u8,
}

/// Multi-touch gesture recognizer.
#[derive(Debug)]
pub struct GestureRecognizer {
    /// Configuration.
    config: GestureConfig,
    /// Active touch points.
    touches: HashMap<TouchId, TouchPoint>,
    /// Currently recognized gesture.
    current_gesture: RecognizedGesture,
    /// Initial distance between two fingers (for pinch).
    initial_pinch_distance: Option<f32>,
    /// Initial angle between two fingers (for rotate).
    initial_rotation_angle: Option<f32>,
    /// Last tap info for double-tap detection.
    last_tap: Option<LastTap>,
    /// Velocity tracking for pan.
    velocity_samples: Vec<(Point, Instant)>,
}

impl GestureRecognizer {
    /// Create a new gesture recognizer with default config.
    pub fn new() -> Self {
        Self::with_config(GestureConfig::default())
    }

    /// Create a new gesture recognizer with custom config.
    pub fn with_config(config: GestureConfig) -> Self {
        Self {
            config,
            touches: HashMap::new(),
            current_gesture: RecognizedGesture::None,
            initial_pinch_distance: None,
            initial_rotation_angle: None,
            last_tap: None,
            velocity_samples: Vec::new(),
        }
    }

    /// Get the current gesture configuration.
    pub fn config(&self) -> &GestureConfig {
        &self.config
    }

    /// Get the number of active touches.
    pub fn touch_count(&self) -> usize {
        self.touches.len()
    }

    /// Get an active touch by ID.
    pub fn touch(&self, id: TouchId) -> Option<&TouchPoint> {
        self.touches.get(&id)
    }

    /// Process an event and return any recognized gesture.
    pub fn process(&mut self, event: &Event) -> Option<Event> {
        match event {
            Event::TouchStart {
                id,
                position,
                pressure,
            } => self.on_touch_start(*id, *position, *pressure),
            Event::TouchMove {
                id,
                position,
                pressure,
            } => self.on_touch_move(*id, *position, *pressure),
            Event::TouchEnd { id, position } => self.on_touch_end(*id, *position),
            Event::TouchCancel { id } => self.on_touch_cancel(*id),
            _ => None,
        }
    }

    fn on_touch_start(&mut self, id: TouchId, position: Point, pressure: f32) -> Option<Event> {
        let touch = TouchPoint::new(id, position, pressure);
        self.touches.insert(id, touch);

        // Reset gesture state when new touch starts
        if self.touches.len() == 2 {
            self.init_two_finger_tracking();
        }

        None
    }

    fn on_touch_move(&mut self, id: TouchId, position: Point, pressure: f32) -> Option<Event> {
        if let Some(touch) = self.touches.get_mut(&id) {
            touch.update(position, pressure);
        } else {
            return None;
        }

        match self.touches.len() {
            1 => self.recognize_single_finger_move(),
            2 => self.recognize_two_finger_move(),
            _ => None,
        }
    }

    fn on_touch_end(&mut self, id: TouchId, _position: Point) -> Option<Event> {
        let touch = self.touches.remove(&id)?;

        // Check for tap if this was the last touch
        if self.touches.is_empty() {
            let duration = touch.duration();
            let distance = touch.total_distance();

            if duration.as_millis() < u128::from(self.config.tap_timeout_ms)
                && distance < self.config.tap_slop
            {
                return self.handle_tap(touch.start_position);
            }

            // End any active gesture
            self.end_active_gesture()
        } else if self.touches.len() == 1 {
            // Went from 2 to 1 finger - end pinch/rotate
            self.end_two_finger_gesture()
        } else {
            None
        }
    }

    fn on_touch_cancel(&mut self, id: TouchId) -> Option<Event> {
        self.touches.remove(&id);

        if self.touches.is_empty() {
            self.end_active_gesture()
        } else {
            None
        }
    }

    fn init_two_finger_tracking(&mut self) {
        let (dist, angle) = if let Some((t1, t2)) = self.get_two_touches() {
            let dist = t1.current_position.distance(&t2.current_position);
            let angle = (t2.current_position.y - t1.current_position.y)
                .atan2(t2.current_position.x - t1.current_position.x);
            (Some(dist), Some(angle))
        } else {
            (None, None)
        };
        self.initial_pinch_distance = dist;
        self.initial_rotation_angle = angle;
    }

    fn recognize_single_finger_move(&mut self) -> Option<Event> {
        let touch = self.touches.values().next()?;
        let distance = touch.total_distance();

        if distance >= self.config.pan_threshold {
            let delta = touch.delta();
            let velocity = self.calculate_velocity();

            let state = match &self.current_gesture {
                RecognizedGesture::Pan { .. } => GestureState::Changed,
                _ => GestureState::Started,
            };

            self.current_gesture = RecognizedGesture::Pan {
                delta,
                velocity,
                state,
            };

            Some(Event::GesturePan {
                delta,
                velocity,
                state,
            })
        } else {
            None
        }
    }

    fn recognize_two_finger_move(&mut self) -> Option<Event> {
        let (t1, t2) = self.get_two_touches()?;

        let current_distance = t1.current_position.distance(&t2.current_position);
        let initial_distance = self.initial_pinch_distance?;
        let scale = current_distance / initial_distance;

        let current_angle = self.angle_between(&t1.current_position, &t2.current_position);
        let initial_angle = self.initial_rotation_angle?;
        let angle_delta = current_angle - initial_angle;

        let center = Point::new(
            (t1.current_position.x + t2.current_position.x) / 2.0,
            (t1.current_position.y + t2.current_position.y) / 2.0,
        );

        // Determine if this is more pinch or rotate
        let scale_change = (scale - 1.0).abs();
        let angle_change = angle_delta.abs();

        if scale_change > self.config.pinch_threshold || angle_change > self.config.rotate_threshold
        {
            // Prefer the larger change
            if scale_change > angle_change {
                let state = match &self.current_gesture {
                    RecognizedGesture::Pinch { .. } => GestureState::Changed,
                    _ => GestureState::Started,
                };

                self.current_gesture = RecognizedGesture::Pinch {
                    scale,
                    center,
                    state,
                };

                Some(Event::GesturePinch {
                    scale,
                    center,
                    state,
                })
            } else {
                let state = match &self.current_gesture {
                    RecognizedGesture::Rotate { .. } => GestureState::Changed,
                    _ => GestureState::Started,
                };

                self.current_gesture = RecognizedGesture::Rotate {
                    angle: angle_delta,
                    center,
                    state,
                };

                Some(Event::GestureRotate {
                    angle: angle_delta,
                    center,
                    state,
                })
            }
        } else {
            None
        }
    }

    fn handle_tap(&mut self, position: Point) -> Option<Event> {
        let now = Instant::now();

        let count = if let Some(last) = &self.last_tap {
            if now.duration_since(last.time).as_millis() < u128::from(self.config.double_tap_ms)
                && position.distance(&last.position) < self.config.tap_slop
            {
                last.count + 1
            } else {
                1
            }
        } else {
            1
        };

        self.last_tap = Some(LastTap {
            position,
            time: now,
            count,
        });

        Some(Event::GestureTap { position, count })
    }

    fn end_active_gesture(&mut self) -> Option<Event> {
        let result = match &self.current_gesture {
            RecognizedGesture::Pan {
                delta, velocity, ..
            } => Some(Event::GesturePan {
                delta: *delta,
                velocity: *velocity,
                state: GestureState::Ended,
            }),
            _ => None,
        };

        self.current_gesture = RecognizedGesture::None;
        self.velocity_samples.clear();
        result
    }

    fn end_two_finger_gesture(&mut self) -> Option<Event> {
        let result = match &self.current_gesture {
            RecognizedGesture::Pinch { scale, center, .. } => Some(Event::GesturePinch {
                scale: *scale,
                center: *center,
                state: GestureState::Ended,
            }),
            RecognizedGesture::Rotate { angle, center, .. } => Some(Event::GestureRotate {
                angle: *angle,
                center: *center,
                state: GestureState::Ended,
            }),
            _ => None,
        };

        self.current_gesture = RecognizedGesture::None;
        self.initial_pinch_distance = None;
        self.initial_rotation_angle = None;
        result
    }

    fn get_two_touches(&self) -> Option<(&TouchPoint, &TouchPoint)> {
        let mut iter = self.touches.values();
        let t1 = iter.next()?;
        let t2 = iter.next()?;
        Some((t1, t2))
    }

    fn angle_between(&self, p1: &Point, p2: &Point) -> f32 {
        (p2.y - p1.y).atan2(p2.x - p1.x)
    }

    fn calculate_velocity(&mut self) -> Point {
        let now = Instant::now();

        // Keep only recent samples (last 100ms)
        self.velocity_samples
            .retain(|(_, time)| now.duration_since(*time).as_millis() < 100);

        if let Some(touch) = self.touches.values().next() {
            self.velocity_samples.push((touch.current_position, now));
        }

        if self.velocity_samples.len() < 2 {
            return Point::ORIGIN;
        }

        let (first_pos, first_time) = self.velocity_samples.first().expect("checked len >= 2");
        let (last_pos, last_time) = self.velocity_samples.last().expect("checked len >= 2");

        let dt = last_time.duration_since(*first_time).as_secs_f32();
        if dt < 0.001 {
            return Point::ORIGIN;
        }

        Point::new(
            (last_pos.x - first_pos.x) / dt,
            (last_pos.y - first_pos.y) / dt,
        )
    }

    /// Check if a long press has occurred.
    /// Call this periodically (e.g., from a timer) to detect long press.
    pub fn check_long_press(&mut self) -> Option<Event> {
        if self.touches.len() != 1 {
            return None;
        }

        let touch = self.touches.values().next()?;

        if touch.duration().as_millis() >= u128::from(self.config.long_press_ms)
            && touch.total_distance() < self.config.tap_slop
            && matches!(self.current_gesture, RecognizedGesture::None)
        {
            self.current_gesture = RecognizedGesture::LongPress {
                position: touch.start_position,
            };

            Some(Event::GestureLongPress {
                position: touch.start_position,
            })
        } else {
            None
        }
    }

    /// Reset the recognizer state.
    pub fn reset(&mut self) {
        self.touches.clear();
        self.current_gesture = RecognizedGesture::None;
        self.initial_pinch_distance = None;
        self.initial_rotation_angle = None;
        self.velocity_samples.clear();
    }
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pointer gesture recognizer that unifies mouse, touch, and pen input.
#[derive(Debug)]
pub struct PointerGestureRecognizer {
    /// Active pointers.
    pointers: HashMap<PointerId, PointerInfo>,
    /// Configuration.
    config: GestureConfig,
    /// Primary pointer ID.
    primary_pointer: Option<PointerId>,
}

/// Information about an active pointer.
#[derive(Debug, Clone)]
pub struct PointerInfo {
    /// Pointer ID.
    pub id: PointerId,
    /// Pointer type.
    pub pointer_type: PointerType,
    /// Starting position.
    pub start_position: Point,
    /// Current position.
    pub current_position: Point,
    /// Start time.
    pub start_time: Instant,
    /// Is primary pointer.
    pub is_primary: bool,
    /// Pressure (0.0-1.0).
    pub pressure: f32,
}

impl PointerGestureRecognizer {
    /// Create a new pointer gesture recognizer.
    pub fn new() -> Self {
        Self::with_config(GestureConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: GestureConfig) -> Self {
        Self {
            pointers: HashMap::new(),
            config,
            primary_pointer: None,
        }
    }

    /// Get the gesture configuration.
    pub fn config(&self) -> &GestureConfig {
        &self.config
    }

    /// Get the number of active pointers.
    pub fn pointer_count(&self) -> usize {
        self.pointers.len()
    }

    /// Get the primary pointer.
    pub fn primary(&self) -> Option<&PointerInfo> {
        self.primary_pointer.and_then(|id| self.pointers.get(&id))
    }

    /// Process a pointer event.
    pub fn process(&mut self, event: &Event) -> Option<Event> {
        match event {
            Event::PointerDown {
                pointer_id,
                pointer_type,
                position,
                pressure,
                is_primary,
                ..
            } => {
                let info = PointerInfo {
                    id: *pointer_id,
                    pointer_type: *pointer_type,
                    start_position: *position,
                    current_position: *position,
                    start_time: Instant::now(),
                    is_primary: *is_primary,
                    pressure: *pressure,
                };

                if *is_primary || self.primary_pointer.is_none() {
                    self.primary_pointer = Some(*pointer_id);
                }

                self.pointers.insert(*pointer_id, info);
                None
            }
            Event::PointerMove {
                pointer_id,
                position,
                pressure,
                ..
            } => {
                if let Some(info) = self.pointers.get_mut(pointer_id) {
                    info.current_position = *position;
                    info.pressure = *pressure;
                }
                None
            }
            Event::PointerUp { pointer_id, .. } | Event::PointerCancel { pointer_id } => {
                self.pointers.remove(pointer_id);
                if self.primary_pointer == Some(*pointer_id) {
                    self.primary_pointer = self.pointers.keys().next().copied();
                }
                None
            }
            _ => None,
        }
    }

    /// Reset the recognizer.
    pub fn reset(&mut self) {
        self.pointers.clear();
        self.primary_pointer = None;
    }
}

impl Default for PointerGestureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GestureConfig tests
    #[test]
    fn test_gesture_config_default() {
        let config = GestureConfig::default();
        assert_eq!(config.pan_threshold, 10.0);
        assert_eq!(config.tap_timeout_ms, 300);
        assert_eq!(config.tap_slop, 10.0);
        assert_eq!(config.long_press_ms, 500);
        assert_eq!(config.double_tap_ms, 300);
        assert!((config.pinch_threshold - 0.05).abs() < 0.001);
        assert!((config.rotate_threshold - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_gesture_config_custom() {
        let config = GestureConfig {
            pan_threshold: 20.0,
            tap_timeout_ms: 200,
            tap_slop: 15.0,
            long_press_ms: 800,
            double_tap_ms: 400,
            pinch_threshold: 0.1,
            rotate_threshold: 0.1,
        };
        assert_eq!(config.pan_threshold, 20.0);
        assert_eq!(config.long_press_ms, 800);
    }

    // TouchPoint tests
    #[test]
    fn test_touch_point_new() {
        let point = TouchPoint::new(TouchId::new(1), Point::new(100.0, 200.0), 0.5);
        assert_eq!(point.id, TouchId(1));
        assert_eq!(point.start_position, Point::new(100.0, 200.0));
        assert_eq!(point.current_position, Point::new(100.0, 200.0));
        assert_eq!(point.pressure, 0.5);
    }

    #[test]
    fn test_touch_point_update() {
        let mut point = TouchPoint::new(TouchId::new(1), Point::new(100.0, 200.0), 0.5);
        point.update(Point::new(150.0, 250.0), 0.8);

        assert_eq!(point.current_position, Point::new(150.0, 250.0));
        assert_eq!(point.previous_position, Point::new(100.0, 200.0));
        assert_eq!(point.pressure, 0.8);
    }

    #[test]
    fn test_touch_point_total_distance() {
        let mut point = TouchPoint::new(TouchId::new(1), Point::new(0.0, 0.0), 0.5);
        point.update(Point::new(3.0, 4.0), 0.5);

        assert!((point.total_distance() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_touch_point_delta() {
        let mut point = TouchPoint::new(TouchId::new(1), Point::new(0.0, 0.0), 0.5);
        point.update(Point::new(10.0, 20.0), 0.5);
        point.update(Point::new(15.0, 25.0), 0.5);

        let delta = point.delta();
        assert_eq!(delta.x, 5.0);
        assert_eq!(delta.y, 5.0);
    }

    // GestureRecognizer tests
    #[test]
    fn test_recognizer_new() {
        let recognizer = GestureRecognizer::new();
        assert_eq!(recognizer.touch_count(), 0);
    }

    #[test]
    fn test_recognizer_with_config() {
        let config = GestureConfig {
            pan_threshold: 20.0,
            ..Default::default()
        };
        let recognizer = GestureRecognizer::with_config(config);
        assert_eq!(recognizer.config().pan_threshold, 20.0);
    }

    #[test]
    fn test_recognizer_touch_start() {
        let mut recognizer = GestureRecognizer::new();

        let event = Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        };

        recognizer.process(&event);
        assert_eq!(recognizer.touch_count(), 1);

        let touch = recognizer.touch(TouchId::new(1)).unwrap();
        assert_eq!(touch.start_position, Point::new(100.0, 200.0));
    }

    #[test]
    fn test_recognizer_touch_move() {
        let mut recognizer = GestureRecognizer::new();

        // Start touch
        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // Move touch
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0),
            pressure: 0.6,
        });

        let touch = recognizer.touch(TouchId::new(1)).unwrap();
        assert_eq!(touch.current_position, Point::new(150.0, 250.0));
    }

    #[test]
    fn test_recognizer_touch_end() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchEnd {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
        });

        assert_eq!(recognizer.touch_count(), 0);
    }

    #[test]
    fn test_recognizer_tap() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // End immediately (tap)
        let result = recognizer.process(&Event::TouchEnd {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
        });

        assert!(matches!(result, Some(Event::GestureTap { count: 1, .. })));
    }

    #[test]
    fn test_recognizer_pan() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // Move beyond pan threshold
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0), // 50px moved
            pressure: 0.5,
        });

        assert!(matches!(
            result,
            Some(Event::GesturePan {
                state: GestureState::Started,
                ..
            })
        ));
    }

    #[test]
    fn test_recognizer_pan_continued() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // First move - starts pan
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
        });

        // Second move - continues pan
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(200.0, 300.0),
            pressure: 0.5,
        });

        assert!(matches!(
            result,
            Some(Event::GesturePan {
                state: GestureState::Changed,
                ..
            })
        ));
    }

    #[test]
    fn test_recognizer_two_touches() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        assert_eq!(recognizer.touch_count(), 2);
    }

    #[test]
    fn test_recognizer_pinch() {
        let mut recognizer = GestureRecognizer::with_config(GestureConfig {
            pinch_threshold: 0.01,
            ..Default::default()
        });

        // Start with two fingers 100px apart
        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        // Move fingers apart to 200px
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(50.0, 200.0),
            pressure: 0.5,
        });

        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(2),
            position: Point::new(250.0, 200.0),
            pressure: 0.5,
        });

        assert!(matches!(result, Some(Event::GesturePinch { .. })));
    }

    #[test]
    fn test_recognizer_reset() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        assert_eq!(recognizer.touch_count(), 1);

        recognizer.reset();
        assert_eq!(recognizer.touch_count(), 0);
    }

    #[test]
    fn test_recognizer_touch_cancel() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchCancel {
            id: TouchId::new(1),
        });

        assert_eq!(recognizer.touch_count(), 0);
    }

    #[test]
    fn test_recognizer_ignores_non_touch_events() {
        let mut recognizer = GestureRecognizer::new();

        let result = recognizer.process(&Event::MouseMove {
            position: Point::new(100.0, 200.0),
        });

        assert!(result.is_none());
        assert_eq!(recognizer.touch_count(), 0);
    }

    // PointerGestureRecognizer tests
    #[test]
    fn test_pointer_recognizer_new() {
        let recognizer = PointerGestureRecognizer::new();
        assert_eq!(recognizer.pointer_count(), 0);
        assert!(recognizer.primary().is_none());
    }

    #[test]
    fn test_pointer_recognizer_pointer_down() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        assert_eq!(recognizer.pointer_count(), 1);

        let primary = recognizer.primary().unwrap();
        assert_eq!(primary.id, PointerId(1));
        assert_eq!(primary.pointer_type, PointerType::Touch);
        assert!(primary.is_primary);
    }

    #[test]
    fn test_pointer_recognizer_pointer_move() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Mouse,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        recognizer.process(&Event::PointerMove {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Mouse,
            position: Point::new(150.0, 250.0),
            pressure: 0.6,
            is_primary: true,
        });

        let primary = recognizer.primary().unwrap();
        assert_eq!(primary.current_position, Point::new(150.0, 250.0));
    }

    #[test]
    fn test_pointer_recognizer_pointer_up() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Pen,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        recognizer.process(&Event::PointerUp {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Pen,
            position: Point::new(100.0, 200.0),
            is_primary: true,
            button: None,
        });

        assert_eq!(recognizer.pointer_count(), 0);
        assert!(recognizer.primary().is_none());
    }

    #[test]
    fn test_pointer_recognizer_multiple_pointers() {
        let mut recognizer = PointerGestureRecognizer::new();

        // First pointer (primary)
        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        // Second pointer (not primary)
        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(2),
            pointer_type: PointerType::Touch,
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
            is_primary: false,
            button: None,
        });

        assert_eq!(recognizer.pointer_count(), 2);

        let primary = recognizer.primary().unwrap();
        assert_eq!(primary.id, PointerId(1));
    }

    #[test]
    fn test_pointer_recognizer_primary_changes_on_remove() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(2),
            pointer_type: PointerType::Touch,
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
            is_primary: false,
            button: None,
        });

        // Remove primary
        recognizer.process(&Event::PointerUp {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            is_primary: true,
            button: None,
        });

        assert_eq!(recognizer.pointer_count(), 1);
        // Primary should now be the remaining pointer
        assert!(recognizer.primary().is_some());
    }

    #[test]
    fn test_pointer_recognizer_reset() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        recognizer.reset();

        assert_eq!(recognizer.pointer_count(), 0);
        assert!(recognizer.primary().is_none());
    }

    #[test]
    fn test_pointer_recognizer_cancel() {
        let mut recognizer = PointerGestureRecognizer::new();

        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: true,
            button: None,
        });

        recognizer.process(&Event::PointerCancel {
            pointer_id: PointerId::new(1),
        });

        assert_eq!(recognizer.pointer_count(), 0);
    }

    // RecognizedGesture tests
    #[test]
    fn test_recognized_gesture_default() {
        let gesture = RecognizedGesture::default();
        assert!(matches!(gesture, RecognizedGesture::None));
    }

    // PointerInfo tests
    #[test]
    fn test_pointer_info_clone() {
        let info = PointerInfo {
            id: PointerId::new(1),
            pointer_type: PointerType::Mouse,
            start_position: Point::new(100.0, 200.0),
            current_position: Point::new(150.0, 250.0),
            start_time: Instant::now(),
            is_primary: true,
            pressure: 0.8,
        };

        let cloned = info.clone();
        assert_eq!(cloned.id, info.id);
        assert_eq!(cloned.pointer_type, info.pointer_type);
        assert_eq!(cloned.pressure, info.pressure);
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_gesture_config_debug() {
        let config = GestureConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("GestureConfig"));
    }

    #[test]
    fn test_gesture_config_clone() {
        let config = GestureConfig {
            pan_threshold: 25.0,
            ..Default::default()
        };
        let cloned = config;
        assert_eq!(cloned.pan_threshold, 25.0);
    }

    #[test]
    fn test_touch_point_debug() {
        let point = TouchPoint::new(TouchId::new(1), Point::new(100.0, 200.0), 0.5);
        let debug = format!("{point:?}");
        assert!(debug.contains("TouchPoint"));
    }

    #[test]
    fn test_touch_point_clone() {
        let point = TouchPoint::new(TouchId::new(1), Point::new(100.0, 200.0), 0.5);
        let cloned = point.clone();
        assert_eq!(cloned.id, point.id);
        assert_eq!(cloned.start_position, point.start_position);
    }

    #[test]
    fn test_touch_point_duration() {
        let point = TouchPoint::new(TouchId::new(1), Point::new(100.0, 200.0), 0.5);
        let duration = point.duration();
        assert!(duration.as_millis() < 100); // Should be very short
    }

    #[test]
    fn test_recognized_gesture_debug() {
        let gesture = RecognizedGesture::Tap {
            position: Point::new(50.0, 50.0),
            count: 2,
        };
        let debug = format!("{gesture:?}");
        assert!(debug.contains("Tap"));
    }

    #[test]
    fn test_recognized_gesture_clone() {
        let gesture = RecognizedGesture::Pan {
            delta: Point::new(10.0, 20.0),
            velocity: Point::new(100.0, 200.0),
            state: GestureState::Started,
        };
        let cloned = gesture;
        assert!(matches!(cloned, RecognizedGesture::Pan { .. }));
    }

    #[test]
    fn test_recognized_gesture_all_variants() {
        let gestures = vec![
            RecognizedGesture::None,
            RecognizedGesture::Tap {
                position: Point::ORIGIN,
                count: 1,
            },
            RecognizedGesture::LongPress {
                position: Point::ORIGIN,
            },
            RecognizedGesture::Pan {
                delta: Point::ORIGIN,
                velocity: Point::ORIGIN,
                state: GestureState::Started,
            },
            RecognizedGesture::Pinch {
                scale: 1.0,
                center: Point::ORIGIN,
                state: GestureState::Changed,
            },
            RecognizedGesture::Rotate {
                angle: 0.5,
                center: Point::ORIGIN,
                state: GestureState::Ended,
            },
        ];

        for gesture in gestures {
            let debug = format!("{gesture:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_gesture_recognizer_default() {
        let recognizer = GestureRecognizer::default();
        assert_eq!(recognizer.touch_count(), 0);
        assert_eq!(recognizer.config().pan_threshold, 10.0);
    }

    #[test]
    fn test_gesture_recognizer_debug() {
        let recognizer = GestureRecognizer::new();
        let debug = format!("{recognizer:?}");
        assert!(debug.contains("GestureRecognizer"));
    }

    #[test]
    fn test_gesture_recognizer_touch_move_unknown_id() {
        let mut recognizer = GestureRecognizer::new();

        // Move without starting touch
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(99),
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
        });

        assert!(result.is_none());
    }

    #[test]
    fn test_gesture_recognizer_touch_end_unknown_id() {
        let mut recognizer = GestureRecognizer::new();

        // End without starting touch
        let result = recognizer.process(&Event::TouchEnd {
            id: TouchId::new(99),
            position: Point::new(100.0, 200.0),
        });

        assert!(result.is_none());
    }

    #[test]
    fn test_gesture_recognizer_pan_end() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // Move to start pan
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
        });

        // End touch
        let result = recognizer.process(&Event::TouchEnd {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0),
        });

        assert!(matches!(
            result,
            Some(Event::GesturePan {
                state: GestureState::Ended,
                ..
            })
        ));
    }

    #[test]
    fn test_gesture_recognizer_rotate() {
        let mut recognizer = GestureRecognizer::with_config(GestureConfig {
            rotate_threshold: 0.01,
            pinch_threshold: 1.0, // High threshold to prefer rotation
            ..Default::default()
        });

        // Start with two fingers
        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        // Rotate (move one finger up, other down, same distance apart)
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(100.0, 150.0),
            pressure: 0.5,
        });

        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(2),
            position: Point::new(200.0, 250.0),
            pressure: 0.5,
        });

        assert!(matches!(result, Some(Event::GestureRotate { .. })));
    }

    #[test]
    fn test_gesture_recognizer_two_finger_end() {
        let mut recognizer = GestureRecognizer::with_config(GestureConfig {
            pinch_threshold: 0.01,
            ..Default::default()
        });

        // Start pinch
        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        // Move to trigger pinch
        recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(50.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchMove {
            id: TouchId::new(2),
            position: Point::new(250.0, 200.0),
            pressure: 0.5,
        });

        // End one finger
        let result = recognizer.process(&Event::TouchEnd {
            id: TouchId::new(1),
            position: Point::new(50.0, 200.0),
        });

        assert!(matches!(
            result,
            Some(Event::GesturePinch {
                state: GestureState::Ended,
                ..
            })
        ));
    }

    #[test]
    fn test_gesture_recognizer_three_touches() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(3),
            position: Point::new(300.0, 200.0),
            pressure: 0.5,
        });

        assert_eq!(recognizer.touch_count(), 3);

        // Move with 3 touches should return None (not handled)
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
        });

        assert!(result.is_none());
    }

    #[test]
    fn test_pointer_recognizer_debug() {
        let recognizer = PointerGestureRecognizer::new();
        let debug = format!("{recognizer:?}");
        assert!(debug.contains("PointerGestureRecognizer"));
    }

    #[test]
    fn test_pointer_recognizer_default() {
        let recognizer = PointerGestureRecognizer::default();
        assert_eq!(recognizer.pointer_count(), 0);
    }

    #[test]
    fn test_pointer_recognizer_with_config() {
        let config = GestureConfig {
            pan_threshold: 30.0,
            ..Default::default()
        };
        let recognizer = PointerGestureRecognizer::with_config(config);
        assert_eq!(recognizer.config().pan_threshold, 30.0);
    }

    #[test]
    fn test_pointer_recognizer_ignores_non_pointer_events() {
        let mut recognizer = PointerGestureRecognizer::new();

        let result = recognizer.process(&Event::MouseMove {
            position: Point::new(100.0, 200.0),
        });

        assert!(result.is_none());
        assert_eq!(recognizer.pointer_count(), 0);
    }

    #[test]
    fn test_pointer_recognizer_move_unknown_pointer() {
        let mut recognizer = PointerGestureRecognizer::new();

        let result = recognizer.process(&Event::PointerMove {
            pointer_id: PointerId::new(99),
            pointer_type: PointerType::Touch,
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
            is_primary: true,
        });

        assert!(result.is_none());
    }

    #[test]
    fn test_pointer_info_debug() {
        let info = PointerInfo {
            id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            start_position: Point::new(100.0, 200.0),
            current_position: Point::new(100.0, 200.0),
            start_time: Instant::now(),
            is_primary: true,
            pressure: 0.5,
        };
        let debug = format!("{info:?}");
        assert!(debug.contains("PointerInfo"));
    }

    #[test]
    fn test_pointer_recognizer_first_non_primary_becomes_primary() {
        let mut recognizer = PointerGestureRecognizer::new();

        // First pointer is not marked as primary, but should become primary
        recognizer.process(&Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
            is_primary: false,
            button: None,
        });

        // Since no primary existed, this should be set as primary
        assert!(recognizer.primary().is_some());
    }

    #[test]
    fn test_gesture_recognizer_below_pan_threshold() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        // Small move below threshold
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(102.0, 202.0), // Only ~2.8px moved
            pressure: 0.5,
        });

        assert!(result.is_none());
    }

    #[test]
    fn test_gesture_recognizer_below_pinch_threshold() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.5,
        });

        recognizer.process(&Event::TouchStart {
            id: TouchId::new(2),
            position: Point::new(200.0, 200.0),
            pressure: 0.5,
        });

        // Tiny movement that doesn't trigger pinch/rotate
        let result = recognizer.process(&Event::TouchMove {
            id: TouchId::new(1),
            position: Point::new(99.0, 200.0),
            pressure: 0.5,
        });

        assert!(result.is_none());
    }
}
