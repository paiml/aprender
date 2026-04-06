#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Animation system with spring physics, easing, and keyframe support.
//!
//! Provides 60fps-capable animations for smooth UI transitions.

use crate::geometry::Point;
use std::collections::HashMap;

// =============================================================================
// Easing Functions - TESTS FIRST
// =============================================================================

/// Standard easing functions for animations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// Linear interpolation (no easing)
    #[default]
    Linear,
    /// Ease in (slow start)
    EaseIn,
    /// Ease out (slow end)
    EaseOut,
    /// Ease in and out (slow start and end)
    EaseInOut,
    /// Cubic ease in
    CubicIn,
    /// Cubic ease out
    CubicOut,
    /// Cubic ease in and out
    CubicInOut,
    /// Exponential ease in
    ExpoIn,
    /// Exponential ease out
    ExpoOut,
    /// Elastic bounce at end
    ElasticOut,
    /// Bounce at end
    BounceOut,
    /// Back ease out (overshoots then returns)
    BackOut,
}

impl Easing {
    /// Apply easing function to a normalized time value (0.0 to 1.0).
    #[must_use]
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => Self::ease_in_quad(t),
            Self::EaseOut => Self::ease_out_quad(t),
            Self::EaseInOut => Self::ease_in_out_quad(t),
            Self::CubicIn => Self::ease_in_cubic(t),
            Self::CubicOut => Self::ease_out_cubic(t),
            Self::CubicInOut => Self::ease_in_out_cubic(t),
            Self::ExpoIn => Self::ease_in_expo(t),
            Self::ExpoOut => Self::ease_out_expo(t),
            Self::ElasticOut => Self::elastic_out(t),
            Self::BounceOut => Self::bounce_out(t),
            Self::BackOut => Self::back_out(t),
        }
    }

    fn ease_in_quad(t: f64) -> f64 {
        t * t
    }

    fn ease_out_quad(t: f64) -> f64 {
        (1.0 - t).mul_add(-(1.0 - t), 1.0)
    }

    fn ease_in_out_quad(t: f64) -> f64 {
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0f64).mul_add(t, 2.0).powi(2) / 2.0
        }
    }

    fn ease_in_cubic(t: f64) -> f64 {
        t * t * t
    }

    fn ease_out_cubic(t: f64) -> f64 {
        1.0 - (1.0 - t).powi(3)
    }

    fn ease_in_out_cubic(t: f64) -> f64 {
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            1.0 - (-2.0f64).mul_add(t, 2.0).powi(3) / 2.0
        }
    }

    fn ease_in_expo(t: f64) -> f64 {
        if t == 0.0 {
            0.0
        } else {
            10.0f64.mul_add(t, -10.0).exp2()
        }
    }

    fn ease_out_expo(t: f64) -> f64 {
        if (t - 1.0).abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - (-10.0 * t).exp2()
        }
    }

    fn elastic_out(t: f64) -> f64 {
        if t == 0.0 || (t - 1.0).abs() < f64::EPSILON {
            t
        } else {
            let c4 = (2.0 * std::f64::consts::PI) / 3.0;
            (-10.0 * t)
                .exp2()
                .mul_add((t.mul_add(10.0, -0.75) * c4).sin(), 1.0)
        }
    }

    fn bounce_out(t: f64) -> f64 {
        const N1: f64 = 7.5625;
        const D1: f64 = 2.75;

        if t < 1.0 / D1 {
            N1 * t * t
        } else if t < 2.0 / D1 {
            let t = t - 1.5 / D1;
            (N1 * t).mul_add(t, 0.75)
        } else if t < 2.5 / D1 {
            let t = t - 2.25 / D1;
            (N1 * t).mul_add(t, 0.9375)
        } else {
            let t = t - 2.625 / D1;
            (N1 * t).mul_add(t, 0.984375)
        }
    }

    fn back_out(t: f64) -> f64 {
        const C1: f64 = 1.70158;
        const C3: f64 = C1 + 1.0;
        C1.mul_add((t - 1.0).powi(2), C3.mul_add((t - 1.0).powi(3), 1.0))
    }
}

// =============================================================================
// SpringConfig - Spring Physics Parameters
// =============================================================================

/// Spring physics configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringConfig {
    /// Mass of the object (affects inertia)
    pub mass: f64,
    /// Stiffness of the spring (affects speed)
    pub stiffness: f64,
    /// Damping coefficient (affects bounciness)
    pub damping: f64,
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self::GENTLE
    }
}

impl SpringConfig {
    /// Gentle spring (slow, smooth)
    pub const GENTLE: Self = Self {
        mass: 1.0,
        stiffness: 100.0,
        damping: 15.0,
    };

    /// Wobbly spring (bouncy)
    pub const WOBBLY: Self = Self {
        mass: 1.0,
        stiffness: 180.0,
        damping: 12.0,
    };

    /// Stiff spring (fast, snappy)
    pub const STIFF: Self = Self {
        mass: 1.0,
        stiffness: 400.0,
        damping: 30.0,
    };

    /// Molasses spring (very slow)
    pub const MOLASSES: Self = Self {
        mass: 1.0,
        stiffness: 50.0,
        damping: 20.0,
    };

    /// Create custom spring config.
    #[must_use]
    pub const fn custom(mass: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            mass,
            stiffness,
            damping,
        }
    }

    /// Calculate damping ratio.
    #[must_use]
    pub fn damping_ratio(&self) -> f64 {
        self.damping / (2.0 * (self.mass * self.stiffness).sqrt())
    }

    /// Whether spring is underdamped (will oscillate).
    #[must_use]
    pub fn is_underdamped(&self) -> bool {
        self.damping_ratio() < 1.0
    }

    /// Whether spring is critically damped (fastest without oscillation).
    #[must_use]
    pub fn is_critically_damped(&self) -> bool {
        (self.damping_ratio() - 1.0).abs() < 0.01
    }

    /// Whether spring is overdamped (slow, no oscillation).
    #[must_use]
    pub fn is_overdamped(&self) -> bool {
        self.damping_ratio() > 1.0
    }
}

// =============================================================================
// Spring - Animated Spring Value
// =============================================================================

/// A spring-animated value.
#[derive(Debug, Clone)]
pub struct Spring {
    /// Current value
    pub value: f64,
    /// Target value
    pub target: f64,
    /// Current velocity
    pub velocity: f64,
    /// Spring configuration
    pub config: SpringConfig,
    /// Whether animation is complete
    pub at_rest: bool,
    /// Precision threshold for settling
    pub precision: f64,
}

impl Spring {
    /// Create a new spring at an initial value.
    #[must_use]
    pub fn new(initial: f64) -> Self {
        Self {
            value: initial,
            target: initial,
            velocity: 0.0,
            config: SpringConfig::default(),
            at_rest: true,
            precision: 0.001,
        }
    }

    /// Set spring configuration.
    #[must_use]
    pub fn with_config(mut self, config: SpringConfig) -> Self {
        self.config = config;
        self
    }

    /// Set target value.
    pub fn set_target(&mut self, target: f64) {
        if (self.target - target).abs() > f64::EPSILON {
            self.target = target;
            self.at_rest = false;
        }
    }

    /// Update spring physics for a time step (dt in seconds).
    pub fn update(&mut self, dt: f64) {
        if self.at_rest {
            return;
        }

        // Spring force: F = -k * x
        let displacement = self.value - self.target;
        let spring_force = -self.config.stiffness * displacement;

        // Damping force: F = -c * v
        let damping_force = -self.config.damping * self.velocity;

        // Total acceleration: a = F / m
        let acceleration = (spring_force + damping_force) / self.config.mass;

        // Verlet integration
        self.velocity += acceleration * dt;
        self.value += self.velocity * dt;

        // Check if at rest
        if displacement.abs() < self.precision && self.velocity.abs() < self.precision {
            self.value = self.target;
            self.velocity = 0.0;
            self.at_rest = true;
        }
    }

    /// Immediately set value without animation.
    pub fn set_immediate(&mut self, value: f64) {
        self.value = value;
        self.target = value;
        self.velocity = 0.0;
        self.at_rest = true;
    }
}

// =============================================================================
// AnimatedValue - Generic Animated Value
// =============================================================================

/// An animated value with easing or spring physics.
#[derive(Debug, Clone)]
pub enum AnimatedValue {
    /// Easing-based animation
    Eased(EasedValue),
    /// Spring physics animation
    Spring(Spring),
}

impl AnimatedValue {
    /// Get current value.
    #[must_use]
    pub fn value(&self) -> f64 {
        match self {
            Self::Eased(e) => e.value(),
            Self::Spring(s) => s.value,
        }
    }

    /// Whether animation is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        match self {
            Self::Eased(e) => e.is_complete(),
            Self::Spring(s) => s.at_rest,
        }
    }

    /// Update animation for a time step.
    pub fn update(&mut self, dt: f64) {
        match self {
            Self::Eased(e) => e.update(dt),
            Self::Spring(s) => s.update(dt),
        }
    }
}

/// An easing-based animated value.
#[derive(Debug, Clone)]
pub struct EasedValue {
    /// Start value
    pub from: f64,
    /// End value
    pub to: f64,
    /// Total duration in seconds
    pub duration: f64,
    /// Elapsed time
    pub elapsed: f64,
    /// Easing function
    pub easing: Easing,
}

impl EasedValue {
    /// Create new eased animation.
    #[must_use]
    pub fn new(from: f64, to: f64, duration: f64) -> Self {
        Self {
            from,
            to,
            duration,
            elapsed: 0.0,
            easing: Easing::EaseInOut,
        }
    }

    /// Set easing function.
    #[must_use]
    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    /// Get current value.
    #[must_use]
    pub fn value(&self) -> f64 {
        let t = if self.duration > 0.0 {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let eased = self.easing.apply(t);
        (self.to - self.from).mul_add(eased, self.from)
    }

    /// Whether animation is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Update animation.
    pub fn update(&mut self, dt: f64) {
        self.elapsed = (self.elapsed + dt).min(self.duration);
    }

    /// Progress from 0.0 to 1.0.
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.duration > 0.0 {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

// =============================================================================
// Keyframe - Keyframe Animation Support
// =============================================================================

/// A keyframe in an animation.
#[derive(Debug, Clone)]
pub struct Keyframe<T: Clone> {
    /// Time of this keyframe (0.0 to 1.0 normalized)
    pub time: f64,
    /// Value at this keyframe
    pub value: T,
    /// Easing to next keyframe
    pub easing: Easing,
}

impl<T: Clone> Keyframe<T> {
    /// Create new keyframe.
    #[must_use]
    pub fn new(time: f64, value: T) -> Self {
        Self {
            time: time.clamp(0.0, 1.0),
            value,
            easing: Easing::Linear,
        }
    }

    /// Set easing to next keyframe.
    #[must_use]
    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }
}

/// Keyframe animation track.
#[derive(Debug, Clone)]
pub struct KeyframeTrack<T: Clone + Interpolate> {
    /// Keyframes sorted by time
    keyframes: Vec<Keyframe<T>>,
    /// Total duration in seconds
    pub duration: f64,
    /// Current elapsed time
    pub elapsed: f64,
    /// Whether to loop
    pub looping: bool,
}

impl<T: Clone + Interpolate> KeyframeTrack<T> {
    /// Create new keyframe track.
    #[must_use]
    pub fn new(duration: f64) -> Self {
        Self {
            keyframes: Vec::new(),
            duration,
            elapsed: 0.0,
            looping: false,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, keyframe: Keyframe<T>) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .expect("keyframe times must be comparable")
        });
    }

    /// Set looping.
    #[must_use]
    pub fn with_loop(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Get value at current time.
    #[must_use]
    pub fn value(&self) -> Option<T> {
        if self.keyframes.is_empty() {
            return None;
        }

        let t = if self.duration > 0.0 {
            let raw = self.elapsed / self.duration;
            if self.looping {
                raw % 1.0
            } else {
                raw.clamp(0.0, 1.0)
            }
        } else {
            1.0
        };

        // Find surrounding keyframes
        let mut prev_idx = 0;
        let mut next_idx = 0;

        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time <= t {
                prev_idx = i;
            }
            if kf.time >= t {
                next_idx = i;
                break;
            }
            next_idx = i;
        }

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        if prev_idx == next_idx {
            return Some(prev.value.clone());
        }

        // Interpolate between keyframes
        let segment_duration = next.time - prev.time;
        let segment_t = if segment_duration > 0.0 {
            (t - prev.time) / segment_duration
        } else {
            1.0
        };

        let eased_t = prev.easing.apply(segment_t);
        Some(T::interpolate(&prev.value, &next.value, eased_t))
    }

    /// Update animation.
    pub fn update(&mut self, dt: f64) {
        self.elapsed += dt;
        if !self.looping && self.elapsed > self.duration {
            self.elapsed = self.duration;
        }
    }

    /// Whether animation is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.looping && self.elapsed >= self.duration
    }

    /// Reset to start.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }
}

// =============================================================================
// Interpolate Trait
// =============================================================================

/// Trait for types that can be interpolated.
pub trait Interpolate {
    /// Interpolate between two values.
    fn interpolate(from: &Self, to: &Self, t: f64) -> Self;
}

impl Interpolate for f64 {
    fn interpolate(from: &Self, to: &Self, t: f64) -> Self {
        from + (to - from) * t
    }
}

impl Interpolate for f32 {
    fn interpolate(from: &Self, to: &Self, t: f64) -> Self {
        (*to - *from).mul_add(t as Self, *from)
    }
}

impl Interpolate for Point {
    fn interpolate(from: &Self, to: &Self, t: f64) -> Self {
        Self {
            x: f32::interpolate(&from.x, &to.x, t),
            y: f32::interpolate(&from.y, &to.y, t),
        }
    }
}

/// Color for animation (RGBA as f32 0-1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl AnimColor {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

impl Interpolate for AnimColor {
    fn interpolate(from: &Self, to: &Self, t: f64) -> Self {
        let t = t as f32;
        Self {
            r: (to.r - from.r).mul_add(t, from.r),
            g: (to.g - from.g).mul_add(t, from.g),
            b: (to.b - from.b).mul_add(t, from.b),
            a: (to.a - from.a).mul_add(t, from.a),
        }
    }
}

// =============================================================================
// AnimationController - Manages Multiple Animations
// =============================================================================

/// Controller for managing multiple animations.
#[derive(Debug, Default)]
pub struct AnimationController {
    /// Named springs
    springs: HashMap<String, Spring>,
    /// Named eased values
    eased: HashMap<String, EasedValue>,
    /// Active animation count
    active_count: usize,
}

impl AnimationController {
    /// Create new controller.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a spring animation.
    pub fn add_spring(&mut self, name: &str, initial: f64, config: SpringConfig) {
        let spring = Spring::new(initial).with_config(config);
        self.springs.insert(name.to_string(), spring);
    }

    /// Add an eased animation.
    pub fn add_eased(&mut self, name: &str, from: f64, to: f64, duration: f64, easing: Easing) {
        let eased = EasedValue::new(from, to, duration).with_easing(easing);
        self.eased.insert(name.to_string(), eased);
    }

    /// Set spring target.
    pub fn set_target(&mut self, name: &str, target: f64) {
        if let Some(spring) = self.springs.get_mut(name) {
            spring.set_target(target);
        }
    }

    /// Get current value.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        if let Some(spring) = self.springs.get(name) {
            return Some(spring.value);
        }
        if let Some(eased) = self.eased.get(name) {
            return Some(eased.value());
        }
        None
    }

    /// Update all animations.
    pub fn update(&mut self, dt: f64) {
        self.active_count = 0;

        for spring in self.springs.values_mut() {
            spring.update(dt);
            if !spring.at_rest {
                self.active_count += 1;
            }
        }

        for eased in self.eased.values_mut() {
            eased.update(dt);
            if !eased.is_complete() {
                self.active_count += 1;
            }
        }
    }

    /// Whether any animations are active.
    #[must_use]
    pub fn is_animating(&self) -> bool {
        self.active_count > 0
    }

    /// Number of active animations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Remove an animation.
    pub fn remove(&mut self, name: &str) {
        self.springs.remove(name);
        self.eased.remove(name);
    }

    /// Clear all animations.
    pub fn clear(&mut self) {
        self.springs.clear();
        self.eased.clear();
        self.active_count = 0;
    }
}

// =============================================================================
// Tests - TDD Style
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Easing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_easing_linear() {
        assert!((Easing::Linear.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::Linear.apply(0.5) - 0.5).abs() < 0.001);
        assert!((Easing::Linear.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_clamps_input() {
        assert!((Easing::Linear.apply(-0.5) - 0.0).abs() < 0.001);
        assert!((Easing::Linear.apply(1.5) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_ease_in() {
        let val = Easing::EaseIn.apply(0.5);
        assert!(val < 0.5); // Should be below linear at midpoint
    }

    #[test]
    fn test_easing_ease_out() {
        let val = Easing::EaseOut.apply(0.5);
        assert!(val > 0.5); // Should be above linear at midpoint
    }

    #[test]
    fn test_easing_ease_in_out() {
        let val = Easing::EaseInOut.apply(0.5);
        assert!((val - 0.5).abs() < 0.01); // Should be near 0.5 at midpoint
    }

    #[test]
    fn test_easing_cubic() {
        assert!((Easing::CubicIn.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::CubicOut.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_expo() {
        assert!((Easing::ExpoIn.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::ExpoOut.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_elastic() {
        let val = Easing::ElasticOut.apply(1.0);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_bounce() {
        let val = Easing::BounceOut.apply(1.0);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_back() {
        let val = Easing::BackOut.apply(1.0);
        assert!((val - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // SpringConfig tests
    // -------------------------------------------------------------------------

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_spring_config_presets() {
        assert!(SpringConfig::GENTLE.stiffness < SpringConfig::STIFF.stiffness);
        assert!(SpringConfig::WOBBLY.damping < SpringConfig::STIFF.damping);
    }

    #[test]
    fn test_spring_config_damping_ratio() {
        let config = SpringConfig::GENTLE;
        let ratio = config.damping_ratio();
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_spring_config_damping_types() {
        // Underdamped (bouncy)
        let underdamped = SpringConfig::custom(1.0, 100.0, 5.0);
        assert!(underdamped.is_underdamped());

        // Overdamped (slow)
        let overdamped = SpringConfig::custom(1.0, 100.0, 50.0);
        assert!(overdamped.is_overdamped());
    }

    // -------------------------------------------------------------------------
    // Spring tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_spring_new() {
        let spring = Spring::new(10.0);
        assert!((spring.value - 10.0).abs() < 0.001);
        assert!((spring.target - 10.0).abs() < 0.001);
        assert!(spring.at_rest);
    }

    #[test]
    fn test_spring_set_target() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);
        assert!(!spring.at_rest);
        assert!((spring.target - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_spring_update() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);

        // Simulate multiple frames
        for _ in 0..100 {
            spring.update(1.0 / 60.0); // 60fps
        }

        // Should be near target
        assert!((spring.value - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_spring_converges() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);

        // Simulate until at rest
        for _ in 0..1000 {
            if spring.at_rest {
                break;
            }
            spring.update(1.0 / 60.0);
        }

        assert!(spring.at_rest);
        assert!((spring.value - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_spring_set_immediate() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);
        spring.update(1.0 / 60.0);

        spring.set_immediate(50.0);
        assert!(spring.at_rest);
        assert!((spring.value - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_spring_no_update_when_at_rest() {
        let mut spring = Spring::new(100.0);
        let initial_value = spring.value;
        spring.update(1.0 / 60.0);
        assert!((spring.value - initial_value).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // EasedValue tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eased_value_new() {
        let eased = EasedValue::new(0.0, 100.0, 1.0);
        assert!((eased.value() - 0.0).abs() < 0.001);
        assert!(!eased.is_complete());
    }

    #[test]
    fn test_eased_value_update() {
        let mut eased = EasedValue::new(0.0, 100.0, 1.0);
        eased.update(0.5);
        assert!(eased.value() > 0.0);
        assert!(eased.value() < 100.0);
    }

    #[test]
    fn test_eased_value_complete() {
        let mut eased = EasedValue::new(0.0, 100.0, 1.0);
        eased.update(2.0); // Past duration
        assert!(eased.is_complete());
        assert!((eased.value() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_progress() {
        let mut eased = EasedValue::new(0.0, 100.0, 1.0);
        assert!((eased.progress() - 0.0).abs() < 0.001);
        eased.update(0.5);
        assert!((eased.progress() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_with_easing() {
        let eased = EasedValue::new(0.0, 100.0, 1.0).with_easing(Easing::CubicOut);
        assert_eq!(eased.easing, Easing::CubicOut);
    }

    // -------------------------------------------------------------------------
    // AnimatedValue tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_animated_value_eased() {
        let mut anim = AnimatedValue::Eased(EasedValue::new(0.0, 100.0, 1.0));
        assert!((anim.value() - 0.0).abs() < 0.001);
        anim.update(1.0);
        assert!(anim.is_complete());
    }

    #[test]
    fn test_animated_value_spring() {
        let mut anim = AnimatedValue::Spring(Spring::new(0.0));
        if let AnimatedValue::Spring(ref mut s) = anim {
            s.set_target(100.0);
        }
        assert!(!anim.is_complete());
    }

    // -------------------------------------------------------------------------
    // Keyframe tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_keyframe_new() {
        let kf: Keyframe<f64> = Keyframe::new(0.5, 50.0);
        assert!((kf.time - 0.5).abs() < 0.001);
        assert!((kf.value - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_clamps_time() {
        let kf: Keyframe<f64> = Keyframe::new(1.5, 50.0);
        assert!((kf.time - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_new() {
        let track: KeyframeTrack<f64> = KeyframeTrack::new(2.0);
        assert!((track.duration - 2.0).abs() < 0.001);
        assert!(track.value().is_none());
    }

    #[test]
    fn test_keyframe_track_single_keyframe() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 100.0));
        assert!((track.value().unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_interpolation() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));

        track.update(0.5);
        let val = track.value().unwrap();
        assert!(val > 40.0 && val < 60.0); // Should be near 50
    }

    #[test]
    fn test_keyframe_track_looping() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0).with_loop(true);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));

        track.update(1.5);
        assert!(!track.is_complete());
    }

    #[test]
    fn test_keyframe_track_reset() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.update(0.5);
        track.reset();
        assert!((track.elapsed - 0.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Interpolate tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_interpolate_f64() {
        let result = f64::interpolate(&0.0, &100.0, 0.5);
        assert!((result - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_f32() {
        let result = f32::interpolate(&0.0, &100.0, 0.5);
        assert!((result - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_point() {
        let from = Point { x: 0.0, y: 0.0 };
        let to = Point { x: 100.0, y: 100.0 };
        let result = Point::interpolate(&from, &to, 0.5);
        assert!((result.x - 50.0).abs() < 0.001);
        assert!((result.y - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color() {
        let result = AnimColor::interpolate(&AnimColor::BLACK, &AnimColor::WHITE, 0.5);
        assert!((result.r - 0.5).abs() < 0.001);
        assert!((result.g - 0.5).abs() < 0.001);
        assert!((result.b - 0.5).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // AnimationController tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_controller_new() {
        let controller = AnimationController::new();
        assert!(!controller.is_animating());
        assert_eq!(controller.active_count(), 0);
    }

    #[test]
    fn test_controller_add_spring() {
        let mut controller = AnimationController::new();
        controller.add_spring("x", 0.0, SpringConfig::GENTLE);
        assert!((controller.get("x").unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_controller_add_eased() {
        let mut controller = AnimationController::new();
        controller.add_eased("opacity", 0.0, 1.0, 0.3, Easing::EaseOut);
        assert!((controller.get("opacity").unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_controller_set_target() {
        let mut controller = AnimationController::new();
        controller.add_spring("x", 0.0, SpringConfig::STIFF);
        controller.set_target("x", 100.0);
        controller.update(1.0 / 60.0);
        assert!(controller.is_animating());
    }

    #[test]
    fn test_controller_update() {
        let mut controller = AnimationController::new();
        controller.add_eased("fade", 0.0, 1.0, 0.5, Easing::Linear);
        controller.update(0.25);
        let val = controller.get("fade").unwrap();
        assert!(val > 0.4 && val < 0.6);
    }

    #[test]
    fn test_controller_remove() {
        let mut controller = AnimationController::new();
        controller.add_spring("x", 0.0, SpringConfig::GENTLE);
        controller.remove("x");
        assert!(controller.get("x").is_none());
    }

    #[test]
    fn test_controller_clear() {
        let mut controller = AnimationController::new();
        controller.add_spring("x", 0.0, SpringConfig::GENTLE);
        controller.add_spring("y", 0.0, SpringConfig::GENTLE);
        controller.clear();
        assert!(controller.get("x").is_none());
        assert!(controller.get("y").is_none());
    }

    #[test]
    fn test_controller_get_nonexistent() {
        let controller = AnimationController::new();
        assert!(controller.get("nonexistent").is_none());
    }

    #[test]
    fn test_controller_active_count() {
        let mut controller = AnimationController::new();
        controller.add_spring("a", 0.0, SpringConfig::GENTLE);
        controller.add_spring("b", 0.0, SpringConfig::GENTLE);
        controller.set_target("a", 100.0);
        controller.set_target("b", 100.0);
        controller.update(1.0 / 60.0);
        assert_eq!(controller.active_count(), 2);
    }

    // =========================================================================
    // Easing Additional Tests
    // =========================================================================

    #[test]
    fn test_easing_default() {
        assert_eq!(Easing::default(), Easing::Linear);
    }

    #[test]
    fn test_easing_all_variants_at_zero() {
        let easings = [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::CubicIn,
            Easing::CubicOut,
            Easing::CubicInOut,
            Easing::ExpoIn,
            Easing::ExpoOut,
            Easing::ElasticOut,
            Easing::BounceOut,
            Easing::BackOut,
        ];
        for easing in easings {
            let val = easing.apply(0.0);
            assert!(val.abs() < 0.01, "{easing:?} at 0.0 = {val}");
        }
    }

    #[test]
    fn test_easing_all_variants_at_one() {
        let easings = [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::CubicIn,
            Easing::CubicOut,
            Easing::CubicInOut,
            Easing::ExpoIn,
            Easing::ExpoOut,
            Easing::ElasticOut,
            Easing::BounceOut,
            Easing::BackOut,
        ];
        for easing in easings {
            let val = easing.apply(1.0);
            assert!((val - 1.0).abs() < 0.01, "{easing:?} at 1.0 = {val}");
        }
    }

    #[test]
    fn test_easing_cubic_in_out_midpoint() {
        let val = Easing::CubicInOut.apply(0.5);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_easing_expo_in_zero() {
        // ExpoIn has special case for 0
        let val = Easing::ExpoIn.apply(0.0);
        assert!((val - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_expo_out_one() {
        // ExpoOut has special case for 1
        let val = Easing::ExpoOut.apply(1.0);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_elastic_out_zero() {
        let val = Easing::ElasticOut.apply(0.0);
        assert!((val - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_bounce_out_segments() {
        // Test all segments of bounce
        assert!(Easing::BounceOut.apply(0.1) < 0.3);
        assert!(Easing::BounceOut.apply(0.5) > 0.5);
        assert!(Easing::BounceOut.apply(0.8) > 0.9);
        assert!(Easing::BounceOut.apply(0.95) > 0.98);
    }

    #[test]
    fn test_easing_back_out_overshoots() {
        // BackOut should overshoot slightly past 1.0 before settling
        let val_mid = Easing::BackOut.apply(0.5);
        assert!(val_mid > 0.5); // Should be ahead of linear
    }

    #[test]
    fn test_easing_clone() {
        let e = Easing::CubicOut;
        let cloned = e;
        assert_eq!(e, cloned);
    }

    #[test]
    fn test_easing_debug() {
        let e = Easing::ElasticOut;
        let debug = format!("{e:?}");
        assert!(debug.contains("ElasticOut"));
    }

    // =========================================================================
    // SpringConfig Additional Tests
    // =========================================================================

    #[test]
    fn test_spring_config_default() {
        let config = SpringConfig::default();
        assert_eq!(config, SpringConfig::GENTLE);
    }

    #[test]
    fn test_spring_config_custom() {
        let config = SpringConfig::custom(2.0, 200.0, 20.0);
        assert!((config.mass - 2.0).abs() < 0.001);
        assert!((config.stiffness - 200.0).abs() < 0.001);
        assert!((config.damping - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_spring_config_molasses() {
        let config = SpringConfig::MOLASSES;
        assert!(config.stiffness < SpringConfig::GENTLE.stiffness);
    }

    #[test]
    fn test_spring_config_critically_damped() {
        // Critical damping = 2 * sqrt(m * k)
        // For m=1, k=100: critical = 2 * 10 = 20
        let config = SpringConfig::custom(1.0, 100.0, 20.0);
        assert!(config.is_critically_damped());
    }

    #[test]
    fn test_spring_config_all_presets_valid() {
        let presets = [
            SpringConfig::GENTLE,
            SpringConfig::WOBBLY,
            SpringConfig::STIFF,
            SpringConfig::MOLASSES,
        ];
        for config in presets {
            assert!(config.mass > 0.0);
            assert!(config.stiffness > 0.0);
            assert!(config.damping > 0.0);
        }
    }

    #[test]
    fn test_spring_config_clone() {
        let config = SpringConfig::STIFF;
        let cloned = config;
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_spring_config_debug() {
        let config = SpringConfig::WOBBLY;
        let debug = format!("{config:?}");
        assert!(debug.contains("SpringConfig"));
    }

    // =========================================================================
    // Spring Additional Tests
    // =========================================================================

    #[test]
    fn test_spring_with_config() {
        let spring = Spring::new(0.0).with_config(SpringConfig::STIFF);
        assert_eq!(spring.config, SpringConfig::STIFF);
    }

    #[test]
    fn test_spring_set_target_same_value() {
        let mut spring = Spring::new(100.0);
        spring.set_target(100.0); // Same as initial
        assert!(spring.at_rest); // Should remain at rest
    }

    #[test]
    fn test_spring_update_small_dt() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);
        spring.update(0.001); // Very small time step
        assert!(spring.value > 0.0);
    }

    #[test]
    fn test_spring_precision_threshold() {
        let mut spring = Spring::new(0.0);
        spring.precision = 0.1; // Larger precision
        spring.set_target(0.05); // Within precision
        spring.update(0.016);
        // Should settle quickly with larger precision
    }

    #[test]
    fn test_spring_negative_values() {
        let mut spring = Spring::new(0.0);
        spring.set_target(-100.0);
        for _ in 0..200 {
            spring.update(1.0 / 60.0);
        }
        assert!((spring.value - (-100.0)).abs() < 1.0);
    }

    #[test]
    fn test_spring_clone() {
        let spring = Spring::new(50.0);
        let cloned = spring;
        assert!((cloned.value - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_spring_debug() {
        let spring = Spring::new(0.0);
        let debug = format!("{spring:?}");
        assert!(debug.contains("Spring"));
    }

    // =========================================================================
    // EasedValue Additional Tests
    // =========================================================================

    #[test]
    fn test_eased_value_zero_duration() {
        let eased = EasedValue::new(0.0, 100.0, 0.0);
        assert!((eased.value() - 100.0).abs() < 0.001); // Instant
        assert!(eased.is_complete());
    }

    #[test]
    fn test_eased_value_negative_update() {
        let mut eased = EasedValue::new(0.0, 100.0, 1.0);
        eased.update(0.5);
        eased.update(-0.2); // Negative dt (shouldn't happen but handle gracefully)
                            // elapsed should not exceed duration and value should stay in valid range
        assert!(eased.elapsed <= eased.duration);
        // value() clamps progress to [0, 1], so value is always in [from, to] range
        assert!(eased.value() >= 0.0 && eased.value() <= 100.0);
        // elapsed should be 0.3 after -0.2 from 0.5
        assert!((eased.elapsed - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_progress_zero_duration() {
        let eased = EasedValue::new(0.0, 100.0, 0.0);
        assert!((eased.progress() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_linear_interpolation() {
        let mut eased = EasedValue::new(0.0, 100.0, 1.0).with_easing(Easing::Linear);
        eased.update(0.5);
        assert!((eased.value() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_clone() {
        let eased = EasedValue::new(10.0, 90.0, 2.0);
        let cloned = eased;
        assert!((cloned.from - 10.0).abs() < 0.001);
        assert!((cloned.to - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_eased_value_debug() {
        let eased = EasedValue::new(0.0, 100.0, 1.0);
        let debug = format!("{eased:?}");
        assert!(debug.contains("EasedValue"));
    }

    // =========================================================================
    // AnimatedValue Additional Tests
    // =========================================================================

    #[test]
    fn test_animated_value_spring_complete() {
        let mut spring = Spring::new(0.0);
        spring.set_immediate(100.0);
        let anim = AnimatedValue::Spring(spring);
        assert!(anim.is_complete());
    }

    #[test]
    fn test_animated_value_update_eased() {
        let mut anim = AnimatedValue::Eased(EasedValue::new(0.0, 100.0, 1.0));
        anim.update(0.5);
        assert!(anim.value() > 0.0);
        assert!(anim.value() < 100.0);
    }

    #[test]
    fn test_animated_value_update_spring() {
        let mut spring = Spring::new(0.0);
        spring.set_target(100.0);
        let mut anim = AnimatedValue::Spring(spring);
        anim.update(1.0 / 60.0);
        assert!(anim.value() > 0.0);
    }

    // =========================================================================
    // Keyframe Additional Tests
    // =========================================================================

    #[test]
    fn test_keyframe_with_easing() {
        let kf: Keyframe<f64> = Keyframe::new(0.5, 50.0).with_easing(Easing::CubicOut);
        assert_eq!(kf.easing, Easing::CubicOut);
    }

    #[test]
    fn test_keyframe_clamps_negative_time() {
        let kf: Keyframe<f64> = Keyframe::new(-0.5, 50.0);
        assert!((kf.time - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_clone() {
        let kf: Keyframe<f64> = Keyframe::new(0.5, 75.0);
        let cloned = kf;
        assert!((cloned.time - 0.5).abs() < 0.001);
        assert!((cloned.value - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_debug() {
        let kf: Keyframe<f64> = Keyframe::new(0.5, 50.0);
        let debug = format!("{kf:?}");
        assert!(debug.contains("Keyframe"));
    }

    // =========================================================================
    // KeyframeTrack Additional Tests
    // =========================================================================

    #[test]
    fn test_keyframe_track_zero_duration() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(0.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));
        // Zero duration should jump to end
        assert!((track.value().unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_multiple_keyframes() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(0.5, 50.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));

        track.elapsed = 0.25;
        let val = track.value().unwrap();
        assert!(val > 20.0 && val < 30.0); // Between 0 and 50

        track.elapsed = 0.75;
        let val = track.value().unwrap();
        assert!(val > 70.0 && val < 80.0); // Between 50 and 100
    }

    #[test]
    fn test_keyframe_track_keyframe_sorting() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        // Add out of order
        track.add_keyframe(Keyframe::new(1.0, 100.0));
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(0.5, 50.0));

        // Should still work correctly
        track.elapsed = 0.0;
        assert!((track.value().unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_looping_wrap() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0).with_loop(true);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));

        track.update(2.5); // 2.5 seconds on 1 second loop
                           // Should be at 0.5 normalized time
        let val = track.value().unwrap();
        assert!(val > 40.0 && val < 60.0);
    }

    #[test]
    fn test_keyframe_track_non_looping_clamps() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 100.0));

        track.update(5.0); // Way past duration
        assert!((track.elapsed - 1.0).abs() < 0.001); // Clamped to duration
        assert!((track.value().unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_is_complete() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        assert!(!track.is_complete());
        track.update(1.0);
        assert!(track.is_complete());
    }

    #[test]
    fn test_keyframe_track_looping_never_complete() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(1.0).with_loop(true);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.update(10.0);
        assert!(!track.is_complete());
    }

    #[test]
    fn test_keyframe_track_clone() {
        let mut track: KeyframeTrack<f64> = KeyframeTrack::new(2.0);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        let cloned = track.clone();
        assert!((cloned.duration - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_track_debug() {
        let track: KeyframeTrack<f64> = KeyframeTrack::new(1.0);
        let debug = format!("{track:?}");
        assert!(debug.contains("KeyframeTrack"));
    }

    // =========================================================================
    // AnimColor Additional Tests
    // =========================================================================

    #[test]
    fn test_anim_color_new() {
        let color = AnimColor::new(0.5, 0.6, 0.7, 0.8);
        assert!((color.r - 0.5).abs() < 0.001);
        assert!((color.g - 0.6).abs() < 0.001);
        assert!((color.b - 0.7).abs() < 0.001);
        assert!((color.a - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_anim_color_constants() {
        assert!((AnimColor::WHITE.r - 1.0).abs() < 0.001);
        assert!((AnimColor::BLACK.r - 0.0).abs() < 0.001);
        assert!((AnimColor::TRANSPARENT.a - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_anim_color_interpolate_alpha() {
        let from = AnimColor::new(1.0, 1.0, 1.0, 0.0);
        let to = AnimColor::new(1.0, 1.0, 1.0, 1.0);
        let result = AnimColor::interpolate(&from, &to, 0.5);
        assert!((result.a - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_anim_color_clone() {
        let color = AnimColor::new(0.1, 0.2, 0.3, 0.4);
        let cloned = color;
        assert_eq!(color, cloned);
    }

    #[test]
    fn test_anim_color_debug() {
        let color = AnimColor::WHITE;
        let debug = format!("{color:?}");
        assert!(debug.contains("AnimColor"));
    }

    // =========================================================================
    // AnimationController Additional Tests
    // =========================================================================

    #[test]
    fn test_controller_default() {
        let controller = AnimationController::default();
        assert!(!controller.is_animating());
    }

    #[test]
    fn test_controller_set_target_nonexistent() {
        let mut controller = AnimationController::new();
        controller.set_target("nonexistent", 100.0); // Should not panic
    }

    #[test]
    fn test_controller_mixed_animations() {
        let mut controller = AnimationController::new();
        controller.add_spring("spring", 0.0, SpringConfig::STIFF);
        controller.add_eased("eased", 0.0, 100.0, 0.5, Easing::Linear);

        controller.set_target("spring", 100.0);
        controller.update(0.25);

        assert!(controller.is_animating());
        // Both should have values
        assert!(controller.get("spring").is_some());
        assert!(controller.get("eased").is_some());
    }

    #[test]
    fn test_controller_eased_completes() {
        let mut controller = AnimationController::new();
        controller.add_eased("fade", 0.0, 1.0, 0.5, Easing::Linear);
        controller.update(0.5);
        assert!(!controller.is_animating()); // Should be complete
    }

    #[test]
    fn test_controller_debug() {
        let controller = AnimationController::new();
        let debug = format!("{controller:?}");
        assert!(debug.contains("AnimationController"));
    }

    // =========================================================================
    // Interpolate Additional Tests
    // =========================================================================

    #[test]
    fn test_interpolate_f64_boundaries() {
        assert!((f64::interpolate(&0.0, &100.0, 0.0) - 0.0).abs() < 0.001);
        assert!((f64::interpolate(&0.0, &100.0, 1.0) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_f32_negative() {
        let result = f32::interpolate(&-50.0, &50.0, 0.5);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_point_negative() {
        let from = Point {
            x: -100.0,
            y: -100.0,
        };
        let to = Point { x: 100.0, y: 100.0 };
        let result = Point::interpolate(&from, &to, 0.5);
        assert!((result.x - 0.0).abs() < 0.001);
        assert!((result.y - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_boundaries() {
        let result_start = AnimColor::interpolate(&AnimColor::BLACK, &AnimColor::WHITE, 0.0);
        assert!((result_start.r - 0.0).abs() < 0.001);

        let result_end = AnimColor::interpolate(&AnimColor::BLACK, &AnimColor::WHITE, 1.0);
        assert!((result_end.r - 1.0).abs() < 0.001);
    }
}
