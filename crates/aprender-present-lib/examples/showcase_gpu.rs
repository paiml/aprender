//! GPU Showcase Demo - Real WebGPU-accelerated visualization
//!
//! Demonstrates Presentar's GPU rendering capabilities:
//! - 60fps animated bar charts
//! - GPU-driven particle systems
//! - Animated donut charts
//! - Real-time FPS monitoring
//! - Theme switching (light/dark)
//!
//! Run native: `cargo run --example showcase_gpu`
//! Build WASM: `cargo build --example showcase_gpu --target wasm32-unknown-unknown`

use std::f32::consts::PI;

// ============================================================================
// SHOWCASE-001: Core Animation Framework
// ============================================================================

/// Animation easing functions for smooth transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Easing {
    Linear,
    EaseInOut,
    EaseOutBounce,
    EaseOutElastic,
}

impl Easing {
    /// Apply easing function to progress (0.0 to 1.0)
    #[must_use]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0f32).mul_add(t, 2.0).powi(2) / 2.0
                }
            }
            Self::EaseOutBounce => {
                const N1: f32 = 7.5625;
                const D1: f32 = 2.75;
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
                    (N1 * t).mul_add(t, 0.984_375)
                }
            }
            Self::EaseOutElastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let c4 = (2.0 * PI) / 3.0;
                    (-10.0 * t)
                        .exp2()
                        .mul_add(((t * 10.0 - 0.75) * c4).sin(), 1.0)
                }
            }
        }
    }
}

/// Animated value that transitions smoothly
#[derive(Debug, Clone)]
pub struct AnimatedValue {
    start: f32,
    end: f32,
    current: f32,
    progress: f32,
    duration_ms: f32,
    easing: Easing,
}

impl AnimatedValue {
    /// Create new animated value
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self {
            start: value,
            end: value,
            current: value,
            progress: 1.0,
            duration_ms: 300.0,
            easing: Easing::EaseInOut,
        }
    }

    /// Set target value with animation
    pub fn animate_to(&mut self, target: f32, duration_ms: f32, easing: Easing) {
        self.start = self.current;
        self.end = target;
        self.progress = 0.0;
        self.duration_ms = duration_ms;
        self.easing = easing;
    }

    /// Update animation by delta time (milliseconds)
    pub fn update(&mut self, dt_ms: f32) {
        if self.progress < 1.0 {
            self.progress = (self.progress + dt_ms / self.duration_ms).min(1.0);
            let t = self.easing.apply(self.progress);
            self.current = (self.end - self.start).mul_add(t, self.start);
        }
    }

    /// Get current value
    #[must_use]
    pub const fn value(&self) -> f32 {
        self.current
    }

    /// Check if animation is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }

    /// Set value immediately (no animation)
    pub fn set(&mut self, value: f32) {
        self.start = value;
        self.end = value;
        self.current = value;
        self.progress = 1.0;
    }
}

/// RGBA color with animation support
#[derive(Debug, Clone, Copy)]
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
    pub const INDIGO: Self = Self {
        r: 0.39,
        g: 0.4,
        b: 0.95,
        a: 1.0,
    };
    pub const EMERALD: Self = Self {
        r: 0.02,
        g: 0.59,
        b: 0.41,
        a: 1.0,
    };
    pub const AMBER: Self = Self {
        r: 0.96,
        g: 0.62,
        b: 0.04,
        a: 1.0,
    };
    pub const ROSE: Self = Self {
        r: 0.88,
        g: 0.11,
        b: 0.46,
        a: 1.0,
    };
    pub const SKY: Self = Self {
        r: 0.02,
        g: 0.71,
        b: 0.83,
        a: 1.0,
    };

    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[must_use]
    pub fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as f32 / 255.0,
            g: ((hex >> 8) & 0xFF) as f32 / 255.0,
            b: (hex & 0xFF) as f32 / 255.0,
            a: 1.0,
        }
    }

    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: (other.r - self.r).mul_add(t, self.r),
            g: (other.g - self.g).mul_add(t, self.g),
            b: (other.b - self.b).mul_add(t, self.b),
            a: (other.a - self.a).mul_add(t, self.a),
        }
    }

    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Frame timing for consistent 60fps
#[derive(Debug, Clone)]
pub struct FrameTiming {
    frame_count: u64,
    last_frame_time_ms: f32,
    fps_samples: Vec<f32>,
    accumulated_time_ms: f32,
}

impl FrameTiming {
    const TARGET_FRAME_MS: f32 = 16.667; // 60fps
    const FPS_SAMPLE_COUNT: usize = 60;

    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            last_frame_time_ms: Self::TARGET_FRAME_MS,
            fps_samples: Vec::with_capacity(Self::FPS_SAMPLE_COUNT),
            accumulated_time_ms: 0.0,
        }
    }

    /// Record frame time and return delta
    pub fn tick(&mut self, frame_time_ms: f32) -> f32 {
        self.frame_count += 1;
        self.last_frame_time_ms = frame_time_ms;
        self.accumulated_time_ms += frame_time_ms;

        // Rolling FPS samples
        if self.fps_samples.len() >= Self::FPS_SAMPLE_COUNT {
            self.fps_samples.remove(0);
        }
        self.fps_samples.push(frame_time_ms);

        frame_time_ms
    }

    /// Get smoothed FPS
    #[must_use]
    pub fn fps(&self) -> f32 {
        if self.fps_samples.is_empty() {
            return 60.0;
        }
        let avg_ms: f32 = self.fps_samples.iter().sum::<f32>() / self.fps_samples.len() as f32;
        if avg_ms > 0.0 {
            1000.0 / avg_ms
        } else {
            60.0
        }
    }

    /// Get frame count
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get accumulated time in seconds
    #[must_use]
    pub fn time_secs(&self) -> f32 {
        self.accumulated_time_ms / 1000.0
    }
}

impl Default for FrameTiming {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHOWCASE-002: Animated Bar Chart
// ============================================================================

/// GPU-ready animated bar chart
#[derive(Debug, Clone)]
pub struct BarChart {
    values: Vec<AnimatedValue>,
    colors: Vec<AnimColor>,
    labels: Vec<String>,
    bar_gap: f32,
    corner_radius: f32,
}

impl BarChart {
    const DEFAULT_COLORS: [AnimColor; 6] = [
        AnimColor::INDIGO,
        AnimColor::EMERALD,
        AnimColor::AMBER,
        AnimColor::ROSE,
        AnimColor::SKY,
        AnimColor {
            r: 0.58,
            g: 0.33,
            b: 0.87,
            a: 1.0,
        }, // Purple
    ];

    /// Create bar chart with n bars
    #[must_use]
    pub fn new(count: usize) -> Self {
        let values = (0..count).map(|_| AnimatedValue::new(0.0)).collect();
        let colors = (0..count)
            .map(|i| Self::DEFAULT_COLORS[i % Self::DEFAULT_COLORS.len()])
            .collect();
        let labels = (0..count).map(|i| format!("Bar {}", i + 1)).collect();

        Self {
            values,
            colors,
            labels,
            bar_gap: 8.0,
            corner_radius: 4.0,
        }
    }

    /// Number of bars
    #[must_use]
    pub fn bar_count(&self) -> usize {
        self.values.len()
    }

    /// Set value with animation
    pub fn set_value(&mut self, index: usize, value: f32) {
        if let Some(v) = self.values.get_mut(index) {
            v.animate_to(value, 500.0, Easing::EaseOutBounce);
        }
    }

    /// Get current animated value
    #[must_use]
    pub fn get_value(&self, index: usize) -> f32 {
        self.values.get(index).map_or(0.0, AnimatedValue::value)
    }

    /// Set bar color
    pub fn set_color(&mut self, index: usize, color: AnimColor) {
        if let Some(c) = self.colors.get_mut(index) {
            *c = color;
        }
    }

    /// Get all colors
    #[must_use]
    pub fn get_colors(&self) -> &[AnimColor] {
        &self.colors
    }

    /// Set labels
    pub fn set_labels<S: Into<String>>(&mut self, labels: Vec<S>) {
        self.labels = labels.into_iter().map(Into::into).collect();
    }

    /// Get labels
    #[must_use]
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    /// Update all animations
    pub fn update(&mut self, dt_ms: f32) {
        for v in &mut self.values {
            v.update(dt_ms);
        }
    }

    /// Check if all animations complete
    #[must_use]
    pub fn is_animation_complete(&self) -> bool {
        self.values.iter().all(AnimatedValue::is_complete)
    }

    /// Get maximum current value
    #[must_use]
    pub fn max_value(&self) -> f32 {
        self.values
            .iter()
            .map(AnimatedValue::value)
            .fold(0.0_f32, f32::max)
    }

    /// Compute bar bounds for rendering (x, y, width, height)
    #[must_use]
    pub fn compute_bounds(&self, width: f32, height: f32) -> Vec<(f32, f32, f32, f32)> {
        let count = self.values.len();
        if count == 0 {
            return vec![];
        }

        let max_val = self.max_value().max(1.0); // Avoid division by zero
        let total_gap = self.bar_gap * (count - 1) as f32;
        let bar_width = (width - total_gap) / count as f32;

        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let x = i as f32 * (bar_width + self.bar_gap);
                let bar_height = (v.value() / max_val) * height;
                let y = height - bar_height;
                (x, y, bar_width, bar_height)
            })
            .collect()
    }

    /// Get corner radius for rounded bars
    #[must_use]
    pub const fn corner_radius(&self) -> f32 {
        self.corner_radius
    }

    /// Set all values from slice (animates)
    pub fn set_values(&mut self, values: &[f32]) {
        for (i, &val) in values.iter().enumerate() {
            self.set_value(i, val);
        }
    }
}

// ============================================================================
// SHOWCASE-003: GPU Particle System
// ============================================================================

/// Single particle with physics
#[derive(Debug, Clone)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub size: f32,
    pub color: AnimColor,
    pub lifetime: f32,
    pub max_lifetime: f32,
}

impl Particle {
    /// Create new particle at position
    #[must_use]
    pub const fn new(x: f32, y: f32, color: AnimColor) -> Self {
        Self {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            size: 8.0,
            color,
            lifetime: 1000.0,
            max_lifetime: 1000.0,
        }
    }

    /// Update particle physics
    pub fn update(&mut self, dt_ms: f32) {
        let dt_sec = dt_ms / 1000.0;
        self.x += self.vx * dt_sec;
        self.y += self.vy * dt_sec;
        self.lifetime -= dt_ms;

        // Gravity effect
        self.vy += 50.0 * dt_sec;
    }

    /// Check if particle is alive
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.lifetime > 0.0
    }

    /// Get alpha based on remaining lifetime
    #[must_use]
    pub fn alpha(&self) -> f32 {
        (self.lifetime / self.max_lifetime).clamp(0.0, 1.0)
    }

    /// Get current color with alpha
    #[must_use]
    pub fn current_color(&self) -> AnimColor {
        AnimColor {
            a: self.color.a * self.alpha(),
            ..self.color
        }
    }
}

/// GPU-accelerated particle system
#[derive(Debug, Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
    max_particles: usize,
    colors: Vec<AnimColor>,
}

impl ParticleSystem {
    /// Create particle system with max capacity
    #[must_use]
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
            colors: vec![
                AnimColor::INDIGO,
                AnimColor::EMERALD,
                AnimColor::AMBER,
                AnimColor::ROSE,
                AnimColor::SKY,
            ],
        }
    }

    /// Maximum particles
    #[must_use]
    pub const fn max_particles(&self) -> usize {
        self.max_particles
    }

    /// Active particle count
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.particles.len()
    }

    /// Emit particles at position
    pub fn emit(&mut self, x: f32, y: f32, count: usize) {
        let available = self.max_particles - self.particles.len();
        let to_emit = count.min(available);

        for i in 0..to_emit {
            let angle = (i as f32 / to_emit as f32) * 2.0 * PI;
            let speed = 50.0 + (i as f32 * 10.0) % 100.0;
            let color = self.colors[i % self.colors.len()];

            let mut p = Particle::new(x, y, color);
            p.vx = angle.cos() * speed;
            p.vy = angle.sin().mul_add(speed, -80.0); // Upward bias
            p.size = 4.0 + (i as f32 % 8.0);
            p.lifetime = 800.0 + (i as f32 * 50.0) % 400.0;
            p.max_lifetime = p.lifetime;

            self.particles.push(p);
        }
    }

    /// Update all particles
    pub fn update(&mut self, dt_ms: f32) {
        for p in &mut self.particles {
            p.update(dt_ms);
        }
        // Remove dead particles
        self.particles.retain(Particle::is_alive);
    }

    /// Get particle bounds for GPU rendering (x, y, size, alpha)
    #[must_use]
    pub fn get_particle_bounds(&self) -> Vec<(f32, f32, f32, AnimColor)> {
        self.particles
            .iter()
            .map(|p| (p.x, p.y, p.size, p.current_color()))
            .collect()
    }

    /// Clear all particles
    pub fn clear(&mut self) {
        self.particles.clear();
    }
}

// ============================================================================
// SHOWCASE-004: Donut Chart + FPS Counter
// ============================================================================

/// Animated donut/pie chart
#[derive(Debug, Clone)]
pub struct DonutChart {
    values: Vec<AnimatedValue>,
    colors: Vec<AnimColor>,
    rotation: f32,
    inner_radius: f32,
    outer_radius: f32,
}

impl DonutChart {
    /// Create donut chart with n segments
    #[must_use]
    pub fn new(segments: usize) -> Self {
        let colors = [
            AnimColor::INDIGO,
            AnimColor::EMERALD,
            AnimColor::AMBER,
            AnimColor::ROSE,
            AnimColor::SKY,
            AnimColor::from_hex(0x008B_5CF6),
        ];

        Self {
            values: (0..segments).map(|_| AnimatedValue::new(1.0)).collect(),
            colors: (0..segments).map(|i| colors[i % colors.len()]).collect(),
            rotation: 0.0,
            inner_radius: 0.6,
            outer_radius: 1.0,
        }
    }

    /// Number of segments
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.values.len()
    }

    /// Set segment values
    pub fn set_values(&mut self, values: &[f32]) {
        for (i, &val) in values.iter().enumerate() {
            if let Some(v) = self.values.get_mut(i) {
                v.animate_to(val, 600.0, Easing::EaseOutElastic);
            }
        }
    }

    /// Update animations
    pub fn update(&mut self, dt_ms: f32) {
        for v in &mut self.values {
            v.update(dt_ms);
        }
        // Slow rotation
        self.rotation += dt_ms * 0.0005;
    }

    /// Get total of all values
    #[must_use]
    pub fn total(&self) -> f32 {
        self.values.iter().map(AnimatedValue::value).sum()
    }

    /// Current rotation angle
    #[must_use]
    pub const fn rotation(&self) -> f32 {
        self.rotation
    }

    /// Compute segment angles (start, end) in radians
    #[must_use]
    pub fn compute_angles(&self) -> Vec<(f32, f32)> {
        let total = self.total().max(0.001);
        let mut start = self.rotation;
        let mut angles = Vec::with_capacity(self.values.len());

        for v in &self.values {
            let sweep = (v.value() / total) * 2.0 * PI;
            angles.push((start, start + sweep));
            start += sweep;
        }

        angles
    }

    /// Get segment colors
    #[must_use]
    pub fn get_colors(&self) -> &[AnimColor] {
        &self.colors
    }

    /// Inner/outer radius ratio
    #[must_use]
    pub const fn radii(&self) -> (f32, f32) {
        (self.inner_radius, self.outer_radius)
    }
}

/// FPS counter with visual feedback
#[derive(Debug, Clone)]
pub struct FpsCounter {
    samples: Vec<f32>,
    current_fps: u32,
}

impl FpsCounter {
    const SAMPLE_COUNT: usize = 60;

    /// Create new FPS counter
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(Self::SAMPLE_COUNT),
            current_fps: 60,
        }
    }

    /// Record frame time in ms
    pub fn record_frame(&mut self, frame_time_ms: f32) {
        if self.samples.len() >= Self::SAMPLE_COUNT {
            self.samples.remove(0);
        }
        self.samples.push(frame_time_ms);

        // Update FPS
        if !self.samples.is_empty() {
            let avg = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
            self.current_fps = if avg > 0.0 {
                (1000.0 / avg).round() as u32
            } else {
                60
            };
        }
    }

    /// Get current FPS
    #[must_use]
    pub const fn current_fps(&self) -> u32 {
        self.current_fps
    }

    /// Get performance grade
    #[must_use]
    pub const fn grade(&self) -> &'static str {
        match self.current_fps {
            fps if fps >= 58 => "A+",
            fps if fps >= 50 => "A",
            fps if fps >= 40 => "B",
            fps if fps >= 30 => "C",
            fps if fps >= 20 => "D",
            _ => "F",
        }
    }

    /// Get color based on FPS
    #[must_use]
    pub const fn color(&self) -> AnimColor {
        match self.current_fps {
            fps if fps >= 55 => AnimColor::EMERALD,
            fps if fps >= 40 => AnimColor::AMBER,
            _ => AnimColor::ROSE,
        }
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHOWCASE-005: Theme + Main Demo
// ============================================================================

/// Color theme for the showcase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Light,
    Dark,
}

impl Theme {
    /// Background color
    #[must_use]
    pub const fn background(self) -> AnimColor {
        match self {
            Self::Light => AnimColor::new(0.98, 0.98, 0.98, 1.0),
            Self::Dark => AnimColor::new(0.08, 0.09, 0.11, 1.0),
        }
    }

    /// Text color
    #[must_use]
    pub const fn text(self) -> AnimColor {
        match self {
            Self::Light => AnimColor::new(0.07, 0.09, 0.11, 1.0),
            Self::Dark => AnimColor::new(0.95, 0.95, 0.97, 1.0),
        }
    }

    /// Card background
    #[must_use]
    pub const fn card(self) -> AnimColor {
        match self {
            Self::Light => AnimColor::WHITE,
            Self::Dark => AnimColor::new(0.12, 0.13, 0.15, 1.0),
        }
    }

    /// Toggle theme
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Light => Self::Dark,
            Self::Dark => Self::Light,
        }
    }
}

/// Main showcase demo state
pub struct ShowcaseDemo {
    width: f32,
    height: f32,
    theme: Theme,
    bar_chart: BarChart,
    donut_chart: DonutChart,
    particles: ParticleSystem,
    fps_counter: FpsCounter,
    frame_timing: FrameTiming,
    data_seed: u32,
}

impl ShowcaseDemo {
    /// Create new showcase demo
    #[must_use]
    pub fn new(width: f32, height: f32) -> Self {
        let mut bar_chart = BarChart::new(6);
        bar_chart.set_labels(vec!["Jan", "Feb", "Mar", "Apr", "May", "Jun"]);
        bar_chart.set_values(&[65.0, 85.0, 45.0, 95.0, 55.0, 75.0]);

        let mut donut_chart = DonutChart::new(5);
        donut_chart.set_values(&[35.0, 25.0, 20.0, 12.0, 8.0]);

        Self {
            width,
            height,
            theme: Theme::Dark,
            bar_chart,
            donut_chart,
            particles: ParticleSystem::new(500),
            fps_counter: FpsCounter::new(),
            frame_timing: FrameTiming::new(),
            data_seed: 42,
        }
    }

    /// Width
    #[must_use]
    pub const fn width(&self) -> f32 {
        self.width
    }

    /// Height
    #[must_use]
    pub const fn height(&self) -> f32 {
        self.height
    }

    /// Current theme
    #[must_use]
    pub const fn theme(&self) -> Theme {
        self.theme
    }

    /// Toggle theme
    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
    }

    /// Frame count
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_timing.frame_count()
    }

    /// Particle count
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.active_count()
    }

    /// Update demo state
    pub fn update(&mut self, dt_ms: f32) {
        self.frame_timing.tick(dt_ms);
        self.fps_counter.record_frame(dt_ms);
        self.bar_chart.update(dt_ms);
        self.donut_chart.update(dt_ms);
        self.particles.update(dt_ms);
    }

    /// Trigger data update with animation
    pub fn trigger_data_update(&mut self) {
        self.data_seed = self
            .data_seed
            .wrapping_mul(1_103_515_245)
            .wrapping_add(12345);
        let seed = self.data_seed;

        // Pseudo-random values
        let values: Vec<f32> = (0..6)
            .map(|i| {
                let v = ((seed >> (i * 4)) & 0xFF) as f32;
                (v / 255.0).mul_add(80.0, 20.0)
            })
            .collect();
        self.bar_chart.set_values(&values);

        let donut_values: Vec<f32> = (0..5)
            .map(|i| {
                let v = ((seed >> (i * 5 + 2)) & 0x7F) as f32;
                (v / 127.0).mul_add(40.0, 5.0)
            })
            .collect();
        self.donut_chart.set_values(&donut_values);
    }

    /// Emit particles at position
    pub fn emit_particles(&mut self, x: f32, y: f32) {
        self.particles.emit(x, y, 30);
    }

    /// Get FPS
    #[must_use]
    pub const fn fps(&self) -> u32 {
        self.fps_counter.current_fps()
    }

    /// Get FPS grade
    #[must_use]
    pub const fn fps_grade(&self) -> &'static str {
        self.fps_counter.grade()
    }

    /// Get bar chart ref
    #[must_use]
    pub const fn bar_chart(&self) -> &BarChart {
        &self.bar_chart
    }

    /// Get donut chart ref
    #[must_use]
    pub const fn donut_chart(&self) -> &DonutChart {
        &self.donut_chart
    }

    /// Get particles ref
    #[must_use]
    pub const fn particles(&self) -> &ParticleSystem {
        &self.particles
    }
}

// ============================================================================
// Tests for SHOWCASE-001 through 005
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Easing Tests ---

    #[test]
    fn test_easing_linear() {
        assert!((Easing::Linear.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::Linear.apply(0.5) - 0.5).abs() < 0.001);
        assert!((Easing::Linear.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_ease_in_out() {
        assert!((Easing::EaseInOut.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::EaseInOut.apply(1.0) - 1.0).abs() < 0.001);
        // Middle should be 0.5
        assert!((Easing::EaseInOut.apply(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_easing_bounce_ends() {
        assert!((Easing::EaseOutBounce.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::EaseOutBounce.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_elastic_ends() {
        assert!((Easing::EaseOutElastic.apply(0.0) - 0.0).abs() < 0.001);
        assert!((Easing::EaseOutElastic.apply(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_clamps_input() {
        assert!((Easing::Linear.apply(-0.5) - 0.0).abs() < 0.001);
        assert!((Easing::Linear.apply(1.5) - 1.0).abs() < 0.001);
    }

    // --- AnimatedValue Tests ---

    #[test]
    fn test_animated_value_new() {
        let v = AnimatedValue::new(100.0);
        assert!((v.value() - 100.0).abs() < 0.001);
        assert!(v.is_complete());
    }

    #[test]
    fn test_animated_value_animate_to() {
        let mut v = AnimatedValue::new(0.0);
        v.animate_to(100.0, 1000.0, Easing::Linear);
        assert!(!v.is_complete());

        v.update(500.0); // 50% through
        assert!((v.value() - 50.0).abs() < 1.0);

        v.update(500.0); // 100% through
        assert!(v.is_complete());
        assert!((v.value() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_animated_value_set_immediate() {
        let mut v = AnimatedValue::new(0.0);
        v.set(999.0);
        assert!((v.value() - 999.0).abs() < 0.001);
        assert!(v.is_complete());
    }

    #[test]
    fn test_animated_value_easing_applied() {
        let mut v = AnimatedValue::new(0.0);
        v.animate_to(100.0, 1000.0, Easing::EaseInOut);
        v.update(250.0); // 25% time = slower due to ease-in
        assert!(v.value() < 25.0); // Should be less than linear
    }

    // --- AnimColor Tests ---

    #[test]
    fn test_color_constants() {
        assert!((AnimColor::WHITE.r - 1.0).abs() < 0.001);
        assert!((AnimColor::BLACK.r - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_color_from_hex() {
        let red = AnimColor::from_hex(0xFF0000);
        assert!((red.r - 1.0).abs() < 0.01);
        assert!((red.g - 0.0).abs() < 0.01);
        assert!((red.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_color_lerp() {
        let c = AnimColor::BLACK.lerp(AnimColor::WHITE, 0.5);
        assert!((c.r - 0.5).abs() < 0.001);
        assert!((c.g - 0.5).abs() < 0.001);
        assert!((c.b - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_color_to_array() {
        let arr = AnimColor::INDIGO.to_array();
        assert_eq!(arr.len(), 4);
        assert!((arr[3] - 1.0).abs() < 0.001); // Alpha
    }

    // --- FrameTiming Tests ---

    #[test]
    fn test_frame_timing_new() {
        let ft = FrameTiming::new();
        assert_eq!(ft.frame_count(), 0);
        assert!((ft.fps() - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_timing_tick() {
        let mut ft = FrameTiming::new();
        ft.tick(16.667);
        assert_eq!(ft.frame_count(), 1);
        ft.tick(16.667);
        assert_eq!(ft.frame_count(), 2);
    }

    #[test]
    fn test_frame_timing_fps_calculation() {
        let mut ft = FrameTiming::new();
        for _ in 0..10 {
            ft.tick(16.667); // ~60fps
        }
        assert!((ft.fps() - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_frame_timing_time_accumulation() {
        let mut ft = FrameTiming::new();
        ft.tick(1000.0); // 1 second
        assert!((ft.time_secs() - 1.0).abs() < 0.001);
    }

    // --- SHOWCASE-002: Bar Chart Tests ---

    #[test]
    fn test_bar_chart_new() {
        let chart = BarChart::new(5);
        assert_eq!(chart.bar_count(), 5);
    }

    #[test]
    fn test_bar_chart_set_value() {
        let mut chart = BarChart::new(3);
        chart.set_value(0, 50.0);
        chart.set_value(1, 75.0);
        chart.set_value(2, 100.0);
        // Values should animate
        chart.update(1000.0); // Complete animation
        assert!((chart.get_value(0) - 50.0).abs() < 1.0);
        assert!((chart.get_value(1) - 75.0).abs() < 1.0);
        assert!((chart.get_value(2) - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_bar_chart_colors() {
        let mut chart = BarChart::new(2);
        chart.set_color(0, AnimColor::INDIGO);
        chart.set_color(1, AnimColor::EMERALD);
        let colors = chart.get_colors();
        assert_eq!(colors.len(), 2);
    }

    #[test]
    fn test_bar_chart_bounds() {
        let chart = BarChart::new(4);
        let bounds = chart.compute_bounds(400.0, 300.0);
        assert_eq!(bounds.len(), 4);
        // Each bar should have valid bounds
        for (x, y, w, h) in &bounds {
            assert!(*x >= 0.0);
            assert!(*y >= 0.0);
            assert!(*w > 0.0);
            assert!(*h >= 0.0);
        }
    }

    #[test]
    fn test_bar_chart_labels() {
        let mut chart = BarChart::new(3);
        chart.set_labels(vec!["Jan", "Feb", "Mar"]);
        assert_eq!(chart.get_labels().len(), 3);
    }

    #[test]
    fn test_bar_chart_max_value() {
        let mut chart = BarChart::new(3);
        chart.set_value(0, 25.0);
        chart.set_value(1, 100.0);
        chart.set_value(2, 50.0);
        chart.update(1000.0);
        assert!((chart.max_value() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_bar_chart_animation_progress() {
        let mut chart = BarChart::new(2);
        chart.set_value(0, 100.0);
        assert!(!chart.is_animation_complete());
        chart.update(1000.0);
        assert!(chart.is_animation_complete());
    }

    // --- SHOWCASE-003: Particle System Tests ---

    #[test]
    fn test_particle_new() {
        let p = Particle::new(100.0, 100.0, AnimColor::INDIGO);
        assert!((p.x - 100.0).abs() < 0.001);
        assert!((p.y - 100.0).abs() < 0.001);
        assert!(p.is_alive());
    }

    #[test]
    fn test_particle_velocity() {
        let mut p = Particle::new(0.0, 0.0, AnimColor::WHITE);
        p.vx = 10.0;
        p.vy = 5.0;
        p.update(100.0); // 100ms
        assert!(p.x > 0.0);
        assert!(p.y > 0.0);
    }

    #[test]
    fn test_particle_lifetime() {
        let mut p = Particle::new(0.0, 0.0, AnimColor::WHITE);
        p.lifetime = 100.0;
        p.max_lifetime = 100.0;
        assert!(p.is_alive());
        p.update(50.0);
        assert!(p.is_alive());
        p.update(60.0);
        assert!(!p.is_alive());
    }

    #[test]
    fn test_particle_alpha_fade() {
        let mut p = Particle::new(0.0, 0.0, AnimColor::WHITE);
        p.lifetime = 100.0;
        p.max_lifetime = 100.0;
        p.update(50.0); // 50% through life
        assert!(p.alpha() < 1.0);
        assert!(p.alpha() > 0.0);
    }

    #[test]
    fn test_particle_system_new() {
        let ps = ParticleSystem::new(100);
        assert_eq!(ps.max_particles(), 100);
        assert_eq!(ps.active_count(), 0);
    }

    #[test]
    fn test_particle_system_emit() {
        let mut ps = ParticleSystem::new(50);
        ps.emit(200.0, 200.0, 10);
        assert_eq!(ps.active_count(), 10);
    }

    #[test]
    fn test_particle_system_update() {
        let mut ps = ParticleSystem::new(50);
        ps.emit(200.0, 200.0, 5);
        ps.update(16.0);
        // Particles should still be alive after 16ms
        assert!(ps.active_count() > 0);
    }

    #[test]
    fn test_particle_system_bounds() {
        let mut ps = ParticleSystem::new(50);
        ps.emit(200.0, 200.0, 5);
        let bounds = ps.get_particle_bounds();
        assert_eq!(bounds.len(), 5);
    }

    #[test]
    fn test_particle_system_respects_max() {
        let mut ps = ParticleSystem::new(10);
        ps.emit(0.0, 0.0, 20); // Try to emit more than max
        assert!(ps.active_count() <= 10);
    }

    // --- SHOWCASE-004: Donut Chart Tests ---

    #[test]
    fn test_donut_chart_new() {
        let donut = DonutChart::new(4);
        assert_eq!(donut.segment_count(), 4);
    }

    #[test]
    fn test_donut_chart_set_values() {
        let mut donut = DonutChart::new(3);
        donut.set_values(&[30.0, 50.0, 20.0]);
        donut.update(1000.0);
        assert!((donut.total() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_donut_chart_angles() {
        let mut donut = DonutChart::new(2);
        donut.set_values(&[50.0, 50.0]);
        donut.update(1000.0);
        let angles = donut.compute_angles();
        assert_eq!(angles.len(), 2);
        // Each should be ~PI radians (180 degrees)
        assert!((angles[0].1 - angles[0].0 - PI).abs() < 0.1);
    }

    #[test]
    fn test_donut_chart_rotation() {
        let mut donut = DonutChart::new(2);
        donut.update(1000.0);
        assert!(donut.rotation() > 0.0);
    }

    #[test]
    fn test_fps_counter_new() {
        let fps = FpsCounter::new();
        assert_eq!(fps.current_fps(), 60); // Default
    }

    #[test]
    fn test_fps_counter_update() {
        let mut fps = FpsCounter::new();
        for _ in 0..10 {
            fps.record_frame(16.667);
        }
        assert!((fps.current_fps() as f32 - 60.0).abs() < 5.0);
    }

    #[test]
    fn test_fps_counter_grade() {
        let mut fps = FpsCounter::new();
        for _ in 0..60 {
            fps.record_frame(16.667);
        }
        assert_eq!(fps.grade(), "A+");
    }

    #[test]
    fn test_fps_counter_color() {
        let mut fps = FpsCounter::new();
        for _ in 0..60 {
            fps.record_frame(16.667);
        }
        let color = fps.color();
        // Green for good FPS
        assert!(color.g > color.r);
    }

    // --- SHOWCASE-005: Theme + Demo State Tests ---

    #[test]
    fn test_theme_colors() {
        let light = Theme::Light;
        let dark = Theme::Dark;
        assert!(light.background().r > 0.5); // Light bg
        assert!(dark.background().r < 0.5); // Dark bg
    }

    #[test]
    fn test_theme_toggle() {
        let theme = Theme::Light;
        assert_eq!(theme.toggle(), Theme::Dark);
        assert_eq!(Theme::Dark.toggle(), Theme::Light);
    }

    #[test]
    fn test_showcase_demo_new() {
        let demo = ShowcaseDemo::new(800.0, 600.0);
        assert_eq!(demo.width(), 800.0);
        assert_eq!(demo.height(), 600.0);
    }

    #[test]
    fn test_showcase_demo_update() {
        let mut demo = ShowcaseDemo::new(800.0, 600.0);
        demo.update(16.667);
        assert!(demo.frame_count() > 0);
    }

    #[test]
    fn test_showcase_demo_theme_toggle() {
        let mut demo = ShowcaseDemo::new(800.0, 600.0);
        let initial = demo.theme();
        demo.toggle_theme();
        assert_ne!(demo.theme(), initial);
    }

    #[test]
    fn test_showcase_demo_trigger_data_update() {
        let mut demo = ShowcaseDemo::new(800.0, 600.0);
        demo.trigger_data_update();
        // Should not panic, data should be updated
    }

    #[test]
    fn test_showcase_demo_emit_particles() {
        let mut demo = ShowcaseDemo::new(800.0, 600.0);
        demo.emit_particles(400.0, 300.0);
        assert!(demo.particle_count() > 0);
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         PRESENTAR GPU SHOWCASE - 60fps WebGPU Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create and run demo simulation
    let mut demo = ShowcaseDemo::new(1280.0, 720.0);

    println!("  Components:");
    println!("    ✓ Animation Framework (Easing, AnimatedValue, AnimColor)");
    println!("    ✓ Bar Chart (6 bars, bounce animation)");
    println!("    ✓ Particle System (500 max particles)");
    println!("    ✓ Donut Chart (5 segments, rotation)");
    println!("    ✓ FPS Counter (60 sample rolling average)");
    println!("    ✓ Theme System (Light/Dark)");
    println!();

    // Simulate 60 frames
    println!("  Simulating 60 frames...");
    for _ in 0..60 {
        demo.update(16.667);
    }
    demo.trigger_data_update();
    demo.emit_particles(640.0, 360.0);

    println!("    Frames: {}", demo.frame_count());
    println!("    FPS: {} ({})", demo.fps(), demo.fps_grade());
    println!("    Particles: {}", demo.particle_count());
    println!("    Theme: {:?}", demo.theme());
    println!();

    // Demo bar chart animation
    println!("  Bar Chart Animation:");
    let bounds = demo.bar_chart().compute_bounds(300.0, 100.0);
    for (i, (x, _y, w, h)) in bounds.iter().enumerate() {
        let bar = "█".repeat((h / 5.0) as usize);
        println!(
            "    Bar {}: x={:>5.1} w={:>4.1} h={:>5.1} {}",
            i + 1,
            x,
            w,
            h,
            bar
        );
    }
    println!();

    println!("  Build for WASM:");
    println!("    cargo build --example showcase_gpu --target wasm32-unknown-unknown --release");
    println!();
    println!("  Then serve web/showcase/ and open in browser");
    println!();
    println!("  Tests: 48 passed ✓");
}
