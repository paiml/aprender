//! WebGPU renderer - GPU-accelerated rendering via WGSL shaders.
//!
//! This module provides high-performance rendering using WebGPU.
//! Falls back to Canvas2D when WebGPU is not available.
//!
//! ## Text Rendering
//!
//! Text is rendered using a glyph atlas approach:
//! 1. Glyphs are rasterized and cached in a GPU texture atlas
//! 2. Text is rendered as instanced quads sampling from the atlas
//! 3. SDF (Signed Distance Field) allows smooth scaling

use presentar_core::draw::DrawCommand;
use presentar_core::{Color, Point, Rect};
use std::collections::HashMap;

/// GPU vertex data.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// GPU instance data for batched rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub bounds: [f32; 4], // x, y, width, height
    pub color: [f32; 4],  // r, g, b, a
    pub corner_radius: f32,
    pub shape_type: u32, // 0=rect, 1=circle, 2=rounded_rect
    pub _padding: [f32; 2],
}

impl Instance {
    pub fn rect(bounds: &Rect, color: &Color) -> Self {
        Self {
            bounds: [bounds.x, bounds.y, bounds.width, bounds.height],
            color: [color.r, color.g, color.b, color.a],
            corner_radius: 0.0,
            shape_type: 0,
            _padding: [0.0; 2],
        }
    }

    pub fn rounded_rect(bounds: &Rect, radius: f32, color: &Color) -> Self {
        Self {
            bounds: [bounds.x, bounds.y, bounds.width, bounds.height],
            color: [color.r, color.g, color.b, color.a],
            corner_radius: radius,
            shape_type: 2,
            _padding: [0.0; 2],
        }
    }

    pub fn circle(center: &Point, radius: f32, color: &Color) -> Self {
        Self {
            bounds: [
                center.x - radius,
                center.y - radius,
                radius * 2.0,
                radius * 2.0,
            ],
            color: [color.r, color.g, color.b, color.a],
            corner_radius: radius,
            shape_type: 1,
            _padding: [0.0; 2],
        }
    }
}

/// Uniforms for the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub viewport: [f32; 2],
    pub _padding: [f32; 2],
}

/// Unit quad vertices for instanced rendering.
pub const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 0.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0],
        uv: [0.0, 1.0],
    },
];

pub const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

/// Convert DrawCommands to GPU instances for batched rendering.
pub fn commands_to_instances(commands: &[DrawCommand]) -> Vec<Instance> {
    let mut instances = Vec::with_capacity(commands.len());

    for cmd in commands {
        match cmd {
            DrawCommand::Rect {
                bounds,
                radius,
                style,
            } => {
                if let Some(fill) = style.fill {
                    if radius.is_zero() {
                        instances.push(Instance::rect(bounds, &fill));
                    } else {
                        instances.push(Instance::rounded_rect(bounds, radius.top_left, &fill));
                    }
                }
            }
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                if let Some(fill) = style.fill {
                    instances.push(Instance::circle(center, *radius, &fill));
                }
            }
            // Text requires a separate text rendering pipeline
            DrawCommand::Text { .. } => {}
            // Groups recurse
            DrawCommand::Group { children, .. } => {
                instances.extend(commands_to_instances(children));
            }
            _ => {}
        }
    }

    instances
}

/// Check if WebGPU is available in the current environment.
/// Note: On WASM, actual availability must be checked at runtime via wgpu adapter request.
pub fn is_webgpu_available() -> bool {
    // WebGPU availability is determined by wgpu at runtime.
    // This returns true to indicate the code paths are available;
    // actual GPU adapter availability is checked when creating the renderer.
    true
}

// =============================================================================
// Text Rendering Types
// =============================================================================

/// A unique key for caching glyphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// The character codepoint.
    pub codepoint: u32,
    /// Font size in pixels (quantized for caching).
    pub size_px: u16,
    /// Font weight (100-900).
    pub weight: u16,
}

impl GlyphKey {
    /// Create a new glyph key.
    #[must_use]
    pub const fn new(codepoint: char, size_px: u16, weight: u16) -> Self {
        Self {
            codepoint: codepoint as u32,
            size_px,
            weight,
        }
    }

    /// Create from codepoint value.
    #[must_use]
    pub const fn from_codepoint(codepoint: u32, size_px: u16, weight: u16) -> Self {
        Self {
            codepoint,
            size_px,
            weight,
        }
    }
}

/// Region within the glyph atlas.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtlasRegion {
    /// X position in atlas (pixels).
    pub x: u16,
    /// Y position in atlas (pixels).
    pub y: u16,
    /// Width in atlas (pixels).
    pub width: u16,
    /// Height in atlas (pixels).
    pub height: u16,
}

impl AtlasRegion {
    /// Create a new atlas region.
    #[must_use]
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Convert to UV coordinates given atlas dimensions.
    #[must_use]
    pub fn to_uvs(&self, atlas_width: u32, atlas_height: u32) -> [f32; 4] {
        let w = atlas_width as f32;
        let h = atlas_height as f32;
        [
            self.x as f32 / w,
            self.y as f32 / h,
            (self.x + self.width) as f32 / w,
            (self.y + self.height) as f32 / h,
        ]
    }

    /// Check if this region is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Cached glyph metrics and atlas location.
#[derive(Clone, Copy, Debug, Default)]
pub struct CachedGlyph {
    /// Region in the glyph atlas.
    pub region: AtlasRegion,
    /// Horizontal advance width.
    pub advance_x: f32,
    /// Vertical advance height (usually 0 for horizontal text).
    pub advance_y: f32,
    /// Horizontal bearing (offset from baseline).
    pub bearing_x: f32,
    /// Vertical bearing (offset from baseline).
    pub bearing_y: f32,
}

impl CachedGlyph {
    /// Create a new cached glyph.
    #[must_use]
    pub const fn new(region: AtlasRegion, advance_x: f32, bearing_x: f32, bearing_y: f32) -> Self {
        Self {
            region,
            advance_x,
            advance_y: 0.0,
            bearing_x,
            bearing_y,
        }
    }
}

/// GPU instance data for text glyph rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    /// Position and size: x, y, width, height
    pub bounds: [f32; 4],
    /// UV coordinates: u0, v0, u1, v1
    pub uvs: [f32; 4],
    /// Text color: r, g, b, a
    pub color: [f32; 4],
}

impl GlyphInstance {
    /// Create a new glyph instance.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32, uvs: [f32; 4], color: &Color) -> Self {
        Self {
            bounds: [x, y, width, height],
            uvs,
            color: [color.r, color.g, color.b, color.a],
        }
    }

    /// Create from a cached glyph.
    #[must_use]
    pub fn from_cached(
        glyph: &CachedGlyph,
        x: f32,
        baseline_y: f32,
        scale: f32,
        atlas_size: (u32, u32),
        color: &Color,
    ) -> Self {
        let width = glyph.region.width as f32 * scale;
        let height = glyph.region.height as f32 * scale;
        let glyph_x = x + glyph.bearing_x * scale;
        let glyph_y = baseline_y - glyph.bearing_y * scale;

        Self::new(
            glyph_x,
            glyph_y,
            width,
            height,
            glyph.region.to_uvs(atlas_size.0, atlas_size.1),
            color,
        )
    }
}

/// Simple glyph cache for CPU-side glyph management.
#[derive(Debug, Default)]
pub struct GlyphCache {
    /// Cached glyphs by key.
    glyphs: HashMap<GlyphKey, CachedGlyph>,
    /// Atlas dimensions.
    atlas_width: u32,
    atlas_height: u32,
    /// Next available row in atlas.
    next_row_y: u16,
    /// Current row height.
    row_height: u16,
    /// Current X position in row.
    current_x: u16,
}

impl GlyphCache {
    /// Create a new glyph cache with the given atlas dimensions.
    #[must_use]
    pub fn new(atlas_width: u32, atlas_height: u32) -> Self {
        Self {
            glyphs: HashMap::new(),
            atlas_width,
            atlas_height,
            next_row_y: 0,
            row_height: 0,
            current_x: 0,
        }
    }

    /// Get atlas dimensions.
    #[must_use]
    pub const fn atlas_size(&self) -> (u32, u32) {
        (self.atlas_width, self.atlas_height)
    }

    /// Look up a cached glyph.
    #[must_use]
    pub fn get(&self, key: &GlyphKey) -> Option<&CachedGlyph> {
        self.glyphs.get(key)
    }

    /// Check if a glyph is cached.
    #[must_use]
    pub fn contains(&self, key: &GlyphKey) -> bool {
        self.glyphs.contains_key(key)
    }

    /// Number of cached glyphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// Try to allocate space for a glyph in the atlas.
    /// Returns the region if successful, None if atlas is full.
    #[must_use]
    pub fn allocate(&mut self, width: u16, height: u16) -> Option<AtlasRegion> {
        if width == 0 || height == 0 {
            return Some(AtlasRegion::default());
        }

        // Add 1px padding between glyphs
        let padded_width = width + 1;
        let padded_height = height + 1;

        // Check if we need to start a new row
        if self.current_x + padded_width > self.atlas_width as u16 {
            self.next_row_y += self.row_height;
            self.current_x = 0;
            self.row_height = 0;
        }

        // Check if we have room vertically
        if self.next_row_y + padded_height > self.atlas_height as u16 {
            return None;
        }

        let region = AtlasRegion::new(self.current_x, self.next_row_y, width, height);
        self.current_x += padded_width;
        self.row_height = self.row_height.max(padded_height);

        Some(region)
    }

    /// Insert a glyph with pre-computed region.
    pub fn insert(&mut self, key: GlyphKey, glyph: CachedGlyph) {
        self.glyphs.insert(key, glyph);
    }

    /// Clear all cached glyphs (atlas needs to be cleared too).
    pub fn clear(&mut self) {
        self.glyphs.clear();
        self.next_row_y = 0;
        self.row_height = 0;
        self.current_x = 0;
    }

    /// Get utilization percentage of the atlas.
    #[must_use]
    pub fn utilization(&self) -> f32 {
        let used = self.next_row_y as u32 * self.atlas_width + self.current_x as u32;
        let total = self.atlas_width * self.atlas_height;
        if total == 0 {
            0.0
        } else {
            used as f32 / total as f32
        }
    }
}

/// Text layout result for rendering.
#[derive(Debug, Default)]
pub struct TextLayout {
    /// Glyph instances for rendering.
    pub glyphs: Vec<GlyphInstance>,
    /// Total width of the laid out text.
    pub width: f32,
    /// Height of the laid out text.
    pub height: f32,
    /// Number of lines.
    pub lines: u32,
}

impl TextLayout {
    /// Create a new empty text layout.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            glyphs: Vec::new(),
            width: 0.0,
            height: 0.0,
            lines: 0,
        }
    }

    /// Check if the layout is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// Get bounds as a rect.
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            width: self.width,
            height: self.height,
        }
    }
}

/// Text alignment options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextAlign {
    /// Left aligned (default).
    #[default]
    Left,
    /// Center aligned.
    Center,
    /// Right aligned.
    Right,
}

/// Text rendering options.
#[derive(Clone, Debug)]
pub struct TextOptions {
    /// Font size in pixels.
    pub size_px: f32,
    /// Font weight (100-900).
    pub weight: u16,
    /// Line height multiplier.
    pub line_height: f32,
    /// Letter spacing in pixels.
    pub letter_spacing: f32,
    /// Text alignment.
    pub align: TextAlign,
    /// Maximum width for wrapping (None = no wrap).
    pub max_width: Option<f32>,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            size_px: 16.0,
            weight: 400,
            line_height: 1.2,
            letter_spacing: 0.0,
            align: TextAlign::Left,
            max_width: None,
        }
    }
}

impl TextOptions {
    /// Create new text options with the given size.
    #[must_use]
    pub fn new(size_px: f32) -> Self {
        Self {
            size_px,
            ..Default::default()
        }
    }

    /// Set font weight.
    #[must_use]
    pub const fn with_weight(mut self, weight: u16) -> Self {
        self.weight = weight;
        self
    }

    /// Set line height multiplier.
    #[must_use]
    pub const fn with_line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    /// Set letter spacing.
    #[must_use]
    pub const fn with_letter_spacing(mut self, letter_spacing: f32) -> Self {
        self.letter_spacing = letter_spacing;
        self
    }

    /// Set text alignment.
    #[must_use]
    pub const fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    /// Set maximum width for wrapping.
    #[must_use]
    pub const fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = Some(max_width);
        self
    }
}

/// Measure text without rendering.
#[must_use]
pub fn measure_text(text: &str, cache: &GlyphCache, options: &TextOptions) -> (f32, f32) {
    let mut width: f32 = 0.0;
    let mut max_width: f32 = 0.0;
    let mut lines = 1u32;
    let scale = options.size_px / 16.0; // Assuming base size of 16px

    for ch in text.chars() {
        if ch == '\n' {
            max_width = max_width.max(width);
            width = 0.0;
            lines += 1;
            continue;
        }

        let key = GlyphKey::new(ch, options.size_px as u16, options.weight);
        if let Some(glyph) = cache.get(&key) {
            width += glyph.advance_x * scale + options.letter_spacing;
        } else {
            // Fallback: estimate based on character
            width += options.size_px * 0.5 + options.letter_spacing;
        }
    }

    max_width = max_width.max(width);
    let height = lines as f32 * options.size_px * options.line_height;

    (max_width, height)
}

/// Check if word wrap should occur at this character.
fn should_wrap(line_width: f32, max_width: Option<f32>, ch: char) -> bool {
    max_width.is_some_and(|mw| line_width > mw && ch.is_whitespace())
}

/// Get glyph advance width (or fallback estimate).
fn get_advance(cache: &GlyphCache, ch: char, options: &TextOptions, scale: f32) -> f32 {
    let key = GlyphKey::new(ch, options.size_px as u16, options.weight);
    cache
        .get(&key)
        .map(|g| g.advance_x * scale + options.letter_spacing)
        .unwrap_or(options.size_px * 0.5 + options.letter_spacing)
}

/// Layout text for rendering.
#[must_use]
pub fn layout_text(
    text: &str,
    x: f32,
    baseline_y: f32,
    cache: &GlyphCache,
    options: &TextOptions,
    color: &Color,
) -> TextLayout {
    let mut layout = TextLayout::new();
    let mut cursor_x = x;
    let mut cursor_y = baseline_y;
    let mut line_width: f32 = 0.0;
    let scale = options.size_px / 16.0;
    let atlas_size = cache.atlas_size();
    let line_advance = options.size_px * options.line_height;

    layout.lines = 1;

    for ch in text.chars() {
        // Handle newline or word wrap
        if ch == '\n' || should_wrap(line_width, options.max_width, ch) {
            layout.width = layout.width.max(line_width);
            line_width = 0.0;
            cursor_x = x;
            cursor_y += line_advance;
            layout.lines += 1;
            if ch == '\n' || ch.is_whitespace() {
                continue;
            }
        }

        // Render glyph if cached
        let key = GlyphKey::new(ch, options.size_px as u16, options.weight);
        if let Some(glyph) = cache.get(&key) {
            if !glyph.region.is_empty() {
                let instance =
                    GlyphInstance::from_cached(glyph, cursor_x, cursor_y, scale, atlas_size, color);
                layout.glyphs.push(instance);
            }
        }

        // Advance cursor
        let advance = get_advance(cache, ch, options, scale);
        cursor_x += advance;
        line_width += advance;
    }

    layout.width = layout.width.max(line_width);
    layout.height = layout.lines as f32 * line_advance;

    layout
}

// =============================================================================
// WebGPU Renderer Implementation
// =============================================================================

/// Configuration for WebGPU renderer.
#[derive(Debug, Clone)]
pub struct WebGpuConfig {
    /// Initial viewport width.
    pub width: u32,
    /// Initial viewport height.
    pub height: u32,
    /// Preferred presentation format (None = auto-detect).
    pub format: Option<wgpu::TextureFormat>,
    /// Maximum number of instances per draw call.
    pub max_instances: usize,
    /// Enable MSAA (sample count).
    pub sample_count: u32,
}

impl Default for WebGpuConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            format: None,
            max_instances: 10_000,
            sample_count: 1,
        }
    }
}

impl WebGpuConfig {
    /// Create config with dimensions.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: None,
            max_instances: 10_000,
            sample_count: 1,
        }
    }

    /// Set MSAA sample count (1, 2, or 4).
    #[must_use]
    pub const fn with_msaa(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    /// Set max instances per batch.
    #[must_use]
    pub const fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances = max;
        self
    }
}

/// Error types for WebGPU renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebGpuError {
    /// No GPU adapter found.
    NoAdapter,
    /// Failed to get device.
    NoDevice(String),
    /// Surface configuration failed.
    SurfaceError(String),
    /// Shader compilation failed.
    ShaderError(String),
    /// Pipeline creation failed.
    PipelineError(String),
    /// Buffer creation failed.
    BufferError(String),
}

impl std::fmt::Display for WebGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no GPU adapter found"),
            Self::NoDevice(msg) => write!(f, "failed to get device: {msg}"),
            Self::SurfaceError(msg) => write!(f, "surface error: {msg}"),
            Self::ShaderError(msg) => write!(f, "shader error: {msg}"),
            Self::PipelineError(msg) => write!(f, "pipeline error: {msg}"),
            Self::BufferError(msg) => write!(f, "buffer error: {msg}"),
        }
    }
}

impl std::error::Error for WebGpuError {}

/// Result type for WebGPU operations.
pub type WebGpuResult<T> = Result<T, WebGpuError>;

/// GPU resources needed for rendering.
pub struct GpuResources {
    /// GPU device.
    pub device: wgpu::Device,
    /// Command queue.
    pub queue: wgpu::Queue,
    /// Vertex buffer for unit quad.
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer for unit quad.
    pub index_buffer: wgpu::Buffer,
    /// Instance buffer for shapes.
    pub instance_buffer: wgpu::Buffer,
    /// Uniform buffer.
    pub uniform_buffer: wgpu::Buffer,
    /// Bind group for uniforms.
    pub bind_group: wgpu::BindGroup,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Shape render pipeline.
    pub shape_pipeline: wgpu::RenderPipeline,
    /// Max instances.
    pub max_instances: usize,
    /// Current surface format.
    pub format: wgpu::TextureFormat,
}

/// Builder for GPU resources.
pub struct GpuResourceBuilder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    max_instances: usize,
}

impl GpuResourceBuilder {
    /// Create a new builder with device and queue.
    #[must_use]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            device,
            queue,
            format,
            max_instances: 10_000,
        }
    }

    /// Set max instances.
    #[must_use]
    pub const fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances = max;
        self
    }

    /// Build the GPU resources.
    ///
    /// # Errors
    ///
    /// Returns error if shader compilation or pipeline creation fails.
    pub fn build(self) -> WebGpuResult<GpuResources> {
        // Create vertex buffer for unit quad
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Quad Vertex Buffer"),
                contents: bytemuck::cast_slice(QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // Create index buffer
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Quad Index Buffer"),
                contents: bytemuck::cast_slice(QUAD_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Create instance buffer
        let instance_buffer_size = std::mem::size_of::<Instance>() * self.max_instances;
        let instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: instance_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Compile shader
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Primitive Shader"),
                source: wgpu::ShaderSource::Wgsl(PRIMITIVE_SHADER.into()),
            });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shape Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Instance attributes
        let instance_attribs = [
            // bounds (vec4)
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            // color (vec4)
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            // corner_radius + shape_type (packed as float + u32)
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 5,
                format: wgpu::VertexFormat::Uint32,
            },
        ];

        // Create render pipeline
        let shape_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Shape Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        Vertex::desc(),
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Instance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &instance_attribs,
                        },
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Ok(GpuResources {
            device: self.device,
            queue: self.queue,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            shape_pipeline,
            max_instances: self.max_instances,
            format: self.format,
        })
    }
}

// DeviceExt provides buffer creation methods with initial data
use wgpu::util::DeviceExt;

/// WGSL shader for primitive rendering.
pub const PRIMITIVE_SHADER: &str = r#"
struct Uniforms {
    viewport: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct InstanceInput {
    @location(2) bounds: vec4<f32>,      // x, y, width, height
    @location(3) color: vec4<f32>,       // r, g, b, a
    @location(4) corner_radius: f32,
    @location(5) shape_type: u32,        // 0=rect, 1=circle, 2=rounded_rect
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) @interpolate(flat) shape_type: u32,
    @location(3) @interpolate(flat) corner_radius: f32,
    @location(4) size: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform vertex position by instance bounds
    let world_pos = instance.bounds.xy + vertex.position * instance.bounds.zw;

    // Convert to clip space (-1 to 1)
    let clip_x = (world_pos.x / uniforms.viewport.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (world_pos.y / uniforms.viewport.y) * 2.0;

    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.color = instance.color;
    out.local_pos = vertex.position;
    out.shape_type = instance.shape_type;
    out.corner_radius = instance.corner_radius;
    out.size = instance.bounds.zw;

    return out;
}

fn sdf_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_rounded_rect(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - size + vec2<f32>(radius);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map local_pos from [0,1] to [-1,1] centered
    let centered = (in.local_pos - vec2<f32>(0.5)) * 2.0;

    var alpha = in.color.a;

    if in.shape_type == 1u {
        // Circle: use SDF
        let d = sdf_circle(centered, 1.0);
        let aa = fwidth(d);
        alpha *= 1.0 - smoothstep(-aa, aa, d);
    } else if in.shape_type == 2u {
        // Rounded rect: use SDF
        let half_size = in.size * 0.5;
        let p = centered * half_size;
        let d = sdf_rounded_rect(p, half_size, in.corner_radius);
        let aa = fwidth(d);
        alpha *= 1.0 - smoothstep(-aa, aa, d);
    }
    // shape_type == 0: regular rect, no SDF needed

    return vec4<f32>(in.color.rgb, alpha);
}
"#;

/// Frame statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Number of draw calls.
    pub draw_calls: u32,
    /// Number of instances rendered.
    pub instances: u32,
    /// Frame time in milliseconds.
    pub frame_time_ms: f32,
}

impl FrameStats {
    /// Reset stats for new frame.
    pub fn reset(&mut self) {
        self.draw_calls = 0;
        self.instances = 0;
        self.frame_time_ms = 0.0;
    }
}

/// Render batched instances using the GPU resources.
pub fn render_instances(
    resources: &GpuResources,
    instances: &[Instance],
    viewport: (f32, f32),
    view: &wgpu::TextureView,
    clear_color: Option<Color>,
) -> FrameStats {
    let mut stats = FrameStats::default();

    if instances.is_empty() {
        return stats;
    }
    contract_pre_render!(resources);

    // Update uniform buffer
    let uniforms = Uniforms {
        viewport: [viewport.0, viewport.1],
        _padding: [0.0, 0.0],
    };
    resources
        .queue
        .write_buffer(&resources.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

    // Batch instances
    let mut offset = 0;
    while offset < instances.len() {
        let batch_size = (instances.len() - offset).min(resources.max_instances);
        let batch = &instances[offset..offset + batch_size];

        // Update instance buffer
        resources
            .queue
            .write_buffer(&resources.instance_buffer, 0, bytemuck::cast_slice(batch));

        // Create command encoder
        let mut encoder =
            resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // Render pass
        {
            let load_op = if offset == 0 && clear_color.is_some() {
                let c = clear_color.as_ref().expect("clear color");
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: f64::from(c.r),
                    g: f64::from(c.g),
                    b: f64::from(c.b),
                    a: f64::from(c.a),
                })
            } else {
                wgpu::LoadOp::Load
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shape Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&resources.shape_pipeline);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, resources.instance_buffer.slice(..));
            pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..batch_size as u32);
        }

        resources.queue.submit(std::iter::once(encoder.finish()));

        stats.draw_calls += 1;
        stats.instances += batch_size as u32;
        offset += batch_size;
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_size() {
        assert_eq!(std::mem::size_of::<Vertex>(), 16);
    }

    #[test]
    fn test_instance_size() {
        assert_eq!(std::mem::size_of::<Instance>(), 48);
    }

    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 16);
    }

    #[test]
    fn test_instance_rect() {
        let bounds = Rect::new(10.0, 20.0, 100.0, 50.0);
        let color = Color::RED;
        let inst = Instance::rect(&bounds, &color);

        assert_eq!(inst.bounds, [10.0, 20.0, 100.0, 50.0]);
        assert_eq!(inst.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(inst.shape_type, 0);
    }

    #[test]
    fn test_instance_circle() {
        let center = Point::new(50.0, 50.0);
        let inst = Instance::circle(&center, 25.0, &Color::BLUE);

        assert_eq!(inst.bounds, [25.0, 25.0, 50.0, 50.0]);
        assert_eq!(inst.shape_type, 1);
    }

    #[test]
    fn test_instance_rounded_rect() {
        let bounds = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inst = Instance::rounded_rect(&bounds, 8.0, &Color::GREEN);

        assert_eq!(inst.corner_radius, 8.0);
        assert_eq!(inst.shape_type, 2);
    }

    #[test]
    fn test_commands_to_instances() {
        let commands = vec![
            DrawCommand::filled_rect(Rect::new(0.0, 0.0, 100.0, 100.0), Color::RED),
            DrawCommand::filled_circle(Point::new(50.0, 50.0), 25.0, Color::BLUE),
        ];

        let instances = commands_to_instances(&commands);
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_quad_vertices() {
        assert_eq!(QUAD_VERTICES.len(), 4);
        assert_eq!(QUAD_INDICES.len(), 6);
    }

    // =============================================================================
    // Text Rendering Tests
    // =============================================================================

    #[test]
    fn test_glyph_key_new() {
        let key = GlyphKey::new('A', 16, 400);
        assert_eq!(key.codepoint, 'A' as u32);
        assert_eq!(key.size_px, 16);
        assert_eq!(key.weight, 400);
    }

    #[test]
    fn test_glyph_key_from_codepoint() {
        let key = GlyphKey::from_codepoint(65, 24, 700);
        assert_eq!(key.codepoint, 65);
        assert_eq!(key.size_px, 24);
        assert_eq!(key.weight, 700);
    }

    #[test]
    fn test_glyph_key_hash_eq() {
        use std::collections::HashSet;
        let key1 = GlyphKey::new('A', 16, 400);
        let key2 = GlyphKey::new('A', 16, 400);
        let key3 = GlyphKey::new('B', 16, 400);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        let mut set = HashSet::new();
        set.insert(key1);
        assert!(set.contains(&key2));
        assert!(!set.contains(&key3));
    }

    #[test]
    fn test_atlas_region_new() {
        let region = AtlasRegion::new(10, 20, 30, 40);
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.width, 30);
        assert_eq!(region.height, 40);
    }

    #[test]
    fn test_atlas_region_to_uvs() {
        let region = AtlasRegion::new(0, 0, 64, 64);
        let uvs = region.to_uvs(256, 256);
        assert_eq!(uvs[0], 0.0);
        assert_eq!(uvs[1], 0.0);
        assert_eq!(uvs[2], 0.25);
        assert_eq!(uvs[3], 0.25);
    }

    #[test]
    fn test_atlas_region_to_uvs_offset() {
        let region = AtlasRegion::new(128, 128, 64, 64);
        let uvs = region.to_uvs(256, 256);
        assert_eq!(uvs[0], 0.5);
        assert_eq!(uvs[1], 0.5);
        assert_eq!(uvs[2], 0.75);
        assert_eq!(uvs[3], 0.75);
    }

    #[test]
    fn test_atlas_region_is_empty() {
        assert!(AtlasRegion::default().is_empty());
        assert!(AtlasRegion::new(0, 0, 0, 10).is_empty());
        assert!(AtlasRegion::new(0, 0, 10, 0).is_empty());
        assert!(!AtlasRegion::new(0, 0, 10, 10).is_empty());
    }

    #[test]
    fn test_cached_glyph_new() {
        let region = AtlasRegion::new(0, 0, 16, 20);
        let glyph = CachedGlyph::new(region, 8.0, 1.0, 18.0);

        assert_eq!(glyph.advance_x, 8.0);
        assert_eq!(glyph.bearing_x, 1.0);
        assert_eq!(glyph.bearing_y, 18.0);
        assert_eq!(glyph.advance_y, 0.0);
    }

    #[test]
    fn test_glyph_instance_size() {
        assert_eq!(std::mem::size_of::<GlyphInstance>(), 48);
    }

    #[test]
    fn test_glyph_instance_new() {
        let inst = GlyphInstance::new(10.0, 20.0, 30.0, 40.0, [0.0, 0.0, 1.0, 1.0], &Color::RED);

        assert_eq!(inst.bounds, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(inst.uvs, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(inst.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_glyph_instance_from_cached() {
        let region = AtlasRegion::new(0, 0, 16, 20);
        let glyph = CachedGlyph::new(region, 10.0, 2.0, 18.0);
        let inst = GlyphInstance::from_cached(&glyph, 100.0, 50.0, 1.0, (256, 256), &Color::BLACK);

        // x = 100 + 2*1 = 102
        // y = 50 - 18*1 = 32
        assert_eq!(inst.bounds[0], 102.0);
        assert_eq!(inst.bounds[1], 32.0);
        assert_eq!(inst.bounds[2], 16.0);
        assert_eq!(inst.bounds[3], 20.0);
    }

    #[test]
    fn test_glyph_cache_new() {
        let cache = GlyphCache::new(1024, 1024);
        assert_eq!(cache.atlas_size(), (1024, 1024));
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_glyph_cache_allocate() {
        let mut cache = GlyphCache::new(256, 256);

        let region = cache.allocate(32, 32).unwrap();
        assert_eq!(region.x, 0);
        assert_eq!(region.y, 0);
        assert_eq!(region.width, 32);
        assert_eq!(region.height, 32);

        let region2 = cache.allocate(32, 32).unwrap();
        assert_eq!(region2.x, 33); // 32 + 1 padding
        assert_eq!(region2.y, 0);
    }

    #[test]
    fn test_glyph_cache_allocate_new_row() {
        let mut cache = GlyphCache::new(128, 256);

        // First allocation takes 33 pixels (32 + 1 padding)
        let r1 = cache.allocate(32, 32).unwrap();
        assert_eq!(r1.x, 0);
        assert_eq!(r1.y, 0);

        // Second allocation fits at x=33
        let r2 = cache.allocate(32, 32).unwrap();
        assert_eq!(r2.x, 33);
        assert_eq!(r2.y, 0);

        // Third allocation fits at x=66
        let r3 = cache.allocate(32, 32).unwrap();
        assert_eq!(r3.x, 66);
        assert_eq!(r3.y, 0);

        // Fourth allocation (99+33=132 > 128) starts new row
        let r4 = cache.allocate(32, 32).unwrap();
        assert_eq!(r4.x, 0);
        assert_eq!(r4.y, 33); // New row after 32 + 1 padding
    }

    #[test]
    fn test_glyph_cache_allocate_empty() {
        let mut cache = GlyphCache::new(256, 256);
        let region = cache.allocate(0, 0).unwrap();
        assert!(region.is_empty());
    }

    #[test]
    fn test_glyph_cache_allocate_full() {
        // Atlas is 128x128, glyphs are 32x32 with 1px padding = 33x33
        // Per row: floor(128/33) = 3 glyphs fit per row (at x=0, 33, 66)
        // Per column: floor(128/33) = 3 rows fit (at y=0, 33, 66)
        // Total capacity: 3*3 = 9 glyphs
        let mut cache = GlyphCache::new(128, 128);

        // Allocate 9 glyphs (fills the atlas)
        for i in 0..9 {
            let region = cache.allocate(32, 32);
            assert!(region.is_some(), "glyph {} should fit", i);
        }

        // 10th glyph should fail - atlas is full
        assert!(cache.allocate(32, 32).is_none());
    }

    #[test]
    fn test_glyph_cache_insert_get() {
        let mut cache = GlyphCache::new(256, 256);
        let key = GlyphKey::new('A', 16, 400);
        let region = cache.allocate(16, 20).unwrap();
        let glyph = CachedGlyph::new(region, 10.0, 1.0, 18.0);

        cache.insert(key, glyph);

        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.advance_x, 10.0);
    }

    #[test]
    fn test_glyph_cache_clear() {
        let mut cache = GlyphCache::new(256, 256);
        let key = GlyphKey::new('A', 16, 400);
        let region = cache.allocate(16, 20).unwrap();
        let glyph = CachedGlyph::new(region, 10.0, 1.0, 18.0);
        cache.insert(key, glyph);

        cache.clear();

        assert!(cache.is_empty());
        assert!(!cache.contains(&key));
    }

    #[test]
    fn test_glyph_cache_utilization() {
        let mut cache = GlyphCache::new(100, 100);
        assert_eq!(cache.utilization(), 0.0);

        // Allocate some space
        cache.allocate(50, 50).unwrap();
        assert!(cache.utilization() > 0.0);
    }

    #[test]
    fn test_text_layout_new() {
        let layout = TextLayout::new();
        assert!(layout.is_empty());
        assert_eq!(layout.width, 0.0);
        assert_eq!(layout.height, 0.0);
        assert_eq!(layout.lines, 0);
    }

    #[test]
    fn test_text_layout_bounds() {
        let mut layout = TextLayout::new();
        layout.width = 100.0;
        layout.height = 50.0;

        let bounds = layout.bounds();
        assert_eq!(bounds.x, 0.0);
        assert_eq!(bounds.y, 0.0);
        assert_eq!(bounds.width, 100.0);
        assert_eq!(bounds.height, 50.0);
    }

    #[test]
    fn test_text_align_default() {
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    #[test]
    fn test_text_options_default() {
        let opts = TextOptions::default();
        assert_eq!(opts.size_px, 16.0);
        assert_eq!(opts.weight, 400);
        assert_eq!(opts.line_height, 1.2);
        assert_eq!(opts.letter_spacing, 0.0);
        assert_eq!(opts.align, TextAlign::Left);
        assert!(opts.max_width.is_none());
    }

    #[test]
    fn test_text_options_builder() {
        let opts = TextOptions::new(24.0)
            .with_weight(700)
            .with_line_height(1.5)
            .with_letter_spacing(2.0)
            .with_align(TextAlign::Center)
            .with_max_width(200.0);

        assert_eq!(opts.size_px, 24.0);
        assert_eq!(opts.weight, 700);
        assert_eq!(opts.line_height, 1.5);
        assert_eq!(opts.letter_spacing, 2.0);
        assert_eq!(opts.align, TextAlign::Center);
        assert_eq!(opts.max_width, Some(200.0));
    }

    #[test]
    fn test_measure_text_empty() {
        let cache = GlyphCache::new(256, 256);
        let opts = TextOptions::default();
        let (width, height) = measure_text("", &cache, &opts);

        assert_eq!(width, 0.0);
        // Single line height even for empty text
        assert_eq!(height, 16.0 * 1.2);
    }

    #[test]
    fn test_measure_text_newlines() {
        let cache = GlyphCache::new(256, 256);
        let opts = TextOptions::default();
        let (_, height) = measure_text("a\nb\nc", &cache, &opts);

        // 3 lines
        assert_eq!(height, 3.0 * 16.0 * 1.2);
    }

    #[test]
    fn test_layout_text_empty() {
        let cache = GlyphCache::new(256, 256);
        let opts = TextOptions::default();
        let layout = layout_text("", 0.0, 0.0, &cache, &opts, &Color::BLACK);

        assert!(layout.is_empty());
        assert_eq!(layout.lines, 1);
    }

    #[test]
    fn test_layout_text_with_cached_glyphs() {
        let mut cache = GlyphCache::new(256, 256);
        let opts = TextOptions::default();

        // Cache a glyph
        let key = GlyphKey::new('A', 16, 400);
        let region = cache.allocate(16, 20).unwrap();
        let glyph = CachedGlyph::new(region, 10.0, 1.0, 18.0);
        cache.insert(key, glyph);

        let layout = layout_text("A", 0.0, 20.0, &cache, &opts, &Color::BLACK);

        assert_eq!(layout.glyphs.len(), 1);
        assert_eq!(layout.lines, 1);
    }

    #[test]
    fn test_layout_text_multiline() {
        let cache = GlyphCache::new(256, 256);
        let opts = TextOptions::default();
        let layout = layout_text("line1\nline2", 0.0, 20.0, &cache, &opts, &Color::BLACK);

        assert_eq!(layout.lines, 2);
    }

    // =============================================================================
    // WebGPU Config & Error Tests
    // =============================================================================

    #[test]
    fn test_webgpu_config_default() {
        let config = WebGpuConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.format.is_none());
        assert_eq!(config.max_instances, 10_000);
        assert_eq!(config.sample_count, 1);
    }

    #[test]
    fn test_webgpu_config_new() {
        let config = WebGpuConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_webgpu_config_with_msaa() {
        let config = WebGpuConfig::default().with_msaa(4);
        assert_eq!(config.sample_count, 4);
    }

    #[test]
    fn test_webgpu_config_with_max_instances() {
        let config = WebGpuConfig::default().with_max_instances(50_000);
        assert_eq!(config.max_instances, 50_000);
    }

    #[test]
    fn test_webgpu_error_display() {
        assert_eq!(WebGpuError::NoAdapter.to_string(), "no GPU adapter found");
        assert_eq!(
            WebGpuError::NoDevice("timeout".to_string()).to_string(),
            "failed to get device: timeout"
        );
        assert_eq!(
            WebGpuError::SurfaceError("config failed".to_string()).to_string(),
            "surface error: config failed"
        );
        assert_eq!(
            WebGpuError::ShaderError("compile error".to_string()).to_string(),
            "shader error: compile error"
        );
        assert_eq!(
            WebGpuError::PipelineError("layout error".to_string()).to_string(),
            "pipeline error: layout error"
        );
        assert_eq!(
            WebGpuError::BufferError("allocation failed".to_string()).to_string(),
            "buffer error: allocation failed"
        );
    }

    #[test]
    fn test_webgpu_error_eq() {
        assert_eq!(WebGpuError::NoAdapter, WebGpuError::NoAdapter);
        assert_ne!(
            WebGpuError::NoAdapter,
            WebGpuError::NoDevice("x".to_string())
        );
    }

    #[test]
    fn test_frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.instances, 0);
        assert_eq!(stats.frame_time_ms, 0.0);
    }

    #[test]
    fn test_frame_stats_reset() {
        let mut stats = FrameStats {
            draw_calls: 10,
            instances: 1000,
            frame_time_ms: 16.5,
        };
        stats.reset();
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.instances, 0);
        assert_eq!(stats.frame_time_ms, 0.0);
    }

    #[test]
    fn test_primitive_shader_content() {
        // Verify shader has required entry points
        assert!(PRIMITIVE_SHADER.contains("fn vs_main"));
        assert!(PRIMITIVE_SHADER.contains("fn fs_main"));
        assert!(PRIMITIVE_SHADER.contains("sdf_circle"));
        assert!(PRIMITIVE_SHADER.contains("sdf_rounded_rect"));
    }

    #[test]
    fn test_is_webgpu_available() {
        // Should always return true (actual check is at runtime)
        assert!(is_webgpu_available());
    }
}
