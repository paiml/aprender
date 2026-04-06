# Trueno-Viz: GPU Rendering Primitives

**Version:** 0.1.1 | **Crate:** `trueno-viz`

Trueno-Viz provides GPU-accelerated 2D rendering primitives built on Trueno's compute foundation. It serves as the rendering backend for Presentar and any visualization needs in the Sovereign AI Stack.

## Position in Stack

```
Presentar (Apps)
    │
    ▼
Trueno-Viz (Rendering)  ← YOU ARE HERE
    │
    ▼
Trueno (Compute)
```

## Core Abstractions

### Canvas

The primary drawing surface:

```rust
pub struct Canvas<'gpu> {
    context: &'gpu GpuContext,
    commands: Vec<DrawCommand>,
    viewport: Viewport,
}

impl Canvas<'_> {
    pub fn clear(&mut self, color: Color);
    pub fn draw(&mut self, cmd: DrawCommand);
    pub fn present(&mut self);
}
```

### Draw Commands

All rendering reduces to these primitives:

```rust
pub enum DrawCommand {
    // Geometry
    Path { points: Vec<Point>, closed: bool, style: StrokeStyle },
    Fill { path: PathRef, color: Color, rule: FillRule },
    Rect { bounds: Rect, radius: CornerRadius, style: BoxStyle },
    Circle { center: Point, radius: f32, style: BoxStyle },

    // Text (fontdue rasterization, GPU compositing)
    Text { content: String, position: Point, style: TextStyle },

    // Images (Trueno tensor → GPU texture)
    Image { tensor: TensorRef, bounds: Rect, sampling: Sampling },

    // Compositing
    Group { children: Vec<DrawCommand>, transform: Transform2D },
    Clip { bounds: Rect, child: Box<DrawCommand> },
    Opacity { alpha: f32, child: Box<DrawCommand> },
}
```

## WGSL Shader Pipeline

Trueno-Viz uses WebGPU Shading Language for GPU rendering:

```wgsl
// Fill shader
@vertex fn vs_fill(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment fn fs_fill(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
```

## Anti-Aliasing Strategy

| Technique | Use Case | Implementation |
|-----------|----------|----------------|
| **Hardware MSAA** | Solid fills | 4x MSAA via WebGPU |
| **SDF** | Text, icons | Shader-based, resolution-independent |
| **Analytical AA** | Lines, curves | Edge distance in fragment shader |

```wgsl
// Analytical AA for lines
@fragment fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    let dist = abs(in.edge_distance);
    let alpha = 1.0 - smoothstep(in.line_width - 1.0, in.line_width, dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
```

## Chart Primitives

Built on the Grammar of Graphics (Wilkinson, 2005):

```rust
pub enum ChartType {
    Line { series: Vec<Series>, interpolation: Interpolation },
    Bar { series: Vec<Series>, orientation: Orientation },
    Scatter { series: Vec<Series>, size_encoding: Option<String> },
    Heatmap { matrix: TensorRef, color_scale: ColorScale },
    Histogram { data: TensorRef, bins: BinStrategy },
}

impl ChartType {
    pub fn to_commands(&self, bounds: Rect, theme: &Theme) -> Vec<DrawCommand>;
}
```

## Color System

Perceptually uniform color operations:

```rust
impl Color {
    /// CIELAB color space (Levkowitz & Herman, 1992)
    pub fn to_lab(&self) -> LabColor;

    /// WCAG 2.1 contrast ratio
    pub fn contrast_ratio(&self, other: &Color) -> f32 {
        let l1 = self.relative_luminance();
        let l2 = other.relative_luminance();
        (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05)
    }
}
```

## Performance Targets

| Operation | Target | Backend |
|-----------|--------|---------|
| Path tessellation (1K points) | <1ms | Trueno SIMD |
| Fill rendering (10K triangles) | <2ms | WebGPU |
| Text layout (1K glyphs) | <5ms | fontdue + GPU |
| Chart update (100K points) | <16ms | Full pipeline |

## Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| WebGPU (native) | Stable | Primary target |
| WebGPU (WASM) | Stable | Browser deployment |
| WGPU fallback | Stable | Vulkan/Metal/DX12 |

## Integration with Trueno

Trueno-Viz leverages Trueno for:

- **Tensor → Texture:** Direct GPU upload for image data
- **SIMD tessellation:** Path point processing
- **Color math:** LAB/sRGB conversions

```rust
// Load tensor as GPU texture
let tensor: Tensor<f32> = trueno::load("image.bin")?;
let texture = canvas.upload_tensor(&tensor)?;
canvas.draw(DrawCommand::Image {
    tensor: texture,
    bounds: Rect::new(0.0, 0.0, 256.0, 256.0),
    sampling: Sampling::Linear,
});
```

## Recent Changes (v0.1.1)

- WebGPU compute physics demo
- WASM target support
- Comprehensive benchmark suite

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Presentar](./presentar.md) | [Trueno](./trueno.md)
