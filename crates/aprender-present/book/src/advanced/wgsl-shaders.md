# WGSL Shaders

Presentar uses WebGPU Shading Language (WGSL) for GPU-accelerated rendering. All primitives are rendered using Signed Distance Fields (SDF) for resolution-independent anti-aliasing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Vertex Shader                                               │
│  - Transform quad vertices                                   │
│  - Pass instance data to fragment shader                     │
├─────────────────────────────────────────────────────────────┤
│  Fragment Shader                                             │
│  - SDF-based shape rendering                                 │
│  - Anti-aliased edges via smoothstep                         │
│  - Color and opacity blending                                │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures

### Vertex Input

```wgsl
struct VertexInput {
    @location(0) position: vec2<f32>,  // Quad corner (-1 to 1)
    @location(1) uv: vec2<f32>,        // Texture coordinates (0 to 1)
}
```

### Instance Data

Each rendered primitive is an instance with:

```wgsl
struct Instance {
    @location(2) pos: vec2<f32>,       // Screen position
    @location(3) size: vec2<f32>,      // Width, height
    @location(4) color: vec4<f32>,     // RGBA color
    @location(5) corner_radius: f32,   // Border radius
    @location(6) shape_type: u32,      // 0=rect, 1=circle, 2=text
}
```

### Uniforms

```wgsl
struct Uniforms {
    screen_size: vec2<f32>,           // Viewport dimensions
    time: f32,                        // Animation time
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
```

## Vertex Shader

The vertex shader transforms quad vertices to screen space:

```wgsl
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) size: vec2<f32>,
    @location(3) corner_radius: f32,
    @location(4) shape_type: u32,
}

@vertex
fn vs_main(vertex: VertexInput, instance: Instance) -> VertexOutput {
    var out: VertexOutput;

    // Transform to screen coordinates
    let screen_pos = instance.pos + vertex.position * instance.size * 0.5;
    let ndc = (screen_pos / uniforms.screen_size) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);

    // Pass through instance data
    out.uv = vertex.uv;
    out.color = instance.color;
    out.size = instance.size;
    out.corner_radius = instance.corner_radius;
    out.shape_type = instance.shape_type;

    return out;
}
```

## Fragment Shaders

### Rectangle with Rounded Corners (SDF)

```wgsl
fn sdf_rounded_rect(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - size + vec2<f32>(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - radius;
}

@fragment
fn fs_rounded_rect(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert UV to local coordinates centered at origin
    let local_pos = (in.uv - 0.5) * in.size;
    let half_size = in.size * 0.5;

    // Compute SDF
    let d = sdf_rounded_rect(local_pos, half_size, in.corner_radius);

    // Anti-aliased edge (1px feather)
    let alpha = 1.0 - smoothstep(-1.0, 1.0, d);

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
```

### Circle (SDF)

```wgsl
fn sdf_circle(p: vec2<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

@fragment
fn fs_circle(in: VertexOutput) -> @location(0) vec4<f32> {
    let local_pos = (in.uv - 0.5) * in.size;
    let radius = min(in.size.x, in.size.y) * 0.5;

    let d = sdf_circle(local_pos, radius);
    let alpha = 1.0 - smoothstep(-1.0, 1.0, d);

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
```

### Main Fragment Shader (Dispatch)

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    switch in.shape_type {
        case 0u: { return fs_rounded_rect(in); }  // Rectangle
        case 1u: { return fs_circle(in); }        // Circle
        case 2u: { return fs_text(in); }          // Text glyph
        default: { return in.color; }             // Fallback
    }
}
```

## Text Rendering

Text uses a glyph atlas with alpha coverage:

```wgsl
@group(0) @binding(1)
var glyph_atlas: texture_2d<f32>;

@group(0) @binding(2)
var glyph_sampler: sampler;

@fragment
fn fs_text(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(glyph_atlas, glyph_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
```

## Embedded Shader

Presentar embeds the primitive shader at compile time:

```rust
const PRIMITIVE_SHADER: &str = include_str!("shaders/primitive.wgsl");

// Or inline definition
const PRIMITIVE_SHADER: &str = r#"
    // Full WGSL shader source...
"#;
```

## Custom Effects

### Gradient Fill

```wgsl
@fragment
fn fs_gradient(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = in.uv.y;  // Vertical gradient
    let start_color = vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Red
    let end_color = vec4<f32>(0.0, 0.0, 1.0, 1.0);    // Blue
    return mix(start_color, end_color, t);
}
```

### Drop Shadow

```wgsl
@fragment
fn fs_shadow(in: VertexOutput) -> @location(0) vec4<f32> {
    let shadow_offset = vec2<f32>(4.0, 4.0);
    let shadow_blur = 8.0;

    // Shadow SDF (offset and blurred)
    let shadow_pos = (in.uv - 0.5) * in.size - shadow_offset;
    let shadow_d = sdf_rounded_rect(shadow_pos, in.size * 0.5, in.corner_radius);
    let shadow_alpha = 1.0 - smoothstep(-shadow_blur, shadow_blur, shadow_d);

    // Main shape
    let local_pos = (in.uv - 0.5) * in.size;
    let d = sdf_rounded_rect(local_pos, in.size * 0.5, in.corner_radius);
    let shape_alpha = 1.0 - smoothstep(-1.0, 1.0, d);

    // Composite shadow under shape
    let shadow_color = vec4<f32>(0.0, 0.0, 0.0, 0.3 * shadow_alpha);
    let shape_color = vec4<f32>(in.color.rgb, in.color.a * shape_alpha);

    return mix(shadow_color, shape_color, shape_alpha);
}
```

### Outline/Border

```wgsl
@fragment
fn fs_outline(in: VertexOutput) -> @location(0) vec4<f32> {
    let border_width = 2.0;
    let local_pos = (in.uv - 0.5) * in.size;

    let d = sdf_rounded_rect(local_pos, in.size * 0.5, in.corner_radius);

    // Inside the border
    let inner_alpha = 1.0 - smoothstep(-1.0, 1.0, d + border_width);
    // Outside the shape
    let outer_alpha = 1.0 - smoothstep(-1.0, 1.0, d);

    // Border = outer - inner
    let border_alpha = outer_alpha - inner_alpha;

    return vec4<f32>(in.color.rgb, in.color.a * border_alpha);
}
```

## Performance Tips

1. **Batch instances** - Group similar shapes into single draw calls
2. **Minimize overdraw** - Sort transparent objects back-to-front
3. **Use SDF** - Resolution-independent, GPU-friendly
4. **Atlas textures** - Single bind for all glyphs

## Verified Test

```rust
#[test]
fn test_sdf_concepts() {
    // SDF: negative inside, positive outside, zero at boundary
    fn sdf_circle(p: (f32, f32), radius: f32) -> f32 {
        (p.0 * p.0 + p.1 * p.1).sqrt() - radius
    }

    // Center of circle (inside)
    assert!(sdf_circle((0.0, 0.0), 10.0) < 0.0);

    // On the boundary
    assert!((sdf_circle((10.0, 0.0), 10.0)).abs() < 0.001);

    // Outside
    assert!(sdf_circle((15.0, 0.0), 10.0) > 0.0);
}

#[test]
fn test_smoothstep_antialiasing() {
    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    // Well inside (full opacity)
    assert!((smoothstep(-1.0, 1.0, -2.0) - 0.0).abs() < 0.001);

    // At boundary center (50% opacity)
    assert!((smoothstep(-1.0, 1.0, 0.0) - 0.5).abs() < 0.001);

    // Well outside (zero opacity)
    assert!((smoothstep(-1.0, 1.0, 2.0) - 1.0).abs() < 0.001);
}
```
