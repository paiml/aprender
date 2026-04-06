// Presentar WGSL Shader - Primitive Rendering
// Renders rectangles, circles, and basic shapes

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct Uniforms {
    viewport: vec2<f32>,
    _padding: vec2<f32>,
};

struct Instance {
    bounds: vec4<f32>,      // x, y, width, height
    color: vec4<f32>,       // r, g, b, a
    corner_radius: f32,
    shape_type: u32,        // 0=rect, 1=circle, 2=rounded_rect
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32
) -> VertexOutput {
    let inst = instances[instance_idx];

    // Transform vertex position to instance bounds
    let pos = vec2<f32>(
        inst.bounds.x + vertex.position.x * inst.bounds.z,
        inst.bounds.y + vertex.position.y * inst.bounds.w
    );

    // Convert to clip space (-1 to 1)
    let clip_pos = vec2<f32>(
        (pos.x / uniforms.viewport.x) * 2.0 - 1.0,
        1.0 - (pos.y / uniforms.viewport.y) * 2.0
    );

    var output: VertexOutput;
    output.clip_position = vec4<f32>(clip_pos, 0.0, 1.0);
    output.uv = vertex.uv;
    output.color = inst.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// Rounded rectangle SDF fragment shader
@fragment
fn fs_rounded_rect(input: VertexOutput, @builtin(instance_index) instance_idx: u32) -> @location(0) vec4<f32> {
    let inst = instances[instance_idx];
    let radius = inst.corner_radius;

    // UV is 0-1, convert to pixel coordinates relative to center
    let size = vec2<f32>(inst.bounds.z, inst.bounds.w);
    let half_size = size * 0.5;
    let p = (input.uv - 0.5) * size;

    // Rounded rectangle SDF
    let q = abs(p) - half_size + radius;
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;

    // Anti-aliased edge
    let alpha = 1.0 - smoothstep(-1.0, 1.0, d);

    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}

// Circle SDF fragment shader
@fragment
fn fs_circle(input: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(input.uv - 0.5) * 2.0;
    let alpha = 1.0 - smoothstep(0.98, 1.0, dist);
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
