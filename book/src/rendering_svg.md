# SVG Rendering & Keyframes

Simular can export simulation output as SVG files and keyframe animation data, enabling integration with video production pipelines like rmedia and resolve-pipeline.

## Architecture

The rendering pipeline follows a three-stage command pattern:

```text
Physics Engine ──► RenderCommand[] ──► SVG Renderer ──► SVG + Keyframes
                                       │
                                       ├─ SvgRenderer  (template SVG)
                                       └─ KeyframeRecorder (per-frame JSON)
```

All renderers (TUI, WASM, SVG) consume the same `RenderCommand` enum, ensuring visual parity across backends.

## CLI Usage

### SVG Keyframes (Template + Animation Data)

```bash
simular render --domain orbit --format svg-keyframes \
  --output /tmp/orbit --fps 60 --duration 10 --seed 42
```

Produces:
- `template.svg` — Static SVG with all elements at frame 0 positions
- `keyframes.json` — Per-frame property values for every element

### SVG Frames (One File Per Frame)

```bash
simular render --domain orbit --format svg-frames \
  --output /tmp/frames --fps 30 --duration 5 --seed 42
```

Produces `frame_0000.svg` through `frame_0149.svg` (30 fps x 5 seconds).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | `orbit` | Simulation domain (`orbit`, `monte_carlo`, `optimization`) |
| `--format` | `svg-keyframes` | Output format (`svg-keyframes` or `svg-frames`) |
| `--output` | `.` | Output directory |
| `--fps` | `60` | Frames per second |
| `--duration` | `10.0` | Simulation duration in seconds |
| `--seed` | `42` | Random seed for deterministic output |

## SVG Grid Protocol

Generated SVGs follow the SVG Grid Protocol specification:

- **Canvas**: 1920x1080 (16:9)
- **Grid**: 16x9 cells at 120px each
- **Element IDs**: Every `<g>` group has a unique ID (`circle-0`, `path-1`, `text-2`)
- **Manifest**: Comment block documenting canvas parameters

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" width="1920" height="1080">
<!-- GRID PROTOCOL MANIFEST
     Canvas: 1920x1080 | Grid: 16x9 | Cell: 120px
     Renderer: simular SVG
-->
  <g id="circle-0">
    <circle cx="960" cy="540" r="20" fill="#ffd700"/>
  </g>
  <g id="path-0">
    <path d="M960.0,540.0 L980.0,520.0 ..." stroke="#4fc3f7" fill="none"/>
  </g>
</svg>
```

## Keyframes JSON Format

The keyframes file captures every element's properties at every frame:

```json
{
  "fps": 60,
  "duration_frames": 600,
  "seed": 42,
  "domain": "orbit",
  "elements": {
    "circle-0": {
      "cx": [960.0, 960.5, 961.2, ...],
      "cy": [540.0, 539.8, 539.2, ...],
      "r": [20.0, 20.0, 20.0, ...],
      "fill": ["#ffd700", "#ffd700", ...]
    },
    "path-0": {
      "d": ["M960.0,540.0 L...", "M960.5,539.8 L...", ...],
      "stroke": ["#4fc3f7", "#4fc3f7", ...]
    }
  }
}
```

Each element has arrays of values indexed by frame number. Values are either numeric (positions, radii) or strings (colors, SVG path data).

## Programmatic API

### SvgRenderer

```rust,ignore
use simular::renderers::{SvgRenderer, SvgConfig};
use simular::orbit::render::{RenderCommand, Color};

let config = SvgConfig {
    width: 1920,
    height: 1080,
    include_manifest: true,
    ..SvgConfig::default()
};

let mut renderer = SvgRenderer::with_config(config);

// Generate render commands from your simulation
let commands: Vec<RenderCommand> = render_state(&state, &config, &camera, &trails);

// Render to SVG string
let svg: String = renderer.render(&commands);
```

### KeyframeRecorder

```rust,ignore
use simular::renderers::{KeyframeRecorder, KeyframesExport};

let mut recorder = KeyframeRecorder::new(60, 42, "orbit");

// Record commands for each frame
for frame in 0..total_frames {
    let commands = render_state(&state, &config, &camera, &trails);
    recorder.record_frame(&commands);
    integrator.step(&mut state, dt);
}

// Export as JSON
let json: String = recorder.to_json();
std::fs::write("keyframes.json", &json)?;

// Or access structured data
let export: KeyframesExport = recorder.export();
assert_eq!(export.duration_frames, total_frames);
```

## RenderCommand Reference

The SVG renderer handles all 8 render command types:

| Command | SVG Element | Element ID | Properties |
|---------|------------|------------|------------|
| `Clear` | `<rect>` | `bg` | width, height, fill |
| `DrawCircle` | `<circle>` | `circle-N` | cx, cy, r, fill/stroke |
| `DrawLine` | `<line>` | `line-N` | x1, y1, x2, y2, stroke |
| `DrawOrbitPath` | `<path>` | `orbit-path-N` | d, stroke, opacity |
| `DrawText` | `<text>` | `text-N` | x, y, fill, font-size |
| `DrawVelocity` | `<line>` | `velocity-N` | x1, y1, x2, y2, stroke, marker |
| `HighlightBody` | `<circle>` | `highlight-N` | cx, cy, r, stroke |
| `SetCamera` | *(none)* | *(none)* | Updates coordinate transforms |

## Integration with rmedia

The keyframes output is designed for direct consumption by rmedia's SVG producer:

```bash
# 1. Generate keyframes from simulation
simular render --domain orbit --format svg-keyframes --output /tmp/orbit

# 2. Use in MLT XML via rmedia SVG producer
#    template.svg provides the base, keyframes.json drives animation
```

The rmedia SVG producer reads `keyframes.json` and applies per-frame property overrides via `animate.{element-id}.{attribute}` properties, enabling smooth physics-driven animation in video output.

## Determinism

SVG output is fully deterministic: the same `--seed` value produces bit-identical SVG and keyframes across runs. This enables:

- **Reproducible renders**: Same seed = same video frames
- **Diff-based validation**: Compare outputs to verify simulation correctness
- **CI integration**: Automated visual regression testing

## Next Steps

- [Replay System](./engine_replay.md) — Record and replay simulations
- [Deterministic RNG](./engine_rng.md) — Reproducible randomness
- [Physics Simulations](./domain_physics.md) — Simulation domains
