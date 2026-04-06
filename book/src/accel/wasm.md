# WebAssembly

Trueno-viz compiles to WebAssembly for browser-based visualization with
near-native performance.

## Building for WASM

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["wasm"] }
```

Build:

```bash
wasm-pack build --target web
```

## JavaScript Integration

```javascript
import init, { create_scatter_plot, render_to_canvas } from './trueno_viz.js';

async function main() {
    await init();

    const x = new Float32Array([1, 2, 3, 4, 5]);
    const y = new Float32Array([2, 4, 1, 5, 3]);

    const plot = create_scatter_plot(x, y);
    render_to_canvas(plot, 'myCanvas', 800, 600);
}

main();
```

## Canvas Rendering

```html
<canvas id="myCanvas" width="800" height="600"></canvas>
<script type="module">
    import init, { ScatterPlot } from './trueno_viz.js';

    async function render() {
        await init();

        const plot = new ScatterPlot()
            .x([1, 2, 3, 4, 5])
            .y([2, 4, 1, 5, 3])
            .color('#4285f4')
            .title('My Plot');

        plot.render_to_canvas('myCanvas');
    }

    render();
</script>
```

## ImageData Output

For manual canvas manipulation:

```javascript
const imageData = plot.render_to_image_data(800, 600);
ctx.putImageData(imageData, 0, 0);
```

## SVG Output in Browser

```javascript
const svgString = plot.render_to_svg(800, 600);
document.getElementById('container').innerHTML = svgString;
```

## SIMD in WASM

WebAssembly SIMD is auto-detected:

```rust
#[cfg(target_arch = "wasm32")]
{
    use trueno_viz::accel;

    if accel::wasm_simd_available() {
        println!("WASM SIMD128 enabled");
    }
}
```

## Bundle Size

Typical bundle sizes:

| Configuration | Size (gzipped) |
|---------------|----------------|
| Minimal | ~150 KB |
| With all plots | ~300 KB |
| With GPU (WebGPU) | ~500 KB |

## Performance

WASM performance is typically 60-80% of native:

```text
Scatter plot (10k points):
  Native:  12ms
  WASM:    18ms

Heatmap (500x500):
  Native:  45ms
  WASM:    65ms
```

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Trueno-Viz WASM Demo</title>
</head>
<body>
    <canvas id="plot" width="800" height="600"></canvas>
    <script type="module">
        import init, { LineChart } from './trueno_viz.js';

        async function main() {
            await init();

            // Generate sine wave
            const x = Array.from({length: 100}, (_, i) => i * 0.1);
            const y = x.map(v => Math.sin(v));

            const chart = new LineChart()
                .x(new Float32Array(x))
                .y(new Float32Array(y))
                .color('#4285f4')
                .title('Sine Wave')
                .build();

            chart.render_to_canvas('plot');
        }

        main();
    </script>
</body>
</html>
```

## Next Chapter

Continue to [Aprender Integration](../integration/aprender.md) for ML pipeline visualization.
