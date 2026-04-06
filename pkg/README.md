# @paiml/trueno-viz

SIMD-accelerated visualization library for the browser, compiled from Rust to WebAssembly.

**Zero JavaScript charting libraries required** - pure Rust rendering to PNG.

## Installation

```bash
npm install @paiml/trueno-viz
```

## Quick Start

```javascript
import init, { scatter_plot, line_chart, histogram, from_prompt, PlotOptions } from '@paiml/trueno-viz';

// Initialize the WASM module
await init();

// Create a scatter plot
const x = new Float32Array([1, 2, 3, 4, 5]);
const y = new Float32Array([2, 4, 3, 5, 4]);

const options = new PlotOptions()
  .width(800)
  .height(600)
  .color('#4285F4')
  .point_size(8);

const pngData = scatter_plot(x, y, options);

// Display in an image element
const blob = new Blob([pngData], { type: 'image/png' });
document.getElementById('chart').src = URL.createObjectURL(blob);
```

## API

### Plot Functions

#### `scatter_plot(x, y, options?)`

Create a scatter plot.

```javascript
const png = scatter_plot(
  new Float32Array([1, 2, 3]),
  new Float32Array([4, 5, 6]),
  new PlotOptions().color('#FF0000')
);
```

#### `line_chart(x, y, options?)`

Create a line chart.

```javascript
const png = line_chart(
  new Float32Array([0, 1, 2, 3]),
  new Float32Array([0, 1, 4, 9])
);
```

#### `histogram(data, bins?, options?)`

Create a histogram.

```javascript
const png = histogram(
  new Float32Array([1, 2, 2, 3, 3, 3, 4, 4, 5]),
  10  // number of bins
);
```

#### `heatmap(data, rows, cols, options?)`

Create a heatmap from row-major flattened data.

```javascript
const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
const png = heatmap(data, 3, 3);  // 3x3 matrix
```

#### `from_prompt(prompt)`

Create any plot from a text prompt (DSL).

```javascript
const png = from_prompt('scatter x=[1,2,3,4,5] y=[2,4,3,5,4] color=blue size=8');
```

Supported prompts:
- `scatter x=[...] y=[...] [color=...] [size=...]`
- `line x=[...] y=[...]`
- `histogram data=[...]`
- `heatmap matrix=[[...],[...]]`
- `boxplot groups=[[...],[...]]`

#### `ggplot(x, y, geom, theme, options?)`

Grammar of Graphics style plotting.

```javascript
const png = ggplot(
  new Float32Array([1, 2, 3, 4, 5]),
  new Float32Array([2, 4, 3, 5, 4]),
  'point',   // geom: 'point', 'line', 'bar', 'area'
  'dark'     // theme: 'grey', 'minimal', 'dark', 'classic', 'bw', 'void'
);
```

### PlotOptions

```javascript
const options = new PlotOptions()
  .width(800)          // Width in pixels
  .height(600)         // Height in pixels
  .color('#4285F4')    // Primary color (hex)
  .background('#FFF')  // Background color (hex)
  .title('My Plot')    // Title
  .point_size(5)       // Point size for scatter plots
  .line_width(2);      // Line width for line charts
```

## Browser Support

Works in all modern browsers with WebAssembly support:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

## Bundle Size

The WASM module is approximately 200KB gzipped, significantly smaller than most JavaScript charting libraries.

## License

MIT OR Apache-2.0
