# Direct Terminal Backend

The Direct Terminal Backend is a high-performance, zero-allocation terminal rendering system that implements the `Canvas` trait directly using crossterm, bypassing ratatui for maximum efficiency.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DirectTerminalCanvas                   │
│              (implements Canvas trait)                  │
├─────────────────────────────────────────────────────────┤
│                     CellBuffer                          │
│  ┌─────────────────────┬─────────────────────────────┐  │
│  │     Vec<Cell>       │       BitVec (dirty)        │  │
│  │   CompactString     │      1 bit per cell         │  │
│  └─────────────────────┴─────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    DiffRenderer                         │
│  - Style caching       - Cursor optimization           │
│  - Buffered I/O        - Skip continuation cells       │
├─────────────────────────────────────────────────────────┤
│                      crossterm                          │
│              (direct terminal control)                  │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### Zero-Allocation Steady State

The system uses `CompactString` for cell symbols, which inlines strings up to 24 bytes. Since terminal characters are typically 1-4 bytes (UTF-8), allocations only happen during buffer creation or resize.

### Smart Dirty Tracking

Each cell has a corresponding bit in a `BitVec`. Only dirty cells are rendered, minimizing terminal I/O.

### Differential Rendering

The `DiffRenderer` tracks:
- Current cursor position (avoids redundant moves)
- Current style state (avoids redundant color/attribute changes)
- Continuation cells (wide characters span multiple cells)

## Usage

### Basic Example

```rust
use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};

fn main() {
    // Create a buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    // Draw using the Canvas trait
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Fill background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.1, 0.1, 0.2, 1.0)
        );

        // Draw text
        let style = TextStyle::default();
        canvas.draw_text("Hello, Terminal!", Point::new(10.0, 5.0), &style);
    }

    // Render to stdout
    let mut output = std::io::stdout();
    renderer.flush(&mut buffer, &mut output).unwrap();
}
```

### With Color Modes

```rust
use presentar_terminal::ColorMode;
use presentar_terminal::direct::DiffRenderer;

// Auto-detect color support
let renderer = DiffRenderer::new();

// Force specific color mode
let renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);
let renderer = DiffRenderer::with_color_mode(ColorMode::Color256);
let renderer = DiffRenderer::with_color_mode(ColorMode::Color16);
```

### Direct Cell Updates

```rust
use presentar_core::Color;
use presentar_terminal::direct::{CellBuffer, Modifiers};

let mut buffer = CellBuffer::new(80, 24);

// Update individual cells
buffer.update(
    10, 5,                    // x, y
    "X",                      // symbol
    Color::RED,               // foreground
    Color::BLACK,             // background
    Modifiers::BOLD           // text modifiers
);

// Fill a rectangle
buffer.fill_rect(
    0, 0, 10, 5,              // x, y, width, height
    " ",                      // symbol
    Color::WHITE,
    Color::BLUE,
    Modifiers::NONE
);
```

## Performance

Benchmarked on a modern system:

| Operation | Time | Target |
|-----------|------|--------|
| Full 80x24 redraw | ~0.03ms | <1ms |
| 10% differential update | ~6μs | <100μs |
| Cell updates | 87-123M/sec | - |

### Cursor Optimization

The renderer minimizes cursor movements:

```
Scattered (10 cells): 10 cursor moves
Consecutive (10 cells): 1 cursor move
```

## Modifiers

Available text modifiers:

```rust
use presentar_terminal::direct::Modifiers;

let mods = Modifiers::BOLD | Modifiers::ITALIC;

// All available modifiers:
// - Modifiers::NONE
// - Modifiers::BOLD
// - Modifiers::DIM
// - Modifiers::ITALIC
// - Modifiers::UNDERLINE
// - Modifiers::BLINK
// - Modifiers::REVERSE
// - Modifiers::HIDDEN
// - Modifiers::STRIKETHROUGH
```

## Wide Character Support

The system correctly handles wide characters (CJK, emoji):

```rust
// Wide characters occupy 2 cells
buffer.update(0, 0, "日", Color::WHITE, Color::BLACK, Modifiers::NONE);
// Cell 1 becomes a "continuation" cell
```

## Running the Examples

```bash
# Interactive demo
cargo run -p presentar-terminal --example direct_canvas_demo

# Performance benchmark (use release mode)
cargo run -p presentar-terminal --example direct_benchmark --release
```

## API Reference

### `CellBuffer`

```rust
impl CellBuffer {
    fn new(width: u16, height: u16) -> Self;
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn len(&self) -> usize;
    fn get(&self, x: u16, y: u16) -> Option<&Cell>;
    fn get_mut(&mut self, x: u16, y: u16) -> Option<&mut Cell>;
    fn update(&mut self, x: u16, y: u16, symbol: &str, fg: Color, bg: Color, mods: Modifiers);
    fn fill_rect(&mut self, x: u16, y: u16, w: u16, h: u16, symbol: &str, fg: Color, bg: Color, mods: Modifiers);
    fn resize(&mut self, width: u16, height: u16);
    fn mark_all_dirty(&mut self);
    fn clear_dirty(&mut self);
    fn dirty_count(&self) -> usize;
}
```

### `DiffRenderer`

```rust
impl DiffRenderer {
    fn new() -> Self;
    fn with_color_mode(mode: ColorMode) -> Self;
    fn flush<W: Write>(&mut self, buffer: &mut CellBuffer, writer: &mut W) -> io::Result<usize>;
    fn render_full<W: Write>(&mut self, buffer: &mut CellBuffer, writer: &mut W) -> io::Result<usize>;
    fn reset(&mut self);
    fn cells_written(&self) -> usize;
    fn cursor_moves(&self) -> usize;
    fn style_changes(&self) -> usize;
}
```

### `DirectTerminalCanvas`

Implements the `presentar_core::Canvas` trait for drawing operations:

```rust
impl<'a> Canvas for DirectTerminalCanvas<'a> {
    fn fill_rect(&mut self, rect: Rect, color: Color);
    fn stroke_rect(&mut self, rect: Rect, color: Color, width: f32);
    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle);
    fn fill_circle(&mut self, center: Point, radius: f32, color: Color);
    fn draw_line(&mut self, start: Point, end: Point, color: Color, width: f32);
    fn push_transform(&mut self, transform: Transform2D);
    fn pop_transform(&mut self);
    fn push_clip(&mut self, rect: Rect);
    fn pop_clip(&mut self);
    // ... and more
}
```

## Specification

This implementation follows `PROBAR-SPEC-009: Bug Hunting Probador - Brick Architecture`. See `docs/specifications/simplified-tui-spec.md` for the complete technical specification with falsification tests.
