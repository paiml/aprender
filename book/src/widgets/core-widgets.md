# Core Widgets

Presentar provides a complete widget library for building UIs.

## Layout Widgets

| Widget | Purpose |
|--------|---------|
| [Container](./container.md) | Single-child wrapper with padding/decoration |
| [Row](./row.md) | Horizontal layout |
| [Column](./column.md) | Vertical layout |
| [Stack](./stack.md) | Overlay layout |

## Input Widgets

| Widget | Purpose |
|--------|---------|
| [Button](./button.md) | Clickable button |
| [TextInput](./text-input.md) | Text entry field |
| [Checkbox](./checkbox.md) | Boolean toggle |
| [Slider](./slider.md) | Numeric range selector |
| [Select](./select.md) | Dropdown selection |

## Display Widgets

| Widget | Purpose |
|--------|---------|
| [Text](./text.md) | Static text display |
| [Image](./image.md) | Image display |

## Data Widgets

| Widget | Purpose |
|--------|---------|
| [DataTable](./data-table.md) | Tabular data |
| [Chart](./chart.md) | Data visualization |
| [DataCard](./data-card.md) | Metric display |
| [ModelCard](./model-card.md) | ML model info |

## Widget Hierarchy

```
Widget (trait)
├── Layout
│   ├── Container
│   ├── Row
│   ├── Column
│   └── Stack
├── Input
│   ├── Button
│   ├── TextInput
│   ├── Checkbox
│   ├── Slider
│   └── Select
└── Display
    ├── Text
    └── Image
```

## Import

```rust
use presentar_widgets::{
    Button, Text, Column, Row, Container, Stack,
    TextInput, Checkbox, Slider, Select,
};
```
