# Image Processing

The `trueno-image` crate provides NPP-parity image processing operations with provable contracts.

## Convolution & Edge Detection

```rust
use trueno_image::{conv2d, gaussian_blur, sobel, canny, BorderMode};

// Gaussian blur with σ=1.5
let blurred = gaussian_blur(&image, width, height, 1.5)?;

// Sobel edge detection
let (gx, gy) = sobel(&image, width, height)?;

// Full Canny pipeline (blur → gradient → NMS → hysteresis)
let edges = canny(&image, width, height, 1.0, 0.05, 0.15)?;
```

## Histogram & Equalization

```rust
use trueno_image::{histogram, cumulative_histogram, equalize};

let hist = histogram(&image, width, height, 256)?;
let cumhist = cumulative_histogram(&hist);
let equalized = equalize(&image, width, height, 256)?;
```

## Morphological Operations

Dilate, erode, opening, and closing with flat structuring elements:

```rust
use trueno_image::{dilate, erode, opening, closing};

let se = vec![1.0f32; 9]; // 3×3 box
let dilated = dilate(&image, w, h, &se, 3, 3)?;
let eroded = erode(&image, w, h, &se, 3, 3)?;
let opened = opening(&image, w, h, &se, 3, 3)?;  // erode then dilate
let closed = closing(&image, w, h, &se, 3, 3)?;  // dilate then erode
```

## Resize

Four interpolation methods: Nearest, Bilinear, Bicubic (Keys' convolution), and Lanczos (a=3):

```rust
use trueno_image::{resize, Interpolation};

let small = resize(&image, 256, 256, 64, 64, Interpolation::Bilinear)?;
let big = resize(&image, 64, 64, 256, 256, Interpolation::Nearest)?;
let sharp = resize(&image, 256, 256, 128, 128, Interpolation::Bicubic)?;
let best = resize(&image, 256, 256, 128, 128, Interpolation::Lanczos)?;
```

## Color Conversion

RGB↔Grayscale and RGB↔HSV using ITU-R BT.601 weights:

```rust
use trueno_image::{rgb_to_gray, rgb_to_hsv, hsv_to_rgb};

let gray = rgb_to_gray(&rgb, width, height)?;
let hsv = rgb_to_hsv(&rgb, width, height)?;
let rgb_back = hsv_to_rgb(&hsv, width, height)?;
```

## Connected Component Labeling

Union-find based 4-connectivity labeling:

```rust
use trueno_image::connected_components;

let labels = connected_components(&binary_image, width, height)?;
// labels[i] = 0 for background, 1..N for component IDs
```

## Provable Contract

All operations are backed by `image-conv2d-v1.yaml`, `image-histogram-v1.yaml`, and `image-color-v1.yaml` contracts with falsification tests for:
- Identity kernel preservation
- Parseval energy conservation for convolution
- HSV roundtrip accuracy
- Connected component correctness (4-connectivity)
