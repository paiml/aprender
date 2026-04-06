#![allow(clippy::too_many_lines, clippy::disallowed_methods)]
//! Image processing demonstration — full feature showcase.
//!
//! ```sh
//! cargo run --example image_demo -p trueno-image
//! ```

use trueno_image::{
    canny, canny_rgb, connected_components, conv2d, dilate, equalize, erode, gaussian_blur,
    gradient_magnitude, histogram, hsv_to_rgb, resize, rgb_to_gray, rgb_to_hsv, sobel, BorderMode,
    ImageBuf, ImageOps, Interpolation,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== trueno-image: Full Demo ===\n");

    let w = 16;
    let h = 16;

    // Create test image: left=0, right=1 (sharp vertical edge)
    let mut image = vec![0.0f32; w * h];
    for y in 0..h {
        for x in w / 2..w {
            image[y * w + x] = 1.0;
        }
    }

    // ── Convolution ────────────────────────────────────────
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let id_out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Zero)?;
    let conv_err: f32 =
        image.iter().zip(id_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("Identity conv max error: {conv_err:.2e}");

    // ── Gaussian blur ──────────────────────────────────────
    let blurred = gaussian_blur(&image, w, h, 1.5)?;
    let (bmin, bmax) = min_max(&blurred);
    println!("Gaussian blur (σ=1.5): range [{bmin:.3}, {bmax:.3}]");

    // ── Sobel + Canny ──────────────────────────────────────
    let (gx, gy) = sobel(&image, w, h)?;
    let mag = gradient_magnitude(&gx, &gy);
    let edge_count = mag.iter().filter(|&&m| m > 0.5).count();
    println!("Sobel: {edge_count} edge pixels");

    let edges = canny(&image, w, h, 1.0, 0.05, 0.15)?;
    let canny_count = edges.iter().filter(|&&v| v > 0.5).count();
    println!("Canny: {canny_count} edge pixels");

    // ── Histogram + equalization ───────────────────────────
    println!("\n--- Histogram ---");
    let test_img = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.0, 0.75_f32];
    let hist = histogram(&test_img, 3, 3, 4)?;
    println!("Histogram (4 bins): {hist:?}");
    let eq = equalize(&test_img, 3, 3, 256)?;
    let (eq_min, eq_max) = min_max(&eq);
    println!("Equalized range: [{eq_min:.3}, {eq_max:.3}]");

    // ── Morphology ─────────────────────────────────────────
    println!("\n--- Morphology ---");
    let se = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32]; // 3×3 box
    let dilated = dilate(&image, w, h, &se, 3, 3)?;
    let eroded = erode(&image, w, h, &se, 3, 3)?;
    let dil_ones = dilated.iter().filter(|&&v| v > 0.5).count();
    let ero_ones = eroded.iter().filter(|&&v| v > 0.5).count();
    let orig_ones = image.iter().filter(|&&v| v > 0.5).count();
    println!("Original ones: {orig_ones}, dilated: {dil_ones}, eroded: {ero_ones}");

    // ── Resize ─────────────────────────────────────────────
    println!("\n--- Resize ---");
    let small = resize(&image, w, h, 8, 8, Interpolation::Bilinear)?;
    println!("Resize {w}×{h} → 8×8 (bilinear): {} pixels", small.len());
    let big = resize(&image, w, h, 32, 32, Interpolation::Nearest)?;
    println!("Resize {w}×{h} → 32×32 (nearest): {} pixels", big.len());
    let bicubic = resize(&image, w, h, 8, 8, Interpolation::Bicubic)?;
    println!("Resize {w}×{h} → 8×8 (bicubic): {} pixels", bicubic.len());
    let lanczos = resize(&image, w, h, 8, 8, Interpolation::Lanczos)?;
    println!("Resize {w}×{h} → 8×8 (lanczos): {} pixels", lanczos.len());

    // ── Color conversion ───────────────────────────────────
    println!("\n--- Color Conversion ---");
    let rgb = vec![
        1.0, 0.0, 0.0, // red
        0.0, 1.0, 0.0, // green
        0.0, 0.0, 1.0, // blue
        1.0, 1.0, 1.0_f32, // white
    ];
    let gray = rgb_to_gray(&rgb, 4, 1)?;
    println!(
        "RGB→Gray: red={:.3}, green={:.3}, blue={:.3}, white={:.3}",
        gray[0], gray[1], gray[2], gray[3]
    );

    let hsv = rgb_to_hsv(&rgb, 4, 1)?;
    let recovered = hsv_to_rgb(&hsv, 4, 1)?;
    let hsv_err: f32 =
        rgb.iter().zip(recovered.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("RGB→HSV→RGB roundtrip error: {hsv_err:.2e}");

    // ── Connected components ───────────────────────────────
    println!("\n--- Connected Components ---");
    #[rustfmt::skip]
    let binary = vec![
        1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0_f32,
    ];
    let labels = connected_components(&binary, 5, 5)?;
    let num_labels = *labels.iter().max().unwrap_or(&0);
    println!("5×5 binary image → {num_labels} connected components");
    for y in 0..5 {
        let row: String = (0..5)
            .map(|x| {
                let l = labels[y * 5 + x];
                if l == 0 {
                    '.'
                } else {
                    (b'A' + (l - 1) as u8) as char
                }
            })
            .collect();
        println!("  {row}");
    }

    // ── Multi-channel Canny (NPP parity) ──────────────────
    println!("\n--- Multi-channel Canny (canny_rgb) ---");
    let mut rgb_img = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in w / 2..w {
            let base = (y * w + x) * 3;
            rgb_img[base] = 1.0;
            rgb_img[base + 1] = 1.0;
            rgb_img[base + 2] = 1.0;
        }
    }
    let rgb_edges = canny_rgb(&rgb_img, w, h, 3, 1.0, 0.05, 0.15)?;
    let rgb_edge_count = rgb_edges.iter().filter(|&&v| v > 0.5).count();
    println!("canny_rgb: {rgb_edge_count} edge pixels from {w}×{h}×3 input");

    // ── ImageOps trait (method dispatch on ImageBuf) ─────
    println!("\n--- ImageOps trait ---");
    let buf = ImageBuf::new(rgb_img, w, h, 3)?;
    let blurred_buf = buf.blur(1.5)?;
    println!(
        "ImageBuf.blur(): {}×{} × {} channels",
        blurred_buf.width(),
        blurred_buf.height(),
        blurred_buf.channels()
    );
    let gray_buf = buf.to_gray()?;
    println!(
        "ImageBuf.to_gray(): {}×{} × {} channel",
        gray_buf.width(),
        gray_buf.height(),
        gray_buf.channels()
    );
    let edge_buf = buf.canny_edges(1.0, 0.05, 0.15)?;
    let trait_edges = edge_buf.data().iter().filter(|&&v| v > 0.5).count();
    println!("ImageBuf.canny_edges(): {trait_edges} edge pixels");

    // ImageOps expanded trait: sobel, dilate, resize, histogram, components
    let (gx_buf, _gy_buf) = buf.sobel_gradients()?;
    println!("ImageBuf.sobel_gradients(): {}×{}", gx_buf.width(), gx_buf.height());

    let se = [1.0f32; 9];
    let dilated = gray_buf.apply_dilate(&se, 3, 3)?;
    println!("ImageBuf.apply_dilate(): {}×{}", dilated.width(), dilated.height());

    let resized = buf.apply_resize(8, 8, Interpolation::Bilinear)?;
    println!(
        "ImageBuf.apply_resize(8×8): {}×{} × {} ch",
        resized.width(),
        resized.height(),
        resized.channels()
    );

    let hist = gray_buf.compute_histogram(10)?;
    let total: u32 = hist.iter().sum();
    println!("ImageBuf.compute_histogram(10): total={total}");

    let hsv_buf = buf.to_hsv()?;
    println!("ImageBuf.to_hsv(): {} channels", hsv_buf.channels());

    println!("\n=== All image demos passed ===");
    Ok(())
}

fn min_max(data: &[f32]) -> (f32, f32) {
    let min = data.iter().copied().fold(f32::MAX, f32::min);
    let max = data.iter().copied().fold(f32::MIN, f32::max);
    (min, max)
}
