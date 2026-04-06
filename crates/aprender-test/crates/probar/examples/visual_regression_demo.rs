//! Visual Regression Testing Demo
//!
//! Demonstrates visual regression testing capabilities:
//! - Image comparison with configurable thresholds
//! - Perceptual diffing (human vision weighted)
//! - Mask regions for dynamic content
//! - Baseline management
//!
//! Run with: cargo run --example visual_regression_demo -p jugar-probar

#![allow(clippy::expect_used)]

use image::{ImageEncoder, Rgba, RgbaImage};
use jugar_probar::{
    perceptual_diff, MaskRegion, ScreenshotComparison, VisualRegressionConfig,
    VisualRegressionTester,
};

fn main() {
    println!("=== Visual Regression Testing Demo ===\n");

    // Demo 1: Configuration
    println!("1. Configuration Options");
    println!("   ----------------------");

    let config = VisualRegressionConfig::default();
    println!("   Default threshold: {}%", config.threshold * 100.0);
    println!("   Default color threshold: {}", config.color_threshold);
    println!("   Default baseline dir: {}", config.baseline_dir);
    println!("   Default diff dir: {}", config.diff_dir);

    let custom_config = VisualRegressionConfig::default()
        .with_threshold(0.05)
        .with_color_threshold(20)
        .with_baseline_dir("my_baselines");
    println!("\n   Custom config:");
    println!("     Threshold: {}%", custom_config.threshold * 100.0);
    println!("     Color threshold: {}", custom_config.color_threshold);
    println!("     Baseline dir: {}\n", custom_config.baseline_dir);

    // Demo 2: Identical Image Comparison
    println!("2. Identical Image Comparison");
    println!("   ---------------------------");

    let tester = VisualRegressionTester::default();
    let red_image = create_solid_image(100, 100, Rgba([255, 0, 0, 255]));

    let result = tester
        .compare_images(&red_image, &red_image)
        .expect("Comparison failed");

    println!("   Comparing identical red images (100x100):");
    println!("     Matches: {}", result.matches);
    println!("     Is identical: {}", result.is_identical());
    println!("     Diff pixels: {}", result.diff_pixel_count);
    println!("     Diff percentage: {:.2}%\n", result.diff_percentage);

    // Demo 3: Different Image Comparison
    println!("3. Different Image Comparison");
    println!("   ---------------------------");

    let green_image = create_solid_image(100, 100, Rgba([0, 255, 0, 255]));

    let strict_tester = VisualRegressionTester::new(
        VisualRegressionConfig::default()
            .with_threshold(0.0)
            .with_color_threshold(0),
    );

    let result = strict_tester
        .compare_images(&red_image, &green_image)
        .expect("Comparison failed");

    println!("   Comparing red vs green images:");
    println!("     Matches: {}", result.matches);
    println!(
        "     Diff pixels: {} / {}",
        result.diff_pixel_count, result.total_pixels
    );
    println!("     Diff percentage: {:.2}%", result.diff_percentage);
    println!("     Max color diff: {}", result.max_color_diff);
    println!("     Avg color diff: {:.1}", result.avg_color_diff);
    println!(
        "     Diff image generated: {}\n",
        result.diff_image.is_some()
    );

    // Demo 4: Within Color Threshold
    println!("4. Color Threshold Testing");
    println!("   ------------------------");

    let gray1 = create_solid_image(50, 50, Rgba([100, 100, 100, 255]));
    let gray2 = create_solid_image(50, 50, Rgba([105, 105, 105, 255])); // +5 diff

    // With color threshold of 20, +5 should pass
    let lenient_tester =
        VisualRegressionTester::new(VisualRegressionConfig::default().with_color_threshold(20));

    let result = lenient_tester
        .compare_images(&gray1, &gray2)
        .expect("Comparison failed");

    println!("   Comparing gray(100,100,100) vs gray(105,105,105):");
    println!("     Color threshold: 20");
    println!(
        "     Matches: {} (small diff within threshold)\n",
        result.matches
    );

    // Demo 5: Perceptual Diff
    println!("5. Perceptual Diff (Human Vision Weighted)");
    println!("   ----------------------------------------");

    let white = Rgba([255, 255, 255, 255]);
    let black = Rgba([0, 0, 0, 255]);
    let red = Rgba([255, 0, 0, 255]);
    let green = Rgba([0, 255, 0, 255]);

    println!("   Perceptual differences (using human vision weights):");
    println!("     RGB weights: Red=0.299, Green=0.587, Blue=0.114");
    println!();
    println!("     White vs Black: {:.2}", perceptual_diff(white, black));
    println!("     White vs Red:   {:.2}", perceptual_diff(white, red));
    println!("     White vs Green: {:.2}", perceptual_diff(white, green));
    println!("     Red vs Black:   {:.2}", perceptual_diff(red, black));
    println!("     Green vs Black: {:.2}", perceptual_diff(green, black));
    println!();
    println!("   Note: Green changes are most noticeable to human eyes\n");

    // Demo 6: Mask Regions
    println!("6. Mask Regions for Dynamic Content");
    println!("   ----------------------------------");

    let mask1 = MaskRegion::new(10, 10, 100, 50);
    let mask2 = MaskRegion::new(200, 300, 150, 100);

    println!("   Mask 1: x=10, y=10, width=100, height=50");
    println!("     Contains (50, 30): {}", mask1.contains(50, 30));
    println!("     Contains (150, 30): {}", mask1.contains(150, 30));

    println!("\n   Mask 2: x=200, y=300, width=150, height=100");
    println!("     Contains (250, 350): {}", mask2.contains(250, 350));
    println!("     Contains (0, 0): {}\n", mask2.contains(0, 0));

    // Demo 7: Screenshot Comparison Config
    println!("7. Screenshot Comparison Configuration");
    println!("   ------------------------------------");

    let comparison = ScreenshotComparison::new()
        .with_threshold(0.02)
        .with_max_diff_pixels(100)
        .with_max_diff_pixel_ratio(0.05)
        .with_mask(MaskRegion::new(0, 0, 100, 50)) // Header
        .with_mask(MaskRegion::new(0, 500, 800, 100)); // Footer

    println!("   Comparison config:");
    println!("     Threshold: {}%", comparison.threshold * 100.0);
    println!("     Max diff pixels: {:?}", comparison.max_diff_pixels);
    println!("     Max diff ratio: {:?}", comparison.max_diff_pixel_ratio);
    println!("     Mask regions: {}", comparison.mask_regions.len());
    for (i, mask) in comparison.mask_regions.iter().enumerate() {
        println!(
            "       Mask {}: ({}, {}) {}x{}",
            i + 1,
            mask.x,
            mask.y,
            mask.width,
            mask.height
        );
    }

    // Demo 8: Gradient Image Comparison
    println!("\n8. Partial Difference Detection");
    println!("   ------------------------------");

    let gradient1 = create_gradient_image(100, 100);
    let gradient2 = create_gradient_with_artifact(100, 100);

    let tester = VisualRegressionTester::new(
        VisualRegressionConfig::default()
            .with_threshold(0.10) // 10% tolerance
            .with_color_threshold(0),
    );

    let result = tester
        .compare_images(&gradient1, &gradient2)
        .expect("Comparison failed");

    println!("   Comparing gradient with artifact:");
    println!("     Total pixels: {}", result.total_pixels);
    println!("     Diff pixels: {}", result.diff_pixel_count);
    println!("     Diff percentage: {:.2}%", result.diff_percentage);
    println!(
        "     Within 10% threshold: {}",
        result.within_threshold(0.10)
    );

    // Demo 9: ImageDiffResult Utilities
    println!("\n9. ImageDiffResult Utilities");
    println!("   --------------------------");

    println!("   result.is_identical(): {}", result.is_identical());
    println!(
        "   result.within_threshold(0.05): {}",
        result.within_threshold(0.05)
    );
    println!(
        "   result.within_threshold(0.15): {}",
        result.within_threshold(0.15)
    );

    println!("\n=== Demo Complete ===");
}

/// Create a solid color image and encode as PNG
fn create_solid_image(width: u32, height: u32, color: Rgba<u8>) -> Vec<u8> {
    let mut img = RgbaImage::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = color;
    }
    encode_png(&img, width, height)
}

/// Create a horizontal gradient image
fn create_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            #[allow(clippy::cast_possible_truncation)]
            let gray = ((x as f32 / width as f32) * 255.0) as u8;
            img.put_pixel(x, y, Rgba([gray, gray, gray, 255]));
        }
    }
    encode_png(&img, width, height)
}

/// Create a gradient with a small artifact (red square in corner)
fn create_gradient_with_artifact(width: u32, height: u32) -> Vec<u8> {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            #[allow(clippy::cast_possible_truncation)]
            let gray = ((x as f32 / width as f32) * 255.0) as u8;
            // Add small red artifact in top-left corner
            if x < 10 && y < 10 {
                img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            } else {
                img.put_pixel(x, y, Rgba([gray, gray, gray, 255]));
            }
        }
    }
    encode_png(&img, width, height)
}

/// Encode image to PNG bytes
fn encode_png(img: &RgbaImage, width: u32, height: u32) -> Vec<u8> {
    let mut buffer = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buffer);
    encoder
        .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgba8)
        .expect("PNG encoding failed");
    buffer
}
