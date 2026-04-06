//! SVG Generation Demo
//!
//! Demonstrates SVG generation with both Material Design 3 and Grid Protocol modes.
//!
//! Run with: cargo run --example svg_generation_demo

use batuta::oracle::svg::shapes::Point;
use batuta::oracle::svg::{
    documentation_diagram, sovereign_stack_diagram, GridProtocol, GridSpan, LayoutTemplate,
    LintConfig, ShapeHeavyRenderer, SvgBuilder, SvgLinter, TextHeavyRenderer, VideoPalette,
    VideoTypography,
};

fn main() {
    println!("=== SVG Generation Demo ===\n");

    // ── Material Design 3 Mode ─────────────────────────────────────────

    // 1. Generate Sovereign Stack architecture diagram (now uses Grid Protocol)
    println!("1. Sovereign Stack Architecture Diagram (Grid Protocol + Template E)");
    let stack_svg = sovereign_stack_diagram();
    println!("   Generated SVG: {} bytes", stack_svg.len());
    println!(
        "   Contains GRID PROTOCOL MANIFEST: {}",
        stack_svg.contains("GRID PROTOCOL MANIFEST")
    );
    println!("   Viewport 1920x1080: {}\n", stack_svg.contains("viewBox=\"0 0 1920 1080\""));

    // 2. Custom shape-heavy diagram
    println!("2. Custom Architecture Diagram (Material Design 3)");
    let custom_svg = ShapeHeavyRenderer::new()
        .title("Data Pipeline Architecture")
        .layer("ingestion", 50.0, 100.0, 800.0, 150.0, "Data Ingestion")
        .horizontal_stack(
            &[("kafka", "Kafka"), ("spark", "Spark"), ("trueno", "Trueno")],
            Point::new(100.0, 130.0),
        )
        .layer("processing", 50.0, 300.0, 800.0, 150.0, "Processing")
        .horizontal_stack(
            &[("aprender", "Aprender"), ("realizar", "Realizar")],
            Point::new(100.0, 330.0),
        )
        .build();
    println!("   Generated SVG: {} bytes\n", custom_svg.len());

    // 3. Text-heavy documentation diagram
    println!("3. Documentation Diagram");
    let doc_svg = documentation_diagram(
        "API Reference",
        &[
            ("Authentication", "All endpoints require Bearer token authentication."),
            ("Rate Limiting", "100 requests per minute per API key."),
            ("Versioning", "API version specified via Accept header."),
        ],
    );
    println!("   Generated SVG: {} bytes\n", doc_svg.len());

    // 4. Dark mode diagram
    println!("4. Dark Mode Diagram");
    let dark_svg = ShapeHeavyRenderer::new()
        .dark_mode()
        .title("Dark Mode Components")
        .component("comp1", 100.0, 200.0, "Component A", "trueno")
        .component("comp2", 300.0, 200.0, "Component B", "aprender")
        .connect(Point::new(260.0, 240.0), Point::new(300.0, 240.0))
        .build();
    println!("   Generated SVG: {} bytes\n", dark_svg.len());

    // 5. Text-heavy with code-like content
    println!("5. Code Documentation Diagram");
    let code_doc = TextHeavyRenderer::new()
        .title("Function Reference")
        .heading("train_model()")
        .paragraph("Trains a machine learning model on the provided dataset.")
        .heading("Parameters")
        .paragraph("data: &Dataset - Training data")
        .paragraph("epochs: usize - Number of training epochs")
        .paragraph("lr: f32 - Learning rate")
        .heading("Returns")
        .paragraph("Result<Model, TrainError>")
        .build();
    println!("   Generated SVG: {} bytes\n", code_doc.len());

    // ── Grid Protocol Mode ─────────────────────────────────────────────

    println!("--- Grid Protocol Mode ---\n");

    // 6. Grid Protocol: cell allocation engine
    println!("6. Grid Protocol Cell Allocation");
    let mut grid = GridProtocol::new();
    grid.allocate("header", GridSpan::new(0, 0, 15, 1)).expect("header");
    grid.allocate("sidebar", GridSpan::new(0, 2, 3, 8)).expect("sidebar");
    grid.allocate("content", GridSpan::new(4, 2, 15, 8)).expect("content");
    println!(
        "   Cells used: {} / 144 ({:.0}%)",
        grid.cells_used(),
        grid.cells_used() as f32 / 144.0 * 100.0
    );
    println!("   Cells free: {}", grid.cells_free());
    println!("   Manifest preview:");
    for line in grid.manifest().lines().take(5) {
        println!("     {}", line);
    }
    println!();

    // 7. Layout Templates A-G
    println!("7. Layout Templates");
    let templates = [
        LayoutTemplate::TitleSlide,
        LayoutTemplate::TwoColumn,
        LayoutTemplate::Dashboard,
        LayoutTemplate::CodeWalkthrough,
        LayoutTemplate::Diagram,
        LayoutTemplate::KeyConcepts,
        LayoutTemplate::ReflectionReadings,
    ];
    for template in &templates {
        let mut gp = GridProtocol::new();
        let allocs = template.apply(&mut gp).expect("template apply");
        let names: Vec<&str> = allocs.iter().map(|(n, _)| *n).collect();
        println!("   {} -> {} regions: {:?}", template, allocs.len(), names);
    }
    println!();

    // 8. ShapeHeavyRenderer with Grid Protocol + Template
    println!("8. Grid Protocol ShapeHeavyRenderer (Template E: Diagram)");
    let grid_svg = ShapeHeavyRenderer::new()
        .template(LayoutTemplate::Diagram)
        .title("Stack Architecture")
        .component("trueno_box", 100.0, 300.0, "Trueno", "trueno")
        .component("aprender_box", 300.0, 300.0, "Aprender", "aprender")
        .build();
    println!("   Generated SVG: {} bytes", grid_svg.len());
    println!("   Has manifest: {}", grid_svg.contains("GRID PROTOCOL MANIFEST"));
    println!("   1920x1080 viewport: {}\n", grid_svg.contains("viewBox=\"0 0 1920 1080\""));

    // 9. TextHeavyRenderer with Grid Protocol
    println!("9. Grid Protocol TextHeavyRenderer (Template B: Two Column)");
    let text_grid_svg = TextHeavyRenderer::new()
        .template(LayoutTemplate::TwoColumn)
        .title("Lecture Notes")
        .heading("Key Concepts")
        .paragraph("Grid Protocol provides provable non-overlap via occupied-set tracking.")
        .build();
    println!("   Generated SVG: {} bytes", text_grid_svg.len());
    println!("   Has manifest: {}\n", text_grid_svg.contains("GRID PROTOCOL MANIFEST"));

    // 10. Video Palette and Typography
    println!("10. Video Palette and Typography");
    let dark_palette = VideoPalette::dark();
    let light_palette = VideoPalette::light();
    println!("   Dark canvas:  {}", dark_palette.canvas);
    println!("   Dark heading: {}", dark_palette.heading);
    println!("   Light canvas: {}", light_palette.canvas);
    println!("   Light heading: {}", light_palette.heading);

    let vt = VideoTypography::dark();
    println!(
        "   Slide title:    {}px {} ({})",
        vt.slide_title.size, vt.slide_title.weight, vt.slide_title.family
    );
    println!(
        "   Section header: {}px {} ({})",
        vt.section_header.size, vt.section_header.weight, vt.section_header.family
    );
    println!("   Body:           {}px {} ({})", vt.body.size, vt.body.weight, vt.body.family);
    println!("   Code:           {}px {} ({})", vt.code.size, vt.code.weight, vt.code.family);
    println!("   Min font size:  {}px\n", VideoTypography::MIN_FONT_SIZE);

    // 11. Video-mode contrast verification
    println!("11. Contrast Verification");
    println!(
        "   heading on canvas:  pass={}",
        VideoPalette::verify_contrast(&dark_palette.heading, &dark_palette.canvas)
    );
    println!(
        "   heading on surface: pass={}",
        VideoPalette::verify_contrast(&dark_palette.heading, &dark_palette.surface)
    );
    println!(
        "   accent_gold on canvas: pass={}",
        VideoPalette::verify_contrast(&dark_palette.accent_gold, &dark_palette.canvas)
    );
    println!(
        "   body on canvas:     pass={}\n",
        VideoPalette::verify_contrast(&dark_palette.body, &dark_palette.canvas)
    );

    // 12. Video-mode lint config
    println!("12. Video-Mode Lint Rules");
    let video_config = LintConfig::video_mode();
    let linter = SvgLinter::with_config(video_config);
    println!(
        "   min_text_size:       {}px",
        linter.lint_text_size(18.0, None).map_or("PASS".to_string(), |v| v.message)
    );
    println!(
        "   17px text:           {}",
        linter.lint_text_size(17.0, None).map_or("PASS".to_string(), |v| v.message)
    );
    println!(
        "   2px stroke:          {}",
        linter.lint_stroke_width(2.0, None).map_or("PASS".to_string(), |v| v.message)
    );
    println!(
        "   1px stroke:          {}",
        linter.lint_stroke_width(1.0, None).map_or("PASS".to_string(), |v| v.message)
    );
    println!(
        "   20px padding:        {}",
        linter.lint_internal_padding(20.0, None).map_or("PASS".to_string(), |v| v.message)
    );
    println!(
        "   15px padding:        {}",
        linter.lint_internal_padding(15.0, None).map_or("PASS".to_string(), |v| v.message)
    );

    // 13. SvgBuilder grid mode with video styles
    println!("\n13. SvgBuilder Grid Protocol Mode");
    let mut builder = SvgBuilder::new().grid_protocol().video_styles();
    builder.allocate("header", GridSpan::new(0, 0, 15, 1)).expect("header");
    builder.allocate("body", GridSpan::new(0, 2, 15, 8)).expect("body");
    let builder_svg = builder.build();
    println!("   Generated SVG: {} bytes", builder_svg.len());
    println!("   Has manifest: {}", builder_svg.contains("GRID PROTOCOL MANIFEST"));
    println!("   Has video CSS: {}", builder_svg.contains("Segoe UI"));

    println!("\n=== Demo Complete ===");
    println!("Modes: Material Design 3 (8px grid) + Grid Protocol (120px cells, 1080p)");
}
