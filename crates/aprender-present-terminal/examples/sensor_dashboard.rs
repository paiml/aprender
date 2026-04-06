#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! `IoT` Sensor Dashboard Example
//!
//! Demonstrates real-time sensor data visualization for `IoT`
//! monitoring applications with temperature, humidity, and pressure.
//!
//! Run with: cargo run -p presentar-terminal --example `sensor_dashboard`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== IoT Sensor Dashboard ===\n");

    // Simulate sensor data
    let temp_history = simulate_temperature(60);
    let humidity_history = simulate_humidity(60);
    let pressure_history = simulate_pressure(60);
    let co2_history = simulate_co2(60);

    let sensors = vec![
        Sensor::new("sensor-001", "Lab A", 23.5, 45.2, 1013.2, 420),
        Sensor::new("sensor-002", "Lab B", 24.1, 42.8, 1012.8, 385),
        Sensor::new("sensor-003", "Server Room", 18.2, 35.5, 1014.1, 350),
        Sensor::new("sensor-004", "Office", 22.8, 48.5, 1013.5, 580),
        Sensor::new("sensor-005", "Warehouse", 15.3, 55.2, 1012.2, 320),
    ];

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.03, 0.05, 0.08, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.3, 0.8, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "IoT Sensor Network Monitor",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Temperature graph
        draw_sensor_graph(
            &mut canvas,
            "Temperature (°C)",
            &temp_history,
            Rect::new(2.0, 3.0, 36.0, 5.0),
            Color::new(1.0, 0.5, 0.3, 1.0),
            15.0,
            30.0,
        );

        // Humidity graph
        draw_sensor_graph(
            &mut canvas,
            "Humidity (%)",
            &humidity_history,
            Rect::new(42.0, 3.0, 36.0, 5.0),
            Color::new(0.3, 0.7, 1.0, 1.0),
            30.0,
            70.0,
        );

        // Pressure graph
        draw_sensor_graph(
            &mut canvas,
            "Pressure (hPa)",
            &pressure_history,
            Rect::new(2.0, 9.0, 36.0, 4.0),
            Color::new(0.8, 0.6, 1.0, 1.0),
            1010.0,
            1020.0,
        );

        // CO2 graph
        draw_sensor_graph(
            &mut canvas,
            "CO2 (ppm)",
            &co2_history,
            Rect::new(42.0, 9.0, 36.0, 4.0),
            Color::new(0.9, 0.8, 0.3, 1.0),
            300.0,
            800.0,
        );

        // Sensor table
        draw_sensor_table(&mut canvas, &sensors, 2.0, 14.0);

        // Alerts
        draw_alerts(&mut canvas, &sensors, 2.0, 21.0);
    }

    // Render
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    println!("Buffer: {}x{}", buffer.width(), buffer.height());
    println!("Cells written: {cells_written}");
    println!("Output bytes: {}\n", output.len());

    println!("Rendered output:");
    println!("{}", "─".repeat(82));
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!();
    println!("{}", "─".repeat(82));
}

struct Sensor {
    id: String,
    location: String,
    temperature: f64,
    humidity: f64,
    pressure: f64,
    co2: u32,
}

impl Sensor {
    fn new(id: &str, location: &str, temp: f64, humidity: f64, pressure: f64, co2: u32) -> Self {
        Self {
            id: id.to_string(),
            location: location.to_string(),
            temperature: temp,
            humidity,
            pressure,
            co2,
        }
    }

    fn temp_status(&self) -> (Color, &'static str) {
        if self.temperature > 28.0 || self.temperature < 16.0 {
            (Color::new(1.0, 0.3, 0.3, 1.0), "WARN")
        } else {
            (Color::new(0.3, 1.0, 0.5, 1.0), " OK ")
        }
    }

    fn co2_status(&self) -> (Color, &'static str) {
        if self.co2 > 1000 {
            (Color::new(1.0, 0.3, 0.3, 1.0), "HIGH")
        } else if self.co2 > 600 {
            (Color::new(1.0, 0.7, 0.2, 1.0), "ELEV")
        } else {
            (Color::new(0.3, 1.0, 0.5, 1.0), " OK ")
        }
    }
}

fn draw_sensor_graph(
    canvas: &mut DirectTerminalCanvas<'_>,
    title: &str,
    history: &[f64],
    bounds: Rect,
    color: Color,
    min: f64,
    max: f64,
) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(title, Point::new(bounds.x, bounds.y), &label_style);

    let current = history.last().copied().unwrap_or(0.0);
    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.1}"),
        Point::new(bounds.x + bounds.width - 8.0, bounds.y),
        &value_style,
    );

    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(min, max)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_sensor_table(canvas: &mut DirectTerminalCanvas<'_>, sensors: &[Sensor], x: f32, y: f32) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Sensor ID     Location       Temp(°C)  Humidity  Pressure   CO2(ppm)  Status",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, sensor) in sensors.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;

        let name_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<12}", sensor.id),
            Point::new(x, row_y),
            &name_style,
        );
        canvas.draw_text(
            &format!("{:<12}", sensor.location),
            Point::new(x + 14.0, row_y),
            &name_style,
        );

        // Temperature with color
        let (temp_color, _) = sensor.temp_status();
        let temp_style = TextStyle {
            color: temp_color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>7.1}", sensor.temperature),
            Point::new(x + 27.0, row_y),
            &temp_style,
        );

        // Humidity
        let hum_style = TextStyle {
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>7.1}%", sensor.humidity),
            Point::new(x + 36.0, row_y),
            &hum_style,
        );

        // Pressure
        let press_style = TextStyle {
            color: Color::new(0.8, 0.6, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>8.1}", sensor.pressure),
            Point::new(x + 46.0, row_y),
            &press_style,
        );

        // CO2 with status color
        let (co2_color, co2_status) = sensor.co2_status();
        let co2_style = TextStyle {
            color: co2_color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>8}", sensor.co2),
            Point::new(x + 56.0, row_y),
            &co2_style,
        );

        // Overall status
        let status_style = TextStyle {
            color: co2_color,
            ..Default::default()
        };
        canvas.draw_text(co2_status, Point::new(x + 68.0, row_y), &status_style);
    }
}

fn draw_alerts(canvas: &mut DirectTerminalCanvas<'_>, sensors: &[Sensor], x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    // Check for any alerts
    let mut alerts = Vec::new();
    for sensor in sensors {
        if sensor.temperature > 28.0 {
            alerts.push(format!(
                "{}: High temperature ({:.1}°C)",
                sensor.id, sensor.temperature
            ));
        }
        if sensor.co2 > 600 {
            alerts.push(format!("{}: Elevated CO2 ({} ppm)", sensor.id, sensor.co2));
        }
    }

    if alerts.is_empty() {
        canvas.draw_text(
            "Alerts: None - All sensors nominal",
            Point::new(x, y),
            &label_style,
        );
    } else {
        let alert_style = TextStyle {
            color: Color::new(1.0, 0.7, 0.2, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("Alerts ({}):", alerts.len()),
            Point::new(x, y),
            &alert_style,
        );
        if let Some(alert) = alerts.first() {
            canvas.draw_text(alert, Point::new(x + 12.0, y), &alert_style);
        }
    }

    canvas.draw_text(
        "[q] quit  [r] refresh  [a] alerts  [c] configure  [h] help",
        Point::new(x, y + 1.0),
        &label_style,
    );
}

fn simulate_temperature(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 3.0f64.mul_add((i as f64 / 20.0).sin(), 22.0);
            let noise = ((i * 7919) % 20) as f64 / 20.0;
            base + noise
        })
        .collect()
}

fn simulate_humidity(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 10.0f64.mul_add((i as f64 / 15.0).cos(), 45.0);
            let noise = ((i * 6971) % 30) as f64 / 10.0;
            (base + noise).clamp(30.0, 70.0)
        })
        .collect()
}

fn simulate_pressure(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 2.0f64.mul_add((i as f64 / 30.0).sin(), 1013.25);
            let noise = ((i * 1103) % 10) as f64 / 10.0;
            base + noise
        })
        .collect()
}

fn simulate_co2(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 100.0f64.mul_add((i as f64 / 12.0).sin(), 420.0);
            let spike = if i % 15 == 0 { 80.0 } else { 0.0 };
            let noise = ((i * 7717) % 50) as f64;
            (base + spike + noise).max(300.0)
        })
        .collect()
}
