//! Default position-scale tick labels follow R's fixed-versus-scientific choice outside
//! |v| in [2^-8, 1e4] (paiml/aprender#6958). Expected strings are ggplot2 4.0.3 / R 4.6.2's, as
//! measured in the issue.

use trueno_viz::grammar::{GGPlot, Geom};

fn y_labels(y: [f32; 4]) -> Vec<String> {
    GGPlot::new()
        .data_xy(&[1.0, 2.0, 3.0, 4.0], &y)
        .geom(Geom::line())
        .build()
        .expect("builds")
        .layout()
        .expect("layout")
        .y
        .labels
}

#[test]
fn line_plot_labels_switch_to_scientific_like_r() {
    assert_eq!(y_labels([1e5, 2e5, 3e5, 4e5]), ["1e+05", "2e+05", "3e+05", "4e+05"]);
    assert_eq!(y_labels([1e-4, 2e-4, 3e-4, 4e-4]), ["1e-04", "2e-04", "3e-04", "4e-04"]);
    assert_eq!(
        y_labels([1e-5, 1.1e-5, 1.2e-5, 1.3e-5]),
        ["1.0e-05", "1.1e-05", "1.2e-05", "1.3e-05"]
    );
}
