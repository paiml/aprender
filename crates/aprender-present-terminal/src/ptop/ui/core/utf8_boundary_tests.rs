//! #4511 defect D1: display truncation byte-sliced system strings (process, container,
//! sensor, cgroup names). A multi-byte char at the cut point panicked the whole TUI
//! ("byte index N is not a char boundary"). Each row below cuts inside a 3-byte char;
//! every one panicked before the `floor_char_boundary` fix.

use crate::ptop::analyzers::{
    Container, ContainerRuntime, ContainerState, ContainerStats, ProcessExtra, SensorReading,
    SensorStatus, SensorType,
};

/// "x" + ten 3-byte chars: byte offsets 1,4,7,.. are boundaries; 2,3,5,6,.. are not.
fn cjk() -> String {
    format!("x{}", "日".repeat(10))
}

#[test]
fn panel_gpu_truncate_name_cuts_on_a_char_boundary() {
    assert_eq!(super::panel_gpu::truncate_name(&cjk(), 6), "x日");
}

#[test]
fn overlays_truncate_name_cuts_on_a_char_boundary() {
    assert_eq!(
        crate::ptop::ui::overlays::truncate_name(&cjk(), 9),
        "x日..."
    );
}

#[test]
fn connections_truncate_process_name_cuts_on_a_char_boundary() {
    assert_eq!(
        crate::ptop::ui::panels::connections::truncate_process_name(&cjk(), 6),
        "x日…"
    );
}

#[test]
fn format_truncate_and_pad_cut_on_a_char_boundary() {
    assert_eq!(super::format::truncate_with_ellipsis(&cjk(), 9), "x日...");
    assert_eq!(super::format::pad_left(&cjk(), 6), "x日");
    assert_eq!(super::format::pad_right(&cjk(), 6), "x日");
}

#[test]
fn sensor_short_label_cuts_on_a_char_boundary() {
    let r = SensorReading {
        device: "d".into(),
        sensor_type: SensorType::Temperature,
        label: format!("x{}", "日".repeat(5)),
        index: 1,
        value: 0.0,
        critical: None,
        max: None,
        min: None,
        status: SensorStatus::Normal,
        hwmon_path: std::path::PathBuf::new(),
    };
    assert_eq!(r.short_label(), "x日日...");
}

#[test]
fn process_extra_badge_and_cgroup_cut_on_a_char_boundary() {
    let p = ProcessExtra {
        container: Some(cjk()),
        cgroup: format!("/sys/{}", cjk()),
        ..Default::default()
    };
    assert_eq!(p.container_badge().as_deref(), Some("[x日日日…]"));
    assert_eq!(p.cgroup_short(), format!("x{}...", "日".repeat(8)));
}

#[test]
fn container_display_name_cuts_on_a_char_boundary_and_survives_zero_width() {
    let c = Container {
        id: "i".into(),
        name: cjk(),
        image: "img".into(),
        state: ContainerState::Running,
        status: String::new(),
        runtime: ContainerRuntime::Docker,
        stats: ContainerStats::default(),
        created: 0,
        ports: Vec::new(),
    };
    assert_eq!(c.display_name(6), "x日…");
    assert_eq!(c.display_name(0), "…");
}
