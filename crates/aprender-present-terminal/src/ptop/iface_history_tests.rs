//! #4511 defect D2: `App::net_iface_history` gained one entry per interface name ever
//! seen and never dropped one. Container veth*/tun*/cali* names are fresh on every
//! start, so a long-running ttop on a busy host grew without bound. The history must
//! track the interfaces in the latest snapshot — no more.

use crate::ptop::app::{App, MetricsSnapshot, NetworkInfo};
use crate::Snapshot;

fn snap(ifaces: &[&str]) -> MetricsSnapshot {
    let mut s = MetricsSnapshot::empty();
    s.network_info = ifaces
        .iter()
        .map(|n| NetworkInfo {
            name: (*n).to_string(),
            received: 1,
            transmitted: 2,
        })
        .collect();
    s
}

#[test]
fn iface_history_drops_interfaces_that_disappeared() {
    let mut app = App::with_config_lightweight(true, crate::ptop::config::PtopConfig::default());
    app.apply_snapshot(snap(&["eth0", "veth1a"]));
    app.apply_snapshot(snap(&["eth0", "veth2b"]));
    let mut keys: Vec<_> = app.net_iface_history.keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        ["eth0", "veth2b"],
        "a vanished interface kept its history"
    );
}

#[test]
fn iface_history_is_bounded_under_churn() {
    let mut app = App::with_config_lightweight(true, crate::ptop::config::PtopConfig::default());
    for i in 0..500 {
        app.apply_snapshot(snap(&["eth0", &format!("veth{i}")]));
    }
    assert_eq!(app.net_iface_history.len(), 2);
    // a surviving interface keeps its accumulated history
    assert!(app.net_iface_history["eth0"].0.as_slice().len() > 1);
}
