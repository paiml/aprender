//! #4511 item 6: the collector's per-phase timings are off by default and, when on,
//! name every collector once, in collect order.

use super::app::MetricsCollector;
use crate::AsyncCollector;

#[test]
fn collect_timings_are_empty_when_off() {
    let mut c = MetricsCollector::new(false);
    assert!(c.collect().collect_phase_us.is_empty());
}

#[test]
fn collect_timings_name_every_collector_in_order_when_on() {
    let mut c = MetricsCollector::new(false);
    c.set_timings(true);
    let names: Vec<&str> = c
        .collect()
        .collect_phase_us
        .iter()
        .map(|(p, _)| *p)
        .collect();
    assert_eq!(
        names,
        ["cpu", "mem", "process", "disk", "net", "gpu", "analyzers"]
    );
}
