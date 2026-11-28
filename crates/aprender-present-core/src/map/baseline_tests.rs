//! Headless 10,530-cell baseline for the map runtime (MAP-5 slice 2).
//!
//! A 117x90 pointy-top grid (10,530 cells) is driven through the camera, cull, level-of-detail,
//! sprite atlas and layer scheduler for one fixed pan/zoom script. Every frame draws each visible
//! cell: the LOD tier and the sprite size bucket come from the camera scale, and one atlas lookup
//! is made per drawn cell. Counts are asserted exactly. Wall-clock time is printed, never asserted.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use super::{
    size_bucket, tier_for_radius, visible_indices, Bounds, Camera2D, EntityId, Event, Layers,
    LensId, LodTier, Outcome, OverlayChange, SpriteAtlas,
};

const COLS: usize = 117;
const ROWS: usize = 90;
const CELLS: usize = COLS * ROWS;
/// `sqrt(3)` in f64, for the pointy-top column pitch.
const SQRT3: f64 = 1.732_050_807_568_877_2;
/// Circumradius of a cell in world units. Its screen radius is `scale * HEX_RADIUS`.
const HEX_RADIUS: f64 = 1.0;
const VIEW: (f64, f64) = (1280.0, 720.0);
/// Cull margin in screen pixels, grown on every side of the viewport.
const CULL_MARGIN_PX: f64 = 32.0;
const SPRITE_KEY: &str = "hex-cell";
const SPRITE_SEED: u64 = 7;
const LENS_A: LensId = 1;
const LENS_B: LensId = 2;
const ENTITY: EntityId = 42;
/// Screen centre of `VIEW`: the anchor of every zoom in the script.
const ANCHOR: (f64, f64) = (640.0, 360.0);
/// Pan step in screen pixels. At scale 2 it is 10 world units.
const PAN_STEP_PX: f64 = 20.0;

const fn pan() -> Event {
    Event::Pan {
        dx_px: PAN_STEP_PX,
        dy_px: 0.0,
    }
}

const fn zoom(scale: f64) -> Event {
    Event::ZoomMotion {
        anchor: ANCHOR,
        scale,
    }
}

/// The fixed script, applied in order after the initial paint at scale 2.
const SCRIPT: [(&str, Event); 27] = [
    ("pan_1", pan()),
    ("pan_2", pan()),
    ("pan_3", pan()),
    ("pan_4", pan()),
    ("pan_5", pan()),
    ("pan_6", pan()),
    ("pan_7", pan()),
    ("pan_8", pan()),
    ("pan_9", pan()),
    ("zoom_4", zoom(4.0)),
    ("zoom_8", zoom(8.0)),
    ("settle_8", Event::ZoomSettle),
    ("pan_at_8", pan()),
    (
        "prepaint_16",
        Event::IdlePrepaint {
            anchor: ANCHOR,
            next_scale: 16.0,
        },
    ),
    ("zoom_16", zoom(16.0)),
    ("settle_16", Event::ZoomSettle),
    ("zoom_40", zoom(40.0)),
    ("settle_40", Event::ZoomSettle),
    ("zoom_80", zoom(80.0)),
    ("settle_80", Event::ZoomSettle),
    ("hover", Event::Overlay(OverlayChange::Hover(Some(ENTITY)))),
    (
        "select",
        Event::Overlay(OverlayChange::Select(Some(ENTITY))),
    ),
    ("toggle", Event::Overlay(OverlayChange::LayerToggle)),
    ("delta_inactive", Event::LensDelta(LENS_B)),
    ("lens_switch_b", Event::LensSwitch(LENS_B)),
    ("delta_active", Event::LensDelta(LENS_B)),
    ("idle", Event::Idle),
];

/// Centre of the cell at `(col, row)`. Odd rows are shifted half a column to the right.
fn cell_centre(col: usize, row: usize) -> (f64, f64) {
    let offset = if row % 2 == 1 { 0.5 } else { 0.0 };
    (
        SQRT3 * HEX_RADIUS * (col as f64 + offset),
        1.5 * HEX_RADIUS * row as f64,
    )
}

/// The 10,530 cell boxes, row-major.
fn grid() -> Vec<Bounds> {
    let (hw, hh) = (SQRT3 * HEX_RADIUS / 2.0, HEX_RADIUS);
    (0..ROWS)
        .flat_map(|row| (0..COLS).map(move |col| (col, row)))
        .map(|(col, row)| {
            let (cx, cy) = cell_centre(col, row);
            Bounds::new((cx - hw, cy - hh), (cx + hw, cy + hh)).expect("cell box is ordered")
        })
        .collect()
}

fn grid_centre() -> (f64, f64) {
    cell_centre(COLS / 2, ROWS / 2)
}

/// Reference cull: project each box to screen space and test overlap with the grown viewport.
/// Shares no arithmetic with `visible_indices`.
fn brute_force_visible(cam: &Camera2D, items: &[Bounds], margin: f64) -> Vec<usize> {
    let (vw, vh) = cam.viewport();
    let mut out = Vec::new();
    for (i, b) in items.iter().enumerate() {
        let a = cam.screen_at(b.min());
        let z = cam.screen_at(b.max());
        let (x0, x1) = (a.0.min(z.0), a.0.max(z.0));
        let (y0, y1) = (a.1.min(z.1), a.1.max(z.1));
        if x1 >= -margin && x0 <= vw + margin && y1 >= -margin && y0 <= vh + margin {
            out.push(i);
        }
    }
    out
}

/// The committed expectation for one row: name, (base, lens, overlay) renders, frame request,
/// drawn cells, LOD tier, sprite bucket.
type Expect = (&'static str, (u32, u32, u32), bool, usize, LodTier, u32);

/// One observed frame, after one event.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Row {
    name: &'static str,
    renders: (u32, u32, u32),
    frame_requested: bool,
    visible: usize,
    tier: LodTier,
    bucket: u32,
}

impl Row {
    fn expect(&self) -> Expect {
        (
            self.name,
            self.renders,
            self.frame_requested,
            self.visible,
            self.tier,
            self.bucket,
        )
    }
}

struct Harness {
    layers: Layers,
    items: Vec<Bounds>,
    atlas: SpriteAtlas,
    lookups: usize,
    oracle_mismatches: usize,
    rows: Vec<Row>,
}

impl Harness {
    /// Records the frame after one event: culls, checks the cull against the reference, looks up
    /// one sprite per drawn cell, and stores the row.
    fn observe(&mut self, name: &'static str, out: Outcome) {
        let cam = *self.layers.camera();
        let visible = visible_indices(&cam, &self.items, CULL_MARGIN_PX).expect("margin is valid");
        if visible != brute_force_visible(&cam, &self.items, CULL_MARGIN_PX) {
            self.oracle_mismatches += 1;
        }
        let radius = cam.scale() * HEX_RADIUS;
        let tier = tier_for_radius(radius).expect("radius is positive");
        let bucket = size_bucket(radius).expect("radius is in sprite range");
        for _ in &visible {
            self.atlas
                .sprite(SPRITE_KEY, SPRITE_SEED, bucket)
                .expect("bucket is in range");
            self.lookups += 1;
        }
        self.rows.push(Row {
            name,
            renders: (out.renders.base, out.renders.lens, out.renders.overlay),
            frame_requested: out.frame_requested,
            visible: visible.len(),
            tier,
            bucket,
        });
    }

    fn apply(&mut self, name: &'static str, event: Event) {
        let out = self.layers.handle(event).expect("script event is valid");
        self.observe(name, out);
    }
}

struct Run {
    rows: Vec<Row>,
    oracle_mismatches: usize,
    generations: u64,
    lookups: usize,
    elapsed: Duration,
}

fn run_script() -> Run {
    let camera = Camera2D::new(grid_centre(), 2.0, VIEW).expect("valid camera");
    let (layers, first) = Layers::new(camera, Some(LENS_A));
    let mut h = Harness {
        layers,
        items: grid(),
        atlas: SpriteAtlas::new(),
        lookups: 0,
        oracle_mismatches: 0,
        rows: Vec::new(),
    };
    h.observe("init", first);
    let started = Instant::now();
    for (name, event) in SCRIPT {
        h.apply(name, event);
    }
    let elapsed = started.elapsed();
    Run {
        rows: h.rows,
        oracle_mismatches: h.oracle_mismatches,
        generations: h.atlas.generations(),
        lookups: h.lookups,
        elapsed,
    }
}

/// Committed table: one entry per row, in script order. Filled from the measured run and
/// checked against the brute-force cull and the layer-cache rules by the tests below.
const EXPECTED: &[Expect] = &[
    ("init", (1, 1, 1), true, 10530, LodTier::Far, 4),
    ("pan_1", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_2", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_3", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_4", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_5", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_6", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_7", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_8", (0, 0, 0), true, 10530, LodTier::Far, 4),
    ("pan_9", (1, 1, 1), true, 10530, LodTier::Far, 4),
    ("zoom_4", (0, 0, 0), true, 9405, LodTier::Far, 8),
    ("zoom_8", (0, 0, 0), true, 3752, LodTier::Far, 12),
    ("settle_8", (1, 1, 1), true, 3752, LodTier::Far, 12),
    ("pan_at_8", (0, 0, 0), true, 3652, LodTier::Far, 12),
    ("prepaint_16", (1, 0, 0), false, 3652, LodTier::Far, 12),
    ("zoom_16", (0, 0, 0), true, 1050, LodTier::Mid, 16),
    ("settle_16", (0, 1, 1), true, 1050, LodTier::Mid, 16),
    ("zoom_40", (0, 0, 0), true, 233, LodTier::Near, 21),
    ("settle_40", (1, 1, 1), true, 233, LodTier::Near, 21),
    ("zoom_80", (0, 0, 0), true, 74, LodTier::Close, 25),
    ("settle_80", (1, 1, 1), true, 74, LodTier::Close, 25),
    ("hover", (0, 0, 1), true, 74, LodTier::Close, 25),
    ("select", (0, 0, 1), true, 74, LodTier::Close, 25),
    ("toggle", (0, 0, 1), true, 74, LodTier::Close, 25),
    ("delta_inactive", (0, 0, 0), false, 74, LodTier::Close, 25),
    ("lens_switch_b", (0, 1, 0), true, 74, LodTier::Close, 25),
    ("delta_active", (0, 1, 0), true, 74, LodTier::Close, 25),
    ("idle", (0, 0, 0), false, 74, LodTier::Close, 25),
];

fn renders_of(rows: &[Row], name: &str) -> (u32, u32, u32) {
    rows.iter()
        .find(|r| r.name == name)
        .expect("named row exists")
        .renders
}

#[test]
fn grid_is_the_117_by_90_pointy_top_grid_of_10530_cells() {
    let items = grid();
    assert_eq!(CELLS, 10_530);
    assert_eq!(items.len(), CELLS);
    let (hw, hh) = (SQRT3 * HEX_RADIUS / 2.0, HEX_RADIUS);
    assert_eq!(items[0].min(), (-hw, -hh), "first cell box");
    let last = items.last().expect("grid is not empty");
    assert_eq!(
        last.max(),
        (cell_centre(COLS - 1, ROWS - 1).0 + hw, 1.5 * 89.0 + hh)
    );
}

#[test]
fn baseline_counts_match_the_committed_table_exactly() {
    let run = run_script();
    let got: Vec<Expect> = run.rows.iter().map(Row::expect).collect();
    assert_eq!(got.len(), EXPECTED.len(), "row count");
    for (i, (g, w)) in got.iter().zip(EXPECTED).enumerate() {
        assert_eq!(g, w, "row {i}");
    }
}

#[test]
fn visible_cells_match_the_brute_force_projection_on_every_frame() {
    let run = run_script();
    assert_eq!(
        run.rows.len(),
        SCRIPT.len() + 1,
        "init plus one row per event"
    );
    assert_eq!(
        run.oracle_mismatches, 0,
        "frames where cull differs from brute force"
    );
}

#[test]
fn whole_grid_frame_draws_all_10530_cells_at_the_finest_scale_of_the_script() {
    let run = run_script();
    let init = run.rows.first().expect("init row");
    assert_eq!(init.name, "init");
    assert_eq!(init.visible, CELLS);
    assert_eq!(init.tier, LodTier::Far);
    assert_eq!(init.bucket, 4);
    assert_eq!(init.renders, (1, 1, 1));
}

#[test]
fn pans_inside_the_cache_margin_render_nothing_and_settles_render_each_layer_once() {
    let run = run_script();
    let rows = &run.rows;
    for name in [
        "pan_1", "pan_2", "pan_3", "pan_4", "pan_5", "pan_6", "pan_7", "pan_8",
    ] {
        assert_eq!(
            renders_of(rows, name),
            (0, 0, 0),
            "{name} stays inside the margin"
        );
    }
    // 180 px from the last paint is past the 160 px margin, so every layer repaints once.
    assert_eq!(
        renders_of(rows, "pan_9"),
        (1, 1, 1),
        "pan_9 leaves the margin"
    );
    assert_eq!(
        renders_of(rows, "pan_at_8"),
        (0, 0, 0),
        "pan after settle stays inside"
    );
    for name in ["zoom_4", "zoom_8", "zoom_16", "zoom_40", "zoom_80"] {
        assert_eq!(
            renders_of(rows, name),
            (0, 0, 0),
            "{name} renders nothing while moving"
        );
    }
    assert_eq!(
        renders_of(rows, "settle_8"),
        (1, 1, 1),
        "settle repaints each layer once"
    );
    assert_eq!(
        renders_of(rows, "prepaint_16"),
        (1, 0, 0),
        "prepaint paints base only"
    );
    assert_eq!(
        renders_of(rows, "settle_16"),
        (0, 1, 1),
        "base served from the prefetch slot"
    );
    assert_eq!(
        renders_of(rows, "settle_40"),
        (1, 1, 1),
        "no prefetch, base repaints"
    );
    assert_eq!(
        renders_of(rows, "settle_80"),
        (1, 1, 1),
        "no prefetch, base repaints"
    );
}

#[test]
fn sprite_atlas_paints_one_sprite_per_distinct_bucket_and_looks_up_every_drawn_cell() {
    let run = run_script();
    let buckets: BTreeSet<u32> = run.rows.iter().map(|r| r.bucket).collect();
    let drawn: usize = run.rows.iter().map(|r| r.visible).sum();
    assert_eq!(run.lookups, drawn, "one atlas lookup per drawn cell");
    let distinct = u64::try_from(buckets.len()).expect("small count");
    assert_eq!(run.generations, distinct, "one paint per distinct bucket");
    assert_eq!(buckets, BTreeSet::from([4, 8, 12, 16, 21, 25]));
}

#[test]
fn baseline_run_is_repeatable_and_wall_clock_is_printed_not_asserted() {
    let first = run_script();
    let second = run_script();
    assert_eq!(
        first.rows, second.rows,
        "two runs of the script give the same rows"
    );
    eprintln!(
        "MAP-5 baseline: run 1 script {:?} ({} rows, {} cell lookups); run 2 script {:?}",
        first.elapsed,
        first.rows.len(),
        first.lookups,
        second.elapsed
    );
}
