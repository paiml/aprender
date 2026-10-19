//! Scenario "course-flow": the FLOW-006 course factory and Shorts line (SIM-7, #5511).
//!
//! Stations are multi-server queues with a factory WIP limit over stations 1-6. edX and Shorts
//! launch with no human step; Coursera courses wait for the daily 07:00 UTC submission batch.
//! Shorts take priority over courses on the shared stations (render GPU, Chrome lane). Service
//! times are Erlang-4 (gamma, coefficient of variation 0.5). Time is minutes from 00:00 UTC.

use std::collections::VecDeque;

use crate::engine::rng::SimRng;
use crate::engine::scheduler::EventScheduler;
use crate::engine::state::SimEvent;
use crate::engine::SimTime;

const DAY: f64 = 1440.0;
const OUTLINE: usize = 0;
const BUILD: usize = 1;
const DESIGN: usize = 2;
const RENDER: usize = 3;
const VERIFY: usize = 4;
const CHROME: usize = 5;
const NAMES: [&str; 6] = ["outline", "build", "design", "render", "verify", "chrome"];

/// Inputs of the factory. Times are minutes per course; `*_lanes`/`builders`/`gpus` are servers.
#[derive(Debug, Clone, PartialEq)]
pub struct CourseFlowConfig {
    pub builders: usize,
    pub build_min: f64,
    pub design_lanes: usize,
    pub design_min: f64,
    pub gpus: usize,
    pub render_min: f64,
    pub chrome_lanes: usize,
    pub stage_min: f64,
    pub coursera_share: f64,
    pub wip: usize,
    pub starts_per_day: f64,
    pub shorts_per_day: f64,
    pub outline_min: f64,
    pub verify_min: f64,
    /// Count a finished course as WIP until the daily batch releases it (flow model B6).
    pub batch_in_wip: bool,
}

/// What one seeded run measured after the warm-up.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CourseFlowResult {
    pub courses_per_day: f64,
    pub shorts_per_day: f64,
    pub lead_days: f64,
    pub l: f64,
    pub util: [f64; 6],
    /// Mean hours a built course waits for the daily batch (courses that take the batch path).
    pub mean_batch_wait_h: f64,
}

impl CourseFlowResult {
    /// The busiest station and its utilisation.
    #[must_use]
    pub fn busiest(&self) -> (&'static str, f64) {
        let i = (0..6).fold(0, |b, i| if self.util[i] > self.util[b] { i } else { b });
        (NAMES[i], self.util[i])
    }
}

#[derive(Clone, Copy)]
struct Job {
    short: bool,
    id: usize,
    step: u8,
}

enum Ev {
    Start,
    ShortTick,
    Done(usize, Job),
    Delay(Job),
    Batch,
}

fn pack(e: &Ev) -> SimEvent {
    let (name, st, j) = match e {
        Ev::Start => ("start", 0, None),
        Ev::ShortTick => ("short_tick", 0, None),
        Ev::Done(s, j) => ("done", *s, Some(*j)),
        Ev::Delay(j) => ("delay", 0, Some(*j)),
        Ev::Batch => ("batch", 0, None),
    };
    let jv = j.map_or(0u64, |j| {
        ((j.id as u64) << 16) | (u64::from(j.step) << 8) | u64::from(j.short)
    });
    SimEvent::Custom {
        name: name.into(),
        data: (jv | ((st as u64) << 56)).to_le_bytes().to_vec(),
    }
}

fn unpack(e: &SimEvent) -> Option<Ev> {
    let SimEvent::Custom { name, data } = e else {
        return None;
    };
    let v = u64::from_le_bytes(data.as_slice().try_into().ok()?);
    let job = Job {
        short: v & 0xff == 1,
        id: usize::try_from((v >> 16) & 0xffff_ffff).ok()?,
        step: ((v >> 8) & 0xff) as u8,
    };
    Some(match name.as_str() {
        "start" => Ev::Start,
        "short_tick" => Ev::ShortTick,
        "done" => Ev::Done((v >> 56) as usize, job),
        "delay" => Ev::Delay(job),
        "batch" => Ev::Batch,
        _ => return None,
    })
}

struct Station {
    free: usize,
    size: usize,
    hi: VecDeque<(Job, f64)>,
    lo: VecDeque<(Job, f64)>,
    busy_min: f64,
}

struct Sim {
    c: CourseFlowConfig,
    rng: SimRng,
    sched: EventScheduler,
    now: f64,
    warm: f64,
    st: Vec<Station>,
    wip: usize,
    pending: usize,
    started_at: Vec<f64>,
    coursera: Vec<bool>,
    batch_wait: Vec<(usize, f64)>,
    bw_sum: f64,
    bw_n: u64,
    sojourn: Vec<f64>,
    shorts_done: u64,
    in_system: usize,
    l_area: f64,
    last_t: f64,
}

impl Sim {
    fn at(&mut self, dt: f64, e: &Ev) {
        self.sched
            .schedule(SimTime::from_secs((self.now + dt) * 60.0), pack(e));
    }

    fn erlang4(&mut self, mean: f64) -> f64 {
        let p: f64 = (0..4).map(|_| 1.0 - self.rng.gen_f64()).product();
        -mean / 4.0 * p.ln()
    }

    fn enqueue(&mut self, s: usize, job: Job, mean: f64) {
        let d = self.erlang4(mean);
        if job.short {
            self.st[s].hi.push_back((job, d));
        } else {
            self.st[s].lo.push_back((job, d));
        }
        self.dispatch(s);
    }

    fn dispatch(&mut self, s: usize) {
        while self.st[s].free > 0 {
            let Some((job, d)) = self.st[s]
                .hi
                .pop_front()
                .or_else(|| self.st[s].lo.pop_front())
            else {
                break;
            };
            self.st[s].free -= 1;
            if self.now >= self.warm {
                self.st[s].busy_min += d;
            }
            self.at(d, &Ev::Done(s, job));
        }
    }

    fn start_course(&mut self) {
        let id = self.started_at.len();
        self.started_at.push(self.now);
        let cs = self.rng.gen_f64() < self.c.coursera_share;
        self.coursera.push(cs);
        self.wip += 1;
        self.in_system += 1;
        let m = self.c.outline_min;
        self.enqueue(
            OUTLINE,
            Job {
                short: false,
                id,
                step: 0,
            },
            m,
        );
    }

    fn launch(&mut self, id: usize) {
        self.in_system -= 1;
        if self.now >= self.warm {
            self.sojourn.push(self.now - self.started_at[id]);
        }
    }

    fn leave_wip(&mut self) {
        self.wip -= 1;
        if self.pending > 0 {
            self.pending -= 1;
            self.start_course();
        }
    }

    fn course_next(&mut self, j: Job) {
        let c = self.c.clone();
        let nj = |step| Job { step, ..j };
        match j.step {
            0 => self.enqueue(BUILD, nj(1), c.build_min),
            1 => self.enqueue(DESIGN, nj(2), c.design_min),
            2 => self.enqueue(RENDER, nj(3), c.render_min),
            3 => self.enqueue(VERIFY, nj(4), c.verify_min),
            4 if self.coursera[j.id] => self.enqueue(CHROME, nj(5), c.stage_min),
            4 => self.at(15.0, &Ev::Delay(nj(6))),
            5 => {
                if !self.c.batch_in_wip {
                    self.leave_wip();
                }
                self.batch_wait.push((j.id, self.now));
            }
            _ => {
                self.leave_wip();
                self.launch(j.id);
            }
        }
    }

    /// The 07:00 batch: release every waiting course, and when the wait counts as WIP, free its slot now.
    fn release_batch(&mut self) {
        for (id, since) in std::mem::take(&mut self.batch_wait) {
            if self.now >= self.warm {
                self.bw_sum += self.now - since;
                self.bw_n += 1;
            }
            if self.c.batch_in_wip {
                self.leave_wip();
            }
            self.launch(id);
        }
    }

    fn short_next(&mut self, j: Job) {
        let nj = |step| Job { step, ..j };
        match j.step {
            0 => self.enqueue(RENDER, nj(1), 5.0),
            1 => self.at(20.0, &Ev::Delay(nj(2))),
            2 => self.enqueue(CHROME, nj(3), 5.0),
            _ => {
                if self.now >= self.warm {
                    self.shorts_done += 1;
                }
            }
        }
    }

    fn step(&mut self, e: &Ev) {
        match *e {
            Ev::Start => {
                if self.wip < self.c.wip {
                    self.start_course();
                } else {
                    self.pending += 1;
                }
                self.at(DAY / self.c.starts_per_day, &Ev::Start);
            }
            Ev::ShortTick => {
                let id = self.started_at.len() + self.shorts_done as usize;
                self.at(
                    20.0,
                    &Ev::Delay(Job {
                        short: true,
                        id,
                        step: 0,
                    }),
                );
                self.at(DAY / self.c.shorts_per_day, &Ev::ShortTick);
            }
            Ev::Done(s, j) => {
                self.st[s].free += 1;
                self.dispatch(s);
                if j.short {
                    self.short_next(j);
                } else {
                    self.course_next(j);
                }
            }
            Ev::Delay(j) if j.short => self.short_next(j),
            Ev::Delay(j) => self.course_next(j),
            Ev::Batch => {
                self.release_batch();
                self.at(DAY, &Ev::Batch);
            }
        }
    }
}

/// Run one seed for `days`, discarding the first `warmup_d`.
#[must_use]
pub fn run_course_flow(
    c: &CourseFlowConfig,
    seed: u64,
    days: f64,
    warmup_d: f64,
) -> CourseFlowResult {
    let size = [1, c.builders, c.design_lanes, c.gpus, 4, c.chrome_lanes];
    let mut s = Sim {
        c: c.clone(),
        rng: SimRng::new(seed),
        sched: EventScheduler::new(),
        now: 0.0,
        warm: warmup_d * DAY,
        st: size
            .iter()
            .map(|&n| Station {
                free: n,
                size: n,
                hi: VecDeque::new(),
                lo: VecDeque::new(),
                busy_min: 0.0,
            })
            .collect(),
        wip: 0,
        pending: 0,
        started_at: Vec::new(),
        coursera: Vec::new(),
        batch_wait: Vec::new(),
        bw_sum: 0.0,
        bw_n: 0,
        sojourn: Vec::new(),
        shorts_done: 0,
        in_system: 0,
        l_area: 0.0,
        last_t: 0.0,
    };
    s.at(0.0, &Ev::Start);
    s.at(0.0, &Ev::ShortTick);
    s.at(7.0 * 60.0, &Ev::Batch);
    let end = days * DAY;
    while let Some(ev) = s.sched.next() {
        let t = ev.time.as_secs_f64() / 60.0;
        if t > end {
            break;
        }
        let from = s.last_t.max(s.warm);
        if t > from {
            s.l_area += s.in_system as f64 * (t - from);
        }
        s.last_t = t;
        s.now = t;
        if let Some(e) = unpack(&ev.event) {
            s.step(&e);
        }
    }
    let span = end - s.warm;
    let n = s.sojourn.len() as f64;
    let mut util = [0.0; 6];
    for (u, st) in util.iter_mut().zip(&s.st) {
        *u = st.busy_min / (span * st.size as f64);
    }
    CourseFlowResult {
        courses_per_day: n / span * DAY,
        shorts_per_day: s.shorts_done as f64 / span * DAY,
        lead_days: s.sojourn.iter().sum::<f64>() / n.max(1.0) / DAY,
        l: s.l_area / span,
        util,
        mean_batch_wait_h: s.bw_sum / (s.bw_n.max(1) as f64) / 60.0,
    }
}

/// Named FLOW-006 factory cases.
#[must_use]
pub fn course_preset(name: &str) -> Option<CourseFlowConfig> {
    let mut c = CourseFlowConfig {
        builders: 11,
        build_min: 2520.0,
        design_lanes: 2,
        design_min: 180.0,
        gpus: 1,
        render_min: 100.0,
        chrome_lanes: 2,
        stage_min: 360.0,
        coursera_share: 0.4,
        wip: 24,
        starts_per_day: 5.0,
        shorts_per_day: 5.0,
        outline_min: 90.0,
        verify_min: 15.0,
        batch_in_wip: false,
    };
    match name {
        "cal-today" => {
            c.builders = 1;
            c.design_lanes = 1;
            c.chrome_lanes = 1;
        }
        "design" => {}
        "nine-builders" => c.builders = 9,
        "coursera-24h" => c.stage_min = 1440.0,
        "render-300" => c.render_min = 300.0,
        _ => return None,
    }
    Some(c)
}

/// Names of the FLOW-006 factory cases.
#[must_use]
pub fn course_cases() -> [&'static str; 5] {
    [
        "cal-today",
        "design",
        "nine-builders",
        "coursera-24h",
        "render-300",
    ]
}

/// The reference run: seeds 1..=5, 300 days, 20 days warm-up, averaged.
#[must_use]
pub fn run_course_reference(c: &CourseFlowConfig) -> CourseFlowResult {
    let mut a = CourseFlowResult::default();
    for seed in 1..=5 {
        let r = run_course_flow(c, seed, 300.0, 20.0);
        a.courses_per_day += r.courses_per_day / 5.0;
        a.shorts_per_day += r.shorts_per_day / 5.0;
        a.lead_days += r.lead_days / 5.0;
        a.l += r.l / 5.0;
        a.mean_batch_wait_h += r.mean_batch_wait_h / 5.0;
        for i in 0..6 {
            a.util[i] += r.util[i] / 5.0;
        }
    }
    a
}

/// Override the case with the ledger's measured inputs (a JSON object); unknown keys are refused.
///
/// # Errors
/// Returns the offending key or type when the object holds something the model does not read.
pub fn apply_course_inputs(c: &mut CourseFlowConfig, v: &serde_json::Value) -> Result<(), String> {
    let obj = v.as_object().ok_or("inputs must be a JSON object")?;
    for (k, x) in obj {
        let f = x.as_f64().ok_or_else(|| format!("{k}: not a number"))?;
        match k.as_str() {
            "builders" => c.builders = f as usize,
            "build_min" => c.build_min = f,
            "design_lanes" => c.design_lanes = f as usize,
            "design_min" => c.design_min = f,
            "gpus" => c.gpus = f as usize,
            "render_min" => c.render_min = f,
            "chrome_lanes" => c.chrome_lanes = f as usize,
            "stage_min" => c.stage_min = f,
            "coursera_share" => c.coursera_share = f,
            "wip" => c.wip = f as usize,
            "starts_per_day" => c.starts_per_day = f,
            "shorts_per_day" => c.shorts_per_day = f,
            "batch_in_wip" => c.batch_in_wip = f != 0.0,
            _ => return Err(format!("unknown input: {k}")),
        }
    }
    Ok(())
}

/// The `sim` block fields this scenario owns, in the shape the cops publish.
#[must_use]
pub fn course_sim_block(
    case: &str,
    c: &CourseFlowConfig,
    r: &CourseFlowResult,
) -> serde_json::Value {
    let (station, util) = r.busiest();
    serde_json::json!({
        "scenario": format!("course-flow:{case}"),
        "seeds": 5, "hours": 300.0 * 24.0, "warmup_h": 20.0 * 24.0,
        "inputs": {
            "builders": c.builders, "build_min": c.build_min, "design_lanes": c.design_lanes,
            "design_min": c.design_min, "gpus": c.gpus, "render_min": c.render_min,
            "chrome_lanes": c.chrome_lanes, "stage_min": c.stage_min,
            "coursera_share": c.coursera_share, "wip": c.wip,
            "starts_per_day": c.starts_per_day, "shorts_per_day": c.shorts_per_day, "batch_in_wip": c.batch_in_wip,
        },
        "predicted": {
            "courses_day": r.courses_per_day, "shorts_day": r.shorts_per_day,
            "lead_days": r.lead_days, "L": r.l, "mean_batch_wait_h": r.mean_batch_wait_h, "busiest_util_pct": util * 100.0,
        },
        "binding_station": station,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FLOW-006 reference bands: (case, courses/day, tolerance).
    const WANT: [(&str, f64, f64); 5] = [
        ("cal-today", 0.54, 0.10),
        ("design", 4.92, 0.05),
        ("nine-builders", 4.90, 0.05),
        ("coursera-24h", 4.84, 0.05),
        ("render-300", 4.70, 0.05),
    ];

    fn got(name: &str) -> CourseFlowResult {
        run_course_reference(&course_preset(name).expect("case"))
    }

    #[test]
    fn courses_per_day_match_the_reference_bands() {
        for (name, want, tol) in WANT {
            let r = got(name);
            assert!(
                (r.courses_per_day - want).abs() <= want * tol,
                "{name}: {:.3} vs {want}",
                r.courses_per_day
            );
        }
    }

    #[test]
    fn shorts_hold_five_a_day_in_every_case() {
        for (name, _, _) in WANT {
            let r = got(name);
            assert!(
                (r.shorts_per_day - 5.0).abs() <= 0.1,
                "{name}: {:.2}",
                r.shorts_per_day
            );
        }
    }

    #[test]
    fn the_binding_station_is_the_one_the_case_squeezes() {
        assert_eq!(got("nine-builders").busiest().0, "build");
        assert!(got("nine-builders").util[BUILD] > 0.95);
        assert_eq!(got("render-300").busiest().0, "render");
        assert!(got("render-300").util[RENDER] > 0.95);
        assert_eq!(got("coursera-24h").busiest().0, "chrome");
    }

    #[test]
    fn littles_law_holds_within_half_a_percent() {
        for (name, _, _) in WANT.iter().filter(|w| w.0 != "cal-today") {
            let r = got(name);
            let lam_w = r.courses_per_day * r.lead_days;
            assert!(
                (lam_w - r.l).abs() <= r.l * 0.005,
                "{name}: {lam_w:.3} vs {:.3}",
                r.l
            );
        }
    }

    #[test]
    fn same_seed_same_output() {
        let c = course_preset("design").expect("case");
        assert_eq!(
            run_course_flow(&c, 4, 40.0, 5.0),
            run_course_flow(&c, 4, 40.0, 5.0)
        );
    }

    #[test]
    fn a_reverted_design_shows_red() {
        let mut c = course_preset("design").expect("case");
        c.builders = 1;
        assert!((run_course_reference(&c).courses_per_day - 4.92).abs() > 4.92 * 0.05);
    }

    fn with_batch_in_wip(name: &str) -> CourseFlowConfig {
        let mut c = course_preset(name).expect("case");
        c.batch_in_wip = true;
        c
    }

    /// B6: the daily batch wait is measured and reported, not hidden.
    #[test]
    fn mean_batch_wait_is_reported_and_inside_a_day() {
        let r = got("design");
        assert!(
            r.mean_batch_wait_h > 0.5 && r.mean_batch_wait_h < 24.0,
            "mean batch wait {:.2} h",
            r.mean_batch_wait_h
        );
    }

    /// B6: with the batch wait counted inside WIP, Little's law holds within 0.5%,
    /// and the WIP cap now binds on it, so it can only cost throughput.
    #[test]
    fn littles_law_holds_with_batch_wait_inside_wip() {
        for name in ["design", "nine-builders", "coursera-24h"] {
            let outside = got(name);
            let r = run_course_reference(&with_batch_in_wip(name));
            let lam_w = r.courses_per_day * r.lead_days;
            assert!(
                (lam_w - r.l).abs() <= r.l * 0.005,
                "{name}: {lam_w:.3} vs {:.3}",
                r.l
            );
            assert!(r.l <= 24.0 + 1e-9, "{name}: L {:.2} over the cap", r.l);
            assert!(
                r.courses_per_day <= outside.courses_per_day + 0.02,
                "{name}"
            );
        }
    }

    /// B6 falsifier: counting the batch wait inside WIP must change the answer, or the flag is dead.
    #[test]
    fn batch_in_wip_is_not_a_no_op() {
        let a = got("coursera-24h");
        let b = run_course_reference(&with_batch_in_wip("coursera-24h"));
        assert!((a.courses_per_day - b.courses_per_day).abs() > 1e-6 || (a.l - b.l).abs() > 1e-6);
    }

    #[test]
    fn unknown_case_is_none() {
        assert!(course_preset("nope").is_none());
    }
}
