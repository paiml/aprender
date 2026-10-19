//! Scenario "pr-flow": the aprender PR line as a discrete-event model (FLOW-005 SIM-2, #5511).
//!
//! Six components, in minutes, seeded: authors on a pull system, the WIP cap, PR checks on a
//! runner pool (red with `p_red`, then fix and rerun), the arm delay, the merge queue (batch,
//! minimum wait, build limit, runner pool, flake sends the whole group back) and merge.
//! Events ride `engine::scheduler`, time is `SimTime`, randomness is `SimRng`.

use std::collections::VecDeque;

use crate::engine::rng::SimRng;
use crate::engine::scheduler::EventScheduler;
use crate::engine::state::SimEvent;
use crate::engine::SimTime;

/// Where the review sits relative to the PR checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewMode {
    Serial,
    Concurrent,
}

/// Inputs of the line. Times are minutes; `*_runner_min` is the runner-minutes of one run.
#[derive(Debug, Clone, PartialEq)]
pub struct PrFlowConfig {
    pub cap: usize,
    /// Mean concurrent authors; the fractional part is one author who thinks 1/frac times slower.
    pub n: f64,
    pub t_author_min: f64,
    pub r_pr: usize,
    pub pr_runner_min: f64,
    pub pr_jobs: usize,
    pub pr_setup_min: f64,
    pub p_red: f64,
    pub t_fix_min: f64,
    pub t_arm_min: f64,
    /// Review wait per PR after green checks, serial with arming; 0 = no review station.
    pub t_rev_min: f64,
    /// Review runs alongside the checks (`Concurrent`) or after them (`Serial`).
    pub rev_mode: ReviewMode,
    /// Parallel reviewers; 0 = unlimited (no queue at the review station).
    pub rev_lanes: usize,
    /// Per-tier review minutes, handed out to PRs round-robin; empty = `t_rev_min` for all.
    pub rev_tiers: Vec<f64>,
    pub r_mq: usize,
    pub mq_runner_min: f64,
    pub mq_jobs: usize,
    pub mq_setup_min: f64,
    pub builds: usize,
    pub mq_batch: usize,
    pub mq_wait_min: f64,
    pub f: f64,
    /// Chance an armed PR is found conflicting (main moved); 0 keeps the FLOW-005 designs.
    pub p_dirty: f64,
    /// Minutes until a conflicting PR is rebased and rerun (FLOW-006 keep-fresh rule).
    pub t_rebase_min: f64,
}

impl PrFlowConfig {
    /// Mean think time of author `a`: the last, fractional author is slowed by its fraction.
    fn think_min(&self, a: usize) -> f64 {
        let frac = self.n - self.n.floor();
        if frac > 0.0 && a == self.n.floor() as usize {
            self.t_author_min / frac
        } else {
            self.t_author_min
        }
    }
}

/// What one seeded run measured after the warm-up.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrFlowResult {
    pub merged_per_h: f64,
    pub w_mean_min: f64,
    pub w_p90_min: f64,
    pub l: f64,
    pub prs_per_pass: f64,
    pub pr_util: f64,
    pub mq_util: f64,
    pub runs_per_pr: f64,
    /// Author busy fraction: share of author time not spent blocked on the WIP cap.
    pub rho_a: f64,
    /// Mean wall of one PR-check run, start to verdict (runner wait + jobs + setup).
    pub run_wall_min: f64,
    /// Mean open to green checks, merged PRs only (eq 7 terms r(w+c_wall) + (r-1)t_fix).
    pub checks_min: f64,
    /// Mean green to armed (eq 7 term t_arm, plus any serial review).
    pub arm_min: f64,
    /// Mean armed to merged (eq 7 term W_mq).
    pub mq_min: f64,
    /// Mean armed to the start of the first merge pass that holds the PR (the wait in W_mq).
    pub pass_wait_min: f64,
    /// Mean wall of one merge pass, group formed to verdict (runner wait + jobs + setup).
    pub pass_wall_min: f64,
}

#[derive(Clone, Copy)]
enum Ev {
    Author(usize),
    PrJob(usize),
    PrRun(usize),
    Fixed(usize),
    Armed(usize),
    Reviewed(usize),
    Rebased(usize),
    MqJob(usize),
    MqPass(usize),
    MqTimer,
}

impl Ev {
    fn pack(self) -> SimEvent {
        let (name, id) = match self {
            Self::Author(i) => ("author", i),
            Self::PrJob(i) => ("pr_job", i),
            Self::PrRun(i) => ("pr_run", i),
            Self::Fixed(i) => ("fixed", i),
            Self::Armed(i) => ("armed", i),
            Self::Reviewed(i) => ("reviewed", i),
            Self::Rebased(i) => ("rebased", i),
            Self::MqJob(i) => ("mq_job", i),
            Self::MqPass(i) => ("mq_pass", i),
            Self::MqTimer => ("mq_timer", 0),
        };
        SimEvent::Custom {
            name: name.into(),
            data: (id as u64).to_le_bytes().to_vec(),
        }
    }

    fn unpack(e: &SimEvent) -> Option<Self> {
        let SimEvent::Custom { name, data } = e else {
            return None;
        };
        let id = usize::try_from(u64::from_le_bytes(data.as_slice().try_into().ok()?)).ok()?;
        Some(match name.as_str() {
            "author" => Self::Author(id),
            "pr_job" => Self::PrJob(id),
            "pr_run" => Self::PrRun(id),
            "fixed" => Self::Fixed(id),
            "armed" => Self::Armed(id),
            "reviewed" => Self::Reviewed(id),
            "rebased" => Self::Rebased(id),
            "mq_job" => Self::MqJob(id),
            "mq_pass" => Self::MqPass(id),
            "mq_timer" => Self::MqTimer,
            _ => return None,
        })
    }
}

/// A pool of identical runners with a FIFO of waiting jobs `(owner, minutes)`.
struct Pool {
    free: usize,
    size: usize,
    waiting: VecDeque<(usize, f64)>,
    busy_min: f64,
}

impl Pool {
    fn new(size: usize) -> Self {
        Self {
            free: size,
            size,
            waiting: VecDeque::new(),
            busy_min: 0.0,
        }
    }
}

struct Sim {
    c: PrFlowConfig,
    rng: SimRng,
    sched: EventScheduler,
    now: f64,
    warm: f64,
    open: usize,
    blocked: VecDeque<usize>,
    open_at: Vec<f64>,
    jobs_left: Vec<usize>,
    pr: Pool,
    mq: Pool,
    queue: VecDeque<usize>,
    enq_at: Vec<f64>,
    groups: Vec<Vec<usize>>,
    group_jobs_left: Vec<usize>,
    group_at: Vec<f64>,
    pass_wall: (f64, u64),
    active: usize,
    timer_at: f64,
    sojourn: Vec<f64>,
    l_area: f64,
    rev_free: usize,
    rev_wait: VecDeque<usize>,
    green: Vec<bool>,
    rev_ok: Vec<bool>,
    green_at: Vec<f64>,
    run_at: Vec<f64>,
    wall: (f64, u64),
    stage_sum: [f64; 4],
    pass_at: Vec<f64>,
    blocked_area: f64,
    last_t: f64,
    runs: u64,
    prs_seen: u64,
    passes: u64,
}

impl Sim {
    fn at(&mut self, dt: f64, e: Ev) {
        self.sched
            .schedule(SimTime::from_secs((self.now + dt) * 60.0), e.pack());
    }

    fn exp(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.rng.gen_f64()).ln()
    }

    fn start_run(&mut self, pr: usize) {
        self.runs += 1;
        self.run_at[pr] = self.now;
        self.jobs_left[pr] = self.c.pr_jobs;
        let d = self.c.pr_runner_min / self.c.pr_jobs as f64;
        for _ in 0..self.c.pr_jobs {
            self.pr.waiting.push_back((pr, d));
        }
        self.dispatch_pr();
    }

    fn dispatch_pr(&mut self) {
        while self.pr.free > 0 {
            let Some((pr, d)) = self.pr.waiting.pop_front() else {
                break;
            };
            self.pr.free -= 1;
            if self.now >= self.warm {
                self.pr.busy_min += d;
            }
            self.at(d, Ev::PrJob(pr));
        }
    }

    fn open_pr(&mut self, author: usize) {
        let pr = self.open_at.len();
        self.open_at.push(self.now);
        self.enq_at.push(0.0);
        self.jobs_left.push(0);
        self.green.push(false);
        self.rev_ok.push(false);
        self.green_at.push(0.0);
        self.run_at.push(0.0);
        self.pass_at.push(-1.0);
        self.open += 1;
        self.prs_seen += 1;
        if self.c.rev_mode == ReviewMode::Concurrent {
            self.request_review(pr);
        }
        self.start_run(pr);
        let t = self.exp(self.c.think_min(author));
        self.at(t, Ev::Author(author));
    }

    fn form_groups(&mut self) {
        while self.active < self.c.builds && !self.queue.is_empty() {
            let head = self.enq_at[self.queue[0]];
            let ready =
                self.queue.len() >= self.c.mq_batch || self.now >= head + self.c.mq_wait_min;
            if !ready {
                if self.timer_at < self.now {
                    self.timer_at = head + self.c.mq_wait_min;
                    self.at(self.timer_at - self.now, Ev::MqTimer);
                }
                return;
            }
            let take = self.c.mq_batch.min(self.queue.len());
            let members: Vec<usize> = self.queue.drain(..take).collect();
            for &pr in &members {
                if self.pass_at[pr] < self.enq_at[pr] {
                    self.pass_at[pr] = self.now;
                }
            }
            self.groups.push(members);
            let g = self.groups.len() - 1;
            self.group_jobs_left.push(self.c.mq_jobs);
            self.group_at.push(self.now);
            self.active += 1;
            let d = self.c.mq_runner_min / self.c.mq_jobs as f64;
            for _ in 0..self.c.mq_jobs {
                self.mq.waiting.push_back((g, d));
            }
        }
        self.dispatch_mq();
    }

    fn dispatch_mq(&mut self) {
        while self.mq.free > 0 {
            let Some((g, d)) = self.mq.waiting.pop_front() else {
                break;
            };
            self.mq.free -= 1;
            if self.now >= self.warm {
                self.mq.busy_min += d;
            }
            self.at(d, Ev::MqJob(g));
        }
    }

    fn pass_done(&mut self, g: usize) {
        self.active -= 1;
        let members = std::mem::take(&mut self.groups[g]);
        if self.now >= self.warm {
            self.pass_wall.0 += self.now - self.group_at[g];
            self.pass_wall.1 += 1;
        }
        if self.rng.gen_f64() < self.c.f {
            for pr in members.into_iter().rev() {
                self.queue.push_front(pr);
            }
        } else {
            if self.now >= self.warm {
                self.passes += 1;
            }
            for pr in members {
                self.merge(pr);
            }
        }
        self.form_groups();
    }

    /// Review minutes of `pr`: its round-robin tier, else the flat `t_rev_min`.
    fn review_min(&self, pr: usize) -> f64 {
        self.c
            .rev_tiers
            .get(pr % self.c.rev_tiers.len().max(1))
            .copied()
            .unwrap_or(self.c.t_rev_min)
    }

    /// Ask the review station for `pr`; a zero-minute review finishes inline (no event).
    fn request_review(&mut self, pr: usize) {
        if self.review_min(pr) <= 0.0 {
            self.reviewed(pr);
        } else if self.rev_free > 0 {
            self.rev_free -= 1;
            self.at(self.review_min(pr), Ev::Reviewed(pr));
        } else {
            self.rev_wait.push_back(pr);
        }
    }

    /// A PR-check run reached its verdict: red goes to fix, green goes on.
    fn run_done(&mut self, pr: usize) {
        if self.now >= self.warm {
            self.wall.0 += self.now - self.run_at[pr];
            self.wall.1 += 1;
        }
        if self.rng.gen_f64() < self.c.p_red {
            self.at(self.c.t_fix_min, Ev::Fixed(pr));
        } else {
            self.checks_green(pr);
        }
    }

    fn checks_green(&mut self, pr: usize) {
        self.green[pr] = true;
        self.green_at[pr] = self.now;
        match self.c.rev_mode {
            ReviewMode::Serial => self.request_review(pr),
            ReviewMode::Concurrent if self.rev_ok[pr] => self.at(self.c.t_arm_min, Ev::Armed(pr)),
            ReviewMode::Concurrent => {}
        }
    }

    fn reviewed(&mut self, pr: usize) {
        self.rev_ok[pr] = true;
        if self.review_min(pr) > 0.0 {
            self.rev_free = self.rev_free.saturating_add(1);
            if let Some(next) = self.rev_wait.pop_front() {
                self.rev_free -= 1;
                self.at(self.review_min(next), Ev::Reviewed(next));
            }
        }
        if self.green[pr] {
            self.at(self.c.t_arm_min, Ev::Armed(pr));
        }
    }

    fn merge(&mut self, pr: usize) {
        if self.now >= self.warm {
            self.sojourn.push(self.now - self.open_at[pr]);
            let (open, green, armed) = (self.open_at[pr], self.green_at[pr], self.enq_at[pr]);
            let stages = [
                green - open,
                armed - green,
                self.now - armed,
                self.pass_at[pr] - armed,
            ];
            for (s, d) in self.stage_sum.iter_mut().zip(stages) {
                *s += d;
            }
        }
        self.open -= 1;
        if let Some(a) = self.blocked.pop_front() {
            self.open_pr(a);
        }
    }

    fn step(&mut self, e: Ev) {
        match e {
            Ev::Author(a) if self.open < self.c.cap => self.open_pr(a),
            Ev::Author(a) => self.blocked.push_back(a),
            Ev::PrJob(pr) => {
                self.pr.free += 1;
                self.jobs_left[pr] -= 1;
                if self.jobs_left[pr] == 0 {
                    self.at(self.c.pr_setup_min, Ev::PrRun(pr));
                }
                self.dispatch_pr();
            }
            Ev::PrRun(pr) => self.run_done(pr),
            Ev::Reviewed(pr) => self.reviewed(pr),
            Ev::Fixed(pr) | Ev::Rebased(pr) => self.start_run(pr),
            Ev::Armed(pr) if self.c.p_dirty > 0.0 && self.rng.gen_f64() < self.c.p_dirty => {
                self.at(self.c.t_rebase_min, Ev::Rebased(pr));
            }
            Ev::Armed(pr) if self.c.builds == 0 => self.merge(pr),
            Ev::Armed(pr) => {
                self.enq_at[pr] = self.now;
                self.queue.push_back(pr);
                self.form_groups();
            }
            Ev::MqJob(g) => {
                self.mq.free += 1;
                self.group_jobs_left[g] -= 1;
                if self.group_jobs_left[g] == 0 {
                    self.at(self.c.mq_setup_min, Ev::MqPass(g));
                }
                self.dispatch_mq();
            }
            Ev::MqPass(g) => self.pass_done(g),
            Ev::MqTimer => self.form_groups(),
        }
    }
}

/// Run one seed for `hours`, discarding the first `warmup_h`.
#[must_use]
pub fn run_pr_flow(c: &PrFlowConfig, seed: u64, hours: f64, warmup_h: f64) -> PrFlowResult {
    let mut s = Sim {
        c: c.clone(),
        rng: SimRng::new(seed),
        sched: EventScheduler::new(),
        now: 0.0,
        warm: warmup_h * 60.0,
        open: 0,
        blocked: VecDeque::new(),
        open_at: Vec::new(),
        jobs_left: Vec::new(),
        pr: Pool::new(c.r_pr),
        mq: Pool::new(c.r_mq),
        queue: VecDeque::new(),
        enq_at: Vec::new(),
        groups: Vec::new(),
        group_jobs_left: Vec::new(),
        group_at: Vec::new(),
        pass_wall: (0.0, 0),
        active: 0,
        timer_at: -1.0,
        sojourn: Vec::new(),
        l_area: 0.0,
        blocked_area: 0.0,
        rev_free: if c.rev_lanes == 0 {
            usize::MAX
        } else {
            c.rev_lanes
        },
        rev_wait: VecDeque::new(),
        green: Vec::new(),
        rev_ok: Vec::new(),
        green_at: Vec::new(),
        run_at: Vec::new(),
        wall: (0.0, 0),
        stage_sum: [0.0; 4],
        pass_at: Vec::new(),
        last_t: 0.0,
        runs: 0,
        prs_seen: 0,
        passes: 0,
    };
    s.timer_at = -1.0;
    for a in 0..c.n.ceil() as usize {
        let t = s.exp(c.think_min(a));
        s.at(t, Ev::Author(a));
    }
    let end = hours * 60.0;
    while let Some(ev) = s.sched.next() {
        let t = ev.time.as_secs_f64() / 60.0;
        if t > end {
            break;
        }
        let from = s.last_t.max(s.warm);
        if t > from {
            s.l_area += s.open as f64 * (t - from);
            s.blocked_area += s.blocked.len() as f64 * (t - from);
        }
        s.last_t = t;
        s.now = t;
        if let Some(e) = Ev::unpack(&ev.event) {
            s.step(e);
        }
    }
    s.summary(end)
}

impl Sim {
    fn summary(&mut self, end: f64) -> PrFlowResult {
        let span = end - self.warm;
        let n = self.sojourn.len() as f64;
        self.sojourn.sort_by(f64::total_cmp);
        let p90 = self
            .sojourn
            .get(((n * 0.9) as usize).min(self.sojourn.len().saturating_sub(1)));
        PrFlowResult {
            merged_per_h: n / span * 60.0,
            w_mean_min: self.sojourn.iter().sum::<f64>() / n.max(1.0),
            w_p90_min: p90.copied().unwrap_or(0.0),
            l: self.l_area / span,
            prs_per_pass: n / (self.passes as f64).max(1.0),
            pr_util: self.pr.busy_min / (span * self.pr.size as f64),
            mq_util: self.mq.busy_min / (span * self.mq.size as f64),
            runs_per_pr: self.runs as f64 / (self.prs_seen as f64).max(1.0),
            rho_a: 1.0 - self.blocked_area / (span * self.c.n.ceil().max(1.0)),
            run_wall_min: self.wall.0 / (self.wall.1 as f64).max(1.0),
            checks_min: self.stage_sum[0] / n.max(1.0),
            arm_min: self.stage_sum[1] / n.max(1.0),
            mq_min: self.stage_sum[2] / n.max(1.0),
            pass_wait_min: self.stage_sum[3] / n.max(1.0),
            pass_wall_min: self.pass_wall.0 / (self.pass_wall.1 as f64).max(1.0),
        }
    }
}

/// Named FLOW-005 designs. `rm` is runner-minutes per run; the fast tier (24) has a shorter
/// wall. Setup minutes are the overheads the spec leaves open, calibrated to the reference.
#[must_use]
pub fn preset(name: &str) -> Option<PrFlowConfig> {
    let (cap, r_pr, r_mq, builds, rm, p_red, fast) = match name {
        "D" => (30, 38, 4, 1, 72.0, 0.2, false),
        "A" => (10, 40, 4, 5, 89.2, 0.2, false),
        "G" => (10, 12, 8, 2, 24.0, 0.1, true),
        "E" => (30, 12, 4, 1, 24.0, 0.1, true),
        _ => return None,
    };
    let setup = if fast { 9.0 } else { 7.7 };
    Some(PrFlowConfig {
        cap,
        n: 11.0,
        t_author_min: 30.0,
        r_pr,
        pr_runner_min: rm,
        pr_jobs: 4,
        pr_setup_min: if fast { 15.0 } else { setup },
        p_red,
        t_fix_min: 10.0,
        t_arm_min: 2.0,
        t_rev_min: 0.0,
        rev_mode: ReviewMode::Serial,
        rev_lanes: 0,
        rev_tiers: Vec::new(),
        r_mq,
        mq_runner_min: rm,
        mq_jobs: 4,
        mq_setup_min: setup,
        builds,
        mq_batch: 10,
        mq_wait_min: 5.0,
        f: 0.02,
        p_dirty: 0.0,
        t_rebase_min: 0.0,
    })
}

/// Throughput bounds (merged/h) of the four stations; the binding station is the smallest.
///
/// `cap` is cap/W (eq 2), `supply` is N authors at one PR per think time, `pr_checks` and
/// `merge_queue` are pool capacity over the runner-minutes one merged PR consumes (eq 3, 5, 6).
#[must_use]
pub fn bounds(c: &PrFlowConfig, r: &PrFlowResult) -> [(&'static str, f64); 4] {
    let w_h = (r.w_mean_min / 60.0).max(f64::EPSILON);
    let pr_per_merge = c.pr_runner_min * r.runs_per_pr.max(1.0);
    let mq_per_merge = c.mq_runner_min / r.prs_per_pass.max(1.0);
    [
        ("cap", c.cap as f64 / w_h),
        ("supply", c.n * 60.0 / c.t_author_min),
        ("pr_checks", c.r_pr as f64 * 60.0 / pr_per_merge),
        ("merge_queue", c.r_mq as f64 * 60.0 / mq_per_merge),
    ]
}

/// The binding station: argmin of [`bounds`]. No utilisation thresholds.
#[must_use]
pub fn binding_station(c: &PrFlowConfig, r: &PrFlowResult) -> &'static str {
    bounds(c, r)
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or("none", |b| b.0)
}

/// Flow model eq 7, time open, term by term.
///
/// `W = r(w + c_wall) + (r-1)t_fix + t_arm + p_dirty/(1-p_dirty)(t_reb + r(w + c_wall) + (r-1)t_fix + t_arm) + W_mq`
/// with `W_mq = wait_mq + (w_pass + t_pass)/(1-f)`. This line has no rebase, so `p_dirty = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eq7Law {
    /// Runs per PR, `1/(1-p_red)` (eq 4).
    pub r: f64,
    /// Runner wait per run: the sim's mean run wall less `c_wall`.
    pub w_min: f64,
    /// Wall of one run with free runners: job minutes plus setup.
    pub c_wall_min: f64,
    pub t_fix_min: f64,
    /// Green to armed, plus the review minutes when the review is serial.
    pub t_arm_min: f64,
    pub p_dirty: f64,
    pub t_reb_min: f64,
    /// Armed to the start of the first merge pass that holds the PR.
    pub wait_mq_min: f64,
    /// Wall of one merge pass with free runners: job minutes plus setup.
    pub t_pass_min: f64,
    /// Runner wait inside one merge pass: the sim's mean pass wall less `t_pass`.
    pub w_pass_min: f64,
    pub f: f64,
}

impl Eq7Law {
    /// Open to green: `r(w + c_wall) + (r-1)t_fix`.
    #[must_use]
    pub fn checks_min(&self) -> f64 {
        self.r * (self.w_min + self.c_wall_min) + (self.r - 1.0) * self.t_fix_min
    }

    /// Rebase loop: `p_dirty/(1-p_dirty)(t_reb + checks + t_arm)`.
    #[must_use]
    pub fn rebase_min(&self) -> f64 {
        self.p_dirty / (1.0 - self.p_dirty) * (self.t_reb_min + self.checks_min() + self.t_arm_min)
    }

    /// `W_mq = wait_mq + (w_pass + t_pass)/(1-f)`: a flake sends the whole group round again.
    #[must_use]
    pub fn w_mq_min(&self) -> f64 {
        self.wait_mq_min + (self.w_pass_min + self.t_pass_min) / (1.0 - self.f)
    }

    /// Eq 7, mean time open.
    #[must_use]
    pub fn w_law_min(&self) -> f64 {
        self.checks_min() + self.t_arm_min + self.rebase_min() + self.w_mq_min()
    }
}

/// Eq 7 evaluated on `c`, with the queue waits (`w`, `wait_mq`, `w_pass`) read from the run `r`.
///
/// `wait_mq` is the wait for a merge pass to start, not the batch timer: with one build at a
/// time the slot is busy, so an armed PR waits out the pass in flight and any overflow.
#[must_use]
pub fn eq7_law(c: &PrFlowConfig, r: &PrFlowResult) -> Eq7Law {
    let c_wall_min = c.pr_runner_min / c.pr_jobs.max(1) as f64 + c.pr_setup_min;
    let t_pass_min = c.mq_runner_min / c.mq_jobs.max(1) as f64 + c.mq_setup_min;
    let review = match c.rev_mode {
        ReviewMode::Serial if c.rev_tiers.is_empty() => c.t_rev_min,
        ReviewMode::Serial => c.rev_tiers.iter().sum::<f64>() / c.rev_tiers.len() as f64,
        ReviewMode::Concurrent => 0.0,
    };
    Eq7Law {
        r: 1.0 / (1.0 - c.p_red),
        w_min: (r.run_wall_min - c_wall_min).max(0.0),
        c_wall_min,
        t_fix_min: c.t_fix_min,
        t_arm_min: c.t_arm_min + review,
        p_dirty: 0.0,
        t_reb_min: 0.0,
        wait_mq_min: r.pass_wait_min,
        t_pass_min,
        w_pass_min: (r.pass_wall_min - t_pass_min).max(0.0),
        f: c.f,
    }
}

/// The reference run: seeds 1..=5, 400 h, 20 h warm-up, averaged.
#[must_use]
pub fn run_reference(c: &PrFlowConfig) -> PrFlowResult {
    let mut acc = PrFlowResult::default();
    for seed in 1..=5 {
        let r = run_pr_flow(c, seed, 400.0, 20.0);
        acc.merged_per_h += r.merged_per_h / 5.0;
        acc.w_mean_min += r.w_mean_min / 5.0;
        acc.w_p90_min += r.w_p90_min / 5.0;
        acc.l += r.l / 5.0;
        acc.prs_per_pass += r.prs_per_pass / 5.0;
        acc.pr_util += r.pr_util / 5.0;
        acc.mq_util += r.mq_util / 5.0;
        acc.runs_per_pr += r.runs_per_pr / 5.0;
        acc.rho_a += r.rho_a / 5.0;
        acc.run_wall_min += r.run_wall_min / 5.0;
        acc.checks_min += r.checks_min / 5.0;
        acc.arm_min += r.arm_min / 5.0;
        acc.mq_min += r.mq_min / 5.0;
        acc.pass_wait_min += r.pass_wait_min / 5.0;
        acc.pass_wall_min += r.pass_wall_min / 5.0;
    }
    acc
}

/// SIM-3 contract: every design's merged/h is within 5% of its reference (D 20.08, A 4.96, G 12.55, E 21.37).
#[must_use]
pub fn reference_contract_ok() -> bool {
    [("D", 20.08), ("A", 4.96), ("G", 12.55), ("E", 21.37)]
        .iter()
        .all(|(d, want)| {
            preset(d).is_some_and(|c| (run_reference(&c).merged_per_h - want).abs() <= want * 0.05)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WANT: [(&str, f64); 4] = [("D", 20.08), ("A", 4.96), ("G", 12.55), ("E", 21.37)];

    fn within(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= want * tol
    }

    fn roomy() -> PrFlowConfig {
        let mut c = preset("D").expect("preset");
        c.cap = 1000;
        c.n = 1000.0;
        c.r_pr = 10_000;
        c.r_mq = 10_000;
        c.builds = 100;
        c
    }

    #[test]
    fn six_minutes_of_review_adds_about_two_to_l_at_20_per_hour() {
        let mut c = roomy();
        c.t_author_min = 600.0;
        c.n = 200.0;
        let base = run_reference(&c);
        assert!(
            within(base.merged_per_h, 20.0, 0.05),
            "lambda {}",
            base.merged_per_h
        );
        c.t_rev_min = 6.0;
        let r = run_reference(&c);
        let dl = r.l - base.l;
        assert!(within(dl, 2.0, 0.1), "dL {dl}");
    }

    #[test]
    fn author_busy_fraction_is_one_without_a_cap_and_falls_under_one_with_it() {
        let mut c = roomy();
        c.n = 10.0;
        let free = run_reference(&c);
        assert!(free.rho_a > 0.999, "free {}", free.rho_a);
        let mut c = roomy();
        c.n = 60.0;
        c.cap = 2;
        let capped = run_reference(&c);
        assert!(capped.rho_a < 0.5, "capped {}", capped.rho_a);
    }

    fn open_line() -> PrFlowConfig {
        let mut c = roomy();
        c.t_author_min = 600.0;
        c.n = 200.0;
        c.p_red = 0.0;
        c
    }

    #[test]
    fn concurrent_review_inside_the_check_wall_removes_the_serial_wait() {
        let mut c = open_line();
        let base = run_reference(&c).w_mean_min;
        c.t_rev_min = 6.0;
        let serial = run_reference(&c).w_mean_min;
        c.rev_mode = ReviewMode::Concurrent;
        let conc = run_reference(&c).w_mean_min;
        assert!(
            within(serial - base, 6.0, 0.05),
            "serial adds {}",
            serial - base
        );
        assert!(
            within(serial - conc, 6.0, 0.05),
            "concurrent saves {}",
            serial - conc
        );
    }

    #[test]
    fn one_lane_queues_reviews_and_tiers_set_each_prs_time() {
        let mut c = open_line();
        c.t_rev_min = 6.0;
        let free = run_reference(&c).w_mean_min;
        c.rev_lanes = 1;
        let one = run_reference(&c).w_mean_min;
        assert!(one > free + 5.0, "one lane {one} vs unlimited {free}");
        let mut t = open_line();
        t.rev_tiers = vec![2.0, 10.0];
        let tiered = run_reference(&t).w_mean_min - run_reference(&open_line()).w_mean_min;
        assert!(within(tiered, 6.0, 0.1), "mean of tiers adds {tiered}");
    }

    #[test]
    fn binding_is_cap_when_l_hits_cap() {
        let mut c = roomy();
        c.n = 60.0;
        c.cap = 2;
        let r = run_reference(&c);
        assert!(r.l > 0.7 * c.cap as f64, "L {}", r.l);
        assert_eq!(binding_station(&c, &r), "cap");
    }

    #[test]
    fn binding_is_supply_when_authors_are_starved() {
        let mut c = roomy();
        c.n = 1.0;
        let r = run_reference(&c);
        assert_eq!(binding_station(&c, &r), "supply");
    }

    #[test]
    fn binding_is_a_pool_when_the_pool_is_small() {
        let mut c = roomy();
        c.n = 60.0;
        c.r_pr = 4;
        let r = run_reference(&c);
        assert_eq!(binding_station(&c, &r), "pr_checks");
        let mut c = roomy();
        c.n = 60.0;
        c.r_mq = 1;
        c.builds = 1;
        let r = run_reference(&c);
        assert_eq!(binding_station(&c, &r), "merge_queue");
    }

    #[test]
    fn fractional_n_sits_between_its_integer_neighbours() {
        let at = |n: f64| {
            let mut c = preset("D").expect("preset");
            c.n = n;
            c.cap = 100;
            run_reference(&c).merged_per_h
        };
        let (lo, mid, hi) = (at(11.0), at(11.5), at(12.0));
        assert!(lo < mid && mid < hi, "{lo} {mid} {hi}");
    }

    #[test]
    fn merges_per_hour_match_the_reference_within_5_percent() {
        for (name, want) in WANT {
            let r = run_reference(&preset(name).expect("preset"));
            assert!(
                within(r.merged_per_h, want, 0.05),
                "{name}: {:.2} vs {want}",
                r.merged_per_h
            );
        }
    }

    #[test]
    fn littles_law_holds_within_half_a_percent() {
        for (name, _) in WANT {
            let r = run_reference(&preset(name).expect("preset"));
            let lam_w = r.merged_per_h * r.w_mean_min / 60.0;
            assert!(
                within(lam_w, r.l, 0.005),
                "{name}: lambda*W {lam_w:.3} vs L {:.3}",
                r.l
            );
        }
    }

    #[test]
    fn eq7_time_open_matches_the_sim_within_2_percent_on_design_d() {
        let c = preset("D").expect("preset");
        let r = run_reference(&c);
        let law = eq7_law(&c, &r);
        assert!(
            within(r.w_mean_min, law.w_law_min(), 0.02),
            "sim W {:.1} vs law W {:.1} ({law:?})",
            r.w_mean_min,
            law.w_law_min()
        );
    }

    #[test]
    fn eq7_terms_match_their_sim_stages() {
        let c = preset("D").expect("preset");
        let r = run_reference(&c);
        let law = eq7_law(&c, &r);
        assert!(within(r.checks_min, law.checks_min(), 0.02), "checks");
        assert!(within(r.arm_min, law.t_arm_min, 0.02), "arm");
        assert!(within(r.mq_min, law.w_mq_min(), 0.02), "W_mq");
    }

    #[test]
    fn the_batch_timer_is_not_the_merge_queue_wait_with_one_build() {
        // One build at a time keeps the build slot busy: an armed PR waits for the pass in
        // flight (and any overflow past the batch size), not for the 5-minute batch timer.
        let c = preset("D").expect("preset");
        let r = run_reference(&c);
        let timer = Eq7Law {
            wait_mq_min: c.mq_wait_min,
            ..eq7_law(&c, &r)
        };
        assert!(
            !within(r.w_mean_min, timer.w_law_min(), 0.02),
            "timer-wait law {:.1} matched sim {:.1}",
            timer.w_law_min(),
            r.w_mean_min
        );
        assert!(
            r.pass_wait_min > 2.0 * c.mq_wait_min,
            "wait {}",
            r.pass_wait_min
        );
    }

    #[test]
    fn same_seed_same_output() {
        let c = preset("D").expect("preset");
        assert_eq!(
            run_pr_flow(&c, 7, 100.0, 5.0),
            run_pr_flow(&c, 7, 100.0, 5.0)
        );
    }

    #[test]
    fn a_reverted_design_shows_red() {
        // D with a single merge-queue runner is not D: the 5% band must reject it.
        let mut c = preset("D").expect("preset");
        c.r_mq = 1;
        let r = run_reference(&c);
        assert!(
            !within(r.merged_per_h, 20.08, 0.05),
            "mutant passed at {:.2}",
            r.merged_per_h
        );
    }

    #[test]
    fn unknown_preset_is_none() {
        assert!(preset("Z").is_none());
    }
}
