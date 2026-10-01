//! Spike 012 — is a delivery that ADDS optional arguments byte-identical when the caller
//! passes none? That is Forecast Coach's acceptance gate for a one-line tag bump.
//!
//! Three parts, because a baseline alone proves nothing:
//!   A. capture a bit-exact signature per case through the PUBLIC door;
//!   B. mutation-prove the signature actually detects a change (a guard that cannot fail
//!      is theatre — CLAUDE.md verification discipline 5 and 7);
//!   C. the mechanism test: run the spike-011 regressor plumbing with ZERO regressors and
//!      compare against the crate's own untouched `predict`, bit for bit.

#[allow(dead_code)]
mod regressors;
mod sig;

use aprender_forecast::dates::days_from_civil;
use aprender_forecast::prophet::{make_design, predict, Mode, Params, Seasonality, Spec};
use aprender_forecast::{forecast, ForecastArgs};
use regressors::{predict_points, splice, Standardized};
use sig::signature;

const FIX: &str = "../../../crates/aprender-forecast/tests/fixtures";

fn csv(name: &str) -> (Vec<String>, Vec<f64>) {
    let text = std::fs::read_to_string(format!("{FIX}/{name}")).expect("csv");
    let mut rows: Vec<(String, f64)> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split(',');
        // The fixture CSVs quote their fields ("2007-12-10"); strip before parsing.
        let ds = it.next().expect("ds").trim().trim_matches('"').to_string();
        let y: f64 = match it.next().map(|v| v.trim().trim_matches('"').parse::<f64>()) {
            Some(Ok(v)) => v,
            _ => continue,
        };
        rows.push((ds[..10].to_string(), y));
    }
    // CONVENTIONS: sort by ds and de-duplicate at load (wp_log_R.csv is not chronological).
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.dedup_by(|a, b| a.0 == b.0);
    (rows.iter().map(|r| r.0.clone()).collect(), rows.iter().map(|r| r.1).collect())
}

fn base(ds: Vec<String>, y: Vec<f64>, horizon: usize) -> ForecastArgs {
    ForecastArgs {
        ds, y, horizon,
        freq: None, model: None, growth: None, cap: None, seasonality_mode: None,
        interval_width: None, holidays: None, n_lags: None, seed: Some(42),
    }
}

fn cases() -> Vec<(String, ForecastArgs)> {
    let mut out: Vec<(String, ForecastArgs)> = Vec::new();
    let (pd, py) = csv("peyton_manning.csv");
    let (ad, ay) = csv("air_passengers.csv");
    let (rd, ry) = csv("retail_sales.csv");
    let (wd, wy) = csv("wp_log_R.csv");

    out.push(("peyton/prophet/default".into(), base(pd.clone(), py.clone(), 30)));

    let mut a = base(ad.clone(), ay.clone(), 12);
    a.freq = Some("MS".into());
    a.seasonality_mode = Some("multiplicative".into());
    out.push(("air/prophet/multiplicative".into(), a));

    let mut r = base(rd.clone(), ry.clone(), 12);
    r.freq = Some("MS".into());
    out.push(("retail/prophet/default".into(), r));

    let mut w = base(wd.clone(), wy.clone(), 30);
    w.growth = Some("logistic".into());
    w.cap = Some(wy.iter().fold(f64::MIN, |m, v| m.max(*v)) * 1.2);
    out.push(("wp_log_R/prophet/logistic".into(), w));

    let mut h = base(pd.clone(), py.clone(), 30);
    h.holidays = Some(vec![aprender_forecast::HolidayArg {
        name: "playoff".into(),
        dates: vec!["2010-01-16".into(), "2014-01-12".into(), "2016-01-17".into()],
        lower_window: -1,
        upper_window: 1,
    }]);
    out.push(("peyton/prophet/holidays+windows".into(), h));

    let mut n = base(pd.clone(), py.clone(), 30);
    n.model = Some("neuralprophet".into());
    n.n_lags = Some(0);
    out.push(("peyton/neuralprophet/lag0".into(), n));

    let mut n7 = base(pd, py, 30);
    n7.model = Some("neuralprophet".into());
    n7.n_lags = Some(7);
    out.push(("peyton/neuralprophet/lag7".into(), n7));

    // NOT retail/MS: `forecast.rs:421` refuses `freq != "D"` for neuralprophet
    // (test `neuralprophet_refuses_non_daily_freq`). A second DAILY series instead.
    let _ = (rd, ry);
    let mut nw = base(wd, wy, 30);
    nw.model = Some("neuralprophet".into());
    nw.n_lags = Some(0);
    out.push(("wp_log_R/neuralprophet/lag0".into(), nw));
    out
}

fn day(s: &str) -> i64 {
    days_from_civil(s[0..4].parse().expect("y"), s[5..7].parse().expect("m"), s[8..10].parse().expect("d"))
}

fn main() {
    println!("# Spike 012 — no-argument bitwise invariance\n");

    // ---- A. signature baseline, twice, through the public door -----------------
    println!("## A. Baseline: same input twice through `forecast()` (determinism)\n");
    println!("| case | signature | repeat | identical | fit s |");
    println!("|---|---|---|---|---|");
    let mut base_sigs: Vec<(String, u64)> = Vec::new();
    let mut all_same = true;
    for (name, args) in cases() {
        let r1 = forecast(&args).expect("forecast");
        let s1 = signature(&r1);
        let r2 = forecast(&args).expect("forecast");
        let s2 = signature(&r2);
        let same = s1 == s2;
        all_same &= same;
        println!("| {name} | `{s1:016x}` | `{s2:016x}` | {} | {:.2} |",
                 if same { "yes" } else { "**NO**" }, r1.fit_seconds);
        base_sigs.push((name, s1));
    }
    println!("\nAll repeat calls identical: **{}**", if all_same { "yes" } else { "NO" });

    // ---- B. mutation proof: can this signature FAIL? ---------------------------
    println!("\n## B. Mutation proof — the signature detects a 1-ULP change\n");
    let (_, args) = cases().remove(0);
    let mut r = forecast(&args).expect("forecast");
    let clean = signature(&r);
    println!("| mutation | signature | detected |");
    println!("|---|---|---|");
    let orig = r.yhat[0];
    r.yhat[0] = f64::from_bits(orig.to_bits() + 1); // one ULP up
    let m1 = signature(&r);
    println!("| yhat[0] += 1 ULP ({orig:.17e} -> {:.17e}) | `{m1:016x}` | {} |",
             r.yhat[0], if m1 != clean { "**yes**" } else { "NO — VACUOUS" });
    r.yhat[0] = orig;
    let last = r.trend.len() - 1;
    let ot = r.trend[last];
    r.trend[last] = f64::from_bits(ot.to_bits() + 1);
    let m2 = signature(&r);
    println!("| trend[last] += 1 ULP | `{m2:016x}` | {} |", if m2 != clean { "**yes**" } else { "NO — VACUOUS" });
    r.trend[last] = ot;
    r.components.insert("phantom".into(), serde_json::json!([0.0]));
    let m3 = signature(&r);
    println!("| one extra component key | `{m3:016x}` | {} |", if m3 != clean { "**yes**" } else { "NO — VACUOUS" });
    r.components.remove("phantom");
    let restored = signature(&r);
    println!("| all mutations reverted | `{restored:016x}` | {} |",
             if restored == clean { "back to clean" } else { "**NOT RESTORED**" });

    // ---- C. the mechanism: is the regressor plumbing inert at zero regressors? --
    println!("\n## C. Mechanism — spike-011 regressor plumbing with ZERO regressors\n");
    println!("Compares the crate's untouched `predict` against the prototype path that carries");
    println!("the regressor machinery but is handed an empty regressor list.\n");
    println!("| dataset | design identical | yhat bits | trend bits | components | verdict |");
    println!("|---|---|---|---|---|---|");
    for (file, horizon) in [("peyton_manning.csv", 30usize), ("retail_sales.csv", 12), ("air_passengers.csv", 12)] {
        let (ds, y) = csv(file);
        let ds_days: Vec<i64> = ds.iter().map(|s| day(s)).collect();
        let seas = vec![Seasonality { name: "yearly".into(), period: 365.25, order: 10, prior_scale: 10.0, mode: Mode::Additive }];
        let spec = Spec::default_linear(seas);

        let d_plain = make_design(&ds_days, &y, &spec);
        let mut d_spliced = make_design(&ds_days, &y, &spec);
        let none: Vec<Standardized> = Vec::new();
        let novals: Vec<Vec<f64>> = Vec::new();
        splice(&mut d_spliced, &none, &novals);

        let design_same = d_plain.k == d_spliced.k
            && d_plain.cols.len() == d_spliced.cols.len()
            && d_plain.x.iter().zip(&d_spliced.x).all(|(a, b)| a.to_bits() == b.to_bits())
            && d_plain.s_a.iter().zip(&d_spliced.s_a).all(|(a, b)| a.to_bits() == b.to_bits())
            && d_plain.s_m.iter().zip(&d_spliced.s_m).all(|(a, b)| a.to_bits() == b.to_bits())
            && d_plain.prior_scales.iter().zip(&d_spliced.prior_scales).all(|(a, b)| a.to_bits() == b.to_bits());

        let mut future = ds_days.clone();
        let step = ds_days[ds_days.len() - 1] - ds_days[ds_days.len() - 2];
        for i in 1..=horizon {
            future.push(ds_days[ds_days.len() - 1] + step * i as i64);
        }
        let (p, _info) = aprender_forecast::fit::fit_prophet(&d_plain, 8);
        let params = Params { k: p.k, m: p.m, delta: p.delta.clone(), beta: p.beta.clone(), sigma_obs: p.sigma_obs };

        let f_plain = predict(&d_plain, &params, &future, 42);
        let (yh, tr, comps) = predict_points(&d_spliced, &params, &future, &none, &novals);

        let yhat_bits = f_plain.yhat.iter().zip(&yh).all(|(a, b)| a.to_bits() == b.to_bits());
        let trend_bits = f_plain.trend.iter().zip(&tr).all(|(a, b)| a.to_bits() == b.to_bits());
        let mut comp_ok = true;
        let mut comp_n = 0;
        for (n, v) in &f_plain.components {
            if let Some((_, w)) = comps.iter().find(|(m, _)| m == n) {
                comp_n += 1;
                if !v.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()) {
                    comp_ok = false;
                }
            }
        }
        let ok = design_same && yhat_bits && trend_bits && comp_ok;
        println!("| {file} | {} | {} | {} | {comp_n} identical: {} | {} |",
                 if design_same { "yes" } else { "**NO**" },
                 if yhat_bits { "identical" } else { "**DIFFER**" },
                 if trend_bits { "identical" } else { "**DIFFER**" },
                 if comp_ok { "yes" } else { "**NO**" },
                 if ok { "INERT ✓" } else { "**NOT INERT**" });
    }
}
