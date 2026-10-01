//! Spike 011 — Prophet external regressors: does the shipped design-matrix path reach
//! Prophet 1.4.0 parity, and does it need restructuring to get there?
//!
//! Ladder (the CONVENTIONS pattern): rung 1 data prep exact -> rung 3 predict + components
//! at Python MAP -> rung 4 fit. Rung 2 (objective at MAP) needs the oracle's -lp, which
//! these fixtures do not publish; rung 5 (bands) is regressor-independent.

mod regressors;

use aprender_forecast::dates::days_from_civil;
use aprender_forecast::fit::fit_prophet;
use aprender_forecast::prophet::{
    columns, make_design, Holiday, Mode, Params, Seasonality, Spec,
};
use regressors::{predict_points, splice, standardize_one, RegressorSpec, Standardized};
use serde_json::Value;

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("f64"))
        .collect()
}
fn day(s: &str) -> i64 {
    let b = s.as_bytes();
    let y: i64 = s[0..4].parse().expect("y");
    let m: u32 = s[5..7].parse().expect("m");
    let d: u32 = s[8..10].parse().expect("d");
    let _ = b;
    days_from_civil(y, m, d)
}
fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

struct Loaded {
    fx: Value,
    spec: Spec,
    ds_hist: Vec<i64>,
    y: Vec<f64>,
    regs: Vec<Standardized>,
    vals_hist: Vec<Vec<f64>>,
    vals_all: Vec<Vec<f64>>,
    ds_all: Vec<i64>,
}

fn load(path: &str) -> Loaded {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let n = fx["n_history"].as_u64().expect("n") as usize;

    let mut seasonalities = Vec::new();
    for (name, s) in fx["seasonalities"].as_object().expect("seasonalities") {
        seasonalities.push(Seasonality {
            name: name.clone(),
            period: s["period"].as_f64().expect("period"),
            order: s["fourier_order"].as_u64().expect("order") as usize,
            prior_scale: s["prior_scale"].as_f64().expect("ps"),
            mode: if s["mode"] == "multiplicative" { Mode::Multiplicative } else { Mode::Additive },
        });
    }
    let mut spec = Spec::default_linear(seasonalities);
    spec.changepoint_prior_scale = fx["changepoint_prior_scale"].as_f64().expect("cps");

    // Holidays, when the fixture carries them: group the rows by name, as Prophet does.
    if let Some(rows) = fx.get("holidays_requested").and_then(|v| v.as_array()) {
        let mut byname: Vec<(String, Vec<i64>, i64, i64)> = Vec::new();
        for r in rows {
            let nm = r["holiday"].as_str().expect("holiday").to_string();
            let d = day(&r["ds"].as_str().expect("ds")[0..10]);
            let lw = r["lower_window"].as_i64().expect("lw");
            let uw = r["upper_window"].as_i64().expect("uw");
            match byname.iter_mut().find(|e| e.0 == nm) {
                Some(e) => e.1.push(d),
                None => byname.push((nm, vec![d], lw, uw)),
            }
        }
        // Prophet iterates the holiday NAMES in sorted order when building columns; the
        // crate sorts the generated column names, which is the same result.
        byname.sort_by(|a, b| a.0.cmp(&b.0));
        for (nm, days, lw, uw) in byname {
            spec.holidays.push(Holiday { name: nm, days, lower_window: lw, upper_window: uw, prior_scale: 10.0 });
        }
    }

    let ds_all: Vec<i64> = fx["forecast"]["ds"].as_array().expect("fds").iter()
        .map(|v| day(v.as_str().expect("ds"))).collect();
    let ds_hist: Vec<i64> = fx["history"]["ds"].as_array().expect("hds").iter()
        .map(|v| day(v.as_str().expect("ds"))).collect();
    let y = f64s(&fx["history"]["y"]);

    let mut regs = Vec::new();
    let mut vals_hist = Vec::new();
    let mut vals_all = Vec::new();
    for r in fx["regressors_requested"].as_array().expect("regs") {
        let name = r["name"].as_str().expect("name").to_string();
        let all = f64s(&fx["regressor_values"][&name]);
        let spec_r = RegressorSpec {
            name: name.clone(),
            mode: if r["mode"] == "multiplicative" { Mode::Multiplicative } else { Mode::Additive },
            prior_scale: r["prior_scale"].as_f64().expect("ps"),
            standardize: match r["standardize"].as_str() { Some("auto") | None => None, Some("True") => Some(true), _ => Some(false) },
        };
        let hist: Vec<f64> = all[..n].to_vec();
        regs.push(standardize_one(&spec_r, &hist));
        vals_hist.push(hist);
        vals_all.push(all);
    }
    Loaded { fx, spec, ds_hist, y, regs, vals_hist, vals_all, ds_all }
}

fn run(stem: &str, path: &str) -> Value {
    println!("\n## {stem}\n");
    let l = load(path);
    let mut d = make_design(&l.ds_hist, &l.y, &l.spec);
    let base_cols: Vec<String> = columns(&l.spec).iter().map(|c| c.name.clone()).collect();
    splice(&mut d, &l.regs, &l.vals_hist);

    // ---- rung 0: the standardisation constants themselves --------------------
    println!("### rung 0 — standardisation constants (history rows only)\n");
    println!("| regressor | mode | rust mu | python mu | rust std | python std | d mu | d std |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut worst_std = 0.0f64;
    for r in &l.regs {
        let py = &l.fx["extra_regressors"][&r.name];
        let (pmu, pstd) = (py["mu"].as_f64().expect("mu"), py["std"].as_f64().expect("std"));
        let (dmu, dstd) = ((r.mu - pmu).abs(), (r.std - pstd).abs());
        worst_std = worst_std.max(dmu).max(dstd);
        println!("| {} | {:?} | {:.10} | {:.10} | {:.10} | {:.10} | {:.1e} | {:.1e} |",
                 r.name, r.mode, r.mu, pmu, r.std, pstd, dmu, dstd);
    }

    // ---- rung 1: data prep exact ---------------------------------------------
    let py_cols: Vec<String> = l.fx["columns"].as_array().expect("cols").iter()
        .map(|v| v.as_str().expect("s").to_string()).collect();
    let rust_cols: Vec<String> = d.cols.iter().map(|c| c.name.clone()).collect();
    let order_ok = rust_cols == py_cols;
    let d_ps = max_abs(&d.prior_scales, &f64s(&l.fx["prior_scales"]));
    let d_sa = max_abs(&d.s_a, &f64s(&l.fx["s_a"]));
    let d_sm = max_abs(&d.s_m, &f64s(&l.fx["s_m"]));
    let mut d_x = 0.0f64;
    for (blk, rows) in [("X_first3", 0usize), ("X_last3", l.ds_hist.len() - 3)] {
        let py = l.fx[blk].as_array().expect("X");
        for (r, prow) in py.iter().enumerate() {
            let prow = f64s(prow);
            let i = rows + r;
            let rrow = &d.x[i * d.k..(i + 1) * d.k];
            d_x = d_x.max(max_abs(rrow, &prow));
        }
    }
    println!("\n### rung 1 — data prep exact\n");
    println!("| quantity | result |");
    println!("|---|---|");
    println!("| base columns (no regressors) | {} |", base_cols.len());
    println!("| columns with regressors | {} (python {}) |", rust_cols.len(), py_cols.len());
    println!("| column ORDER identical to python | {} |", if order_ok { "yes" } else { "**NO**" });
    if !order_ok {
        println!("| rust order | `{}` |", rust_cols.join(", "));
        println!("| python order | `{}` |", py_cols.join(", "));
    }
    println!("| max abs d prior_scales | {d_ps:.2e} |");
    println!("| max abs d s_a | {d_sa:.2e} |");
    println!("| max abs d s_m | {d_sm:.2e} |");
    println!("| max abs d X (first3 + last3 rows) | {d_x:.2e} |");
    println!("| max abs d standardisation constants | {worst_std:.2e} |");

    // ---- rung 3: predict + components at Python MAP --------------------------
    let pp = &l.fx["params"];
    let p = Params {
        k: f64s(&pp["k"])[0],
        m: f64s(&pp["m"])[0],
        delta: f64s(&pp["delta"]),
        beta: f64s(&pp["beta"]),
        sigma_obs: f64s(&pp["sigma_obs"])[0],
    };
    let (yhat, trend, comps) = predict_points(&d, &p, &l.ds_all, &l.regs, &l.vals_all);
    let d_yhat = max_abs(&yhat, &f64s(&l.fx["forecast"]["yhat"]));
    let d_trend = max_abs(&trend, &f64s(&l.fx["forecast"]["trend"]));
    let y_scale = l.fx["y_scale"].as_f64().expect("ys");
    println!("\n### rung 3 — predict at Python MAP (y_scale {y_scale:.3})\n");
    println!("| quantity | max abs diff | relative to y_scale |");
    println!("|---|---|---|");
    println!("| yhat | {d_yhat:.2e} | {:.2e} |", d_yhat / y_scale);
    println!("| trend | {d_trend:.2e} | {:.2e} |", d_trend / y_scale);

    let py_comps = l.fx["forecast"]["components"].as_object().expect("components");
    println!("\n| component | max abs diff | compared |");
    println!("|---|---|---|");
    let mut checked = 0;
    let mut worst_comp = 0.0f64;
    let mut names: Vec<&String> = py_comps.keys().collect();
    names.sort();
    for nm in names {
        if let Some((_, v)) = comps.iter().find(|(n, _)| n == nm) {
            let dd = max_abs(v, &f64s(&py_comps[nm]));
            worst_comp = worst_comp.max(dd);
            checked += 1;
            println!("| {nm} | {dd:.2e} | yes |");
        } else {
            println!("| {nm} | — | **MISSING from rust** |");
        }
    }
    println!("\n{checked} components compared, worst {worst_comp:.2e}");

    // ---- rung 4: fit ----------------------------------------------------------
    let t0 = std::time::Instant::now();
    let (fitp, info) = fit_prophet(&d, 8);
    let secs = t0.elapsed().as_secs_f64();
    let (fy, _ft, _fc) = predict_points(&d, &fitp, &l.ds_all, &l.regs, &l.vals_all);
    let d_fit_yhat = max_abs(&fy, &f64s(&l.fx["forecast"]["yhat"]));
    // The CONTROL for rung 4: an optimiser that lands on a different beta is only a
    // defect if it lands on a WORSE objective. The crate scales by 1/T; the contract's
    // 0.5 bar is on the unscaled objective, so both are reported.
    let model = aprender_forecast::prophet::Model::new(&d);
    let t_n = l.ds_hist.len() as f64;
    let f_py = model.objective(&model.pack(&p));
    let f_rs = model.objective(&model.pack(&fitp));
    let slack = (f_rs - f_py) * t_n;

    // Is each regressor column identifiable, or collinear with the trend / seasonality
    // basis already in X? Pearson r against t and against the strongest existing column.
    let corr = |a: &[f64], b: &[f64]| -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let mut num = 0.0; let mut da = 0.0; let mut db = 0.0;
        for i in 0..a.len() { let (x, y) = (a[i] - ma, b[i] - mb); num += x * y; da += x * x; db += y * y; }
        if da == 0.0 || db == 0.0 { 0.0 } else { num / (da.sqrt() * db.sqrt()) }
    };

    println!("\n### rung 4 — Rust MAP fit with regressor columns\n");
    println!("| quantity | value |");
    println!("|---|---|");
    println!("| status | {} |", info.status);
    println!("| rounds / iters | {} / {} |", info.rounds, info.iterations);
    println!("| fit seconds | {secs:.2} |");
    println!("| max abs d yhat vs python forecast | {d_fit_yhat:.4} ({:.2}% of y_scale) |", 100.0 * d_fit_yhat / y_scale);
    println!("| objective at python MAP (unscaled) | {:.4} |", f_py * t_n);
    println!("| objective at rust fit (unscaled) | {:.4} |", f_rs * t_n);
    println!("| **slack f_rust - f_python** | **{slack:+.4}** (contract bar 0.5; negative = rust found a BETTER optimum) |");
    let py_beta = f64s(&pp["beta"]);
    let mut ident_rows: Vec<Value> = Vec::new();
    println!("\n| regressor | python beta | rust beta | r vs t (trend) | max r vs other X col | identifiable? |");
    println!("|---|---|---|---|---|---|");
    for (j, r) in l.regs.iter().enumerate() {
        let c = d.k - l.regs.len() + j;
        let colj: Vec<f64> = (0..l.ds_hist.len()).map(|i| d.x[i * d.k + c]).collect();
        let r_t = corr(&colj, &d.t);
        let mut r_other: f64 = 0.0;
        let mut which = String::new();
        for c2 in 0..d.k {
            if c2 == c { continue; }
            let col2: Vec<f64> = (0..l.ds_hist.len()).map(|i| d.x[i * d.k + c2]).collect();
            let rr = corr(&colj, &col2);
            if rr.abs() > r_other.abs() { r_other = rr; which = d.cols[c2].name.clone(); }
        }
        let ident = if r_t.abs() > 0.9 || r_other.abs() > 0.9 { "**no — collinear**" } else { "yes" };
        println!("| {} | {:+.6} | {:+.6} | {:+.3} | {:+.3} (`{}`) | {} |",
                 r.name, py_beta[c], fitp.beta[c], r_t, r_other, which, ident);
        ident_rows.push(serde_json::json!({
            "name": r.name, "mode": format!("{:?}", r.mode),
            "py_beta": py_beta[c], "rust_beta": fitp.beta[c],
            "r_trend": r_t, "r_other": r_other, "r_other_col": which,
            "collinear": r_t.abs() > 0.9 || r_other.abs() > 0.9,
        }));
    }

    let comp_json: Vec<Value> = comps.iter().map(|(n, v)| {
        let py = py_comps.get(n).map(|x| f64s(x));
        serde_json::json!({ "name": n, "rust": v, "python": py })
    }).collect();
    serde_json::json!({
        "stem": stem,
        "ds": l.fx["forecast"]["ds"],
        "n_history": l.ds_hist.len(),
        "y": l.fx["history"]["y"],
        "y_scale": y_scale,
        "py_yhat": l.fx["forecast"]["yhat"],
        "rust_yhat_at_py_params": yhat,
        "rust_yhat_fit": fy,
        "components": comp_json,
        "identifiability": ident_rows,
        "rungs": {
            "std_consts": worst_std, "col_order_ok": order_ok,
            "d_prior_scales": d_ps, "d_s_a": d_sa, "d_s_m": d_sm, "d_x": d_x,
            "d_yhat": d_yhat, "d_trend": d_trend, "worst_component": worst_comp,
            "n_components": checked, "n_cols": rust_cols.len(),
            "slack": slack, "fit_status": info.status.clone(), "fit_seconds": secs,
        }
    })
}

fn main() {
    println!("# Spike 011 — Prophet external regressors, parity ladder");
    let a = run("retail_sales + 4 regressors", "fixtures/retail_regressors_prophet140.json");
    let b = run("retail_sales + 2 holidays (windows) + 4 regressors", "fixtures/retail_regressors_holidays_prophet140.json");
    let out = serde_json::json!({ "regressors_only": a, "regressors_and_holidays": b });
    std::fs::write("results.json", serde_json::to_string(&out).expect("ser")).expect("write results.json");
}
