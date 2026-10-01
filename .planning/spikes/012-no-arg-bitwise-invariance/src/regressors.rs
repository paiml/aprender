//! The prototype under test: external regressors spliced onto the shipped `Design`.
//!
//! Nothing here is a copy of `prophet.rs`. The point of the spike is to find out whether
//! the SHIPPED types admit regressors without restructuring, so the base design comes from
//! `make_design` and the base feature rows from `feature_row`; only the regressor columns
//! are new code.

use aprender_forecast::prophet::{
    feature_row, holiday_day_sets, Column, Design, Mode, Model, Params,
};

#[derive(Clone, Debug)]
pub struct RegressorSpec {
    pub name: String,
    pub mode: Mode,
    pub prior_scale: f64,
    /// `None` = Prophet's "auto".
    pub standardize: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct Standardized {
    pub name: String,
    pub mu: f64,
    pub std: f64,
    pub mode: Mode,
    pub prior_scale: f64,
}

/// Prophet 1.4 `initialize_scales`: over the HISTORY rows only,
/// `auto` means "do not standardize a column whose unique values are exactly {0, 1}",
/// and the spread is pandas `Series.std()` — the SAMPLE standard deviation, ddof = 1.
///
/// The ddof is the trap: numpy's default is ddof = 0, which on this fixture gives
/// 6.500320 where Prophet stores 6.511442.
#[must_use]
pub fn standardize_one(spec: &RegressorSpec, history: &[f64]) -> Standardized {
    let binary = {
        let mut seen_other = false;
        let mut seen: Vec<f64> = Vec::new();
        for &v in history {
            if v != 0.0 && v != 1.0 {
                seen_other = true;
                break;
            }
            if !seen.contains(&v) {
                seen.push(v);
            }
        }
        !seen_other && seen.len() == 2
    };
    let do_std = spec.standardize.unwrap_or(!binary);
    let (mu, std) = if do_std {
        let n = history.len() as f64;
        let mu = history.iter().sum::<f64>() / n;
        // ddof = 1
        let var = history.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0);
        (mu, var.sqrt())
    } else {
        (0.0, 1.0)
    };
    Standardized {
        name: spec.name.clone(),
        mu,
        std,
        mode: spec.mode,
        prior_scale: spec.prior_scale,
    }
}

/// Append one column per regressor, in INSERTION order, after the seasonality and holiday
/// columns the crate already emits. Measured against Prophet 1.4.0: the oracle's column
/// order is seasonalities -> holidays (name-sorted) -> extra regressors (insertion order),
/// so regressors never reorder an existing column.
pub fn splice(d: &mut Design, regs: &[Standardized], values_history: &[Vec<f64>]) {
    let base_k = d.k;
    let t = d.t.len();
    let r = regs.len();
    if r == 0 {
        return;
    }
    let new_k = base_k + r;
    let mut x = vec![0.0f64; t * new_k];
    for i in 0..t {
        x[i * new_k..i * new_k + base_k].copy_from_slice(&d.x[i * base_k..(i + 1) * base_k]);
        for (j, reg) in regs.iter().enumerate() {
            x[i * new_k + base_k + j] = (values_history[j][i] - reg.mu) / reg.std;
        }
    }
    for reg in regs {
        d.cols.push(Column {
            name: reg.name.clone(),
            component: reg.name.clone(),
            mode: reg.mode,
            prior_scale: reg.prior_scale,
            holiday: None,
        });
        d.prior_scales.push(reg.prior_scale);
        d.s_a.push(if reg.mode == Mode::Additive { 1.0 } else { 0.0 });
        d.s_m.push(if reg.mode == Mode::Multiplicative { 1.0 } else { 0.0 });
    }
    d.x = x;
    d.k = new_k;
}

/// A regressor-aware `predict`, point estimates only (the uncertainty simulation is
/// untouched by regressors and is rung 5's business, not this spike's).
///
/// This is the function that answers the CR's architectural worry. `feature_row` is keyed
/// on `day`; a regressor value is NOT a function of the day, so the base rows come from
/// `feature_row` and the regressor cells are written beside them from the caller's array.
#[must_use]
pub fn predict_points(
    d: &Design,
    p: &Params,
    ds_days: &[i64],
    regs: &[Standardized],
    values_all: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>, Vec<(String, Vec<f64>)>) {
    let n = ds_days.len();
    let base_k = d.k - regs.len();
    let base_cols = &d.cols[..base_k];
    let t: Vec<f64> = ds_days
        .iter()
        .map(|&x| (x - d.start_days) as f64 / d.t_scale_days)
        .collect();
    let cap: Option<Vec<f64>> = d.cap_scaled.as_ref().map(|c| vec![c[0]; n]);
    let model = Model::new(d);
    let trend_s = model.trend(p, &t, cap.as_deref());

    let hol_sets = holiday_day_sets(&d.spec);
    let mut x = vec![0.0f64; n * d.k];
    let mut row: Vec<f64> = Vec::with_capacity(base_k);
    for (i, &day) in ds_days.iter().enumerate() {
        row.clear();
        feature_row(day, &d.spec, base_cols, &hol_sets, &mut row);
        x[i * d.k..i * d.k + base_k].copy_from_slice(&row);
        for (j, reg) in regs.iter().enumerate() {
            x[i * d.k + base_k + j] = (values_all[j][i] - reg.mu) / reg.std;
        }
    }

    let sum_over = |sel: &dyn Fn(&Column) -> bool, i: usize| -> f64 {
        let mut v = 0.0;
        for c in 0..d.k {
            if sel(&d.cols[c]) {
                v += x[i * d.k + c] * p.beta[c];
            }
        }
        v
    };
    let series = |sel: &dyn Fn(&Column) -> bool, additive: bool| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let v = sum_over(sel, i);
                if additive {
                    v * d.y_scale
                } else {
                    v
                }
            })
            .collect()
    };

    let mut components: Vec<(String, Vec<f64>)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for c in &d.cols {
        if !names.contains(&c.component) {
            names.push(c.component.clone());
        }
    }
    for nm in &names {
        let mode = d
            .cols
            .iter()
            .find(|c| &c.component == nm)
            .expect("col")
            .mode;
        let nm2 = nm.clone();
        components.push((
            nm.clone(),
            series(&move |c: &Column| c.component == nm2, mode == Mode::Additive),
        ));
    }
    // Prophet publishes the holiday roll-up and the two extra-regressor roll-ups by name.
    let reg_names: Vec<String> = regs.iter().map(|r| r.name.clone()).collect();
    let rn_a = reg_names.clone();
    let rn_m = reg_names.clone();
    components.push((
        "extra_regressors_additive".into(),
        series(
            &move |c: &Column| c.mode == Mode::Additive && rn_a.contains(&c.name),
            true,
        ),
    ));
    components.push((
        "extra_regressors_multiplicative".into(),
        series(
            &move |c: &Column| c.mode == Mode::Multiplicative && rn_m.contains(&c.name),
            false,
        ),
    ));
    let additive = series(&|c: &Column| c.mode == Mode::Additive, true);
    let multiplicative = series(&|c: &Column| c.mode == Mode::Multiplicative, false);
    if d.spec.holidays.iter().len() > 0 {
        components.push((
            "holidays".into(),
            series(&|c: &Column| c.holiday.is_some(), true),
        ));
    }
    components.push(("additive_terms".into(), additive.clone()));
    components.push(("multiplicative_terms".into(), multiplicative.clone()));

    let trend: Vec<f64> = trend_s.iter().map(|v| v * d.y_scale).collect();
    let yhat: Vec<f64> = (0..n)
        .map(|i| trend[i] * (1.0 + multiplicative[i]) + additive[i])
        .collect();
    (yhat, trend, components)
}
