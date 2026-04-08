//! aprender-monte-carlo CLI
//!
//! Monte Carlo simulation tool for financial modeling and risk analysis.

use aprender_monte_carlo::{
    cli::{Cli, Commands, OutputFormat},
    data::{CsvLoader, Sp500Data, Sp500Period},
    models::{BayesianRevenueModel, ProductData},
    prelude::*,
};
use std::error::Error;
use std::path::Path;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let cli = Cli::parse_args();

    match cli.command {
        Commands::Sp500 {
            years,
            simulations,
            initial,
            withdrawal_rate,
            real_returns,
        } => {
            run_sp500_simulation(
                years,
                simulations,
                initial,
                withdrawal_rate,
                real_returns,
                cli.seed,
                cli.format,
                cli.verbose,
            )?;
        }
        Commands::Csv {
            file,
            column,
            years,
            simulations,
            initial,
        } => {
            run_csv_simulation(
                &file,
                column.as_deref(),
                years,
                simulations,
                initial,
                cli.seed,
                cli.format,
                cli.verbose,
            )?;
        }
        Commands::Revenue {
            file,
            quarters,
            simulations,
            bayesian,
        } => {
            run_revenue_simulation(
                &file,
                quarters,
                simulations,
                bayesian,
                cli.seed,
                cli.format,
                cli.verbose,
            )?;
        }
        Commands::Stats { monthly, decades } => {
            show_sp500_stats(monthly, decades);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_sp500_simulation(
    years: u32,
    simulations: usize,
    initial: f64,
    withdrawal_rate: Option<f64>,
    real_returns: bool,
    seed: u64,
    format: OutputFormat,
    verbose: bool,
) -> std::result::Result<(), Box<dyn Error>> {
    if verbose {
        println!("Loading S&P 500 historical data...");
    }

    let sp500 = Sp500Data::load();
    let returns = sp500.monthly_returns(Sp500Period::All, real_returns);

    if verbose {
        let stats = sp500.statistics(Sp500Period::All, real_returns);
        println!("{stats}");
    }

    // Create bootstrap model
    let model = EmpiricalBootstrap::new(initial, returns);

    // Configure engine
    let engine = MonteCarloEngine::reproducible(seed)
        .with_n_simulations(simulations)
        .with_variance_reduction(VarianceReduction::Antithetic);

    if verbose {
        println!("Running {simulations} simulations over {years} years...");
    }

    // Run simulation
    let horizon = TimeHorizon::years(years).with_step(TimeStep::Monthly);
    let result = engine.simulate(&model, &horizon);

    // Generate risk report
    let risk_free_rate = 0.02; // 2% annual, 0.167% monthly
    let report = RiskReport::from_paths(&result.paths, risk_free_rate / 12.0)
        .map_err(|e| format!("Failed to generate risk report: {e}"))?;

    // Handle withdrawal scenarios
    let final_values = if let Some(rate) = withdrawal_rate {
        // Simulate withdrawals
        let monthly_withdrawal = initial * rate / 12.0;
        result
            .paths
            .iter()
            .map(|p| {
                let mut value = initial;
                for (i, &ret) in p.values.iter().skip(1).enumerate() {
                    let growth = ret / p.values.get(i).unwrap_or(&initial);
                    value = (value - monthly_withdrawal) * growth;
                    if value < 0.0 {
                        return 0.0;
                    }
                }
                value
            })
            .collect::<Vec<_>>()
    } else {
        result.final_values()
    };

    let final_stats = Statistics::from_values(&final_values);

    // Output results
    output_simulation_results(
        &final_stats,
        &report,
        simulations,
        years,
        initial,
        withdrawal_rate,
        format,
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_csv_simulation(
    file: &Path,
    column: Option<&str>,
    years: u32,
    simulations: usize,
    initial: f64,
    seed: u64,
    format: OutputFormat,
    verbose: bool,
) -> std::result::Result<(), Box<dyn Error>> {
    if verbose {
        println!("Loading data from {}...", file.display());
    }

    let loader = CsvLoader::load(file, column)?;

    if verbose {
        println!(
            "Loaded {} observations from column '{}'",
            loader.n_rows, loader.column_name
        );
        println!("{}", loader.stats());
    }

    let model = EmpiricalBootstrap::new(initial, loader.returns);
    let engine = MonteCarloEngine::reproducible(seed)
        .with_n_simulations(simulations)
        .with_variance_reduction(VarianceReduction::Antithetic);

    if verbose {
        println!("Running {simulations} simulations over {years} years...");
    }

    let horizon = TimeHorizon::years(years);
    let result = engine.simulate(&model, &horizon);

    let report = RiskReport::from_paths(&result.paths, 0.0)
        .map_err(|e| format!("Failed to generate risk report: {e}"))?;

    let final_stats = result.final_value_statistics();

    output_simulation_results(
        &final_stats,
        &report,
        simulations,
        years,
        initial,
        None,
        format,
    );

    Ok(())
}

fn run_revenue_simulation(
    file: &Path,
    quarters: u32,
    simulations: usize,
    _bayesian: bool,
    seed: u64,
    format: OutputFormat,
    verbose: bool,
) -> std::result::Result<(), Box<dyn Error>> {
    if verbose {
        println!("Loading product data from {}...", file.display());
    }

    // For now, use a simple example if file doesn't exist
    let products = if file.exists() {
        // Try to load from CSV
        let loader = CsvLoader::load(file, Some("revenue"))?;
        vec![ProductData::new(
            "Portfolio",
            loader.returns.first().copied().unwrap_or(100_000.0),
            loader.stats().mean,
            loader.stats().std,
        )]
    } else {
        // Use example products
        if verbose {
            println!("File not found, using example products...");
        }
        vec![
            ProductData::new("Widget", 100_000.0, 0.15, 0.20),
            ProductData::new("Gadget", 50_000.0, 0.25, 0.30),
        ]
    };

    let model = BayesianRevenueModel::new(products.clone());
    let engine = MonteCarloEngine::reproducible(seed)
        .with_n_simulations(simulations)
        .with_variance_reduction(VarianceReduction::Antithetic);

    if verbose {
        println!("Products:");
        for p in &products {
            println!(
                "  {}: ${:.0} base, {:.1}% growth, {:.1}% volatility",
                p.name,
                p.base_revenue,
                p.growth_rate * 100.0,
                p.volatility * 100.0
            );
        }
        println!("Running {simulations} simulations over {quarters} quarters...");
    }

    let horizon = TimeHorizon::quarters(quarters);
    let result = engine.simulate(&model, &horizon);

    let report = RiskReport::from_paths(&result.paths, 0.0)
        .map_err(|e| format!("Failed to generate risk report: {e}"))?;

    let final_stats = result.final_value_statistics();
    let initial = model.total_base_revenue();

    output_revenue_results(
        &final_stats,
        &report,
        simulations,
        quarters,
        initial,
        format,
    );

    Ok(())
}

fn show_sp500_stats(monthly: bool, decades: bool) {
    let sp500 = Sp500Data::load();

    println!("S&P 500 Historical Data Summary");
    println!("================================");
    println!("Total months: {}", sp500.len());
    println!();

    let all_stats = sp500.statistics(Sp500Period::All, false);
    println!("Overall (Nominal):");
    println!("{all_stats}");

    let real_stats = sp500.statistics(Sp500Period::All, true);
    println!("Overall (Real/Inflation-Adjusted):");
    println!("{real_stats}");

    if monthly {
        println!("\nMonthly Return Percentiles:");
        let returns = sp500.monthly_returns(Sp500Period::All, false);
        let p5 = percentile(&returns, 0.05);
        let p25 = percentile(&returns, 0.25);
        let p50 = percentile(&returns, 0.50);
        let p75 = percentile(&returns, 0.75);
        let p95 = percentile(&returns, 0.95);
        println!("  5th:  {:.2}%", p5 * 100.0);
        println!("  25th: {:.2}%", p25 * 100.0);
        println!("  50th: {:.2}%", p50 * 100.0);
        println!("  75th: {:.2}%", p75 * 100.0);
        println!("  95th: {:.2}%", p95 * 100.0);
    }

    if decades {
        println!("\nDecade-by-Decade Performance:");
        println!(
            "{:<8} {:>12} {:>12} {:>12}",
            "Decade", "Return", "Volatility", "MaxDD"
        );
        println!("{:-<8} {:-<12} {:-<12} {:-<12}", "", "", "", "");

        for (label, stats) in sp500.decade_stats(false) {
            println!(
                "{:<8} {:>11.2}% {:>11.2}% {:>11.2}%",
                label,
                stats.annual_mean * 100.0,
                stats.annual_std * 100.0,
                stats.max_drawdown * 100.0
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)] // serde_json::json! macro uses unwrap internally
fn output_simulation_results(
    stats: &Statistics,
    report: &RiskReport,
    simulations: usize,
    years: u32,
    initial: f64,
    withdrawal_rate: Option<f64>,
    format: OutputFormat,
) {
    match format {
        OutputFormat::Table => {
            println!("\nMonte Carlo Simulation Results");
            println!("==============================");
            println!("Simulations: {simulations}");
            println!("Horizon:     {years} years");
            println!("Initial:     ${initial:.2}");
            if let Some(rate) = withdrawal_rate {
                println!("Withdrawal:  {:.2}% annual", rate * 100.0);
            }
            println!();
            println!("Final Value Statistics:");
            println!("  Mean:      ${:.2}", stats.mean);
            println!("  Std Dev:   ${:.2}", stats.std);
            println!("  Min:       ${:.2}", stats.min);
            println!("  Max:       ${:.2}", stats.max);
            println!();
            println!("Risk Metrics:");
            println!("  VaR (95%):      {:.2}%", report.var_95 * 100.0);
            println!("  CVaR (95%):     {:.2}%", report.cvar_95 * 100.0);
            println!("  Max Drawdown:   {:.2}%", report.drawdown.mean * 100.0);
            println!("  Sharpe Ratio:   {:.3}", report.sharpe_ratio);
            println!("  Sortino Ratio:  {:.3}", report.sortino_ratio);

            if withdrawal_rate.is_some() {
                let ruin_pct = stats.min.max(0.0) / initial * 100.0;
                println!("\nWithdrawal Analysis:");
                println!("  Ruin probability: estimate based on minimum value");
                println!(
                    "  Minimum value: ${:.2} ({:.1}% of initial)",
                    stats.min.max(0.0),
                    ruin_pct
                );
            }
        }
        OutputFormat::Json => {
            let json = serde_json::json!({
                "simulations": simulations,
                "years": years,
                "initial": initial,
                "withdrawal_rate": withdrawal_rate,
                "statistics": {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max
                },
                "risk": {
                    "var_95": report.var_95,
                    "cvar_95": report.cvar_95,
                    "max_drawdown_mean": report.drawdown.mean,
                    "sharpe_ratio": report.sharpe_ratio,
                    "sortino_ratio": report.sortino_ratio
                }
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        }
        OutputFormat::Csv => {
            println!("metric,value");
            println!("simulations,{simulations}");
            println!("years,{years}");
            println!("initial,{initial}");
            println!("mean,{:.2}", stats.mean);
            println!("std,{:.2}", stats.std);
            println!("var_95,{:.4}", report.var_95);
            println!("cvar_95,{:.4}", report.cvar_95);
            println!("max_drawdown,{:.4}", report.drawdown.mean);
            println!("sharpe_ratio,{:.4}", report.sharpe_ratio);
            println!("sortino_ratio,{:.4}", report.sortino_ratio);
        }
    }
}

#[allow(clippy::disallowed_methods)] // serde_json::json! macro uses unwrap internally
fn output_revenue_results(
    stats: &Statistics,
    report: &RiskReport,
    simulations: usize,
    quarters: u32,
    initial: f64,
    format: OutputFormat,
) {
    match format {
        OutputFormat::Table => {
            println!("\nRevenue Forecast Results");
            println!("========================");
            println!("Simulations: {simulations}");
            println!("Horizon:     {quarters} quarters");
            println!("Base Revenue: ${initial:.2}");
            println!();
            println!("Projected Revenue:");
            println!("  Mean:      ${:.2}", stats.mean);
            println!("  Std Dev:   ${:.2}", stats.std);
            println!("  Min:       ${:.2}", stats.min);
            println!("  Max:       ${:.2}", stats.max);
            println!("  Growth:    {:.2}%", (stats.mean / initial - 1.0) * 100.0);
            println!();
            println!("Risk Metrics:");
            println!("  VaR (95%):      {:.2}%", report.var_95 * 100.0);
            println!("  CVaR (95%):     {:.2}%", report.cvar_95 * 100.0);
            println!("  Max Drawdown:   {:.2}%", report.drawdown.mean * 100.0);
        }
        OutputFormat::Json => {
            let json = serde_json::json!({
                "simulations": simulations,
                "quarters": quarters,
                "base_revenue": initial,
                "projected": {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "growth_pct": (stats.mean / initial - 1.0) * 100.0
                },
                "risk": {
                    "var_95": report.var_95,
                    "cvar_95": report.cvar_95,
                    "max_drawdown_mean": report.drawdown.mean
                }
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        }
        OutputFormat::Csv => {
            println!("metric,value");
            println!("simulations,{simulations}");
            println!("quarters,{quarters}");
            println!("base_revenue,{initial}");
            println!("mean,{:.2}", stats.mean);
            println!("std,{:.2}", stats.std);
            println!("growth_pct,{:.2}", (stats.mean / initial - 1.0) * 100.0);
            println!("var_95,{:.4}", report.var_95);
            println!("cvar_95,{:.4}", report.cvar_95);
        }
    }
}
