#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * Statistical analysis utilities for benchmark results
 *
 * Provides functions for:
 * - Descriptive statistics (mean, median, std dev, CV)
 * - Confidence intervals (bootstrap method)
 * - Comparative analysis (effect sizes, speedup ratios)
 * - Outlier detection (IQR method)
 */

export interface Statistics {
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
  coefficientOfVariation: number;
  confidenceInterval95: [number, number];
}

/**
 * Calculate descriptive statistics for a dataset
 */
export function calculateStatistics(values: number[]): Statistics {
  if (values.length === 0) {
    throw new Error("Cannot calculate statistics for empty dataset");
  }

  const sorted = [...values].sort((a, b) => a - b);
  const mean = calculateMean(values);
  const stdDev = calculateStdDev(values, mean);

  return {
    mean,
    median: calculateMedian(sorted),
    stdDev,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    coefficientOfVariation: stdDev / mean,
    confidenceInterval95: bootstrapConfidenceInterval(values, mean),
  };
}

/**
 * Calculate arithmetic mean
 */
export function calculateMean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Calculate median
 */
export function calculateMedian(sortedValues: number[]): number {
  const mid = Math.floor(sortedValues.length / 2);
  if (sortedValues.length % 2 === 0) {
    return (sortedValues[mid - 1] + sortedValues[mid]) / 2;
  }
  return sortedValues[mid];
}

/**
 * Calculate standard deviation
 */
export function calculateStdDev(values: number[], mean: number): number {
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((sum, v) => sum + v, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Bootstrap confidence interval estimation
 * Uses percentile method with 1000 bootstrap samples
 */
export function bootstrapConfidenceInterval(
  values: number[],
  statistic: number,
  iterations = 1000,
  alpha = 0.05
): [number, number] {
  const bootstrapMeans: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const sample = bootstrapSample(values);
    bootstrapMeans.push(calculateMean(sample));
  }

  bootstrapMeans.sort((a, b) => a - b);

  const lowerIndex = Math.floor(iterations * (alpha / 2));
  const upperIndex = Math.floor(iterations * (1 - alpha / 2));

  return [bootstrapMeans[lowerIndex], bootstrapMeans[upperIndex]];
}

/**
 * Generate bootstrap sample (sample with replacement)
 */
function bootstrapSample(values: number[]): number[] {
  const sample: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const randomIndex = Math.floor(Math.random() * values.length);
    sample.push(values[randomIndex]);
  }
  return sample;
}

/**
 * Detect outliers using IQR method
 * Returns { outliers, cleaned, outlierIndices }
 */
export function detectOutliers(values: number[]): {
  outliers: number[];
  cleaned: number[];
  outlierIndices: number[];
} {
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = calculateQuantile(sorted, 0.25);
  const q3 = calculateQuantile(sorted, 0.75);
  const iqr = q3 - q1;

  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  const outliers: number[] = [];
  const cleaned: number[] = [];
  const outlierIndices: number[] = [];

  values.forEach((v, i) => {
    if (v < lowerBound || v > upperBound) {
      outliers.push(v);
      outlierIndices.push(i);
    } else {
      cleaned.push(v);
    }
  });

  return { outliers, cleaned, outlierIndices };
}

/**
 * Calculate quantile
 */
function calculateQuantile(sortedValues: number[], q: number): number {
  const pos = (sortedValues.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;

  if (sortedValues[base + 1] !== undefined) {
    return sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base]);
  }
  return sortedValues[base];
}

/**
 * Calculate Cohen's d effect size
 */
export function calculateCohenD(
  baseline: number[],
  current: number[]
): number {
  const baselineMean = calculateMean(baseline);
  const currentMean = calculateMean(current);

  const pooledStdDev = calculatePooledStdDev(baseline, current);

  return (currentMean - baselineMean) / pooledStdDev;
}

/**
 * Calculate pooled standard deviation
 */
function calculatePooledStdDev(group1: number[], group2: number[]): number {
  const n1 = group1.length;
  const n2 = group2.length;

  const mean1 = calculateMean(group1);
  const mean2 = calculateMean(group2);

  const var1 = calculateStdDev(group1, mean1) ** 2;
  const var2 = calculateStdDev(group2, mean2) ** 2;

  const pooledVariance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);

  return Math.sqrt(pooledVariance);
}

/**
 * Calculate speedup ratio with confidence interval
 */
export function calculateSpeedup(
  baseline: number[],
  current: number[]
): {
  speedupRatio: number;
  confidenceInterval95: [number, number];
} {
  const baselineMean = calculateMean(baseline);
  const currentMean = calculateMean(current);

  const speedupRatio = baselineMean / currentMean;

  // Bootstrap CI for speedup ratio
  const bootstrapRatios: number[] = [];
  for (let i = 0; i < 1000; i++) {
    const baseSample = bootstrapSample(baseline);
    const currSample = bootstrapSample(current);
    bootstrapRatios.push(calculateMean(baseSample) / calculateMean(currSample));
  }

  bootstrapRatios.sort((a, b) => a - b);
  const lowerIndex = Math.floor(1000 * 0.025);
  const upperIndex = Math.floor(1000 * 0.975);

  return {
    speedupRatio,
    confidenceInterval95: [bootstrapRatios[lowerIndex], bootstrapRatios[upperIndex]],
  };
}

/**
 * Simplified Welch's t-test p-value approximation
 * For proper implementation, would need t-distribution
 * This is a simplified version for demonstration
 */
export function welchTTest(
  baseline: number[],
  current: number[]
): number {
  const mean1 = calculateMean(baseline);
  const mean2 = calculateMean(current);

  const var1 = calculateStdDev(baseline, mean1) ** 2;
  const var2 = calculateStdDev(current, mean2) ** 2;

  const n1 = baseline.length;
  const n2 = current.length;

  // Welch's t-statistic
  const t = (mean1 - mean2) / Math.sqrt(var1 / n1 + var2 / n2);

  // Degrees of freedom (Welch-Satterthwaite equation)
  const df = Math.pow(var1 / n1 + var2 / n2, 2) /
    (Math.pow(var1 / n1, 2) / (n1 - 1) + Math.pow(var2 / n2, 2) / (n2 - 1));

  // Simplified p-value approximation
  // For production use, should use proper t-distribution CDF
  // This approximation uses normal distribution for df > 30
  const pValue = 2 * (1 - normalCDF(Math.abs(t)));

  return pValue;
}

/**
 * Normal CDF approximation (for large samples)
 */
function normalCDF(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

  return z > 0 ? 1 - prob : prob;
}

/**
 * Format number with fixed precision
 */
export function formatNumber(value: number, precision = 2): string {
  return value.toFixed(precision);
}

/**
 * Format percentage
 */
export function formatPercent(value: number, precision = 1): string {
  return (value * 100).toFixed(precision) + "%";
}

// CLI usage
if (import.meta.main) {
  console.log("Statistical Analysis Utilities");
  console.log("Usage: import { calculateStatistics } from './statistical_analysis.ts'");

  // Example usage
  const sampleData = [21.42, 21.31, 20.15, 21.89, 21.45, 21.67, 21.23, 21.56, 21.78, 23.67];
  const stats = calculateStatistics(sampleData);

  console.log("\nExample with sample data:", sampleData);
  console.log("Statistics:", {
    mean: formatNumber(stats.mean),
    median: formatNumber(stats.median),
    stdDev: formatNumber(stats.stdDev),
    cv: formatPercent(stats.coefficientOfVariation),
    ci95: `[${formatNumber(stats.confidenceInterval95[0])}, ${formatNumber(stats.confidenceInterval95[1])}]`,
  });

  const outlierResult = detectOutliers(sampleData);
  console.log("\nOutlier detection:");
  console.log("  Outliers:", outlierResult.outliers);
  console.log("  Cleaned values:", outlierResult.cleaned.length, "samples");
}
