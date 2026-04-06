/**
 * Entrenar Monitor WASM TypeScript Definitions
 *
 * Real-time training visualization for the browser.
 *
 * @example
 * ```typescript
 * import init, { WasmMetricsCollector, WasmDashboardOptions } from 'entrenar-monitor';
 *
 * await init();
 *
 * const collector = new WasmMetricsCollector();
 * collector.record_loss(0.5);
 * collector.record_accuracy(0.85);
 *
 * const stats = JSON.parse(collector.summary_json());
 * console.log(`Loss: ${stats.loss.mean}, Accuracy: ${stats.accuracy.mean}`);
 * ```
 */

/**
 * Initialize the WASM module.
 * Must be called before using any other functions.
 */
export default function init(): Promise<void>;

/**
 * Statistics for a single metric.
 */
export interface MetricStats {
  /** Number of recorded values */
  count: number;
  /** Mean value */
  mean: number;
  /** Standard deviation */
  std: number;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Whether any NaN values were recorded */
  has_nan: boolean;
  /** Whether any Inf values were recorded */
  has_inf: boolean;
}

/**
 * Summary of all metrics.
 */
export interface MetricsSummary {
  loss?: MetricStats;
  accuracy?: MetricStats;
  learning_rate?: MetricStats;
  gradient_norm?: MetricStats;
  [key: string]: MetricStats | undefined;
}

/**
 * WASM-compatible metrics collector for training.
 *
 * @example
 * ```typescript
 * const collector = new WasmMetricsCollector();
 *
 * // Record metrics during training
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   collector.record_loss(1.0 / (epoch + 1));
 *   collector.record_accuracy(0.5 + 0.005 * epoch);
 * }
 *
 * // Get summary statistics
 * const summary: MetricsSummary = JSON.parse(collector.summary_json());
 * console.log(`Final loss: ${summary.loss?.mean}`);
 * ```
 */
export class WasmMetricsCollector {
  /**
   * Create a new metrics collector.
   */
  constructor();

  /**
   * Record a loss value.
   * @param value - The loss value to record
   */
  record_loss(value: number): void;

  /**
   * Record an accuracy value.
   * @param value - The accuracy value to record (0.0 to 1.0)
   */
  record_accuracy(value: number): void;

  /**
   * Record a learning rate value.
   * @param value - The learning rate value to record
   */
  record_learning_rate(value: number): void;

  /**
   * Record a gradient norm value.
   * @param value - The gradient L2 norm to record
   */
  record_gradient_norm(value: number): void;

  /**
   * Record a custom metric.
   * @param name - The metric name
   * @param value - The metric value
   */
  record_custom(name: string, value: number): void;

  /**
   * Get the total number of recorded metrics.
   */
  count(): number;

  /**
   * Check if the collector is empty.
   */
  is_empty(): boolean;

  /**
   * Clear all recorded metrics.
   */
  clear(): void;

  /**
   * Get summary statistics as a JSON string.
   * Parse with JSON.parse() to get MetricsSummary.
   */
  summary_json(): string;

  /**
   * Get the mean loss value.
   * Returns NaN if no loss has been recorded.
   */
  loss_mean(): number;

  /**
   * Get the mean accuracy value.
   * Returns NaN if no accuracy has been recorded.
   */
  accuracy_mean(): number;
}

/**
 * Dashboard rendering options.
 *
 * @example
 * ```typescript
 * const opts = new WasmDashboardOptions()
 *   .width(1024)
 *   .height(768)
 *   .background_color('#1a1a2e')
 *   .loss_color('#ff6b6b')
 *   .accuracy_color('#4ecdc4');
 * ```
 */
export class WasmDashboardOptions {
  /**
   * Create default dashboard options.
   * Default: 800x400, dark theme, sparklines enabled.
   */
  constructor();

  /**
   * Set width in pixels.
   * @param width - Width in pixels (default: 800)
   */
  width(width: number): WasmDashboardOptions;

  /**
   * Set height in pixels.
   * @param height - Height in pixels (default: 400)
   */
  height(height: number): WasmDashboardOptions;

  /**
   * Set background color.
   * @param color - Hex color (e.g., '#1a1a2e')
   */
  background_color(color: string): WasmDashboardOptions;

  /**
   * Set loss curve color.
   * @param color - Hex color (e.g., '#ff6b6b')
   */
  loss_color(color: string): WasmDashboardOptions;

  /**
   * Set accuracy curve color.
   * @param color - Hex color (e.g., '#4ecdc4')
   */
  accuracy_color(color: string): WasmDashboardOptions;

  /**
   * Enable or disable sparklines.
   * @param show - Whether to show sparklines (default: true)
   */
  show_sparklines(show: boolean): WasmDashboardOptions;
}

/**
 * Dashboard for canvas rendering and sparkline generation.
 *
 * @example
 * ```typescript
 * const dashboard = new WasmDashboard();
 *
 * // Add metrics during training
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   dashboard.add_loss(1.0 / (epoch + 1));
 *   dashboard.add_accuracy(0.5 + 0.005 * epoch);
 * }
 *
 * // Get sparkline for terminal display
 * console.log(`Loss: ${dashboard.loss_sparkline()}`);
 *
 * // Get state for canvas rendering
 * const state = JSON.parse(dashboard.state_json());
 * ```
 */
export class WasmDashboard {
  /**
   * Create a new dashboard with default options.
   */
  constructor();

  /**
   * Set maximum history length.
   * @param max - Maximum number of values to keep (default: 100)
   */
  max_history(max: number): WasmDashboard;

  /**
   * Update dashboard with metrics from collector.
   * @param collector - The metrics collector
   */
  update(collector: WasmMetricsCollector): void;

  /**
   * Add a loss value directly.
   * @param value - Loss value (NaN/Inf ignored)
   */
  add_loss(value: number): void;

  /**
   * Add an accuracy value directly.
   * @param value - Accuracy value (NaN/Inf ignored)
   */
  add_accuracy(value: number): void;

  /**
   * Get loss history length.
   */
  loss_history_len(): number;

  /**
   * Get accuracy history length.
   */
  accuracy_history_len(): number;

  /**
   * Clear all history.
   */
  clear(): void;

  /**
   * Get canvas width.
   */
  width(): number;

  /**
   * Get canvas height.
   */
  height(): number;

  /**
   * Generate sparkline characters for loss.
   */
  loss_sparkline(): string;

  /**
   * Generate sparkline characters for accuracy.
   */
  accuracy_sparkline(): string;

  /**
   * Get dashboard state as JSON.
   * Contains width, height, loss_history, accuracy_history, colors.
   */
  state_json(): string;
}

/**
 * Dashboard state structure (from state_json()).
 */
export interface DashboardState {
  width: number;
  height: number;
  loss_history: number[];
  accuracy_history: number[];
  loss_color: string;
  accuracy_color: string;
  background_color: string;
}
