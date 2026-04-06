/* tslint:disable */
/* eslint-disable */
/**
 * Get correct GEMM output PNG
 */
export function get_correct_gemm(): Uint8Array;
/**
 * Run special values test
 */
export function test_special_values(): WasmTestResult;
/**
 * Run identity matrix test
 */
export function test_identity_matrix(): WasmTestResult;
/**
 * Get version string
 */
export function version(): string;
/**
 * Run bug detection test - returns buggy output for comparison
 */
export function test_bug_detection(): WasmTestResult;
/**
 * Run gradient test
 */
export function test_gradient(): WasmTestResult;
/**
 * Run deterministic RNG test
 */
export function test_deterministic_rng(seed: bigint): WasmTestResult;
/**
 * Run all visual tests
 */
export function run_all_tests(): WasmTestResult[];
/**
 * WASM test result
 */
export class WasmTestResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Number of different pixels
   */
  readonly diff_pixels: number;
  /**
   * Percentage of different pixels
   */
  readonly diff_percent: number;
  /**
   * Total number of pixels
   */
  readonly total_pixels: number;
  /**
   * Test name
   */
  readonly name: string;
  /**
   * Whether test passed
   */
  readonly passed: boolean;
  /**
   * PNG image data
   */
  readonly png_data: Uint8Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmtestresult_free: (a: number, b: number) => void;
  readonly get_correct_gemm: () => [number, number];
  readonly run_all_tests: () => [number, number];
  readonly test_bug_detection: () => number;
  readonly test_deterministic_rng: (a: bigint) => number;
  readonly test_gradient: () => number;
  readonly test_identity_matrix: () => number;
  readonly test_special_values: () => number;
  readonly version: () => [number, number];
  readonly wasmtestresult_diff_percent: (a: number) => number;
  readonly wasmtestresult_diff_pixels: (a: number) => number;
  readonly wasmtestresult_name: (a: number) => [number, number];
  readonly wasmtestresult_passed: (a: number) => number;
  readonly wasmtestresult_png_data: (a: number) => [number, number];
  readonly wasmtestresult_total_pixels: (a: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
