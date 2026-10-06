# Row 13: intel's AMD GPUs under wgpu (driver and pinned adapter)

Read at origin/main 11f844a772. `D:` cites are
`crates/aprender-compute/src/backends/gpu/device/mod.rs`. No host was probed.

## How an inference route picks its adapter today
Every wgpu inference route builds its device with `GpuDevice::new()`:
serve (`crates/apr-cli/src/commands/serve/handlers.rs:754`), batch
(`crates/aprender-serve/src/infer/batch_wgpu.rs:141`) and generate
(`crates/aprender-serve/src/infer/gguf_gpu_generate.rs:172`, `:638`). That calls
`request_adapter` with `PowerPreference::HighPerformance` and
`force_fallback_adapter: false` (D:125-129), and nothing else: no device-type check, no
adapter index, no name. `new_with_adapter_index` (D:160) and the pool
(`backends/gpu/pool.rs:49`, `:111`) can pick by index, and no inference route calls
them. GLES is masked out (PMAT-925, `backends/gpu/mod.rs:69-71`); Vulkan software
adapters are not.

## Finding R13-1: a software Vulkan adapter is refused by the probe and taken by the routes
The kernel registry's probe classifies `DeviceType::Cpu` as "software rasteriser ... is
not a GPU", `Unavailable` (`crates/aprender-compute/src/registry/wgpu_probe.rs:58-70`).
The inference routes never ask: `request_adapter` returns whatever Vulkan adapter wgpu
ranks first, and on a host with no working AMD driver the only Vulkan adapter is the
Mesa software one (lavapipe), which wgpu lists with `DeviceType::Cpu`. So on intel today
a "wgpu" run can be a CPU run, with `used_gpu=true` and no message. E4 forbids that
shape, and E4's census (row 9) did not list it: add it as T13.
Supporting evidence, not proof: #3757's table (quoted at
`crates/aprender-serve/src/infer/gguf_gpu_generate.rs:108-111`) lists intel as
"Vulkan, no working driver", and still gives intel a cosine, 0.955376, equal to the
RTX 4090's to six figures. Which adapter that run used is [U].

## Finding R13-2: two AMD GPUs, and no way to choose one
R1's C1 and C2 are AMD GPU 0 and GPU 1 on intel. `HighPerformance` with two discrete
GPUs returns one of them, whichever wgpu ranks first, and the receipt does not say which.
So C1 and C2 cannot be separate cells until a route takes an adapter choice.

## Finding R13-3: receipts do not name the adapter
No inference route writes `AdapterInfo` (name, vendor, device id, device type, driver,
driver_info) to its output. `get_info` is read only by the probe, the monitor
(`monitor/backends.rs:34`, `:78`) and the pool filter (`pool.rs:59`). R1 F4 checks the
comparator's sha (row 11); nothing checks the adapter on apr's side.

## Draft for the row-13 ticket
Code (one PR, no host needed):
1. One adapter chooser for all four call sites: `--gpu-adapter <index|name-substring>`
   (and an env var for the harness), resolved through `enumerate_adapters`, refusing a
   `DeviceType::Cpu` adapter unless a test-only flag allows it. No choice given: the
   current `HighPerformance` pick, but still refusing `Cpu`.
2. Every wgpu receipt and `--json` output carries `adapter: {name, vendor, device,
   device_type, backend, driver, driver_info, index}`.
3. E1/E6: a C1 or C2 receipt whose `device_type` is not `DiscreteGpu`, or whose index
   is not the cell's, fails (joins row 16's (host, backend) match, which becomes
   (host, backend, adapter)).
Host (intel, operator or infra lane, not L3):
4. A working AMD Vulkan driver (Mesa RADV) for both GPUs; `vulkaninfo --summary` lists
   two AMD `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` entries, and their names go into R1's cell
   table.
5. The pinned choice: C1 = index of GPU 0, C2 = index of GPU 1, by name substring
   (indices can reorder across driver updates).

## Falsifiers
| id | claim | how |
|---|---|---|
| F13-1 | a software adapter is refused on every route | run with `VK_ICD_FILENAMES` pointing at lavapipe only: serve, batch and generate exit non-zero naming the adapter |
| F13-2 | the chosen adapter is the one used | two-GPU host: `--gpu-adapter 0` and `1` give receipts with different `device` ids |
| F13-3 | a receipt without `adapter` fails E1's checker | planted receipt with the field removed |
| F13-4 | a C1 receipt from GPU 1 fails C1 | planted adapter index |
| F13-5 | the #3757 intel figure is re-measured on a named DiscreteGpu adapter | E1 receipt on C1 with `device_type = DiscreteGpu` |

## Open
- Intel's driver state and which adapter #3757's intel run used [U]: a read-only
  `vulkaninfo --summary` on intel, when a lane may touch it.
- Whether E1 means one AMD GPU or both: R1 has two cells (C1, C2); PRM has one (C1
  intel-wgpu). Default: both, as R1 says.
