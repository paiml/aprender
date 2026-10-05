# Row 11: the llama.cpp comparator build for each E1 cell

Read at origin/main 11f844a772 (`scripts/llama_pin.toml`, `scripts/llama_bin.sh`). No
host was probed: which hosts hold which build today stays [U].

## What each cell needs (R1 cell table)
| cell | host | apr backend | comparator (R1, RQ-2 ruling) |
|---|---|---|---|
| C0 | lambda | CUDA sm_89 + x86 CPU | ggml-cuda (the existing parity_host_receipt.sh path) |
| C1, C2 | intel | wgpu/Vulkan, AMD GPU 0 and 1 | ggml-webgpu (like-for-like) + ggml-vulkan |
| C3 | mini | wgpu/Metal | ggml-webgpu + ggml-metal (MLX advisory only) |
| C4 | gx10 | aarch64 CPU | ggml-cpu |
| C5 | gx10 | CUDA sm_121 | ggml-cuda |

## What the pin declares and what the resolver checks
- The pin is one commit, `build_commit = "d1d3c3396"` (`scripts/llama_pin.toml`), and
  `scripts/llama_bin.sh` refuses a binary whose `--version` names another
  (`wrong_build`). That part already serves every cell.
- How it was built is declared as one cmake line per host: `build_flags_lambda`,
  `build_flags_gx10` (CUDA, arch 121), `build_flags_intel` (`-DGGML_NATIVE=ON`) and
  `build_flags_mini` (`-DGGML_METAL=ON`). `llama_pin_host()` maps the hostname to one
  of those four.
- The resolver compares the build's CMakeCache.txt with that line on two keys only,
  `GGML_CUDA` and `CMAKE_CUDA_ARCHITECTURES` (`llama_bin.sh:451-464`). Any mismatch is
  `cmake_mismatch`.

## Gaps
- **G11-1, one line per host, two builds per cell.** C1-C3 need two comparators each
  (webgpu and vulkan on intel, webgpu and metal on mini), and the pin has one line for
  the host. Neither webgpu nor vulkan is declared anywhere (`grep -i webgpu` and
  `vulkan` in the pin: 0 hits). C4 and C5 share gx10's CUDA line; whether C4's
  ggml-cpu leg may be that CUDA build run with `-ngl 0` is not written down [U].
- **G11-2, the backend is not checked off CUDA.** On intel and mini both declared keys
  are unset, and so are they in a CPU, webgpu, vulkan or metal build. So the resolver
  passes any of those builds on those hosts. `build_flags_mini` names GGML_METAL, and
  nothing reads it. A C1 receipt today could carry a ggml-cpu ratio under a webgpu
  label, and R1 F4 (the sha check) would pass it. That is the silent-substitution
  shape E4 forbids, on the comparator side.
- **G11-3, the receipt does not name the comparator backend.** F4 checks the commit.
  No field says which ggml backend produced the denominator, so a reader cannot
  tell webgpu from vulkan in a C1 receipt.

## Draft fix (for the row-11 ticket; bash only, per C301)
1. Key the declaration by cell backend, not host: `build_flags_<host>_<backend>`, e.g.
   `build_flags_intel_webgpu = "cmake -B build-webgpu -DGGML_WEBGPU=ON"`,
   `build_flags_intel_vulkan = "cmake -B build-vulkan -DGGML_VULKAN=ON"`,
   `build_flags_mini_webgpu`, `build_flags_mini_metal`, `build_flags_gx10_cpu`. Keep
   the four host lines as the `cuda`/default entries, so lambda and gx10 numbers keep
   their basis. Exact cmake option names to be read from the pinned checkout's
   `ggml/CMakeLists.txt` [U].
2. The resolver takes the backend (`LLAMA_PIN_BACKEND`, required, no default) and
   compares every backend key it knows (`GGML_CUDA`, `GGML_VULKAN`, `GGML_WEBGPU`,
   `GGML_METAL`, plus CUDA archs), not two: an `ON` the line does not declare is a
   mismatch too, so a vulkan build cannot pass as webgpu.
3. The receipt records `comparator_backend` next to the sha, and F4 also fails a
   receipt whose `comparator_backend` is not the cell's.

## Falsifiers (case-table rows for `scripts/check_llama_pin.sh`)
| case | input | want |
|---|---|---|
| F11-1 | backend webgpu, cache has GGML_WEBGPU=ON only | ok |
| F11-2 | backend webgpu, cache is a plain CPU build | cmake_mismatch |
| F11-3 | backend webgpu, cache has GGML_VULKAN=ON | cmake_mismatch |
| F11-4 | backend metal on mini, cache lacks GGML_METAL | cmake_mismatch (today: ok, which is G11-2) |
| F11-5 | no LLAMA_PIN_BACKEND | refused, never defaulted |
| F11-6 | lambda and gx10 cuda as today | ok, unchanged |

## Open
- Which builds exist on intel, mini and gx10 today: a read-only probe per host
  (`--version` and the CMakeCache keys above), when L3 may touch hosts.
- C4: a separate CPU build or the CUDA build at `-ngl 0`. Default: a separate
  `build_flags_gx10_cpu` build, so the denominator holds no CUDA code path.
