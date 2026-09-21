#!/usr/bin/env bash
# PMAT-3091 SCALAR llama.cpp d1d3c3396 build (intel): no x86 instruction sets beyond baseline x86-64, CPU repack OFF,
# llamafile sgemm OFF, out-of-tree in ~/parity-ref/llama-scalar-d1d3c3396. Option names read from ggml/CMakeLists.txt
# (GGML_NATIVE, GGML_SSE42/AVX/AVX2/BMI2/FMA/F16C/AVX_VNNI/AVX512*, GGML_CPU_REPACK, GGML_LLAMAFILE) and
# ggml/src/ggml-cpu/CMakeLists.txt (x86 non-MSVC: NATIVE=OFF adds -m<isa> only for enabled options; INS_ENB defaults
# them ON when NATIVE is OFF, so each is set OFF explicitly). Then the producer, as build.sh does, against this dir.
set -uo pipefail
L="$HOME/src/llama.cpp-d1d3c3396"; B="$HOME/parity-ref/llama-scalar-d1d3c3396"; S="$HOME/parity-ref/scalar-3091"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) dirty=$(git -C "$L" status --porcelain --untracked-files=no | wc -l) host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) free=$(df -h --output=avail "$HOME" | tail -1)"
gcc --version | head -1
nice -n 10 cmake -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_NATIVE=OFF -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
  -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
  -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF \
  -DGGML_CCACHE=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_OPENSSL=OFF \
  < /dev/null; echo "configure rc=$?"
nice -n 10 timeout 3600 cmake --build "$B" -j 8 --target llama llama-common ggml ggml-base ggml-cpu < /dev/null > "$S/build.log" 2>&1; echo "build rc=$?"; tail -3 "$S/build.log"
grep -E '^(GGML_[A-Z0-9_]+|CMAKE_BUILD_TYPE|BUILD_SHARED_LIBS):' "$B/CMakeCache.txt" | grep -E 'NATIVE|SSE|AVX|FMA|F16C|BMI|AMX|REPACK|LLAMAFILE|OPENMP:|BUILD_TYPE|SHARED' > "$S/cmake_cache_flags.txt"; cat "$S/cmake_cache_flags.txt"
echo "--- producer"
mkdir -p "$S/bin"
c++ -O2 -std=c++17 -Wall -Wextra -I"$L/include" -I"$L/common" -I"$L/ggml/include" "$S/producer/apr_raw_logits.cpp" -o "$S/bin/apr_raw_logits" \
  -L"$B/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" < /dev/null; echo "producer rc=$?"
ldd "$S/bin/apr_raw_logits" | grep -E 'llama|ggml'
sha256sum "$S/bin/apr_raw_logits" "$B"/bin/libggml-cpu.so.* "$B"/bin/libggml-base.so.* "$B"/bin/libllama.so.*
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
