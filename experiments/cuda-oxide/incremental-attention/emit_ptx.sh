#!/usr/bin/env bash
# PMAT-883 — emit the standalone embeddable PTX for the oxide `attn_warp_rawptr`
# decode-attention kernel (sm_121, GB10 Blackwell).
#
# This is the source-of-record emit path. Kernel C uses f32 `exp()` (softmax),
# which cuda-oxide lowers to a libdevice `__nv_expf` call and therefore emits as
# NVVM IR (it skips llc, leaving libNVVM lowering to the consumer). To produce a
# fully self-contained `.ptx` (no extern `__nv_*`) that `include_str!` ->
# CudaModule::from_ptx can load directly, we:
#   1. cargo oxide pipeline  -> NVVM IR (.ll)
#   2. llvm-link libdevice   -> resolve __nv_expf
#   3. opt internalize/nvvm-reflect/O3
#   4. llc nvptx64 sm_121    -> full module .ptx
#   5. trim to the single `attn_warp_rawptr` entry (+header)
#   6. ptxas -arch=sm_121    -> validate it assembles
#
# Run on gx10 (GB10, sm_121) ONLY:
#   ssh gx10
#   cd /tmp/incattn883_spike   # or rsync this dir
#   ./emit_ptx.sh [out.ptx]
set -euo pipefail

OUT="${1:-generated/attn_warp.sm121.ptx}"
ARCH="${ARCH:-sm_121}"
ENTRY="attn_warp_rawptr"
LIBDEVICE="${LIBDEVICE:-/usr/local/cuda-13.0/nvvm/libdevice/libdevice.10.bc}"

export PATH="${HOME}/.cargo/bin:/usr/lib/llvm-21/bin:${PATH}"
export LLVM_SYS_211_PREFIX="${LLVM_SYS_211_PREFIX:-/usr/lib/llvm-21}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT

echo "[1/6] cargo oxide pipeline --arch ${ARCH} -> NVVM IR (.ll)"
cargo oxide pipeline --arch "${ARCH}" >/dev/null 2>&1 || true
LL="$(find . -maxdepth 1 -name '*.ll' -print -quit)"
if [ -z "${LL}" ]; then
  echo "ERROR: no .ll emitted by cargo oxide pipeline" >&2
  exit 1
fi
echo "      .ll = ${LL}"

echo "[2/6] llvm-link ${LL} + libdevice"
test -f "${LIBDEVICE}" || { printf 'ERROR: libdevice not found at %s\n' "${LIBDEVICE}" >&2; exit 1; }
llvm-link "${LL}" "${LIBDEVICE}" -o "${WORK}/linked.bc"

echo "[3/6] opt: internalize (keep all 5 entries) + nvvm-reflect + globaldce + O3"
opt -passes="internalize,nvvm-reflect,globaldce" \
    -internalize-public-api-list="attn_warp,${ENTRY},attn_chunk,attn_reduce,incremental_attention" \
    "${WORK}/linked.bc" -o "${WORK}/int.bc"
opt -O3 "${WORK}/int.bc" -o "${WORK}/opt.bc"

echo "[4/6] llc nvptx64 ${ARCH} -> full module .ptx"
llc -mcpu="${ARCH}" -mtriple=nvptx64-nvidia-cuda "${WORK}/opt.bc" -o "${WORK}/full.ptx"

echo "[5/6] trim to single entry: ${ENTRY}"
mkdir -p "$(dirname "${OUT}")"
SCRIPT_DIR="$(cd "$(dirname "$0")" || exit 1; pwd)"
{
  echo "//"
  echo "// PMAT-883 oxide ${ENTRY} decode-attention kernel, source-of-record."
  echo "// sm_121 (GB10 Blackwell). Self-contained (libdevice exp call inlined)."
  echo "// Emitted by emit_ptx.sh: cargo oxide pipeline -> llvm-link libdevice ->"
  echo "// opt internalize/nvvm-reflect/O3 -> llc nvptx64 -> trim -> ptxas-verified."
  echo "// ABI: ${ENTRY}(q,k,v:*const f32, out:*mut f32, kv_len,head_dim,n_heads,"
  echo "//      n_kv_heads:u32, scale:f32). Launch grid=(n_heads,1,1) block=(1024,1,1)."
  echo "//"
  awk -v entry="${ENTRY}" -f "${SCRIPT_DIR}/trim_entry.awk" "${WORK}/full.ptx"
} > "${OUT}.raw"
# ptxas rejects non-ASCII bytes: keep only ASCII printable + tab/newline/CR.
LC_ALL=C tr -cd '\11\12\15\40-\176' < "${OUT}.raw" > "${OUT}"
rm -f "${OUT}.raw"

ENTRIES="$(grep -c '\.visible \.entry' "${OUT}")"
# Count only PTX directives (exclude // comment lines) for unresolved externs.
EXTERNS="$(grep -v '^//' "${OUT}" | grep -c '\.extern \.func\|__nv_' || true)"
echo "      ${OUT}: $(wc -l < "${OUT}") lines, ${ENTRIES} entry, ${EXTERNS} extern __nv refs"
[ "${ENTRIES}" = "1" ] || { echo "ERROR: expected exactly 1 entry" >&2; exit 1; }
[ "${EXTERNS}" = "0" ] || { echo "ERROR: unresolved extern __nv refs remain" >&2; exit 1; }

echo "[6/6] ptxas -arch=${ARCH} (verify it assembles)"
ptxas -arch="${ARCH}" "${OUT}" -o "${WORK}/verify.cubin"
echo "      ptxas OK ($(wc -c < "${WORK}/verify.cubin") byte cubin)"
echo "DONE: ${OUT}"
