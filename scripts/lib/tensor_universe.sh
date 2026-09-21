# tensor_universe.sh -- the Q4_K universe definition for shell callers (#3763, #3712 row A2; #3742).
#
# THE definition lives in scripts/lib/tensor_universe.py (a file's dominant >= 2-D tensor dtypes by COUNT,
# read from `apr tensors --json`; a member iff DTYPE is among them, ties included). This file only reads the
# header and hands it over, so a shell gate and the model ladder cannot disagree about which files are Q4_K.
#
#   . scripts/lib/tensor_universe.sh || exit 1
#   tu_dominant_dtypes <apr-bin> <file>      -> the dominant dtypes, space-separated; rc 2 if unreadable
#   tu_is_member <apr-bin> <file> <DTYPE>    -> rc 0 member, 1 not a member, 2 unreadable
#
# A header read is not GPU work: it runs with no GPU visible (CUDA_VISIBLE_DEVICES set and empty) and takes
# no fleet lock. SOURCED and option-neutral: it sets no shell options (a sourced `set` mutates the caller,
# scripts/check_sourced_libs_option_neutral.sh) and fails by return status only.

TU_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tensor_universe.py"

# tu__header <apr-bin> <file> <out.json> -> the header read's status
tu__header() {
  CUDA_VISIBLE_DEVICES="" "$1" tensors "$2" --json > "$3" 2> /dev/null
}

tu_dominant_dtypes() {
  local hdr rc
  hdr=$(mktemp) || return 2
  if tu__header "$1" "$2" "$hdr"; then python3 "$TU_PY" dominant "$hdr"; rc=$?; else rc=2; fi
  rm -f -- "$hdr"
  return "$rc"
}

tu_is_member() {
  local hdr rc
  hdr=$(mktemp) || return 2
  if tu__header "$1" "$2" "$hdr"; then python3 "$TU_PY" member "$hdr" "$3"; rc=$?; else rc=2; fi
  rm -f -- "$hdr"
  return "$rc"
}
