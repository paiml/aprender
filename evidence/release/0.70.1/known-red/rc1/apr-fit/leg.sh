#!/usr/bin/env bash
# C276 rule 2 / C272 5b evidence: the 5 lambda .apr rows on v0.69.5-rc.1 (released cuda asset), current ladder at cc4463f.
# The #4016 fit gate runs before any apr call, so this records whether rc.1 is refused the same way.
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
O=/mnt/nvme-raid0/tmp/a-0701/apr-rc1
cd /mnt/nvme-raid0/agent-wt/rel-0701 || exit 2
[ "$(git rev-parse HEAD)" = cc4463f7a609f04c0c4fd1ba4229a985ddaeeead ] || { echo "WRONG-TREE $(git rev-parse HEAD)"; exit 2; }
while systemctl --user is-active --quiet evid0701b-rerun8b; do sleep 30; done
RC1=/mnt/nvme-raid0/tmp/e6-rc0695/gh-assets-rc1/x-apr-v0.69.5-rc.1-x86_64-unknown-linux-gnu-cuda/apr-v0.69.5-rc.1-x86_64-unknown-linux-gnu-cuda/apr
echo "== RC1 $("$RC1" --version | head -1) sha256=$(sha256sum "$RC1" | cut -c1-16) START $(date -u +%FT%TZ)"
for f in qwen2.5-coder-0.5b-instruct.apr qwen2.5-coder-1.5b-instruct-q4k-v2.apr qwen2.5-coder-1.5b-instruct-q4k.apr qwen2.5-coder-1.5b-instruct-st.apr qwen2.5-coder-1.5b-q4k.apr; do
  s=$(echo "$f" | tr -c 'A-Za-z0-9.\n-' _)
  DOGFOOD_ALLOW_UNPINNED=1 APR="$RC1" choom -n 1000 -- bash scripts/model_ladder.sh --host lambda --only "inv:$f" --out "$O/only-$s" > "$O/$s.log" 2>&1
  echo "$f rc=$? $(date -u +%FT%TZ) :: $(grep -E '^\s+\[' "$O/$s.log" | tail -1 | cut -c1-200)"
done
echo DONE
