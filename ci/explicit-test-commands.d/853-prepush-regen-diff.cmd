# RS-6 (C49): the pre-push regen-diff check passes on a clean graph and refuses on drift and on a regen that cannot run
bash scripts/hooks/prepush_regen_diff.sh --self-test
