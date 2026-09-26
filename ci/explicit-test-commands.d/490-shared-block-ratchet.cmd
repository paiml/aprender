# #3422 PP-ARCH-001 Phase 0: RMSNorm/RoPE/softmax/attention duplicates may not grow past docs/audits/shared-block-duplicates-baseline.tsv
python3 scripts/classify_shared_block_adoption.py --self-test && python3 scripts/classify_shared_block_adoption.py
