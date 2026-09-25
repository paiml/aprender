# #3431: bare-integer qtype lines may not grow past docs/audits/bare-int-qtype-baseline.tsv
python3 scripts/classify_bare_int_qtype.py --self-test && python3 scripts/classify_bare_int_qtype.py
