# #7277 dogfood-nightly LAB stop: no schedule trigger, every step has timeout-minutes; planted defects caught
bash scripts/check_dogfood_nightly_lab.sh --self-test && bash scripts/check_dogfood_nightly_lab.sh
