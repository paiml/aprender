# A PR lands only via the train: the heavy tier runs once per train (#7001); each removed conjunct, need or refusal must turn --self-test RED.
bash scripts/check_ci_train_only.sh --self-test
