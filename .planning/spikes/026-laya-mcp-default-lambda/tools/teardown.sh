#!/usr/bin/env bash
# Remove spike-026 resources (tag spike=026). The weights live under s3://kev-spike-021-AWS_ACCOUNT_ID/laya-en-55cf4c4e/
# and go with spike 021's bucket teardown (../021-kev-mcp-default-lambda/tools/teardown.sh).
set -uo pipefail
for fn in laya-spike-10g laya-spike-4g; do
  aws lambda delete-function --region us-east-1 --function-name "$fn" || true
  aws logs delete-log-group --region us-east-1 --log-group-name "/aws/lambda/$fn" 2>/dev/null || true
done
aws s3 rm s3://kev-spike-021-AWS_ACCOUNT_ID/laya-en-55cf4c4e/ --recursive --region us-east-1 || true
echo "spike-026 teardown done"
