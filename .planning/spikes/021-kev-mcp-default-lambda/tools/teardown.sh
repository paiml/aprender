#!/usr/bin/env bash
# Remove every resource spikes 021-023 created (all tagged spike=021-023, region us-east-1).
# LMI (spike 023) was torn down at the end of the session; this removes the rest.
set -uo pipefail
R=us-east-1; B=kev-spike-021-AWS_ACCOUNT_ID
aws lambda delete-function --region "$R" --function-name kev-spike-lambda-s3 || true
aws lambda delete-function --region "$R" --function-name kev-spike-lambda-baked || true
for td in kev-spike-fargate-baked:1 kev-spike-fargate-s3:1; do aws ecs deregister-task-definition --region "$R" --task-definition "$td" >/dev/null || true; done
aws ecs delete-cluster --region "$R" --cluster kev-spike >/dev/null || true
aws codebuild delete-project --region "$R" --name kev-spike-images || true
aws ecr delete-repository --region "$R" --repository-name kev-spike --force >/dev/null || true
aws s3 rm "s3://$B" --recursive --region "$R" && aws s3api delete-bucket --bucket "$B" --region "$R" || true
aws ec2 delete-security-group --region "$R" --group-id sg-REDACTED || true
for g in /ecs/kev-spike /aws/lambda/kev-spike-lambda-s3 /aws/lambda/kev-spike-lambda-baked /aws/lambda/kev-spike-lmi /aws/lambda/kev-spike-lmi2 \
         /aws/codebuild/kev-spike-images /aws/lambda/capacity-provider/kev-spike-cp /aws/lambda/capacity-provider/kev-spike-cp2; do
  aws logs delete-log-group --region "$R" --log-group-name "$g" 2>/dev/null || true
done
for role in kev-spike-lambda kev-spike-ecs-exec kev-spike-ecs-task kev-spike-codebuild kev-spike-lmi-operator; do
  for p in $(aws iam list-attached-role-policies --role-name "$role" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
    aws iam detach-role-policy --role-name "$role" --policy-arn "$p"; done
  for p in $(aws iam list-role-policies --role-name "$role" --query 'PolicyNames' --output text 2>/dev/null); do
    aws iam delete-role-policy --role-name "$role" --policy-name "$p"; done
  aws iam delete-role --role-name "$role" 2>/dev/null || true
done
# Managed instances are HIDDEN from describe-instances unless IncludeManagedResources=true:
aws ec2 describe-instances --region "$R" --include-managed-resources \
  --filters Name=instance-state-name,Values=pending,running Name=tag-key,Values=aws:lambda:capacity-provider \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text 2>/dev/null || \
  echo "(aws-cli too old for --include-managed-resources; check with boto3 IncludeManagedResources=True)"
echo "teardown done"
