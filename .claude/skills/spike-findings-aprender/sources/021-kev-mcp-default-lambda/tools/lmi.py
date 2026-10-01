"""Spike 023: Lambda Managed Instances minimum footprint + Graviton latency.

  uv run --with boto3 python tools/lmi.py up        # capacity provider + function + publish (times each step)
  uv run --with boto3 python tools/lmi.py scale 1   # PutFunctionScalingConfig min=max=N on the published version
  uv run --with boto3 python tools/lmi.py census    # EC2 instances the capacity provider runs (type, AZ, launch time)
  uv run --with boto3 python tools/lmi.py down      # scale 0, delete function + capacity provider

Records every step to results/lmi.jsonl with seconds since the step began.
"""
import json, sys, time
import boto3

R = "us-east-1"; A = "AWS_ACCOUNT_ID"; CP = "kev-spike-cp"; FN = "kev-spike-lmi"
SUBNETS = ["subnet-REDACTED1", "subnet-REDACTED2", "subnet-REDACTED4"]  # 1a, 1b, 1d (LMI refuses us-east-1c)
SG = "sg-REDACTED"
lam = boto3.client("lambda", region_name=R); ec2 = boto3.client("ec2", region_name=R)

def log(rec):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(rec, default=str), flush=True)
    with open("results/lmi.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")

def wait(pred, what, t0, every=5, limit=1800):
    while time.time() - t0 < limit:
        ok, info = pred()
        if ok:
            return info
        time.sleep(every)
    raise TimeoutError(what)

def census():
    res = ec2.describe_instances(IncludeManagedResources=True, Filters=[{"Name": "vpc-id", "Values": ["vpc-REDACTED"]}, {"Name": "operator.managed", "Values": ["true"]},
                                          {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "shutting-down"]}])
    out = []
    for r in res["Reservations"]:
        for i in r["Instances"]:
            tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
            out.append({"id": i["InstanceId"], "type": i["InstanceType"], "az": i["Placement"]["AvailabilityZone"], "state": i["State"]["Name"],
                        "launched": i["LaunchTime"].isoformat(), "operator": i.get("Operator", {}), "tags": {k: v for k, v in tags.items() if "lambda" in k.lower() or k == "spike"}})
    return out

def up(mem_mb=16384, gib_per_vcpu=2.0):
    t0 = time.time()
    try:
        lam.create_capacity_provider(
            CapacityProviderName=CP,
            VpcConfig={"SubnetIds": SUBNETS, "SecurityGroupIds": [SG]},
            PermissionsConfig={"CapacityProviderOperatorRoleArn": f"arn:aws:iam::{A}:role/kev-spike-lmi-operator"},
            InstanceRequirements={"Architectures": ["arm64"]},
            CapacityProviderScalingConfig={"ScalingMode": "Auto", "MaxVCpuCount": 64},
            Tags={"spike": "021-023"})
    except lam.exceptions.ResourceConflictException:
        pass
    wait(lambda: (lam.get_capacity_provider(CapacityProviderName=CP)["CapacityProvider"]["State"] == "Active", None), "cp active", t0)
    cp_arn = lam.get_capacity_provider(CapacityProviderName=CP)["CapacityProvider"]["CapacityProviderArn"]
    log({"step": "capacity provider Active", "s": round(time.time() - t0, 1), "instances": census()})
    t1 = time.time()
    code = open("build/lambda-s3.zip", "rb").read()
    lam.create_function(FunctionName=FN, Runtime="provided.al2023", Architectures=["arm64"], Handler="bootstrap", Code={"ZipFile": code},
                        Role=f"arn:aws:iam::{A}:role/kev-spike-lambda", MemorySize=mem_mb, Timeout=900,
                        Environment={"Variables": {"KEV_GGUF_S3": "s3://kev-spike-021-AWS_ACCOUNT_ID/kev-0.8b-merged-f32.gguf", "PLATFORM": "lmi",
                                                   "RAYON_NUM_THREADS": str(int(mem_mb / 1024 / gib_per_vcpu)), "KEV_LOCAL_DIR": "/tmp"}},
                        CapacityProviderConfig={"LambdaManagedInstancesCapacityProviderConfig": {
                            "CapacityProviderArn": cp_arn, "PerExecutionEnvironmentMaxConcurrency": 1, "ExecutionEnvironmentMemoryGiBPerVCpu": gib_per_vcpu}},
                        Tags={"spike": "021-023"})
    lam.get_waiter("function_active_v2").wait(FunctionName=FN)
    log({"step": "function created", "s": round(time.time() - t1, 1)})
    t2 = time.time()
    v = lam.publish_version(FunctionName=FN)["Version"]
    def active():
        c = lam.get_function_configuration(FunctionName=FN, Qualifier=v)
        return c["State"] in ("Active", "Failed"), c
    c = wait(active, "version active", t2, every=10)
    log({"step": f"version {v} {c['State']}", "s": round(time.time() - t2, 1), "state_reason": c.get("StateReason"),
         "scaling": lam.get_function_scaling_config(FunctionName=FN, Qualifier=v).get("FunctionScalingConfig") if c["State"] == "Active" else None,
         "instances": census()})

def scale(n):
    v = max(int(x["Version"]) for x in lam.list_versions_by_function(FunctionName=FN)["Versions"] if x["Version"] != "$LATEST")
    t0 = time.time()
    lam.put_function_scaling_config(FunctionName=FN, Qualifier=str(v), FunctionScalingConfig={"MinExecutionEnvironments": n, "MaxExecutionEnvironments": max(n, 1) if n else 0})
    log({"step": f"scale min=max={n} on v{v}", "s": 0, "applied": lam.get_function_scaling_config(FunctionName=FN, Qualifier=str(v)), "instances": census()})

def down():
    try:
        for x in lam.list_versions_by_function(FunctionName=FN)["Versions"]:
            if x["Version"] != "$LATEST":
                lam.put_function_scaling_config(FunctionName=FN, Qualifier=x["Version"], FunctionScalingConfig={"MinExecutionEnvironments": 0, "MaxExecutionEnvironments": 0})
        lam.delete_function(FunctionName=FN)
    except lam.exceptions.ResourceNotFoundException:
        pass
    t0 = time.time()
    while True:
        try:
            lam.delete_capacity_provider(CapacityProviderName=CP); break
        except lam.exceptions.ResourceNotFoundException:
            break
        except Exception as e:  # versions still detaching
            if time.time() - t0 > 900: raise
            time.sleep(15)
    log({"step": "down requested", "s": round(time.time() - t0, 1), "instances": census()})

if __name__ == "__main__":
    cmd = sys.argv[1]
    {"up": up, "census": lambda: log({"step": "census", "instances": census()}), "down": down}.get(cmd, lambda: scale(int(sys.argv[2])))()
