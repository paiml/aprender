"""Fargate scale-from-zero timeline: RunTask -> ECS lifecycle timestamps -> port open -> first MCP answer -> stop.

  uv run --with boto3 python tools/fargate_cold.py kev-spike-fargate-baked --rounds 3 --out results/fargate-baked.jsonl

Every time is seconds after the RunTask call (client clock); ECS timestamps are converted to the same origin.
"""
import argparse, json, socket, subprocess, sys, time
import boto3

ap = argparse.ArgumentParser()
ap.add_argument("taskdef"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--out", required=True)
ap.add_argument("--cluster", default="kev-spike"); ap.add_argument("--region", default="us-east-1")
ap.add_argument("--subnets", default="subnet-REDACTED1,subnet-REDACTED2,subnet-REDACTED3")
ap.add_argument("--sg", default="sg-REDACTED"); ap.add_argument("--row", type=int, default=1)
a = ap.parse_args()
ecs = boto3.client("ecs", region_name=a.region); ec2 = boto3.client("ec2", region_name=a.region)

def rel(ts, t0):
    return round(ts.timestamp() - t0, 2) if ts else None

def one(r):
    t0 = time.time()
    run = ecs.run_task(cluster=a.cluster, taskDefinition=a.taskdef, launchType="FARGATE", count=1,
                       networkConfiguration={"awsvpcConfiguration": {"subnets": a.subnets.split(","), "securityGroups": [a.sg], "assignPublicIp": "ENABLED"}},
                       tags=[{"key": "spike", "value": "021-023"}])
    if run.get("failures"):
        return {"label": f"cold-{r}", "failures": run["failures"]}
    arn = run["tasks"][0]["taskArn"]
    ip, task = None, None
    try:
        while True:
            task = ecs.describe_tasks(cluster=a.cluster, tasks=[arn])["tasks"][0]
            if task["lastStatus"] == "STOPPED":
                return {"label": f"cold-{r}", "stopped": task.get("stoppedReason"), "containers": [c.get("reason") for c in task["containers"]]}
            if not ip:
                eni = next((d["value"] for att in task.get("attachments", []) for d in att["details"] if d["name"] == "networkInterfaceId"), None)
                if eni:
                    assoc = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni])["NetworkInterfaces"][0].get("Association")
                    ip = assoc and assoc.get("PublicIp")
            if ip and task["lastStatus"] == "RUNNING":
                try:
                    socket.create_connection((ip, 8080), timeout=1).close()
                    break
                except OSError:
                    pass
            time.sleep(0.5)
        port_open = round(time.time() - t0, 2)
        out = subprocess.run([sys.executable, "tools/probe.py", "http", f"http://{ip}:8080/", "--row", str(a.row), "--warm", "0", "--label", f"cold-{r}"],
                             capture_output=True, text=True)
        first = json.loads(out.stdout)
        first_answer = round(time.time() - t0, 2)
        warm = json.loads(subprocess.run([sys.executable, "tools/probe.py", "http", f"http://{ip}:8080/", "--row", str(a.row), "--warm", "5", "--label", f"warm-{r}"],
                                         capture_output=True, text=True).stdout)
        task = ecs.describe_tasks(cluster=a.cluster, tasks=[arn])["tasks"][0]
        return {"label": f"cold-{r}", "taskdef": a.taskdef, "task": arn.split("/")[-1], "az": task.get("availabilityZone"),
                "timeline_s": {"run_task": 0.0, "pull_started": rel(task.get("pullStartedAt"), t0), "pull_stopped": rel(task.get("pullStoppedAt"), t0),
                               "container_started": rel(task.get("startedAt"), t0), "port_open (weights loaded)": port_open, "first_answer": first_answer},
                "first": first, "warm": warm}
    finally:
        ecs.stop_task(cluster=a.cluster, task=arn, reason="spike 022 measurement done")

with open(a.out, "a") as f:
    for r in range(a.rounds):
        rec = one(r)
        f.write(json.dumps(rec) + "\n"); f.flush()
        s = rec.get("first", {}).get("server", {}); w = rec.get("warm", {}).get("server", {})
        print(rec["label"], rec.get("timeline_s") or rec, "decision", s.get("decision_ms"), "warm_p50", w.get("warm_p50_ms"),
              "dp", s.get("max_abs_dp"), "cpu", s.get("host", {}).get("cpu_guess"), flush=True)
