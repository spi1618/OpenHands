# Run evals of swe bench output.jsonl using the sb cli (assumed conda env was mr-env, not modal-eval-new)

import os
import sys
import json
import subprocess
import datetime

# automatically(?) save to same directory
OUTPUT_JSONL_PATH = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-v3_maxiter_100_N_v0.43.0-no-hint-run_1_20250916_102755/output.jsonl"
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# extract output directory from output jsonl path
OUTPUT_DIR = os.path.dirname(OUTPUT_JSONL_PATH)

# Run the command from the OpenHands directory
os.chdir("/home/sophiapi/model-routing/OpenHands")

# convert to json
cmd = [
    "poetry",
    "run",
    "python3",
    "evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py",
    OUTPUT_JSONL_PATH
]
result = subprocess.run(cmd, capture_output=True, text=True)
swe_file = OUTPUT_JSONL_PATH.replace('.jsonl', '.swebench.jsonl')
print(f"Output jsonl --> swebench file: {swe_file}")

with open(swe_file) as f: f = f.readlines()
output = [json.loads(line) for line in f]
out_file = OUTPUT_JSONL_PATH.replace("output.jsonl", "preds.json")
with open(out_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"Output jsonl --> json file: {out_file}")

# submit using the sb cli
print(f"Submitting using the sb cli...")
cmd = [
    "sb-cli",
    "submit",
    "swe-bench_verified", # swe-bench_verified or swe-gym-train
    "test",
    "--predictions_path",
    out_file,
    "--run_id",
    RUN_ID,
    "--output_dir",
    OUTPUT_DIR
]
result = subprocess.run(cmd, capture_output=True, text=True)

# get report using the sb cli
print(f"Getting report using the sb cli...")
cmd = [
    "sb-cli",
    "get-report",
    "swe-bench_verified",
    "test",
    RUN_ID, 
    "-o",
    OUTPUT_DIR
]
result = subprocess.run(cmd, capture_output=True, text=True)

print("Finished.")
