import os
import glob
import json
import csv
import pandas as pd
import re

res_files = glob.glob(os.path.join("/Users/asedova/PycharmProjects/ambiguties/out_study_1/ask_for_ambiguity/mixtral-8x7B-Instruct-v0.1/temp_0.2", "*.json"))

for file in res_files:
    with open(file) as f:
        data = json.load(f)
        for d in data:
            if "Yes" in d["response"]:
                d["response"] = True
            elif "No" in d["response"]:
                d["response"] = False

            d["entity"] = re.findall('Can (.+) mean', d["prompt"])[0]

        df = pd.DataFrame(data).drop(["system_prompt", "prompt"], axis=1)
        df["response"], df["entity"] = df["entity"], df["response"]

        out_path = os.path.join("/Users/asedova/PycharmProjects/ambiguties/out_study_1/ask_for_ambiguity/processed", f"{'/'.join(file.split(os.sep)[-3:-1])}")
        os.makedirs(out_path, exist_ok=True)

        output_file = os.path.join(out_path, f"{file.split(os.sep)[-3]}.csv")

        df.to_csv(output_file, mode="a", header=False)

print("ok")