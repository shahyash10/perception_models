import os
import json
import csv
from datetime import datetime
import re

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def compute_metrics(jsonl_file, csv_file, extra_outdir=None):
    pred_list = []
    test_list = []

    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            answer = data.get('answer', '')
            if isinstance(answer, list):
                answer = answer[0]
            questionId = data.get('questionId', '')
            model = data.get("model_id", '')
            test_list.append({
                "questionId": int(questionId),
                "answer": answer
            })
    match = re.search(r'answers_(.*)\.jsonl', jsonl_file)
    file_path = f"./answers/docvqa_submission{match.group(1)}.json"
    with open(file_path, "w") as json_file:
        json.dump(test_list, json_file)
    combined_data = {
        "model": model,
        "time": time_string,
        "accuracy": "add here after submission",
    }
    add_data_to_csv(csv_file, combined_data)
    print(f"submission file: {file_path} is created, please submit at [https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2]\n and update the {csv_file} sheet")
    with open("./docvqa_submission_url.txt", "w") as f:
        f.write("https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2")

    # add a duplicate copy of the info to the extra_outdir
    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_submission_file = os.path.join(extra_outdir, f"docvqa_submission_{model}.json")
        with open(extra_submission_file, "w") as json_file:
            json.dump(test_list, json_file)
        with open(os.path.join(extra_outdir, "docvqa_submission_url.txt"), "w") as f:
            f.write("https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2")

        print(f"Added a copy of the submission file to {extra_submission_file}")

        extra_csv_file = os.path.join(extra_outdir, f"docvqa_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Added a copy of the submission info to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str, required=True, help="Path to the answers file")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")
    args = parser.parse_args()

    compute_metrics(args.answers_file, args.csv_file, args.extra_outdir)
