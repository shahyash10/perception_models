import os
import json
import csv
from datetime import datetime


current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    correct, total = 0, 0
    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                total += 1.0
                data = json.loads(line)
                answer = data.get('answer', '')[0].lower()
                model = data.get('model_id', '')
                gt_answer = data.get('gt_answer', [''])[0].lower()
                if is_number(gt_answer) and is_number(answer):
                    correct += relaxed_accuracy(float(answer), float(gt_answer))
                elif answer == gt_answer:
                    correct += 1.0
                else:
                    out_file.write(line)

    combined_data = {
        "model": model,
        "time": time_string,
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct/total,
    }
    add_data_to_csv(csv_file, combined_data)
    print(f"Model: {model}, Total: {total}, Correct: {correct}, Accuracy: {100.0 * correct/total}")
    print(f"Saved experiment data to {csv_file}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"chartqa_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Added a copy of the experiment data to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, required=True, help="Path to the answers file")
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to the output file to store the incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir)
