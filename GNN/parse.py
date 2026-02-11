import csv
import json
import glob

files = sorted(
    glob.glob(
        "wfomi_data/json/color*_50k/domain*/tv*/1.txt.val_metrics.seed=0.multiedgefixed.json"
    )
)

print(len(files))

output_rows = []
for filepath in files:
    parts = filepath.split("/")
    color_part = parts[2]  # 'color1_50k'
    domain_part = parts[3]  # 'domain10'
    tv_part = parts[4]  # 'tv0.0'
    color_number = color_part.replace("color", "")  # '1'
    domain_number = domain_part.replace("domain", "")  # '10'
    tv_number = tv_part.replace("tv", "")  # '0.0'

    # Read the JSON file for seed=0
    with open(filepath, "r") as f:
        data0 = json.load(f)
    val_accuracy_seed0 = data0.get("val_accuracy")

    # Construct file paths for seed=1 and seed=2
    filepath_seed1 = filepath.replace("seed=0", "seed=1")
    filepath_seed2 = filepath.replace("seed=0", "seed=2")

    # Read the JSON file for seed=1
    try:
        with open(filepath_seed1, "r") as f:
            data1 = json.load(f)
        val_accuracy_seed1 = data1.get("val_accuracy")
    except FileNotFoundError:
        val_accuracy_seed1 = None

    # Read the JSON file for seed=2
    try:
        with open(filepath_seed2, "r") as f:
            data2 = json.load(f)
        val_accuracy_seed2 = data2.get("val_accuracy")
    except FileNotFoundError:
        val_accuracy_seed2 = None

    # Calculate average val_accuracy
    val_accuracies = [
        va
        for va in [val_accuracy_seed0, val_accuracy_seed1, val_accuracy_seed2]
        if va is not None
    ]
    if val_accuracies:
        average_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    else:
        average_val_accuracy = None

    expected_accuracy = 1 - (1 - float(tv_number)) / 2

    output_rows.append(
        [
            color_number,
            domain_number,
            tv_number,
            expected_accuracy,
            val_accuracy_seed0,
            val_accuracy_seed1,
            val_accuracy_seed2,
            average_val_accuracy,
        ]
    )

# Write the output to a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "color number",
            "domain number",
            "tv number",
            "expected_accuracy",
            "val_accuracy_seed0",
            "val_accuracy_seed1",
            "val_accuracy_seed2",
            "average_val_accuracy",
        ]
    )
    writer.writerows(output_rows)
