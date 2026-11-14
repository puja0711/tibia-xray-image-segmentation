
import csv
from pathlib import Path

csv_folder = Path("data/CSVs")

# Update training and validation CSV files
for filename in ["train.csv", "validation.csv"]:

    csv_path = csv_folder / filename

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    for row in rows:
        if row["masks"] != "None":
            image_name = Path(row["masks"]).name
            row["masks"] = f"data/processed_masks/{image_name}"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["xrays", "masks"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(filename, "updated successfully!")

print("CSV paths updated!")
