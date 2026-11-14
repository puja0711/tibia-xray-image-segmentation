
import csv
from pathlib import Path
import random

# Project folders
xray_folder = Path("data/xrays")
mask_folder = Path("data/masks")
csv_folder = Path("data/CSVs")

csv_folder.mkdir(parents=True, exist_ok=True)

# Collect all 150 X-ray images
dataset = []

for i in range(150):
    xray_path = f"data/xrays/{i}.png"

    if (mask_folder / f"{i}.png").exists():
        mask_path = f"data/masks/{i}.png"
    else:
        mask_path = "None"

    dataset.append({
        "xrays": xray_path,
        "masks": mask_path
    })

# Create dataset.csv
columns = ["xrays", "masks"]

with open(csv_folder / "dataset.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(dataset)

print("dataset.csv created successfully!")
print("Total images:", len(dataset))
print("Images with masks:", sum(row["masks"] != "None" for row in dataset))
print("Images without masks:", sum(row["masks"] == "None" for row in dataset))
