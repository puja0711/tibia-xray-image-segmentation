
import csv
from pathlib import Path

csv_folder = Path("data/CSVs")

# Read the complete dataset
with open(csv_folder / "dataset.csv", "r") as file:
    reader = csv.DictReader(file)
    dataset = list(reader)

# Split the dataset
train_data = dataset[:80]
validation_data = dataset[80:100]
test_data = dataset[100:150]

# Save the three CSV files
columns = ["xrays", "masks"]

for filename, data in [
    ("train.csv", train_data),
    ("validation.csv", validation_data),
    ("test.csv", test_data)
]:
    with open(csv_folder / filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

    print(filename, "created:", len(data), "images")

print("All CSV files created successfully!")
