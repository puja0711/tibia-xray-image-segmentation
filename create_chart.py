import matplotlib.pyplot as plt

files = ["dataset.csv", "train.csv", "validation.csv", "test.csv"]
counts = [150, 80, 20, 50]

plt.figure(figsize=(9, 5))

bars = plt.bar(files, counts)

plt.title("Number of Samples in Each CSV File")
plt.xlabel("CSV Files")
plt.ylabel("Number of Samples")

plt.bar_label(bars, padding=3)
plt.tight_layout()

plt.savefig("csv_sample_counts.png", dpi=300)
plt.show()