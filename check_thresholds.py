
import torch
import numpy as np

from dataset import TibiaDataset
from models.unet import UNet


device = torch.device("cpu")

dataset = TibiaDataset("data/CSVs/validation.csv")

model = UNet().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

results = {t: [] for t in thresholds}

with torch.no_grad():

    for image, mask in dataset:

        image = image.unsqueeze(0).to(device)
        mask = mask.to(device)

        output = model(image)
        probability = torch.sigmoid(output).squeeze()

        target = mask.squeeze()

        for threshold in thresholds:

            prediction = (probability > threshold).float()

            intersection = (prediction * target).sum()

            dice = (
                2 * intersection + 1e-6
            ) / (
                prediction.sum() + target.sum() + 1e-6
            )

            results[threshold].append(dice.item())


print("\nThreshold comparison")
print("--------------------")

for threshold in thresholds:

    average = np.mean(results[threshold])

    print(
        f"Threshold: {threshold:.1f} | "
        f"Average Dice: {average:.4f} "
        f"({average * 100:.2f}%)"
    )
