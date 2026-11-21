
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from dataset import TibiaDataset
from models.unet_lext import UNetLext
from dice_score import dice_score


# 1. Configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Evaluation device:", DEVICE)


# 2. Load the best saved model

model = UNetLext(
    basic_width=24,
    depth=6,
    center_depth=2,
    input_channels=1,
    output_channels=1,
    convolution="regular",
    activation="relu"
).to(DEVICE)

checkpoint = torch.load(
    "checkpoints/best_model_lext.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded best model from epoch:", checkpoint["epoch"])
print("Best saved validation Dice:", checkpoint["best_validation_dice"])


# 3. Create output folders

Path("outputs/predictions/validation").mkdir(
    parents=True, exist_ok=True
)

Path("outputs/predictions/test").mkdir(
    parents=True, exist_ok=True
)

Path("outputs/plots").mkdir(
    parents=True, exist_ok=True
)


# 4. Evaluate validation images

validation_dataset = TibiaDataset("data/CSVs/validation.csv")

validation_scores = []

print("\nEvaluating validation images...")

with torch.no_grad():

    for index in range(len(validation_dataset)):

        image, mask = validation_dataset[index]

        image = image.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)

        output = model(image)
        prediction = torch.sigmoid(output)

        score = dice_score(prediction, mask, threshold=0.5)
        validation_scores.append(score)

        predicted_mask = (
            prediction.squeeze().cpu().numpy() > 0.5
        ).astype(np.uint8)

        actual_mask = mask.squeeze().cpu().numpy()

        # Save predicted validation mask
        Image.fromarray(
            predicted_mask * 255
        ).save(
            f"outputs/predictions/validation/{index + 80}.png"
        )

        # Save visual comparison
        if index < 5:

            original_image = image.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(original_image, cmap="gray")
            axes[0].set_title("Original X-ray")

            axes[1].imshow(actual_mask, cmap="gray")
            axes[1].set_title("Actual mask")

            axes[2].imshow(predicted_mask, cmap="gray")
            axes[2].set_title("Predicted mask")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()

            plt.savefig(
                f"outputs/plots/validation_{index + 80}.png"
            )

            plt.close()


average_dice = np.mean(validation_scores)

print("\nValidation results")
print("------------------")
print("Number of images:", len(validation_dataset))
print(f"Average Dice score: {average_dice:.4f}")
print(f"Average Dice percentage: {average_dice * 100:.2f}%")


# 5. Generate predictions for 50 test images

test_data = pd.read_csv("data/CSVs/test.csv")

print("\nGenerating test predictions...")

with torch.no_grad():

    for index, row in test_data.iterrows():

        image = Image.open(row["xrays"]).convert("L")

        image_array = np.array(
            image, dtype=np.float32
        ) / 255.0

        image_tensor = torch.tensor(
            image_array
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        output = model(image_tensor)

        prediction = torch.sigmoid(output)

        predicted_mask = (
            prediction.squeeze().cpu().numpy() > 0.5
        ).astype(np.uint8)

        Image.fromarray(
            predicted_mask * 255
        ).save(
            f"outputs/predictions/test/{row['xrays'].split('/')[-1]}"
        )

print("Test images processed:", len(test_data))
print("Test predictions saved successfully!")

print("\nEvaluation completed!")
