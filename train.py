
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np


from dataset import TibiaDataset
from models.unet_lext import UNetLext
from dice_score import dice_score


# -------------------------------
# 1. Configuration
# -------------------------------
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", DEVICE)


# -------------------------------
# 2. Load datasets
# -------------------------------

train_dataset = TibiaDataset("data/CSVs/train.csv")
validation_dataset = TibiaDataset("data/CSVs/validation.csv")

# Set up a reproducible shuffle generator
generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

# Validation DataLoader
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# -------------------------------
# 3. Initialize model
# -------------------------------

model = UNetLext(
    basic_width=24,
    depth=6,
    center_depth=2,
    input_channels=1,
    output_channels=1,
    convolution="regular",
    activation="relu"
).to(DEVICE)

# Binary Cross Entropy loss
bce_loss = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------------
# 4. Dice loss function
# -------------------------------

def calculate_dice_loss(predictions, targets, smooth=1e-6):
    """
    Calculate the average Dice loss separately for each image.
    """

    predictions = torch.sigmoid(predictions)
    targets = targets.float()

    dims = (1, 2, 3)

    intersection = (predictions * targets).sum(dim=dims)

    union = predictions.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice.mean()

# -------------------------------
# 5. Prepare checkpoint folder
# -------------------------------

Path("checkpoints").mkdir(exist_ok=True)

best_dice = 0.0


# -------------------------------
# 6. Training loop
# -------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss_bce = bce_loss(outputs, masks)
        loss_dice = calculate_dice_loss(outputs, masks)

        loss = 0.5 * loss_bce + 0.5 * loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_train_loss = running_loss / len(train_loader)


    # -------------------------------
    # 7. Validation
    # -------------------------------

    model.eval()

    validation_dice = 0.0
    validation_loss = 0.0
    validation_images = 0

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            loss_dice = calculate_dice_loss(outputs, masks)

            loss = 0.5 * loss_bce + 0.5 * loss_dice

            validation_loss += loss.item()
            validation_images += images.size(0)
            
            predictions = torch.sigmoid(outputs)

            batch_dice = dice_score(predictions, masks)

            validation_dice += batch_dice * images.size(0)

    average_validation_loss = (
        validation_loss / len(validation_loader)
    )

    average_validation_dice = (
        validation_dice / validation_images
    )

    # -------------------------------
    # 8. Display results
    # -------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Validation Loss: {average_validation_loss:.4f} "
        f"Validation Dice: {average_validation_dice:.4f}"
    )


    # -------------------------------
    # 9. Save best model
    # -------------------------------

    if average_validation_dice > best_dice:

        best_dice = average_validation_dice

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_dice": best_dice
            },
            "checkpoints/best_model_lext.pth"
        )

        print("Best model saved!")

print("\nTraining completed!")
print(f"Best validation Dice: {best_dice:.4f}")
print("Model saved in checkpoints/best_model_lext.pth")
