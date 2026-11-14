
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import TibiaDataset
from models.unet import UNet
from dice_score import dice_score


# -------------------------------
# 1. Configuration
# -------------------------------

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", DEVICE)


# -------------------------------
# 2. Load datasets
# -------------------------------

train_dataset = TibiaDataset("data/CSVs/train.csv")
validation_dataset = TibiaDataset("data/CSVs/validation.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

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

model = UNet().to(DEVICE)

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

    predictions = torch.sigmoid(predictions)

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (2.0 * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )

    return 1.0 - dice


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

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            loss_dice = calculate_dice_loss(outputs, masks)

            loss = 0.5 * loss_bce + 0.5 * loss_dice

            validation_loss += loss.item()

            predictions = torch.sigmoid(outputs)

            batch_dice = dice_score(predictions, masks)

            validation_dice += batch_dice

    average_validation_loss = (
        validation_loss / len(validation_loader)
    )

    average_validation_dice = (
        validation_dice / len(validation_loader)
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
            "checkpoints/best_model.pth"
        )

        print("Best model saved!")

print("\nTraining completed!")
print(f"Best validation Dice: {best_dice:.4f}")
print("Model saved in checkpoints/best_model.pth")
