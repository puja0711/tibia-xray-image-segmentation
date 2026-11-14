
import torch


def dice_score(predictions, targets, threshold=0.5, smooth=1e-6):

    predictions = (predictions > threshold).float()
    targets = targets.float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (2.0 * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )

    return dice.item()
