
import torch


def dice_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate the average Dice score per image.

    predictions: Model probabilities (B, 1, H, W)
    targets: Ground-truth masks (B, 1, H, W)
    """

    predictions = (predictions > threshold).float()
    targets = targets.float()

    # Calculate Dice separately for each image
    dims = (1, 2, 3)

    intersection = (predictions * targets).sum(dim=dims)

    union = predictions.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return average Dice across the batch
    return dice.mean().item()
