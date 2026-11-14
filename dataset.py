
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class TibiaDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # Load X-ray image
        image = Image.open(row["xrays"]).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        # Load binary segmentation mask
        mask = Image.open(row["masks"]).convert("L")
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Convert to PyTorch tensors
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
