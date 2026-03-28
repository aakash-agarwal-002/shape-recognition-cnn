import torch
from torch.utils.data import Dataset
import numpy as np
from .augment import augment_points
from ..utils.preprocessing import points_to_model_input

class StrokeDataset(Dataset):
    def __init__(self, data, label_names, label_to_idx, indices=None, augment=False):
        self.data = data
        self.label_names = label_names
        self.label_to_idx = label_to_idx
        self.indices = indices if indices is not None else list(range(len(data)))
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.data[self.indices[idx]]
        source = sample.get("source", "synthetic")
        points = sample["points"]
        label = sample["label"]
        
        # Determine label name and index
        if isinstance(label, int):
            label_idx = label
            label_name = self.label_names[label_idx]
        else:
            label_name = label
            label_idx = self.label_to_idx[label]

        if self.augment:
            points = augment_points(points, source, label_name)

        line_width = np.random.randint(1, 3) if self.augment else 2
        img = points_to_model_input(points, line_width=line_width)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label_idx), source
