# src/datasets/cub_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class CUBDataset(Dataset):
    """
    Caltech-UCSD Birds-200-2011 Dataset loader.
    Expects directory structure:
      root/
        class_001/
          img1.jpg
          img2.jpg
        class_002/
        ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # discover classes
        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # gather (path, label) pairs
        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
