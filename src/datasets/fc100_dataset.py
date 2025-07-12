# src/datasets/fc100_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class FC100Dataset(Dataset):
    """
    FC100 (subset of CIFAR-100) loader.
    Directory structure:
      root/
        class_001/
          img1.png
          img2.png
        class_002/
        ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
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
