# src/utils.py

import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_episode(dataset, n_way, n_shot, n_query):
    """
    Sample a few-shot episode from the dataset.

    Returns:
        support_x, support_y, query_x, query_y tensors.
    """
    class_idxs = random.sample(range(len(dataset.classes)), n_way)
    support_x, support_y, query_x, query_y = [], [], [], []

    for cls_idx in class_idxs:
        # Filter all samples of this class
        cls_samples = [s for s in dataset.samples if s[1] == cls_idx]
        selected = random.sample(cls_samples, n_shot + n_query)
        support = selected[:n_shot]
        query   = selected[n_shot:]

        for path, label in support:
            img = Image.open(path).convert('RGB')
            img = dataset.transform(img)
            support_x.append(img)
            support_y.append(label)

        for path, label in query:
            img = Image.open(path).convert('RGB')
            img = dataset.transform(img)
            query_x.append(img)
            query_y.append(label)

    support_x = torch.stack(support_x)
    query_x   = torch.stack(query_x)
    support_y = torch.tensor(support_y)
    query_y   = torch.tensor(query_y)

    return support_x, support_y, query_x, query_y


def save_checkpoint(model, optimizer, episode, output_dir):
    """Save model + optimizer state to a checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        'episode': episode,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    path = os.path.join(output_dir, f'checkpoint_{episode}.pth')
    torch.save(ckpt, path)


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load model (and optionally optimizer) state from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('episode', None)


def log_metrics(logs, csv_path):
    """Save training logs (episode, loss, accuracy) to CSV."""
    df = pd.DataFrame(logs, columns=['episode', 'loss', 'accuracy'])
    df.to_csv(csv_path, index=False)


def compute_confidence_interval(acc_list, confidence=0.95):
    """Compute mean accuracy and 95% CI from a list of accuracies."""
    a = np.array(acc_list)
    mean = a.mean()
    sem  = a.std(ddof=1) / np.sqrt(len(a))
    h    = sem * 1.96  # 95% CI
    return mean, h
