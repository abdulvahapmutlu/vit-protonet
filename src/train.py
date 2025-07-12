# src/train.py

import os
import argparse
import torch
from torch import optim
from torchvision import transforms
from datasets.cub_dataset import CUBDataset
from datasets.fc100_dataset import FC100Dataset
from datasets.mini_imagenet_dataset import MiniImageNetDataset
from datasets.cifar_fs_dataset import CIFARFSDataset
from models.protonet_vit import ProtonetViT
from utils import set_seed, get_episode, save_checkpoint, log_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Prototypical ViT')
    parser.add_argument('--dataset', choices=['mini','cub','cifarfs','fc100'], required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--queries', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Select dataset class
    if args.dataset == 'cub':
        DatasetClass = CUBDataset
    elif args.dataset == 'fc100':
        DatasetClass = FC100Dataset
    elif args.dataset == 'mini':
        DatasetClass = MiniImageNetDataset
    else:
        DatasetClass = CIFARFSDataset

    dataset = DatasetClass(args.data_root, transform)

    # Model, optimizer, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProtonetViT().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logs = []

    # Training loop
    for episode in range(1, args.episodes + 1):
        support_x, support_y, query_x, query_y = get_episode(
            dataset, args.ways, args.shots, args.queries
        )
        support_x, query_x = support_x.to(device), query_x.to(device)

        model.train()
        loss, acc, _ = model.loss(
            support_x, query_x, args.ways, args.shots, args.queries
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logs.append((episode, loss.item(), acc.item()))

        if episode % 100 == 0:
            print(f'Episode {episode}/{args.episodes} — '
                  f'Loss: {loss.item():.4f}, Acc: {acc.item()*100:.2f}%')

        # Save intermediate checkpoints
        if episode % (args.episodes // 5) == 0:
            save_checkpoint(model, optimizer, episode, args.output_dir)

    # Final checkpoint & logs
    save_checkpoint(model, optimizer, args.episodes, args.output_dir)
    log_metrics(logs, os.path.join(args.output_dir, 'training_log.csv'))


if __name__ == '__main__':
    main()
