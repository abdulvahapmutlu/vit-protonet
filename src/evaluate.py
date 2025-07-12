# src/evaluate.py

import argparse
import torch
from tqdm import tqdm
from torchvision import transforms
from datasets.cub_dataset import CUBDataset
from datasets.fc100_dataset import FC100Dataset
from datasets.mini_imagenet_dataset import MiniImageNetDataset
from datasets.cifar_fs_dataset import CIFARFSDataset
from models.protonet_vit import ProtonetViT
from utils import set_seed, get_episode, load_checkpoint, compute_confidence_interval


def main():
    parser = argparse.ArgumentParser(description='Evaluate Prototypical ViT')
    parser.add_argument('--dataset', choices=['mini','cub','cifarfs','fc100'], required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--queries', type=int, default=15)
    parser.add_argument('--eval_episodes', type=int, default=1000)
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

    # Model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProtonetViT().to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    # Collect accuracies
    acc_list = []
    with torch.no_grad():
        for _ in tqdm(range(args.eval_episodes)):
            support_x, support_y, query_x, query_y = get_episode(
                dataset, args.ways, args.shots, args.queries
            )
            support_x, query_x = support_x.to(device), query_x.to(device)
            _, acc, _ = model.loss(
                support_x, query_x, args.ways, args.shots, args.queries
            )
            acc_list.append(acc.item())

    mean_acc, conf_interval = compute_confidence_interval(acc_list)
    print(f'{args.dataset} — {args.ways}-way {args.shots}-shot: '
          f'{mean_acc*100:.2f}% ± {conf_interval*100:.2f}%')


if __name__ == '__main__':
    main()
