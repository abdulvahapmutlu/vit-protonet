# src/models/protonet_vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ProtonetViT(nn.Module):
    """
    Prototypical Network with a ViT backbone.
    """
    def __init__(self, backbone_name='vit_small_patch16_224', pretrained=True):
        super(ProtonetViT, self).__init__()
        # Load a ViT backbone without classification head
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.embed_dim = self.backbone.num_features

    def forward(self, x):
        # Returns the CLS token embeddings
        return self.backbone.forward_features(x)

    def loss(self, support, query, n_way, n_shot, n_query):
        """
        Computes prototypical loss and accuracy for one episode.

        Args:
            support: Tensor of shape [n_way*n_shot, C, H, W]
            query:   Tensor of shape [n_way*n_query, C, H, W]
            n_way:   number of classes per episode
            n_shot:  support examples per class
            n_query: query examples per class

        Returns:
            loss:   cross-entropy loss
            acc:    accuracy (float)
            preds:  predicted class indices for query examples
        """
        # Extract embeddings
        z_support = self.forward(support)  # [n_way*n_shot, embed_dim]
        z_query   = self.forward(query)    # [n_way*n_query, embed_dim]

        # Compute prototypes by averaging support embeddings
        prototypes = z_support.view(n_way, n_shot, -1).mean(dim=1)  # [n_way, embed_dim]

        # Compute squared Euclidean distances between query and prototypes
        dists = torch.cdist(z_query, prototypes)  # [n_way*n_query, n_way]

        # Create target labels for query set
        target_inds = torch.arange(n_way, device=dists.device)
        target_inds = target_inds.unsqueeze(1).expand(n_way, n_query).reshape(-1)

        # Negative distances as logits
        logits = -dists
        loss   = F.cross_entropy(logits, target_inds)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc   = torch.eq(preds, target_inds).float().mean()
        return loss, acc, preds
