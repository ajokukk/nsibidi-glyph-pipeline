"""
PatchNCE loss components for CUT training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampleMLP(nn.Module):
    """Projects feature vectors sample at spatial locations into embedding space."""

    def __init__(self, num_patches: int = 256, embed_dim: int = 256):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.mlps = nn.ModuleDict()  # built lazily keyed by "layer_{idx}"

    def _get_mlp(self, layer_key: str, in_channels: int, device):
        if layer_key not in self.mlps:
            mlp = nn.Sequential(
                nn.Linear(in_channels, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
            ).to(device)
            # Weight init for MLP
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, 0.0, 0.02)
                    torch.nn.init.constant_(m.bias, 0.0)
            self.mlps[layer_key] = mlp
        return self.mlps[layer_key]

    def forward(self, feats: list, patch_ids: list = None):
        """Returns (embeddings, patch_ids)."""
        return_ids = []
        return_feats = []

        for layer_idx, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

            if patch_ids is not None:
                patch_id = patch_ids[layer_idx]
            else:
                n_patches = min(self.num_patches, H * W)
                patch_id = torch.randperm(H * W, device=feat.device)[:n_patches]

            sampled = feat_flat[:, patch_id, :]  # (B, n_patches, C)
            sampled = sampled.reshape(-1, C)     # (B*n_patches, C)

            mlp = self._get_mlp(f"layer_{layer_idx}", C, feat.device)
            embedded = mlp(sampled)
            embedded = F.normalize(embedded, dim=1)  # L2 normalize

            return_feats.append(embedded)
            return_ids.append(patch_id)

        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    """InfoNCE loss between corresponding patch embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q_layers: list, feat_k_layers: list) -> torch.Tensor:
        total_loss = 0.0
        for feat_q, feat_k in zip(feat_q_layers, feat_k_layers):
            n = feat_q.shape[0]
            # Similarity matrix
            logits = torch.mm(feat_q, feat_k.t()) / self.temperature
            # Positives are diagonal (i==j)
            labels = torch.arange(n, device=feat_q.device)
            total_loss = total_loss + self.cross_entropy(logits, labels)
        return total_loss / len(feat_q_layers)