
"""Normal residual patch memory."""

import torch
from torch.nn import functional as F


class ResidualPatchMemory:
    def __init__(self, patch_size=16, stride=8, max_patches=30000, k=3, chunk_size=4096, seed=1203):
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.max_patches = int(max_patches)
        self.k = int(k)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.memory = None
        self.mean = None
        self.std = None

    def _as_4d(self, maps):
        if not torch.is_tensor(maps):
            maps = torch.as_tensor(maps)
        maps = maps.float()
        if maps.ndim == 3:
            maps = maps.unsqueeze(1)
        if maps.ndim != 4:
            raise ValueError("residual maps must be [N,H,W] or [N,1,H,W]")
        return maps

    def _patches(self, maps):
        maps = self._as_4d(maps)
        patches = F.unfold(maps, kernel_size=self.patch_size, stride=self.stride)
        return patches.transpose(1, 2).reshape(-1, self.patch_size * self.patch_size)

    def fit(self, residual_maps):
        patches = self._patches(residual_maps).detach().cpu()
        self.mean = patches.mean(dim=0, keepdim=True)
        self.std = patches.std(dim=0, keepdim=True).clamp_min(1e-6)
        patches = (patches - self.mean) / self.std
        if patches.shape[0] > self.max_patches:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            idx = torch.randperm(patches.shape[0], generator=gen)[:self.max_patches]
            patches = patches[idx]
        self.memory = patches.contiguous()
        return self

    def to(self, device):
        if self.memory is not None:
            self.memory = self.memory.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

    def _distances(self, flat_patches):
        if self.memory is None:
            raise RuntimeError("memory is empty; call fit first")
        flat_patches = flat_patches.to(self.memory.device)
        flat_patches = (flat_patches - self.mean) / self.std
        scores = []
        for start in range(0, flat_patches.shape[0], self.chunk_size):
            chunk = flat_patches[start:start + self.chunk_size]
            dist = torch.cdist(chunk, self.memory)
            topk = torch.topk(dist, k=min(self.k, dist.shape[1]), largest=False, dim=1).values
            scores.append(topk.mean(dim=1))
        return torch.cat(scores, dim=0)

    def score_map(self, residual_maps):
        maps = self._as_4d(residual_maps)
        n, _, h, w = maps.shape
        patches = F.unfold(maps, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)
        flat = patches.reshape(-1, self.patch_size * self.patch_size)
        patch_scores = self._distances(flat).reshape(n, -1).to(maps.device)
        ones = torch.ones(n, self.patch_size * self.patch_size, patch_scores.shape[1], device=maps.device)
        weighted = ones * patch_scores.unsqueeze(1)
        score = F.fold(weighted, output_size=(h, w), kernel_size=self.patch_size, stride=self.stride)
        norm = F.fold(ones, output_size=(h, w), kernel_size=self.patch_size, stride=self.stride).clamp_min(1e-6)
        score = (score / norm).squeeze(1)
        image_score = torch.topk(score.flatten(1), k=max(1, int(h * w * 0.01)), dim=1).values.mean(dim=1)
        return score.detach().cpu(), image_score.detach().cpu()

    def state_dict(self):
        return {
            "memory": None if self.memory is None else self.memory.cpu(),
            "mean": None if self.mean is None else self.mean.cpu(),
            "std": None if self.std is None else self.std.cpu(),
            "patch_size": self.patch_size,
            "stride": self.stride,
            "max_patches": self.max_patches,
            "k": self.k,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        self.patch_size = int(state["patch_size"])
        self.stride = int(state["stride"])
        self.max_patches = int(state.get("max_patches", self.max_patches))
        self.k = int(state.get("k", self.k))
        self.chunk_size = int(state.get("chunk_size", self.chunk_size))
        self.seed = int(state.get("seed", self.seed))
        self.memory = state["memory"].cpu()
        self.mean = state["mean"].cpu()
        self.std = state["std"].cpu()
        return self
