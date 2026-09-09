from collections import defaultdict
from typing import Optional

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from unet import UNet
from utilities import median_pool


DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimiser,
        train_step,
        val_step,
        callback_dict,
        device,
        identifier,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimiser = optimiser
        self.train_step = train_step
        self.val_step = val_step
        self.callback_dict = callback_dict
        self.device = device
        self.identifier = identifier
        self.state = {}

    def train(self, epochs):
        for _ in range(epochs):
            self.model.train()
            for batch in self.train_dataloader:
                loss = self.train_step(self, batch)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

    def attach_method(self, name, method):
        setattr(self, name, method)


def noise(x, noise_std=0.2, noise_res=16):
    del noise_res
    return x + noise_std * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)


def denoising(
    identifier: str,
    data: Optional[str] = None,
    lr=0.001,
    depth=4,
    wf=7,
    n_input=4,
    noise_std=0.2,
    noise_res=16,
    modality_weights=None,
    device=None,
):
    del data
    device = torch.device(device) if device is not None else DEFAULT_DEVICE

    if modality_weights is None:
        modality_weights = [1.0 / n_input] * n_input

    model = UNet(
        in_channels=n_input,
        n_classes=n_input,
        norm="group",
        up_mode="upconv",
        depth=depth,
        wf=wf,
        padding=True,
    ).to(device)

    def get_scores(trainer, batch, median_f=True):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        trainer.model.eval()
        with torch.no_grad():
            clean = x.clone().to(trainer.device)
            mask = clean.sum(dim=1, keepdim=True) > 0.01
            mask = F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95
            reconstructed = trainer.model(clean)

            err_maps = []
            for i in range(n_input):
                err = ((clean[:, i : i + 1] - reconstructed[:, i : i + 1]) * mask).abs()
                if median_f:
                    err = median_pool(err, kernel_size=5, stride=1, padding=2)
                err_maps.append(err)

            final_err = torch.zeros_like(err_maps[0])
            for err, weight in zip(err_maps, modality_weights):
                final_err += weight * err
        return final_err.cpu()

    def loss_f(batch, batch_results):
        y = batch[1] if isinstance(batch, (list, tuple)) else batch
        mask = y.sum(dim=1, keepdim=True) > 0.01
        return (torch.pow(batch_results - y, 2) * mask.float()).mean()

    def forward(batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        y = x.clone()
        x_noisy = noise(x.clone(), noise_std=noise_std, noise_res=noise_res)
        return trainer.model(x_noisy), y

    def train_step(trainer, batch):
        batch = batch.to(trainer.device)
        output, y = forward(batch)
        return loss_f([None, y], output)

    def val_step(trainer, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(trainer.device)
        output, y = forward(x)
        return {"loss": loss_f([None, y], output)}

    trainer = Trainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
        optimiser=torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001),
        train_step=train_step,
        val_step=val_step,
        callback_dict=defaultdict(list),
        device=device,
        identifier=identifier,
    )

    trainer.noise = noise
    trainer.get_scores = get_scores
    trainer.loss_f = loss_f
    trainer.forward = forward
    trainer.lr_scheduler = CosineAnnealingLR(trainer.optimiser, T_max=100)
    return trainer
