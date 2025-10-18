# basic models for face angle estimation

import torch
import pytorch_lightning as pl

from torch import nn
from transformers import ResNetModel, AutoImageProcessor

from huggingface_hub import PyTorchModelHubMixin


class EstimateAngle(pl.LightningModule, PyTorchModelHubMixin):
    """Uses architecture that is selected for well-behaved MSE curves.
    """
    def __init__(self, base_model="microsoft/resnet-18", lr=3E-4):
        super().__init__()
        # !!!!!!!!!! dev: what to seed?

        # initialize from an untrained model 
        self.base_model = ResNetModel.from_pretrained(base_model)
        self.preprocess = AutoImageProcessor.from_pretrained(base_model, use_fast=True)

        output_size = self.base_model.config.hidden_sizes[-1]
        self.decoder = nn.Sequential(nn.Linear(output_size, output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size, 2)
                                     )

        self.cosSim = nn.CosineSimilarity()
        self.loss = lambda pred, target: torch.mean(1. - self.cosSim(pred, target))
        self.save_hyperparameters()

    def decodeAngles(self, encodings):
        return torch.atan2(encodings[:, 1], encodings[:, 0])

    def forward(self, images):
        # For now, I'm doing the processing within the model
        processed = self.preprocess(images, return_tensors="pt")['pixel_values']
        embedding = self.base_model(processed).pooler_output.squeeze()

        return self.decoder(embedding)

    def training_step(self, batch, batchidx=None):
        images, targets = batch

        predictions = self.forward(images)

        loss = self.loss(predictions, targets)
        self.log('train/loss', loss.item())

        return loss

    def validation_step(self, batch, batchidx=None):
        images, targets = batch

        predictions = self.forward(images)
        loss = self.loss(predictions, targets)

        self.log('eval/loss', loss.item())

        norm = torch.norm(predictions, dim=1).mean()
        self.log('eval/norm', norm.item())

        # !!!!! Also log raw angle distances

    def test_step(self, batch, batchidx=None):
        images, targets = batch

        predictions = self.forward(images)

        loss = self.loss(predictions, targets)
        self.log('test/loss', loss.item())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def freeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
