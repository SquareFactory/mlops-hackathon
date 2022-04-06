from argparse import ArgumentParser

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST


class Classifier(pl.LightningModule):
    """General class for the model."""

    def __init__(
        self, learning_rate: float = 0.001, model_name="mobilenetv3_rw", num_classes=10
    ):
        """Initialize the model."""
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate

        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=1
        )
        self.loss = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1(num_classes, average="weighted")
        self.test_cm = torchmetrics.ConfusionMatrix(num_classes)

    @staticmethod
    def add_argparse_args(parent_parser):
        """Argument parser for model."""
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--learning_rate", type=float, default=0.0005)
        parser.add_argument("--model_name",type=str,default="mobilenetv3_rw")
        return parent_parser

    def forward(self, x):
        """Forward pass."""

        return self.model(x)

    def _step(self, batch, name):
        """General step."""
        inputs, targets = batch

        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        _, preds = torch.max(outputs, 1)
        if name == "train":
            self.log("train_loss", loss)
            self.train_acc(preds, targets)
            self.log(
                "train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=False
            )
        elif name == "val":
            self.val_acc(preds, targets)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val_acc",
                self.val_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        else:
            raise ValueError(f"Invalid step name given: {name}")

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        _, preds = torch.max(outputs, 1)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_cm(preds, targets)

        return loss

    def test_epoch_end(self, outputs):
        """Metrics calculation at th end of the training."""
        self.log("test_acc", self.test_acc.compute(), sync_dist=True)
        self.log("test_f1", self.test_f1.compute(), sync_dist=True)
        print("Confusion matrix:")
        print(self.test_cm.compute().cpu().numpy())

    def configure_optimizers(self):
        """Optimizer settings."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        MNIST("./data", download=True)

    def train_dataloader(self):
        train_dataset = MNIST(
            "./data", train=True, download=False, transform=self.transform
        )
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=self.hparams.batch_size
        )

    def val_dataloader(self):
        val_dataset = MNIST(
            "./data", train=False, download=False, transform=self.transform
        )
        val_dataset, _ = train_test_split(val_dataset, test_size=0.5)
        return torch.utils.data.DataLoader(
            val_dataset, batch_size=self.hparams.batch_size
        )

    def test_dataloader(self):
        test_dataset = MNIST(
            "./data", train=False, download=False, transform=self.transform
        )
        _, test_dataset = train_test_split(test_dataset, test_size=0.5)
        return torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams.batch_size
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="lightning_logs",
        filename=f"MNIST_classifier_{args.model_name}" + "{epoch}-{val_loss:.2f}",
        monitor="val_acc",
        mode="max",
    )
    dm = MNISTDataModule()
    model = Classifier(learning_rate=args.learning_rate,model_name=args.model_name)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    trainer.test(model, dm)
