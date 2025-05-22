import pandas as pd
from torch import nn
from torch.nn.functional import mse_loss
import torch as tr
from tqdm import tqdm
import mlflow
from src.models.unet.conv_layers import N_Conv, DownBlock, OutConv
from src.metrics import compute_metrics


class SimpleCNN(nn.Module):
    def __init__(
        self,
        train_len: int,
        device: str,
        lr: float,
        verbose: bool,
        embedding_dim: int,
        features: list,
        kernels: list,
        input_length: int,
        **kwargs,
    ):
        super().__init__()

        self.device = tr.device(device)
        self.hyperparameters = {
            "train_len": train_len,
            "device": device,
            "lr": lr,
            "verbose": verbose,
        }
        self.input_length = input_length

        self.build_graph(
            embedding_dim=embedding_dim,
            features=features,
            kernels=kernels,
        )

        self.verbose = verbose
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        self.log_model()
        self.to(self.device)

    def build_graph(
        self,
        embedding_dim: int,
        features: list,
        kernels: list,
    ):

        num_convs = len(features) - 1
        self.L_min = self.input_length // (2 ** (num_convs))
        volume = [(self.input_length / 2**i) * f for i, f in enumerate(features)]
        self.out_dim = 1
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_initial_volume": 4 * self.input_length,
            "arc_latent_volume": volume[-1],
            "arc_features": features,
            "arc_kernels": kernels,
        }

        self.inc = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=features[0],
                kernel_size=kernels[0],
                padding="same",
                stride=1,
            ),
            nn.BatchNorm1d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList()
        for i in range(num_convs):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=features[i],
                        out_channels=features[i + 1],
                        kernel_size=kernels[i + 1],
                        padding="same",
                        stride=1,
                    ),
                    nn.BatchNorm1d(features[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.outc = OutConv(features[-1], self.out_dim)

    def forward(self, x):
        x = x.to(self.device)
        x = self.inc(x)
        feature_maps = [x]
        for _, conv in enumerate(self.layers):
            x = conv(x)
            feature_maps.append(x)
        x = self.outc(x)
        y_pred = nn.Sigmoid()(x)
        return y_pred

    def loss_func(self, y_pred, y):
        """yhat and y are [N, L]"""
        y = y.view(y.shape[0], -1)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        return mse_loss(y_pred, y)

    def fit(self, loader):
        self.train()

        metrics = {"loss": 0, "F1": 0, "Accuracy": 0, "Accuracy_seq": 0}
        if self.verbose:
            loader = tqdm(loader)

        for batch in loader:
            y = batch["pseudo_probing"].to(self.device)
            x_model = batch["embedding"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            y_pred = self(x_model)
            loss = self.loss_func(y_pred, y)
            metrics["loss"] += loss.item()

            batch_metrics = compute_metrics(y_pred, y, mask, binary=True)
            for k, v in batch_metrics.items():
                metrics[k] += v

            loss.backward()
            self.optimizer.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()

        metrics = {"loss": 0, "F1": 0, "Accuracy": 0, "Accuracy_seq": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:
                y = batch["pseudo_probing"].to(self.device)
                x_model = batch["embedding"].to(self.device)
                mask = batch["mask"].to(self.device)

                y_pred = self(x_model)
                loss = self.loss_func(y_pred, y)
                metrics["loss"] += loss.item()
                batch_metrics = compute_metrics(y_pred, y, mask, binary=True)

                for k, v in batch_metrics.items():
                    metrics[k] += v

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def pred(self):
        pass

    def log_model(self):
        """Logs the model architecture and hyperparameters to MLflow."""
        mlflow.log_params(self.hyperparameters)
        mlflow.log_params(self.architecture)
        mlflow.log_param("arc_num_params", sum(p.numel() for p in self.parameters()))
