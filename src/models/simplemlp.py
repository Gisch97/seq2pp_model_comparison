import pandas as pd
from torch import nn
from torch.nn.functional import mse_loss
import torch as tr
from tqdm import tqdm
import mlflow
from src.models.unet.conv_layers import N_Conv, DownBlock, OutConv
from src.metrics import compute_metrics


class SimpleMLP(nn.Module):
    def __init__(
        self,
        train_len: int,
        device: str,
        lr: float,
        verbose: bool,
        embedding_dim: int,
        features: list,
        input_length: int,
        **kwargs
    ):
        super().__init__()
        self.device = tr.device(device)
        self.verbose = verbose
        self.input_length = input_length
        self.embedding_dim = embedding_dim

        # Guardar hiperparámetros y arquitectura
        self.hyperparameters = {
            "train_len": train_len,
            "device": device,
            "lr": lr,
            "verbose": verbose,
        }
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_features": features,
        }

        # Construir MLP análogo al convolucional
        self._build_mlp(features)

        # Optimizador
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # Log y mover a device
        self.log_model()
        self.to(self.device)

    def _build_mlp(self, features: list):
        """
        Construye el MLP con ModuleList:
        - flatten de [batch, embedding_dim, input_length] a un vector.
        - capas según 'features'.
        - capa final para restaurar longitud de salida.
        - salida [batch, input_length, 1].
        """
        in_dim = self.embedding_dim * self.input_length
        self.layers = nn.ModuleList()
        for hidden_dim in features:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        # capa final de restauración
        self.restore = nn.Linear(in_dim, self.input_length)

    def forward(self, x):
        """
        x: [batch, embedding_dim, input_length]
        retorna: [batch, input_length, 1]
        """
        batch = x.size(0)
        x = x.to(self.device)
        x = x.view(batch, -1)
        # pasar por MLP
        for layer in self.layers:
            x = layer(x)

        x = self.restore(x)
        x = tr.sigmoid(x)
        return x.unsqueeze(1)

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
