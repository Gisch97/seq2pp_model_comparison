import pandas as pd
from torch import nn
from torch.nn.functional import mse_loss
import torch as tr
from tqdm import tqdm
import mlflow
from src.models.unet.conv_layers import N_Conv, UpBlock, DownBlock, OutConv
from src.metrics import compute_metrics


def seq2motif(weights=None, **model_cfg):
    """
    seq2motif: a deep learning-based autoencoder for RNA sequence connections predictions.
    weights (str): Path to weights file
    **kwargs: Model hyperparameters
    """

    model = Seq2Motif(**model_cfg)
    if weights is not None:
        print(f"Load weights from {weights}")
        model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
    else:
        print("No weights provided, using random initialization")
    mlflow.set_tag("model", "seq2motif-pseudo_probing")
    return model


class Seq2Motif(nn.Module):
    def __init__(
        self,
        train_len: int,
        device: str,
        lr: float,
        verbose: bool,
        embedding_dim: int,
        num_conv: int,
        pool_mode: str,
        up_mode: str,
        skip: int,
        addition: str,
        features: list,
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
        # 4) arquitectura
        self.build_graph(
            embedding_dim=embedding_dim,
            num_conv=num_conv,
            pool_mode=pool_mode,
            up_mode=up_mode,
            skip=skip,
            addition=addition,
            features=features,
        )

        self.verbose = verbose
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        self.log_model()
        self.to(self.device)

    def build_graph(
        self,
        embedding_dim: int,
        num_conv: int,
        pool_mode: str,
        up_mode: str,
        skip: int,
        addition: str,
        features: list,
    ):

        rev_features = features[::-1]
        encoder_blocks = len(features) - 1

        self.L_min = self.input_length // (2**encoder_blocks)
        volume = [(self.input_length / 2**i) * f for i, f in enumerate(features)]
        self.out_dim = 1
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_encoder_blocks": encoder_blocks,
            "arc_initial_volume": 4 * self.input_length,
            "arc_latent_volume": volume[-1],
            "arc_features": features,
            "arc_num_conv": num_conv,
            "arc_pool_mode": pool_mode,
            "arc_up_mode": up_mode,
            "arc_addition": addition,
            "arc_skip": skip,
        }

        self.inc = N_Conv(embedding_dim, features[0], num_conv)
        self.down = nn.ModuleList(
            [
                DownBlock(
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    num_conv=num_conv,
                    pool_mode=pool_mode,
                )
                for i in range(encoder_blocks)
            ]
        )
        self.up = nn.ModuleList(
            [
                UpBlock(
                    in_channels=rev_features[i],
                    out_channels=rev_features[i + 1],
                    num_conv=num_conv,
                    up_mode=up_mode,
                    addition=addition,
                    skip=skip,
                )
                for i in range(len(rev_features) - 1)
            ]
        )
        self.outc = OutConv(features[0], self.out_dim)

    def forward(self, x):
        x = x.to(self.device)
        x = self.inc(x)
        encoder_outputs = [x]
        for _, down in enumerate(self.down):
            x = down(x)
            encoder_outputs.append(x)

        x_latent = x

        skips = encoder_outputs[:-1][::-1]
        for up, skip in zip(self.up, skips):
            x = up(x, skip)

        y_pred = nn.Sigmoid()(self.outc(x))

        return y_pred, x_latent

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
            y_pred, _ = self(x_model)
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

                y_pred, z = self(x_model)
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

    # def pred(self, loader, logits=False):
    #     self.eval()

    #     if self.verbose:
    #         loader = tqdm(loader)

    #     predictions, logits_list = [], []
    #     with tr.no_grad():
    #         for batch in loader:

    #             seqid = batch["id"]
    #             embedding = batch["embedding"]
    #             sequences = batch["sequence"]
    #             lengths = batch["length"]
    #             pseudo_probing = batch["pseudo_probing"],
    #             motif_emb = batch["motif_emb"],
    #             y_pred, z = self(embedding)

    #             for k in range(y_pred.shape[0]):
    #                 seq_len = lengths[k]

    #                 predictions.append(
    #                     (
    #                         seqid[k],
    #                         sequences[k],
    #                         seq_len,
    #                         embedding[k, :, :seq_len].cpu().numpy(),
    #                         pseudo_probing[k, :, :seq_len].cpu().numpy(),
    #                         motif_emb[k, :, :seq_len].cpu().numpy(),
    #                         y_pred[k, :, :seq_len].cpu().numpy(),
    #                         z[k].cpu().numpy(),
    #                     )
    #                 )

    #     predictions = pd.DataFrame(
    #         predictions,
    #         columns=[
    #             "id",
    #             "sequence",
    #             "length",
    #             "embedding",
    #             "pseudo_probing",
    #             "motif_emb",
    #             "connection_pred",
    #             "latent",
    #         ],
    #     )

    #     return predictions, logits_list
