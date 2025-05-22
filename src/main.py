# scripts/main.py
import os, shutil, json, random
from datetime import datetime
from functools import partial

import numpy as np
import torch as tr
import pandas as pd
import mlflow, mlflow.pytorch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import SeqDataset, pad_batch
from hydra.utils import instantiate
from src.utils import validate_file


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    repo_root = hydra.utils.get_original_cwd()
    os.chdir(repo_root)
    # Cache y reproducibilidad
    if cfg.cache_path and cfg.command == "train":
        shutil.rmtree(cfg.cache_path, ignore_errors=True)
        os.makedirs(cfg.cache_path, exist_ok=True)
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # MLflow
    mlflow.set_tracking_uri(f"sqlite:///{repo_root}/results/mlflow/mlruns.db")
    art_loc = f"{repo_root}/results/mlflow/mlruns/artifacts"

    try:
        mlflow.create_experiment(cfg.exp, artifact_location=art_loc)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(cfg.exp)

    # AGREGAR RUIDO - fix
    with mlflow.start_run(run_name=cfg.run):
        mlflow.log_params(
            {
                "command": cfg.command,
                "exp": cfg.exp,
                "run": cfg.run,
                "model_name": cfg.model_name,
                "max_len": cfg.max_len,
                "batch_size": cfg.batch_size,
                "nworkers": cfg.nworkers,
                "patience": cfg.patience,
                "max_epochs": cfg.max_epochs,
            }
        )

        mlflow.set_tag("model", cfg.model._target_)
        mlflow.set_tag("command", cfg.command)
        if cfg.command == "train":
            run_train(cfg)
        elif cfg.command == "test":
            run_test(cfg)
        elif cfg.command == "pred":
            run_pred(cfg)
        else:
            raise ValueError(f"Unknown command: {cfg.command}")


def run_train(cfg):
    mlflow.log_params(OmegaConf.to_container(cfg.train, resolve=True))
    net, train_loader, valid_loader = _setup_train(cfg)
    _loop_train(net, train_loader, valid_loader, cfg)
    mlflow.pytorch.log_model(net, "model")


def run_test(cfg):
    mlflow.log_params(OmegaConf.to_container(cfg.test, resolve=True))
    _do_test(cfg)


def run_pred(cfg):
    _do_pred(cfg)


def _setup_train(cfg):
    if cfg.train.train_file is None:
        raise ValueError("No train file")
    # 1) Preparar csv de train/valid si es necesario
    if cfg.train.valid_file is None:
        df = pd.read_csv(cfg.train.train_file)
        val = df.sample(frac=cfg.valid_split)

        train_path = os.path.join(cfg.train.out_path, "train_tmp.csv")
        valid_path = os.path.join(cfg.train.out_path, "valid_tmp.csv")

        df.drop(val.index).to_csv(train_path, index=False)
        val.to_csv(valid_path, index=False)
    else:
        train_path, valid_path = cfg.train.train_file, cfg.train.valid_file

    # 2) DataLoaders
    pad_fn = partial(pad_batch, fixed_length=cfg.max_len)
    train_ds = SeqDataset(
        train_path,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        verbose=cfg.verbose,
        cache_path=cfg.cache_path,
        training=True,
    )
    valid_ds = SeqDataset(
        valid_path,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        verbose=cfg.verbose,
        cache_path=cfg.cache_path,
        training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.nworkers,
        collate_fn=pad_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.nworkers,
        collate_fn=pad_fn,
    )
    # 3) Modelo

    # en run_train:
    net = instantiate(cfg.model, train_len=len(train_loader), verbose=cfg.verbose)

    return net, train_loader, valid_loader


def _loop_train(net, train_loader, valid_loader, cfg):
    if cfg.train.out_path is None:
        out_path = (
            f"results/{cfg.exp}/{cfg.run}/{str(datetime.now().replace(':', '-'))}"
        )
    else:
        out_path = cfg.train.out_path
    os.makedirs(out_path, exist_ok=True)

    set_out_path(net, out_path, cfg.train.continue_training, cfg.verbose)

    best_loss, patience_cnt = np.inf, 0
    if cfg.verbose:
        print("Start training...")
        print(f"Output path: {out_path}")
    logfile = os.path.join(out_path, "train_log.csv")

    for epoch in range(cfg.max_epochs):
        # 1 --- entrenamiento ---
        train_metrics = net.fit(train_loader)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v, step=epoch)

        # 2 --- validación ---
        val_metrics = net.test(valid_loader)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"valid_{k}", v, step=epoch)

        # 3 --- guardo mejor modelo ---
        loss = val_metrics["loss"]
        if loss < best_loss:
            best_loss = loss
            tr.save(net.state_dict(), os.path.join(out_path, "weights.pmt"))
            mlflow.log_metric("best_epoch", epoch)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt > cfg.patience:
                if cfg.verbose:
                    print(f"Parando por paciencia > {cfg.patience}")
                break

        # -4 -- log local ---
        header = (
            ["epoch"]
            + [f"train_{k}" for k in sorted(train_metrics.keys())]
            + [f"valid_{k}" for k in sorted(val_metrics.keys())]
        )
        row = (
            [str(epoch)]
            + [f"{train_metrics[k]:.4f}" for k in sorted(train_metrics.keys())]
            + [f"{val_metrics[k]:.4f}" for k in sorted(val_metrics.keys())]
        )
        mode = "w" if epoch == 0 else "a"
        if cfg.verbose:
            print(",".join(header) + "\n")
            print(",".join(row) + "\n")
        with open(logfile, mode) as f:
            if mode == "w":
                f.write(",".join(header) + "\n")
            f.write(",".join(row) + "\n")

    # limpieza de caché temporal
    shutil.rmtree(cfg.cache_path, ignore_errors=True)

    tmp_file = os.path.join(out_path, "train_tmp.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    tmp_file = os.path.join(out_path, "valid_tmp.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def _do_test(cfg):

    # Lectura de test file
    if cfg.test.test_file is None:
        raise ValueError("No test file")
    test_file = validate_file(cfg.test.test_file)

    # 2 DataLoader
    pad_batch_with_fixed_length = partial(pad_batch, fixed_length=cfg.max_len)
    test_loader = DataLoader(
        SeqDataset(test_file, **cfg),
        batch_size=cfg.batch_size if cfg.batch_size else 4,
        shuffle=False,
        num_workers=cfg.nworkers,
        collate_fn=pad_batch_with_fixed_length,
    )

    # 3 Carga de modelo
    net = instantiate(cfg.model, train_len=0, verbose=cfg.verbose)
    net.load_state_dict(tr.load(os.path.join(cfg.test.model_weights)))

    if cfg.verbose:
        print(f"Start testing {test_file} with {cfg.model._target_}")
        print(f"Saving to {cfg.test.out_path}")

    # 4 Test
    test_metrics = net.test(test_loader)

    for k, v in test_metrics.items():
        mlflow.log_metric(key=f"test_{k}", value=v)

    # 5 Log
    summary = (
        ",".join([f"test_{k}" for k in sorted(test_metrics.keys())])
        + "\n"
        + ",".join([f"{test_metrics[k]:.3f}" for k in sorted(test_metrics.keys())])
        + "\n"
    )
    if cfg.test.out_path is not None:
        with open(cfg.test.out_path, "w") as f:
            f.write(summary)
    if cfg.verbose:
        print(summary)


def _do_pred(cfg):
    pass


def set_out_path(net, out_path: str, continue_training: bool, verbose: bool):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    elif os.path.exists(os.path.join(out_path, "weights.pmt")):
        # Si el modelo ya existe y quiero continuar, lo cargo
        if continue_training:
            if verbose:
                print(f"Continuing training from {out_path}")
            net.load_state_dict(tr.load(os.path.join(out_path, "weights.pmt")))
            # Si el modelo ya existe y no quiero continuar, lanzo error
        else:
            raise ValueError(
                f"Output path {out_path} already exists with weights and not continue training is set"
            )
    else:
        # Si los pesos del modelo no existen pero si la carpeta, lanzo error
        pass
    mlflow.log_param("out_path", out_path)


if __name__ == "__main__":
    main()
