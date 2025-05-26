#!/usr/bin/env python3
import subprocess
import os
from itertools import combinations_with_replacement


def run_experiment(params, exp, run, data, command):
    """
    Lanza un experimento Hydra:
      params: dict de parámetros (clave: valor) que Hydra entiende (p.ej. 'model.features': '[4,8,16]')
      exp: nombre de la experimentación (exp)
      run: nombre de la corrida (run)
      data: nombre del dataset (srp, tRNA, etc.)
      command: 'command=train' o 'command=test'
    """
    cmd = ["python3", "-m", "src.main"]
    for key, val in params.items():
        cmd.append(f"{key}={val}")
    cmd += [f"exp={exp}", f"run={run}", command, f"{command.split('=')[1]}={data}"]
    print(f"→ {command.split('=')[1].upper():5s}:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --- configuración general ---
base_feats = [4, 8, 16, 32, 64]
feature_sets = [
    combo for r in range(2, 4) for combo in combinations_with_replacement(base_feats, r)
]
datasets = ["srp", "tRNA"]


# === Experimentos SimpleCNN ===
for data in datasets:
    exp_cnn = f"SimpleCNN_{data}"
    results_cnn = os.path.join("results", data, exp_cnn)
    os.makedirs(results_cnn, exist_ok=True)

    for feats in feature_sets:
        feat_list = list(feats)
        kern_list = [3] * len(feat_list)
        feat_str = ",".join(map(str, feat_list))
        kernel_str = ",".join(map(str, kern_list))
        run_name = f"scnn_k3_f{'_'.join(map(str, feat_list))}"
        run_path = os.path.join(results_cnn, run_name)
        if os.path.exists(run_path):
            print(f"→ SKIP: {run_name} ya existe")
            continue

        params = {
            "model": "simplecnn",
            "model.features": f"[{feat_str}]",
            "model.kernels": f"[{kernel_str}]",
        }
        run_experiment(params, exp_cnn, run_name, data, "command=train")
        run_experiment(params, exp_cnn, run_name, data, "command=test")


# === Experimentos Seq2Motif ===
for data in datasets:
    exp_s2m = f"Seq2Motif_{data}"
    results_s2m = os.path.join("results", data, exp_s2m)
    os.makedirs(results_s2m, exist_ok=True)

    for feats in feature_sets:
        feat_list = list(feats)
        feat_str = ",".join(map(str, feat_list))
        for skip in [0, 1]:
            run_name = f"s2m_skip{skip}_f{'_'.join(map(str, feat_list))}"
            run_path = os.path.join(results_s2m, run_name)
            if os.path.exists(run_path):
                print(f"→ SKIP: {run_name} ya existe")
                continue

            params = {
                "model": "seq2motif",
                "model.skip": skip,
                "model.features": f"[{feat_str}]",
            }
            run_experiment(params, exp_s2m, run_name, data, "command=train")
            run_experiment(params, exp_s2m, run_name, data, "command=test")


# === Experimentos SimpleMLP ===
for data in datasets:
    exp_mlp = f"SimpleMLP_{data}"
    results_mlp = os.path.join("results", data, exp_mlp)
    os.makedirs(results_mlp, exist_ok=True)

    for feats in feature_sets:
        feat_list = list(feats)
        feat_str = ",".join(map(str, feat_list))
        run_name = f"mlp_f{'_'.join(map(str, feat_list))}"
        run_path = os.path.join(results_mlp, run_name)
        if os.path.exists(run_path):
            print(f"→ SKIP: {run_name} ya existe")
            continue

        params = {"model": "simplemlp", "model.features": f"[{feat_str}]"}
        run_experiment(params, exp_mlp, run_name, data, "command=train")
        run_experiment(params, exp_mlp, run_name, data, "command=test")


# === Experimentos SeqEncoder ===
for data in datasets:
    exp_enc = f"SeqEncoder_{data}"
    results_enc = os.path.join("results", data, exp_enc)
    os.makedirs(results_enc, exist_ok=True)

    # mantengo num_conv 2 y pool = avg
    for feats in feature_sets:
        feat_list = list(feats)
        feat_str = ",".join(map(str, feat_list))
        run_name = f"enc_f{'_'.join(map(str, feat_list))}"
        run_path = os.path.join(results_enc, run_name)
        if os.path.exists(run_path):
            print(f"→ SKIP: {run_name} ya existe")
            continue

        params = {
            "model": "seqencoder",
            "model.features": f"[{feat_str}]",
        }
        run_experiment(params, exp_enc, run_name, data, "command=train")
        run_experiment(params, exp_enc, run_name, data, "command=test")
