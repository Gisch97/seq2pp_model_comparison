#!/usr/bin/env python3
from itertools import combinations_with_replacement
import subprocess
import os

# Sólo dos valores de num_conv
base_feats = [8, 16, 32, 64]

feature_sets = [
    list(combo)
    for r in range(2, 5)
    for combo in combinations_with_replacement(base_feats, r)
]

# Directorio de resultados
exp = "SimpleCNN_no_pooling"
results_dir = f"results/{exp}"
os.makedirs(results_dir, exist_ok=True)

for feats in feature_sets:
    k = [3 for _ in range(len(feats))]

    # nombre de la corrida
    feat_str = ",".join(str(f) for f in feats)
    kernel_str = ",".join(str(k) for k in k)
    run = f"scnn_k_3_f{'_'.join(str(f) for f in feats)}"
    if os.path.exists(f"{results_dir}/{run}"):
        print(f"→ SKIP: {run} already exists")
        continue

    # Training
    train_cmd = [
        "python3",
        "-m",
        "src.main",
        "model=simplecnn",
        f"model.features=[{feat_str}]",
        f"model.kernels=[{kernel_str}]",
        f"exp={exp}",
        f"run={run}",
        "command=train",
        f"train.out_path={results_dir}/{run}",
    ]
    print("→ TRAIN:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    # Testing
    test_cmd = [
        "python3",
        "-m",
        "src.main",
        f"exp={exp}",
        f"run={run}",
        "model=simplecnn",
        f"model.features=[{feat_str}]",
        f"model.kernels=[{kernel_str}]",
        "command=test",
    ]
    print("→ TEST: ", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
