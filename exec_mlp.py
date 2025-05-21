#!/usr/bin/env python3
from itertools import combinations_with_replacement
import subprocess
import os

base_feats = [16, 32, 64]

feature_sets = [
    list(combo)
    for r in range(1, 4)
    for combo in combinations_with_replacement(base_feats, r)
]

# Directorio de resultados
exp = "SimpleMLP"
results_dir = f"results/{exp}"
os.makedirs(results_dir, exist_ok=True)

for feats in feature_sets:
    # formatear lista de features para Hydra: "[4,8,16]"
    feat_str = ",".join(str(f) for f in feats)
    # nombre de la corrida
    run = f"SMLP_f{'_'.join(str(f) for f in feats)}"

    # Training
    train_cmd = [
        "python3",
        "-m",
        "src.main",
        "model=simplemlp",
        f"model.features=[{feat_str}]",
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
        "model=simplemlp",
        f"model.features=[{feat_str}]",
        "command=test",
    ]
    print("→ TEST: ", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
