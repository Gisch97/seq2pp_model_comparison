#!/usr/bin/env python3
from itertools import combinations_with_replacement
import subprocess
import os

# Sólo dos valores de num_conv
ncs = [1, 2]
base_feats = [4, 8, 16, 32]

feature_sets = [
    list(combo)
    for r in range(1, 4)
    for combo in combinations_with_replacement(base_feats, r)
]


# Directorio de resultados
exp = "seq2pp"
results_dir = f"results/{exp}"
os.makedirs(results_dir, exist_ok=True)

for nc in ncs:
    for feats in feature_sets:
        for skip in ["0", "1"]:
            # nombre de la corrida
            feat_str = ",".join(str(f) for f in feats)
            run = f"s2pp_nc{nc}_skip_{skip}_f{'_'.join(str(f) for f in feats)}"

            # Training
            train_cmd = [
                "python3",
                "-m",
                "src.main",
                "model=seq2motif",
                f"model.num_conv={nc}",
                f"model.features=[{feat_str}]",
                f"model.skip={skip}",
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
                "model=seq2motif",
                f"model.num_conv={nc}",
                f"model.features=[{feat_str}]",
                f"model.skip={skip}",
                "command=test",
            ]
            print("→ TEST: ", " ".join(test_cmd))
            subprocess.run(test_cmd, check=True)
