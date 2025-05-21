# imports
import os
import subprocess as sp
from platform import system
import warnings
import numpy as np
import torch as tr
import pandas as pd
import json
from src.embeddings import NT_DICT, VOCABULARY


# All possible matching brackets for base pairing
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]
# Normalization.
BRACKET_DICT = {"!": "A", "?": "a", "C": "B", "D": "b"}



def valid_sequence(seq):
    """Check if sequence is valid"""
    return set(seq.upper()) <= (set(NT_DICT.keys()).union(set(VOCABULARY)))


def validate_file(pred_file):
    """Validate input file fasta/csv format and return csv file"""
    if os.path.splitext(pred_file)[1] == ".fasta":
        table = []
        with open(pred_file) as f:
            row = []  # id, seq, (optionally) struct
            for line in f:
                if line.startswith(">"):
                    if row:
                        table.append(row)
                        row = []
                    row.append(line[1:].strip())
                else:
                    if len(row) == 1:  # then is seq
                        row.append(line.strip())
                        if not valid_sequence(row[-1]):
                            raise ValueError(
                                f"Sequence {row.upper()} contains invalid characters"
                            )
                    else:  # struct
                        row.append(
                            line.strip()[: len(row[1])]
                        )  # some fasta formats have extra information in the structure line
        if row:
            table.append(row)

        pred_file = pred_file.replace(".fasta", ".csv")

        if len(table[-1]) == 2:
            columns = ["id", "sequence"]
        else:
            columns = ["id", "sequence", "dotbracket"]

        pd.DataFrame(table, columns=columns).to_csv(pred_file, index=False)

    elif os.path.splitext(pred_file)[1] != ".csv":
        raise ValueError(
            "Predicting from a file with format different from .csv or .fasta is not supported"
        )

    return pred_file