import pandas as pd
from torch.utils.data import Dataset
import torch as tr
import ast
import os
import json
import pickle
import random
from src.embeddings import OneHotEmbedding
from typing import Union


class SeqDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        min_len=0,
        max_len=512,
        verbose=False,
        cache_path=None,
        for_prediction=False,
        training=False,
        **kargs,
    ):
        """
        interaction_prior: none, probmat
        """
        self.max_len = max_len
        self.verbose = verbose
        self.training = training

        # Loading dataset
        data = pd.read_csv(dataset_path)
        assert (
            "sequence" in data.columns and "id" in data.columns
        ), "Dataset should contain 'id' and 'sequence' columns"

        data["len"] = data.sequence.str.len()
        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len
        datalen = len(data)

        data = data[(data.len >= min_len) & (data.len <= max_len)]
        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.pseudo_probing = data.pseudo_probe
        # self.stem = data.stem
        # self.motifs = data.motifs.tolist()
        self.embedding = OneHotEmbedding()
        self.embedding_size = self.embedding.emb_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        sequence = self.sequences[idx]
        # motif = self.motifs[idx]
        pseudo_probing = self.pseudo_probing[idx]
        pseudo_probing = tr.Tensor(ast.literal_eval(pseudo_probing)).unsqueeze(dim=0)
        # stem = self.stem[idx]
        # stem = tr.Tensor(ast.literal_eval(stem)).unsqueeze(dim=0)
        L = len(sequence)
        seq_emb = self.embedding.seq2emb(sequence)
        # motif_emb = self.embedding.motif2emb(motif)

        item = {
            "id": seqid,
            "length": L,
            "sequence": sequence,
            "embedding": seq_emb,
            "pseudo_probing": pseudo_probing,
            # "stem" : stem,
            # "motif_emb": motif_emb,
        }
        return item


def pad_batch(batch, fixed_length=0):
    """batch is a dictionary with different variables lists"""
    L = [b["length"] for b in batch]
    if fixed_length == 0:
        fixed_length = max(L)
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], fixed_length))
    pseudo_probing_pad = tr.zeros(
        (len(batch), batch[0]["pseudo_probing"].shape[0], fixed_length), dtype=tr.float
    )
    # stem_pad = tr.zeros((len(batch), batch[0]["stem"].shape[0], fixed_length), dtype=tr.float)
    # motif_emb_pad = tr.zeros((len(batch), batch[0]["motif_emb"].shape[0], fixed_length), dtype=tr.float)
    mask = tr.zeros((len(batch), fixed_length), dtype=tr.bool)

    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]
        pseudo_probing_pad[k, :, : L[k]] = batch[k]["pseudo_probing"]
        # stem_pad[k, :, : L[k]] = batch[k]["stem"]
        # motif_emb_pad[k, :, : L[k]] = batch[k]["motif_emb"]
        mask[k, : L[k]] = 1

    out_batch = {
        "id": [b["id"] for b in batch],
        "length": L,
        "sequence": [b["sequence"] for b in batch],
        "embedding": embedding_pad,
        "pseudo_probing": pseudo_probing_pad,
        # "stem":stem_pad,
        # "motif_emb": motif_emb_pad,
        "mask": mask,
    }

    return out_batch
