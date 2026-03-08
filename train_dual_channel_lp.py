#!/usr/bin/env python3
"""Dual-channel (low-pass / high-pass) link prediction with adaptive MoE decoder.

Target datasets:
- Homophilic: cora, citeseer, pubmed
- Heterophilic: texas, cornell, wisconsin, chameleon, squirrel

Results are persisted to local logs (JSONL + CSV summary).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import add_self_loops, remove_self_loops


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(name: str, root: Path) -> Data:
    name = name.lower()
    transform = NormalizeFeatures()
    if name in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=str(root / "planetoid"), name=name.capitalize(), transform=transform)
    elif name in {"texas", "cornell", "wisconsin"}:
        dataset = WebKB(root=str(root / "webkb"), name=name.capitalize(), transform=transform)
    elif name in {"chameleon", "squirrel"}:
        dataset = WikipediaNetwork(root=str(root / "wikipedia"), name=name, geom_gcn_preprocess=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index
    return data


class ChannelProp(nn.Module):
    """One propagation step using low-pass or high-pass operator."""

    def __init__(self, high_pass: bool = False):
        super().__init__()
        self.high_pass = high_pass

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.zeros_like(x)
        msg = x[col] * norm.unsqueeze(-1)
        out.index_add_(0, row, msg)

        if self.high_pass:
            return x - out
        return out


class DualChannelEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        self.lin_low_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_low_2 = nn.Linear(hidden_dim, out_dim)
        self.low_prop = ChannelProp(high_pass=False)

        self.lin_high_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_high_2 = nn.Linear(hidden_dim, out_dim)
        self.high_prop = ChannelProp(high_pass=True)

    def _tower(self, x: Tensor, edge_index: Tensor, tower: str) -> Tensor:
        if tower == "low":
            x = self.low_prop(x, edge_index, x.size(0))
            x = self.lin_low_1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.low_prop(x, edge_index, x.size(0))
            x = self.lin_low_2(x)
            return x

        x = self.high_prop(x, edge_index, x.size(0))
        x = self.lin_high_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.high_prop(x, edge_index, x.size(0))
        x = self.lin_high_2(x)
        return x

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        z_low = self._tower(x, edge_index, "low")
        z_high = self._tower(x, edge_index, "high")
        return z_low, z_high


class DotDecoder(nn.Module):
    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


class MLPDecoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        zi, zj = z[src], z[dst]
        feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        return self.mlp(feats).view(-1)


class AdaptiveMoEDecoder(nn.Module):
    """Mixture of experts over low/high/sum channels for each node pair."""

    def __init__(self, dim: int, gate_hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 8, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 3),
        )

    def _pair_feats(self, zi: Tensor, zj: Tensor) -> Tensor:
        return torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)

    def forward(self, z_low: Tensor, z_high: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        li, lj = z_low[src], z_low[dst]
        hi, hj = z_high[src], z_high[dst]

        e_low = (li * lj).sum(dim=-1)
        e_high = (hi * hj).sum(dim=-1)
        mix_i = 0.5 * (li + hi)
        mix_j = 0.5 * (lj + hj)
        e_mix = (mix_i * mix_j).sum(dim=-1)

        gate_input = torch.cat([self._pair_feats(li, lj), self._pair_feats(hi, hj)], dim=-1)
        w = torch.softmax(self.gate(gate_input), dim=-1)
        experts = torch.stack([e_low, e_high, e_mix], dim=-1)
        return (w * experts).sum(dim=-1)


@dataclass
class Metrics:
    auc: float
    ap: float


def evaluate_auc_ap(logits: Tensor, labels: Tensor) -> Metrics:
    from sklearn.metrics import average_precision_score, roc_auc_score

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    ys = labels.detach().cpu().numpy()
    auc = float(roc_auc_score(ys, probs))
    ap = float(average_precision_score(ys, probs))
    return Metrics(auc=auc, ap=ap)


def run_single(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    seed_everything(seed)
    raw = load_dataset(dataset_name, args.data_root)
    split = RandomLinkSplit(
        num_val=args.val_ratio,
        num_test=args.test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True,
    )
    train_data, val_data, test_data = split(raw)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    encoder = DualChannelEncoder(
        in_dim=train_data.x.size(-1),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(device)

    decoder_name = args.decoder.lower()
    if decoder_name == "dot":
        decoder = DotDecoder().to(device)
    elif decoder_name == "mlp":
        decoder = MLPDecoder(args.out_dim).to(device)
    elif decoder_name == "moe":
        decoder = AdaptiveMoEDecoder(args.out_dim).to(device)
    else:
        raise ValueError(f"Unknown decoder: {args.decoder}")

    params: List[nn.Parameter] = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    def decode(data_obj: Data) -> Tuple[Tensor, Tensor]:
        z_low, z_high = encoder(data_obj.x, data_obj.edge_index)
        edge_label_index = data_obj.edge_label_index
        labels = data_obj.edge_label.float()

        if decoder_name == "dot":
            logits = decoder(0.5 * (z_low + z_high), edge_label_index)
        elif decoder_name == "mlp":
            logits = decoder(0.5 * (z_low + z_high), edge_label_index)
        else:
            logits = decoder(z_low, z_high, edge_label_index)
        return logits, labels

    best_val_auc = -math.inf
    best_state = None
    patience = 0

    for _ in range(args.epochs):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        logits, labels = decode(train_data)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_logits, val_labels = decode(val_data)
        val_metrics = evaluate_auc_ap(val_logits, val_labels)

        if val_metrics.auc > best_val_auc:
            best_val_auc = val_metrics.auc
            best_state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        val_logits, val_labels = decode(val_data)
        test_logits, test_labels = decode(test_data)

    val_metrics = evaluate_auc_ap(val_logits, val_labels)
    test_metrics = evaluate_auc_ap(test_logits, test_labels)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "decoder": decoder_name,
        "val_auc": val_metrics.auc,
        "val_ap": val_metrics.ap,
        "test_auc": test_metrics.auc,
        "test_ap": test_metrics.ap,
    }


def aggregate(records: Sequence[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [float(x[key]) for x in records]
    return float(np.mean(vals)), float(np.std(vals))


def write_logs(records: Sequence[Dict[str, float]], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = out_dir / f"dual_channel_lp_{ts}.jsonl"
    csv_path = out_dir / f"dual_channel_lp_{ts}_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in records:
        grouped.setdefault((str(row["dataset"]), str(row["decoder"])), []).append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "decoder", "runs", "test_auc_mean", "test_auc_std", "test_ap_mean", "test_ap_std"])
        for (dataset, decoder), rows in sorted(grouped.items()):
            auc_mean, auc_std = aggregate(rows, "test_auc")
            ap_mean, ap_std = aggregate(rows, "test_ap")
            writer.writerow([dataset, decoder, len(rows), f"{auc_mean:.4f}", f"{auc_std:.4f}", f"{ap_mean:.4f}", f"{ap_std:.4f}"])

    return jsonl_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-channel link prediction with adaptive MoE decoder")
    parser.add_argument("--datasets", nargs="+", default=[
        "texas", "cornell", "wisconsin", "chameleon", "squirrel", "cora", "citeseer", "pubmed"
    ])
    parser.add_argument("--decoder", type=str, default="moe", choices=["moe", "dot", "mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    all_records: List[Dict[str, float]] = []
    for ds in args.datasets:
        for seed in args.seeds:
            record = run_single(ds, seed, args, device)
            all_records.append(record)
            print(
                f"[{record['dataset']}] seed={record['seed']} decoder={record['decoder']} "
                f"test_auc={record['test_auc']:.4f} test_ap={record['test_ap']:.4f}"
            )

    jsonl_path, csv_path = write_logs(all_records, args.log_dir)
    print(f"Saved raw run logs to: {jsonl_path}")
    print(f"Saved summary logs to: {csv_path}")


if __name__ == "__main__":
    main()
