import os
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset import DataManager


training_sets = (
    "musique",
    "2wikimultihopqa",
    "hotpotqa",
    "multihoprag",
    "nq",
)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _to_numpy(embedding) -> np.ndarray:
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32)
    if torch.is_tensor(embedding):
        return embedding.detach().cpu().numpy().astype(np.float32)
    return np.asarray(embedding, dtype=np.float32)


class HopDiscriminatorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class HopDiscriminator:
    def __init__(
        self,
        conf: Dict,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.conf = conf
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_ratio = val_ratio
        self.seed = seed
        self.device = torch.device("cpu")

        self.target_dataset = conf["dataset"].lower()
        self.source_datasets = self.get_training_datasets(self.target_dataset)
        self.checkpoint_path = self.get_checkpoint_path(conf, self.target_dataset)

        self.input_dim = None
        self.model = None
        self.metadata = {}

    @classmethod
    def get_training_datasets(cls, target_dataset: str) -> Tuple[str, ...]:
        return tuple(
            dataset
            for dataset in training_sets
            if dataset != target_dataset.lower()
        )

    @classmethod
    def get_checkpoint_path(cls, conf: Dict, target_dataset: str | None = None) -> str:
        if conf["save_dir"] is None:
            raise ValueError('Using the query hop discriminator requires a non-empty save_dir.')

        target_dataset = (target_dataset or conf["dataset"]).lower()
        source_datasets = cls.get_training_datasets(target_dataset)
        checkpoint_dir = os.path.join(conf["save_dir"], "hop_discriminator")
        file_name = (
            f'target_{_sanitize_name(target_dataset)}'
            f'__sources_{"_".join(_sanitize_name(dataset) for dataset in source_datasets)}'
            f'__embed_{_sanitize_name(conf["embed_name"])}.pt'
        )
        return os.path.join(checkpoint_dir, file_name)

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    @staticmethod
    def _extract_hop_count(dataset_name: str, sample: Dict) -> int:
        if dataset_name == "musique":
            return len(sample.get("question_decomposition", []))
        if dataset_name == "2wikimultihopqa":
            return len(sample.get("evidences_id", []))
        if dataset_name == "hotpotqa":
            return len(sample.get("supporting_facts", []))
        if dataset_name == "multihoprag":
            return len(sample.get("evidence_list", []))
        if dataset_name == "nq":
            return 1
        raise NotImplementedError(
            f"Unsupported query hop discriminator dataset: {dataset_name}"
        )

    @staticmethod
    def _to_hop_label(hop_count: int) -> int:
        if hop_count <= 0:
            return 0
        return min(hop_count, 4)

    def _build_training_samples(self) -> Tuple[List[str], np.ndarray]:
        queries = []
        labels = []

        for dataset_name in self.source_datasets:
            data = DataManager(
                dataset_name=dataset_name,
                data_dir=self.conf["hop_discriminator_training_data_dir"],
                test_samples=-1,
            )
            if len(data.all_queries) != len(data.data):
                raise ValueError(
                    f"Query count ({len(data.all_queries)}) does not match sample count ({len(data.data)}) "
                    f'for dataset "{dataset_name}".'
                )

            for query, sample in zip(data.all_queries, data.data):
                hop_count = self._extract_hop_count(dataset_name, sample)
                if hop_count <= 0:
                    continue
                labels.append(self._to_hop_label(hop_count) - 1)
                queries.append(query)

        if not queries:
            raise ValueError("Query hop discriminator training set is empty.")

        return queries, np.asarray(labels, dtype=np.int64)

    def _embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        if hasattr(self.conf["embed_model"], "load_model"):
            self.conf["embed_model"].load_model()

        embeddings = []
        bar = tqdm(queries, desc="embedding query hop discriminator queries")
        for query in bar:
            embeddings.append(_to_numpy(self.conf["embed_model"].embed(query)).squeeze())
        bar.close()

        features = np.asarray(embeddings, dtype=np.float32)
        return self._normalize_embeddings(features)

    def _split_train_val(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        train_indices = []
        val_indices = []

        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)

            if len(label_indices) == 1:
                train_indices.extend(label_indices.tolist())
                continue

            val_count = max(1, int(round(len(label_indices) * self.val_ratio)))
            val_count = min(val_count, len(label_indices) - 1)
            val_indices.extend(label_indices[:val_count].tolist())
            train_indices.extend(label_indices[val_count:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        if not val_indices:
            val_indices = train_indices[:]

        return (
            features[train_indices],
            features[val_indices],
            labels[train_indices],
            labels[val_indices],
        )

    def _build_dataloader(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long(),
        )
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=shuffle,
        )

    def fit(self):
        queries, labels = self._build_training_samples()
        features = self._embed_queries(queries)
        train_x, val_x, train_y, val_y = self._split_train_val(features, labels)

        self.input_dim = int(features.shape[1])
        self.model = HopDiscriminatorMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        train_loader = self._build_dataloader(train_x, train_y, shuffle=True)
        val_loader = self._build_dataloader(val_x, val_y, shuffle=False)

        class_counts = np.bincount(train_y, minlength=4)
        class_weights = class_counts.sum() / np.clip(4 * class_counts, a_min=1, a_max=None)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        best_state_dict = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss_total = 0.0
            val_sample_count = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    logits = self.model(batch_features)
                    loss = criterion(logits, batch_labels)
                    val_loss_total += loss.item() * batch_features.size(0)
                    val_sample_count += batch_features.size(0)
            val_loss = val_loss_total / max(val_sample_count, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state_dict is None:
            raise RuntimeError(
                "Query hop discriminator training did not produce a checkpoint."
            )

        self.model.load_state_dict(best_state_dict)
        self.model.eval()
        self.metadata = {
            "target_dataset": self.target_dataset,
            "source_datasets": list(self.source_datasets),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_size": int(len(train_y)),
            "val_size": int(len(val_y)),
            "train_label_distribution": dict(
                sorted(Counter((train_y + 1).tolist()).items())
            ),
            "val_label_distribution": dict(
                sorted(Counter((val_y + 1).tolist()).items())
            ),
        }
        return self

    def save(self, path: str | None = None) -> str:
        if self.model is None:
            raise ValueError("There is no trained query hop discriminator to save.")

        path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "target_dataset": self.target_dataset,
                "source_datasets": list(self.source_datasets),
                "embed_name": self.conf["embed_name"],
                "metadata": self.metadata,
            },
            path,
        )
        return path

    def load(self, path: str | None = None):
        path = path or self.checkpoint_path
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint.get("target_dataset") != self.target_dataset:
            raise ValueError(
                f'Query discriminator target dataset mismatch: expected "{self.target_dataset}", '
                f'got "{checkpoint.get("target_dataset")}".'
            )
        if checkpoint.get("embed_name") != self.conf["embed_name"]:
            raise ValueError(
                f'Query discriminator embed model mismatch: expected "{self.conf["embed_name"]}", '
                f'got "{checkpoint.get("embed_name")}".'
            )

        self.source_datasets = tuple(
            checkpoint.get("source_datasets", self.source_datasets)
        )
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
        self.dropout = checkpoint.get("dropout", self.dropout)
        self.metadata = checkpoint.get("metadata", {})

        self.model = HopDiscriminatorMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self

    def prepare(self):
        if os.path.exists(self.checkpoint_path):
            tqdm.write(f'Loading query hop discriminator from "{self.checkpoint_path}"...')
            return self.load()

        tqdm.write(
            "Training query hop discriminator on datasets: "
            + ", ".join(self.source_datasets)
        )
        self.fit()
        saved_path = self.save()
        tqdm.write(f'Query hop discriminator saved to "{saved_path}".')
        return self

    def predict_from_embedding(self, embedding) -> int:
        if self.model is None:
            raise ValueError("Query hop discriminator has not been prepared.")

        feature = _to_numpy(embedding).reshape(1, -1)
        feature = self._normalize_embeddings(feature)
        feature_tensor = torch.from_numpy(feature).float().to(self.device)

        with torch.no_grad():
            predicted_label = int(self.model(feature_tensor).argmax(dim=1).item())
        return predicted_label + 1
