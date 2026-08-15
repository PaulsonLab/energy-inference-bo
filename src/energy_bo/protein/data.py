"""Pinned measured TrpB and CreiLOV data loading.

The source files are the processed, experimentally measured fitness tables from
the public SGPO repository.  They are downloaded on demand and never committed.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
SGPO_COMMIT = "290fa8a4cc99d50980cb8d7cf85ae76744552ead"

DATASETS = {
    "trpb": {
        "relative_path": "data/TrpB/fitness.csv",
        "sequence_column": "Combo",
        "fitness_column": "fitness",
        "rows": 111_883,
        "length": 15,
        "sha256": "f4627857282bef95be5485ff13aef99579c10b5b0e7bf8098ad0cdcbc6a8013a",
    },
    "creilov": {
        "relative_path": "data/CreiLOV/fitness.csv",
        "sequence_column": "Combo",
        "fitness_column": "fitness",
        "rows": 167_530,
        "length": 119,
        "sha256": "e3c09bb02ea6e6da0046ddf3d16b4c370dadeaad79d8e0cb3bcda4e02d5d7f8f",
    },
}


@dataclass(frozen=True)
class ProteinLandscape:
    name: str
    sequences: tuple[str, ...]
    encoded: torch.Tensor
    fitness: torch.Tensor
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        if self.encoded.dtype != torch.long or self.encoded.ndim != 2:
            raise ValueError("encoded sequences must be a two-dimensional long tensor")
        if self.fitness.dtype != torch.double or self.fitness.shape != (len(self.sequences),):
            raise ValueError("fitness must be a one-dimensional double tensor")
        if self.encoded.shape[0] != len(self.sequences):
            raise ValueError("sequence and tensor row counts differ")
        if not torch.isfinite(self.fitness).all():
            raise ValueError("fitness contains nonfinite values")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_url(name: str) -> str:
    spec = DATASETS[name]
    return (
        "https://raw.githubusercontent.com/jsunn-y/SGPO/"
        f"{SGPO_COMMIT}/{spec['relative_path']}"
    )


def fetch_landscape(name: str, cache_dir: Path, timeout: int = 180) -> Path:
    if name not in DATASETS:
        raise KeyError(f"unknown protein landscape: {name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{name}.csv"
    expected = str(DATASETS[name]["sha256"])
    if target.exists() and sha256_file(target) == expected:
        return target
    if target.exists():
        raise RuntimeError(f"checksum mismatch for existing data file {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{name}-", suffix=".csv", dir=cache_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(dataset_url(name), timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
        observed = sha256_file(temporary)
        if observed != expected:
            raise RuntimeError(f"download checksum mismatch for {name}: {observed}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def encode_sequences(sequences: list[str] | tuple[str, ...]) -> torch.Tensor:
    if not sequences:
        raise ValueError("at least one sequence is required")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("all sequences must have equal length")
    try:
        values = [[AA_TO_INDEX[aa] for aa in sequence] for sequence in sequences]
    except KeyError as error:
        raise ValueError(f"unsupported amino-acid symbol: {error.args[0]}") from error
    return torch.tensor(values, dtype=torch.long)


def load_landscape(name: str, cache_dir: Path, *, download: bool = True) -> ProteinLandscape:
    name = name.lower()
    spec = DATASETS[name]
    path = fetch_landscape(name, cache_dir) if download else cache_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if observed_hash != spec["sha256"]:
        raise RuntimeError(f"checksum mismatch for {path}: {observed_hash}")
    records: list[tuple[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row[str(spec["sequence_column"])].strip().upper()
            fitness = float(row[str(spec["fitness_column"])])
            records.append((sequence, fitness))
    records.sort(key=lambda item: item[0])
    sequences = tuple(item[0] for item in records)
    if len(records) != int(spec["rows"]) or len(set(sequences)) != len(records):
        raise RuntimeError(f"unexpected row count or duplicate sequence in {name}")
    if any(len(sequence) != int(spec["length"]) for sequence in sequences):
        raise RuntimeError(f"unexpected sequence length in {name}")
    fitness = torch.tensor([item[1] for item in records], dtype=torch.double)
    return ProteinLandscape(
        name=name,
        sequences=sequences,
        encoded=encode_sequences(sequences),
        fitness=fitness,
        source_path=path,
        source_sha256=observed_hash,
    )


def frozen_permutation(size: int, dataset: str, seed: int) -> np.ndarray:
    """Return a stable dataset-specific permutation independent of Python hashing."""
    token = hashlib.sha256(f"task05a-v1:{dataset}:{seed}".encode()).digest()
    numeric_seed = int.from_bytes(token[:8], "little")
    return np.random.Generator(np.random.PCG64(numeric_seed)).permutation(size)


def smoke_subset(landscape: ProteinLandscape, seed: int, size: int) -> ProteinLandscape:
    if size < 48:
        raise ValueError("smoke pool must contain at least 48 points")
    indices = frozen_permutation(len(landscape.sequences), landscape.name, seed)[:size]
    sequences = tuple(landscape.sequences[int(index)] for index in indices)
    return ProteinLandscape(
        name=landscape.name,
        sequences=sequences,
        encoded=landscape.encoded[torch.as_tensor(indices)].clone(),
        fitness=landscape.fitness[torch.as_tensor(indices)].clone(),
        source_path=landscape.source_path,
        source_sha256=landscape.source_sha256,
    )
