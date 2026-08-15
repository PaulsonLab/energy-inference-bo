"""Download and checksum the two pinned Task 05A measured datasets."""
from pathlib import Path
from energy_bo.protein.data import DATASETS, load_landscape

for name in DATASETS:
    data = load_landscape(name, Path("data/task05a"), download=True)
    print(name, len(data.sequences), data.source_sha256)
