# Task 01 Colab run

The checked-in smoke result uses only three seeds locally. Run the following in a new
Google Colab notebook for the planned 30-seed CPU/GPU-optional validation; it is still
only a scalar nine-parameter residual fit, but avoids consuming the local laptop.

```python
!git clone https://github.com/PaulsonLab/energy-inference-bo.git
%cd energy-inference-bo
!pip install -r requirements.txt
!pip install -e .
!pytest -q
!python -m energy_bo.experiments.run_task01 \
    --seeds $(seq 0 29) \
    --output-dir artifacts/task01_colab \
    --report-path reports/TASK_01_COLAB.md
```

The run is CPU-compatible and should remain well below 1 GB RAM. To download outputs:

```python
!zip -r task01_colab_outputs.zip artifacts/task01_colab reports/TASK_01_COLAB.md
from google.colab import files
files.download("task01_colab_outputs.zip")
```

Do not run q>1, large-dimensional BO, particle sweeps beyond the configuration above,
or Task 02 from this notebook.
