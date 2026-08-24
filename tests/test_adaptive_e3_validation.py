from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/sun_oxide/configs/adaptive_e3_validation.json"
DRIVER = ROOT / "experiments/sun_oxide/adaptive_e3_validation.py"
NOTEBOOK = ROOT / "experiments/sun_oxide/colab_adaptive_e3_validation.ipynb"


def test_fresh_adaptive_config_freezes_requested_protocol() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["bo"]["seeds"] == list(range(12, 32))
    assert config["engineering_smoke"]["seeds"] == [0, 1, 2]
    assert config["bo"]["methods"] == [
        "NO_PBE",
        "FULL_PBE_OPT",
        "ADAPTIVE_PBE",
    ]
    assert config["bo"]["initial_action_count"] == 8
    assert config["bo"]["sequential_query_count"] == 12
    assert config["factor_bank"]["factor_count"] == 124718
    assert config["factor_bank"]["weight_exact"] == "1/499"
    assert config["adaptive"] == {
        "activation_rule": "stable decreasing c_j; smallest top prefix leaving current-pair contribution <= rho*(epsilon_struct-active_EI_gap)",
        "cumulative_active_set_within_seed": True,
        "epsilon_struct": 0.02,
        "full_bank_fallback_after_max_stages": True,
        "max_stages": 8,
        "rho": 0.8,
        "warm_start_across_bo_iterations": True,
        "warm_start_across_stages": True,
    }
    assert config["importance_validation"]["sample_count"] == 4096
    assert [item["name"] for item in config["importance_validation"]["states"]] == [
        "seed_12_initial",
        "seed_12_after_6_queries",
        "seed_12_after_12_queries",
    ]
    assert config["bootstrap"]["resamples"] == 10000
    assert config["environment"]["runtime"] == "standard_colab_cpu"
    assert config["environment"]["uv"] == "0.10.11"
    assert config["environment"]["python"] == "3.12.13"
    assert config["output"]["zip_name"] == "sun_oxide_adaptive_e3_outputs.zip"
    assert config["tuning_after_gw_results"] is False


def test_target_free_smoke_is_oracle_isolated_and_precedes_oracle_loader() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    smoke_start = source.index("def scientific_smoke(")
    oracle_loader_start = source.index("def _load_oracle_after_smoke(")
    smoke_source = source[smoke_start:oracle_loader_start]
    assert "include_oracle=False" in smoke_source
    assert "gw_band_gap_ev" not in smoke_source
    engineering_start = source.index("def run_engineering_smoke(")
    engineering_source = source[engineering_start:source.index("def _paired_bootstrap(")]
    assert engineering_source.index("scientific_smoke(") < engineering_source.index(
        "_load_oracle_after_smoke("
    )
    validation_start = source.index("def run_validation(")
    validation_source = source[validation_start:]
    assert validation_source.index("scientific_smoke(") < validation_source.index(
        "_load_oracle_after_smoke("
    )


def test_shadow_full_is_computed_without_oracle_and_excluded_from_runtime() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    start = source.index("def _run_adaptive_trajectory(")
    stop = source.index("def _old_full_actions(")
    adaptive_source = source[start:stop]
    shadow_start = adaptive_source.index("shadow_started")
    query_start = adaptive_source.index("value = oracle.query(")
    assert shadow_start < query_start
    shadow_section = adaptive_source[shadow_start:query_start]
    assert "oracle.query" not in shadow_section
    assert "shadow_full_seconds_excluded_from_adaptive_runtime" in shadow_section
    assert "shadow_affected_adaptive_policy" in shadow_section


def test_driver_sets_headless_cache_before_matplotlib_import() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert source.index('os.environ["MPLBACKEND"] = "Agg"') < source.index(
        "import matplotlib"
    )
    assert source.index('os.environ.setdefault("MPLCONFIGDIR"') < source.index(
        "import matplotlib"
    )
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_colab_notebook_freezes_cpu_bootstrap_run_and_single_zip() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "standard CPU runtime" in source
    assert "uv=={UV_BOOTSTRAP_VERSION}" in source
    assert "UV_BOOTSTRAP_VERSION = '0.10.11'" in source
    assert "PYTHON_VERSION = '3.12.13'" in source
    assert "pip', 'check'" in source
    assert "adaptive_e3_validation.json" in source
    assert "adaptive_e3_validation.py" in source
    assert "'smoke'" in source and "'run'" in source
    assert "sun_oxide_adaptive_e3_outputs.zip" in source
    assert "files.download(str(ZIP_PATH))" in source
    assert "--require-hashes" in source
    assert notebook["metadata"].get("accelerator", "") == ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
