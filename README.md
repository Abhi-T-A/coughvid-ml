# COUGHVID ML: Machine Learning for Acoustic Cough Analysis

An engineering-grade, modular, and reproducible machine learning pipeline for acoustic cough analysis based on the crowdsourced [COUGHVID dataset (Zenodo v3.0)](https://zenodo.org/records/7024894).

> **IMPORTANT DISCLAIMER**  
> This software is a research and engineering demonstration pipeline. It is **NOT** a clinically validated diagnostic, medical screening, or healthcare system. It must not be used for medical decision-making or patient diagnosis.

---

## 1. Project Architecture

```
coughvid-ml/
├── configs/                  # Centralized YAML configurations
│   ├── base_config.yaml      # Audio, DSP, feature, and split parameters
│   └── logging_config.yaml   # Structured logging configuration
├── src/coughvid/             # Core Python package
│   ├── audio/                # Audio decoding, DSP, and energy-envelope segmentation
│   ├── data/                 # Ingestion, schema validation, and leak-free splitters
│   ├── features/             # Acoustic feature extraction and Parquet caching
│   ├── models/               # Scikit-learn and deep learning model pipelines
│   ├── evaluation/           # Clinical metrics, threshold tuning, and reports
│   └── config.py             # Type-safe configuration loader (Pydantic)
├── tests/                    # Automated test suite
├── notebooks/                # Exploratory and experimental notebooks
├── artifacts/                # Model checkpoints and legacy benchmarks
├── data/                     # Working splits and extracted features (gitignored)
├── reports/                  # Generated figures and evaluation metrics
└── docs/                     # Dataset guides and technical references
```

---

## 2. Quickstart & Configuration

### A. Environment Setup
Install the core dependencies:
```bash
pip install -r requirements.txt
# Or install in editable mode:
pip install -e .
```

### B. Configuring the Raw Dataset Path
The raw dataset directory is **never duplicated** and is configured externally.
You can specify the location in two ways:

1. **Environment Variable (Recommended):**
   ```bash
   # Windows PowerShell
   $env:COUGHVID_DATA_DIR = "C:\path\to\coughvid_20211012"

   # Linux / macOS
   export COUGHVID_DATA_DIR="/path/to/coughvid_20211012"
   ```

2. **Configuration File:**
   Edit `configs/base_config.yaml`:
   ```yaml
   dataset:
     raw_data_dir: "../AT/ED/coughvid_20211012"  # Relative to project root or absolute
   ```

---

## 3. Running Automated Tests
Run the test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 4. Development Status
- [x] **Milestone 1:** Architecture scaffolding, dependency isolation, external dataset linkage, and configuration validation.
- [ ] **Milestone 2:** Audio DSP and energy-envelope hysteresis cough burst segmentation.
- [ ] **Milestone 3:** Leak-free dataset ingestion and group-aware cross-validation splits.
- [ ] **Milestone 4:** Segmented feature extraction pipeline.
- [ ] **Milestone 5:** Model training, calibration, and regression testing.
- [ ] **Milestone 6:** Final evaluation and historical benchmark comparison.
