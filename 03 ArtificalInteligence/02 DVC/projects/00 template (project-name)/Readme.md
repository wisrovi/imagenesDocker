# ⭐ The Definitive MLOps Project Template ⭐

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Poetry](https://img.shields.io/badge/environment-Poetry-blue)
![DVC](https://img.shields.io/badge/Data_Versioning-DVC-blue)
![Status: Ready](https://img.shields.io/badge/Status-Ready_for_Use-brightgreen)

This repository is a **production-grade template** for modern Machine Learning projects. It is designed to enforce MLOps best practices from the start, ensuring your work is reproducible, scalable, and maintainable.

---\n

## ✨ Core Philosophy & Best Practices

This template isn't just a folder structure; it's a workflow that promotes:

*   **Reproducibility:** By versioning code (Git), data (DVC), and environment (`pyproject.toml`), anyone can reproduce your experiments exactly.
*   **Modularity:** Code is organized into distinct scripts for processing, training, etc.
*   **Configuration as Code:** Hyperparameters and settings are managed in `params.yaml`, not hardcoded in scripts.
*   **Automation:** The `scripts/` directory provides a home for all automation, making the project easy to run.
*   **Traceability:** A clear and enforced documentation workflow ensures you always know the lineage of your models.

---\n

## 🚀 Quickstart: Your First 5 Minutes

**Prerequisites:** [Git](https://git-scm.com/), [Python 3.8+](https://www.python.org/), [Poetry](https://python-poetry.org/), and [DVC](https://dvc.org/doc/install).

### 1. Set Up the Project
```bash
# Clone the repository
git clone git@github.com:cimacorporate/dataset-IA.git
cd dataset-IA

# Initialize DVC (if not already done)
dvc init
```

### 2. Configure DVC Remote Storage
You need to tell DVC where to store your large files. Edit the `.dvc/config` file to add your remote storage (e.g., S3, GCS, or even a local directory).

**Example for a local remote (on another drive):**
```ini
# In .dvc/config
[core]
    autostage = true
    remote = minio
['remote "my-local-storage"']
    url = s3://datasets
    endpointurl = http://<host-ip>:30702
    access_key_id = DVC
    secret_access_key = uTA.......OJm
    jobs = 8
    read_timeout = 300
    connect_timeout = 60
```
Commit the change: `git commit .dvc/config -m "config: set DVC remote"`

### 3. Run a Full Pipeline with Placeholder Data
This template is runnable out-of-the-box.

```bash
# Create dummy raw data
mkdir -p data/raw/20251121_dummy_dataset
echo "feature1,feature2,label" > data/raw/20251121_dummy_dataset/dummy_data.csv
echo "1,2,0" >> data/raw/20251121_dummy_dataset/dummy_data.csv

# Track the raw data with DVC
dvc add data/raw/20251121_dummy_dataset

# Run the data processing script
poetry run python scripts/process_data.py \
  --raw-path data/raw/20251121_dummy_dataset \
  --processed-path data/processed/20251121_processed_dummy

# Track the processed data
dvc add data/processed/20251121_processed_dummy

# Run the training script
poetry run python scripts/train.py \
  --data-path data/processed/20251121_processed_dummy \
  --output-path model/dummy_v1 \
  --params params.yaml

# Track the new model
dvc add model/dummy_v1

# Commit your first full run!
git add .
git commit -m "feat: initial run of the full pipeline"

# Push everything to your remotes
git push
dvc push
```
You have now run a full, versioned, and reproducible ML pipeline!

---

## 📂 In-Depth File Guide

*   **`pyproject.toml`**: Defines all Python dependencies for both development and production. Use `poetry add <package>` to add new ones.
*   **`params.yaml`**: Your project's control panel. All hyperparameters for training and data processing should be stored here.
*   **`.gitignore`**: Carefully configured to ignore environment folders, IDE files, and local data, ensuring your repository stays clean.
*   **`scripts/`**: Contains the core logic of your project.
    *   `process_data.py`: Takes raw data as input, performs transformations, and saves the result to `data/processed`.
    *   `train.py`: Takes processed data and `params.yaml` as input, trains the model, and saves the artifacts (weights, logs) to `model/`.
*   **`data/`**: The heart of your project, managed by DVC.
    *   `external/`: For documenting datasets downloaded from third-party sources.
    *   `raw/`: For immutable, original data. **Never edit files here.**
    *   `processed/`: For cleaned, transformed data ready for training.
*   **`#model-name#/`**: The main project package. **Remember to rename this.**
    *   `Readme.md`: The **Experiment Log**. This is where you connect the dots, noting which model version was trained on which data version.
*   **`model/`**: Stores trained model artifacts. Each training run should output to a new versioned subdirectory (e.g., `model/yolo_v1.2`).

---

## 🔄 The MLOps Workflow in Practice

1.  **New Data Arrives**:
    *   Place it in `data/raw/`.
    *   `dvc add data/raw/<new_data_folder>`
    *   `git commit -m "data: add raw data version X"`

2.  **Run an Experiment**:
    *   Modify `params.yaml` with new hyperparameters if needed.
    *   Run your processing and training scripts.
    *   `dvc add data/processed/<new_processed_data>`
    *   `dvc add model/<new_model_version>`
    *   `git commit -m "exp: run experiment with model vX and data vY"`

3.  **Switch Between Experiments**:
    *   Want to revert to an old experiment? It's as simple as:
        ```bash
        git checkout <commit_hash_of_old_experiment>
        dvc checkout
        ```
    *   This command will revert your code, `params.yaml`, and use DVC to pull the exact data and model files associated with that experiment.

---

## 🤝 Contribution Guidelines

1.  **Fork** the repository.
2.  Create a new feature branch: `git checkout -b feature/my-new-feature`.
3.  Make your changes. Add your code, update `params.yaml`, etc.
4.  Commit your changes with clear, descriptive messages.
5.  Push to your branch: `git push origin feature/my-new-feature`.
6.  Open a **Pull Request**.

---

## 📜 License

This project template is licensed under the **MIT License**.