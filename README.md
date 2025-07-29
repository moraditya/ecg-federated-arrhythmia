# Federated, Privacy-Preserving ECG Arrhythmia Detection
A project for developing a federated learning system for ECG arrhythmia detection on wearables, ensuring privacy with TensorFlow Federated and TensorFlow Privacy.

## Project Structure 
- `src/`: Source code
  - `data/`: Data processing and loading
  - `federated/`: Federated learning logic
  - `models/`: Model definitions
  - `pipelines/`: Training and evaluation pipelines
- `tests/`: Unit and integration tests
- `docs/`: Documentation
- `infra/`: Infrastructure (Docker, Kubernetes, Kubeflow)

## Setup
(To be updated with setup instructions)

## Data Setup

To use the real MIT-BIH Arrhythmia Database for this project, download the data to `data/mitbih/`:

```bash
wget https://physionet.org/files/mitdb/1.0.0/mitdbdir.tar
tar -xvf mitdbdir.tar -C data/mitbih/
```

Note: After untarring, ensure the .dat and .atr files for all 48 records are directly in `data/mitbih/` (move or rename as needed). This is required for the loading functions in `src/data/load_ecg.py`. For testing without real data, synthetic generation is included in the code.
