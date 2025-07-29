# Project Stack Documentation

## Overview
This document defines the production stack for the Federated, Privacy-Preserving ECG Arrhythmia Detection for Wearables project. Versions are pinned for reproducibility but can be updated via semver constraints in `requirements.txt`. As of July 27, 2025, we've updated to latest compatible versions to address availability and compatibility issues on macOS arm64 (Apple Silicon).

## Core Stack
- **Python**: 3.11.13 (base for all; install via Homebrew: `brew install python@3.11`).
- **TensorFlow**: 2.19.0 (core ML; supports arm64 GPU via tensorflow-metal).
- **TensorFlow-Federated**: 0.87.0 (federated learning; for sharding ECG data across wearables).
- **TensorFlow-Privacy**: 0.9.0 (differential privacy; integrated in model optimizers).
- **Syft (PySyft)**: 0.9.5 (optional secure computation for multi-party federated sim).
- **MLflow**: 3.1.4 (experiment tracking; logs federated rounds).
- **pytest**: 8.4.1 (unit/integration tests; run via `pytest tests/`).
- **Docker**: Base image `python:3.11-slim` (for local/prod containers in `/infra/Dockerfile`).
- **Kubernetes + Kubeflow**: 1.9 on AWS EKS (pipelines deployment; use Kubeflow for federated workflows).

## Installation in `.venv`
Activate venv and run:
```bash
pip install tensorflow==2.19.0 tensorflow-federated==0.87.0 tensorflow-privacy==0.9.0 syft==0.9.5 mlflow==3.1.4 pytest==8.4.1
pip freeze > requirements.txt