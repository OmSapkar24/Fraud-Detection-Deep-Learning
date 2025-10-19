# Fraud Detection Platform with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-informational)](https://www.docker.com/)
[![CI](https://github.com/OmSapkar24/Fraud-Detection-Deep-Learning/actions/workflows/ci.yml/badge.svg)](./actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Enterprise-grade fraud detection for transaction streams using deep learning (tabular transformers + GNN for entity linkage) with real-time inference and explainability.

## Overview
This platform ingests transaction data, builds labeled graphs of entities (cards, devices, IPs, merchants), and trains a hybrid model:
- TabTransformer for per-transaction features (categorical + continuous)
- GraphSAGE for entity network risk propagation
- Late-fusion risk head with focal loss and cost-sensitive training
- Real-time inference via Kafka -> FastAPI -> Redis feature store

## Business Context
- Reduce fraud loss and chargebacks while minimizing false positives
- Detect synthetic identities and mule networks
- SLA: <50ms P99 scoring, >90% recall at 1% FPR on recent windows

## Tech Stack
- Data: Kafka, Spark, Delta Lake; Redis for feature store
- Modeling: PyTorch, PyTorch Geometric, scikit-learn, Optuna
- Serving: FastAPI, Uvicorn, Docker, Kubernetes (Helm)
- MLOps: DVC, MLflow, GitHub Actions, pre-commit

## Repository Structure
```
fraud-dl/
  data/                 # sample schemas, synthetic generators
  notebooks/            # EDA, prototyping
  src/
    data/ingest.py      # kafka consumers, batch loaders
    features/build.py   # windowed, aggregations, entity graph build
    models/tabtransformer.py
    models/graph.py
    models/fusion_head.py
    train.py            # training loop, CV, class weights
    infer.py            # batch inference
  serving/
    api.py              # FastAPI scoring with feature fetch
    kafka_consumer.py   # streaming scoring worker
  configs/
    model.yaml          # hyperparams
    features.yaml       # aggregation definitions
  tests/
  docker/
    Dockerfile
  ci/
    ci.yml
```

## Sample Results (synthetic benchmark)
- AUROC: 0.987
- PR-AUC: 0.812
- Recall @ 1% FPR: 92.3%
- Latency (P99): 38ms per transaction (CPU), 12ms (GPU)

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Usage
Train:
```
python -m src.train --config configs/model.yaml
```
Batch inference:
```
python -m src.infer --input data/sample_transactions.parquet --output outputs/scores.parquet
```
Serve API:
```
uvicorn serving.api:app --host 0.0.0.0 --port 8080
```
Docker:
```
docker build -t fraud-dl:latest -f docker/Dockerfile .
```

## Explainability
- SHAP for tabular features
- GNN model explanations via PGExplainer
- Per-score JSON with top factors and entity paths

## Roadmap
- Online training with drift detection (KS/PSI)
- Feature store integration (Feast)
- Graph temporal attention (TGAT)
- Rule learning for analyst review queues

## Disclaimer
This repository uses synthetic data only. Do not upload real PII/PCI.
