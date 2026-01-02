# ModelDoctorDriftDetection
**Linux | Windows | MacOS**

Python-based sample project demonstrating **data drift monitoring for deployed machine learning models**, designed as a **beginner-friendly technical guide for ML support and MLOps roles**.

This repository can be used as a **practical reference** to understand how ML models are monitored in production and how **data distribution shifts are detected before failures occur**.

---

## Table of Contents

- What is this?
- Goal of this project
- Project concept
- Requirements
- How to use
- Project workflow
- Drift detection logic
- Alerting logic
- What this project does NOT do
- Use cases
- Project structure
- Contribution
- Author
- License

---

## What is this?

This repository contains **sample Python scripts** that demonstrate how to monitor a deployed machine learning model for **data drift** using Evidently AI.

All code is intentionally kept **simple and readable**, focusing on **monitoring logic rather than complex modeling**, so beginners can understand how real-world ML systems are supported and maintained.

You can freely fork and use this repository for:
- Learning
- Education
- Interview preparation
- ML support / MLOps practice

---

## Goal of this project

The goal of this project is to demonstrate **how ML models fail silently in production** and how a **monitoring system can detect early warning signals**.

Instead of building a complex model, this project focuses on:
- Detecting **covariate drift**
- Alerting when the **input environment changes**
- Supporting **human-in-the-loop troubleshooting**

This mirrors the responsibilities of **ML support engineers and MLOps engineers**.

---

## Project Concept

### The “Model Doctor” idea

- A trained model is treated as a **healthy patient**
- Incoming production data represents the **environment**
- When data distributions shift, the model is considered **at risk**
- The system acts as a **doctor**, diagnosing changes and raising alerts

### Key analogy

> The *banana* (target concept) stays the same,  
> but the *plate* (data distribution) changes.

This project detects **plate changes**, not banana changes.

---

## Requirements

### Runtime requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- evidently (0.4.x)

### Optional (development)

- virtualenv / conda
- VS Code (recommended)

---

## How to use

### 1. Clone this repository

```bash
git clone https://github.com/your-username/model-doctor-drift-detection.git
cd model-doctor-drift-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Project Workflow

### Step 1: Train a baseline model

```bash
python train_model.py
```

---

### Step 2: Create drifted data

```bash
python create_drift_data.py
```

---

### Step 3: Detect data drift

```bash
python detect_drift.py
```

---

## Drift Detection Logic

This project uses **Evidently AI** to detect statistically significant shifts in feature distributions between baseline and new data.

---

## Alerting Logic

An alert is triggered when dataset-level drift is detected and a significant number of features have shifted.

---

## What this project does NOT do

This project intentionally **does NOT**:
- Measure real production accuracy
- Detect label drift or concept drift
- Automatically retrain the model
- Guarantee that drift causes performance degradation

---

## Use Cases

- ML Support Engineer roles
- Entry-level MLOps roles
- Understanding production ML monitoring

---

## Project Structure

```
model-doctor-drift-detection/
├── train_model.py
├── create_drift_data.py
├── detect_drift.py
├── data/
├── models/
├── reports/
└── requirements.txt
```

---

## Author

Darshna Drisha Konwar

---

## License

MIT License
