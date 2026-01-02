# 🏥 Model Doctor: ML Drift Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.15-orange.svg)](https://www.evidentlyai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready machine learning monitoring system that detects **data drift** in deployed models and triggers alerts before performance degrades. This project demonstrates MLOps best practices for model maintenance and troubleshooting.

## 🎯 Problem Statement

In production ML systems, models can fail silently when input data distributions change (data drift). This project builds a "Model Doctor" that:
- Monitors incoming data for distribution shifts
- Detects when the "environment" changes (even if the model stays the same)
- Sends alerts before model performance degrades
- Provides actionable insights for model maintenance

## 🧠 The "Plate vs Banana" Analogy

**Scenario**: A model trained to classify bananas on white plates.

- **The Model**: Recognizes bananas (the "Banana")
- **The Data**: Always expects white plates (the "Plate")
- **The Problem**: What if production data shows bananas on blue plates?

The model might fail—not because bananas changed, but because the **plate** (data distribution) changed. This project detects those "plate changes" automatically.

## 📊 Project Architecture

```
Baseline Model → Drift Detection → Alert System → Retraining Pipeline
     ↓                  ↓                 ↓
 Healthy Data    Statistical Tests    Notifications
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/model-doctor-drift-detection.git
cd model-doctor-drift-detection

# Install dependencies
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
# Step 1: Train baseline model
python train_model.py

# Step 2: Create drifted data (simulate financial crisis)
python create_drift_data.py

# Step 3: Detect drift and generate report
python detect_drift.py
```

### View Results

Open `reports/drift_report.html` in your browser to see detailed drift analysis with:
- Feature-by-feature drift scores
- Distribution comparisons (original vs drifted)
- Statistical test results (Kolmogorov-Smirnov, Chi-squared)

## 📁 Project Structure

```
model-doctor-drift-detection/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── data/
│   ├── original_data.csv      # Baseline training data
│   └── drifted_data.csv       # Simulated production data with drift
├── models/
│   └── loan_model.pkl         # Trained Random Forest model
├── reports/
│   └── drift_report.html      # Evidently AI drift report
├── train_model.py             # Script 1: Train baseline model
├── create_drift_data.py       # Script 2: Generate drifted data
└── detect_drift.py            # Script 3: Run drift detection
```

## 🔬 How It Works

### 1. Baseline Model (The "Healthy Patient")

Trains a **Random Forest Classifier** on loan approval data with features:
- Income, Credit Score, Loan Amount
- Employment Years, Debt-to-Income Ratio
- Education Level, Employment Type

**Performance**: ~65% accuracy on test set

### 2. Drift Simulation (The "Sick Patient")

Simulates a **financial crisis** scenario where:
- Incomes drop by **40%** (job losses)
- Credit scores decrease by **100 points** (late payments)
- Loan amounts increase by **20%** (higher need)
- Employment years decrease by **34%** (job instability)

### 3. Drift Detection (The "Doctor's Diagnosis")

Uses **Evidently AI** to automatically:
- Compare distributions between baseline and new data
- Run statistical tests (KS test for numerical, Chi² for categorical)
- Calculate drift score (% of features that drifted)
- Generate visual HTML report

### 4. Alert System

Triggers alerts when:
- Dataset-level drift detected (≥50% features drifted)
- Critical features drift (income, credit_score)
- Drift score exceeds threshold

## 📈 Example Results

```
DRIFT DETECTION RESULTS
======================================================================
Dataset Drift Detected: True
Number of Drifted Features: 6/8
Drift Score: 75.0%

⚠️  ALERT: DATA DRIFT DETECTED!
======================================================================
Drift severity: 75.0% of features have drifted
Recommended actions:
  1. Review the drift report for detailed analysis
  2. Investigate root causes (data pipeline, user behavior)
  3. Consider retraining the model with new data
  4. Monitor model performance metrics closely
```

## 🎓 Key Learnings & Skills Demonstrated

### MLOps & Production ML
- Data drift detection in production systems
- Model monitoring and observability
- Automated alerting pipelines
- Troubleshooting degraded models

### Technical Skills
- **Libraries**: scikit-learn, Evidently AI, pandas, numpy
- **Concepts**: Statistical hypothesis testing, distribution shifts
- **Tools**: Git version control, modular Python code

### Business Impact
- Prevents silent model failures in production
- Reduces downtime through early detection
- Enables proactive model maintenance
- Critical for ML support and DevOps roles

## 🔧 Customization

### Use Your Own Dataset

Replace `create_dataset()` in `train_model.py`:

```python
def create_dataset():
    # Load your data
    df = pd.read_csv('your_dataset.csv')
    return df
```

### Change Drift Scenario

Modify drift parameters in `create_drift_data.py`:

```python
# Example: Simulate seasonal shift
income_drift = np.random.normal(60000, 12000, n_samples)  # Higher income
```

### Adjust Alert Thresholds

Edit drift sensitivity in `detect_drift.py`:

```python
TestNumberOfDriftedColumns(gte=5)  # Alert if ≥5 features drift
```

## 📚 Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Data Drift Explained](https://www.evidentlyai.com/ml-in-production/data-drift)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

**GitHub**: [github.com/yourusername](https://github.com/yourusername)

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Evidently AI team for the excellent drift detection library
- Inspired by real-world MLOps challenges in production systems

---

⭐ **Star this repo** if you find it helpful!
