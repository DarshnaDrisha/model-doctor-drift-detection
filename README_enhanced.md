# 🏥 Model Doctor: ML Drift Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.15-orange.svg)](https://www.evidentlyai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready machine learning monitoring system that detects **data drift** in deployed models and triggers alerts before performance degrades. This project demonstrates MLOps best practices for model maintenance and troubleshooting.

![Drift Detection Alert](screenshots/alert_screenshot.png)
*Automated alert system detecting 75% feature drift*

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
Baseline Model → Drift Detection → Alert System → Support Playbook
     ↓                  ↓                 ↓              ↓
 Healthy Data    Statistical Tests    Notifications   Troubleshooting
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DarshnaDrisha/model-doctor-drift-detection.git
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

Open `reports/drift_report.html` in your browser to see detailed drift analysis.

![Drift Report](screenshots/drift_report_screenshot.png)
*Interactive Evidently AI report showing distribution shifts*

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
├── screenshots/
│   ├── alert_screenshot.png   # Terminal alert output
│   ├── drift_report_screenshot.png  # HTML report visualization
│   └── code_screenshot.png    # Data drift simulation code
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

![Drift Simulation Code](screenshots/code_screenshot.png)
*Code snippet showing intentional data poisoning for drift simulation*

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

## 🔬 The Statistical Logic (The "Why")

As someone with an **M.Sc. in Applied Mathematics**, I didn't just use a library—I understand the mathematics behind drift detection:

### Numerical Features (Kolmogorov-Smirnov Test)
For continuous variables like **Income** and **Credit Score**, I utilize the **Kolmogorov-Smirnov (K-S) test**:

- **Hypothesis**: \(H_0\): Both distributions are identical
- **Test Statistic**: \(D = \sup_x |F_1(x) - F_2(x)|\)
- **Decision Rule**: If \(D > D_{\alpha}\) (critical value), reject \(H_0\) → drift detected
- **Why K-S?**: Non-parametric test that compares entire cumulative distribution functions, not just means

### Categorical Features (Chi-Squared Test)
For discrete variables like **Education** and **Employment Type**, I apply the **Chi-squared (χ²) test**:

- **Test Statistic**: \(\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}\)
- **Where**: \(O_i\) = observed frequency, \(E_i\) = expected frequency
- **Decision Rule**: If \(\chi^2 > \chi^2_{\alpha, df}\), frequency distribution has shifted

### Population Stability Index (PSI)
The system calculates PSI to quantify drift magnitude:

\[
PSI = \sum_{i=1}^{n} (P_{current,i} - P_{reference,i}) \times \ln\left(\frac{P_{current,i}}{P_{reference,i}}\right)
\]

**Interpretation**:
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Moderate drift (monitor closely)
- PSI ≥ 0.2: Severe drift (retrain model immediately)

**Why This Matters**: By using rigorous statistical tests instead of simple accuracy metrics, the system detects **environmental changes** (data distribution shifts) before they degrade model performance.

## 🛠️ Support Engineer Playbook

This project isn't just about detecting errors—it's a **complete troubleshooting workflow** for ML support teams:

### Scenario: Production Model Alert

#### Step 1: The Alert 🚨
```
⚠️  ALERT: DATA DRIFT DETECTED!
Drift severity: 75.0% of features have drifted
Time: 2026-01-02 20:35:00 UTC
Model: loan_approval_v2.pkl
```

**Support Engineer Action**: Acknowledge alert and open drift report

---

#### Step 2: Root Cause Analysis 🔍
Open `reports/drift_report.html` and investigate:

| Feature | Drift Detected | Statistical Test | P-Value | Severity |
|---------|---------------|------------------|---------|----------|
| **income** | ✅ YES | K-S Test | 0.0001 | HIGH |
| **credit_score** | ✅ YES | K-S Test | 0.0023 | HIGH |
| loan_amount | ✅ YES | K-S Test | 0.0456 | MEDIUM |
| employment_years | ✅ YES | K-S Test | 0.0089 | HIGH |
| education | ✅ YES | Chi² Test | 0.0234 | MEDIUM |
| employment_type | ✅ YES | Chi² Test | 0.0178 | MEDIUM |

**Key Finding**: `income` and `credit_score` show extreme drift (p < 0.001)

**Support Engineer Insight**: "Multiple financial features drifted simultaneously—this suggests an external economic event, not a data pipeline bug."

---

#### Step 3: Hypothesis Formation 💡
Correlate drift timing with external events:

- **Check**: Did a new credit-scoring regulation get passed?
- **Check**: Is there an ongoing financial crisis (recession, layoffs)?
- **Check**: Did the data ingestion pipeline change source systems?

**Investigation Result**: Financial crisis started 2 weeks ago (news reports confirm mass layoffs and falling credit scores).

---

#### Step 4: Troubleshooting Decision Tree 🌳

```
Is drift due to data quality issues?
├─ YES → Fix data pipeline, validate upstream sources
└─ NO → Continue

Is drift temporary (seasonal/event-driven)?
├─ YES → Monitor closely, implement temporary thresholds
└─ NO → Permanent shift detected

Permanent shift confirmed?
└─ YES → Model retraining required
```

**Decision**: Drift is permanent (structural economic change) → **Retrain model**

---

#### Step 5: Mitigation Actions 🔧

**Immediate (0-24 hours)**:
1. Enable **manual review queue** for high-risk predictions (loan amounts > $300k)
2. Adjust **decision thresholds** temporarily (increase approval threshold from 0.5 to 0.6)
3. Notify stakeholders (risk team, product managers)

**Short-term (1-7 days)**:
1. Collect **new labeled data** from past 2 weeks (ground truth loan outcomes)
2. Perform **exploratory data analysis** on new data distribution
3. Validate that drift is consistent (not a one-time anomaly)

**Long-term (1-4 weeks)**:
1. **Retrain model** using recent data with updated feature distributions
2. Implement **feature engineering** to make model more robust (e.g., income-to-debt ratio normalization)
3. Update **monitoring thresholds** based on new baseline
4. Deploy **retrained model** with A/B testing (10% traffic initially)

---

#### Step 6: Documentation & Post-Mortem 📋

**Incident Report**:
```
Incident ID: DRIFT-2026-01-02
Model: loan_approval_v2.pkl
Detection Time: 2026-01-02 20:35:00
Root Cause: Economic recession causing 40% income drop, 100-point credit score decrease
Impact: Model accuracy dropped from 65% to 42% (estimated)
Resolution: Model retrained with recent data (v3.pkl deployed 2026-01-09)
Lessons Learned: Need faster retraining pipeline for economic shocks
```

**Support Engineer Value**: This playbook transforms a technical alert into **actionable business decisions**, demonstrating cross-functional troubleshooting skills.

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
- Root cause analysis for ML systems

### Technical Skills
- **Libraries**: scikit-learn, Evidently AI, pandas, numpy
- **Statistics**: Kolmogorov-Smirnov test, Chi-squared test, PSI
- **Concepts**: Distribution shifts, hypothesis testing, population stability
- **Tools**: Git version control, modular Python code

### Business Impact
- Prevents silent model failures in production
- Reduces downtime through early detection
- Enables proactive model maintenance
- Provides clear troubleshooting workflows for support teams
- Critical for ML support, DevOps, and SRE roles

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

## 🖼️ Screenshots

### Terminal Alert Output
![Alert Output](screenshots/alert_screenshot.png)

### Evidently AI Drift Report
![Drift Report](screenshots/drift_report_screenshot.png)

### Data Drift Simulation Code
![Code Snippet](screenshots/code_screenshot.png)

## 📚 Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Data Drift Explained](https://www.evidentlyai.com/ml-in-production/data-drift)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Population Stability Index (PSI)](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a pull request

## 👩‍💻 About the Author

**Darshna Drisha**

M.Tech in Computer Science (3rd Semester) | M.Sc. in Applied Mathematics

Specializing in Computer Vision, Visual Question Answering, and MLOps. Passionate about building production-ready ML systems with robust monitoring and troubleshooting capabilities.

**Connect with me:**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐱 GitHub: [github.com/DarshnaDrisha](https://github.com/DarshnaDrisha)

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Evidently AI** team for the excellent drift detection library
- Inspired by real-world MLOps challenges in production systems
- Statistical methods based on classical hypothesis testing theory

---

⭐ **Star this repo** if you find it helpful for your ML monitoring journey!

## 📊 Project Statistics

- **Lines of Code**: ~400+
- **Technologies**: Python, scikit-learn, Evidently AI, pandas, numpy
- **Dataset**: 1,300 samples (1,000 baseline + 300 drifted)
- **Drift Detection Accuracy**: 75% feature drift successfully identified
- **Statistical Tests**: K-S test, Chi-squared test, PSI calculation
