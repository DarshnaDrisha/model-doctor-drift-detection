"""
Model Doctor - Drift Detection System
Script 1: Train Baseline Model
Author: Your Name
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def create_dataset():
    """Generate synthetic loan prediction dataset"""
    np.random.seed(42)
    n_samples = 1000

    age = np.random.randint(22, 65, n_samples)
    income = np.random.normal(50000, 15000, n_samples).clip(20000, 150000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    loan_amount = np.random.normal(200000, 80000, n_samples).clip(50000, 500000)
    employment_years = np.random.randint(0, 30, n_samples)
    debt_to_income = (loan_amount / income).clip(0.1, 0.6)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                  n_samples, p=[0.3, 0.4, 0.2, 0.1])
    employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Freelancer'], 
                                       n_samples, p=[0.6, 0.3, 0.1])

    loan_approval_prob = (
        (credit_score / 850) * 0.4 +
        (income / 150000) * 0.3 +
        (1 - debt_to_income) * 0.2 +
        (employment_years / 30) * 0.1
    )
    loan_approved = (loan_approval_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'employment_years': employment_years,
        'debt_to_income_ratio': debt_to_income,
        'education': education,
        'employment_type': employment_type,
        'loan_approved': loan_approved
    })

    return df

def train_model(data, model_path='models/loan_model.pkl'):
    """Train Random Forest classifier"""
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']

    X_encoded = pd.get_dummies(X, columns=['education', 'employment_type'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Training Complete!")
    print(f"Accuracy: {accuracy*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    feature_names = X_encoded.columns.tolist()
    return model, feature_names, accuracy

if __name__ == "__main__":
    print("="*70)
    print("MODEL DOCTOR: Training Baseline Model")
    print("="*70 + "\n")

    # Create dataset
    data = create_dataset()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/original_data.csv', index=False)
    print(f"✓ Dataset created: {len(data)} samples")
    print(f"✓ Loan approval rate: {data['loan_approved'].mean()*100:.1f}%\n")

    # Train model
    model, features, acc = train_model(data)
    print(f"\n✓ Model saved to: models/loan_model.pkl")
    print(f"✓ Training complete with {acc*100:.1f}% accuracy")
