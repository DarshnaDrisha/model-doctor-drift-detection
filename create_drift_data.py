"""
Model Doctor - Drift Detection System
Script 2: Create Drifted Dataset (Financial Crisis Simulation)
Author: Your Name
Date: January 2026
"""

import pandas as pd
import numpy as np
import os

def create_drifted_data():
    """Simulate financial crisis with data drift"""
    np.random.seed(100)
    n_samples = 300

    # Financial Crisis: Lower incomes, worse credit, higher loan needs
    age = np.random.randint(22, 65, n_samples)
    income = np.random.normal(30000, 10000, n_samples).clip(15000, 80000)  # 40% drop
    credit_score = np.random.normal(550, 90, n_samples).clip(300, 750)  # 100 point drop
    loan_amount = np.random.normal(240000, 85000, n_samples).clip(60000, 500000)  # 20% increase
    employment_years = np.random.randint(0, 20, n_samples)  # Job instability
    debt_to_income = (loan_amount / income).clip(0.1, 0.6)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                 n_samples, p=[0.4, 0.35, 0.2, 0.05])
    employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Freelancer'], 
                                      n_samples, p=[0.4, 0.3, 0.3])

    loan_approval_prob = (
        (credit_score / 850) * 0.4 +
        (income / 150000) * 0.3 +
        (1 - debt_to_income) * 0.2 +
        (employment_years / 30) * 0.1
    )
    loan_approved = (loan_approval_prob + np.random.normal(0, 0.1, n_samples) > 0.6).astype(int)

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

if __name__ == "__main__":
    print("="*70)
    print("MODEL DOCTOR: Creating Drifted Data (Financial Crisis)")
    print("="*70 + "\n")

    # Load original data for comparison
    original_data = pd.read_csv('data/original_data.csv')

    # Create drifted data
    drifted_data = create_drifted_data()

    # Save
    os.makedirs('data', exist_ok=True)
    drifted_data.to_csv('data/drifted_data.csv', index=False)

    # Show comparison
    print("DRIFT SIMULATION: Financial Crisis Scenario\n")
    print(f"Original Income Mean: ${original_data['income'].mean():,.0f}")
    print(f"Drifted Income Mean: ${drifted_data['income'].mean():,.0f}")
    print(f"Change: {((drifted_data['income'].mean() / original_data['income'].mean() - 1) * 100):.1f}%\n")

    print(f"Original Credit Score: {original_data['credit_score'].mean():.0f}")
    print(f"Drifted Credit Score: {drifted_data['credit_score'].mean():.0f}")
    print(f"Change: {((drifted_data['credit_score'].mean() / original_data['credit_score'].mean() - 1) * 100):.1f}%\n")

    print(f"Original Approval Rate: {original_data['loan_approved'].mean()*100:.1f}%")
    print(f"Drifted Approval Rate: {drifted_data['loan_approved'].mean()*100:.1f}%")
    print(f"Change: {((drifted_data['loan_approved'].mean() / original_data['loan_approved'].mean() - 1) * 100):.1f}%\n")

    print(f"✓ Drifted dataset saved to: data/drifted_data.csv")
