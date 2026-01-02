"""
Model Doctor - Drift Detection System
Script 3: Detect Data Drift using Evidently AI
Author: Your Name
Date: January 2026
"""

import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

def detect_drift(reference_data, current_data, report_path='reports/drift_report.html'):
    """
    Detect data drift between reference and current datasets

    Args:
        reference_data: Original training data (baseline)
        current_data: New production data to check
        report_path: Path to save HTML report

    Returns:
        drift_detected: Boolean indicating if drift occurred
        drift_score: Percentage of features that drifted
    """

    # Create Evidently Report
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    # Run the report
    drift_report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML report
    os.makedirs('reports', exist_ok=True)
    drift_report.save_html(report_path)

    # Extract drift metrics
    report_dict = drift_report.as_dict()

    # Get dataset drift status
    dataset_drift = report_dict['metrics'][0]['result']['dataset_drift']
    n_drifted_features = report_dict['metrics'][0]['result']['number_of_drifted_columns']
    n_total_features = report_dict['metrics'][0]['result']['number_of_columns']
    drift_score = (n_drifted_features / n_total_features) * 100

    print("="*70)
    print("DRIFT DETECTION RESULTS")
    print("="*70)
    print(f"Dataset Drift Detected: {dataset_drift}")
    print(f"Number of Drifted Features: {n_drifted_features}/{n_total_features}")
    print(f"Drift Score: {drift_score:.1f}%")
    print(f"\n✓ Detailed report saved to: {report_path}")
    print("="*70)

    return dataset_drift, drift_score

def run_drift_test(reference_data, current_data):
    """Run automated drift tests"""

    # Create test suite
    drift_tests = TestSuite(tests=[
        TestNumberOfDriftedColumns(gte=3)  # Alert if ≥3 features drift
    ])

    drift_tests.run(reference_data=reference_data, current_data=current_data)

    # Get test results
    test_results = drift_tests.as_dict()
    all_tests_passed = test_results['summary']['all_passed']

    return all_tests_passed

if __name__ == "__main__":
    print("="*70)
    print("MODEL DOCTOR: Drift Detection System")
    print("="*70 + "\n")

    # Load datasets
    print("Loading datasets...")
    reference_data = pd.read_csv('data/original_data.csv')
    current_data = pd.read_csv('data/drifted_data.csv')

    print(f"✓ Reference data: {len(reference_data)} samples")
    print(f"✓ Current data: {len(current_data)} samples\n")

    # Run drift detection
    drift_detected, drift_score = detect_drift(reference_data, current_data)

    # Alert system
    if drift_detected:
        print("\n" + "="*70)
        print("⚠️  ALERT: DATA DRIFT DETECTED!")
        print("="*70)
        print(f"Drift severity: {drift_score:.1f}% of features have drifted")
        print("Recommended actions:")
        print("  1. Review the drift report for detailed analysis")
        print("  2. Investigate root causes (data pipeline, user behavior)")
        print("  3. Consider retraining the model with new data")
        print("  4. Monitor model performance metrics closely")
        print("="*70)
    else:
        print("\n✓ No significant drift detected. Model is healthy!")
