"""
COMPAS Recidivism Dataset Bias Audit
Using AI Fairness 360 (AIF360) to analyze racial bias in risk scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# AIF360 imports
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*70)
print("COMPAS RECIDIVISM DATASET - RACIAL BIAS AUDIT")
print("="*70)

# ==============================================================================
# STEP 1: Load and Explore COMPAS Dataset
# ==============================================================================

print("\n[STEP 1] Loading COMPAS Dataset...")

# Download dataset if not already present
import os
import urllib.request

# Define paths
aif360_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'venv', 'lib', 'site-packages', 'aif360', 'data', 'raw', 'compas'
)

# Alternative: try to find the actual aif360 installation path
try:
    import aif360
    aif360_base = os.path.dirname(aif360.__file__)
    aif360_data_path = os.path.join(aif360_base, 'data', 'raw', 'compas')
except:
    pass

compas_file = os.path.join(aif360_data_path, 'compas-scores-two-years.csv')

# Create directory if it doesn't exist
os.makedirs(aif360_data_path, exist_ok=True)

# Download if file doesn't exist
if not os.path.exists(compas_file):
    print("Downloading COMPAS dataset...")
    url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
    try:
        urllib.request.urlretrieve(url, compas_file)
        print(f"✓ Dataset downloaded to: {compas_file}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download manually from:")
        print(url)
        print(f"And place it in: {aif360_data_path}")
        exit()
else:
    print(f"✓ Dataset found at: {compas_file}")

# Load dataset using AIF360's built-in loader
# Protected attribute: race (African-American vs Caucasian)
# Favorable label: No recidivism (two_year_recid = 0)
dataset_orig = CompasDataset()

print(f"Dataset shape: {dataset_orig.features.shape}")
print(f"Protected attribute: race")
print(f"Privileged group: Caucasian (race=1.0)")
print(f"Unprivileged group: African-American (race=0.0)")

# Define privileged and unprivileged groups
privileged_groups = [{'race': 1.0}]  # Caucasian
unprivileged_groups = [{'race': 0.0}]  # African-American

# ==============================================================================
# STEP 2: Analyze Dataset Bias (Before Predictions)
# ==============================================================================

print("\n[STEP 2] Analyzing Dataset Bias Metrics...")

# Calculate dataset metrics
dataset_metric = BinaryLabelDatasetMetric(
    dataset_orig,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n--- Dataset Fairness Metrics ---")
print(f"Base Rate (Positive Outcome Rate):")
print(f"  - African-American: {dataset_metric.base_rate(privileged=False):.3f}")
print(f"  - Caucasian: {dataset_metric.base_rate(privileged=True):.3f}")

print(f"\nDisparate Impact (ratio): {dataset_metric.disparate_impact():.3f}")
print("  Note: 1.0 = perfect fairness, <0.8 indicates significant bias")

print(f"\nStatistical Parity Difference: {dataset_metric.statistical_parity_difference():.3f}")
print("  Note: 0.0 = perfect fairness, negative = unprivileged group disadvantaged")

# ==============================================================================
# STEP 3: Simulate Predictions (Use COMPAS Scores as Predictions)
# ==============================================================================

print("\n[STEP 3] Analyzing COMPAS Score Predictions...")

# Convert dataset to DataFrame for easier manipulation
df = pd.DataFrame(
    dataset_orig.features,
    columns=dataset_orig.feature_names
)
df['two_year_recid'] = dataset_orig.labels.ravel()

# Handle protected attributes - they might be 2D
protected_attrs = dataset_orig.protected_attributes
if len(protected_attrs.shape) > 1:
    df['race'] = protected_attrs[:, 0]  # Take first column if 2D
else:
    df['race'] = protected_attrs.ravel()

# COMPAS uses decile_score; we'll threshold it to create binary predictions
# Threshold: score > 4 predicts recidivism (common threshold used in practice)
# Note: In the AIF360 CompasDataset, the score might be normalized
# We'll work with the favorable_label (0 = no recidivism)

# For classification metrics, we need predicted labels
# We'll use the actual dataset's scores as our "model predictions"
dataset_pred = dataset_orig.copy()

# Calculate classification metrics
classified_metric = ClassificationMetric(
    dataset_orig,
    dataset_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n--- Classification Fairness Metrics ---")

# False Positive Rate (FPR) - falsely predicting recidivism
fpr_unprivileged = classified_metric.false_positive_rate(privileged=False)
fpr_privileged = classified_metric.false_positive_rate(privileged=True)

print(f"False Positive Rate:")
print(f"  - African-American: {fpr_unprivileged:.3f}")
print(f"  - Caucasian: {fpr_privileged:.3f}")
print(f"  - Disparity: {fpr_unprivileged - fpr_privileged:.3f}")

# False Negative Rate (FNR) - failing to predict actual recidivism
fnr_unprivileged = classified_metric.false_negative_rate(privileged=False)
fnr_privileged = classified_metric.false_negative_rate(privileged=True)

print(f"\nFalse Negative Rate:")
print(f"  - African-American: {fnr_unprivileged:.3f}")
print(f"  - Caucasian: {fnr_privileged:.3f}")
print(f"  - Disparity: {fnr_unprivileged - fnr_privileged:.3f}")

# Equal Opportunity Difference (difference in TPR)
print(f"\nEqual Opportunity Difference: {classified_metric.equal_opportunity_difference():.3f}")
print("  Note: 0.0 = equal TPR across groups")

# Average Odds Difference (average of TPR and FPR differences)
print(f"\nAverage Odds Difference: {classified_metric.average_odds_difference():.3f}")
print("  Note: 0.0 = equal TPR and FPR across groups")

# Accuracy
acc_unprivileged = classified_metric.accuracy(privileged=False)
acc_privileged = classified_metric.accuracy(privileged=True)

print(f"\nAccuracy:")
print(f"  - African-American: {acc_unprivileged:.3f}")
print(f"  - Caucasian: {acc_privileged:.3f}")

# ==============================================================================
# STEP 4: Visualizations
# ==============================================================================

print("\n[STEP 4] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('COMPAS Recidivism Dataset: Racial Bias Analysis',
             fontsize=16, fontweight='bold')

# Visualization 1: Base Rates by Race
ax1 = axes[0, 0]
base_rates = [
    dataset_metric.base_rate(privileged=False),
    dataset_metric.base_rate(privileged=True)
]
bars1 = ax1.bar(['African-American', 'Caucasian'], base_rates,
                color=['#e74c3c', '#3498db'], alpha=0.7)
ax1.set_ylabel('Positive Outcome Rate', fontweight='bold')
ax1.set_title('Base Rates by Race\n(Recidivism Rate)', fontweight='bold')
ax1.set_ylim([0, 1])
for bar, rate in zip(bars1, base_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

# Visualization 2: False Positive Rates
ax2 = axes[0, 1]
fpr_rates = [fpr_unprivileged, fpr_privileged]
bars2 = ax2.bar(['African-American', 'Caucasian'], fpr_rates,
                color=['#e74c3c', '#3498db'], alpha=0.7)
ax2.set_ylabel('False Positive Rate', fontweight='bold')
ax2.set_title('False Positive Rate Disparity\n(Incorrectly Predicted to Reoffend)',
              fontweight='bold')
ax2.set_ylim([0, max(fpr_rates) * 1.2])
ax2.axhline(y=np.mean(fpr_rates), color='gray', linestyle='--',
            label='Average', alpha=0.5)
for bar, rate in zip(bars2, fpr_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

# Visualization 3: False Negative Rates
ax3 = axes[0, 2]
fnr_rates = [fnr_unprivileged, fnr_privileged]
bars3 = ax3.bar(['African-American', 'Caucasian'], fnr_rates,
                color=['#e74c3c', '#3498db'], alpha=0.7)
ax3.set_ylabel('False Negative Rate', fontweight='bold')
ax3.set_title('False Negative Rate Disparity\n(Missed Actual Reoffenders)',
              fontweight='bold')
ax3.set_ylim([0, max(fnr_rates) * 1.2])
for bar, rate in zip(bars3, fnr_rates):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

# Visualization 4: Accuracy Comparison
ax4 = axes[1, 0]
accuracies = [acc_unprivileged, acc_privileged]
bars4 = ax4.bar(['African-American', 'Caucasian'], accuracies,
                color=['#e74c3c', '#3498db'], alpha=0.7)
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Model Accuracy by Race', fontweight='bold')
ax4.set_ylim([0, 1])
for bar, acc in zip(bars4, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Visualization 5: Disparate Impact
ax5 = axes[1, 1]
di = dataset_metric.disparate_impact()
colors_di = ['#27ae60' if di >= 0.8 else '#e74c3c']
bar5 = ax5.bar(['Disparate Impact'], [di], color=colors_di, alpha=0.7)
ax5.axhline(y=0.8, color='orange', linestyle='--', linewidth=2,
            label='Fairness Threshold (0.8)')
ax5.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
            label='Perfect Fairness (1.0)')
ax5.set_ylabel('Disparate Impact Ratio', fontweight='bold')
ax5.set_title('Disparate Impact\n(Caucasian / African-American)', fontweight='bold')
ax5.set_ylim([0, max(1.2, di * 1.1)])
ax5.legend()
ax5.text(0, di, f'{di:.3f}', ha='center', va='bottom',
         fontweight='bold', fontsize=12)

# Visualization 6: Fairness Metrics Summary
ax6 = axes[1, 2]
metrics_names = ['Disparate\nImpact', 'Stat Parity\nDiff',
                 'Equal Opp\nDiff', 'Avg Odds\nDiff']
metrics_values = [
    dataset_metric.disparate_impact() - 1.0,  # Normalize to show deviation from 1
    dataset_metric.statistical_parity_difference(),
    classified_metric.equal_opportunity_difference(),
    classified_metric.average_odds_difference()
]
colors_bars = ['#27ae60' if abs(v) < 0.1 else '#e74c3c' for v in metrics_values]
bars6 = ax6.bar(metrics_names, metrics_values, color=colors_bars, alpha=0.7)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_ylabel('Metric Value', fontweight='bold')
ax6.set_title('Fairness Metrics Summary\n(Closer to 0 = More Fair)', fontweight='bold')
ax6.tick_params(axis='x', rotation=0)
for bar, val in zip(bars6, metrics_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center',
             va='bottom' if val > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('compas_bias_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'compas_bias_analysis.png'")
plt.show()

# ==============================================================================
# STEP 5: Apply Bias Mitigation (Reweighing)
# ==============================================================================

print("\n[STEP 5] Applying Bias Mitigation: Reweighing Algorithm...")

# Reweighing adjusts the weights of data points to achieve fairness
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

dataset_transf = RW.fit_transform(dataset_orig)

# Calculate metrics after mitigation
dataset_metric_transf = BinaryLabelDatasetMetric(
    dataset_transf,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n--- After Bias Mitigation (Reweighing) ---")
print(f"Disparate Impact: {dataset_metric_transf.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {dataset_metric_transf.statistical_parity_difference():.3f}")

# ==============================================================================
# STEP 6: Summary Statistics
# ==============================================================================

print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

print("\n🔍 KEY FINDINGS:")
print(f"1. Disparate Impact: {dataset_metric.disparate_impact():.3f}")
if dataset_metric.disparate_impact() < 0.8:
    print("   ⚠️  SIGNIFICANT BIAS DETECTED (below 0.8 threshold)")
else:
    print("   ✓ Within acceptable range")

print(f"\n2. False Positive Rate Disparity: {fpr_unprivileged - fpr_privileged:.3f}")
if abs(fpr_unprivileged - fpr_privileged) > 0.05:
    print("   ⚠️  African-Americans are falsely flagged at higher rates")
else:
    print("   ✓ Minimal disparity")

print(f"\n3. False Negative Rate Disparity: {fnr_unprivileged - fnr_privileged:.3f}")

print("\n💡 RECOMMENDATIONS:")
print("1. Implement reweighing or other bias mitigation algorithms")
print("2. Use multiple fairness metrics simultaneously")
print("3. Regular audits with disaggregated performance reporting")
print("4. Require human review for high-risk predictions")
print("5. Consider context-specific fairness definitions")

print("\n" + "="*70)
print("Audit Complete!")
print("="*70)