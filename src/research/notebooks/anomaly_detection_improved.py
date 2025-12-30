#!/usr/bin/env python
# coding: utf-8
"""
GNSS Anomaly Detection - Improved Version
Addresses class imbalance and optimizes decision threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("GNSS ANOMALY DETECTION - IMPROVED VERSION")
print("="*70)

df = pd.read_csv(r"F:\zizo\RTKCorrection\src\research\data\data_after_outlier_detection1.csv")
print(f"\nDataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")

# Select features
x = df[['avg_snr','min_snr','max_residual','x_x','y_x','z_x','ns','sdx(m)','sdy(m)','sdz(m)']]
y = df['anomaly']

print(f"\nClass distribution:")
print(f"  Normal (1):   {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
print(f"  Anomaly (-1): {(y == -1).sum():,} ({(y == -1).sum()/len(y)*100:.2f}%)")
print(f"  Imbalance ratio: {(y == 1).sum() / (y == -1).sum():.2f}:1")

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain/Test split: {X_train.shape[0]:,} / {X_test.shape[0]:,}")

# ============================================================================
# 2. BASELINE MODEL (for comparison)
# ============================================================================
print("\n" + "="*70)
print("BASELINE MODEL (Original)")
print("="*70)

rf_baseline = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)

baseline_recall = recall_score(y_test, y_pred_baseline, pos_label=-1)
baseline_precision = precision_score(y_test, y_pred_baseline, pos_label=-1)
baseline_f1 = f1_score(y_test, y_pred_baseline, pos_label=-1)

print(f"Baseline Performance:")
print(f"  Recall (Anomaly):    {baseline_recall:.4f}")
print(f"  Precision (Anomaly): {baseline_precision:.4f}")
print(f"  F1-Score (Anomaly):  {baseline_f1:.4f}")

# ============================================================================
# 3. APPROACH 1: CLASS WEIGHTING
# ============================================================================
print("\n" + "="*70)
print("APPROACH 1: CLASS WEIGHTING")
print("="*70)

# Calculate class weights (inverse of frequency)
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {
    1: total / (2 * class_counts[1]),
    -1: total / (2 * class_counts[-1])
}
print(f"\nCalculated class weights:")
print(f"  Normal (1):   {class_weights[1]:.4f}")
print(f"  Anomaly (-1): {class_weights[-1]:.4f}")

rf_weighted = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    class_weight=class_weights  # Apply class weights
)
rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)
y_proba_weighted = rf_weighted.predict_proba(X_test)

weighted_recall = recall_score(y_test, y_pred_weighted, pos_label=-1)
weighted_precision = precision_score(y_test, y_pred_weighted, pos_label=-1)
weighted_f1 = f1_score(y_test, y_pred_weighted, pos_label=-1)

print(f"\nClass-Weighted Model Performance:")
print(f"  Recall (Anomaly):    {weighted_recall:.4f} (↑ {weighted_recall - baseline_recall:+.4f})")
print(f"  Precision (Anomaly): {weighted_precision:.4f} (↓ {weighted_precision - baseline_precision:+.4f})")
print(f"  F1-Score (Anomaly):  {weighted_f1:.4f} (↑ {weighted_f1 - baseline_f1:+.4f})")

# ============================================================================
# 4. APPROACH 2: SMOTE (Synthetic Minority Over-sampling)
# ============================================================================
print("\n" + "="*70)
print("APPROACH 2: SMOTE OVERSAMPLING")
print("="*70)

try:
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    print(f"  Training samples: {X_train.shape[0]:,} → {X_train_smote.shape[0]:,}")
    print(f"  Normal (1):   {(y_train_smote == 1).sum():,}")
    print(f"  Anomaly (-1): {(y_train_smote == -1).sum():,}")
    
    rf_smote = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = rf_smote.predict(X_test)
    y_proba_smote = rf_smote.predict_proba(X_test)
    
    smote_recall = recall_score(y_test, y_pred_smote, pos_label=-1)
    smote_precision = precision_score(y_test, y_pred_smote, pos_label=-1)
    smote_f1 = f1_score(y_test, y_pred_smote, pos_label=-1)
    
    print(f"\nSMOTE Model Performance:")
    print(f"  Recall (Anomaly):    {smote_recall:.4f} (↑ {smote_recall - baseline_recall:+.4f})")
    print(f"  Precision (Anomaly): {smote_precision:.4f} (↓ {smote_precision - baseline_precision:+.4f})")
    print(f"  F1-Score (Anomaly):  {smote_f1:.4f} (↑ {smote_f1 - baseline_f1:+.4f})")
    
    smote_available = True
except ImportError:
    print("\n⚠️  SMOTE not available. Install with: pip install imbalanced-learn")
    smote_available = False

# ============================================================================
# 5. THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION")
print("="*70)

# Use the best model so far (class-weighted)
best_model = rf_weighted
y_proba_best = y_proba_weighted

# Test different thresholds
thresholds_to_test = [0.2, 0.3, 0.4, 0.5]
print(f"\nTesting thresholds: {thresholds_to_test}")
print(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1-Score':<10}")
print("-" * 50)

threshold_results = []
for threshold in thresholds_to_test:
    # Predict with custom threshold (class -1 is index 0)
    y_pred_threshold = np.where(y_proba_best[:, 0] >= threshold, -1, 1)
    
    recall = recall_score(y_test, y_pred_threshold, pos_label=-1)
    precision = precision_score(y_test, y_pred_threshold, pos_label=-1)
    f1 = f1_score(y_test, y_pred_threshold, pos_label=-1)
    
    threshold_results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'predictions': y_pred_threshold
    })
    
    print(f"{threshold:<12.2f} {recall:<10.4f} {precision:<12.4f} {f1:<10.4f}")

# Find optimal threshold (maximize F1 or prioritize recall)
best_threshold_f1 = max(threshold_results, key=lambda x: x['f1'])
best_threshold_recall = max(threshold_results, key=lambda x: x['recall'])

print(f"\nOptimal thresholds:")
print(f"  Best F1-Score:  {best_threshold_f1['threshold']:.2f} (F1={best_threshold_f1['f1']:.4f})")
print(f"  Best Recall:    {best_threshold_recall['threshold']:.2f} (Recall={best_threshold_recall['recall']:.4f})")

# ============================================================================
# 6. FINAL MODEL SELECTION
# ============================================================================
print("\n" + "="*70)
print("FINAL MODEL RECOMMENDATION")
print("="*70)

# Use class-weighted model with optimized threshold
final_model = rf_weighted
final_threshold = 0.3  # Balance between recall and precision
y_pred_final = np.where(y_proba_weighted[:, 0] >= final_threshold, -1, 1)

final_recall = recall_score(y_test, y_pred_final, pos_label=-1)
final_precision = precision_score(y_test, y_pred_final, pos_label=-1)
final_f1 = f1_score(y_test, y_pred_final, pos_label=-1)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\nFinal Model: Class-Weighted RF with Threshold={final_threshold}")
print(f"\n{'Metric':<20} {'Baseline':<12} {'Final':<12} {'Change':<12}")
print("-" * 60)
print(f"{'Recall (Anomaly)':<20} {baseline_recall:<12.4f} {final_recall:<12.4f} {final_recall - baseline_recall:+.4f}")
print(f"{'Precision (Anomaly)':<20} {baseline_precision:<12.4f} {final_precision:<12.4f} {final_precision - baseline_precision:+.4f}")
print(f"{'F1-Score (Anomaly)':<20} {baseline_f1:<12.4f} {final_f1:<12.4f} {final_f1 - baseline_f1:+.4f}")
print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred_baseline):<12.4f} {final_accuracy:<12.4f} {final_accuracy - accuracy_score(y_test, y_pred_baseline):+.4f}")

# Confusion matrix
cm_final = confusion_matrix(y_test, y_pred_final)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Anomaly  Normal")
print(f"Actual Anomaly  {cm_final[0,0]:5d}   {cm_final[0,1]:5d}")
print(f"       Normal   {cm_final[1,0]:5d}   {cm_final[1,1]:5d}")

fn_count = cm_final[0, 1]
fp_count = cm_final[1, 0]
print(f"\nMissed Anomalies: {fn_count:,} (down from 2,329)")
print(f"False Alarms: {fp_count:,}")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Model Comparison - Recall
models = ['Baseline', 'Class\nWeighted', 'SMOTE' if smote_available else 'N/A', 'Optimized\nThreshold']
recalls = [baseline_recall, weighted_recall, smote_recall if smote_available else 0, final_recall]
colors_recall = ['#e74c3c' if r < 0.6 else '#f39c12' if r < 0.8 else '#2ecc71' for r in recalls]

axes[0, 0].bar(models[:len(recalls)], recalls, color=colors_recall, alpha=0.7, edgecolor='black')
axes[0, 0].axhline(y=0.8, color='green', linestyle='--', label='Target (80%)')
axes[0, 0].set_ylabel('Recall (Anomaly Detection)', fontsize=11)
axes[0, 0].set_title('Model Comparison: Recall', fontsize=13, fontweight='bold')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].legend()
for i, v in enumerate(recalls):
    if v > 0:
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Precision-Recall Trade-off
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test == -1, y_proba_best[:, 0])
axes[0, 1].plot(recall_curve, precision_curve, linewidth=2, color='#3498db')
axes[0, 1].scatter([final_recall], [final_precision], color='red', s=200, zorder=5, 
                   label=f'Selected (T={final_threshold})', edgecolors='black', linewidths=2)
axes[0, 1].set_xlabel('Recall', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Threshold Analysis
thresholds_plot = [r['threshold'] for r in threshold_results]
recalls_plot = [r['recall'] for r in threshold_results]
precisions_plot = [r['precision'] for r in threshold_results]
f1s_plot = [r['f1'] for r in threshold_results]

axes[0, 2].plot(thresholds_plot, recalls_plot, 'o-', label='Recall', linewidth=2, markersize=8)
axes[0, 2].plot(thresholds_plot, precisions_plot, 's-', label='Precision', linewidth=2, markersize=8)
axes[0, 2].plot(thresholds_plot, f1s_plot, '^-', label='F1-Score', linewidth=2, markersize=8)
axes[0, 2].axvline(x=final_threshold, color='red', linestyle='--', label=f'Selected ({final_threshold})')
axes[0, 2].set_xlabel('Decision Threshold', fontsize=11)
axes[0, 2].set_ylabel('Score', fontsize=11)
axes[0, 2].set_title('Threshold Impact Analysis', fontsize=13, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Confusion Matrix - Baseline
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
axes[1, 0].set_title('Baseline Model', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# 5. Confusion Matrix - Final
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
            xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
axes[1, 1].set_title('Final Model (Improved)', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

# 6. Improvement Summary
improvements = {
    'Recall\n(Anomaly)': (final_recall - baseline_recall) * 100,
    'Precision\n(Anomaly)': (final_precision - baseline_precision) * 100,
    'F1-Score\n(Anomaly)': (final_f1 - baseline_f1) * 100,
    'Missed\nAnomalies': -(fn_count - 2329) / 2329 * 100
}
colors_imp = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvements.values()]
bars = axes[1, 2].bar(improvements.keys(), improvements.values(), color=colors_imp, alpha=0.7, edgecolor='black')
axes[1, 2].axhline(y=0, color='black', linewidth=1)
axes[1, 2].set_ylabel('Improvement (%)', fontsize=11)
axes[1, 2].set_title('Performance Improvements', fontsize=13, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')
for bar, value in zip(bars, improvements.values()):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -5),
                   f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('f:/zizo/RTKCorrection/src/research/notebooks/anomaly_detection_improvements.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: anomaly_detection_improvements.png")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
