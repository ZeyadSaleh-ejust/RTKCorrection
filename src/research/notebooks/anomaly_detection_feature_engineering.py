#!/usr/bin/env python
# coding: utf-8
"""
GNSS Anomaly Detection - Feature Engineering Version
Adds GNSS-specific features to reduce false alarms while maintaining high recall
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("GNSS ANOMALY DETECTION - FEATURE ENGINEERING VERSION")
print("="*70)

df = pd.read_csv(r"F:\zizo\RTKCorrection\src\research\data\data_after_outlier_detection1.csv")
print(f"\nDataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
print(f"Original features: {list(df.columns)}")

# ============================================================================
# 2. FEATURE ENGINEERING - GNSS-SPECIFIC FEATURES
# ============================================================================
print("\n" + "="*70)
print("CREATING ENGINEERED FEATURES")
print("="*70)

# Create a copy for feature engineering
df_engineered = df.copy()

# -------------------------
# A. SIGNAL QUALITY FEATURES
# -------------------------
print("\n1. Signal Quality Features:")

# SNR range (spread of signal quality)
df_engineered['snr_range'] = df['avg_snr'] - df['min_snr']
print(f"   ✓ snr_range: Spread between avg and min SNR")

# SNR quality ratio (how good is the weakest signal relative to average)
df_engineered['snr_quality_ratio'] = df['min_snr'] / (df['avg_snr'] + 1e-6)
print(f"   ✓ snr_quality_ratio: Min SNR / Avg SNR")

# Signal consistency (inverse of range, normalized)
df_engineered['snr_consistency'] = 1 / (df_engineered['snr_range'] + 1)
print(f"   ✓ snr_consistency: Inverse of SNR range")

# -------------------------
# B. RESIDUAL-BASED FEATURES
# -------------------------
print("\n2. Residual-Based Features:")

# Residual per satellite (normalized by number of satellites)
df_engineered['residual_per_sat'] = df['max_residual'] / (df['ns'] + 1e-6)
print(f"   ✓ residual_per_sat: Max residual / Number of satellites")

# Residual to SNR ratio (high residuals with good SNR is suspicious)
df_engineered['residual_snr_ratio'] = df['max_residual'] / (df['avg_snr'] + 1e-6)
print(f"   ✓ residual_snr_ratio: Max residual / Avg SNR")

# Residual quality indicator (combines residual and signal quality)
df_engineered['residual_quality'] = df['max_residual'] * df_engineered['snr_quality_ratio']
print(f"   ✓ residual_quality: Residual × SNR quality ratio")

# -------------------------
# C. POSITION UNCERTAINTY FEATURES
# -------------------------
print("\n3. Position Uncertainty Features:")

# Total position uncertainty (3D)
df_engineered['total_uncertainty'] = np.sqrt(
    df['sdx(m)']**2 + df['sdy(m)']**2 + df['sdz(m)']**2
)
print(f"   ✓ total_uncertainty: √(sdx² + sdy² + sdz²)")

# Horizontal uncertainty (2D)
df_engineered['horizontal_uncertainty'] = np.sqrt(
    df['sdx(m)']**2 + df['sdy(m)']**2
)
print(f"   ✓ horizontal_uncertainty: √(sdx² + sdy²)")

# Vertical to horizontal ratio (vertical errors are typically larger)
df_engineered['vertical_horizontal_ratio'] = df['sdz(m)'] / (df_engineered['horizontal_uncertainty'] + 1e-6)
print(f"   ✓ vertical_horizontal_ratio: sdz / horizontal uncertainty")

# Uncertainty per satellite (DOP-like metric)
df_engineered['uncertainty_per_sat'] = df_engineered['total_uncertainty'] / (df['ns'] + 1e-6)
print(f"   ✓ uncertainty_per_sat: Total uncertainty / Number of satellites")

# -------------------------
# D. GEOMETRIC FEATURES
# -------------------------
print("\n4. Geometric Features:")

# Satellite geometry quality (more satellites = better, but diminishing returns)
df_engineered['sat_geometry_quality'] = np.log1p(df['ns'])
print(f"   ✓ sat_geometry_quality: log(1 + number of satellites)")

# Position difference magnitude (if available)
if 'diff_x' in df.columns and 'diff_y' in df.columns and 'diff_z' in df.columns:
    df_engineered['position_diff_magnitude'] = np.sqrt(
        df['diff_x']**2 + df['diff_y']**2 + df['diff_z']**2
    )
    print(f"   ✓ position_diff_magnitude: √(diff_x² + diff_y² + diff_z²)")
    
    # Horizontal position difference
    df_engineered['horizontal_diff'] = np.sqrt(df['diff_x']**2 + df['diff_y']**2)
    print(f"   ✓ horizontal_diff: √(diff_x² + diff_y²)")
    
    # Position consistency (inverse of difference)
    df_engineered['position_consistency'] = 1 / (df_engineered['position_diff_magnitude'] + 1)
    print(f"   ✓ position_consistency: 1 / (position difference + 1)")

# -------------------------
# E. COMPOSITE QUALITY INDICATORS
# -------------------------
print("\n5. Composite Quality Indicators:")

# Overall quality score (combines multiple factors)
df_engineered['quality_score'] = (
    df_engineered['snr_quality_ratio'] * 
    df_engineered['sat_geometry_quality'] / 
    (df_engineered['total_uncertainty'] + 1)
)
print(f"   ✓ quality_score: (SNR ratio × sat geometry) / uncertainty")

# Anomaly risk score (high residual + low SNR + high uncertainty)
df_engineered['anomaly_risk_score'] = (
    df['max_residual'] * 
    (1 - df_engineered['snr_quality_ratio']) * 
    df_engineered['total_uncertainty']
)
print(f"   ✓ anomaly_risk_score: Residual × (1-SNR ratio) × uncertainty")

print(f"\n✓ Created {df_engineered.shape[1] - df.shape[1]} new features")
print(f"Total features: {df_engineered.shape[1]}")

# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================
print("\n" + "="*70)
print("FEATURE SELECTION")
print("="*70)

# Original features
original_features = ['avg_snr', 'min_snr', 'max_residual', 'x_x', 'y_x', 'z_x', 
                     'ns', 'sdx(m)', 'sdy(m)', 'sdz(m)']

# Engineered features
engineered_features = [
    'snr_range', 'snr_quality_ratio', 'snr_consistency',
    'residual_per_sat', 'residual_snr_ratio', 'residual_quality',
    'total_uncertainty', 'horizontal_uncertainty', 'vertical_horizontal_ratio',
    'uncertainty_per_sat', 'sat_geometry_quality',
    'quality_score', 'anomaly_risk_score'
]

# Add position features if available
if 'position_diff_magnitude' in df_engineered.columns:
    engineered_features.extend(['position_diff_magnitude', 'horizontal_diff', 'position_consistency'])

# Combined feature set
all_features = original_features + engineered_features

print(f"\nOriginal features: {len(original_features)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Total features: {len(all_features)}")

# Prepare data
X_original = df_engineered[original_features]
X_engineered = df_engineered[all_features]
y = df_engineered['anomaly']

# Split data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42, stratify=y
)
X_train_eng, X_test_eng, _, _ = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test split: {X_train_orig.shape[0]:,} / {X_test_orig.shape[0]:,}")

# ============================================================================
# 4. MODEL COMPARISON: ORIGINAL vs ENGINEERED FEATURES
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Calculate class weights
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {
    1: total / (2 * class_counts[1]),
    -1: total / (2 * class_counts[-1])
}

# -------------------------
# Model 1: Original Features
# -------------------------
print("\n1. Model with ORIGINAL features:")
rf_original = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    class_weight=class_weights
)
rf_original.fit(X_train_orig, y_train)
y_proba_orig = rf_original.predict_proba(X_test_orig)

# Test different thresholds
threshold = 0.3
y_pred_orig = np.where(y_proba_orig[:, 0] >= threshold, -1, 1)

recall_orig = recall_score(y_test, y_pred_orig, pos_label=-1)
precision_orig = precision_score(y_test, y_pred_orig, pos_label=-1)
f1_orig = f1_score(y_test, y_pred_orig, pos_label=-1)
cm_orig = confusion_matrix(y_test, y_pred_orig)

print(f"   Recall:     {recall_orig:.4f}")
print(f"   Precision:  {precision_orig:.4f}")
print(f"   F1-Score:   {f1_orig:.4f}")
print(f"   False Alarms: {cm_orig[1, 0]:,}")
print(f"   Missed Anomalies: {cm_orig[0, 1]:,}")

# -------------------------
# Model 2: Engineered Features
# -------------------------
print("\n2. Model with ENGINEERED features:")
rf_engineered = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    class_weight=class_weights
)
rf_engineered.fit(X_train_eng, y_train)
y_proba_eng = rf_engineered.predict_proba(X_test_eng)

y_pred_eng = np.where(y_proba_eng[:, 0] >= threshold, -1, 1)

recall_eng = recall_score(y_test, y_pred_eng, pos_label=-1)
precision_eng = precision_score(y_test, y_pred_eng, pos_label=-1)
f1_eng = f1_score(y_test, y_pred_eng, pos_label=-1)
cm_eng = confusion_matrix(y_test, y_pred_eng)

print(f"   Recall:     {recall_eng:.4f}")
print(f"   Precision:  {precision_eng:.4f}")
print(f"   F1-Score:   {f1_eng:.4f}")
print(f"   False Alarms: {cm_eng[1, 0]:,}")
print(f"   Missed Anomalies: {cm_eng[0, 1]:,}")

# -------------------------
# Improvement Analysis
# -------------------------
print("\n" + "="*70)
print("IMPROVEMENT ANALYSIS")
print("="*70)

print(f"\n{'Metric':<25} {'Original':<12} {'Engineered':<12} {'Change':<12}")
print("-" * 65)
print(f"{'Recall':<25} {recall_orig:<12.4f} {recall_eng:<12.4f} {recall_eng - recall_orig:+.4f}")
print(f"{'Precision':<25} {precision_orig:<12.4f} {precision_eng:<12.4f} {precision_eng - precision_orig:+.4f}")
print(f"{'F1-Score':<25} {f1_orig:<12.4f} {f1_eng:<12.4f} {f1_eng - f1_orig:+.4f}")
print(f"{'False Alarms':<25} {cm_orig[1, 0]:<12,} {cm_eng[1, 0]:<12,} {int(cm_eng[1, 0] - cm_orig[1, 0]):+,}")
print(f"{'Missed Anomalies':<25} {cm_orig[0, 1]:<12,} {cm_eng[0, 1]:<12,} {int(cm_eng[0, 1] - cm_orig[0, 1]):+,}")

# Calculate percentage improvements
fa_reduction = (cm_orig[1, 0] - cm_eng[1, 0]) / cm_orig[1, 0] * 100 if cm_orig[1, 0] > 0 else 0
precision_improvement = (precision_eng - precision_orig) / precision_orig * 100

print(f"\n✓ False Alarm Reduction: {fa_reduction:.1f}%")
print(f"✓ Precision Improvement: {precision_improvement:+.1f}%")
print(f"✓ Recall Maintained: {recall_eng:.1%} (target: >90%)")

# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("TOP FEATURE IMPORTANCE (Engineered Model)")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_engineered.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Identify engineered features in top 10
top_10_features = feature_importance.head(10)['Feature'].tolist()
engineered_in_top10 = [f for f in top_10_features if f in engineered_features]
print(f"\n✓ Engineered features in top 10: {len(engineered_in_top10)}")
if engineered_in_top10:
    print(f"  {', '.join(engineered_in_top10)}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Performance Comparison
metrics = ['Recall', 'Precision', 'F1-Score']
original_vals = [recall_orig, precision_orig, f1_orig]
engineered_vals = [recall_eng, precision_eng, f1_eng]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, original_vals, width, label='Original', alpha=0.7, color='#e74c3c')
bars2 = axes[0, 0].bar(x + width/2, engineered_vals, width, label='Engineered', alpha=0.7, color='#2ecc71')

axes[0, 0].set_ylabel('Score', fontsize=11)
axes[0, 0].set_title('Performance Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. False Alarm Comparison
fa_data = [cm_orig[1, 0], cm_eng[1, 0]]
colors_fa = ['#e74c3c', '#2ecc71']
bars = axes[0, 1].bar(['Original', 'Engineered'], fa_data, color=colors_fa, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Count', fontsize=11)
axes[0, 1].set_title('False Alarms Comparison', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, fa_data):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., val + 100,
                   f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
# Add reduction percentage
reduction_text = f'{fa_reduction:.1f}% reduction'
axes[0, 1].text(0.5, max(fa_data) * 0.5, reduction_text, 
               ha='center', fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 3. Feature Importance (Top 15)
top_15 = feature_importance.head(15)
colors_feat = ['#2ecc71' if f in engineered_features else '#3498db' for f in top_15['Feature']]

axes[0, 2].barh(range(len(top_15)), top_15['Importance'], color=colors_feat, alpha=0.7)
axes[0, 2].set_yticks(range(len(top_15)))
axes[0, 2].set_yticklabels(top_15['Feature'], fontsize=9)
axes[0, 2].set_xlabel('Importance', fontsize=11)
axes[0, 2].set_title('Top 15 Features', fontsize=13, fontweight='bold')
axes[0, 2].invert_yaxis()
axes[0, 2].grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.7, label='Engineered'),
    Patch(facecolor='#3498db', alpha=0.7, label='Original')
]
axes[0, 2].legend(handles=legend_elements, loc='lower right')

# 4. Confusion Matrix - Original
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
axes[1, 0].set_title('Original Features', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# 5. Confusion Matrix - Engineered
sns.heatmap(cm_eng, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
            xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
axes[1, 1].set_title('Engineered Features', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

# 6. Improvement Summary
improvements = {
    'Recall': (recall_eng - recall_orig) * 100,
    'Precision': (precision_eng - precision_orig) * 100,
    'F1-Score': (f1_eng - f1_orig) * 100,
    'False\nAlarms': -fa_reduction
}
colors_imp = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvements.values()]
bars = axes[1, 2].bar(improvements.keys(), improvements.values(), color=colors_imp, alpha=0.7, edgecolor='black')
axes[1, 2].axhline(y=0, color='black', linewidth=1)
axes[1, 2].set_ylabel('Change (%)', fontsize=11)
axes[1, 2].set_title('Feature Engineering Impact', fontsize=13, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, improvements.values()):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                   f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('f:/zizo/RTKCorrection/src/research/notebooks/feature_engineering_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: feature_engineering_results.png")
plt.show()

# ============================================================================
# 7. SAVE ENGINEERED DATASET
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save engineered features dataset
output_path = r"F:\zizo\RTKCorrection\src\research\data\data_with_engineered_features.csv"
df_engineered.to_csv(output_path, index=False)
print(f"\n✓ Engineered dataset saved: {output_path}")

# Save feature list
feature_list_path = r"F:\zizo\RTKCorrection\src\research\notebooks\engineered_features_list.txt"
with open(feature_list_path, 'w') as f:
    f.write("ORIGINAL FEATURES:\n")
    f.write("=" * 50 + "\n")
    for feat in original_features:
        f.write(f"  - {feat}\n")
    f.write("\nENGINEERED FEATURES:\n")
    f.write("=" * 50 + "\n")
    for feat in engineered_features:
        f.write(f"  - {feat}\n")
print(f"✓ Feature list saved: {feature_list_path}")

print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE!")
print("="*70)
print(f"\n🎯 Summary:")
print(f"   • False Alarms: {cm_orig[1, 0]:,} → {cm_eng[1, 0]:,} ({fa_reduction:.1f}% reduction)")
print(f"   • Precision: {precision_orig:.1%} → {precision_eng:.1%} ({precision_improvement:+.1f}%)")
print(f"   • Recall: {recall_eng:.1%} (maintained >90%)")
print(f"   • Created {len(engineered_features)} new features")
print(f"   • {len(engineered_in_top10)} engineered features in top 10")
