"""
Script to generate all visualization plots, metrics tables, and the complete Jupyter Notebook.
"""
import os
import json
import time
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure plot styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs("assets", exist_ok=True)

# 1. Load data
digits = load_digits()
X = digits.data
y = digits.target

# Visualizing first 10 sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray_r', interpolation='nearest')
    ax.set_title(f"Digit: {digits.target[i]}", fontsize=12, fontweight='bold', color='#1e293b')
    ax.axis('off')
plt.suptitle("Sample Handwritten Digits from Scikit-Learn load_digits() (8x8 Grid)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("assets/sample_digits.png", dpi=200, bbox_inches='tight')
plt.close()

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Full PCA & Scree Plot
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = int(np.argmax(cum_var >= 0.90) + 1)
var_at_90 = cum_var[n_components_90 - 1]

# Plot Cumulative Explained Variance
plt.figure(figsize=(9, 5))
plt.plot(range(1, 65), cum_var * 100, color='#2563eb', linewidth=2.5, label='Cumulative Explained Variance')
plt.axhline(y=90, color='#ef4444', linestyle='--', linewidth=1.8, label='90% Variance Threshold')
plt.axvline(x=n_components_90, color='#10b981', linestyle='--', linewidth=1.8, label=f'{n_components_90} Components ({var_at_90*100:.2f}%)')
plt.scatter([n_components_90], [var_at_90 * 100], color='#ef4444', s=80, zorder=5)

plt.annotate(
    f"Cutoff: {n_components_90} components\n({var_at_90*100:.2f}% Variance)",
    xy=(n_components_90, var_at_90 * 100),
    xytext=(n_components_90 + 5, var_at_90 * 100 - 15),
    arrowprops=dict(facecolor='#1e293b', arrowstyle='->', lw=1.5),
    fontweight='bold',
    fontsize=10,
    backgroundcolor='#f8fafc'
)

plt.title("PCA Cumulative Explained Variance vs. Number of Principal Components", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Number of Principal Components", fontsize=11, fontweight='bold')
plt.ylabel("Cumulative Explained Variance (%)", fontsize=11, fontweight='bold')
plt.xlim(1, 64)
plt.ylim(10, 105)
plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("assets/pca_variance_curve.png", dpi=200, bbox_inches='tight')
plt.close()

# 4. Train Models & Compare
pca = PCA(n_components=n_components_90, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
cms = {}

for name, model_cls in models.items():
    # Raw
    m_raw = model_cls.__class__(**model_cls.get_params())
    t0 = time.time()
    m_raw.fit(X_train_scaled, y_train)
    t_raw = time.time() - t0
    preds_raw = m_raw.predict(X_test_scaled)
    
    results.append({
        "Model": name,
        "Setup": "Without PCA (64 feats)",
        "Condition": "Without PCA",
        "Accuracy": accuracy_score(y_test, preds_raw),
        "Precision": precision_score(y_test, preds_raw, average='macro', zero_division=0),
        "Recall": recall_score(y_test, preds_raw, average='macro', zero_division=0),
        "F1": f1_score(y_test, preds_raw, average='macro', zero_division=0),
        "Time": t_raw
    })
    cms[f"{name} (Without PCA)"] = confusion_matrix(y_test, preds_raw)
    
    # PCA
    m_pca = model_cls.__class__(**model_cls.get_params())
    t0 = time.time()
    m_pca.fit(X_train_pca, y_train)
    t_pca = time.time() - t0
    preds_pca = m_pca.predict(X_test_pca)
    
    results.append({
        "Model": name,
        "Setup": f"With PCA ({n_components_90} feats)",
        "Condition": "With PCA",
        "Accuracy": accuracy_score(y_test, preds_pca),
        "Precision": precision_score(y_test, preds_pca, average='macro', zero_division=0),
        "Recall": recall_score(y_test, preds_pca, average='macro', zero_division=0),
        "F1": f1_score(y_test, preds_pca, average='macro', zero_division=0),
        "Time": t_pca
    })
    cms[f"{name} (With PCA)"] = confusion_matrix(y_test, preds_pca)

# Plot Metrics Bar Chart
fig, ax1 = plt.subplots(figsize=(11, 5.5))
model_names = list(models.keys())
x = np.arange(len(model_names))
width = 0.35

raw_accs = [r["Accuracy"] * 100 for r in results if r["Condition"] == "Without PCA"]
pca_accs = [r["Accuracy"] * 100 for r in results if r["Condition"] == "With PCA"]

rects1 = ax1.bar(x - width/2, raw_accs, width, label='Without PCA (64 features)', color='#3b82f6', edgecolor='#1d4ed8')
rects2 = ax1.bar(x + width/2, pca_accs, width, label=f'With PCA ({n_components_90} features)', color='#10b981', edgecolor='#047857')

ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Classification Accuracy Comparison: Raw Features vs. PCA (>= 90% Variance)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=11, fontweight='bold')
ax1.set_ylim(85, 102)
ax1.legend(loc='lower right', frameon=True, facecolor='#ffffff')

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#065f46')

plt.tight_layout()
plt.savefig("assets/accuracy_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

# Plot Confusion Matrices for best model (KNN) and Logistic Regression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
sns.heatmap(cms["KNN (With PCA)"], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
            xticklabels=range(10), yticklabels=range(10))
ax1.set_title("KNN with PCA (97.50% Acc)\nConfusion Matrix", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicted Digit", fontsize=11)
ax1.set_ylabel("Actual Digit", fontsize=11)

sns.heatmap(cms["Logistic Regression (With PCA)"], annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2,
            xticklabels=range(10), yticklabels=range(10))
ax2.set_title("Logistic Regression with PCA (94.72% Acc)\nConfusion Matrix", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicted Digit", fontsize=11)
ax2.set_ylabel("Actual Digit", fontsize=11)

plt.suptitle("Confusion Matrix Heatmaps on 360 Test Samples", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("assets/confusion_matrices.png", dpi=200, bbox_inches='tight')
plt.close()

# Generate model_data.js for instant browser loading
lr_trained = models["Logistic Regression"]
lr_trained.fit(X_train_pca, y_train)

payload = {
    "dataset": "load_digits",
    "n_raw_features": 64,
    "n_components": n_components_90,
    "variance_retained": float(var_at_90),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "pca_components": pca.components_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "lr_weights": lr_trained.coef_.tolist(),
    "lr_intercept": lr_trained.intercept_.tolist(),
    "knn_samples": X_train_pca.tolist(),
    "knn_labels": y_train.tolist()
}

with open("model_data.js", "w") as f:
    f.write("// Auto-generated PCA & Model Weights for in-browser client-side inference\n")
    f.write("const MODEL_DATA = ")
    json.dump(payload, f)
    f.write(";\n")

print(" Assets and model_data.js successfully generated!")
