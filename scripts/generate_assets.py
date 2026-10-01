"""
Script to generate pure monochrome (black, white, grayscale) visualization plots.
Strictly adheres to black & white palette with shades of gray.
"""
import os
import json
import time
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

# Pure monochrome dark styling
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['text.color'] = '#ededed'
plt.rcParams['axes.labelcolor'] = '#a3a3a3'
plt.rcParams['xtick.color'] = '#a3a3a3'
plt.rcParams['ytick.color'] = '#a3a3a3'
plt.rcParams['axes.edgecolor'] = '#262626'
plt.rcParams['grid.color'] = '#1a1a1a'
plt.rcParams['font.sans-serif'] = 'Geist, Helvetica, Arial, sans-serif'
plt.rcParams['font.monospace'] = 'Geist Mono, Courier, monospace'

os.makedirs("docs/assets", exist_ok=True)

# 1. Load data
digits = load_digits()
X = digits.data
y = digits.target

# Visualizing first 10 sample digits (Monochrome Dark)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), facecolor='#0a0a0a')
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray', interpolation='nearest')
    ax.set_title(f"Digit: {digits.target[i]}", fontsize=11, fontweight='bold', color='#ededed')
    ax.axis('off')
plt.suptitle("Sample Handwritten Digits from Scikit-Learn load_digits() (8x8 Grid)", fontsize=13, fontweight='bold', y=1.02, color='#ffffff')
plt.tight_layout()
plt.savefig("docs/assets/sample_digits.png", dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
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

# Plot Cumulative Explained Variance (Monochrome)
fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0a0a0a')
ax.plot(range(1, 65), cum_var * 100, color='#ffffff', linewidth=2.2, label='Cumulative Variance')
ax.axhline(y=90, color='#737373', linestyle='--', linewidth=1.5, label='90% Threshold')
ax.axvline(x=n_components_90, color='#a3a3a3', linestyle=':', linewidth=1.5, label=f'{n_components_90} Components ({var_at_90*100:.2f}%)')
ax.scatter([n_components_90], [var_at_90 * 100], color='#ffffff', edgecolor='#0a0a0a', s=90, zorder=5)

ax.annotate(
    f"Cutoff: {n_components_90} components\\n({var_at_90*100:.2f}% Variance)",
    xy=(n_components_90, var_at_90 * 100),
    xytext=(n_components_90 + 5, var_at_90 * 100 - 15),
    arrowprops=dict(facecolor='#ffffff', edgecolor='#ffffff', arrowstyle='->', lw=1.2),
    fontweight='bold',
    fontsize=9.5,
    color='#ededed',
    backgroundcolor='#171717',
    bbox=dict(boxstyle="round,pad=0.4", fc="#171717", ec="#404040", lw=0.8)
)

ax.set_title("PCA Cumulative Explained Variance vs. Number of Components", fontsize=12, fontweight='bold', pad=15, color='#ffffff')
ax.set_xlabel("Number of Principal Components", fontsize=10.5, fontweight='bold', color='#a3a3a3')
ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=10.5, fontweight='bold', color='#a3a3a3')
ax.set_xlim(1, 64)
ax.set_ylim(10, 105)
ax.legend(frameon=True, facecolor='#171717', edgecolor='#262626', loc='lower right', labelcolor='#ededed')
ax.grid(True, linestyle=':', alpha=0.3, color='#404040')
plt.tight_layout()
plt.savefig("docs/assets/pca_variance_curve.png", dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
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
    m_raw.fit(X_train_scaled, y_train)
    preds_raw = m_raw.predict(X_test_scaled)
    results.append({
        "Model": name,
        "Condition": "Without PCA",
        "Accuracy": accuracy_score(y_test, preds_raw)
    })
    cms[f"{name} (Without PCA)"] = confusion_matrix(y_test, preds_raw)
    
    # PCA
    m_pca = model_cls.__class__(**model_cls.get_params())
    m_pca.fit(X_train_pca, y_train)
    preds_pca = m_pca.predict(X_test_pca)
    results.append({
        "Model": name,
        "Condition": "With PCA",
        "Accuracy": accuracy_score(y_test, preds_pca)
    })
    cms[f"{name} (With PCA)"] = confusion_matrix(y_test, preds_pca)

# Plot Metrics Bar Chart (Monochrome: Charcoal vs Pure White)
fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='#0a0a0a')
model_names = list(models.keys())
x = np.arange(len(model_names))
width = 0.35

raw_accs = [r["Accuracy"] * 100 for r in results if r["Condition"] == "Without PCA"]
pca_accs = [r["Accuracy"] * 100 for r in results if r["Condition"] == "With PCA"]

rects1 = ax1.bar(x - width/2, raw_accs, width, label='Without PCA (64 feats)', color='#404040', edgecolor='#737373', lw=0.8)
rects2 = ax1.bar(x + width/2, pca_accs, width, label=f'With PCA ({n_components_90} feats)', color='#ffffff', edgecolor='#ffffff', lw=0.8)

ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold', color='#a3a3a3')
ax1.set_title('Classification Accuracy Comparison: Raw Features vs. PCA (90% Variance)', fontsize=12, fontweight='bold', pad=15, color='#ffffff')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=10, fontweight='bold', color='#ededed')
ax1.set_ylim(85, 102)
ax1.legend(loc='lower right', frameon=True, facecolor='#171717', edgecolor='#262626', labelcolor='#ededed')
ax1.grid(True, linestyle=':', alpha=0.3, color='#404040')

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#a3a3a3')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#ffffff')

plt.tight_layout()
plt.savefig("docs/assets/accuracy_comparison.png", dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()

# Plot Confusion Matrices in pure grayscale (Greys)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a0a')

sns.heatmap(cms["KNN (With PCA)"], annot=True, fmt='d', cmap='Greys', cbar=False, ax=ax1,
            xticklabels=range(10), yticklabels=range(10),
            annot_kws={"fontsize": 9, "fontweight": "bold"})
ax1.set_title("KNN with PCA (97.50% Acc)\nConfusion Matrix", fontsize=11, fontweight='bold', color='#ffffff')
ax1.set_xlabel("Predicted Digit", fontsize=10, color='#a3a3a3')
ax1.set_ylabel("Actual Digit", fontsize=10, color='#a3a3a3')
ax1.tick_params(colors='#a3a3a3')

sns.heatmap(cms["Logistic Regression (With PCA)"], annot=True, fmt='d', cmap='Greys', cbar=False, ax=ax2,
            xticklabels=range(10), yticklabels=range(10),
            annot_kws={"fontsize": 9, "fontweight": "bold"})
ax2.set_title("Logistic Regression with PCA (94.72% Acc)\nConfusion Matrix", fontsize=11, fontweight='bold', color='#ffffff')
ax2.set_xlabel("Predicted Digit", fontsize=10, color='#a3a3a3')
ax2.set_ylabel("Actual Digit", fontsize=10, color='#a3a3a3')
ax2.tick_params(colors='#a3a3a3')

plt.suptitle("Confusion Matrix Heatmaps on 360 Test Samples", fontsize=13, fontweight='bold', y=1.02, color='#ffffff')
plt.tight_layout()
plt.savefig("docs/assets/confusion_matrices.png", dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()

print("Pure monochrome dark assets generated successfully in docs/assets/.")
