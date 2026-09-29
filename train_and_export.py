"""
Case Study 46: Handwritten Digit Recognition Using PCA and Classification
Dataset: Scikit-Learn load_digits() (8x8 images, 64 features)
Algorithms: Logistic Regression, KNN, Random Forest, Gradient Boosting
Comparison: With and Without PCA (retaining >= 90% variance)
"""

import time
import json
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def main():
    print("=" * 70)
    print("STEP 1: LOADING & EXPLORING DATASET")
    print("=" * 70)
    digits = load_digits()
    X = digits.data          # (1797, 64)
    y = digits.target        # (1797,)
    print(f"Total Samples: {X.shape[0]}")
    print(f"Original Feature Count (Pixels): {X.shape[1]} (8x8 grid)")
    print(f"Classes: {np.unique(y)} (Digits 0 through 9)")
    print(f"Pixel Intensity Range: min={X.min():.1f}, max={X.max():.1f}")
    
    # Train / Test split (80% train, 20% test stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

    # Standardize data (crucial before PCA and distance-based models like KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("STEP 2: PCA DIMENSIONALITY REDUCTION (>= 90% VARIANCE)")
    print("=" * 70)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Find minimum components for 90% variance
    n_components_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
    variance_at_90 = cumulative_variance[n_components_90 - 1]
    print(f"Components needed for >= 90% variance: {n_components_90} out of 64 features")
    print(f"Exact cumulative variance retained: {variance_at_90 * 100:.2f}%")
    print(f"Feature reduction: {((64 - n_components_90) / 64) * 100:.1f}% reduction in input size!")

    # Fit PCA with 90% variance
    pca = PCA(n_components=n_components_90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Classifiers to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = []
    trained_models = {}

    print("\n" + "=" * 70)
    print("STEP 3: 8-WAY COMPARATIVE EXPERIMENT")
    print("=" * 70)
    print(f"{'Algorithm':<22} | {'Condition':<10} | {'Acc (%)':<8} | {'F1-Macro':<9} | {'F1-Weight':<9} | {'Train Time (s)':<14}")
    print("-" * 80)

    for name, model_cls in models.items():
        # Condition 1: Without PCA (Raw Scaled 64 features)
        m_raw = model_cls.__class__(**model_cls.get_params())
        t0 = time.time()
        m_raw.fit(X_train_scaled, y_train)
        time_raw = time.time() - t0
        
        preds_raw = m_raw.predict(X_test_scaled)
        acc_raw = accuracy_score(y_test, preds_raw)
        p_raw = precision_score(y_test, preds_raw, average='macro', zero_division=0)
        r_raw = recall_score(y_test, preds_raw, average='macro', zero_division=0)
        f1_raw_macro = f1_score(y_test, preds_raw, average='macro', zero_division=0)
        f1_raw_weighted = f1_score(y_test, preds_raw, average='weighted', zero_division=0)
        cm_raw = confusion_matrix(y_test, preds_raw)

        results.append({
            "model": name,
            "condition": "Without PCA",
            "features": 64,
            "accuracy": acc_raw,
            "precision": p_raw,
            "recall": r_raw,
            "f1_macro": f1_raw_macro,
            "f1_weighted": f1_raw_weighted,
            "train_time": time_raw,
            "confusion_matrix": cm_raw
        })
        trained_models[f"{name}_raw"] = m_raw

        print(f"{name:<22} | {'Raw (64)':<10} | {acc_raw*100:>7.2f}% | {f1_raw_macro:>9.4f} | {f1_raw_weighted:>9.4f} | {time_raw:>13.4f}s")

        # Condition 2: With PCA (n_components_90)
        m_pca = model_cls.__class__(**model_cls.get_params())
        t0 = time.time()
        m_pca.fit(X_train_pca, y_train)
        time_pca = time.time() - t0

        preds_pca = m_pca.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, preds_pca)
        p_pca = precision_score(y_test, preds_pca, average='macro', zero_division=0)
        r_pca = recall_score(y_test, preds_pca, average='macro', zero_division=0)
        f1_pca_macro = f1_score(y_test, preds_pca, average='macro', zero_division=0)
        f1_pca_weighted = f1_score(y_test, preds_pca, average='weighted', zero_division=0)
        cm_pca = confusion_matrix(y_test, preds_pca)

        results.append({
            "model": name,
            "condition": "With PCA",
            "features": n_components_90,
            "accuracy": acc_pca,
            "precision": p_pca,
            "recall": r_pca,
            "f1_macro": f1_pca_macro,
            "f1_weighted": f1_pca_weighted,
            "train_time": time_pca,
            "confusion_matrix": cm_pca
        })
        trained_models[f"{name}_pca"] = m_pca

        time_saved = ((time_raw - time_pca) / time_raw) * 100 if time_raw > 0 else 0
        acc_diff = (acc_pca - acc_raw) * 100
        print(f"{name:<22} | {f'PCA ({n_components_90})':<10} | {acc_pca*100:>7.2f}% | {f1_pca_macro:>9.4f} | {f1_pca_weighted:>9.4f} | {time_pca:>13.4f}s (Speedup: {time_saved:+.1f}%, Acc diff: {acc_diff:+.2f}%)")
        print("-" * 80)

    # Save summary stats to json
    summary_data = {
        "n_components_90": n_components_90,
        "variance_retained": float(variance_at_90),
        "results": [
            {
                "model": r["model"],
                "condition": r["condition"],
                "features": r["features"],
                "accuracy": float(r["accuracy"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1_macro": float(r["f1_macro"]),
                "f1_weighted": float(r["f1_weighted"]),
                "train_time": float(r["train_time"])
            }
            for r in results
        ]
    }
    with open("experiment_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save sklearn models
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(pca, "pca_transformer.joblib")
    joblib.dump(trained_models["Logistic Regression_pca"], "logistic_regression_pca.joblib")
    joblib.dump(trained_models["KNN_pca"], "knn_pca.joblib")
    joblib.dump(trained_models["Random Forest_pca"], "random_forest_pca.joblib")

    # Export a fully self-contained model.json for in-browser JavaScript inference on GitHub Pages!
    # For Logistic Regression: W is (10, n_components), b is (10,)
    lr_model = trained_models["Logistic Regression_pca"]
    
    # Also export KNN samples for in-browser KNN if desired (X_train_pca and y_train)
    export_payload = {
        "dataset": "load_digits",
        "n_features_raw": 64,
        "n_components": n_components_90,
        "variance_retained": float(variance_at_90),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "pca": {
            "components": pca.components_.tolist(),   # (n_components, 64)
            "mean": pca.mean_.tolist()                # (64,)
        },
        "logistic_regression": {
            "weights": lr_model.coef_.tolist(),       # (10, n_components)
            "intercept": lr_model.intercept_.tolist() # (10,)
        },
        "knn_reference": {
            # Subsample 400 points stratified for superfast JS distance calculation
            "vectors": X_train_pca[:400].tolist(),
            "labels": y_train[:400].tolist()
        }
    }
    with open("model.json", "w") as f:
        json.dump(export_payload, f)
    print("\n Exported model.json for standalone in-browser inference!")
    print(" Saved joblib artifacts and experiment_results.json successfully.")

if __name__ == "__main__":
    main()
