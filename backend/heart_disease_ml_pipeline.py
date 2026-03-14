# Heart disease prediction — ML training pipeline
# Dataset: UCI Cleveland Heart Disease (303 patients, 13 features)
# Run this first before starting the API

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
from lime import lime_tabular

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# UCI dataset column names
FEATURE_NAMES = [
    "age",       # years
    "sex",       # 1=Male, 0=Female
    "cp",        # chest pain type (0-3)
    "trestbps",  # resting blood pressure mm Hg
    "chol",      # serum cholesterol mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl
    "restecg",   # resting ECG (0-2)
    "thalach",   # max heart rate achieved
    "exang",     # exercise induced angina
    "oldpeak",   # ST depression from exercise
    "slope",     # slope of peak ST segment
    "ca",        # major vessels colored by fluoroscopy (0-3)
    "thal",      # thalassemia type
]

FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120",
    "restecg": "Resting ECG Results",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Exercise)",
    "slope": "Slope of Peak ST Segment",
    "ca": "Major Vessels (Fluoroscopy)",
    "thal": "Thalassemia Type",
}


def load_data(filepath: str = None) -> pd.DataFrame:
    if filepath:
        df = pd.read_csv(filepath)
    else:
        # pull straight from UCI if no local file provided
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, header=None, names=FEATURE_NAMES + ["target"])

    # UCI uses '?' for missing values
    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)

    # values 1-4 all mean disease present, so binarize
    df["target"] = (df["target"] > 0).astype(int)

    print(f"Dataset loaded: {df.shape[0]} patients, {df.shape[1]-1} features")
    print(f"Disease prevalence: {df['target'].mean():.1%}")
    return df


def preprocess_data(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    # median imputation works better than mean for clinical data with outliers
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_NAMES)

    # derived features that tend to help with heart disease prediction
    X_imputed["age_thalach_ratio"] = X_imputed["age"] / (X_imputed["thalach"] + 1)
    X_imputed["bp_age_interaction"] = X_imputed["trestbps"] * X_imputed["age"]
    X_imputed["chol_risk"] = (X_imputed["chol"] > 200).astype(int)
    X_imputed["severe_chest_pain"] = (X_imputed["cp"] == 3).astype(int)

    enhanced_features = FEATURE_NAMES + [
        "age_thalach_ratio", "bp_age_interaction", "chol_risk", "severe_chest_pain"
    ]

    # stratify so both splits have the same class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split: {len(X_train)} train / {len(X_test)} test")

    # SMOTE only on training data — never touch the test set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_resampled)} training samples (balanced)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train_resampled, y_test,
        scaler, imputer,
        enhanced_features
    )


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42
        ),
        "SVM": SVC(
            C=10, kernel="rbf", gamma="scale",
            probability=True, class_weight="balanced", random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            learning_rate_init=0.001, max_iter=500,
            early_stopping=True, random_state=42
        ),
    }
    return models


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    return metrics, y_pred, y_prob


def train_all_models(models, X_train, X_test, y_train, y_test, feature_names):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_experiment("Heart_Disease_Prediction")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
            print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

            model.fit(X_train, y_train)
            metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)

            mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
            mlflow.log_metrics({
                **metrics,
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
            })
            mlflow.sklearn.log_model(model, name.replace(" ", "_"))

            results[name] = {
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "cv_scores": cv_scores,
            }

    return results


def build_ensemble(models, X_train, y_train, X_test, y_test):
    print("\nBuilding Soft Voting Ensemble...")

    # soft voting uses probabilities — more reliable than hard majority vote
    estimators = [(name, model["model"]) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    ensemble.fit(X_train, y_train)

    metrics, y_pred, y_prob = evaluate_model(ensemble, X_test, y_test, "Ensemble (Soft Voting)")

    with mlflow.start_run(run_name="Ensemble_Voting"):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(ensemble, "Ensemble")

    return {"model": ensemble, "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}


def explain_with_shap(model, X_train, X_test, feature_names, save_dir="."):
    print("\nGenerating SHAP explanations...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is a list for binary classification — index 1 is the positive class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      plot_type="bar", show=False, color="#E85D24")
    plt.title("SHAP Feature Importance — Heart Disease Prediction", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP summary plot saved")

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Distribution", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP beeswarm plot saved")

    return explainer, shap_values


def explain_with_lime(model, X_train, X_test, feature_names, patient_idx=0):
    print(f"\nLIME explanation for patient #{patient_idx}...")

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["No Disease", "Heart Disease"],
        mode="classification",
        random_state=42,
    )

    explanation = lime_explainer.explain_instance(
        data_row=X_test[patient_idx],
        predict_fn=model.predict_proba,
        num_features=10,
    )

    return explanation


class HeartDiseasePredictor:
    """Convenience class for loading and running predictions from saved artifacts."""

    def __init__(self, model_path, scaler_path, imputer_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        self.threshold = 0.5

    def predict(self, patient_data: dict) -> dict:
        df = pd.DataFrame([patient_data])
        imputed = self.imputer.transform(df)
        scaled = self.scaler.transform(imputed)

        probability = self.model.predict_proba(scaled)[0][1]
        prediction = int(probability >= self.threshold)

        risk_level = (
            "HIGH" if probability >= 0.7
            else "MODERATE" if probability >= 0.4
            else "LOW"
        )

        return {
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "risk_level": risk_level,
            "confidence": round(float(max(probability, 1 - probability)), 4),
            "threshold_used": self.threshold,
        }


def plot_model_comparison(results, ensemble_result, save_dir="."):
    all_results = {**results, "Ensemble": ensemble_result}
    model_names = list(all_results.keys())
    metrics_list = ["accuracy", "f1_score", "roc_auc", "precision", "recall"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Heart Disease Prediction — Model Comparison", fontsize=16, fontweight="bold")
    colors = ["#3B8BD4", "#E85D24", "#1D9E75", "#7F77DD", "#D4537E", "#BA7517"]

    ax = axes[0, 0]
    accs = [all_results[m]["metrics"]["accuracy"] for m in model_names]
    bars = ax.bar(model_names, accs, color=colors[:len(model_names)], alpha=0.85)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=30)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    aucs = [all_results[m]["metrics"]["roc_auc"] for m in model_names]
    bars = ax.bar(model_names, aucs, color=colors[:len(model_names)], alpha=0.85)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("ROC AUC Comparison")
    ax.set_ylabel("AUC Score")
    ax.tick_params(axis="x", rotation=30)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for i, (name, res) in enumerate(all_results.items()):
        fpr, tpr, _ = roc_curve(y_test_global, res["y_prob"])
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    x = np.arange(len(metrics_list))
    width = 0.12
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res["metrics"][m] for m in metrics_list]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * (len(all_results) / 2))
    ax.set_xticklabels(metrics_list, rotation=20)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("All Metrics Comparison")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Model comparison dashboard saved")


def main():
    print("=" * 55)
    print("   HEART DISEASE PREDICTION SYSTEM — ML Pipeline")
    print("=" * 55)

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, imputer, features = preprocess_data(df)

    global y_test_global
    y_test_global = y_test

    models = build_models()
    results = train_all_models(models, X_train, X_test, y_train, y_test, features)
    ensemble_result = build_ensemble(results, X_train, y_train, X_test, y_test)

    # use Random Forest for SHAP since it's tree-based and fast to explain
    best_rf = results["Random Forest"]["model"]
    explainer, shap_values = explain_with_shap(best_rf, X_train, X_test, features)
    lime_exp = explain_with_lime(best_rf, X_train, X_test, features, patient_idx=0)

    plot_model_comparison(results, ensemble_result)

    joblib.dump(ensemble_result["model"], "models/ensemble_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(imputer, "models/imputer.pkl")

    print("\n" + "=" * 55)
    print("Pipeline complete! Artifacts saved to /models/")
    print(f"Best Ensemble AUC: {ensemble_result['metrics']['roc_auc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
