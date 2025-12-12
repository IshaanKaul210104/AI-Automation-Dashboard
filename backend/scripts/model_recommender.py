# backend/scripts/model_recommender.py
import os
import io
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, silhouette_score
from scipy.stats import skew

def calculate_meta_features(X: pd.DataFrame, y: pd.Series | None, task: str = "regression"):
    meta = {}
    # avg absolute skew across numeric features
    meta["avg_skewness"] = float(np.mean(np.abs(skew(X.values, axis=0))))
    if task in ("regression", "classification") and y is not None:
        cors = []
        for col in X.columns:
            try:
                c = np.corrcoef(X[col].values, y.values)[0, 1]
            except Exception:
                c = 0.0
            cors.append(0.0 if np.isnan(c) else float(np.abs(c)))
        meta["avg_correlation_with_target"] = float(np.nanmean(cors)) if len(cors) else 0.0
    else:
        meta["avg_correlation_with_target"] = 0.0

    try:
        # multicollinearity score: trace of pseudo-inverse of correlation matrix
        corr = np.corrcoef(X.values.T)
        pinv = np.linalg.pinv(corr)
        meta["multicollinearity_score"] = float(np.trace(pinv))
    except Exception:
        meta["multicollinearity_score"] = 0.0

    return meta

def recommend_model(meta: dict, task: str = "regression"):
    task = task.lower()
    if task == "regression":
        if meta["avg_skewness"] > 2 or meta["multicollinearity_score"] > 50:
            return RandomForestRegressor(), "RandomForestRegressor", "High skewness or high multicollinearity detected — using Random Forest."
        if meta["avg_correlation_with_target"] > 0.5:
            return Ridge(), "Ridge", "Strong linear correlation with target — using Ridge regression."
        return LinearRegression(), "LinearRegression", "Default: moderate skew/correlation — using Linear Regression."

    if task == "classification":
        if meta["multicollinearity_score"] > 50:
            return RandomForestClassifier(), "RandomForestClassifier", "High multicollinearity — using Random Forest classifier."
        if meta["avg_correlation_with_target"] > 0.4:
            return LogisticRegression(max_iter=1000), "LogisticRegression", "High correlation with target — using Logistic Regression."
        return RandomForestClassifier(), "RandomForestClassifier", "Default fallback — using Random Forest classifier."

    # clustering
    if meta["avg_skewness"] > 2:
        return DBSCAN(), "DBSCAN", "High skewness — using DBSCAN."
    return KMeans(n_clusters=3, random_state=42), "KMeans", "Default clustering: using KMeans (3 clusters)."

def evaluate_model(model, X, y=None, task="regression"):
    res = {}
    if task == "regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        res["r2"] = float(r2_score(y_test, preds))
        res["mse"] = float(mean_squared_error(y_test, preds))
    elif task == "classification":
        # try stratify, fallback gracefully
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        res["accuracy"] = float(accuracy_score(y_test, preds))
    else:  # clustering
        model.fit(X)

        if hasattr(model, "labels_"):
            labels = model.labels_
        else:
            labels = model.predict(X)

        # unique cluster count
        unique_labels = np.unique(labels)

        # Invalid for silhouette
        if len(unique_labels) < 2:
            res["silhouette"] = None
        else:
            try:
                score = silhouette_score(X, labels)
                res["silhouette"] = None if (np.isnan(score) or np.isinf(score)) else float(score)
            except Exception:
                res["silhouette"] = None
    return res

def clean_json(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def run_from_fileobj(fileobj: io.BytesIO, task: str, target_col: str | None = None):
    # Read CSV into pandas
    df = pd.read_csv(fileobj)
    if df.empty:
        return {"status": "failed", "error": "Uploaded CSV is empty."}

    # If task requires target, ensure provided
    if task in ("regression", "classification"):
        if not target_col:
            return {"status": "failed", "error": "Target column required for regression/classification."}
        if target_col not in df.columns:
            return {"status": "failed", "error": f"Target column '{target_col}' not found in CSV."}

    # Keep numeric columns only for features
    # For regression/classification, drop target from X
    if task in ("regression", "classification"):
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    # Detect categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if len(categorical_cols) == 0 and len(numeric_cols) == 0:
        return {"status": "failed", "error": "No usable columns found. Dataset must contain numeric or categorical columns."}

    # Prepare ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Drop NA rows
    combined = X.copy()
    if y is not None:
        combined[target_col if task in ("regression","classification") else "_y"] = y
        combined = combined.dropna()
        if combined.shape[0] == 0:
            return {"status": "failed", "error": "All rows contain missing values after dropna."}
        # recover X/y aligned
        if task in ("regression", "classification"):
            y = combined[target_col]
            X_numeric = combined.drop(columns=[target_col])
    else:
        combined = X.dropna()
        if combined.shape[0] == 0:
            return {"status": "failed", "error": "All rows contain missing values after dropna."}
        X = combined

    # scale
    scaler = StandardScaler()
    # Fit + transform
    X = combined.drop(columns=[target_col]) if y is not None else combined
    X_processed = preprocessor.fit_transform(X)

    # For meta-features, keep numeric only (your calculator expects numeric input)
    X_numeric_only = combined[numeric_cols].copy()

    # If no numeric cols exist, create placeholder  (prevents skew/corr errors)
    if X_numeric_only.shape[1] == 0:
        X_numeric_only = pd.DataFrame({"dummy": np.zeros(len(X))})
    if len(numeric_cols) == 0:
        X_numeric_only = pd.DataFrame({"dummy": np.zeros(len(combined))})

    # Scale numeric features before meta calculations
    scaled_num = StandardScaler().fit_transform(X_numeric_only)
    X_numeric_scaled = pd.DataFrame(scaled_num, columns=X_numeric_only.columns)

    
    # meta features
    meta = calculate_meta_features(X_numeric_scaled, y if task in ("regression","classification") else None, task)

    # recommendation
    model, model_name, reason = recommend_model(meta, task)

    # evaluate (train+test)
    metrics = evaluate_model(model, X_processed, y if task in ("regression","classification") else None, task)

    return clean_json({
        "status": "success",
        "recommended_model": model_name,
        "reason": reason,
        "meta_features": meta,
        "metrics": metrics,
    })