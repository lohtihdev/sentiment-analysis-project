"""
train_model.py
--------------
TF-IDF + Logistic Regression sentiment classification pipeline.
Trains on 80% of the 10,000-tweet dataset and evaluates on the remaining 20%.
Saves the trained pipeline and evaluation artefacts.
"""

import os, time, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

from preprocessor import preprocess
from dataset_generator import generate_dataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "tweets_dataset.csv"
MODEL_PATH = "outputs/sentiment_model.pkl"
REPORT_PATH = "outputs/classification_report.txt"
os.makedirs("outputs", exist_ok=True)

RANDOM_STATE = 42


# ── 1. Load / Generate dataset ────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("[INFO] Generating synthetic dataset ...")
    df = generate_dataset(10_000)
    df.to_csv(DATA_PATH, index=False)
    print(f"[SAVED] Dataset saved -> {DATA_PATH}")
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Preprocessing tweets ...", end=" ", flush=True)
    t0 = time.time()
    df = df.copy()
    df["processed_text"] = df["text"].map(preprocess)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")
    return df


# ── 3. Build pipeline ─────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),        # unigrams + bigrams
        sublinear_tf=True,         # log-scale TF
        min_df=2,
        max_df=0.95,
    )
    lr = LogisticRegression(
        max_iter=1_000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("clf", lr)])


# ── 4. Train & evaluate ───────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    X = df["processed_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")

    pipeline = build_pipeline()
    print("[INFO] Training TF-IDF + Logistic Regression ...", end=" ", flush=True)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"done in {time.time()-t0:.1f}s")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"{'='*50}")

    report = classification_report(y_test, y_pred)
    print(report)

    # 5-fold cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"Cross-Val Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[SAVED] Model saved -> {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
        f.write(f"\nCross-Val: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")

    # Return everything needed for visualisation
    le = LabelEncoder().fit(y)
    return {
        "pipeline": pipeline,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "accuracy": acc,
        "cv_scores": cv_scores,
        "classes": pipeline.classes_.tolist(),
        "df_full": df,
    }


if __name__ == "__main__":
    df = load_data()
    df = preprocess_dataframe(df)
    results = train_and_evaluate(df)
    print("\n[DONE] Training complete. Run visualize.py to generate plots.")
