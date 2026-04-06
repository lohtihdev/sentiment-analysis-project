"""
main.py
-------
One-shot entry point that:
  1. Generates the 10,000-tweet dataset
  2. Preprocesses text (tokenise → stopword removal → lemmatise)
  3. Trains TF-IDF + Logistic Regression pipeline
  4. Evaluates and prints full metrics
  5. Generates all 8 visualisation plots
  6. Demonstrates live inference on sample tweets
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

# ── Force UTF-8 output on Windows so emoji / Unicode don't crash ──────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from dataset_generator import generate_dataset
from preprocessor       import preprocess
from train_model        import preprocess_dataframe, build_pipeline, train_and_evaluate, REPORT_PATH
from visualize          import (
    plot_distribution, plot_text_lengths, plot_confusion_matrix,
    plot_metrics_dashboard, plot_top_features, plot_wordclouds,
    plot_sentiment_trends, plot_engagement,
)

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
os.makedirs("outputs", exist_ok=True)
DATA_PATH  = "tweets_dataset.csv"
RANDOM_STATE = 42

BANNER = """
================================================================
  SENTIMENT ANALYSIS ON SOCIAL MEDIA TEXT (2024)
  TF-IDF + Logistic Regression  |  10,000 Tweets
================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
def step(n, title):
    print(f"\n{'--'*30}")
    print(f"  STEP {n}: {title}")
    print(f"{'--'*30}")


def main():
    print(BANNER)
    t_total = time.time()

    # -- Step 1: Dataset
    step(1, "Generating / Loading Dataset")
    if os.path.exists(DATA_PATH):
        print(f"  [INFO] Found existing file: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset(10_000)
        df.to_csv(DATA_PATH, index=False)
    print(f"  Rows : {len(df):,}")
    print(f"  Class Distribution:\n{df['sentiment'].value_counts().to_string()}")

    # -- Step 2: Preprocessing
    step(2, "Preprocessing (tokenise -> stopwords -> lemmatise)")
    df = preprocess_dataframe(df)
    sample = df.sample(3, random_state=42)[["text", "processed_text", "sentiment"]]
    print("\n  Sample before / after preprocessing:")
    for _, row in sample.iterrows():
        def safe(s): return str(s).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"\n  [{row['sentiment']}]")
        print(f"  BEFORE : {safe(row['text'][:90])}")
        print(f"  AFTER  : {safe(row['processed_text'][:90])}")

    # -- Step 3: Train
    step(3, "Training TF-IDF + Logistic Regression Pipeline")
    results = train_and_evaluate(df)

    # -- Step 4: Detailed Metrics
    step(4, "Evaluation Metrics")
    report_dict = classification_report(
        results["y_test"], results["y_pred"], output_dict=True
    )
    print(f"  Overall Accuracy : {results['accuracy']*100:.2f}%")
    print(f"  Cross-Val (5-fold): {results['cv_scores'].mean()*100:.2f}%"
          f" ± {results['cv_scores'].std()*100:.2f}%")

    # -- Step 5: Visualisations
    step(5, "Generating Visualisations")
    plot_distribution(df)
    plot_text_lengths(df)
    plot_confusion_matrix(results["y_test"], results["y_pred"], results["classes"])
    plot_metrics_dashboard(report_dict, results["cv_scores"], results["accuracy"])
    plot_top_features(results["pipeline"])
    plot_wordclouds(df)
    plot_sentiment_trends(df)
    plot_engagement(df)

    # -- Step 6: Live Demo
    step(6, "Live Inference on Custom Tweets")
    demo_tweets = [
        "Absolutely love the new update!! This is the best thing ever 😍 #amazing",
        "Worst product I've ever bought. Total waste of money. Refund requested immediately.",
        "Just received my order today. Seems okay so far, will update after using it.",
        "The customer support team was incredibly helpful and patient. 5 stars!",
        "App keeps crashing every 5 minutes. Completely unusable. Terrible experience.",
        "Meh, it does what it says I guess. Nothing special about it really.",
    ]

    pipeline = results["pipeline"]
    preds = pipeline.predict([preprocess(t) for t in demo_tweets])
    probas = pipeline.predict_proba([preprocess(t) for t in demo_tweets])

    def safe(s): return str(s).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
    print(f"\n  {'Tweet':<62} {'Predicted':<12} {'Confidence'}")
    print(f"  {'='*95}")
    for tweet, pred, prob in zip(demo_tweets, preds, probas):
        idx  = pipeline.classes_.tolist().index(pred)
        conf = prob[idx]
        icon = "[+]" if pred=="Positive" else ("[-]" if pred=="Negative" else "[~]")
        print(f"  {icon} {safe(tweet[:58]):<60} {pred:<12} {conf*100:.1f}%")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time()-t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  All outputs saved to: outputs/")
    print(f"      - sentiment_model.pkl")
    print(f"      - classification_report.txt")
    print(f"      - 1_sentiment_distribution.png")
    print(f"      - 2_text_length_analysis.png")
    print(f"      - 3_confusion_matrix.png")
    print(f"      - 4_model_performance.png")
    print(f"      - 5_top_tfidf_features.png")
    print(f"      - 6_word_clouds.png")
    print(f"      - 7_sentiment_trends.png")
    print(f"      - 8_engagement_by_sentiment.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
