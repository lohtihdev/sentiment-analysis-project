"""
visualize.py
------------
Generates publication-quality charts and saves them to outputs/.
Requires the dataset CSV and trained model (run train_model.py first).
"""

import os, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

from preprocessor import preprocess

# ── Style & Palette ───────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.15)
PALETTE   = {"Positive": "#2DD4BF", "Negative": "#F87171", "Neutral": "#A78BFA"}
BG_COLOR  = "#0F172A"   # dark slate
CARD_COLOR = "#1E293B"
TEXT_COLOR = "#F1F5F9"
ACCENT    = "#38BDF8"

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   CARD_COLOR,
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       "#1E293B",
    "legend.facecolor": CARD_COLOR,
    "legend.edgecolor": "#334155",
})

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
def save(name: str):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [SAVED] -> {path}")


# ── 1. Sentiment Distribution ─────────────────────────────────────────────────
def plot_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tweet Sentiment Distribution (10,000 tweets)", fontsize=16, color=TEXT_COLOR, y=1.02)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index],
                  edgecolor="#0F172A", linewidth=1.5, width=0.6)
    ax.set_title("Count per Sentiment Class", color=TEXT_COLOR)
    ax.set_xlabel("Sentiment", color=TEXT_COLOR)
    ax.set_ylabel("Number of Tweets", color=TEXT_COLOR)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f"{val:,}", ha="center", va="bottom", fontsize=11, color=TEXT_COLOR, fontweight="bold")

    # Donut chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[PALETTE[c] for c in counts.index],
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG_COLOR, linewidth=2),
        textprops={"color": TEXT_COLOR, "fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color(BG_COLOR)
        at.set_fontweight("bold")
    ax2.set_title("Class Proportions", color=TEXT_COLOR)

    plt.tight_layout()
    save("1_sentiment_distribution.png")


# ── 2. Text Length Analysis ───────────────────────────────────────────────────
def plot_text_lengths(df: pd.DataFrame):
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Text Length Distribution by Sentiment", fontsize=16, color=TEXT_COLOR, y=1.02)

    for ax, col, label in zip(axes, ["char_len", "word_len"], ["Character Count", "Word Count"]):
        for sent, grp in df.groupby("sentiment"):
            sns.kdeplot(grp[col], ax=ax, label=sent, color=PALETTE[sent],
                        fill=True, alpha=0.25, linewidth=2)
        ax.set_xlabel(label, color=TEXT_COLOR)
        ax.set_ylabel("Density", color=TEXT_COLOR)
        ax.set_title(label, color=TEXT_COLOR)
        ax.legend()

    plt.tight_layout()
    save("2_text_length_analysis.png")


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=16, color=TEXT_COLOR, y=1.02)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Raw Counts", "Normalised (Row %)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, cbar=True,
                    linewidths=0.5, linecolor=BG_COLOR,
                    annot_kws={"size": 12, "color": BG_COLOR, "fontweight": "bold"})
        ax.set_xlabel("Predicted", color=TEXT_COLOR)
        ax.set_ylabel("Actual", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)

    plt.tight_layout()
    save("3_confusion_matrix.png")


# ── 4. Model Performance Dashboard ───────────────────────────────────────────
def plot_metrics_dashboard(report_dict, cv_scores, accuracy):
    classes = [k for k in report_dict if k not in ("accuracy","macro avg","weighted avg")]

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Model Performance Dashboard", fontsize=16, color=TEXT_COLOR, y=1.02)

    # Grouped bar — per-class metrics
    ax = axes[0]
    colors = [ACCENT, "#F472B6", "#FB923C"]
    for i, metric in enumerate(metrics):
        vals = [report_dict[c][metric] for c in classes]
        bars = ax.bar(x + i*width, vals, width, label=metric.replace("-score",""),
                      color=colors[i], alpha=0.85, edgecolor=BG_COLOR)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c[:3] for c in classes])
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class Metrics", color=TEXT_COLOR)
    ax.set_ylabel("Score", color=TEXT_COLOR)
    ax.legend()

    # Cross-validation accuracy box
    ax2 = axes[1]
    ax2.set_facecolor(CARD_COLOR)
    parts = ax2.violinplot([cv_scores*100], positions=[1], showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(ACCENT)
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("#F472B6")
    parts["cmeans"].set_linewidth(2)

    ax2.scatter([1]*len(cv_scores), cv_scores*100, color=TEXT_COLOR, s=60, zorder=3, alpha=0.8)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["5-Fold CV"])
    ax2.set_ylabel("Accuracy (%)", color=TEXT_COLOR)
    ax2.set_title(f"Cross-Validation  (Test Acc: {accuracy*100:.2f}%)", color=TEXT_COLOR)
    ax2.set_ylim(50, 105)

    for fold, score in enumerate(cv_scores, 1):
        ax2.annotate(f"F{fold}: {score*100:.1f}%", xy=(1, score*100),
                     xytext=(1.15, score*100),
                     fontsize=9, color=TEXT_COLOR, va="center")

    plt.tight_layout()
    save("4_model_performance.png")


# ── 5. TF-IDF Top Features ────────────────────────────────────────────────────
def plot_top_features(pipeline, n=18):
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]
    classes = pipeline.classes_
    feat_names = np.array(tfidf.get_feature_names_out())

    fig, axes = plt.subplots(1, len(classes), figsize=(16, 6))
    fig.suptitle(f"Top {n} TF-IDF Features per Sentiment Class", fontsize=16, color=TEXT_COLOR, y=1.02)

    for ax, cls, coef_row in zip(axes, classes, clf.coef_):
        top_idx = np.argsort(coef_row)[-n:][::-1]
        top_feats = feat_names[top_idx]
        top_vals  = coef_row[top_idx]

        colors = [PALETTE[cls]] * n
        ax.barh(range(n), top_vals[::-1], color=[PALETTE[cls]]*n,
                alpha=0.85, edgecolor=BG_COLOR)
        ax.set_yticks(range(n))
        ax.set_yticklabels(top_feats[::-1], fontsize=9.5)
        ax.set_title(f"{cls}", color=TEXT_COLOR, fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient Weight", color=TEXT_COLOR)
        ax.invert_yaxis()

    plt.tight_layout()
    save("5_top_tfidf_features.png")


# ── 6. Word Clouds ────────────────────────────────────────────────────────────
def plot_wordclouds(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Word Clouds by Sentiment", fontsize=16, color=TEXT_COLOR, y=1.02)
    fig.patch.set_facecolor(BG_COLOR)

    wc_colors = {
        "Positive": ["#2DD4BF", "#06B6D4", "#34D399"],
        "Negative": ["#F87171", "#FB923C", "#F43F5E"],
        "Neutral":  ["#A78BFA", "#818CF8", "#C4B5FD"],
    }

    for ax, sentiment in zip(axes, ["Positive", "Negative", "Neutral"]):
        corpus = " ".join(df[df["sentiment"] == sentiment]["processed_text"].dropna())

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return np.random.choice(wc_colors[sentiment])

        wc = WordCloud(
            width=600, height=400,
            background_color="#1E293B",
            max_words=120,
            collocations=False,
            color_func=color_func,
        ).generate(corpus)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(sentiment, color=PALETTE[sentiment], fontsize=14, fontweight="bold")
        ax.set_facecolor(CARD_COLOR)

    plt.tight_layout()
    save("6_word_clouds.png")


# ── 7. Sentiment Trends over Tweet IDs ────────────────────────────────────────
def plot_sentiment_trends(df: pd.DataFrame):
    df = df.copy().sort_values("tweet_id")
    df["bucket"] = pd.cut(df["tweet_id"], bins=50, labels=False)
    trend = df.groupby(["bucket", "sentiment"]).size().unstack(fill_value=0)
    trend = trend.div(trend.sum(axis=1), axis=0) * 100  # % share

    fig, ax = plt.subplots(figsize=(14, 5))
    for sent in trend.columns:
        ax.plot(trend.index, trend[sent], color=PALETTE[sent], linewidth=2.5,
                label=sent, marker="o", markersize=3, alpha=0.85)
        ax.fill_between(trend.index, trend[sent], alpha=0.12, color=PALETTE[sent])

    ax.set_title("Sentiment Share Trends Across Dataset (50 equal-width buckets)",
                 fontsize=14, color=TEXT_COLOR)
    ax.set_xlabel("Tweet Bucket (chronological)", color=TEXT_COLOR)
    ax.set_ylabel("Share (%)", color=TEXT_COLOR)
    ax.legend()
    ax.set_ylim(0, 80)

    plt.tight_layout()
    save("7_sentiment_trends.png")


# ── 8. Engagement by Sentiment ────────────────────────────────────────────────
def plot_engagement(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Engagement Metrics by Sentiment", fontsize=16, color=TEXT_COLOR, y=1.02)

    for ax, col, label in zip(axes, ["likes", "retweets"], ["Likes", "Retweets"]):
        data = [df[df["sentiment"]==s][col].values for s in ["Positive","Negative","Neutral"]]
        vp = ax.violinplot(data, positions=[1,2,3], showmedians=True, showextrema=True)
        for i, (body, col_name) in enumerate(zip(vp["bodies"], ["Positive","Negative","Neutral"])):
            body.set_facecolor(PALETTE[col_name])
            body.set_alpha(0.7)
        vp["cmedians"].set_color(TEXT_COLOR)
        vp["cmedians"].set_linewidth(2)
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(["Positive","Negative","Neutral"])
        ax.set_ylabel(label, color=TEXT_COLOR)
        ax.set_title(f"{label} Distribution", color=TEXT_COLOR)
        ax.set_ylim(0, None)

    plt.tight_layout()
    save("8_engagement_by_sentiment.png")


# ── Master Runner ─────────────────────────────────────────────────────────────
def run_all():
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score

    # Load data
    df = pd.read_csv("tweets_dataset.csv")
    df["processed_text"] = df["text"].map(preprocess)

    # Load model
    pipeline = joblib.load("outputs/sentiment_model.pkl")
    classes  = pipeline.classes_.tolist()

    X = df["processed_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = pipeline.predict(X_test)
    acc    = (y_pred == y_test.values).mean()

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)

    from sklearn.metrics import classification_report as cr
    report_dict = cr(y_test, y_pred, output_dict=True)

    print("\n[INFO] Generating visualisations ...\n")
    plot_distribution(df)
    plot_text_lengths(df)
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_metrics_dashboard(report_dict, cv_scores, acc)
    plot_top_features(pipeline)
    plot_wordclouds(df)
    plot_sentiment_trends(df)
    plot_engagement(df)
    print("\n[DONE] All plots saved to outputs/")


if __name__ == "__main__":
    run_all()
