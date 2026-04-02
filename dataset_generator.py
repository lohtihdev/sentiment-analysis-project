"""
dataset_generator.py
--------------------
Generates a realistic synthetic dataset of 10,000 tweets
with Positive, Negative, and Neutral sentiments.
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Twitter-style vocabulary pools ────────────────────────────────────────────

POSITIVE_PHRASES = [
    "absolutely love this", "so happy today", "best day ever", "feeling great",
    "amazing experience", "highly recommend", "this is wonderful", "so excited",
    "grateful for everything", "life is beautiful", "fantastic product", "super helpful",
    "incredible service", "totally worth it", "beyond expectations", "pure joy",
    "made my day", "feeling blessed", "awesome vibes", "can't believe how good",
    "totally recommend", "loving every moment", "top notch quality", "five stars",
    "exceeded my expectations", "outstanding performance", "really impressed",
    "great customer support", "perfect in every way", "so satisfying",
]

NEGATIVE_PHRASES = [
    "terrible experience", "worst product ever", "so disappointed", "complete waste",
    "never buying again", "horrible service", "really frustrated", "so angry right now",
    "broken on arrival", "refund requested", "absolute garbage", "do not recommend",
    "totally useless", "feels like a scam", "couldn't be worse", "deeply unhappy",
    "such a letdown", "extremely poor quality", "avoid at all costs", "wasted my money",
    "constant crashes", "customer service is awful", "waited for hours", "no response at all",
    "still broken", "false advertising", "nothing works", "what a disaster",
    "unacceptable behavior", "pathetic excuse",
]

NEUTRAL_PHRASES = [
    "just received my order", "tried the new update", "checking the reviews",
    "wondering if it works", "any thoughts on this", "reading the manual",
    "standard shipping arrived", "medium quality overall", "it does what it says",
    "typical product", "nothing special really", "met basic expectations",
    "average at best", "okay I guess", "neither good nor bad",
    "waiting for more information", "comparing different options", "got the package today",
    "will update later", "so far so good I think", "looks normal to me",
    "haven't tried it yet", "some features work some don't", "meh honestly",
    "results may vary", "depends on your use case", "not sure about this one",
    "seems fine for now", "could go either way", "pretty standard stuff",
]

POSITIVE_WORDS = [
    "love", "happy", "great", "awesome", "fantastic", "excellent", "brilliant",
    "joy", "grateful", "blessed", "perfect", "wonderful", "superb", "incredible",
    "delighted", "thrilled", "cheerful", "positive", "beautiful", "amazing",
]

NEGATIVE_WORDS = [
    "hate", "terrible", "awful", "horrible", "disgusting", "bad", "broken",
    "useless", "frustrating", "angry", "disappointed", "failed", "worst",
    "pathetic", "garbage", "trash", "ruined", "waste", "nightmare", "dreadful",
]

NEUTRAL_WORDS = [
    "okay", "average", "standard", "normal", "regular", "typical", "moderate",
    "mediocre", "ordinary", "common", "usual", "decent", "fair", "acceptable",
    "reasonable", "adequate", "sufficient", "midrange", "general", "basic",
]

HASHTAGS_POS = ["#happy", "#blessed", "#love", "#winning", "#grateful",
                "#amazing", "#bestday", "#success", "#positive", "#joy"]
HASHTAGS_NEG = ["#fail", "#disappointed", "#angry", "#neveragain", "#terrible",
                "#scam", "#avoid", "#broken", "#rant", "#frustrated"]
HASHTAGS_NEU = ["#review", "#update", "#justgot", "#thoughts", "#feedback",
                "#opinion", "#checking", "#info", "#standard", "#fyi"]

USERNAMES = [f"@user{i}" for i in range(1000, 9999)]
RT_PREFIX = ["RT", "via", ""]


def make_tweet(sentiment: str) -> str:
    """Compose a realistic-looking tweet for the given sentiment label."""
    if sentiment == "Positive":
        core = random.choice(POSITIVE_PHRASES)
        extras = random.sample(POSITIVE_WORDS, k=random.randint(1, 3))
        tags = random.sample(HASHTAGS_POS, k=random.randint(0, 2))
    elif sentiment == "Negative":
        core = random.choice(NEGATIVE_PHRASES)
        extras = random.sample(NEGATIVE_WORDS, k=random.randint(1, 3))
        tags = random.sample(HASHTAGS_NEG, k=random.randint(0, 2))
    else:
        core = random.choice(NEUTRAL_PHRASES)
        extras = random.sample(NEUTRAL_WORDS, k=random.randint(0, 2))
        tags = random.sample(HASHTAGS_NEU, k=random.randint(0, 1))

    rt = random.choice(RT_PREFIX)
    mention = random.choice(USERNAMES) if random.random() < 0.35 else ""

    parts = [p for p in [rt, mention, core, " ".join(extras), " ".join(tags)] if p]
    tweet = " ".join(parts)

    # Sprinkle realistic noise
    if random.random() < 0.15:
        tweet = tweet.upper()
    if random.random() < 0.2:
        tweet += " 😊😍🙌" if sentiment == "Positive" else (" 😠😤💔" if sentiment == "Negative" else " 🤔😐")
    if random.random() < 0.1:
        tweet += "!!!" if sentiment != "Neutral" else "..."

    return tweet[:280]  # Twitter hard cap


def generate_dataset(n: int = 10_000) -> pd.DataFrame:
    # Realistic class distribution: Positive 45%, Negative 35%, Neutral 20%
    labels = (
        ["Positive"] * int(n * 0.45) +
        ["Negative"] * int(n * 0.35) +
        ["Neutral"]  * (n - int(n * 0.45) - int(n * 0.35))
    )
    random.shuffle(labels)

    rows = []
    for i, label in enumerate(labels):
        rows.append({
            "tweet_id": 1_000_000 + i,
            "text": make_tweet(label),
            "sentiment": label,
            "likes": int(np.random.exponential(scale=50)),
            "retweets": int(np.random.exponential(scale=15)),
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset(10_000)
    out_path = "tweets_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset saved → {out_path}")
    print(df["sentiment"].value_counts())
    print(df.head(3))
