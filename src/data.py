"""Download and prepare the Twitter US Airline Sentiment dataset.

Two tasks are built from it:
  * sentiment  - negative / neutral / positive, all tweets
  * topic      - complaint reason for negative tweets (8 classes)
"""
from pathlib import Path
import re
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment"
    "/resolve/main/Tweets.csv"
)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "Tweets.csv"
SEED = 42

# Merge very small / overlapping classes into readable labels.
TOPIC_MAP = {
    "Customer Service Issue": "Customer service",
    "Late Flight": "Late flight",
    "Cancelled Flight": "Cancelled flight",
    "Lost Luggage": "Luggage",
    "Damaged Luggage": "Luggage",
    "Bad Flight": "Bad flight experience",
    "Flight Booking Problems": "Booking problems",
    "Flight Attendant Complaints": "Cabin crew",
    "longlines": "Long queues",
}
TOPICS = sorted(set(TOPIC_MAP.values()))
SENTIMENTS = ["negative", "neutral", "positive"]


def download() -> Path:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW} ...")
        urllib.request.urlretrieve(DATA_URL, RAW)
    return RAW


def clean(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)          # links
    text = re.sub(r"@\w+", " ", text)             # @mentions (airline handles leak the label)
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def load() -> pd.DataFrame:
    df = pd.read_csv(download())
    df["clean_text"] = df["text"].astype(str).map(clean)
    df["created"] = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True)
    df["topic"] = df["negativereason"].map(TOPIC_MAP)
    return df


def splits(task: str):
    """Return (train_df, test_df) with a stratified 80/20 split."""
    df = load()
    if task == "topic":
        df = df[df["airline_sentiment"].eq("negative") & df["topic"].notna()]
        y = "topic"
    elif task == "sentiment":
        y = "airline_sentiment"
    else:
        raise ValueError(task)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[y])
    return train.reset_index(drop=True), test.reset_index(drop=True), y


if __name__ == "__main__":
    for task in ("sentiment", "topic"):
        tr, te, y = splits(task)
        print(task, len(tr), len(te))
        print(tr[y].value_counts(normalize=True).round(3), "\n")
