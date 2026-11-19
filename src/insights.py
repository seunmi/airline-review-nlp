"""Business insight: what drives negative sentiment, and how it differs by airline.

Uses the trained TF-IDF topic model to label every negative tweet, including the
'Can't Tell' ones the original annotators couldn't place, then summarises.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import ROOT, load

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    model = joblib.load(ROOT / "models" / "tfidf_topic.joblib")

    neg = df[df["airline_sentiment"].eq("negative")].copy()
    unlabelled = neg["topic"].isna()
    neg.loc[unlabelled, "topic"] = model.predict(neg.loc[unlabelled, "clean_text"])

    # 1. Share of negative tweets by airline
    sent = pd.crosstab(df["airline"], df["airline_sentiment"], normalize="index").sort_values("negative")
    # 2. Complaint mix overall and by airline
    overall = neg["topic"].value_counts(normalize=True)
    by_airline = pd.crosstab(neg["airline"], neg["topic"], normalize="index")[overall.index]

    summary = {
        "n_tweets": int(len(df)),
        "n_negative": int(len(neg)),
        "n_unlabelled_negatives_classified": int(unlabelled.sum()),
        "negative_share_by_airline": sent["negative"].round(3).to_dict(),
        "complaint_share_overall": overall.round(3).to_dict(),
        "top_two_topics_share": round(float(overall.iloc[:2].sum()), 3),
        "top_complaint_by_airline": by_airline.idxmax(axis=1).to_dict(),
        "customer_service_share_by_airline": by_airline["Customer service"].round(3).to_dict(),
    }
    (RESULTS / "insights.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Figures
    ax = (overall * 100).sort_values().plot.barh(figsize=(7, 4), color="#2b5d8a")
    ax.set_xlabel("% of negative tweets"); ax.set_ylabel("")
    ax.set_title("What airline customers complain about")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "complaint_topics.png", dpi=150); plt.close()

    ax = (by_airline * 100).plot.barh(stacked=True, figsize=(9, 4.5), colormap="tab10", width=0.75)
    ax.set_xlabel("% of each airline's negative tweets"); ax.set_ylabel("")
    ax.set_title("Complaint mix by airline")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "complaint_mix_by_airline.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
