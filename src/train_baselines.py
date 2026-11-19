"""Supervised models: TF-IDF + logistic regression, and sentence embeddings + logistic regression."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from data import ROOT, splits

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def scores(y_true, y_pred):
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def tfidf_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def embed(texts, model=None):
    from sentence_transformers import SentenceTransformer
    model = model or SentenceTransformer(EMBED_MODEL)
    return model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMBED_MODEL)
    all_metrics = {}

    for task in ("topic", "sentiment"):
        train, test, y = splits(task)
        all_metrics[task] = {"n_train": len(train), "n_test": len(test)}

        # Majority-class floor, so every model has something to beat.
        majority = train[y].mode()[0]
        all_metrics[task]["majority_class"] = scores(test[y], [majority] * len(test))

        # 1. TF-IDF + logistic regression
        pipe = tfidf_pipeline().fit(train["clean_text"], train[y])
        pred = pipe.predict(test["clean_text"])
        all_metrics[task]["tfidf_logreg"] = scores(test[y], pred)
        joblib.dump(pipe, MODELS / f"tfidf_{task}.joblib", compress=3)
        (RESULTS / f"report_tfidf_{task}.txt").write_text(classification_report(test[y], pred, digits=3))

        # 2. Sentence embeddings + logistic regression
        Xtr, Xte = embed(train["clean_text"], st), embed(test["clean_text"], st)
        emb_clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(Xtr, train[y])
        pred_e = emb_clf.predict(Xte)
        all_metrics[task]["embedding_logreg"] = scores(test[y], pred_e)
        (RESULTS / f"report_embedding_{task}.txt").write_text(classification_report(test[y], pred_e, digits=3))

        print(task, json.dumps(all_metrics[task], indent=2))

    path = RESULTS / "metrics_supervised.json"
    path.write_text(json.dumps(all_metrics, indent=2))
    print("Saved", path)


if __name__ == "__main__":
    main()
