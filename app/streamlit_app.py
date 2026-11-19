"""Paste a customer message and see its sentiment and complaint topic.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data import clean  # noqa: E402


@st.cache_resource
def models():
    return (joblib.load(ROOT / "models" / "tfidf_sentiment.joblib"),
            joblib.load(ROOT / "models" / "tfidf_topic.joblib"))


st.set_page_config(page_title="Airline complaint classifier", page_icon="✈️")
st.title("Airline customer message classifier")
st.caption("TF-IDF + logistic regression models trained on 14,640 airline tweets.")

sent_model, topic_model = models()
examples = [
    "Been on hold for 2 hours and still no one has answered, this is ridiculous",
    "My bag never arrived in Heathrow and nobody can tell me where it is",
    "Flight delayed 3 hours with no explanation, missed my connection",
    "Crew on today's flight were brilliant, thank you!",
]
choice = st.selectbox("Try an example or write your own", ["(write your own)"] + examples)
text = st.text_area("Customer message", "" if choice == "(write your own)" else choice, height=100)

if text.strip():
    x = [clean(text)]
    s_probs = pd.Series(sent_model.predict_proba(x)[0], index=sent_model.classes_).sort_values(ascending=False)
    st.subheader(f"Sentiment: {s_probs.index[0]}  ({s_probs.iloc[0]:.0%})")

    if s_probs.index[0] == "negative":
        t_probs = pd.Series(topic_model.predict_proba(x)[0], index=topic_model.classes_).sort_values(ascending=False)
        st.subheader(f"Complaint topic: {t_probs.index[0]}  ({t_probs.iloc[0]:.0%})")
        st.bar_chart(t_probs.head(5))
    else:
        st.info("Not a complaint, so no topic is assigned.")
