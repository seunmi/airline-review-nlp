"""Zero-shot complaint-topic classification with the Gemini API.

Needs a free API key from https://aistudio.google.com/apikey:
    export GEMINI_API_KEY=...
    pip install google-genai
    python gemini_classify.py --limit 300

Results are written to results/metrics_zeroshot_gemini.json, so they can be
compared directly with the other models in the README table.
"""
import argparse
import json
import os
import time

from data import TOPICS, splits
from train_baselines import RESULTS, scores
from zero_shot import DESCRIPTIONS

MODEL = "gemini-2.5-flash"


def build_prompt(text: str) -> str:
    menu = "\n".join(f"- {t}: {d}" for t, d in DESCRIPTIONS.items())
    return (
        "You classify airline customer complaints into exactly one topic.\n\n"
        f"Topics:\n{menu}\n\n"
        f'Complaint: "{text}"\n\n'
        "Reply with the topic name only, copied exactly from the list."
    )


def classify(client, text: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=build_prompt(text))
            answer = (resp.text or "").strip().strip(".").lower()
            for t in TOPICS:                       # map free text back to a valid label
                if t.lower() == answer or t.lower() in answer:
                    return t
            return "Customer service"              # fallback: majority class
        except Exception as e:                     # rate limits on the free tier
            wait = 2 ** attempt * 5
            print(f"  retry in {wait}s ({e.__class__.__name__})")
            time.sleep(wait)
    return "Customer service"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="stratified sample size (free tier is rate-limited)")
    args = ap.parse_args()

    from google import genai
    from sklearn.model_selection import train_test_split
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    _, test, y = splits("topic")
    if args.limit and args.limit < len(test):
        test, _ = train_test_split(test, train_size=args.limit, random_state=42, stratify=test[y])
    test = test.reset_index(drop=True)

    t0, preds = time.time(), []
    for i, text in enumerate(test["clean_text"], 1):
        preds.append(classify(client, text))
        if i % 25 == 0:
            print(f"{i}/{len(test)}")
    res = scores(test[y], preds) | {"n_test": len(test), "seconds": round(time.time() - t0), "model": MODEL}
    test.assign(pred=preds)[["clean_text", y, "pred"]].to_csv(RESULTS / "preds_zeroshot_gemini.csv", index=False)
    (RESULTS / "metrics_zeroshot_gemini.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
