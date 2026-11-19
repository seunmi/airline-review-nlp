"""Zero-shot complaint-topic classification: no training labels used, only topic descriptions.

Methods
  nli  - natural-language-inference model scoring "This complaint is about {topic}."
  llm  - a small open instruction-tuned LLM (Qwen2.5-Instruct) prompted to pick one topic;
         each label is scored by log-likelihood so the answer is always a valid topic.

Usage:  python zero_shot.py --method nli|llm [--limit N]
"""
import argparse
import json
import time

import numpy as np
import torch

from data import TOPICS, splits
from train_baselines import RESULTS, scores

NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Plain-English descriptions help zero-shot models more than terse label names.
DESCRIPTIONS = {
    "Bad flight experience": "an uncomfortable or unpleasant flight (seats, wifi, food, entertainment, the plane)",
    "Booking problems": "problems booking, changing or paying for a flight, or with the website or app",
    "Cabin crew": "rude or unhelpful flight attendants",
    "Cancelled flight": "a cancelled flight",
    "Customer service": "poor customer service, being on hold, or no response from the airline",
    "Late flight": "a delayed or late flight",
    "Long queues": "long lines or queues at the airport",
    "Luggage": "lost, delayed or damaged luggage or bags",
}
assert set(DESCRIPTIONS) == set(TOPICS)


def run_nli(texts, batch=16):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=NLI_MODEL, device=-1)
    labels = list(DESCRIPTIONS.values())
    back = {v: k for k, v in DESCRIPTIONS.items()}
    out = clf(list(texts), candidate_labels=labels,
              hypothesis_template="This airline complaint is about {}.", batch_size=batch)
    return [back[o["labels"][0]] for o in out]


def run_llm(texts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32).eval()
    menu = "\n".join(f"- {t}: {d}" for t, d in DESCRIPTIONS.items())
    label_ids = [tok(" " + t, add_special_tokens=False).input_ids for t in TOPICS]

    preds = []
    for text in texts:
        msgs = [
            {"role": "system", "content": "You classify airline customer complaints."},
            {"role": "user", "content": f"Topics:\n{menu}\n\nComplaint: \"{text}\"\n\n"
                                         "Which single topic best fits? Answer with the topic name only."},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Topic:"
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        # Score all candidate labels in one batched forward pass. Left-padding puts every
        # label at the end, so only the last few positions' logits are needed (saves memory).
        seqs = [p_ids + ids for ids in label_ids]
        width = max(map(len, seqs))
        k = max(map(len, label_ids)) + 1
        pad = tok.pad_token_id or 0
        batch = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(batch, attention_mask=mask, position_ids=pos, logits_to_keep=k).logits
        logp = torch.log_softmax(logits, -1)            # [n_labels, k, vocab]

        def label_score(i):
            ids = label_ids[i]
            # token ids[j] sits at absolute position width-len(ids)+j, predicted by the previous position
            return sum(logp[i, k - len(ids) - 1 + j, t].item() for j, t in enumerate(ids)) / len(ids) ** 0.5

        best = max(range(len(TOPICS)), key=label_score)
        preds.append(TOPICS[best])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["nli", "llm"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="evaluate on a stratified sample of N test tweets")
    args = ap.parse_args()

    _, test, y = splits("topic")
    if args.limit:
        from sklearn.model_selection import train_test_split
        test, _ = train_test_split(test, train_size=args.limit, random_state=42, stratify=test[y])
        test = test.reset_index(drop=True)
    t0 = time.time()
    preds = run_nli(test["clean_text"]) if args.method == "nli" else run_llm(test["clean_text"])
    res = scores(test[y], preds) | {"n_test": len(test), "seconds": round(time.time() - t0)}
    test.assign(pred=preds)[["clean_text", y, "pred"]].to_csv(RESULTS / f"preds_zeroshot_{args.method}.csv", index=False)
    (RESULTS / f"metrics_zeroshot_{args.method}.json").write_text(json.dumps(res, indent=2))
    print(args.method, res)


if __name__ == "__main__":
    main()
