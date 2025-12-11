import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from inference import load_model, predict_sentiment


def analyse_transformer(df, text_col, label_col, model_name):
    tokenizer, model = load_model(model_name)
    gold = df[label_col].astype(str).tolist()
    texts = df[text_col].astype(str).tolist()

    preds = []
    scores = []

    for text in texts:
        label, score = predict_sentiment(text, tokenizer, model)
        preds.append(label)
        scores.append(score)

    df_out = df.copy()
    df_out["pred_label"] = preds
    df_out["confidence"] = scores
    df_errors = df_out[df_out[label_col] != df_out["pred_label"]]
    print("=== Transformer classification report ===")
    print(classification_report(gold, preds))
    return df_errors


def analyse_baseline(df, text_col, label_col, model_path):
    artefacts = joblib.load(model_path)
    vectorizer = artefacts["vectorizer"]
    clf = artefacts["clf"]

    texts = df[text_col].astype(str).tolist()
    gold = df[label_col].astype(str).tolist()

    X = vectorizer.transform(texts)
    preds = clf.predict(X)

    df_out = df.copy()
    df_out["pred_label"] = preds
    df_errors = df_out[df_out[label_col] != df_out["pred_label"]]
    print("=== Baseline classification report ===")
    print(classification_report(gold, preds))
    return df_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--model", choices=["transformer", "baseline"], required=True)
    parser.add_argument("--model-name", type=str, default="Davlan/afrisenti-twitter-sentiment-afroxlmr-large")
    parser.add_argument("--baseline-path", type=Path, default=Path("models/twi_sentiment_tfidf_logreg.joblib"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.model == "transformer":
        df_errors = analyse_transformer(df, args.text_col, args.label_col, args.model_name)
    else:
        df_errors = analyse_baseline(df, args.text_col, args.label_col, args.baseline_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(args.out, index=False)
    print(f"Saved {len(df_errors)} error rows to {args.out}")


if __name__ == "__main__":
    main()
