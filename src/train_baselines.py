import argparse
from pathlib import Path
from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data(path: Path, text_col: str, label_col: str):
    df = pd.read_csv(path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def train_tfidf_logreg(train_texts, train_labels):
    # Simple, standard TF–IDF setup
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(X_train, train_labels)
    return vectorizer, clf


def evaluate_majority(train_labels, test_labels):
    majority_label = Counter(train_labels).most_common(1)[0][0]
    y_pred = [majority_label] * len(test_labels)
    acc = accuracy_score(test_labels, y_pred)
    print("=== Majority baseline ===")
    print(f"Majority label: {majority_label}")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(test_labels, y_pred))
    print()
    return majority_label


def evaluate_tfidf_logreg(vectorizer, clf, test_texts, test_labels):
    X_test = vectorizer.transform(test_texts)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    print("=== TF–IDF + Logistic Regression ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(test_labels, y_pred))
    print()
    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_data(args.data, args.text_col, args.label_col)

    # --- Hard requirement: dataset must be non-toy ---
    counts = Counter(labels)
    print("Label counts:", counts)

    if len(counts) < 2:
        raise ValueError(
            f"Need at least 2 classes for baselines, found {len(counts)}: {counts}"
        )

    if min(counts.values()) < 2:
        raise ValueError(
            "Each label must appear at least 2 times before running baselines.\n"
            f"Current counts: {counts}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=42,
    )

    majority_label = evaluate_majority(y_train, y_test)

    vectorizer, clf = train_tfidf_logreg(X_train, y_train)
    evaluate_tfidf_logreg(vectorizer, clf, X_test, y_test)

    model_path = args.out_dir / "twi_sentiment_tfidf_logreg.joblib"
    joblib.dump(
        {"vectorizer": vectorizer, "clf": clf, "majority_label": majority_label},
        model_path,
    )
    print(f"Saved baseline model to {model_path}")


if __name__ == "__main__":
    main()
