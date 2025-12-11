from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse

# Pretrained African Twitter sentiment model that supports Twi
MODEL_NAME = "Davlan/afrisenti-twitter-sentiment-afroxlmr-large"

# According to the model card: 0 positive, 1 neutral, 2 negative
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**encoded).logits[0]

    probs = torch.softmax(logits, dim=0)
    label_id = int(torch.argmax(probs))
    label = ID2LABEL[label_id]
    score = float(probs[label_id])
    return label, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input Twi text")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)
    label, score = predict_sentiment(args.text, tokenizer, model)
    print(f"Text: {args.text}")
    print(f"Predicted sentiment: {label} (confidence {score:.3f})")


if __name__ == "__main__":
    main()
