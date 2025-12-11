import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Davlan/afrisenti-twitter-sentiment-afroxlmr-large"
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


@st.cache_resource
def load_model():
    """Load tokenizer and model once and reuse them."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    """Run sentiment prediction on a single text."""
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


# --- Streamlit UI ---

st.title("Twi Sentiment Analysis Demo")

st.write(
    "Enter a sentence in Twi and the model will predict whether "
    "the sentiment is positive, neutral, or negative."
)

user_text = st.text_area(
    "Twi text",
    placeholder="Type a Twi sentence here..."
)

if st.button("Analyse sentiment"):
    if not user_text.strip():
        st.warning("Please enter a Twi sentence first.")
    else:
        tokenizer, model = load_model()
        label, score = predict(user_text, tokenizer, model)
        st.write(f"**Sentiment:** {label}")
        st.write(f"Confidence: {score:.3f}")
