# Twi Sentiment Analysis

This project explores sentence-level sentiment analysis for Ghanaian Twi using a pretrained transformer model and simple baseline classifiers.

The focus is on Twi and Twi-dominant texts in an African social media / conversational setting.

**Status:** Working prototype with  

- a command line interface for inference,  
- a Streamlit web demo,  
- a small labelled Twi sentiment dataset,  
- baseline models (majority, TF–IDF + Logistic Regression), and  
- a simple script for exporting model errors for qualitative analysis.  

Ongoing work includes expanding and rebalancing the Twi dataset and doing more detailed error analysis, especially on code-switched examples.

---

## Overview

The project currently provides:

- A wrapper around the pretrained African Twitter sentiment model  
  `Davlan/afrisenti-twitter-sentiment-afroxlmr-large`.
- `inference.py` for command line sentiment prediction in Twi.
- `app/demo_app.py`, a Streamlit app for interactive exploration.
- `src/train_baselines.py` to train and evaluate non-neural baselines.
- `src/error_analysis.py` to export model errors for manual inspection.
- A labelled Twi sentiment dataset in `data/twi_sentiment.csv`.

The labelled dataset is still relatively small and should be seen as a seed for experimentation rather than a final benchmark.

---

## Repository structure

Roughly:

```text
twi_sentiment_analysis/
  app/
    demo_app.py                      # Streamlit web demo
  data/
    twi_sentiment.csv                # Twi sentiment dataset (text, label)
  models/
    twi_sentiment_tfidf_logreg.joblib  # saved baseline model (created after training)
  src/
    train_baselines.py               # majority + TF–IDF + Logistic Regression baselines
    error_analysis.py                # export misclassified examples for inspection
  inference.py                       # CLI inference using the pretrained transformer
  requirements.txt
  README.md
````

---

## Models

### Pretrained transformer

The main model is:

* `Davlan/afrisenti-twitter-sentiment-afroxlmr-large`

Label mapping (from the model card):

* `0` → positive
* `1` → neutral
* `2` → negative

`inference.py` wraps this model and exposes a simple command line interface for single-sentence prediction. The same model is used in the Streamlit demo.

### Baselines

`src/train_baselines.py` trains two baselines on a labelled Twi sentiment CSV:

* **Majority baseline**
  Always predicts the most frequent label seen in the training data.

* **TF–IDF + Logistic Regression**
  Represents each text with word / n-gram features and trains a multinomial logistic regression classifier.

The script expects each label to appear at least a few times; if a class is too rare, it will raise a clear error so you know the dataset needs to be expanded.

---

## Data

### Twi sentiment dataset

`data/twi_sentiment.csv` is a small labelled Twi sentiment dataset in CSV format with columns:

* `text` – Twi sentence or short text
* `label` – sentiment label (`positive`, `neutral`, `negative`)

In the current version, the dataset contains **100 examples per class** (300 rows in total). The examples were created as a seed set and may be refined or extended in future versions. The dataset is intended for:

* quick experiments with Twi sentiment modelling, and
* testing and comparing baselines vs the pretrained transformer.

Planned improvements:

* More diverse and natural Twi examples per label.
* More code-switched Twi–English sentences.
* Clearer documentation of annotation choices and label criteria.

---

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Key dependencies:

* `torch`, `transformers` for the pretrained model
* `streamlit` for the web demo
* `pandas`, `scikit-learn`, `joblib` for baselines and evaluation

---

## Command line inference

`inference.py` provides a simple CLI for single-sentence sentiment prediction in Twi.

Example:

```bash
python inference.py --text "Me pɛ wo paa"
```

Typical output:

```text
Text: Me pɛ wo paa
Predicted sentiment: positive (confidence 0.92)
```

You can also override the model name if you want to experiment with other checkpoints:

```bash
python inference.py \
  --text "Ɛnyɛ adeɛ pa koraa" \
  --model_name Davlan/afrisenti-twitter-sentiment-afroxlmr-large
```

---

## Streamlit demo

The Streamlit app offers a minimal web interface around the same model.

Run:

```bash
streamlit run app/demo_app.py
```

Then open the URL printed by Streamlit in your browser.

The app:

* shows a text area with a placeholder prompt,
* runs the transformer model when you click **Analyse sentiment**, and
* displays the predicted label (positive / neutral / negative) together with the model’s confidence score.

---

## Baselines and quantitative evaluation

### Training baselines

Train and evaluate baselines on the Twi dataset:

```bash
python src/train_baselines.py --data data/twi_sentiment.csv
```

The script:

1. Loads `data/twi_sentiment.csv`.
2. Checks label counts and uses a stratified train/test split.
3. Trains:

   * a majority baseline, and
   * a TF–IDF + Logistic Regression classifier.
4. Prints accuracy and a per-class classification report for both models.
5. Saves the baseline model to `models/twi_sentiment_tfidf_logreg.joblib`.

This gives a simple quantitative comparison point for Twi sentiment classification without neural models.

---

## Qualitative error analysis

`src/error_analysis.py` provides a small utility for exporting misclassified examples for manual inspection.

In its current form, it is designed to be run on the labelled dataset and model predictions, and to write out a CSV containing at least:

* input text,
* gold label,
* predicted label, and (optionally)
* model confidence.

The resulting CSV can be opened in a spreadsheet or notebook to explore:

* common confusion patterns (e.g. neutral vs negative),
* behaviour on informal Twi spelling and variation,
* cases involving Twi–English mixing or code-switching.

The script is intended as a starting point and can be extended to analyse both baselines and the transformer model in more detail.

---

## Planned work

Short-term plans:

* Refine and expand the Twi sentiment dataset beyond the current 300 examples.
* Run more systematic comparisons between baselines and the transformer model.
* Document typical error patterns (e.g. negation, irony, code-switching) based on exported error sets.

Longer-term ideas:

* Fine-tuning the transformer specifically on Twi / English–Twi data.
* Exploring domain adaptation for different genres (social media vs more formal text).
* Linking this work to broader English–Twi code-switching studies.

The current repository is intended as a clean, inspectable starting point for Twi sentiment research, with room to grow into a more complete study.