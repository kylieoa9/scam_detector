import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import streamlit as st

# -----------------------------
# Load NLP tools
# -----------------------------
import spacy
from spacy.cli import download

# Load spaCy model, install if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()

# -----------------------------
# Load pre-trained model + vectorizer
# -----------------------------
tfidf = joblib.load("tfidf.pkl")
final_model = joblib.load("scam_detector_model.pkl")

# -----------------------------
# Load dataset (optional)
# -----------------------------
df = pd.read_excel("project_data.xlsx")

# -----------------------------
# Helper functions
# -----------------------------
def clean_email(email):
    email = email.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    email = " ".join(email.split())
    return email

def scam_score_keyword(email):
    email_lower = clean_email(email.lower())
    keywords = {"urgent": .25, "password": .1, "verify": .1, "click": .2,
                "winner": .25, "now": .1, "money": .25, "buy": .25, "gift": .25, "quickly": .25}
    score = sum(weight for word, weight in keywords.items() if word in email_lower)
    return min(score, 0.25)

def scam_score_misspell(email):
    email = clean_email(email)
    doc = nlp(email)
    words_to_check = [token.text.lower() for token in doc if token.is_alpha and token.ent_type_ not in ["PERSON","ORG","GPE","PRODUCT"]]
    misspelled = spell.unknown(words_to_check)
    if len(misspelled) == 0: return 0
    elif len(misspelled) == 1: return 0.15
    else: return 0.25

def scam_score_address(email_address, email):
    email_lower = clean_email(email.lower())
    parts = email_address.replace("@", ".").split(".")
    found = [part for part in parts if part.lower() in email_lower]
    return 0 if len(found) > 0 else 0.5

def text_stats(email):
    email_clean = clean_email(email)
    return [
        len(email_clean.split()),
        sum(1 for c in email_clean if c.isupper()),
        email_clean.count("!"),
        int("http" in email_clean)
    ]

def rule_flags(email, email_address):
    return [
        scam_score_keyword(email),
        scam_score_misspell(email),
        scam_score_address(email_address, email)
    ]

# -----------------------------
# Prediction function
# -----------------------------
def predict_scam(email_address, email_text):
    tfidf_feat = tfidf.transform([email_text])
    stats_feat = np.array([text_stats(email_text)])
    rule_feat = np.array([rule_flags(email_text, email_address)])
    numeric_feat = csr_matrix(np.hstack([stats_feat, rule_feat]))
    features = hstack([tfidf_feat, numeric_feat])
    prob = final_model.predict_proba(features)[0][1]
    return prob

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Scam Email Detector")
st.write("""
Enter the **email address** and **email content** below to see the likelihood
that it is a scam. The probability ranges from 0 (not a scam) to 1 (highly likely a scam).
""")

email_address = st.text_input("Email Address")
email_text = st.text_area("Email Content")

if st.button("Check Email"):
    with st.spinner("Analyzing email..."):
        prob = predict_scam(email_address, email_text)

    if prob < 0.25:
        st.success(f"Low Scam Probability: {prob:.2f}")
    elif prob < 0.5:
        st.warning(f"Medium Scam Probability: {prob:.2f}")
    else:
        st.error(f"High Scam Probability: {prob:.2f}")

    st.progress(prob)

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.title("About")
st.sidebar.info("""
This tool uses a **Machine Learning Logistic Regression Model** trained on email features like:
- Keywords (`urgent`, `verify`, etc.)
- Misspellings or unusual words
- Email address appearing in the content
- Email statistics (frequency of words, etc.)

This is a **rule-based/ML hybrid detector**; results are indicative.

**Note:** This is a predictive tool and may not be 100% accurate. Always exercise caution with suspicious emails.

Created using Python (sklearn, pandas, spellchecker, numpy, Streamlit)
""")