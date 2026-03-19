import streamlit as st
import joblib
import spacy
from spellchecker import SpellChecker

# Load your trained model
final_model = joblib.load("scam_detector_model.pkl")
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

def clean_email(email_text):
    """Clean email text: remove newlines, tabs, multiple spaces"""
    text = email_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())  # collapse multiple spaces
    return text

# Feature functions
def scam_score_keyword(email_text):
    """Compute keyword-based scam score (max 0.25)"""
    keywords = {
        "urgent": 0.25, "password": 0.1, "verify": 0.1,
        "click": 0.2, "winner": 0.25, "now": 0.1,
        "money": 0.25, "buy": 0.25, "gift": 0.25, "quickly": 0.25
    }
    email_lower = email_text.lower()
    score = sum(weight for word, weight in keywords.items() if word in email_lower)
    return min(score, 0.25)

def scam_score_misspell(email_text):
    """Compute misspelling-based scam score (skip PERSON, ORG, GPE, PRODUCT)"""
    doc = nlp(email_text)
    words_to_check = [
        token.text.lower()
        for token in doc
        if token.is_alpha and token.ent_type_ not in ["PERSON", "ORG", "GPE", "PRODUCT"]
    ]
    if not words_to_check:
        return 0
    misspelled = spell.unknown(words_to_check)
    ratio = len(misspelled) / len(words_to_check)
    return min(ratio, 0.25)  # cap at 0.25

def scam_score_address(email_address, email_text):
    """Return 0 if any part of email address appears in email text"""
    email_lower = email_text.lower()
    parts = email_address.replace("@", ".").split(".")
    found = [part for part in parts if part.lower() in email_lower]
    return 0 if len(found) > 0 else 0.5

def predict_scam(email_address, email_text):
    """Compute probability of scam using trained model"""
    cleaned_text = clean_email(email_text)

    features = [
        scam_score_keyword(cleaned_text),
        scam_score_misspell(cleaned_text),
        scam_score_address(email_address, cleaned_text)
    ]

    prob = final_model.predict_proba([features])[0][1]  # probability of scam
    return prob

# Streamlit UI
st.title("Scam Email Detector")
st.write(
    """
    Enter the **email address** and **email content** below to see the likelihood
    that it is a scam. The probability ranges from 0 (not a scam) to 1 (highly likely a scam).
    """
)
email_address = st.text_input("Email Address")
email_text = st.text_area("Email Content")

if st.button("Check Email"):
    prob = predict_scam(email_address, email_text)
    if prob < 0.25:
        st.success(f"Low Scam Probability: {prob:.2f}")
    elif prob < 0.5:
        st.warning(f"Medium Scam Probability: {prob:.2f}")
    else:
        st.error(f"High Scam Probability: {prob:.2f}")

    # 3️⃣ Progress bar
    st.progress(prob)  # value between 0 and 1



st.sidebar.title("About")
st.sidebar.info(
    """
    This tool uses a **Machine Learning Logistic Regression Model** trained on email features like:
    - Keywords (`urgent`, `verify`, etc.)
    - Misspellings or unusual words
    - Email address appearing in the content

    This is a **rule-based/ML hybrid detector**; results are indicative.

    **Note:** This is a predictive tool and may not be 100% accurate. Always exercise caution with suspicious emails.

    This tool was created using Python (sklearn, pandas, spellchecker, string, numpy)
    """
)