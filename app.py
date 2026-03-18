import streamlit as st
import joblib
from spellchecker import SpellChecker

# Load your trained model
final_model = joblib.load("scam_detector_model.pkl")

# Feature functions
def scam_score_keyword(email):
    keywords = {"urgent": .25, "password": .1, "verify": .1, "click": .2, "winner": .25, "now": .1}
    score = 0
    email_lower = email.lower()
    for word, weight in keywords.items():
        if word in email_lower:
            score += weight
    return min(score, 0.25)

def scam_score_misspell(email):
    spell = SpellChecker()
    words = email.lower().split()
    misspelled = spell.unknown(words)
    if len(misspelled) == 0:
        return 0
    elif len(misspelled) == 1:
        return 0.15
    else:
        return 0.25

def email_in_text(email_address, email_text):
    text_lower = email_text.lower()
    parts = email_address.replace("@", ".").split(".")
    found = [part for part in parts if part.lower() in text_lower]
    return 0.5 if len(found) > 0 else 0

def predict_scam(email_address, email_text):
    features = [
        scam_score_keyword(email_text),
        scam_score_misspell(email_text),
        email_in_text(email_address, email_text)
    ]
    prob = final_model.predict_proba([features])[0][1]
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

    # 4️⃣ Feature breakdown in expander
    with st.expander("Why this score?"):
        st.write(f"Keyword Score: {scam_score_keyword(email_text):.2f}")
        st.write(f"Misspell Score: {scam_score_misspell(email_text):.2f}")
        st.write(f"Email Address Match: {email_in_text(email_address, email_text):.2f}")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a **Logistic Regression Model** trained on email features like:
    - Keywords (`urgent`, `verify`, etc.)
    - Misspellings
    - Email address matches

    This is a **rule-based/ML hybrid detector**; results are indicative.
    """
)