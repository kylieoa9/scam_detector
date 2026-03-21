import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_excel("project_data.xlsx")

def clean_email(email):
    email = email.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    email = " ".join(email.split())
    return email

def text_stats(email):
    email_clean = clean_email(email)
    return [
        len(email_clean.split()),
        sum(1 for c in email_clean if c.isupper()),
        email_clean.count("!"),
        int("http" in email_clean)
    ]

emails = df["email_text"].tolist()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
tfidf_matrix = tfidf.fit_transform(emails)

stats_features = np.array([text_stats(email) for email in emails])
numeric_features_sparse = csr_matrix(stats_features)

X = hstack([tfidf_matrix, numeric_features_sparse])
y = df["label"].values

model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "scam_detector_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")