import os
import re
import joblib
import streamlit as st
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_news_model.joblib")


def clean_text(s: str) -> str:
    """Same cleaning used during training."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Fake News Detector")

title = st.text_input("News Title")
text = st.text_area("News Text")

if st.button("Predict"):
    if not title.strip() and not text.strip():
        st.warning("Please enter a title or text.")
    else:
        combined = clean_text(f"{title} {text}")
        prediction = model.predict([combined])[0]

        if prediction == 1:
            st.success("This is Real News 🟢")
        else:
            st.error("This is Fake News 🔴")

# cd frontend
#  python -m streamlit run app.py