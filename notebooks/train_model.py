import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
CSV_PATH = "dataset/WELFake_Dataset.csv"
N_SAMPLES = 10000


def clean_text(s: str) -> str:
    """Same cleaning function used in the notebook."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    data = pd.read_csv(CSV_PATH)
    data = data.drop(columns=["Unnamed: 0"])

    df_fake = data[data["label"] == 0]
    df_real = data[data["label"] == 1]

    df_fake_sample = df_fake.sample(N_SAMPLES, random_state=RANDOM_STATE)
    df_real_sample = df_real.sample(N_SAMPLES, random_state=RANDOM_STATE)

    df = (
        pd.concat([df_fake_sample, df_real_sample])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    df = df.dropna()
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2
            )),
            ("model", LogisticRegression())
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(model_pipeline, "fake_news_model.joblib")
    print("Saved model to fake_news_model.joblib")


if __name__ == "__main__":
    main()
