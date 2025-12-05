import pickle
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import re

# load the saved model and tf-idf vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# some small helper functions
def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_url(url: str) -> str:
    """Fetch the web page and join all <p> tags as article text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    text = clean_whitespace(text)

    if not text:
        raise RuntimeError("Could not find readable text on the page.")
    return text


def extract_text_from_file(uploaded_file) -> str:
    """Read a txt or pdf file and return the text inside it."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        content = uploaded_file.read()
        uploaded_file.seek(0)
        return clean_whitespace(content.decode("utf-8", errors="ignore"))

    elif name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.read()))
        uploaded_file.seek(0)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
        return clean_whitespace(" ".join(pages_text))

    else:
        raise RuntimeError("Please upload a .txt or .pdf file only.")


def predict_text(text: str):
    """Run the model on given text and return label + probabilities."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    fake_prob = real_prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        try:
            fake_idx = np.where(classes == "fake")[0][0]
            real_idx = np.where(classes == "real")[0][0]
            fake_prob = float(probs[fake_idx]) * 100
            real_prob = float(probs[real_idx]) * 100
        except Exception:
            pass

    return pred, fake_prob, real_prob, vec


def top_keywords_for_prediction(text: str, vec, pred_label: str, top_n: int = 10):
    """
    Use model coefficients + tf-idf vocabulary
    to show some words that influenced the prediction.
    (This is just a rough idea of "why".)
    """
    if not hasattr(model, "coef_"):
        return []

    coef = model.coef_[0]
    classes = model.classes_
    positive_class = classes[1]

    # if prediction is the second class, keep sign,
    # otherwise flip it so that bigger score = more support
    if pred_label == positive_class:
        weights = coef
    else:
        weights = -coef

    inv_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}

    feature_indices = vec.nonzero()[1]
    word_scores = []
    for idx in feature_indices:
        word = inv_vocab.get(idx)
        if not word:
            continue
        score = float(weights[idx])
        word_scores.append((word, score))

    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    seen = set()
    top = []
    for w, s in word_scores:
        if w in seen:
            continue
        seen.add(w)
        top.append((w, s))
        if len(top) >= top_n:
            break
    return top


def analyze_and_display(text: str, source_label: str):
    text = clean_whitespace(text)
    if not text:
        st.warning("No useful text found.")
        return

    with st.expander("Show extracted text"):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    pred, fake_prob, real_prob, vec = predict_text(text)

    st.subheader("Result")

    if pred == "real":
        st.success("Prediction: REAL news")
    else:
        st.error("Prediction: FAKE news")

    if fake_prob is not None and real_prob is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real (approx.)", f"{real_prob:.2f}%")
        with col2:
            st.metric("Fake (approx.)", f"{fake_prob:.2f}%")

    st.caption(f"Source: {source_label}")

    st.subheader("Some words that affected the prediction")
    keywords = top_keywords_for_prediction(text, vec, pred)
    if not keywords:
        st.write("Could not compute word importance for this model.")
        return

    for word, score in keywords:
        st.markdown(f"- **{word}** (influence: {'+' if score >= 0 else ''}{score:.3f})")


# basic page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.title("Fake News Detection using Machine Learning")
st.write(
    "This is a small project where I trained a machine learning model to classify "
    "news as **fake** or **real**. You can try it out in three different ways below."
)

tab1, tab2, tab3 = st.tabs(
    ["Type / paste text", "Use a news URL", "Upload a file"]
)

# tab 1: text input
with tab1:
    st.subheader("1. Type or paste news text")
    text_input = st.text_area(
        "Enter a headline or article:",
        height=220,
        placeholder="Paste some news text here...",
    )
    if st.button("Analyze text", key="btn_text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            analyze_and_display(text_input, "Typed / pasted text")

# tab 2: URL input
with tab2:
    st.subheader("2. Paste a news article URL")
    url = st.text_input("URL (starting with http or https):", "")
    if st.button("Fetch and analyze", key="btn_url"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Downloading and reading the page..."):
                    url_text = extract_text_from_url(url)
                analyze_and_display(url_text, f"URL: {url}")
            except Exception as e:
                st.error(str(e))

# tab 3: file upload
with tab3:
    st.subheader("3. Upload a text or PDF file")
    uploaded = st.file_uploader(
        "Choose a file (.txt or .pdf):", type=["txt", "pdf"], accept_multiple_files=False
    )

    if st.button("Analyze file", key="btn_file"):
        if not uploaded:
            st.warning("Please upload a file first.")
        else:
            try:
                file_text = extract_text_from_file(uploaded)
                analyze_and_display(file_text, f"File: {uploaded.name}")
            except Exception as e:
                st.error(str(e))

# sidebar info
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
- Made as a practice project for fake news detection  
- Model: Logistic Regression  
- Features: TF-IDF representation of text  

The predictions are not perfect – the idea is to show how
text data can be turned into numbers and used with a simple
machine learning model.
"""
    )
