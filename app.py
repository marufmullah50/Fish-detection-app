# app.py
# Run with: streamlit run app.py

import os
import re
import cv2
import numpy as np
import requests
import streamlit as st
import plotly.express as px
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from keras.models import load_model

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="🐟 Fish Species Detector", page_icon="🐠", layout="wide")
MODEL_PATH = "Fish_model.h5"
IMG_SIZE = (128, 128)  # Match your model input

CLASS_NAMES = [
    "Black_Rohu","Catla","Common_Carp","Freshwater_Shark","Grass_Carp",
    "Long_whiskered_Catfish","Mirror_Carp","Mrigal","Nile_Tilapia",
    "Rohu","Silver_Carp","Striped_Catfish"
]

# ---------------- CSS STYLE ---------------- #
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #d9f0ff, #f0fff9);
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #004d66;
    }
    div.stButton > button {
        background-color: #0077b6;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #023e8a;
        transform: scale(1.05);
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 18px;
        color: #006d77;
    }
    .stSuccess, .stWarning, .stInfo, .stError {
        color: #000000;
    }
    section[data-testid="stSidebar"] {
        background-color: #caf0f8;
        color: #000000;
    }
    .stMarkdown p, .stMarkdown li {
        color: #000000;
    }
    .stPlotlyChart {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- UTILS ---------------- #
@st.cache_resource
def load_keras_model(path):
    return load_model(path)

def preprocess(img):
    img = cv2.resize(img, IMG_SIZE).astype(np.float32)/255.0
    return np.expand_dims(img, 0)

def read_cv2(file):
    return cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

def topk(arr, k=3):
    idx = np.argsort(arr)[::-1][:k]
    return idx, arr[idx]

def clean(txt):
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]", "", txt)).strip()

# ---------------- WEB SCRAPING ---------------- #
def fetch_wikipedia_info(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        data = r.json()
        title = data.get("title", "")
        summary = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        thumbnail = data.get("thumbnail", {}).get("source", "")
        return {"title": title, "summary": summary, "url": page_url, "thumb": thumbnail}
    return None

def fetch_duckduckgo_info(query, n=3):
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}+fish+species"
    headers = {"User-Agent":"Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:n]:
        text = a.get_text()
        link = a.get("href")
        snippet_tag = a.find_next("a")
        snippet = snippet_tag.get_text() if snippet_tag else ""
        results.append({"title": text, "link": link, "snippet": snippet})
    return results

def auto_summary(text, max_sentences=5):
    """
    Generate a concise summary from plain text (Wikipedia + snippets).
    """
    if not text:
        return "No summary available."
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:max_sentences])

# ---------------- APP ---------------- #
st.title("🐟 Fish Species Detector")
st.markdown("Upload a fish image and explore species details from the web .")

col1, col2 = st.columns([1,1])

# Initialize session_state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "predicted" not in st.session_state:
    st.session_state.predicted = False

with col1:
    uploaded = st.file_uploader("📤 Upload an image(jpg,jpeg or png)", type=["jpg","jpeg","png"])
    detect_btn = st.button("🔍 Detect")
    reset_btn = st.button("🔄 Reset")

# Reset button clears previous detection
if reset_btn:
    st.session_state.uploaded_image = None
    st.session_state.predicted = False

# Store uploaded image in session_state
if uploaded:
    st.session_state.uploaded_image = uploaded
    st.session_state.predicted = False  # Reset prediction when new image uploaded

# Run detection
if detect_btn and st.session_state.uploaded_image:
    img = read_cv2(st.session_state.uploaded_image)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    # Prediction
    model = load_keras_model(MODEL_PATH)
    x = preprocess(img)
    preds = model.predict(x)
    probs = softmax(preds)[0]

    idxs, vals = topk(probs, 3)
    best = CLASS_NAMES[idxs[0]].replace("_"," ")
    conf = vals[0]*100

    with col1:
        st.success(f"**Prediction:** {best} ({conf:.1f}% confidence)")

        # Confidence chart
        fig = px.bar(
            x=[CLASS_NAMES[i].replace("_"," ") for i in idxs],
            y=vals*100,
            labels={"x":"Species","y":"Confidence (%)"},
            text=[f"{v*100:.1f}%" for v in vals],
            title="Top Predictions"
        )
        fig.update_traces(marker_color="lightblue", textposition="outside")
        st.plotly_chart(fig)

    with col2:
        st.subheader("📚 Species Information")

        # Wikipedia info
        wiki = fetch_wikipedia_info(best)
        collected_text = ""
        if wiki:
            with st.expander("🐠 Wikipedia Summary", expanded=True):
                if wiki.get("thumb"): st.image(wiki["thumb"], width=150)
                st.markdown(f"**[{wiki['title']}]({wiki['url']})**")
                st.write(clean(wiki.get("summary","")))
                collected_text += wiki.get("summary","")

        # DuckDuckGo info
        duck_info = fetch_duckduckgo_info(best)
        if duck_info:
            with st.expander("🌍 External Sources & Snippets"):
                for item in duck_info:
                    st.markdown(f"- [{item['title']}]({item['link']})")
                    if item['snippet']:
                        st.write(f"  {item['snippet']}")

        # Automatic concise summary
        if collected_text:
            summary_text = auto_summary(collected_text)
            with st.expander("✨ Concise Summary", expanded=True):
                st.write(summary_text)

    st.session_state.predicted = True

elif detect_btn:
    st.warning("⚠️ Please upload an image first.")
