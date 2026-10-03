"""
Fruit & Vegetable Classifier - Streamlit app
Model: custom 3-block CNN (PyTorch), 88 classes, 128x128 input.
Run:   streamlit run app.py
"""
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fruit_classification.pth"
CLASSES_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = 128

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Themes
# --------------------------------------------------------------------------
THEMES = {
    "☀️ Light": dict(
        bg="#F6F8FB", surface="#FFFFFF", surface2="#EEF2F7", text="#1B2433",
        muted="#6B7788", border="#E1E7EF", primary="#E8505B", primary2="#F59E0B",
        on_primary="#FFFFFF", chart="#E8505B",
    ),
    "🌙 Dark": dict(
        bg="#0E1117", surface="#171C26", surface2="#1F2633", text="#E8ECF3",
        muted="#93A0B4", border="#2A3344", primary="#FF6B6B", primary2="#FFB347",
        on_primary="#0E1117", chart="#FF6B6B",
    ),
    "🌊 Ocean": dict(
        bg="#0B1F2E", surface="#10303F", surface2="#164155", text="#E6F4F8",
        muted="#8DB4C2", border="#1E5068", primary="#2DD4BF", primary2="#38BDF8",
        on_primary="#06202B", chart="#2DD4BF",
    ),
    "🌿 Forest": dict(
        bg="#F1F6F0", surface="#FFFFFF", surface2="#E3EEE1", text="#1D2B1F",
        muted="#5F7562", border="#D2E0CF", primary="#2F9E5B", primary2="#A3C644",
        on_primary="#FFFFFF", chart="#2F9E5B",
    ),
}

if "theme" not in st.session_state:
    st.session_state.theme = "🌙 Dark"
if "history" not in st.session_state:
    st.session_state.history = []


def inject_css(t: dict) -> None:
    st.markdown(
        f"""
<style>
:root {{
  --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']};
  --text:{t['text']}; --muted:{t['muted']}; --border:{t['border']};
  --primary:{t['primary']}; --primary2:{t['primary2']}; --on-primary:{t['on_primary']};
}}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important; color: var(--text) !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
  background: var(--surface) !important; border-right: 1px solid var(--border);
}}
.stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp div[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {{
  color: var(--text);
}}
[data-testid="stCaptionContainer"], .muted {{ color: var(--muted) !important; }}

/* Hero */
.hero {{
  background: linear-gradient(135deg, var(--primary), var(--primary2));
  color: var(--on-primary); border-radius: 18px; padding: 28px 32px;
  margin-bottom: 22px; box-shadow: 0 10px 30px rgba(0,0,0,.18);
}}
.hero h1 {{ margin:0; font-size: 2.1rem; color: var(--on-primary) !important; }}
.hero p  {{ margin:6px 0 0; opacity:.92; font-size:1.02rem; color: var(--on-primary) !important; }}

/* Cards */
.card {{
  background: var(--surface); border:1px solid var(--border);
  border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;
}}
.pred-label {{ font-size: 2rem; font-weight: 800; color: var(--primary); line-height:1.1; }}
.pred-sub   {{ color: var(--muted); font-size: .9rem; margin-top: 4px; }}
.badge {{
  display:inline-block; padding: 3px 12px; border-radius: 999px; font-size:.78rem;
  font-weight:700; background: var(--surface2); border:1px solid var(--border);
  color: var(--text);
}}
.badge.ok   {{ background: rgba(47,158,91,.15); color:#2F9E5B; border-color: rgba(47,158,91,.4); }}
.badge.mid  {{ background: rgba(245,158,11,.15); color:#D98A06; border-color: rgba(245,158,11,.4); }}
.badge.low  {{ background: rgba(232,80,91,.15); color:#E8505B; border-color: rgba(232,80,91,.4); }}

.bar-row {{ margin: 10px 0; }}
.bar-top {{ display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:4px; }}
.bar-bg  {{ background: var(--surface2); border-radius: 999px; height: 10px; overflow:hidden; }}
.bar-fg  {{ background: linear-gradient(90deg, var(--primary), var(--primary2));
            height:100%; border-radius:999px; }}

/* Widgets */
.stButton > button, .stDownloadButton > button {{
  background: var(--primary); color: var(--on-primary); border: none;
  border-radius: 10px; font-weight: 700; padding: .55rem 1.1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  filter: brightness(1.08); color: var(--on-primary);
}}
[data-testid="stFileUploader"] section {{
  background: var(--surface); border: 2px dashed var(--border); border-radius: 14px;
}}
[data-testid="stFileUploader"] section * {{ color: var(--text) !important; }}
[data-testid="stFileUploader"] button {{ background: var(--surface2); color: var(--text); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ color: var(--muted); font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: var(--primary) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--primary) !important; }}
[data-testid="stMetric"] {{
  background: var(--surface); border:1px solid var(--border);
  border-radius: 14px; padding: 12px 16px;
}}
[data-testid="stMetricValue"] {{ color: var(--primary) !important; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {{
  background: var(--surface2) !important; color: var(--text) !important;
  border-color: var(--border) !important;
}}
[data-baseweb="select"] * {{ color: var(--text) !important; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:12px; }}
hr {{ border-color: var(--border) !important; }}
footer, #MainMenu {{ visibility: hidden; }}
</style>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Model (architecture copied exactly from the training notebook)
# --------------------------------------------------------------------------
class my_cnn(nn.Module):
    def __init__(self, num_feature, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_feature, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_cnn(3, len(class_names))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names, device


PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# The dataset contains near-duplicate folders (spelling variants). Optionally
# merge their probabilities so the ranking is not split between twins.
ALIASES = {
    "Black Berry": "Blackberry",
    "Chilli Peper": "Chilli",
    "Dragonfruit": "Dragon Fruit",
    "Raddish": "Radish",
    "Sweetpotato": "Sweet Potato",
    "Sweetcorn": "Corn",
    "Eggplant": "Brinjal",
    "Capsicum": "Bell Pepper",
    "Jalepeno": "Chilli",
}


@torch.no_grad()
def predict(image: Image.Image, merge_duplicates: bool):
    model, class_names, device = load_model()
    x = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    scores: dict[str, float] = {}
    for name, p in zip(class_names, probs):
        key = ALIASES.get(name, name) if merge_duplicates else name
        scores[key] = scores.get(key, 0.0) + p
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def conf_badge(conf: float, threshold: float) -> str:
    if conf >= max(threshold, 0.8):
        return '<span class="badge ok">High confidence</span>'
    if conf >= threshold:
        return '<span class="badge mid">Moderate confidence</span>'
    return '<span class="badge low">Low confidence</span>'


def render_bars(top: list[tuple[str, float]]) -> None:
    html = ""
    for name, p in top:
        html += (
            f'<div class="bar-row"><div class="bar-top"><span>{name}</span>'
            f'<span><b>{p*100:.1f}%</b></span></div>'
            f'<div class="bar-bg"><div class="bar-fg" style="width:{max(p*100,0.5):.1f}%"></div></div></div>'
        )
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    st.selectbox("Theme", list(THEMES.keys()), key="theme")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top predictions to show", 1, 10, 5)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Predictions below this are flagged as low confidence.")
    merge = st.toggle("Merge duplicate labels", value=True,
                      help="Combines spelling variants such as Black Berry / Blackberry "
                           "or Eggplant / Brinjal into one result.")
    st.markdown("---")
    st.caption("Custom CNN · 128×128 input · 88 classes")

theme = THEMES[st.session_state.theme]
inject_css(theme)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🍎 Fruit &amp; Vegetable Classifier</h1>
  <p>Upload a photo and a convolutional neural network will identify it from 88 fruit and vegetable classes.</p>
</div>
""",
    unsafe_allow_html=True,
)

try:
    model, class_names, device = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}")
    st.stop()

tab_single, tab_batch, tab_history, tab_about = st.tabs(
    ["🔍 Classify", "🗂️ Batch", "🕘 History", "ℹ️ About"]
)

# ----------------------------- Single image --------------------------------
with tab_single:
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True,
                   label_visibility="collapsed")
    if src == "Upload":
        file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"],
                                key="single_upload")
    else:
        file = st.camera_input("Take a photo")

    if file is None:
        st.info("Upload or capture an image to get a prediction.")
    else:
        image = Image.open(file).convert("RGB")
        results = predict(image, merge)
        best, conf = results[0]

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.image(image, use_container_width=True)
        with right:
            st.markdown(
                f'<div class="card"><div class="muted" style="font-size:.85rem">PREDICTION</div>'
                f'<div class="pred-label">{best}</div>'
                f'<div class="pred-sub">{conf*100:.2f}% confidence &nbsp; {conf_badge(conf, threshold)}</div></div>',
                unsafe_allow_html=True,
            )
            if conf < threshold:
                st.warning("The model is unsure about this image. Try a clearer, "
                           "well-lit photo with the item centered.")
            st.markdown(f"**Top {top_k} predictions**")
            render_bars(results[:top_k])

        name = getattr(file, "name", "camera_capture.jpg")
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": name, "Prediction": best, "Confidence (%)": round(conf * 100, 2),
        })
        # avoid logging the same file on every rerun
        h = st.session_state.history
        if len(h) > 1 and h[-1]["File"] == h[-2]["File"] and h[-1]["Prediction"] == h[-2]["Prediction"]:
            h.pop()

# ------------------------------- Batch -------------------------------------
with tab_batch:
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="batch_upload")
    if files:
        rows, cols = [], st.columns(4)
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            res = predict(img, merge)
            label, p = res[0]
            rows.append({"File": f.name, "Prediction": label,
                         "Confidence (%)": round(p * 100, 2),
                         "Runner-up": res[1][0] if len(res) > 1 else ""})
            with cols[i % 4]:
                st.image(img, use_container_width=True)
                st.caption(f"**{label}** · {p*100:.1f}%")
            bar.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(df))
        c2.metric("Avg confidence", f"{df['Confidence (%)'].mean():.1f}%")
        c3.metric("Low confidence", int((df["Confidence (%)"] < threshold * 100).sum()))
        st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
    else:
        st.info("Upload several images to classify them all at once.")

# ------------------------------- History -----------------------------------
with tab_history:
    if st.session_state.history:
        hdf = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Export history (CSV)", hdf.to_csv(index=False).encode(),
                           "history.csv", "text/csv")
        if c2.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Your single-image predictions will appear here during this session.")

# -------------------------------- About ------------------------------------
with tab_about:
    n_params = sum(p.numel() for p in model.parameters())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", len(class_names))
    c2.metric("Parameters", f"{n_params/1e6:.2f}M")
    c3.metric("Input size", f"{IMG_SIZE}×{IMG_SIZE}")
    c4.metric("Device", str(device).upper())

    st.markdown(
        """
<div class="card">
**Fruit & Vegetable Classifier** is a deep learning-based image classification application built using **PyTorch and Streamlit**.
The application uses a custom **Convolutional Neural Network (CNN)** to identify images across **88 fruit and vegetable classes**. Users can upload an image or capture one using their camera and receive the predicted class along with its confidence score.
The application also supports **batch image classification, top predictions, confidence analysis, prediction history, CSV export, and customizable themes**.
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander(f"All {len(class_names)} supported classes"):
        st.write(" · ".join(class_names))
