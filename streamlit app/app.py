import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


st.set_page_config(
    page_title="Flickr8k Image Captioning",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(135deg, #E6F0FF, #FFFFFF);
    padding: 35px;
    border-radius: 24px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    color: #1E3A8A;
    margin-bottom: 10px;
}
.hero p {
    font-size: 18px;
    color: #4B5563;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.07);
}
.caption-box {
    background-color: #E6F0FF;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #1E3A8A;
    margin-top: 20px;
}
.caption-text {
    font-size: 26px;
    font-weight: 700;
    color: #1E3A8A;
}
.small-text {
    color: #6B7280;
    font-size: 14px;
}
.stButton > button {
    background-color: #1E3A8A;
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #FF6B6B;
    color: white;
}
</style>
""", unsafe_allow_html=True)


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "resnet50_lstm_captioner.pt"
VOCAB_PATH = ARTIFACT_DIR / "vocab.json"


class CNNLSTMCaptioner(nn.Module):
    def __init__(self, vocab_size, feature_dim, embed_dim=256, hidden_dim=512, pad_idx=0, dropout=0.3):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        caption_input = captions[:, :-1]
        img_embed = self.dropout(self.feature_proj(features)).unsqueeze(1)
        word_embed = self.dropout(self.embedding(caption_input))
        lstm_input = torch.cat([img_embed, word_embed], dim=1)
        output, _ = self.lstm(lstm_input)
        output = self.dropout(output)
        return self.fc(output)


@st.cache_resource
def load_feature_extractor():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    extractor = nn.Sequential(*list(model.children())[:-1])
    extractor.eval()
    return extractor


@st.cache_resource
def load_caption_model():
    with open(VOCAB_PATH, "r") as f:
        vocab_data = json.load(f)

    stoi = vocab_data["stoi"]
    itos = vocab_data["itos"]

    pad_idx = stoi["<pad>"]

    model = CNNLSTMCaptioner(
        vocab_size=len(itos),
        feature_dim=2048,
        pad_idx=pad_idx
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, stoi, itos


model, stoi, itos = load_caption_model()
feature_extractor = load_feature_extractor()

START_IDX = stoi["<start>"]
END_IDX = stoi["<end>"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def clean_output(words):
    bad_words = ["<unk>", "<start>", "<end>", "<pad>"]
    cleaned = []

    for word in words:
        if word in bad_words:
            continue

        if len(cleaned) > 0 and cleaned[-1] == word:
            continue

        cleaned.append(word)

    caption = " ".join(cleaned)

    caption = caption.replace(" .", ".")
    caption = caption.replace(" ,", ",")
    caption = caption.strip()

    if len(caption) > 0:
        caption = caption[0].upper() + caption[1:]

    return caption


def generate_caption(image, max_len=20):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)

    caption = [START_IDX]

    for _ in range(max_len):
        caption_tensor = torch.tensor([caption], dtype=torch.long)

        with torch.no_grad():
            outputs = model(features, caption_tensor)

        next_token = outputs[0, -1].argmax().item()
        caption.append(next_token)

        if next_token == END_IDX:
            break

    words = [itos[idx] for idx in caption]
    caption = clean_output(words)
    if caption.endswith((" a", " of", " in", " on", " at", " and", "with")):
        caption = " ".join(caption.split()[:-1])

    return caption


st.markdown("""
<div class="hero">
    <h1>🖼️ Flickr8k Image Captioning</h1>
    <p>Upload an image and generate a caption using an improved ResNet50 + LSTM deep learning model.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Task:** Image Captioning")
    st.write("**Encoder:** ResNet50")
    st.write("**Decoder:** LSTM")
    st.write("**Dataset:** Flickr8k")
    st.info("This is a student deep learning project. Captions may be simple because the model was trained on a small dataset.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Model Output")
    st.write("After uploading an image, click the button to generate a caption.")
    st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Uploaded Image")
        st.image(image, width=450)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Generated Caption")

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image)

            st.markdown(f"""
            <div class="caption-box">
                <div class="caption-text">"{caption}"</div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Caption generated successfully!")

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Please upload an image to start.")