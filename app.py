# app.py
import streamlit as st
import torch
import json
import os
import random

# --- Base folder for all resources ---
BASE_DIR = os.path.dirname(__file__)  # folder where app.py is located

# --- Load vocab ---
@st.cache_data
def load_vocab(stoi_file="stoi.json", itos_file="itos.json"):
    stoi_path = os.path.join(BASE_DIR, stoi_file)
    itos_path = os.path.join(BASE_DIR, itos_file)

    if not os.path.exists(stoi_path) or not os.path.exists(itos_path):
        st.error("Vocabulary files not found. Please make sure 'stoi.json' and 'itos.json' are in the folder.")
        st.stop()

    with open(stoi_path) as f:
        stoi = json.load(f)
        if isinstance(stoi, list):
            stoi = {w: i for i, w in enumerate(stoi)}

    with open(itos_path) as f:
        itos = json.load(f)

    return stoi, itos


stoi, itos = load_vocab()
vocab_size = len(stoi)

# --- Define model ---
class MLPTextGen(torch.nn.Module):
    def __init__(self, vocab, ctx=5, emb=64, hid=[1024, 1024], act='relu', drop=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, emb)  # renamed from self.emb to self.embedding
        fn = torch.nn.ReLU if act == 'relu' else torch.nn.Tanh
        layers, inp = [], emb * ctx
        for h in hid:
            layers += [torch.nn.Linear(inp, h), fn(), torch.nn.Dropout(drop)]
            inp = h
        layers.append(torch.nn.Linear(inp, vocab))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)  # also update here
        return self.net(x)


# --- Load model ---
@st.cache_resource
def load_model(model_file, vocab, ctx=5, emb=64, act='relu'):
    model_path = os.path.join(BASE_DIR, model_file)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_file}")
        st.stop()
    model = MLPTextGen(vocab, ctx, emb, [1024, 1024], act)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# --- Prepare input tensor ---
def prep_input(text, block):
    words = text.lower().strip().split()
    idx = [stoi.get(w, stoi.get('.', 0)) for w in words][-block:]
    pad = [0] * (block - len(idx)) + idx
    return torch.tensor([pad], dtype=torch.long)

# --- Predict next words ---
def predict(model, text, block, k=5, temp=1.0, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    ctx = prep_input(text, block)
    out = []
    for _ in range(k):
        with torch.no_grad():
            p = torch.softmax(model(ctx)[0] / temp, dim=0)
            i = torch.multinomial(p, 1).item()
        out.append(itos.get(str(i), '?'))
        ctx = torch.tensor([[*ctx[0][1:], i]], dtype=torch.long)
    return ' '.join(out)

# --- Streamlit UI ---
st.title("Next Word Generator")

# Model selection and settings
model_options = {"Sherlock": "model_sherlock.pt", "Linux": "mlp_cpp_dataset.pt"}
model_opt = st.selectbox("Select Model:", list(model_options.keys()))
act = st.selectbox("Activation:", ["relu", "tanh"])
temp = st.slider("Temperature:", 0.1, 2.0, 1.0)
k = st.number_input("Number of words to predict:", 1, 20, 5)
seed = st.number_input("Random seed:", 0, 9999, 42)

# Load model
model_file = model_options[model_opt]
model = load_model(model_file, vocab_size, ctx=5, emb=64, act=act)

# Input text + generate
text = st.text_input("Enter your input text:")
if st.button("Generate"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        generated_text = predict(model, text, block=5, k=k, temp=temp, seed=seed)
        st.write("### Generated text:")
        st.success(f"{text} {generated_text}")
