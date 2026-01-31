import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from utils import build_transformer, load_vectorizers

VOCAB_SIZE = 25000
ENGLISH_SEQUENCE_LENGTH = 40
HINDI_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 256
NUM_HEADS = 4
LATENT_DIM = 2048
NUM_LAYERS = 2

st.set_page_config(page_title="English → Hindi Translator", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp, .css-18e3th9 {
        background-color: white !important;
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    .stTextArea>div>textarea, .stTextInput>div>input {
        background-color: #f5f5f5 !important;
        color: black !important;
    }
    .stButton>button {
        background-color: black !important;
        color: red !important;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: black;
        font-weight: bold;
        border-top: 2px solid black;
        padding-top: 8px;
        margin-top: 20px;
    }
    .note-text {
        color: black;
        font-style: normal;
        font-size: 14px;
    }
    .translation-output {
        color: black !important;
        font-size: 18px;
        font-weight: 500;
        padding: 10px;
        border-left: 5px solid black;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

source_vectorizer, target_vectorizer = load_vectorizers()
transformer = build_transformer()
transformer.load_weights("translator_transformer.weights.h5")

def translator(sentence):
    index_to_word = {i: word for i, word in enumerate(target_vectorizer.get_vocabulary())}
    src_tokens = source_vectorizer(tf.constant([sentence]))
    shifted_target = ["starttoken"]
    output = []

    for i in range(HINDI_SEQUENCE_LENGTH):
        tgt_tokens = target_vectorizer(tf.constant([" ".join(shifted_target)]))
        logits = transformer([src_tokens, tgt_tokens], training=False)
        next_id = tf.argmax(logits[0, i, :]).numpy()
        next_word = index_to_word.get(next_id, "[UNK]")
        if next_word in ["endtoken", "[UNK]"]:
            break
        output.append(next_word)
        shifted_target.append(next_word)

    return " ".join(output)

st.title("English → Hindi Translator")
sentence = st.text_area("Enter English text here:", height=150)

if st.button("Translate"):
    if sentence.strip() != "":
        with st.spinner("Translating..."):
            translation = translator(sentence)
        st.markdown(f'<p class="translation-output">{translation}</p>', unsafe_allow_html=True)
        st.markdown('<p class="note-text">Note: Translation might not always work accurately.</p>', unsafe_allow_html=True)

st.markdown("---")
with st.expander("Model Parameters & Preprocessing Info"):
    st.markdown(f"""
        <p class="note-text"><b>Transformer Model:</b> 30 million parameters</p>
        <p class="note-text"><b>Encoder & Decoder Layers:</b> {NUM_LAYERS}</p>
        <p class="note-text"><b>Embedding Dimension:</b> {EMBEDDING_DIM}</p>
        <p class="note-text"><b>Number of Attention Heads:</b> {NUM_HEADS}</p>
        <p class="note-text"><b>English Sequence Length:</b> {ENGLISH_SEQUENCE_LENGTH}</p>
        <p class="note-text"><b>Hindi Sequence Length:</b> {HINDI_SEQUENCE_LENGTH}</p>
        <p class="note-text"><b>Vocabulary Size:</b> {VOCAB_SIZE}</p>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>MADE BY THE-NIKHIL07 © 2026</div>", unsafe_allow_html=True)