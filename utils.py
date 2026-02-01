import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from src.transformer import Embeddings, TransformerEncoder, TransformerDecoder
import os

VOCAB_SIZE = 25000
ENGLISH_SEQUENCE_LENGTH = 40
HINDI_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 256
NUM_HEADS = 4
LATENT_DIM = 2048
NUM_LAYERS = 2

def build_transformer():
    encoder_inputs = Input(shape=(None,), dtype="int64", name="input_1")
    x = Embeddings(ENGLISH_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
    for _ in range(NUM_LAYERS):
        x = TransformerEncoder(EMBEDDING_DIM, LATENT_DIM, NUM_HEADS)(x)
    encoder_outputs = x
    
    decoder_inputs = Input(shape=(None,), dtype="int64", name="input_2")
    x = Embeddings(HINDI_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
    for _ in range(NUM_LAYERS):
        x = TransformerDecoder(EMBEDDING_DIM, LATENT_DIM, NUM_HEADS)(x, encoder_outputs)
    
    x = Dropout(0.5)(x)
    decoder_outputs = Dense(VOCAB_SIZE, activation="softmax")(x)
    transformer = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return transformer

def load_vectorizers():
    source_vectorizer_model = load_model("source_vectorizer_model.keras")
    source_layer = source_vectorizer_model.layers[1]

    target_vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=HINDI_SEQUENCE_LENGTH
    )

    with open("target_vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    seen = set()
    unique_vocab = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            unique_vocab.append(w)

    target_vectorizer.set_vocabulary(unique_vocab)

    return source_layer, target_vectorizer