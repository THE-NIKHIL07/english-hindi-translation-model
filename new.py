from tensorflow.keras.layers import TextVectorization

MAX_TOKENS = 25000
SEQ_LENGTH = 40

def load_vectorizers():
    # --- Source Vectorizer ---
    with open("source_vocab.txt", "r", encoding="utf-8") as f:
        source_vocab = [line.strip() for line in f.readlines()]
    source_vocab = list(dict.fromkeys(source_vocab))  # remove duplicates while keeping order

    source_vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH
    )
    source_vectorizer.set_vocabulary(source_vocab)

    # --- Target Vectorizer ---
    with open("target_vocab.txt", "r", encoding="utf-8") as f:
        target_vocab = [line.strip() for line in f.readlines()]
    target_vocab = list(dict.fromkeys(target_vocab))  # remove duplicates

    target_vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH
    )
    target_vectorizer.set_vocabulary(target_vocab)

    return source_vectorizer, target_vectorizer


source_vectorizer, target_vectorizer = load_vectorizers()