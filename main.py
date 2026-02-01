import uvicorn
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from new import load_vectorizers

VOCAB_SIZE = 25000
ENGLISH_SEQUENCE_LENGTH = 40
HINDI_SEQUENCE_LENGTH = 40

app = FastAPI(title="English → Hindi Translator")

app.mount("/static", StaticFiles(directory="static"), name="static")

class TranslateRequest(BaseModel):
    text: str

source_vectorizer = None
target_vectorizer = None
transformer = None

def load_model():
    global source_vectorizer, target_vectorizer, transformer
    source_vectorizer, target_vectorizer = load_vectorizers()
    transformer = tf.keras.models.load_model(
        "transformer_model.keras",
        compile=False
    )

load_model()

def translator(sentence: str):
    sentence = " ".join(sentence.strip().split())
    words = sentence.split()
    truncated_note = ""

    if len(words) > ENGLISH_SEQUENCE_LENGTH:
        sentence = " ".join(words[:ENGLISH_SEQUENCE_LENGTH])
        truncated_note = f"Input truncated to {ENGLISH_SEQUENCE_LENGTH} tokens."

    index_to_word = {
        i: word for i, word in enumerate(target_vectorizer.get_vocabulary())
    }

    src_tokens = source_vectorizer(tf.constant([sentence]))
    shifted_target = ["starttoken"]
    output = []

    for _ in range(HINDI_SEQUENCE_LENGTH):
        tgt_text = " ".join(shifted_target)
        tgt_tokens = target_vectorizer(tf.constant([tgt_text]))
        logits = transformer([src_tokens, tgt_tokens], training=False)
        next_id = tf.argmax(logits[0, len(shifted_target) - 1]).numpy()
        next_word = index_to_word.get(next_id, "[UNK]")

        if next_word in ("endtoken", "[UNK]"):
            break

        output.append(next_word)
        shifted_target.append(next_word)

    return " ".join(output), truncated_note

@app.post("/translate")
async def translate(request: TranslateRequest):
    if not request.text.strip():
        return {"translation": "", "note": "No input provided."}

    translation, note = translator(request.text)
    return {"translation": translation, "note": note}

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
