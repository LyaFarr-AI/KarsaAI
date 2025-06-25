from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.utils.generate_poem import generate_poem, load_resources_poem
from app.utils.generate_pantun import generate_pantun, load_resources_pantun
from app.utils.preprocessing import preprocess_text
import os

# ================== Load Model Poem ==================
poem_model, poem_word2idx, poem_idx2word = load_resources_poem(
    model_path="app/model/poem_lstm.pth",
    poem_word2idx_path="app/model/poem_word2idx.pkl",
    poem_idx2word_path="app/model/poem_idx2word.pkl"
)

# ================== Load Model Pantun ==================
pantun_model, pantun_word2idx, pantun_idx2word = load_resources_pantun(
    model_path="app/model/pantun_lstm.pth",
    pantun_word2idx_path="app/model/pantun_word2idx.pkl",
    pantun_idx2word_path = "app/model/pantun_idx2word.pkl"
    )

app = FastAPI()
@app.get("/")
def home():
    return {"message": "KarsaAI backend is running."}

class Prompt(BaseModel):
    prompt: str

#Poem
@app.post("/generate-poem")
def puisi_api(data: Prompt):
    cleaned = preprocess_text(data.prompt)
    result = generate_poem(poem_model, cleaned, poem_word2idx, poem_idx2word)
    return {"result": result}

#Pantun
#@app.post("/generate-pantun")
#def pantun_api(data: Prompt):
    #cleaned = preprocess_text(data.prompt)
    #result = generate_pantun(pantun_model, cleaned, pantun_word2idx, pantun_idx2word)
    #return {"result": result}

@app.post("/generate-pantun")
def puisi_api(data: Prompt):
    try:
        cleaned = preprocess_text(data.prompt)
        result = generate_poem(poem_model, cleaned, poem_word2idx, poem_idx2word)
        return {"result": result}
        #return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}