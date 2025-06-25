from fastapi import FastAPI, Request
from llama_cpp import Llama

app = FastAPI()

llm = Llama(
    model_path="model/Eunoia-Gemma-9B-o1-Indo.i1-Q4_K_M.gguf",
    n_ctx=1536,
    n_threads=4,
    n_gpu_layers=0,  # CPU-only
    verbose=False
)

@app.post("/chatbot")
async def chatbot_endpoint(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")

    full_prompt = f"[INST] {user_prompt} [/INST]"

    response = llm(full_prompt, max_tokens=256, stop=["</s>"])
    result = response["choices"][0]["text"].strip()

    return {"result": result}
