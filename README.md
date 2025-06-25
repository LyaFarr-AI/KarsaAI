# 📘 KarsaAI - Indonesian Generative AI for Poetry, Pantun, and Chatbot

**KarsaAI** is a generative AI application built to create Indonesian literary works such as:

- 📝 **Poetry** (Puisi)
- 🎭 **Pantun** 
- 🤖 **Conversational responses** (Chatbot)

Users can input a theme or keyword, and the app generates meaningful text based on it. KarsaAI combines **Natural Language Processing (NLP)** techniques with custom-trained models to support creative expression in the Indonesian language.

---

## 🚀 Features

- **🎛️ Model Selection**  
  Choose between two types of models:
  - `Quality`: Fine-tuned transformer model `Eunoia-Gemma-9B-Q5_K_M`
  - `Fast`: Lightweight BiLSTM RNN models for faster generation

- **🌐 Full Indonesian Support**  
  All text generation is optimized for Bahasa Indonesia

- **🧠 AI Chatbot**  
  A simple chatbot powered by the same Gemma-based language model, fine-tuned for Indonesian dialogue

- **🎨 Creative Output**  
  Ideal for fun, inspiration, or educational use

---

## 🏗️ Technologies Used

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) — for frontend UI
- [FastAPI](https://fastapi.tiangolo.com/) — for backend model inference
- [PyTorch](https://pytorch.org/) — for custom LSTM models
- [Transformers (Hugging Face)](https://huggingface.co/) — for Gemma-based LLMs
- Hugging Face Spaces — to deploy the backend API securely

---

## 📂 Project Structure

```bash
.
├── app.py                 # Streamlit frontend
├── backend/
│   ├── main.py           # FastAPI entrypoint
│   ├── model_poem/       # LSTM model & resources for poems
│   └── model_pantun/     # LSTM model & resources for pantun
├── utils/
│   ├── generate_poem.py
│   ├── generate_pantun.py
│   └── preprocessing.py
|
├── data/
│   ├── pantun.csv
│   ├── pantun1.csv
│   └── preprocessing.py
