import torch
import torch.nn.functional as F
import pickle
import os

from app.model.lstm_poem import PoemLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== LOAD MODEL & DICTIONARIES ==========

def load_resources_poem(model_path, poem_word2idx_path, poem_idx2word_path, device="cpu"):
    with open(poem_word2idx_path, "rb") as f:
        word2idx = pickle.load(f)
    with open(poem_idx2word_path, "rb") as f:
        idx2word = pickle.load(f)

    vocab_size = len(word2idx)
    model = PoemLSTM(vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, word2idx, idx2word


# ========== GENERATE POEM ==========
def generate_poem(model, seed_text, word2idx, idx2word, max_len=50, device="cpu", temperature=1.0):
    model.eval()
    UNK_INDEX = 0
    words = seed_text.lower().split()
    input_seq = [word2idx.get(word, UNK_INDEX) for word in words]
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    generated = input_seq.copy()

    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor)
            
            if isinstance(output, tuple):
                output = output[0]

            last_token_logits = output[0] / temperature
            probs = F.softmax(last_token_logits, dim=-1)

            next_token = torch.multinomial(probs, 1).item()

            if next_token == word2idx.get("<eos>", -1):
                break

            generated.append(next_token)
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)

    # Ambil kata dari indeks dan hilangkan semua '<unk>'
    result = [idx2word.get(idx, "<unk>") for idx in generated]
    result = [word for word in result if word != "<unk>"]

    return " ".join(result)