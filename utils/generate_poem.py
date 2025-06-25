import torch
import torch.nn.functional as F
import pickle
import os

from app.model_poem.lstm_poem import PoemLSTM

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
def generate_poem(model, seed_text, word2idx, idx2word, max_len=50, temperature=1.0, top_k=50):
    model.eval()
    words = seed_text.lower().split()
    input_seq = [word2idx.get(w, word2idx['<unk>']) for w in words]

    unk_id = word2idx.get('<unk>')
    pad_id = word2idx.get('<pad>')

    for _ in range(max_len):
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze() / temperature
            probs = F.softmax(output, dim=-1).cpu()

            # Top-k sampling
            top_probs, top_idx = torch.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()  # re-normalisasi
            predicted_id = top_idx[torch.multinomial(top_probs, 1).item()].item()

        # Hindari pad dan unk di output
        if predicted_id in [pad_id, unk_id]:
            continue

        predicted_word = idx2word.get(predicted_id, "<unk>")
        words.append(predicted_word)
        input_seq.append(predicted_id)

        # Batasi panjang input agar tetap efisien
        if len(input_seq) > 30:
            input_seq = input_seq[-30:]

    return ' '.join(words)