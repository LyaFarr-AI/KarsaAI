import torch
from torch import nn
import torch.nn.functional as F
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PantunLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    


# ====== LOAD RESOURCES ======

def load_resources_pantun(model_path, pantun_word2idx_path, pantun_idx2word_path, embed_dim=128, hidden_dim=256, device='cpu'):
    with open(pantun_word2idx_path, "rb") as f:
        word2idx = pickle.load(f)
    with open(pantun_idx2word_path, "rb") as f:
        idx2word = pickle.load(f)

    vocab_size = len(word2idx)
    model = PantunLSTM(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, word2idx, idx2word

# ====== GENERATE PANTUN ======
def generate_pantun(model, seed_text, word2idx, idx2word, max_words=100, target_baris=4):
    model.eval()
    words = seed_text.lower().split()
    br_count = 0
    hasil = []

    for _ in range(max_words):
        token_ids = [word2idx.get(w, word2idx['<UNK>']) for w in words]
        padded = [0] * (max_seq_len - 1 - len(token_ids)) + token_ids
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=-1)
            #predicted_idx = torch.argmax(probs, dim=1).item()
            predicted_idx = torch.multinomial(probs, num_samples=1).item()

        next_word = idx2word.get(predicted_idx, '<UNK>')
        words.append(next_word)

        if next_word == '<br>':
            br_count += 1
            if br_count >= target_baris:
                break

    # Ambil sampai 4 baris saja
    full_text = ' '.join(words)
    baris = [b.strip() for b in full_text.split('<br>') if b.strip()]
    baris = baris[:4]  # potong jadi 4 baris
    return '\n'.join(baris)