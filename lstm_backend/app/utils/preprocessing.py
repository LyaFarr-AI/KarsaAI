import re
import json
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")


# ========== LOAD RESOURCES ==========

# KBBI
#with open(os.path.join(DATA_DIR, "kbbi.txt"), "r", encoding="utf-8") as f:
#    KBBI = set(word.strip().lower() for word in f if word.strip())

# Stopwords
with open(os.path.join(DATA_DIR, "stopwords.txt"), "r", encoding="utf-8") as f:
    STOPWORDS = set(word.strip().lower() for word in f if word.strip())

# Slangwords: CSV
slang_df = pd.read_csv(os.path.join(DATA_DIR, "slang_indo.csv"))
slang_df = slang_df.dropna()
slang_df["slang"] = slang_df["slang"].str.lower().str.strip()
slang_df["formal"] = slang_df["formal"].str.lower().str.strip()
slang_dict_csv = dict(zip(slang_df["slang"], slang_df["formal"]))

# Slangwords: JSON
with open(os.path.join(DATA_DIR, "slangwords.txt"), "r", encoding="utf-8") as f:
    slang_dict_txt = json.load(f)

SLANG_DICT = {**slang_dict_csv, **slang_dict_txt}

# Suku kata: JSON
with open(os.path.join(DATA_DIR, "suku-kata-id.json"), "r", encoding="utf-8") as f:
    SUKU_KATA_DICT = json.load(f)


# ========== CLEANING & NORMALIZATION ==========

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang(text):
    words = text.split()
    return " ".join([SLANG_DICT.get(w, w) for w in words])

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in STOPWORDS])

#def keep_kbbi_only(text):
    return " ".join([w for w in text.split() if w in KBBI])


# ========== SUKU KATA ==========

def hitung_suku_kata(kalimat):
    kata_list = kalimat.lower().split()
    total = 0
    for kata in kata_list:
        kata = kata.lower()
        if kata in SUKU_KATA_DICT:
            total += len(SUKU_KATA_DICT[kata])
        else:
            total += len([c for c in kata if c in "aiueo"])
    return total


# ========== Preprocess Text ==========

def preprocess_text(text, mode="puisi"):
    if not text or not isinstance(text, str):
        return ""

    text = clean_text(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    #text = keep_kbbi_only(text) <---- I will not use this for now

    #if mode == "pantun":
        #return text, hitung_suku_kata(text)
    return text