# app.py
import streamlit as st
import requests

API_URL_CB = "https://lyafarr-ai-chatbot-backend.hf.space/chatbot"


#def generate_poem_remote(prompt):
#    response = requests.post(f"{API_URL_CB}/generate-puisi", json={"prompt": prompt})
#    return response.json().get("result", "❌ Terjadi kesalahan.")
#
#def generate_pantun_remote(prompt):
#    response = requests.post(f"{API_URL_CB}/generate-pantun", json={"prompt": prompt})
#    return response.json().get("result", "❌ Terjadi kesalahan.")

def chatbot_remote(prompt):
    response = requests.post(f"{API_URL_CB}/chatbot", json={"prompt": prompt})
    return response.json().get("result", "❌ Terjadi kesalahan.")

def generate_poem_remote(prompt):
    response = requests.post(f"{API_URL_CB}/chatbot", json={"prompt": f"Tolong buatkan puisi bertema: {prompt}"})
    return response.json().get("result", "Terjadi kesalahan.")

def generate_pantun_remote(prompt):
    response = requests.post(f"{API_URL_CB}/chatbot", json={"prompt": f"Tolong buatkan pantun dengan input {prompt}"})
    return response.json().get("result", "Terjadi kesalahan.")


st.set_page_config(page_title="AI Poet Generator", page_icon="🧠", layout="centered")
st.title("📝 AI Generator: Puisi, Pantun, & Chatbot")

menu = st.sidebar.radio("Pilih Fitur", ["Beranda","Puisi", "Pantun", "Chatbot", "Tentang"])

if "history" not in st.session_state:
    st.session_state["history"] = []


# ========================= BERANDA =========================
if menu == "Beranda":
    st.markdown(
        """
        <div style='text-align: center; padding: 30px 0;'>
            <h1 style='font-size: 48px; color: #4CAF50; margin-bottom: 0;'>KarsaAI</h1>
            <p style='font-size: 20px; color: #555;'>Sastra Digital • Bahasa Indonesia • Kecerdasan Buatan</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align: center; font-size: 17px; line-height: 1.8;'>
            <p>
                <b>KarsaAI</b> adalah aplikasi AI generatif yang dapat menciptakan karya sastra Indonesia seperti
                <span style='color:#4CAF50;'><b>puisi</b></span>, 
                <span style='color:#4CAF50;'><b>pantun</b></span>, 
                dan <span style='color:#4CAF50;'><b>respon percakapan cerdas</b></span> berdasarkan tema yang Anda berikan.
            </p>
            <p>
                Dirancang dengan teknologi pemrosesan bahasa alami (NLP) dan model AI lokal, KarsaAI membantu Anda menuangkan inspirasi menjadi teks yang bermakna.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🚀 Mulai Menggunakan")
    st.markdown(
        """
        - Gunakan menu di samping untuk mulai membuat **puisi** atau **pantun** berdasarkan tema pilihan Anda.
        - Untuk percakapan interaktif, kunjungi menu **Chatbot**.
        - Semua hasil bisa Anda salin dan simpan sesuai keperluan.
        """
    )

    st.markdown("### Note !")
    st.markdown(
        """
        - AI Chatbot masih dalam tahap pengembangan, jadi mungkin belum sempurna dan juga output balasannya sangat lama (Dikarenakan backend menggunakan cpu).
        - AI Puisi/Pantun hasil puisi dan pantun bersifat generatif, jadi bisa bervariasi setiap kali Anda mencoba dan juga masih dalam tahap pengembangan.
        - Untuk feedback atau saran, silakan hubungi pengembang melalui email yang tersedia di menu **Tentang**.
        - Aplikasi ini masih dalam versi beta, jadi mohon maaf jika ada kendala.
        - Aplikasi ini dibuat untuk pembelajaran dan eksplorasi AI.

        """
    )

    st.markdown("---")

    st.markdown(
        "<div style='text-align: center; font-size: 14px; color: #888;'>Versi Beta • Dibuat oleh Rafael Robert</div>",
        unsafe_allow_html=True
    )


# ========================= PUISI =========================
if menu == "Puisi":
    st.header("📜 Generator Puisi")
    seed = st.text_input("Masukkan tema / kata awal:")
    generate = st.button("Generate Puisi")

    if generate and seed:
        with st.spinner("🔄 Sedang membuat puisi..."):
            result = generate_poem_remote(seed)
        st.subheader("📝 Hasil Puisi")
        st.markdown(result)
        st.session_state["history"].append(("Puisi", seed, result))

# ========================= PANTUN =========================
elif menu == "Pantun":
    st.header("🎭 Generator Pantun")
    jenis = st.selectbox("Pilih jenis pantun", [
        "Pantun Cinta", "Pantun Jenaka", "Pantun Pendidikan", "Pantun Agama",
        "Pantun Anak-anak", "Pantun Budi", "Pantun Adat dan Alam", "Pantun Teka-Teki"
    ])
    tema = st.text_input("Masukkan tema / kata awal:")
    generate = st.button("Generate Pantun")

    if generate and tema:
        full_prompt = f"{jenis} - {tema}"
        with st.spinner("🌀 Sedang membuat pantun..."):
            result = generate_pantun_remote(full_prompt)
        st.subheader("🎭 Hasil Pantun")
        st.markdown(result)
        st.session_state["history"].append(("Pantun", full_prompt, result))

# ========================= CHATBOT =========================
elif menu == "Chatbot":
    st.header("🤖 Chatbot AI Sederhana")
    user_input = st.text_input("Tanyakan sesuatu:")
    ask = st.button("Kirim")

    if ask and user_input:
        with st.spinner("💬 Bot sedang mengetik..."):
            result = chatbot_remote(user_input)
        st.markdown(f"**Bot:** {result}")
        st.session_state["history"].append(("Chatbot", user_input, result))

# ========================= TENTANG =========================
elif menu == "Tentang":
    st.header("📘 Tentang Aplikasi KarsaAI")
    st.markdown(
        """
        <div style='font-size: 17px; line-height: 1.6;'>
        <p><b>KarsaAI</b> adalah aplikasi AI generatif berbahasa Indonesia yang dirancang untuk membuat karya sastra tematik dan interaktif. Dengan menggabungkan model deep learning dan natural language processing (NLP), KarsaAI mampu menghasilkan:</p>

        - 📝 <b>Puisi pendek</b> berdasarkan tema yang Anda tentukan  
        - 🎭 <b>Pantun</b> dari berbagai kategori dan gaya  
        - 🤖 <b>Chatbot sederhana</b> dengan model Eunoia-Gemma-9B-o1-Indo-i1.  

        <p>Aplikasi ini dibangun menggunakan teknologi:</p>
        <ul>
            <li><code>PyTorch</code></li>
            <li><code>Streamlit</code></li>
            <li><code>Transformers (Hugging Face)</code></li>
            <li><code>LSTM</code> untuk puisi & pantun, dan <code>GPT-2</code> untuk chatbot</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("🕘 Riwayat Generate Terakhir")
    if not st.session_state.get("history"):
        st.info("Belum ada riwayat generate.")
    else:
        for idx, (fitur, tema, hasil) in enumerate(reversed(st.session_state["history"][-10:]), 1):
            st.markdown(f"**{idx}. [{fitur}]** `{tema}` → _{hasil[:80]}..._")

    st.markdown("---")
    st.markdown(
        """
        📩 Untuk pelaporan bug atau feedback, silakan hubungi:  
        **📧 [lyafarr@gmail.com](mailto:lyafarr@gmail.com)**
        """)


# ========================= FOOTER =========================
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer-custom {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 0.9rem;
        color: #333;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        z-index: 100;
    }
    </style>
    <div class="footer-custom">
        🚀 Created by: <b>Rafael R.T</b> | 💡 Powered by Streamlit & FastAPI
    </div>
""", unsafe_allow_html=True)
