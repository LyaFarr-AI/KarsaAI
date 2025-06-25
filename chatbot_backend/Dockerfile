FROM python:3.10

# Install dependensi sistem (pastikan cepat & ringan)
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    build-essential \
    cmake \
    curl \
    unzip \
    git

WORKDIR /code

# Download model
RUN mkdir -p /code/model && \
    curl -L -o /code/model/Eunoia-Gemma-9B-o1-Indo.i1-Q4_K_M.gguf \
    https://huggingface.co/mradermacher/Eunoia-Gemma-9B-o1-Indo-i1-GGUF/resolve/main/Eunoia-Gemma-9B-o1-Indo.i1-Q4_K_M.gguf


# Salin requirements (tanpa llama-cpp-python dulu)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn python-multipart

# Install llama-cpp-python from repo GitHub

RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git && \
    cd llama-cpp-python && \
    pip install .


# Copy app
COPY app /code/app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
