FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the FastAPI app code
COPY app /code/app

# Expose port (7860 or 8000 are typical for Hugging Face Spaces)
EXPOSE 7860

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
