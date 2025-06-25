import requests

API_URL = "https://lyafarr-ai-chatbot-backend.hf.space/chatbot"
data = {"prompt": "Apa itu kecerdasan buatan?"}

response = requests.post(API_URL, json=data)

print(response.json())
