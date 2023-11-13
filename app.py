# Deploy da Aplicação

# Imports
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer
from pinferencia import Server

# App
app = FastAPI()

# Carrega o tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Carrega o modelo treinado
MODEL_PATH = 'modelos/modelo_dsa_mp1.pt'
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/sentimento")
async def analyze_sentiment(text: str):
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    output = model(**inputs)
    scores = output.logits.softmax(dim=1)
    sentiment = torch.argmax(scores).item()
    return {"sentimento": sentiment, "scores": scores.tolist()}
