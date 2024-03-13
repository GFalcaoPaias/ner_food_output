from fastapi import FastAPI
from api.ner import ner_food_output
from transformers import pipeline

app = FastAPI()

# Load the NER pipeline using a fine-tuned model for food entities
pipe = pipeline("ner", model="davanstrien/deberta-v3-base_fine_tuned_food_ner")

# Define a root `/` endpoint
@app.get('/')
def food(text):

    output = ner_food_output(text, pipe).to_dict()

    return output
