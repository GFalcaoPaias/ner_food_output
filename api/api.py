from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ner import ner_food_output
from transformers import pipeline
from api.get_nutrition_values import get_nutrition_values

app = FastAPI()

# Allowing all origins for testing locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the NER pipeline using a fine-tuned model for food entities
pipe = pipeline("ner", model="davanstrien/deberta-v3-base_fine_tuned_food_ner")

# Define a root `/` endpoint
@app.get('/')
def food(text):

    # Replacing some buzzwords that should not be detected as food
    text = text.lower()
    text = text.replace('breakfast', '         ')
    text = text.replace('lunch', '    ')
    text = text.replace('dinner', '     ')

    output = ner_food_output(text, pipe)
    nutrition_values = get_nutrition_values(output)

    nutrition_values.dropna(inplace=True)

    return {col: nutrition_values[col].to_list() for col in nutrition_values.columns}
