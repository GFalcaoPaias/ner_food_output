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

    output = ner_food_output(text, pipe)
    nutrition_values = get_nutrition_values(output)

    # Increase all indices by 2 for every new line before
    # float_columns = [
    #     'name_start',
    #     'name_end',
    #     'unit_start',
    #     'unit_end',
    #     'amount_start',
    #     'amount_end']

    # for key in nutrition_values:
    #     nutrition_values[key] = [i + 2 for i in nutrition_values[key]]

    nutrition_values.dropna(inplace=True)

    print(nutrition_values)
    return {col: nutrition_values[col].to_list() for col in nutrition_values.columns}

# Just a simple endpoint to identify, whether slow execution times are due to the network
@app.get("/ping")
def root():
    return "pong"
