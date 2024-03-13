from fastapi import FastAPI
from api.ner import ner_food_output

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def food(text):
    output = ner_food_output(text).to_dict()

    return output
