import pandas as pd
import string
from transformers import pipeline


def clean_text(result):

    # Lists to store information
    foods = []
    quantities = []
    units = []

    # Iterate over the resulting entities and append the information
    for entity in result:
        if "FOOD" in entity["entity"]:
            current_food = {"food": entity["word"], "food_start": entity["start"], "food_end": entity["end"]}
            foods.append(current_food)
        elif "QUANTITY" in entity["entity"]:
            current_quantity = {"quantity": entity["word"], "quantity_start": entity["start"], "quantity_end": entity["end"]}
            quantities.append(current_quantity)
        elif "UNIT" in entity["entity"]:
            current_unit = {"unit": entity["word"], "unit_start": entity["start"], "unit_end": entity["end"]}
            units.append(current_unit)

    # Create separate DataFrames for each type of information
    df_food = pd.DataFrame(foods)
    df_quantity = pd.DataFrame(quantities)
    df_unit = pd.DataFrame(units)

    # Check if DataFrames are empty, and create them if needed
    if df_food.empty:
        df_food = pd.DataFrame(columns=["quantity", "quantity_start", "quantity_end"])

    if df_quantity.empty:
        df_quantity = pd.DataFrame(columns=["quantity", "quantity_start", "quantity_end"])

    if df_unit.empty:
        df_unit = pd.DataFrame(columns=["unit", "unit_start", "unit_end"])


    # Combine the DataFrames
    df_edited = pd.concat([df_food, df_quantity, df_unit], axis=1)

    # Fill NaN values with -1
    df_edited = df_edited.fillna(-1)

    # Clean the text
    for i, rows in df_edited.iterrows():
        df_edited.loc[i,'food'] = str(df_edited.loc[i,'food']).replace("▁"," ").strip().lower().capitalize()
        df_edited.loc[i,'quantity'] = str(df_edited.loc[i,'quantity']).replace("▁"," ").strip().lower().capitalize()
        df_edited.loc[i,'unit'] = str(df_edited.loc[i,'unit']).replace("▁"," ").strip()


    # Replace NaN, nan, and NaN with a specific non-null value (e.g., -1) across the entire DataFrame
    df_edited.replace({pd.NaT: -1, 'Nan': -1, 'nan': -1, 'NaN': -1}, inplace=True)
    df_edited = df_edited.fillna(-1)

    # Change type for each columns
    df_edited['food'] = df_edited['food'].astype(str)
    df_edited['food_start'] = df_edited['food_start'].astype(int)
    df_edited['food_end'] = df_edited['food_end'].astype(int)

    df_edited['quantity_start'] = df_edited['quantity_start'].astype(int)
    df_edited['quantity_end'] = df_edited['quantity_end'].astype(int)

    df_edited['unit'] = df_edited['unit'].astype(str)
    df_edited['unit_start'] = df_edited['unit_start'].astype(int)
    df_edited['unit_end'] = df_edited['unit_end'].astype(int)

    # Ignore warnings
    #pd.set_option("mode.chained_assignment", None)

    return df_edited


def ner_food_output(food_name):

    # Load the NER pipeline using a fine-tuned model for food entities
    pipe = pipeline("ner", model="ner_model")

    # Get the NER results for the given food_name
    result = pipe(food_name)

    # Filter entities with confidence score >= 0.3
    result = [entity for entity in result if entity['score'] >= 0.3]

    # Iterate over the results to merge consecutive rows with the same entity
    for i in range(len(result) - 1, 0, -1):
        current_entity = result[i]["entity"]
        previous_entity = result[i - 1]["entity"].split('-')[1]

        # Merge consecutive rows with the same entity
        if current_entity.split('-')[1] == previous_entity:
            if current_entity.split('-')[0] == "U":
                None
            if current_entity.split('-')[0] == "I" or current_entity.split('-')[0] == "L":
                # Append the word from the row below to the "word" column
                result[i - 1]["word"] += result[i]["word"]

                # Update the "end" value of the first row with the "end" value of the row below
                result[i - 1]["end"] = result[i]["end"]

                # Delete the current row
                del result[i]

    # Identify sentence boundaries based on start and end indices
    sentences=[]
    end_sentence = result[-1]['end']

    for i in range(len(result) - 1, 0, -1):
        current_entity = result[i]["entity"]
        previous_entity = result[i - 1]["entity"].split('-')[1]

        if result[i]["start"] != result[i-1]["end"]:
            start_index = result[i-1]["end"]
            end_index = result[i]["start"]

            # Check for conjunctions or punctuation to determine sentence boundaries
            if ' and' in food_name[start_index:end_index] or any([punc in food_name[start_index:end_index] for punc in string.punctuation]):
                sentences.append(end_sentence)
                end_sentence = start_index

    sentences.append(end_sentence)
    sentences = sentences[::-1]

    # Create a DataFrame from the NER results
    a=pd.DataFrame(result)
    output = []

    # Extract rows corresponding to each identified sentence
    for end in sentences:
        i = a[a["end"]==end].index[0]
        output.append([])
        for row in range(i+1):
            output[-1].append(a.loc[row].to_dict())
        a = a.loc[i+1:].reset_index(drop=True)

    # Concatenate cleaned text output (function used) for each sentence
    result = pd.concat([clean_text(o) for o in output])

    result.reset_index(drop=True, inplace=True)

    return result
