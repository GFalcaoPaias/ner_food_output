import pandas as pd
import numpy as np
import difflib
import time
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('data/nutrition_dataset.csv', delimiter=',')
df_portion_level = df[~df['calories_per_portion'].isna()]
df_gram_level = df[~df['calories_per_gram'].isna()]

food_names_portion_level = df_portion_level['name'].tolist()
food_names_gram_level = df_gram_level['name'].tolist()

match_columns = ['matched_calories', 'matched_fat', 'matched_carbs', 'matched_protein']

def get_nutrition_values(meals: pd.DataFrame) -> pd.DataFrame:
    meals[match_columns] = None

    for index, row in meals.iterrows():
        df_best_match = get_calories_for_meal(row['amount'], row['unit'], row['name'])
        if df_best_match is None:
            meals.drop(index, inplace=True)
            continue

        meals.loc[index, match_columns] = df_best_match

    float_columns = [
        'name_start',
        'name_end',
        'unit_start',
        'unit_end',
        'amount_start',
        'amount_end']
    float_columns.extend(match_columns)

    meals[float_columns] = meals[float_columns].astype(float)

    # Scale fats/carbs/proteins to fit to calories, cause due to rounding errors in the dataset, calories and macros drift apart
    fit_macros_to_calories(meals)

    return meals

def fit_macros_to_calories(meals):
    calculated_calories = meals['matched_fat'].sum() * 9 + meals['matched_carbs'].sum() * 4 + meals['matched_protein'].sum() * 4
    ratio = meals['matched_calories'].sum() / calculated_calories
    meals['matched_fat'] = meals['matched_fat'] * ratio
    meals['matched_carbs'] = meals['matched_carbs'] * ratio
    meals['matched_protein'] = meals['matched_protein'] * ratio

def get_calories_for_meal(amount_str: str, unit: str, meal_name: str) -> pd.Series:
    meal_name = clean_text(meal_name)
    if len(meal_name) == 0:
        return None

    if meal_name in cache:
        print("RETURNING FROM CACHE")
        return cache[meal_name]
    else:
        print("NOT FOUND IN CACHE", meal_name, cache.keys())

    # Extraction also returns amount like "a", "some", etc.
    amount = get_numeric_amount(amount_str)

    if str(amount_str) == str(-1) and unit:
        amount = float("0" + "".join(re.findall(r'[\d.,]', unit)).replace(',', '.'))
        unit = re.sub(r'[\d.,]', '', unit)

    calories = fats = carbohydrates = proteins = None
    found = False
    base_unit = get_base_unit(unit)

    # Try to calculate on portion level
    if base_unit is None:
        matches = closest_matches(meal_name, portion_level=True)
        matches_df = df_portion_level[df_portion_level['name'].isin(matches)]
        calories_per_portion_median = matches_df['calories_per_portion'].median()
        #import ipdb; ipdb.set_trace()
        # Find the match with that median value
        best_matches = matches_df.iloc[(matches_df['amount'] - calories_per_portion_median).abs().argsort()[:2]]

        calories = best_matches['calories'].mean() * amount
        fats = best_matches['fats'].mean() * amount
        carbohydrates = best_matches['carbohydrates'].mean() * amount
        proteins = best_matches['proteins'].mean() * amount

        # Plausability check. Bigger than 2000ccal might be due to a misinterpretation and we try to treat the unit as grams instead
        if calories > 2000:
            base_unit = get_base_unit("g")
        else:
            found = True

    # If no result yet, we try to calculate on gram level
    if not found:
        amount_grams = get_base_amount_in_grams(amount, base_unit)
        matches = closest_matches(meal_name, portion_level=False)
        matches_df = df_gram_level[df_gram_level['name'].isin(matches)]
        calories_per_gram_median = matches_df['calories_per_gram'].median()

        best_matches = matches_df.iloc[(matches_df['amount'] - calories_per_gram_median).abs().argsort()[:2]]

        calories_per_gram = best_matches['calories_per_gram'].mean()
        calories = amount_grams * calories_per_gram

        original_calories_mean = best_matches['calories'].mean()
        fats = calories * ((best_matches['fats'].mean() * 9) / original_calories_mean) / 9
        carbohydrates = calories * ((best_matches['carbohydrates'].mean() * 4) / original_calories_mean) / 4
        proteins = calories * ((best_matches['proteins'].mean() * 4) / original_calories_mean) / 4

        if calories:
            found = True

    return pd.Series(
        [calories, fats, carbohydrates, proteins],
        index=match_columns) if found else None

number_identifiers = {
    0.125: ['1/8', 'eighth'],
    0.2: ['1/5', 'fifth'],
    0.33: ['1/3', 'third'],
    0.25: ['1/4', 'quarter', 'fourth'],
    0.5: ['1/2', 'half', 'halve', 'halved', 'middle', 'midway', 'part'],
    1: ['a', 'one', 'an', 'the'],
    2: ['two', 'couple', 'pair', 'double', 'twice', 'both'],
    3: ['three', 'triple', 'trio', 'thrice', 'some'],
    4: ['four'],
    5: ['five', 'few'],
    6: ['six'],
    7: ['seven', 'several'],
    8: ['eight'],
    9: ['nine'],
    10: ['ten', 'dozen']
}

# TODO: Use machine learning to identify the amount
def get_numeric_amount(amount):
    amount = amount.lower().strip()

    if amount.isdigit():
        return np.float64(amount)

    for number, identifiers in number_identifiers.items():
        if amount in identifiers:
            return number

    return 1

# TODO: Use machine learning to identify the base unit
def get_base_unit(unit: str) -> str:
    base_unit_mappings = {
        'cc': 'cl',
        'cl': 'cl',
        '-flasche': 'cup',
        'cup': 'cup',
        'cup 175': 'cup',
        'cup 200g': 'cup',
        'cup 206g': 'cup',
        'cup 213g': 'cup',
        'cup 227g': 'cup',
        'cup 236g': 'cup',
        'cup 240': 'cup',
        'cup 240ml': 'cup',
        'cup 250': 'cup',
        'cup 250gm': 'cup',
        'cup 250ml': 'cup',
        'cup 251g': 'cup',
        'cup s': 'cup',
        'cup serving': 'cup',
        'cups': 'cup',
        'g 200ml': 'cup',
        'glas': 'cup',
        'glass': 'cup',
        'glass 250ml': 'cup',
        'glass-5oz': 'cup',
        'dl': 'dl',
        '-fl': 'fl oz',
        'fl': 'fl oz',
        'fl oz': 'fl oz',
        'flox': 'fl oz',
        'floz': 'fl oz',
        'fluid': 'fl oz',
        'lf': 'fl oz',
        'g': 'g',
        'g 08dl': 'g',
        'g 1': 'g',
        'g 1 16oz': 'g',
        'g 1 1oz': 'g',
        'g 1 2': 'g',
        'g 1 2 5can': 'g',
        'g 1 3cup': 'g',
        'g 1 3cup 30g': 'g',
        'g 1 4': 'g',
        'g 1 4cup': 'g',
        'g 1 50g': 'g',
        'g 1 71g': 'g',
        'g 1 9': 'g',
        'g 1 9oz': 'g',
        'g 12 5g-1bar': 'g',
        'g 125ml': 'g',
        'g 12g-1piece': 'g',
        'g 12g-1sachet': 'g',
        'g 14g-1piece': 'g',
        'g 15': 'g',
        'g 15 5-1piece': 'g',
        'g 15g-about': 'g',
        'g 16': 'g',
        'g 17g-1': 'g',
        'g 18g-1piece': 'g',
        'g 1bag': 'g',
        'g 1bar': 'g',
        'g 1cone': 'g',
        'g 1ear': 'g',
        'g 1muffin': 'g',
        'g 1oz': 'g',
        'g 1pce 6 1g': 'g',
        'g 1row 15 45g': 'g',
        'g 1sachet 35g': 'g',
        'g 1scoop': 'g',
        'g 1serve 125g': 'g',
        'g 1serve 200g': 'g',
        'g 1serve 212g': 'g',
        'g 1serve 220g': 'g',
        'g 1serve 32g': 'g',
        'g 1serve 40g': 'g',
        'g 1serve 47g': 'g',
        'g 1serve 50g': 'g',
        'g 1serve 75g': 'g',
        'g 1tablespoon': 'g',
        'g 1tbsp': 'g',
        'g 1tbspn 20g': 'g',
        'g 2': 'g',
        'g 2 3cup': 'g',
        'g 2 6': 'g',
        'g 2 71': 'g',
        'g 2 90g': 'g',
        'g 20': 'g',
        'g 20g-1piece': 'g',
        'g 20g-1small': 'g',
        'g 25': 'g',
        'g 250 350ml': 'g',
        'g 250ml 1': 'g',
        'g 25g-3crackers': 'g',
        'g 25g-3pieces': 'g',
        'g 26': 'g',
        'g 28g-4pieces': 'g',
        'g 28g-about': 'g',
        'g 2oz': 'g',
        'g 2pieces-30g': 'g',
        'g 2sl 59g': 'g',
        'g 2sl 60g': 'g',
        'g 2sl 67g': 'g',
        'g 2tbsp': 'g',
        'g 3': 'g',
        'g 3 4': 'g',
        'g 3 4cup-30g': 'g',
        'g 3 5': 'g',
        'g 30': 'g',
        'g 32g-2tbsp': 'g',
        'g 35': 'g',
        'g 36g-3pieces': 'g',
        'g 38g-1packet': 'g',
        'g 3oz': 'g',
        'g 3pieced-21g': 'g',
        'g 3pieces-26g': 'g',
        'g 4': 'g',
        'g 4 5oz': 'g',
        'g 43g': 'g',
        'g 45g-2 3cup': 'g',
        'g 49g-1 3cup': 'g',
        'g 4oz': 'g',
        'g 4pieces': 'g',
        'g 4pieces-21g': 'g',
        'g 5 3oz': 'g',
        'g 5 5': 'g',
        'g 50g-1pack': 'g',
        'g 50g-1piece': 'g',
        'g 58g-2pieces': 'g',
        'g 59g-1cup': 'g',
        'g 5pieces': 'g',
        'g 60g-1': 'g',
        'g 62 5g-5pieces': 'g',
        'g 6oz': 'g',
        'g 7 37': 'g',
        'g 7 5g 1 4c': 'g',
        'g 7 9oz': 'g',
        'g 8 3oz': 'g',
        'g 8oz': 'g',
        'g 9': 'g',
        'g 9oz': 'g',
        'g about': 'g',
        'g container': 'g',
        'g ml': 'g',
        'g one': 'g',
        'g reg': 'g',
        'g s': 'g',
        'g serve': 'g',
        'g-': 'g',
        'g-1': 'g',
        'g-1pz': 'g',
        'g-1tbls': 'g',
        'g-2': 'g',
        'g-2tbsp': 'g',
        'g-drained': 'g',
        'g-ish': 'g',
        'ge': 'g',
        'get': 'g',
        'gm': 'g',
        'gm1': 'g',
        'gms': 'g',
        'gms 2tbsp': 'g',
        'gr': 'g',
        'gr 1': 'g',
        'gr 1pack': 'g',
        'gram': 'g',
        'gram s': 'g',
        'gram w': 'g',
        'gramas': 'g',
        'grame': 'g',
        'gramm': 'g',
        'grammes': 'g',
        'grammi': 'g',
        'gramms': 'g',
        'gramos': 'g',
        'grams': 'g',
        'grams 1': 'g',
        'grams 1 2': 'g',
        'grams 1 4': 'g',
        'grams 18': 'g',
        'grams 2': 'g',
        'grams 31': 'g',
        'grams 4oz': 'g',
        'grams 6"': 'g',
        'grams 8': 'g',
        'grams aprox': 'g',
        'grams-': 'g',
        'grm': 'g',
        'grms': 'g',
        'grs': 'g',
        'kg': 'kg',
        'kg s': 'kg',
        'l': 'l',
        'liter': 'l',
        'litre': 'l',
        'litro': 'l',
        'lb': 'lb',
        'lb 16': 'lb',
        'lb s': 'lb',
        'lbs': 'lb',
        'mcg': 'mg',
        'mg': 'mg',
        'micrograms': 'mg',
        'milligram': 'mg',
        'kl': 'ml',
        'm': 'ml',
        'mil': 'ml',
        'mililitros': 'ml',
        'milliliter': 'ml',
        'milliliters': 'ml',
        'mils': 'ml',
        'ml': 'ml',
        'ml 1': 'ml',
        'ml 100ml-48g': 'ml',
        'ml 100ml-84g': 'ml',
        'ml 11 15': 'ml',
        'ml 118ml-99g': 'ml',
        'ml 1cone': 'ml',
        'ml 1glass': 'ml',
        'ml 1serve 1 4cup': 'ml',
        'ml 236ml-1packet': 'ml',
        'ml 24oz': 'ml',
        'ml 2fl': 'ml',
        'ml 2tsp': 'ml',
        'ml 3oz': 'ml',
        'ml 68ml-1bottle': 'ml',
        'ml 8 5oz': 'ml',
        'ml s': 'ml',
        'ml-3 4': 'ml',
        'mls': 'ml',
        '25': 'oz',
        '28': 'oz',
        '29': 'oz',
        '30': 'oz',
        '1oz': 'oz',
        '26g': 'oz',
        '27g': 'oz',
        '30g': 'oz',
        '30g': 'oz',
        '30grms': 'oz',
        '32g': 'oz',
        '80g 2 8oz': 'oz',
        'cup-30g': 'oz',
        'o': 'oz',
        'o z': 'oz',
        'onz': 'oz',
        'ounce': 'oz',
        'ounce 112g': 'oz',
        'ounce 15': 'oz',
        'ounce 36': 'oz',
        'ounce-': 'oz',
        'ounces': 'oz',
        'ounces 113g': 'oz',
        'ounces 162': 'oz',
        'ounces 56': 'oz',
        'ounces 8': 'oz',
        'ounces 85': 'oz',
        'ounces-': 'oz',
        'ounces1': 'oz',
        'ounches': 'oz',
        'ounzes': 'oz',
        'ournces': 'oz',
        'ox': 'oz',
        'oz': 'oz',
        'oz   25': 'oz',
        'oz  1': 'oz',
        'oz  112g': 'oz',
        'oz  113': 'oz',
        'oz  118': 'oz',
        'oz  14': 'oz',
        'oz  142g': 'oz',
        'oz  18': 'oz',
        'oz  2': 'oz',
        'oz  240ml': 'oz',
        'oz  28': 'oz',
        'oz  28g': 'oz',
        'oz  28g about': 'oz',
        'oz  312': 'oz',
        'oz  340g': 'oz',
        'oz  45': 'oz',
        'oz  591ml': 'oz',
        'oz  6': 'oz',
        'oz  85g about': 'oz',
        'oz  9chips': 'oz',
        'oz  cooked': 'oz',
        'oz  slice': 'oz',
        'oz -': 'oz',
        'oz 1': 'oz',
        'oz 1 4': 'oz',
        'oz 103 5g': 'oz',
        'oz 105g': 'oz',
        'oz 112g': 'oz',
        'oz 113': 'oz',
        'oz 113g': 'oz',
        'oz 125': 'oz',
        'oz 140g': 'oz',
        'oz 15': 'oz',
        'oz 150g': 'oz',
        'oz 153g': 'oz',
        'oz 156': 'oz',
        'oz 16': 'oz',
        'oz 165g': 'oz',
        'oz 16floz': 'oz',
        'oz 170g': 'oz',
        'oz 18g': 'oz',
        'oz 1bread': 'oz',
        'oz 1serving  85g': 'oz',
        'oz 2': 'oz',
        'oz 200g': 'oz',
        'oz 240ml': 'oz',
        'oz 24fl': 'oz',
        'oz 24floz': 'oz',
        'oz 25g': 'oz',
        'oz 28': 'oz',
        'oz 28 3g': 'oz',
        'oz 28g': 'oz',
        'oz 28g 13chips': 'oz',
        'oz 28g 31chips': 'oz',
        'oz 28g 39': 'oz',
        'oz 28g 3chips': 'oz',
        'oz 28g about': 'oz',
        'oz 28g pack': 'oz',
        'oz 28grams about': 'oz',
        'oz 3': 'oz',
        'oz 3 5g': 'oz',
        'oz 30g': 'oz',
        'oz 31 1g 1': 'oz',
        'oz 330': 'oz',
        'oz 35g pack': 'oz',
        'oz 3slices': 'oz',
        'oz 40g': 'oz',
        'oz 42': 'oz',
        'oz 42 5': 'oz',
        'oz 43g': 'oz',
        'oz 45g': 'oz',
        'oz 49 6': 'oz',
        'oz 52g': 'oz',
        'oz 56g': 'oz',
        'oz 56g  25': 'oz',
        'oz 56g 3 4': 'oz',
        'oz 57': 'oz',
        'oz 57g': 'oz',
        'oz 57g0': 'oz',
        'oz 6': 'oz',
        'oz 60': 'oz',
        'oz 70g': 'oz',
        'oz 74g': 'oz',
        'oz 8': 'oz',
        'oz 84': 'oz',
        'oz 84g': 'oz',
        'oz 85g': 'oz',
        'oz 85gm': 'oz',
        'oz 9': 'oz',
        'oz 90g': 'oz',
        'oz about': 'oz',
        'oz can': 'oz',
        'oz cup': 'oz',
        'oz s': 'oz',
        'oz slice': 'oz',
        'oz slices': 'oz',
        'oz tall': 'oz',
        'oz-': 'oz',
        'oz--about': 'oz',
        'oz-1': 'oz',
        'oz-142g': 'oz',
        'oz-1chips': 'oz',
        'oz-28g': 'oz',
        'oz-8': 'oz',
        'oz-grande': 'oz',
        'ozs': 'oz',
        'shot 1 30ml': 'oz',
        'shot-1oz': 'oz',
        'shot1': 'oz',
        'shots': 'oz',
        't': 'tbsp',
        't  30g': 'tbsp',
        'tabelspoon': 'tbsp',
        'tabelspoons': 'tbsp',
        'tabl': 'tbsp',
        'table': 'tbsp',
        'tablepspoon': 'tbsp',
        'tablesoon': 'tbsp',
        'tablesoons': 'tbsp',
        'tablespool': 'tbsp',
        'tablespoom': 'tbsp',
        'tablespoon': 'tbsp',
        'tablespoon  15g': 'tbsp',
        'tablespoon 15g': 'tbsp',
        'tablespoon- 15ml': 'tbsp',
        'tablespoons': 'tbsp',
        'tablespoons 14g': 'tbsp',
        'tablespoons 17g': 'tbsp',
        'tablespoons 20g': 'tbsp',
        'tablespoons 32g': 'tbsp',
        'tablespoons 50g': 'tbsp',
        'tablespoonss': 'tbsp',
        'tablespooons': 'tbsp',
        'tablespoos': 'tbsp',
        'tablesppon': 'tbsp',
        'tablesppons': 'tbsp',
        'tb': 'tbsp',
        'tbl': 'tbsp',
        'tblespn': 'tbsp',
        'tbls': 'tbsp',
        'tblsp': 'tbsp',
        'tblspn': 'tbsp',
        'tblspns': 'tbsp',
        'tblspoon': 'tbsp',
        'tblsps': 'tbsp',
        'tbp': 'tbsp',
        'tbpn': 'tbsp',
        'tbps': 'tbsp',
        'tbs': 'tbsp',
        'tbs 15': 'tbsp',
        'tbs 1oz': 'tbsp',
        'tbs 6 92g': 'tbsp',
        'tbs- 14g': 'tbsp',
        'tbs--21g': 'tbsp',
        'tbsb': 'tbsp',
        'tbsn': 'tbsp',
        'tbsp': 'tbsp',
        'tbsp  14g': 'tbsp',
        'tbsp  14gr  1 2': 'tbsp',
        'tbsp  15g': 'tbsp',
        'tbsp  28g': 'tbsp',
        'tbsp  46g': 'tbsp',
        'tbsp  5': 'tbsp',
        'tbsp  7g  0 25oz': 'tbsp',
        'tbsp 1': 'tbsp',
        'tbsp 12g': 'tbsp',
        'tbsp 14': 'tbsp',
        'tbsp 14g': 'tbsp',
        'tbsp 15': 'tbsp',
        'tbsp 15 2grams': 'tbsp',
        'tbsp 15g': 'tbsp',
        'tbsp 15ml': 'tbsp',
        'tbsp 17': 'tbsp',
        'tbsp 17g': 'tbsp',
        'tbsp 19': 'tbsp',
        'tbsp 20g': 'tbsp',
        'tbsp 21g': 'tbsp',
        'tbsp 28': 'tbsp',
        'tbsp 28g': 'tbsp',
        'tbsp 28g 1oz': 'tbsp',
        'tbsp 30': 'tbsp',
        'tbsp 30g': 'tbsp',
        'tbsp 30ml': 'tbsp',
        'tbsp 31': 'tbsp',
        'tbsp 31g': 'tbsp',
        'tbsp 32g': 'tbsp',
        'tbsp 34g': 'tbsp',
        'tbsp 3g': 'tbsp',
        'tbsp 45ml': 'tbsp',
        'tbsp 4g': 'tbsp',
        'tbsp 5 ml': 'tbsp',
        'tbsp 5g': 'tbsp',
        'tbsp 60ml': 'tbsp',
        'tbsp 6g': 'tbsp',
        'tbsp 7': 'tbsp',
        'tbsp 7g': 'tbsp',
        'tbsp 9': 'tbsp',
        'tbsp oz': 'tbsp',
        'tbsp s': 'tbsp',
        'tbsp-': 'tbsp',
        'tbsp-15ml': 'tbsp',
        'tbsp-30g': 'tbsp',
        'tbspn': 'tbsp',
        'tbspns': 'tbsp',
        'tbspp': 'tbsp',
        'tbsps': 'tbsp',
        'tbsps 29g': 'tbsp',
        'tbst': 'tbsp',
        'tdsp': 'tbsp',
        'tlb': 'tbsp',
        'tlbs': 'tbsp',
        'tlbsp': 'tbsp',
        'tpsp': 'tbsp',
        'tsb': 'tbsp',
        'tsbp': 'tbsp',
        'tspb': 'tbsp',
        'ttbsp': 'tbsp',
        '1tsp': 'tsp',
        't 14': 'tsp',
        'teaspon': 'tsp',
        'teaspoon': 'tsp',
        'teaspoons': 'tsp',
        'teaspooon': 'tsp',
        'tps': 'tsp',
        'ts': 'tsp',
        'tsp': 'tsp',
        'tsp  5g': 'tsp',
        'tsp  8g': 'tsp',
        'tsp  9g': 'tsp',
        'tsp 0 3': 'tsp',
        'tsp 1 3g': 'tsp',
        'tsp 2': 'tsp',
        'tsp 3gms': 'tsp',
        'tsp 5g': 'tsp',
        'tsp 5ml': 'tsp',
        'tsp 7g': 'tsp',
        'tsp s': 'tsp',
        'tspn': 'tsp',
        'tspns': 'tsp',
        'tsps': 'tsp',
    }

    return base_unit_mappings.get(unit.strip().lower())

def get_base_amount_in_grams(amount: float, base_unit: str) -> float:
    to_gram_factors = {
        'g': 1,
        'oz': 28.3495,
        'cup': 236.588,
        'tbsp': 14.7868,
        'fl oz': 29.5735,
        'tsp': 4.92892,
        'kg': 1000,
        'mg': 0.001,
        'lb': 453.592,
        'l': 1000,
        'dl': 100,
        'cl': 10,
        'ml': 1,
    }

    return amount * to_gram_factors.get(base_unit)

cache = {
    'portion_level': {},
    'gram_level': {}
}
def closest_matches(meal_name: str, portion_level: bool) -> list:
    cache_key = 'portion_level' if portion_level else 'gram_level'
    if meal_name in cache[cache_key]:
        return cache[cache_key][meal_name]

    food_names = food_names_portion_level if portion_level else food_names_gram_level

    time = current_milli_time()
    matches = difflib.get_close_matches(meal_name, food_names, n=50, cutoff=0.6)
    print("TIME TO FIND ALL IN DATASET", current_milli_time() - time)

    cache_key = 'portion_level' if portion_level else 'gram_level'
    cache[cache_key][meal_name] = matches

    return matches

def clean_text(text):
    text = text.strip()
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos = "n") for word in tokens]
    return ' '.join(tokens)


if __name__ == "__main__":
    print(get_nutrition_values(pd.DataFrame({
        'name': ['bla', 'milk'],
        'amount': ['1', '1'],
        'unit': ['piece', 'glass'],
    })))


def current_milli_time():
    return round(time.time() * 1000)
