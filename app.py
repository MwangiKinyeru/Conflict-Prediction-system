import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import sys
import io
import folium

# ✅ Force UTF-8 Encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

app = Flask(__name__, template_folder="Templates")

# 🔹 Get Absolute Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "xgboost_model.pkl")
conflict_data_path = os.path.join(BASE_DIR, "Data", "conflict_data.csv")

# ✅ Load trained XGBoost model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ✅ Load conflict dataset
try:
    conflict_data = pd.read_csv(conflict_data_path)
    print("Conflict dataset loaded successfully.")
except Exception as e:
    print(f"Error loading conflict dataset: {e}")
    conflict_data = None

# 🔹 Precompute average latitude & longitude for each country
if conflict_data is not None:
    country_avg_coords = conflict_data.groupby("country")[["latitude", "longitude"]].mean().to_dict()
else:
    country_avg_coords = {}

# 🔹 Define feature names
feature_names = [
    "year", "latitude", "longitude", "log_fatalities", "civilian_targeting_Yes",
    "country_Angola", "country_Benin", "country_Botswana", "country_Burkina Faso", "country_Burundi",
    "country_Cameroon", "country_Cape Verde", "country_Central African Republic", "country_Chad",
    "country_Comoros", "country_Democratic Republic of Congo", "country_Djibouti", "country_Egypt",
    "country_Equatorial Guinea", "country_Eritrea", "country_Ethiopia", "country_Gabon", "country_Gambia",
    "country_Ghana", "country_Guinea", "country_Guinea-Bissau", "country_Ivory Coast", "country_Kenya",
    "country_Lesotho", "country_Liberia", "country_Libya", "country_Madagascar", "country_Malawi",
    "country_Mali", "country_Mauritania", "country_Mauritius", "country_Mayotte", "country_Morocco",
    "country_Mozambique", "country_Namibia", "country_Niger", "country_Nigeria", "country_Republic of Congo",
    "country_Reunion", "country_Rwanda", "country_Saint Helena, Ascension and Tristan da Cunha",
    "country_Sao Tome and Principe", "country_Senegal", "country_Seychelles", "country_Sierra Leone",
    "country_Somalia", "country_South Africa", "country_South Sudan", "country_Sudan", "country_Tanzania",
    "country_Togo", "country_Tunisia", "country_Uganda", "country_Zambia", "country_Zimbabwe", "country_eSwatini"
]

# 🛠 Preprocess User Input
def preprocess_input(country):
    """Creates a one-hot encoded dataframe with the exact features the model was trained on."""
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data["year"] = 2025  # Default prediction year
    input_data["latitude"] = country_avg_coords["latitude"].get(country, 0)
    input_data["longitude"] = country_avg_coords["longitude"].get(country, 0)

    # One-hot encode country
    country_col = f"country_{country}"
    if country_col in input_data.columns:
        input_data[country_col] = 1
    else:
        raise ValueError(f"Error: Country '{country}' not found in model features.")

    return input_data

# 1️⃣ Calculate Probability of Conflict for the last 3-5 years (Recent Period)
def calculate_recent_probability(country):
    # Filter the conflict data for the period 2020-2025 (or adjust for the last 3-5 years)
    recent_data = conflict_data[(conflict_data['country'] == country) & (conflict_data['year'] >= 2020)]
    
    # If there's no conflict data in the recent period, return a default (low) probability
    if recent_data.empty:
        return 5.3  # Default low probability for lack of data
    
    # Calculate the total number of conflicts and fatalities in this period
    total_conflicts = len(recent_data)
    total_fatalities = recent_data['fatalities'].sum()
    
    # Let's assume the probability of conflict is based on the frequency of conflicts
    probability = (total_conflicts / len(conflict_data[conflict_data['country'] == country])) * 100  # Rough approximation
    return round(probability, 1)

def classify_conflict_state(probability):
    if probability > 75:
        return "High Risk", "worsening", "The Country is not advisable for investment."
    elif 40 <= probability <= 75:
        return "Medium Risk", "improving", "The Country could be considered for investment with caution. It is important to monitor the situation closely, assess the latest developments, and evaluate the potential risks before making any decisions."
    else:
        return "Low Risk", "stable", "The Country is advisable for investment."

# Function to create Folium map
def create_map(selected_country, conflict_probability):
    # Define risk colors based on probability
    if conflict_probability >= 75:
        country_color = "red"  # High-risk
    elif 40 <= conflict_probability < 75:
        country_color = "blue"  # Medium-risk
    else:
        country_color = "green"  # Stablef

    # Get country coordinates from precomputed averages
    lat = country_avg_coords["latitude"].get(selected_country, 0)
    lon = country_avg_coords["longitude"].get(selected_country, 0)

    # Initialize Folium Map (centered on the country)
    folium_map = folium.Map(location=[lat, lon], zoom_start=5)

    # Add a marker with conflict risk information
    folium.Marker(
        [lat, lon],
        popup=f"{selected_country}: {conflict_probability}% Conflict Probability",
        tooltip=selected_country,
        icon=folium.Icon(color=country_color),
    ).add_to(folium_map)

    return folium_map



# 🏠 Homepage Route
@app.route("/")
def home():
    countries = conflict_data["country"].unique()  # Get the list of unique countries
    return render_template("index.html", countries=countries)


# Safe countries list
safe_countries = [
    'Saint Helena, Ascension and Tristan da Cunha', 'Seychelles', 'Mauritius', 'Mayotte', 
    'Reunion', 'Sao Tome and Principe', 'Comoros', 'Cape Verde'
]

# 🚀 Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model:
            raise ValueError("Model is not loaded correctly.")
        if conflict_data is None:
            raise ValueError("Conflict dataset is not loaded correctly.")

        # 1️⃣ Get User Input
        country = request.form.get("country", "").strip()
        if not country:
            raise ValueError("Country is missing from the input form!")

        print(f"Predicting for country: {country}")

        # Check if the country is in the safe countries list
        if country in safe_countries:
            # For safe countries, set low conflict risk level and probability
            conflict_summary = (
                f"The conflict in {country} has been classified as minimal, with no significant conflict events in recent years. "
                "The country has experienced almost zero fatalities from conflict, indicating a stable environment. "
                "The conflict trend is stable or improving, with a very low likelihood of future conflict recurrence. "
                f"Given the near-zero conflict risk, {country} is considered extremely safe for investment. "
                "The country offers a peaceful environment conducive to business growth and long-term investment. "
                "Investors can have high confidence in the stability of the region, making it an ideal choice for expansion and investment opportunities."
            )
            return render_template("result.html", 
                                   risk="Almost Zero or Low Risk", 
                                   probability="less than 5.0%", 
                                   event_type="No significant conflicts recorded", 
                                   summary=conflict_summary, map_html="")

        # 2️⃣ Preprocess Input for non-safe countries
        input_data = preprocess_input(country)
        if input_data.isnull().values.any():
            raise ValueError(f"Input data contains NaN values: \n{input_data}")

        print("Processed Input Data:\n", input_data)

        # 3️⃣ Make Prediction
        try:
            probability = round(model.predict_proba(input_data)[0][1] * 100, 1)
        except Exception as pred_error:
            raise ValueError(f"Model prediction error: {pred_error}")

        # 4️⃣ Retrieve Recent Conflict Data (2020-2025 or Last 3-5 Years)
        recent_probability = calculate_recent_probability(country)
        
        # 5️⃣ Classify Conflict State based on Recent Probability
        risk_level, trend, investment_advice = classify_conflict_state(recent_probability)

        # 6️⃣ Retrieve Past Conflict Data
        region_conflicts = conflict_data[conflict_data["country"] == country]
        print(f"Found {len(region_conflicts)} past conflict records.")

        if not region_conflicts.empty:
            most_common_event = region_conflicts["event_type"].mode()[0] if "event_type" in region_conflicts else "Unknown Conflict"

            # 🔹 Find the three years with the highest fatalities
            top_fatal_years = (
                region_conflicts.groupby("year")["fatalities"]
                .sum()
                .nlargest(3)
                .reset_index()
            )
            past_years = ", ".join(map(str, top_fatal_years["year"].tolist()))
            total_fatalities = top_fatal_years["fatalities"].sum()

            # 🔹 Generate Conflict Summary
            conflict_summary = (
                f"The conflict in {country} was classified as {most_common_event.lower()}. "
                f"The three most severe years were {past_years}, with a total of {total_fatalities} fatalities. "
                f"The conflict trend appears to be {trend}. "
                f"For the last 3-5 years, the probability of conflict recurrence is {recent_probability}%, "
                f"this indicates that the country conflict state is {trend}. "
                f"Given the {recent_probability}% probability of recurrence, {investment_advice}"
            )
        else:
            conflict_summary = f"{country} has no significant recorded conflict history. It is considered a relatively stable investment location."
            most_common_event = "No past conflicts recorded"

        # ✅ Create Folium Map for the selected country
        folium_map = create_map(country, recent_probability)
        map_html = folium_map._repr_html_()

        # ✅ Render the result page with the map
        return render_template("result.html", risk=risk_level, probability=recent_probability, event_type=most_common_event, summary=conflict_summary, map_html=map_html)

    except Exception as e:
        print(f"ERROR in predict(): {str(e)}")
        return render_template("error.html", error_message=str(e))

# 🎯 Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
