# Conflict-Prediction-system

## Project Overview

> This project aims to develop a Conflict Prediction System to help businesses and investors identify safe locations for investment in Africa. Using the ACLED dataset, we built a machine learning model (XGBoost) to predict the likelihood of conflict in different regions. The project follows the CRISP-DM methodology, covering data analysis, feature engineering, model training, evaluation, and deployment using Flask.

### Problem Statement

> Investors require stable and conflict-free locations for business expansion. Political instability, civil unrest, and violent conflicts impact investment decisions. This system:
>> * Predicts the probability of conflict in African regions.
>> * Identifies high-risk and low-risk locations.
>> * Provides a historical summary of past conflicts.
>> * Generates a risk-based investment recommendation.
> visualization of the africa map risk conflict areas
![Description](https://github.com/MwangiKinyeru/Conflict-Prediction-system/blob/main/Images/Capture%204.PNG)

### Methodology

> The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework:
>> * Business Understanding
>> * Data Understanding
>> * Data Preparation
>> * Modeling
>> * Evaluation
>> * Deployment

### Data Analysis & Preprocessing

> The dataset used is sourced from ACLED (Armed Conflict Location & Event Data). Key preprocessing steps include:
>> * Data cleaning and data sanity check up
>> * Exploratory data analysis
>> * Data preprocessing
>> * Feature Engineering

### Model Training

> Three models were considered:
>> * Logistic Regression (Baseline model)
>> * Random Forest (For capturing non-linear relationships)
>> * XGBoost Model
> Model Selection, The XGBoost outperformed other models in terms of accuracy, precision, and recall. The final model was fine-tuned using RandomizedSearchCV.

### Evaluation

#### Model Performance Comparison

| **Model**                 | **Accuracy** | **Precision** | **Recall** | **ROC-AUC** |
|---------------------------|-------------|--------------|------------|-------------|
| **Logistic Regression**   | 73.2%       | 0.68         | 0.65       | 0.72        |
| **Random Forest**         | 78.5%       | 0.76         | 0.74       | 0.81        |
| **XGBoost (Final Model)** | **85.2%**   | **0.82**     | **0.80**   | **0.87**    |

### Deployment
> The Conflict Prediction System was deployed using Flask as a lightweight web framework. The model was trained using XGBoost and saved as a serialized .pkl file for efficient loading during inference.

#### *Web Application (Flask-based)
> The web Flask application was built to allow users to:
>> * Input a country and region.
>> * Get a conflict risk prediction.
>> * View a summary of past conflicts.
>> * See a map visualization of high-risk areas.
> visualization of the web application
<p align="center">
  <img src="https://github.com/MwangiKinyeru/Conflict-Prediction-system/blob/main/Images/Capture%201.PNG" width="45%" />
  <img src="https://github.com/MwangiKinyeru/Conflict-Prediction-system/blob/main/Images/Capture%202.PNG" width="45%" />
</p>

### Future Improvements
>> * Expand Geographic Scope - Extend the model to predict conflicts in other regions outside Africa.
>> * Enhanced Data Visualization - Add interactive dashboards with real-time conflict trends using Plotly or Dash. 
>> * AI Chatbot for Conflict Queries - Develop an NLP-based chatbot to answer user queries on conflict trends.
>> * Real-time Data Integration - Automate data updates from ACLED or other conflict databases via APIs that will enable creating a Predictive  Alerts System that will Send email or SMS alerts when a country’s conflict probability exceeds a dangerous threshold.

### Deployment Link
The Africa conflict prediction system was deployed in render to access the prediction service follow the link below:
>>>> 🔗[Web_Link](https://conflict-prediction-system-2.onrender.com)

### Author
#### *** DS Martin Waweru***



