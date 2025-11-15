# Traffic-Accidents-Severity-Prediction
An interactive Streamlit-based machine learning App that predicts the severity of traffic accidents based on real-time data.

🚗 Traffic Accident Severity Prediction System

An interactive Streamlit-based machine learning dashboard that predicts the severity of traffic accidents based on real-time data.
The system uses a Random Forest classifier to identify whether an accident is likely to be Minor, Serious, or Fatal and provides feature importance insights for better understanding.

	
🚀 Features
📁 1. CSV Upload & Data Overview

Upload accident datasets in CSV format

View raw data, missing values, and metrics (total records, features)

🔧 2. Automated Data Preprocessing

Cleans column names and trims spaces

Handles invalid or missing data

Encodes categorical variables

Scales numeric features

🤖 3. Machine Learning Model

Random Forest Classifier for severity prediction

Train-test split with accuracy metrics

Displays training and testing accuracy

📊 4. Feature Importance Analysis

Identifies top 10 features affecting accident severity

Interactive bar chart and table for insights

🔮 5. Real-Time Prediction

Users can input accident details via dashboard

Predicts severity (Minor, Serious, Fatal)

Shows prediction probabilities with confidence scores

📋 Required CSV Format

Your CSV should include the following columns:

Column	Description
Accident_ID	Unique identifier for the accident
Speed_kmph	Vehicle speed in km/h
Age_of_Driver	Age of the driver
Location	Location of the accident
Weather_Condition	Weather conditions
Road_Condition	Road surface conditions
Light_Condition	Lighting conditions (day/night)
Vehicle_Type	Type of vehicle involved
Severity	Accident severity (Minor, Serious, Fatal)
🛠️ Tech Stack

Python

Streamlit

Pandas / NumPy

Scikit-learn

Matplotlib / Seaborn

<img width="1812" height="807" alt="ss2" src="https://github.com/user-attachments/assets/4d49c090-f734-4ced-927a-53ec43bc7962" />
<img width="1892" height="795" alt="ss1" src="https://github.com/user-attachments/assets/1b0aac30-2331-40d2-8563-b5661689e9d7" />
<img width="1858" height="790" alt="ss3" src="https://github.com/user-attachments/assets/d3254c7b-9419-485f-89dd-85b407a78b36" />
