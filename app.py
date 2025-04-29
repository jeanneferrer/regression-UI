import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from geopy.distance import geodesic
import ssl
import certifi
import os

app = Flask(__name__)
CORS(app)

# Declare trained_model at the top (global scope)
trained_model = None

# Load the trained model (we'll train it once when the app starts)
def train_model():
    df_train = pd.read_csv('uploaded_dataset.csv') # Placeholder, will be updated
    update_column_name(df_train)
    extract_column_value(df_train)
    drop_columns(df_train)
    update_datatype(df_train)
    convert_nan(df_train)
    handle_null_values(df_train)
    extract_date_features(df_train)
    df_train = calculate_travel_time(df_train)
    label_encoding(df_train)
    X = df_train.drop(['Time_taken(min)', 'Order_Date'], axis=1)
    y = df_train['Time_taken(min)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42) # Added random_state for reproducibility
    model.fit(X_train, y_train)
    return model

def download_file(url, filename):
    if not os.path.exists(filename):
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as response, open(filename, "wb") as out_file:
            out_file.write(response.read())
        print(f"Downloaded: {filename}")

def update_column_name(df):
    df.rename(columns={'Weatherconditions':'Weather_conditions'},inplace=True)

def extract_column_value(df):
    def safe_extract_time(x):
        try:
            if pd.isnull(x) or str(x).strip() == '':
                return np.nan
            # Remove '(min)' and strip whitespace, then convert
            cleaned_x = str(x).replace('(min)', '').strip()
            return int(cleaned_x)
        except Exception:
            return np.nan

    def safe_extract_weather(x):
        try:
            if pd.isnull(x) or str(x).strip() == '':
                return np.nan
            return str(x).replace('conditions', '').strip()
        except Exception:
            return np.nan
    
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(safe_extract_time)
    df['Weather_conditions'] = df['Weather_conditions'].apply(safe_extract_weather)
    df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]

def drop_columns(df):
    df.drop(['ID','Delivery_person_ID'],axis=1,inplace=True)

def update_datatype(df):
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%m/%d/%Y")

def convert_nan(df):
    df.replace('NaN', float(np.nan), regex=True,inplace=True)

def handle_null_values(df):
    for col in ['Delivery_person_Age', 'Weather_conditions', 'Festival', 'multiple_deliveries', 'Road_traffic_density']:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(np.random.choice(df[col].dropna()), inplace=True)
    if 'Delivery_person_Ratings' in df.columns:
        df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)

def extract_date_features(data):
    if 'Order_Date' in data.columns:
        data["day"] = data.Order_Date.dt.day
        data["month"] = data.Order_Date.dt.month
        data["quarter"] = data.Order_Date.dt.quarter
        data["year"] = data.Order_Date.dt.year
        data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
        data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
        data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
        data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
        data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
        data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
        data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
        data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]),1,0)

def calculate_travel_time(df):
    if all(col in df.columns for col in ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Weather_conditions', 'Road_traffic_density']):
        df['distance_km'] = df.apply(lambda row: geodesic((row['Restaurant_latitude'], row['Restaurant_longitude']),
                                                          (row['Delivery_location_latitude'], row['Delivery_location_longitude'])).km, axis=1)
        weather_speed_mapping = {'Sunny': 1.2, 'Cloudy': 1.0, 'Windy': 0.9, 'Fog': 0.7, 'Stormy': 0.5}
        traffic_speed_mapping = {'Low': 1.2, 'Medium': 1.0, 'High': 0.8, 'Jam': 0.5}
        average_speed_km_min = 0.8
        df['base_travel_time_min'] = df['distance_km'] / average_speed_km_min
        df['weather_modifier'] = df['Weather_conditions'].map(weather_speed_mapping).fillna(1.0)
        df['traffic_modifier'] = df['Road_traffic_density'].map(traffic_speed_mapping).fillna(1.0)
        df['estimated_travel_time_min'] = df['base_travel_time_min'] * df['weather_modifier'] * df['traffic_modifier']
        df['estimated_travel_time_min'] = df['estimated_travel_time_min'].astype(int)
        average_shelf_life_min = 1440
        df['estimated_time_of_arrival'] = df['estimated_travel_time_min'] - average_shelf_life_min
    return df

def label_encoding(df):
    categorical_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))

def preprocess_new_data(df):
    update_column_name(df)
    extract_column_value(df)
    drop_columns(df)
    update_datatype(df)
    convert_nan(df)
    handle_null_values(df)
    extract_date_features(df)
    df = calculate_travel_time(df)
    label_encoding(df)
    return df

@app.route('/', methods=['GET'])
def index():
    return 'Flask server running!!!'
    # return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            new_df = pd.read_csv(file)
            processed_df = preprocess_new_data(new_df.copy())

            X_new = processed_df.drop(['Time_taken(min)', 'Order_Date'], axis=1, errors='ignore')
            model_features = trained_model.feature_names_in_
            missing_cols = set(model_features) - set(X_new.columns)
            for c in missing_cols:
                X_new[c] = 0
            extra_cols = set(X_new.columns) - set(model_features)
            X_new = X_new[model_features]

            predictions = trained_model.predict(X_new)
            predictions_list = [round(pred, 2) for pred in predictions]
            return jsonify({'predictions': predictions_list})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format. Only .csv files are allowed'}), 400

if __name__ == '__main__':
    with app.app_context():
        dummy_data = {
            'ID': [1, 2], 'Delivery_person_ID': ['RES1', 'RES2'], 'Delivery_person_Age': [25, 30],
            'Delivery_person_Ratings': [4.5, 4.8], 'Restaurant_latitude': [12.0, 12.1],
            'Restaurant_longitude': [100.0, 100.1], 'Delivery_location_latitude': [12.2, 12.3],
            'Delivery_location_longitude': [100.2, 100.3], 'Order_Date': ['04/16/2025', '04/16/2025'],
            'Time_taken(min)': ['30 mins', '35 mins'], 'Weatherconditions': ['Sunny 24', 'Cloudy 22'],
            'Road_traffic_density': ['Low', 'Medium'], 'Vehicle_condition': ['good', 'good'],
            'Type_of_order': ['Snack', 'Meal'], 'Type_of_vehicle': ['motorcycle', 'scooter'],
            'multiple_deliveries': [1, 2], 'Festival': ['No', 'No'], 'City': ['A', 'B']
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv('uploaded_dataset.csv', index=False)
        trained_model = train_model()
        os.remove('uploaded_dataset.csv')

    app.run(debug=True, port=5000) # Keep the backend on port 5000