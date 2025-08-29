import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
import re
import warnings

warnings.filterwarnings('ignore')

# SETUP
LOOK_BACK = 10
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# LOAD DATA
print("Loading all data sources...")
try:
    rainfall_df = pd.read_csv("data/rainfall in india 1901-2015.csv")
    normals_df = pd.read_csv("data/district wise rainfall normal.csv")
    map_df = pd.read_csv("data/district_master_map.csv")
except FileNotFoundError as e:
    print(f"Error: Could not find a required data file. {e}")
    exit()

# DATA CLEANING
print("Cleaning and preparing data...")
for df in [rainfall_df, normals_df, map_df]:
    if 'SUBDIVISION' in df.columns:
        df['SUBDIVISION'] = df['SUBDIVISION'].str.replace('&', 'And').str.upper().str.strip()
        df['SUBDIVISION'] = df['SUBDIVISION'].replace('MATATHWADA', 'MARATHWADA')
    if 'STATE_UT_NAME' in df.columns:
        df.rename(columns={'STATE_UT_NAME': 'STATE'}, inplace=True)

normals_df['DISTRICT'] = normals_df['DISTRICT'].str.upper().str.strip()
map_df['DISTRICT'] = map_df['DISTRICT'].str.upper().str.strip()
normals_df.fillna(normals_df.mean(numeric_only=True), inplace=True)
rainfall_df.fillna(rainfall_df.mean(numeric_only=True), inplace=True)

# CALCULATE SCALING FACTORS
print("Calculating district-level rainfall scaling factors...")
# Calculate historical seasonal rainfall
rainfall_df['Kharif_Rainfall'] = rainfall_df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
rainfall_df['Rabi_Rainfall'] = rainfall_df[['OCT', 'NOV', 'DEC']].sum(axis=1)

# Calculate the long-term average (normal) for each subdivision from the historical data
subdivision_normals = rainfall_df.groupby('SUBDIVISION')[['Kharif_Rainfall', 'Rabi_Rainfall']].mean().reset_index()
subdivision_normals.rename(
    columns={'Kharif_Rainfall': 'Subdivision_Kharif_Normal', 'Rabi_Rainfall': 'Subdivision_Rabi_Normal'}, inplace=True)

# Get the normals for each district
district_normals = normals_df[['DISTRICT', 'Jun-Sep', 'Oct-Dec']]
district_normals.rename(columns={'Jun-Sep': 'District_Kharif_Normal', 'Oct-Dec': 'District_Rabi_Normal'}, inplace=True)

# Merge to create the scaling factor dataframe
scaling_factors_df = pd.merge(map_df, district_normals, on='DISTRICT')
scaling_factors_df = pd.merge(scaling_factors_df, subdivision_normals, on='SUBDIVISION')

# Calculate the scaling factor as the ratio of district normal to subdivision normal
scaling_factors_df['Kharif_Factor'] = scaling_factors_df['District_Kharif_Normal'] / scaling_factors_df[
    'Subdivision_Kharif_Normal']
scaling_factors_df['Rabi_Factor'] = scaling_factors_df['District_Rabi_Normal'] / scaling_factors_df[
    'Subdivision_Rabi_Normal']
scaling_factors_df.fillna(1, inplace=True)  # Default to 1 if a value is missing


# HELPER FUNCTIONS
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def build_model(input_shape):
    model = Sequential([LSTM(50, return_sequences=True, input_shape=input_shape), LSTM(50), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def sanitize_filename(name):
    """Removes special characters from a string to make it a valid filename."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')


# TRAINING LOOP (PER DISTRICT)
districts_to_train = scaling_factors_df['DISTRICT'].unique()
print(f"Found {len(districts_to_train)} districts to train. Starting or resuming loop...")

for district in districts_to_train:
    safe_district_name = sanitize_filename(district)

    # RESUME LOGIC
    final_model_path = f"models/{safe_district_name}_rabi_model.keras"
    if os.path.exists(final_model_path):
        print(f"--- Models for District: {district} already exist. Skipping. ---")
        continue

    print(f"\n--- Training models for District: {district} ---")

    district_info = scaling_factors_df[scaling_factors_df['DISTRICT'] == district].iloc[0]
    subdivision = district_info['SUBDIVISION']
    kharif_factor = district_info['Kharif_Factor']
    rabi_factor = district_info['Rabi_Factor']

    subdivision_history = rainfall_df[rainfall_df['SUBDIVISION'] == subdivision]
    if len(subdivision_history) < 20:
        print(f"Skipping {district} (insufficient subdivision data).")
        continue

    # Create synthetic district history using the scaling factor
    district_history = pd.DataFrame({
        'Kharif_Rainfall': subdivision_history['Kharif_Rainfall'] * kharif_factor,
        'Rabi_Rainfall': subdivision_history['Rabi_Rainfall'] * rabi_factor
    })

    # Train and Save Kharif Model
    kharif_data = district_history[['Kharif_Rainfall']].values
    scaler_kharif = MinMaxScaler(feature_range=(0, 1))
    X_kharif, y_kharif = create_dataset(scaler_kharif.fit_transform(kharif_data), LOOK_BACK)
    if len(X_kharif) > 0:
        X_kharif = np.reshape(X_kharif, (X_kharif.shape[0], X_kharif.shape[1], 1))
        kharif_model = build_model(input_shape=(LOOK_BACK, 1))
        kharif_model.fit(X_kharif, y_kharif, epochs=100, batch_size=32, verbose=0)
        kharif_model.save(f"models/{safe_district_name}_kharif_model.keras")
        joblib.dump(scaler_kharif, f"scalers/{safe_district_name}_kharif_scaler.pkl")
        print(f"  > Kharif model and scaler saved.")

    # Train and Save Rabi Model
    rabi_data = district_history[['Rabi_Rainfall']].values
    scaler_rabi = MinMaxScaler(feature_range=(0, 1))
    X_rabi, y_rabi = create_dataset(scaler_rabi.fit_transform(rabi_data), LOOK_BACK)
    if len(X_rabi) > 0:
        X_rabi = np.reshape(X_rabi, (X_rabi.shape[0], X_rabi.shape[1], 1))
        rabi_model = build_model(input_shape=(LOOK_BACK, 1))
        rabi_model.fit(X_rabi, y_rabi, epochs=100, batch_size=32, verbose=0)
        rabi_model.save(f"models/{safe_district_name}_rabi_model.keras")
        joblib.dump(scaler_rabi, f"scalers/{safe_district_name}_rabi_scaler.pkl")
        print(f"  > Rabi model and scaler saved.")

print("\n--- All district-level rainfall models trained successfully! ---")
