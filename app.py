from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# LOAD ASSETS FOR RAINFALL PREDICTION
print("Loading assets for rainfall prediction...")
LOOK_BACK = 10
try:
    # Load Data Files
    master_map = pd.read_csv("data/district_master_map.csv")
    rainfall_df = pd.read_csv("data/rainfall in india 1901-2015.csv")
    normals_df = pd.read_csv("data/district wise rainfall normal.csv")
    # NOTE: We are NOT loading the crop_model in this version

    # Clean and Prepare Data
    for df in [rainfall_df, normals_df, master_map]:
        if 'SUBDIVISION' in df.columns:
            df['SUBDIVISION'] = df['SUBDIVISION'].str.replace('&', 'And').str.upper().str.strip()
            df['SUBDIVISION'] = df['SUBDIVISION'].replace('MATATHWADA', 'MARATHWADA')
        if 'STATE_UT_NAME' in df.columns:
            df.rename(columns={'STATE_UT_NAME': 'STATE'}, inplace=True)

    master_map['DISTRICT'] = master_map['DISTRICT'].str.upper().str.strip()
    normals_df['DISTRICT'] = normals_df['DISTRICT'].str.upper().str.strip()
    normals_df.fillna(normals_df.mean(numeric_only=True), inplace=True)
    rainfall_df.fillna(rainfall_df.mean(numeric_only=True), inplace=True)

    # Calculate Scaling Factors
    rainfall_df['Kharif_Rainfall'] = rainfall_df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
    rainfall_df['Rabi_Rainfall'] = rainfall_df[['OCT', 'NOV', 'DEC']].sum(axis=1)
    subdivision_normals = rainfall_df.groupby('SUBDIVISION')[['Kharif_Rainfall', 'Rabi_Rainfall']].mean().reset_index()
    subdivision_normals.rename(
        columns={'Kharif_Rainfall': 'Subdivision_Kharif_Normal', 'Rabi_Rainfall': 'Subdivision_Rabi_Normal'},
        inplace=True)
    district_normals = normals_df[['DISTRICT', 'Jun-Sep', 'Oct-Dec']]
    district_normals.rename(columns={'Jun-Sep': 'District_Kharif_Normal', 'Oct-Dec': 'District_Rabi_Normal'},
                            inplace=True)
    scaling_factors_df = pd.merge(master_map, district_normals, on='DISTRICT')
    scaling_factors_df = pd.merge(scaling_factors_df, subdivision_normals, on='SUBDIVISION')
    scaling_factors_df['Kharif_Factor'] = scaling_factors_df['District_Kharif_Normal'] / scaling_factors_df[
        'Subdivision_Kharif_Normal']
    scaling_factors_df['Rabi_Factor'] = scaling_factors_df['District_Rabi_Normal'] / scaling_factors_df[
        'Subdivision_Rabi_Normal']
    scaling_factors_df.fillna(1, inplace=True)

    print("Rainfall assets loaded successfully.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load a required asset. {e}")
    master_map, rainfall_df, scaling_factors_df = None, None, None


# HELPER FUNCTIONS
def sanitize_filename(name):
    # Sanitize district names for use in file paths
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_').replace('(', '').replace(')', '')


def predict_rainfall_for_district(district_name, season):
    safe_district_name = sanitize_filename(district_name)
    district_info = scaling_factors_df[scaling_factors_df['DISTRICT'] == district_name]
    if district_info.empty:
        raise ValueError(f"District '{district_name}' not found.")

    district_info = district_info.iloc[0]
    subdivision = district_info['SUBDIVISION']
    factor = district_info['Kharif_Factor'] if season == 'Kharif' else district_info['Rabi_Factor']

    model_path = f"models/{safe_district_name}_{season.lower()}_model.keras"
    scaler_path = f"scalers/{safe_district_name}_{season.lower()}_scaler.pkl"
    rainfall_model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    subdivision_history = rainfall_df[rainfall_df['SUBDIVISION'] == subdivision]
    seasonal_col = f'{season}_Rainfall'
    last_10_years_subdivision = subdivision_history[[seasonal_col]].values[-LOOK_BACK:]
    last_10_years_district = last_10_years_subdivision * factor

    input_data = scaler.transform(last_10_years_district)
    input_data = np.reshape(input_data, (1, LOOK_BACK, 1))
    scaled_prediction = rainfall_model.predict(input_data, verbose=0)
    final_prediction = scaler.inverse_transform(scaled_prediction)[0][0]

    return round(float(final_prediction), 2)


# API ENDPOINTS
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/predict-rainfall', methods=['GET'])
def predict_rainfall_endpoint():
    if scaling_factors_df is None:
        return jsonify({'error': 'Server is not ready. Assets could not be loaded.'}), 503

    district_name = request.args.get('district', '').upper()
    if not district_name:
        return jsonify({'error': 'District parameter is required.'}), 400

    try:
        kharif_rain = predict_rainfall_for_district(district_name, 'Kharif')
        rabi_rain = predict_rainfall_for_district(district_name, 'Rabi')

        return jsonify({
            'district': district_name,
            'predicted_kharif_rainfall_mm': kharif_rain,
            'predicted_rabi_rainfall_mm': rabi_rain
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except FileNotFoundError:
        return jsonify({'error': f"Prediction model for district '{district_name}' not found."}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

