District-Level Rainfall Prediction for IndiaThis project uses a hybrid machine learning approach to predict seasonal (Kharif & Rabi) rainfall for every district in India. It leverages a Long Short-Term Memory (LSTM) model trained on a downscaled, district-specific time series.The project is deployed as a Flask web application that exposes an API endpoint for real-time predictions.!Image of a weather forecast map of IndiaProject StructureThe repository is organized as follows:Rainfall/
├── data/
│   ├── district_master_map.csv
│   ├── district wise rainfall normal.csv
│   └── rainfall in india 1901-2015.csv
├── models/
│   └── (This folder will be populated by the training script)
├── scalers/
│   └── (This folder will also be populated by the training script)
├── templates/
│   └── index.html
├── app.py
├── train_rainfall_models.py
├── requirements.txt
└── README.md
data/: Contains the raw data files.models/ & scalers/: Store the trained, district-specific LSTM models and their corresponding scalers.templates/: Contains the HTML frontend.train_rainfall_models.py: A one-time script to train all models.app.py: The Flask web server that serves the frontend and the prediction API.requirements.txt: A list of all necessary Python libraries.How to Run This Project1. SetupClone the repository:git clone <your-repository-url>
cd Rainfall
Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the dependencies:pip install -r requirements.txt
2. Train the ModelsBefore you can run the web application, you must train the models for all districts. This is a one-time process.Run the training script from your terminal:python train_rainfall_models.py
This will take some time. Once it's finished, your models/ and scalers/ folders will be filled with the necessary files.3. Run the Web ApplicationAfter the training is complete, you can start the Flask server.python app.py
Now, open your web browser and go to the following address to see the user interface:http://127.0.0.1:5000API UsageThe application exposes an API endpoint to get rainfall predictions programmatically.URL: /predict-rainfallMethod: GETQuery Parameter: district (e.g., ?district=PUNE)Example Request:[http://127.0.0.1:5000/predict-rainfall?district=PUNE](http://127.0.0.1:5000/predict-rainfall?district=PUNE)
Example Success Response:{
  "district": "PUNE",
  "predicted_kharif_rainfall_mm": 635.78,
  "predicted_rabi_rainfall_mm": 88.12
}
