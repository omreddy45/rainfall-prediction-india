# 🌧️ Rainfall Prediction in Indian Districts

## 📌 Overview
This project provides **district-level rainfall prediction** for India using historical rainfall data (1901–2015).  
We train **LSTM (Long Short-Term Memory) deep learning models** for each district and season (Kharif & Rabi), then deploy them using a **Flask backend API** with a simple web interface.

The project uses subdivision-level historical data and district-wise normals from IMD to create **synthetic district histories** using scaling factors. Separate models are trained for **Kharif (Jun–Sep)** and **Rabi (Oct–Dec)** rainfall.

---

## 📂 Project Structure

-   **`data/`**: Contains the raw data files.
-   **`models/` & `scalers/`**: Store the trained, district-specific LSTM models and their corresponding scalers.
-   **`templates/`**: Contains the HTML frontend.
-   **`train_rainfall_models.py`**: A one-time script to train all models.
-   **`app.py`**: The Flask web server that serves the frontend and the prediction API.
-   **`requirements.txt`**: A list of all necessary Python libraries.

---

### 🧪 Methodology

The core of this project is a hybrid **"scaling factor" approach** that overcomes the limitation of having historical data only at the meteorological subdivision level.

1. **Calculate Scaling Factor:**  
   For each district, a unique scaling factor is calculated by comparing its long-term average seasonal rainfall (`district_wise_rainfall_normal.csv`) to the average rainfall of its parent subdivision.

2. **Create District-Specific History:**  
   This factor is then applied to the subdivision's entire 115-year rainfall history (`rainfall_in_india_1901-2015.csv`) to generate a more realistic, localized time series for the district.

3. **Train Specialist Models:**  
   A separate LSTM model is trained on this unique historical data for every district, turning each model into a specialist for its specific location.

---

## 💻 Tech Stack

This project is built using the following technologies:

**Backend:**
- **Python**: The core programming language.
- **Flask**: A micro web framework for serving the API and the frontend.

**Machine Learning & Data Science:**
- **TensorFlow (with Keras)**: For building and training the LSTM neural network models.
- **Scikit-learn**: Used for data preprocessing (`MinMaxScaler`) and saving models (`joblib`).
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.

**Frontend:**
- **HTML**: For the basic structure of the user interface.
- **JavaScript**: For handling form submission and fetching predictions from the API.

---
## 🚀 How to Run This Project

### 1. Setup

**Clone the repository:**
```bash
git clone < https://github.com/omreddy45/rainfall-prediction-india.git>
cd <rainfall-prediction-india>
```

Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
### 2. Train the Models

Before you can run the web application, you must train the models for all districts.
This is a one-time process that will populate the models/ and scalers/ folders.

Run the training script from your terminal:
```bash
python train_rainfall_models.py
```
Note: This process can take a significant amount of time as it trains hundreds of models.

### 3. Run the Web Application

After the training is complete, start the Flask server:
```bash
python app.py
```

Once the server is running, it will automatically open a web interface in your browser.
Here, you can enter district names to get rainfall predictions for that district.

---
## 📜 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🌐 Data Sources

- [Rainfall in India (1901–2015) Dataset](https://www.kaggle.com/datasets/ashishjstar/rainfall-in-india-1901-2015)
- [Indian Rainfall Analysis and Prediction Notebook by Anbarivan N.L.](https://www.kaggle.com/code/anbarivan/indian-rainfall-analysis-and-prediction)
