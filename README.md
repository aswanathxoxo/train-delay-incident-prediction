# train-delay-incident-prediction

# 📘 **Train Delay Incident Prediction (TNR) – Machine Learning System**

MSc Applied Artificial Intelligence – 7DATA001W**
Author:** Aswanath Jayanath Sumi – 21660070


## ⭐ **Project Overview**

This project builds a complete Machine Learning system to **predict train incidents** before they occur, helping TransNational Railways (TNR) reduce avoidable delays and operational costs.

The workflow includes:

* Data ingestion
* Preprocessing (cleaning, missing value imputation, outlier handling)
* Feature engineering
* Model training & evaluation
* Hyperparameter tuning
* MLflow experiment tracking
* Model saving
* API deployment (FastAPI)

The final deployed model is a **Tuned XGBoost Classifier**, achieving the best performance in detecting incidents.

---

## 📁 **Project Structure**




ML_COURSEWORK
│
├── data/
│     └── TNR_Data.csv
│
├── mlruns/                         <-- MLflow experiment tracking folder
│
├── model_artifacts/
│     ├── tuned_xgboost_v1_0/       <-- your saved tuned model (MLflow format)
│     └── model.pkl                 <-- (likely manually exported)
│
├── Notebook/                       <-- Jupyter notebook outputs
│     └── api.py                          <-- FastAPI app for model deployment
├── requirements.txt





## 🧹 **Data Preprocessing Summary**

### ✦ 1. Missing Value Handling

* `dwell_time` was missing in **6,365 rows**.
* Missingness was identified as **MAR (Missing At Random)**.
* Imputed using **Linear Regression** with strong predictors:

  * `on_train_bookings`
  * `on_train_forecast`

### ✦ 2. Outlier Detection

* IQR method selected (robust, non-parametric, dataset skewed).
* Outliers capped only for relevant features.

### ✦ 3. Feature Engineering

* OneHotEncoding for categorical features
* StandardScaler for numeric features
* Fully integrated inside **ColumnTransformer**
* Avoids leakage and ensures reproducibility

---

## 🤖 **Model Training**

Three baseline models were implemented:

XGBoost performed best, so it was tuned using:

* **RandomizedSearchCV**
* 5-fold **Stratified CV**
* Scoring metric: **F1-score**



## 📊 **MLflow Experiment Tracking**

MLflow is used for:

* Logging hyperparameters
* Logging accuracy, precision, recall, F1
* Saving the entire preprocessing + model pipeline
* Comparing baseline vs tuned models

### Start MLflow UI:

```
% mlflow ui --backend-store-uri mlruns
```

Then open:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📦 **Saved Model (for Deployment)**

The final tuned model is saved at:

```
model_artifacts/tuned_xgboost_v1.0/
```


## 🌐 **API Deployment (FastAPI)**

The file **api.py** loads the saved model and exposes a prediction endpoint.

### Run the API:

```
uvicorn api:app --reload --host 127.0.0.1 --port 8001
```

### Example Request (POST)

```
POST http://127.0.0.1:8001/predict
Content-Type: application/json
```

Example JSON:

```json
{
  "origin": "DON",
  "dest": "KGX",
  "temp": 12.5,
  "rain_1h": 0.2,
  "dwell_time": 3.5,
  ...
}
```

### Example Response:

```json
{
  "prediction": 1,
  "message": "Incident likely"
}
```

---

## 🔧 **Installation**

### 1. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run notebook or API

---

## 🧪 **Testing**

A basic functional test ensures correct output shape:

```python
assert predictions.ndim == 1  
assert len(predictions) == len(X_input)
```

Test passed ✔️

---

## 📝 **Key Technologies Used**

* Python
* Scikit-learn
* XGBoost
* Pandas & NumPy
* FastAPI
* MLflow
* Matplotlib / Seaborn

---

## 🎯 Final Remarks

This project implements a complete ML lifecycle:

✔ Data processing
✔ Feature engineering
✔ Baseline & tuned models
✔ Validation & visualisation
✔ MLflow logging
✔ Deployment-ready API

The **Tuned XGBoost model** is the final chosen solution due to superior F1 performance and strong ROC-AUC results.

---





































































