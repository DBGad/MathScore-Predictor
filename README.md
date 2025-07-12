
# 📊 Math Score Predictor

An end-to-end Machine Learning project for predicting students’ math scores based on demographic and academic features. This repo includes all production-level components: data pipelines, model training and evaluation scripts, logging, exception handling, and deployment through a Flask web app.

---

## 🔍 Project Overview

1. **Production-Ready Data Ingestion**  
   - Custom Python class (`DataIngestion`) reads the raw data from CSV.
   - Automatically stores raw, train, and test splits in `artifacts/` directory.
   - Uses logging and exception handling for traceability.

2. **Exploratory Data Analysis (EDA)**  
   - Conducted separately but not inside the repository.
   - Informed feature engineering decisions and model selection.

3. **Feature Engineering & Data Transformation**  
   - Feature: `average_score = (math + reading + writing) / 3`.
   - Encoded categorical variables using OneHotEncoding.
   - Saved the transformer object for reuse during prediction.

4. **Model Training Pipeline**  
   - Used `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`.
   - Scripts are modular (`model_trainer.py`) and save trained model as a `.pkl` file in `artifacts/`.
   - Includes evaluation logic: *MAE*, *RMSE*, *R²*.

5. **Flask Deployment**  
   - Web interface to collect user inputs.
   - Predicts and displays math score in real-time.
   - Routes:
     - `/` — index page
     - `/predict` — form submission and result display

6. **Modular Project Structure**
   - `src/` contains components:
     - `data_ingestion.py`, `data_transformation.py`, `model_trainer.py`, `predict_pipeline.py`
   - `logger.py` and `exception.py` for clean debugging
   - `app.py` for hosting the prediction web app

---

## ⚙️ Getting Started

### 📋 Prerequisites
- Python 3.8+
- `virtualenv` or `venv`

### 🛠️ Installation

```bash
git clone https://github.com/DBGad/MathScore-Predictor.git
cd MathScore-Predictor
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 🚀 Run the Pipeline

```bash
# Launch Flask app
python app.py
```

Go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📈 Screenshots

- **Home page**

  ![Home Page Screenshot](https://github.com/user-attachments/assets/7b0fec4d-e236-478b-a344-acee4b01c4fb)

- **Prediction Page**

  ![Predict Page Screenshot](https://github.com/user-attachments/assets/33796609-7c09-4c47-ab3d-e27849ce1442)

---

## 🧠 Model Performance

| Model            | MAE   | RMSE  | R²     |
|------------------|-------|--------|--------|
| Linear Regression| 3.90  | 5.18   | 0.872  |
| Ridge Regression | 3.89  | 5.17   | 0.872  |
| Lasso Regression | 3.95  | 5.22   | 0.869  |
| ElasticNet       | 3.94  | 5.21   | 0.870  |

>  Linear Regression Regression was selected 

---

## 📂 Project Structure

```
MathScore-Predictor/
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipelines/
│   │   ├── predict_pipeline.py
│   ├── logger.py
│   ├── exception.py
├── templates/
│   ├── index.html
│   ├── predict.html
├── static/
├── app.py
├── requirements.txt
└── README.md
```

---

## 👤 Author & Contact

**Gad Amr** – Data Scientist  
Cairo University  
[LinkedIn](https://www.linkedin.com/in/gaadamr/)

