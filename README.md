# 💨 AIROX: Real-Time Air Quality Prediction System

A **Machine Learning** project that predicts air quality categories (**Good**, **Moderate**, **Poor**) in real-time using a **CatBoost Classifier**.  
This project demonstrates a complete end-to-end **MLOps workflow** — from robust data cleaning and modeling to a stylish **Streamlit deployment**.

---

## 🌟 Overview

The **AIROX Prediction System** aims to accurately estimate air quality by analyzing crucial environmental inputs, including:

- Pollutant concentrations ($\text{PM}_{2.5}$, $\text{CO}$, $\text{NO}_2$, etc.) 😷  
- Meteorological factors (Temperature, Humidity) 🌡️  
- Urban and industrial proximity 🏭  

The final product is a **sleek, black and gold Streamlit dashboard** allowing users to interactively test different environmental scenarios.

---

## ⚙️ Project Workflow

### 🧩 Data Preparation

- **🧹 Data Cleaning:** Implemented robust handling for impossible readings, including clamping negative pollutant values to zero and capping humidity at $100\%$.  
- **⚖️ Scaling:** Used a `RobustScaler` to normalize feature distributions against outliers, improving model stability.  
- **✂️ Data Splitting:** Divided data into training and testing sets.  

---

### 📊 Exploratory Data Analysis (EDA)

- **🔗 Correlation Matrix:** Analyzed feature-to-feature and feature-to-target relationships.  
- **📈 Distribution Plots:** Visualized univariate distributions of pollutants and metrics (e.g., using Violin Plots).  
- **🎯 Error Analysis:** Used Confusion Matrices and ROC-AUC curves to identify misclassifications (e.g., confusion between *Moderate* and *Poor* air quality).  

---

### 🤖 Model Building

- **💪 Model:** Trained a **CatBoost Classifier**, chosen for its excellent performance on structured data and native handling of categorical features.  
- **🧠 Evaluation Metrics:**
  - Accuracy : 99%  
  - ROC AUC (Weighted) : 1.00

---

### 💾 Deployment & Production

- **💾 Model Saving:** Stored the trained CatBoost model (`AiroX.pkl`) and fitted scaler (`RobustScaler.pkl`) using `joblib`.  
- **💻 Deployment:** Deployed the prediction logic via a custom **Streamlit** app (`app.py`).  

---
