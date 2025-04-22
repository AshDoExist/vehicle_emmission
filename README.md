# Vehicle Emission Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the **emission category** of vehicles using supervised machine learning techniques. As environmental concerns continue to grow, it is crucial to classify vehicles based on their emissions accurately. This helps with enforcing regulations, supporting eco-friendly initiatives, and informing consumers.

Using a dataset containing vehicle information such as **engine size**, **fuel type**, and **CO2 emissions**, a **Random Forest Classifier** is trained to predict the emission category.

---

## 📊 Problem Statement

To build a machine learning model that predicts a vehicle's **emission category** based on input features such as:
- Engine Size
- Fuel Type
- CO2 Emissions

The goal is to assist regulators and manufacturers in identifying high-emission vehicles and promoting cleaner alternatives.

---

## 🎯 Objectives

- Preprocess the dataset for machine learning.
- Train a **Random Forest Classifier** to classify emission categories.
- Evaluate the model using standard classification metrics.
- Visualize model performance using a **confusion matrix heatmap**.
- Allow user input for live prediction of emission category.

---

## 🧪 Methodology

### 1. **Data Collection**
- A CSV file (`vehicle_emissions.csv`) containing vehicle data is used as input.

### 2. **Data Preprocessing**
- Missing values handled using mean imputation.
- Categorical variables encoded using `LabelEncoder`.
- Data split into training (80%) and testing (20%) sets.

### 3. **Model Building**
- A `RandomForestClassifier` from `scikit-learn` is used.
- Model is trained on the training set and evaluated on the test set.

### 4. **Model Evaluation**
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- Confusion matrix visualized with **Seaborn heatmap**.

### 5. **Prediction Interface**
- Accepts user input for a new vehicle’s engine size, fuel type, and CO2 emissions.
- Predicts the vehicle’s emission category.

---

## 📈 Evaluation Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Confusion Matrix**: Visual insight into classification errors

---

## 📊 Visualizations

- **Confusion Matrix**: Heatmap showing actual vs predicted labels
- **Feature Importances**: Bar plot showing the impact of each input feature
- **Category Distribution**: Count plot showing emission category frequency in the dataset

---

## 🖥️ Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 How to Run

1. Clone the repository or download the script.
2. Make sure you have the required packages installed:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
