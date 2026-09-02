# 🚗 Car Price Prediction with Machine Learning

A machine learning project that predicts the **selling price of used cars** based on vehicle characteristics such as present price, car age, mileage, fuel type, seller type, transmission, and ownership.

This project was developed as part of the **EXPERT PETROLEUM SERVICES internship** and focuses on the complete machine learning workflow, from data cleaning and exploratory data analysis to model training and deployment with Streamlit.

---

## 📌 Project Overview

Used car prices depend on multiple factors, including the original price, vehicle age, mileage, fuel type, transmission, and ownership history.

The objective of this project is to build a regression model capable of estimating a car's selling price from these features.

The project covers:

- Data cleaning and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Regression model training
- Model comparison and tuning
- Model evaluation
- Model serialization
- Interactive prediction using Streamlit

---

## 🎯 Objective

Build a machine learning regression model that predicts:

> **Selling Price of a Used Car**

The final model is a **tuned XGBoost regression pipeline**.

---

## 📊 Dataset

The dataset contains **301 car records**, with information about used cars and their selling prices.

### Main Features

| Feature | Description |
|---|---|
| `Car_Name` | Name of the car |
| `Year` | Manufacturing year |
| `Selling_Price` | Selling price of the car — target variable |
| `Present_Price` | Current/original market price |
| `Kms_Driven` | Kilometers driven |
| `Fuel_Type` | Petrol or Diesel |
| `Seller_Type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

### Engineered Features

#### `Car_Age`

Calculated from the manufacturing year using **2018 as the reference year**:

```text
Car_Age = 2018 - Year
