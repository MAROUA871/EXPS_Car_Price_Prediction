# 🚗 Car Price Prediction with Machine Learning

A machine learning project that predicts the **selling price of used vehicles** based on characteristics such as present price, vehicle age, mileage, brand, fuel type, seller type, transmission, and ownership.

This project was developed as part of the **EXPERT PETROLEUM SERVICES internship** and demonstrates an end-to-end machine learning workflow, including data cleaning, feature engineering, exploratory data analysis, preprocessing, regression model comparison, hyperparameter tuning, model serialization, and deployment with Streamlit.

---

## 📌 Project Overview

Used vehicle prices depend on several factors, including the original/current market price, vehicle age, mileage, brand, fuel type, transmission, and ownership history.

The objective of this project is to build a regression model capable of estimating the selling price of a used vehicle from these characteristics.

The project covers:

- Data cleaning and validation
- Duplicate removal
- Feature engineering
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Categorical feature encoding
- Numerical feature scaling
- Regression model comparison
- Hyperparameter tuning
- Model evaluation
- Target leakage prevention
- Model serialization with Joblib
- Interactive prediction with Streamlit

---

## 🎯 Objective

Build a machine learning regression model that predicts:

> **Selling Price of a Used Vehicle**

The final model is a **tuned XGBoost regression pipeline** combining preprocessing and the trained regression model.

---

## 📊 Dataset

The dataset contains **301 original observations** and **9 original features**.

After identifying and removing **2 exact duplicate rows**, the final dataset used for analysis and modeling contains:

> **299 observations**

The data contains Indian used-vehicle resale information, with prices expressed in **lakh INR**.

### Original Features

| Feature | Description |
|---|---|
| `Car_Name` | Vehicle/model name |
| `Year` | Manufacturing year |
| `Selling_Price` | Selling price — target variable |
| `Present_Price` | Current/original market price |
| `Kms_Driven` | Kilometers driven |
| `Fuel_Type` | Fuel type |
| `Seller_Type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

### Dataset Characteristics

| Property | Value |
|---|---:|
| Original observations | 301 |
| Duplicate rows removed | 2 |
| Final observations | 299 |
| Original features | 9 |
| Manufacturing years | 2003–2018 |
| Selling price range | 0.10–35.00 lakh INR |
| Present price range | 0.32–92.60 lakh INR |
| Kilometers driven | 500–500,000 |
| Previous owners | 0–3 |

---

## 🧹 Data Cleaning & Preprocessing

### Duplicate Removal

Two exact duplicate observations were identified and removed.

The dataset was then reset to a clean index, resulting in:

```text
301 → 299 observations
```

No valid vehicle observations were removed solely because of unusual values.

### Missing Values

After duplicate removal:

```text
Missing values = 0
```

Therefore, no missing-value imputation was required.

### Categorical Validation

The final categorical distributions include:

**Fuel Type**

```text
Petrol    241
Diesel     58
```

**Seller Type**

```text
Dealer       193
Individual   106
```

**Transmission**

```text
Manual       260
Automatic     39
```

The original dataset was also checked for CNG observations. After duplicate removal, there were **no CNG observations**, so no CNG-to-Petrol replacement was performed.

---

## ⚙️ Feature Engineering

### `Car_Age`

The manufacturing year was transformed into vehicle age.

The reference year was determined from the maximum year in the dataset:

```text
2018
```

The feature was calculated as:

```text
Car_Age = 2018 - Year
```

This provides a more interpretable representation of vehicle age than the raw manufacturing year.

---

### `Brand`

The original `Car_Name` column contains individual vehicle/model names rather than consistently formatted manufacturer names.

A model-to-manufacturer mapping was therefore created to extract a `Brand` feature.

The mapping successfully covered all observations:

```text
Unmapped vehicle names = 0
```

The resulting dataset contains brands such as:

- Honda
- Hyundai
- Maruti
- Toyota
- Bajaj
- Royal Enfield
- Hero
- Yamaha
- TVS
- KTM
- UM
- Hyosung
- Mahindra
- Suzuki

The dataset includes both **cars and two-wheelers**, so the modeling pipeline also uses an explicit `Vehicle_Type` feature to distinguish between:

```text
Car
Two-Wheeler
```

---

### `Price_Ratio`

An additional feature was calculated for exploratory analysis:

```text
Price_Ratio = Selling_Price / Present_Price
```

This feature was useful for investigating depreciation and pricing relationships.

However, **`Price_Ratio` was not used for model training**.

#### Why?

`Price_Ratio` directly incorporates `Selling_Price`, which is the target variable.

Using it as an input feature would introduce **target leakage**, allowing information from the target to enter the model.

Therefore:

```text
Price_Ratio → EDA only
Price_Ratio → NOT used for model training
```

This is an important methodological decision in the project.

---

## 📈 Exploratory Data Analysis

Several visualizations were created to investigate relationships between vehicle characteristics and selling price.

### Generated Figures

The following figures are available in:

```text
outputs/figures/
```

- `correlation_heatmap.png`
- `present_price_vs_selling_price.png`
- `selling_price_vs_car_age.png`
- `selling_price_vs_kms_driven.png`
- `selling_price_vs_kms_driven_zoomed.png`
- `selling_price_by_fuel.png`

---

### Correlation Analysis

The correlation with `Selling_Price` showed:

| Feature | Correlation |
|---|---:|
| `Present_Price` | 0.876 |
| `Price_Ratio` | 0.230 |
| `Kms_Driven` | 0.029 |
| `Owner` | -0.088 |
| `Car_Age` | -0.234 |

### Main Findings

- **Present Price** has a strong positive linear relationship with Selling Price.
- **Car Age** has a negative relationship with Selling Price, indicating that older vehicles generally tend to have lower resale prices.
- **Kms Driven** has almost no linear correlation with Selling Price in this dataset.
- The weak linear correlation of mileage does **not** necessarily mean mileage is irrelevant; nonlinear relationships and interactions can still be captured by machine learning models.
- `Price_Ratio` was analyzed for exploratory purposes but excluded from model training because of target leakage.

---

### Selling Price by Fuel Type

Average selling prices differed considerably between the two fuel categories:

| Fuel Type | Observations | Mean Selling Price |
|---|---:|---:|
| Diesel | 58 | 10.10 lakh |
| Petrol | 241 | 3.26 lakh |

Diesel vehicles have a substantially higher average selling price in this dataset.

This should be interpreted as a characteristic of this particular dataset rather than as a general market rule.

---

### Data Distribution

Several numerical variables showed right-skewed distributions:

| Feature | Skewness |
|---|---:|
| `Kms_Driven` | 6.418 |
| `Present_Price` | 4.187 |
| `Selling_Price` | 2.537 |
| `Car_Age` | 1.237 |

The extreme mileage observation around **500,000 km** was retained rather than removed automatically. A zoomed visualization was created to inspect the relationship more clearly.

---

## 🤖 Machine Learning Workflow

The target variable is:

```text
Selling_Price
```

The final model features are:

```text
Present_Price
Kms_Driven
Car_Age
Owner
Brand
Vehicle_Type
Fuel_Type
Seller_Type
Transmission
```

The following features were intentionally excluded:

### `Price_Ratio`

Excluded because it uses the target variable and would cause target leakage.

### `Car_Name`

Excluded because the more general `Brand` feature was engineered from vehicle/model names.

### `Year`

Excluded because its information is represented through the engineered `Car_Age` feature.

---

## 🔄 Preprocessing Pipeline

A consistent preprocessing pipeline was used to ensure that preprocessing is learned only from the training data.

### Numerical Features

The following numerical features were standardized using `StandardScaler`:

```text
Present_Price
Kms_Driven
Car_Age
Owner
```

### Categorical Features

The following categorical features were encoded using one-hot encoding:

```text
Brand
Vehicle_Type
Fuel_Type
Seller_Type
Transmission
```

`OneHotEncoder(handle_unknown="ignore")` was used so that unexpected categorical values do not cause prediction failures.

The preprocessing and model were combined using a Scikit-learn `Pipeline` and `ColumnTransformer`.

This approach helps prevent data leakage between training and test data.

---

## 🧪 Model Training

Several regression algorithms were compared using the same train/test split and preprocessing methodology:

1. **DummyRegressor**
2. **Linear Regression**
3. **Lasso Regression**
4. **Random Forest Regressor**
5. **XGBoost Regressor**

A **DummyRegressor** was included as a baseline to provide a reference point for evaluating whether the machine learning models provide meaningful predictive improvement.

The dataset was split into training and testing sets using:

```text
test_size = 0.20
random_state = 42
```

The same split was used when comparing the models.

---

## 🌲 XGBoost Model

XGBoost was selected as the final model after model comparison and subsequent hyperparameter tuning.

The final system is not simply an XGBoost estimator. It is a complete pipeline containing:

```text
Input Data
    ↓
Preprocessing
    ↓
Feature Encoding / Scaling
    ↓
Tuned XGBoost Regressor
    ↓
Predicted Selling Price
```

The complete pipeline was serialized using Joblib and saved as:

```text
models/car_price_model.pkl
```

This allows the Streamlit application to load the trained model directly without retraining it.

---

## 📏 Model Evaluation

Regression models were evaluated using three standard metrics:

### Mean Absolute Error — MAE

Measures the average absolute difference between predicted and actual selling prices.

Lower values indicate better performance.

### Root Mean Squared Error — RMSE

Measures prediction error while giving greater weight to larger errors.

Lower values indicate better performance.

### R² Score

Measures the proportion of target variance explained by the model.

Higher values indicate better performance.

The model comparison and evaluation were performed using the same test set to ensure a fair comparison between models.

---

## 🎯 Hyperparameter Tuning

After comparing the initial regression models, the strongest candidate was further optimized using **GridSearchCV**.

The tuning process followed these principles:

- Cross-validation was performed on the training data.
- The test set was kept separate from the tuning process.
- The final model was evaluated only after the tuning stage.
- The tuned model was saved as the final production pipeline.

This helps reduce the risk of overestimating the model's performance.

---

## 🚀 Streamlit Application

An interactive Streamlit application was developed to allow users to enter vehicle characteristics and receive an estimated selling price.

### Application Inputs

The application accepts:

- Brand
- Vehicle Type
- Present Price
- Kilometers Driven
- Car Age
- Previous Owners
- Fuel Type
- Seller Type
- Transmission

The prediction is displayed in:

```text
lakh INR
```

The application loads the saved model:

```text
models/car_price_model.pkl
```

rather than retraining the model every time the application starts.

---

## 🖥️ Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/MAROUA871/EXPS_Car_Price_Prediction.git
```

### 2. Navigate to the project

```bash
cd EXPS_Car_Price_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit server.

---

## 📁 Project Structure

```text
EXPS_car_price_prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── car_data.csv
│   │
│   └── processed/
│       └── processed_car_data.csv
│
├── models/
│   └── car_price_model.pkl
│
├── notebooks/
│   └── 01_data_cleaning_feature_engineering.ipynb
│
└── outputs/
    └── figures/
        ├── correlation_heatmap.png
        ├── present_price_vs_selling_price.png
        ├── selling_price_by_fuel.png
        ├── selling_price_vs_car_age.png
        ├── selling_price_vs_kms_driven.png
        └── selling_price_vs_kms_driven_zoomed.png
```

---

## 🧠 Key Machine Learning Decisions

### 1. Target Variable

```text
Selling_Price
```

The model is trained to estimate the resale price of the vehicle.

### 2. Target Leakage Prevention

`Price_Ratio` was calculated for EDA but excluded from model training because:

```text
Price_Ratio = Selling_Price / Present_Price
```

Since `Selling_Price` is the target, using this feature during training would leak target information into the model.

### 3. Feature Engineering

`Car_Age`, `Brand`, and `Vehicle_Type` were created to provide more useful representations of the original vehicle information.

### 4. Consistent Preprocessing

Numerical scaling and categorical encoding are included inside the machine learning pipeline rather than being performed separately before the train/test split.

### 5. Model Reproducibility

A fixed random state of `42` was used for the train/test split and relevant models to make the workflow reproducible.

---

## ⚠️ Limitations

This project has several limitations that should be considered when interpreting predictions.

### Small Dataset

The final modeling dataset contains only **299 observations**. A larger and more diverse dataset would likely provide more robust predictions.

### Historical Data

The dataset covers vehicles from approximately **2003–2018**. The model therefore reflects patterns in historical data and may not accurately represent current used-vehicle market prices.

### Limited Features

Important real-world pricing factors such as:

- Vehicle condition
- Accident history
- Service history
- Location
- Number of previous owners beyond the provided category
- Specific trim/variant
- Engine specifications
- Insurance status
- Market demand

are not available in the dataset.

### Mixed Vehicle Types

The dataset contains both cars and two-wheelers. An explicit `Vehicle_Type` feature was introduced to represent this distinction, but the relatively small number of observations still limits how reliably the model can learn separate pricing patterns for different vehicle categories.

### Predictions Are Estimates

The application provides an estimated selling price based on historical patterns. It should not be interpreted as a guaranteed market valuation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Preprocessing, pipelines, regression models, evaluation, and tuning |
| XGBoost | Gradient boosting regression |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Jupyter Notebook | Data analysis and experimentation |
| Git / GitHub | Version control and project management |

---

## 📚 Project Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Duplicate Removal
     ↓
Missing Value Validation
     ↓
Feature Engineering
     ↓
Exploratory Data Analysis
     ↓
Train / Test Split
     ↓
Preprocessing Pipeline
     ↓
Model Comparison
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Final XGBoost Pipeline
     ↓
Model Serialization
     ↓
Streamlit Application
```

---

## 🎓 Internship Context

This project was developed as part of the **EXPERT PETROLEUM SERVICES internship** to demonstrate practical machine learning skills through a complete regression problem.

The project focuses on applying professional data science practices rather than only training a predictive model, including:

- Data quality validation
- Feature engineering
- Exploratory analysis
- Leakage prevention
- Reproducible preprocessing
- Model comparison
- Hyperparameter optimization
- Model serialization
- Interactive deployment

---

## 👩‍💻 Author

**Maroua Bouderraz**

Machine Learning / Data Science Internship Project

**EXPERT PETROLEUM SERVICES**

---

## 📌 Repository

GitHub repository:

**EXPS Car Price Prediction**

https://github.com/MAROUA871/EXPS_Car_Price_Prediction
