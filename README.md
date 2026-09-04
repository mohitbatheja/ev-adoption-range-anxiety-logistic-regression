# EV Adoption and Range Anxiety Prediction

This project focuses on predicting Electric Vehicle (EV) adoption intent based on buyer demographics, driving behavior, charging infrastructure accessibility, and psychological factors such as range anxiety.

---

## 📌 Project Overview

The goal of this project is to analyze buyer patterns and build a machine learning model to classify whether a consumer will purchase an EV (`Will_Buy_EV`). The dataset contains 10,000 records detailing key features like income, daily commute distance, available charging infrastructure, subsidies, and perceived range anxiety.

---

## 📊 Dataset Overview

* **Dataset Name:** `EV_Adoption_and_Range_Anxiety_Dataset.csv
* **Total Instances:** 10,000 rows
* **Features:** 15 columns
* **Target Variable:** `Will_Buy_EV` (Binary: `Yes` / `No` mapped to `1` / `0`)

### Feature Description
| Feature Name | Description |
| :--- | :--- |
| `Buyer_ID` | Unique identifier for each buyer |
| `Age` | Age of the potential buyer |
| `Gender` | Gender identity (Male, Female, Other) |
| `Annual_Income_USD` | Annual income in USD |
| `City_Type` | Residential area type (Urban, Suburban, Rural) |
| `Daily_Commute_km` | Average daily commute in kilometers |
| `Number_of_Cars_Owned` | Total cars currently owned |
| `Current_Car_Type` | Primary current vehicle type (Sedan, SUV, Truck, etc.) |
| `Charging_Stations_Near_Home` | Count of nearby home charging stations |
| `Charging_Stations_Near_Work` | Count of nearby workplace charging stations |
| `Home_Charging_Possible` | Availability of home charging setup (Yes/No) |
| `Environmental_Concern_Level`| Scale rating of environmental awareness |
| `Subsidy_Available` | Availability of government EV incentives (Yes/No) |
| `Range_Anxiety_Level` | Perceived level of range anxiety (Low, Medium, High) |
| **`Will_Buy_EV`** | Target variable indicating EV purchase intent |

---

## 🛠️ Data Preprocessing Pipeline

1. **Missing Value Imputation:**
   * `Annual_Income_USD`: Imputed using **Median**.
   * `Daily_Commute_km`: Imputed using **Median**.
   * `Environmental_Concern_Level`: Imputed using **Mean**.

2. **Target Encoding:**
   * Converted target variable `Will_Buy_EV` from categorical values (`Yes`/`No`) to numeric binary flags (`1`/`0`).

3. **Categorical Feature Encoding:**
   * Applied One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` on categorical columns including `Gender`, `Daily_Commute_km`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`, and `City_Type`.

4. **Data Splitting & Scaling:**
   * Stratified train-test split (80% training, 20% testing, `random_state=42`).
   * Feature scaling using `StandardScaler` to normalize feature distributions.

---

## 🤖 Model Implementation

* **Model Used:** Logistic Regression
* **Handling Class Imbalance:** Set `class_weight="balanced"` to account for target class distribution.
* **Hyperparameters:** `max_iter=1000`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, stratify=y, random_state=42
)

# Standardize Features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train_scaled, y_train)
```[cite: 1]

---

## 📈 Model Evaluation & Performance Results

### 1. Initial Model Execution
* **Accuracy:** 79.50%
* **Precision (Class 1):** 44.19%
* **Recall (Class 1):** 65.14%
* **F1-Score (Class 1):** 52.66%

**Confusion Matrix:**
$$\begin{bmatrix} 1362 & 288 \\ 122 & 228 \end{bmatrix}$$

**Classification Report Summary:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No EV)** | 0.92 | 0.83 | 0.87 | 1650 |
| **1 (Will Buy EV)** | 0.44 | 0.65 | 0.53 | 350 |
| **Accuracy** | | | **0.80** | **2000** |
| **Macro Avg** | 0.68 | 0.74 | 0.70 | 2000 |
| **Weighted Avg** | 0.83 | 0.80 | 0.81 | 2000 |

---

### 2. Balanced Model Execution (`class_weight='balanced'`)
To address class imbalance (350 positive instances vs 1650 negative instances), `class_weight='balanced'` was applied, significantly boosting minority class recall.

* **Accuracy:** 80.25%
* **Precision (Class 1):** 47%
* **Recall (Class 1):** 88%
* **F1-Score (Class 1):** 61%

**Confusion Matrix:**
$$\begin{bmatrix} 1298 & 352 \\ 43 & 307 \end{bmatrix}$$

**Classification Report Summary:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No EV)** | 0.97 | 0.79 | 0.87 | 1650 |
| **1 (Will Buy EV)** | 0.47 | 0.88 | 0.61 | 350 |
| **Accuracy** | | | **0.80** | **2000** |
| **Macro Avg** | 0.72 | 0.83 | 0.74 | 2000 |
| **Weighted Avg** | 0.88 | 0.80 | 0.82 | 2000 |

---

### 💡 Key Takeaways
* **Recall Improvement:** Class 1 recall increased from **65.14%** to **88%** after applying `class_weight='balanced'`, enabling the model to correctly identify 307 out of 350 potential EV buyers.
* **Reduction in False Negatives:** False negatives dropped significantly from 122 down to 43.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/mohitbatheja/ev-adoption-prediction.git](https://github.com/your-username/ev-adoption-prediction.git)
   cd ev-adoption-prediction

```

2. **Install Required Packages:**
```bash
pip install pandas scikit-learn

```


3. **Execute Notebook:**
Launch Jupyter Notebook or JupyterLab to execute the pipeline:
```bash
jupyter notebook

