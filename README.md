# 💧 Water Potability Analysis & Prediction



## 📌 Overview
Access to safe drinking water is one of the most fundamental public health challenges worldwide. This project uses **data science and machine learning** to analyze water quality data and predict whether a given water sample is **safe (potable)** or **unsafe (non-potable)** for human consumption, based on its chemical and physical properties.

The project walks through a complete data science pipeline — from raw data cleaning to a trained, evaluated classification model — and exports results for further visualization in Tableau.

---

## 🎯 Problem Statement
Traditional water testing relies on manual lab analysis, which is slow, resource-intensive, and not always accessible in remote or under-resourced areas. This project explores whether **machine learning can predict water potability directly from measurable water quality parameters**, offering a faster, scalable, and data-driven alternative/supplement to conventional testing.

---

## 📊 Dataset

**Source:** `waterpotability.csv`
**Records:** Water quality readings with 9 input features + 1 target label

| Feature | Description |
|---|---|
| `ph` | pH level of water (0–14 scale) |
| `Hardness` | Capacity of water to precipitate soap (mg/L) |
| `Solids` | Total dissolved solids (ppm) |
| `Chloramines` | Amount of chloramines (ppm) |
| `Sulfate` | Sulfate concentration (mg/L) |
| `Conductivity` | Electrical conductivity (μS/cm) |
| `Organic_carbon` | Organic carbon content (ppm) |
| `Trihalomethanes` | THM concentration (μg/L) |
| `Turbidity` | Cloudiness of water (NTU) |
| `Potability` | **Target** — 0 = Not Potable, 1 = Potable |

---

## 🛠️ Tech Stack

| Category | Tools Used |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Machine Learning | Scikit-learn (Random Forest Classifier) |
| Dashboarding | Tableau (external, using exported CSV) |
| Environment | Jupyter Notebook / VS Code |

---

## 🔍 Project Workflow

### 1. Data Loading & Inspection
Loaded the dataset using Pandas and performed an initial inspection with `.head()` to understand structure and data types.

### 2. Data Cleaning
- Checked for missing values across all columns using `isnull().sum()`
- Imputed missing values using **median imputation** (chosen over mean because it's robust to outliers and skewed distributions common in environmental data)
- Re-verified data completeness post-cleaning

### 3. Exploratory Data Analysis (EDA)
- **Correlation Heatmap** — Identified relationships between features and their influence on potability
- **Class Distribution Plot** — Checked for class imbalance between potable and non-potable samples
- **Boxplot (pH vs Potability)** — Compared pH spread across both classes to spot visible differences

### 4. Feature Engineering
- Separated features (`X`) from the target variable (`y`)
- Split data into **70% training / 30% testing** sets using `train_test_split` with a fixed `random_state` for reproducibility
- Applied **StandardScaler** to normalize feature scales (mean = 0, std = 1), essential since features like `Solids` and `pH` operate on very different numeric ranges

### 5. Model Building
- Trained a **Random Forest Classifier** (100 decision trees)
- Chosen for its robustness to noise, ability to handle non-linear relationships, resistance to overfitting compared to a single decision tree, and built-in feature importance

### 6. Model Evaluation
- **Accuracy Score** — Overall correctness of predictions
- **Confusion Matrix** — Visual breakdown of true/false positives and negatives
- **Classification Report** — Precision, Recall, and F1-score per class, giving a fuller picture than accuracy alone (especially important with imbalanced classes)

### 7. Exporting Results
Saved actual vs. predicted values to `predicted_results.csv`, enabling further visualization and dashboarding in **Tableau** for non-technical stakeholders.

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | **~67%** |
| Precision (Potable) | 0.65 |
| Recall (Potable) | 0.45 |
| F1-Score (Potable) | 0.53 |
| Precision (Not Potable) | 0.68 |
| Recall (Not Potable) | 0.83 |
| F1-Score (Not Potable) | 0.75 |

> ⚠️ *These are approximate/reference values. Run `water_quality_analysis.py` on your machine and replace them with the actual numbers printed in your terminal before publishing.*

**Key takeaway:** The model performs noticeably better at identifying **non-potable** water (higher recall) than potable water, largely due to class imbalance in the dataset (more non-potable samples than potable ones) and the relatively weak individual correlation between features and the target — meaning no single water property alone strongly predicts safety.

---

## 💡 Real-World Applications

- 🏭 **Water Treatment Plants** — Early-stage screening of water batches before full lab testing
- 🌍 **NGOs & Researchers** — Rapidly assess large volumes of field-collected water samples
- 🏘️ **Rural & Developing Regions** — Low-cost potability estimation where lab infrastructure is limited
- 📡 **IoT Integration** — Could feed real-time sensor data into the model for continuous contamination alerts
- 🏛️ **Public Health Policy** — Provides regulators with a data-driven decision-support tool

---

## 🚀 Future Improvements

- [ ] Hyperparameter tuning using `GridSearchCV` / `RandomizedSearchCV`
- [ ] Compare performance against XGBoost, SVM, and Logistic Regression
- [ ] Handle class imbalance using SMOTE or class-weighting (to improve potable-class recall)
- [ ] Deploy as an interactive web app using **Streamlit** or **Flask**
- [ ] Build a live, interactive **Tableau dashboard** linked to `predicted_results.csv`
- [ ] Add feature importance visualization to explain model decisions

---



---

## 📁 Project Structure
