# 🩺 Diabetes — Data Cleaning & Feature Engineering

A beginner Machine Learning project focused on cleaning messy medical data and engineering meaningful features from the Pima Indians Diabetes dataset. This is **Project 2 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Dataset | Pima Indians Diabetes Dataset |
| Patients | 768 records |
| Features | 8 medical attributes → expanded to 12 after engineering |
| Target | Diabetes (0 = No, 1 = Yes) |
| Techniques | Data Cleaning, Outlier Handling, Feature Engineering, Scaling, Train/Test Split |
| Libraries | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` |

---

## 🧠 What This Project Covers

### 1. Detecting Impossible Zeros
Columns like `Glucose`, `BloodPressure`, and `BMI` contained zeros — medically impossible values that represent missing recordings. These were replaced with **group-wise medians** (separately for diabetic and non-diabetic patients).

### 2. Outlier Handling
Used the **IQR (Interquartile Range)** method to detect and cap extreme values — preserving data while preventing outliers from distorting model training.

### 3. Feature Engineering
Created 4 new clinically meaningful features:
- **BMI_Category** — Underweight / Normal / Overweight / Obese
- **Age_Group** — Young / Middle-aged / Senior
- **Glucose_Category** — Normal / Pre-diabetic / Diabetic Range
- **Glucose_Insulin_Ratio** — Interaction feature capturing insulin resistance

### 4. Min-Max Scaling
Normalized all numerical features to the range [0, 1] so no single feature dominates the model due to its measurement scale.

### 5. Train/Test Split
Split data 80/20 with stratification to ensure both sets maintain the same ratio of diabetic to non-diabetic patients.

---

## 📊 Visualizations Generated

| Plot | What It Shows |
|---|---|
| Before Cleaning | Feature distributions with zero spikes |
| After Cleaning | Same distributions with zeros replaced |
| Outlier Box Plots | Features after IQR capping |
| Engineered Features | BMI category, Age group, Glucose category vs diabetes |
| Correlation Heatmap | Feature relationships after cleaning |
| Train/Test Split | Sample counts and class balance in each split |

---

## 🔍 Key Findings

- **Glucose** is the strongest single predictor of diabetes (correlation: ~0.49)
- **BMI** and **Age** are the next most important features
- Obese patients have dramatically higher diabetes rates than normal-weight patients
- Patients in the "Senior" age group show higher diabetes prevalence
- Patients in the "Diabetic Range" glucose category are almost all diabetic — as expected clinically

---

## 📂 Project Structure

```
diabetes-cleaning/
├── diabetes.csv                   # Original dataset
├── diabetes_cleaned.csv           # Cleaned dataset with engineered features
├── diabetes_scaled.csv            # Scaled dataset ready for ML models
├── diabetes_cleaning.py           # Main script
├── plot1_before_cleaning.png      # Distributions before cleaning
├── plot2_after_cleaning.png       # Distributions after cleaning
├── plot3_outliers_boxplot.png     # Box plots after outlier capping
├── plot4_engineered_features.png  # New features vs outcome
├── plot5_correlation_heatmap.png  # Correlation heatmap
├── plot6_train_test_split.png     # Split visualization
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/diabetes-cleaning.git
cd diabetes-cleaning
```

**2. Install dependencies**
```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the script**
```bash
python3 diabetes_cleaning.py
```

---

## 🔬 Connection to Research Proposal

This project directly maps to the data preparation phase of a computational biology research proposal on **colorectal cancer risk prediction in the Emirati population**. In that research:
- Clinical metadata (age, BMI, dietary patterns) requires the same cleaning steps
- Pathway-level features are engineered from raw genomic variants — the same concept as BMI categories from raw BMI
- All features are scaled before being passed into elastic net, random forest, or XGBoost models
- Data is split 60/20/20 (train/validation/test) — an extension of the 80/20 split practiced here

---

## 📚 What I Learned

- How to detect and fix **impossible zero values** in medical data
- How to handle **outliers** using IQR capping without removing data
- How to create new, more informative features through **feature engineering**
- How **Min-Max scaling** works and why it matters for ML models
- How to perform a proper **train/test split** with stratification
- How cleaning decisions directly impact what an ML model can learn

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | 🔜 Next |
| 4 | Survival Analysis | ⏳ Upcoming |
| 5 | Customer Segmentation | ⏳ Upcoming |
| 6 | Gene Expression Clustering | ⏳ Upcoming |
| 7 | Explainable AI with SHAP | ⏳ Upcoming |
| 8 | Counterfactual Explanations | ⏳ Upcoming |
| 9 | Multi-Modal Data Fusion | ⏳ Upcoming |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
