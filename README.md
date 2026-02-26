# 🩺 Data Cleaning & Feature Engineering — Diabetes Dataset

Raw medical data is almost never ready for modelling. This project takes a widely-used clinical dataset full of real-world messiness — impossible zero values masquerading as missing data, outliers that could distort every downstream analysis, and raw measurements that don't yet capture the biology they're meant to represent — and transforms it into a clean, feature-rich dataset ready for machine learning.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Dataset** | Pima Indians Diabetes Dataset |
| **Records** | 768 patients |
| **Features** | 8 raw → 12 after engineering |
| **Target** | Diabetes diagnosis (binary) |
| **Libraries** | `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` |

---

## 🗂️ The Dataset

The Pima Indians Diabetes dataset records clinical measurements from female patients of Pima Indian heritage — a population with elevated rates of type 2 diabetes. The dataset presents a classic real-world data quality problem: five features use zero as a placeholder for missing measurements, which is biologically impossible for values like blood pressure, BMI, and glucose. Accepting these zeros at face value would corrupt any model trained on the data.

**Features:**
`Pregnancies` · `Glucose` · `BloodPressure` · `SkinThickness` · `Insulin` · `BMI` · `DiabetesPedigreeFunction` · `Age`

---

## 🔧 Cleaning Pipeline

### Step 1 — Detecting Impossible Zeros
Five columns cannot legitimately be zero in a living patient: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`. Zero entries in these columns are missing values entered as zeros — a common data entry convention in older clinical datasets.

| Feature | Zero Count | % of Records |
|---------|-----------|--------------|
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

Insulin is missing for nearly half the dataset — naive deletion would lose almost half the data entirely.

### Step 2 — Group-Wise Median Imputation
Rather than replacing missing values with the overall column median, missing values are imputed separately within each outcome group (diabetic vs non-diabetic). This preserves the class-specific distributions — diabetic patients have characteristically different insulin and glucose profiles, and pooling the two groups for imputation would dilute that signal.

### Step 3 — Outlier Detection and Capping
IQR-based outlier detection flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. Rather than removing entire rows (which loses other valid measurements), extreme values are capped at the IQR boundaries. This neutralises distortion while keeping every patient in the dataset.

### Step 4 — Feature Engineering
Four new features are derived from existing measurements to better capture the underlying biology:

| New Feature | Derivation | Rationale |
|-------------|-----------|-----------|
| `BMI_Category` | BMI binned into Underweight / Normal / Overweight / Obese | Captures the non-linear risk jump at obesity threshold |
| `Age_Group` | Age binned into Young / Middle / Senior | Captures age-related risk stages |
| `Glucose_Category` | Glucose binned into Normal / Prediabetic / Diabetic range | Directly encodes clinical glucose classification |
| `Glucose_Insulin_Ratio` | Glucose ÷ Insulin | Proxy for insulin resistance — a key diabetes mechanism |

### Step 5 — Scaling and Splitting
Min-Max scaling applied to all numerical features, followed by an 80/20 stratified train/test split. Stratification ensures the proportion of diabetic patients is preserved in both splits.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_missing_zeros.png` | Zero count per feature before cleaning |
| `plot2_before_after_distributions.png` | Feature distributions before and after imputation |
| `plot3_outlier_boxplots.png` | Outlier visualisation before and after capping |
| `plot4_engineered_features.png` | New feature distributions vs outcome |
| `plot5_correlation_heatmap.png` | Full correlation matrix after cleaning |
| `plot6_train_test_split.png` | Class balance in train and test splits |

---

## 🔍 Key Findings

- Glucose is the strongest predictor of diabetes outcome (Pearson r ≈ 0.49) — both before and after cleaning
- Obese patients show dramatically higher diabetes rates than overweight or normal BMI patients — the engineered `BMI_Category` captures this threshold effect that raw BMI misses
- Insulin was missing for 48.7% of records — group-wise median imputation recovers this signal without introducing bias toward either class
- The `Glucose_Insulin_Ratio` interaction feature captures insulin resistance patterns invisible in either feature individually

---

## 📂 Repository Structure

```
diabetes-cleaning/
├── diabetes.csv
├── diabetes_cleaning.py
├── plot1_missing_zeros.png
├── plot2_before_after_distributions.png
├── plot3_outlier_boxplots.png
├── plot4_engineered_features.png
├── plot5_correlation_heatmap.png
├── plot6_train_test_split.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/diabetes-cleaning.git
cd diabetes-cleaning
pip3 install pandas numpy scikit-learn matplotlib seaborn
python3 diabetes_cleaning.py
```

---

## 📚 Skills Developed

- Identifying and handling missing data encoded as impossible values — a common clinical data quality problem
- Understanding the three types of missingness (MCAR, MAR, MNAR) and choosing imputation strategy accordingly
- Group-wise imputation to preserve class-specific feature distributions
- IQR-based outlier detection and capping vs removal trade-offs
- Feature engineering — binning, interaction terms, and domain-driven transformations
- Min-Max scaling and StandardScaler — when to use each and why scaling matters
- Stratified train/test splitting and the importance of preserving class ratios

---

## 🗺️ Learning Roadmap

**Project 2 of 10** — a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | **Diabetes Data Cleaning** ← | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
