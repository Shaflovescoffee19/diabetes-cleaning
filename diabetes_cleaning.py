# ============================================================
# PROJECT 2: Diabetes — Data Cleaning & Feature Engineering
# ============================================================
# WHAT THIS SCRIPT DOES (step by step):
#   1. Loads and explores the dataset
#   2. Detects and fixes impossible zero values
#   3. Handles outliers
#   4. Engineers new features (BMI category, Age group, etc.)
#   5. Scales/normalizes features
#   6. Splits data into train and test sets
#   7. Visualizes everything with clear charts
#   8. Saves a clean, model-ready CSV
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Visual style ──────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# ===========================================================
# STEP 1: LOAD AND EXPLORE
# ===========================================================

df = pd.read_csv("diabetes.csv")

print("=" * 55)
print("STEP 1: LOADING THE DATA")
print("=" * 55)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print()
print("Column names and types:")
print(df.dtypes)
print()

# Column meanings:
# Pregnancies           - Number of times pregnant
# Glucose               - Plasma glucose concentration
# BloodPressure         - Diastolic blood pressure (mm Hg)
# SkinThickness         - Triceps skin fold thickness (mm)
# Insulin               - 2-Hour serum insulin (mu U/ml)
# BMI                   - Body mass index
# DiabetesPedigreeFunction - Diabetes likelihood based on family history
# Age                   - Age in years
# Outcome               - 0 = No Diabetes, 1 = Diabetes ← TARGET

print("=" * 55)
print("STATISTICAL SUMMARY")
print("=" * 55)
print(df.describe().round(2))
print()

# ===========================================================
# STEP 2: DETECT IMPOSSIBLE ZEROS
# ===========================================================
# These columns CANNOT be zero in a living person.
# Zero here means the value was never recorded.

print("=" * 55)
print("STEP 2: DETECTING IMPOSSIBLE ZEROS")
print("=" * 55)

zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    pct = zero_count / len(df) * 100
    print(f"  {col:30s}: {zero_count:3d} zeros ({pct:.1f}%)")

print()

# Visualize zeros BEFORE cleaning
fig, axes = plt.subplots(1, len(zero_cols), figsize=(16, 4))
for i, col in enumerate(zero_cols):
    axes[i].hist(df[col], bins=20, color="#4C72B0", edgecolor="white")
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
fig.suptitle("Feature Distributions BEFORE Cleaning (notice the zero spikes)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_before_cleaning.png")
plt.close()
print("  Saved: plot1_before_cleaning.png")

# ===========================================================
# STEP 3: REPLACE ZEROS WITH MEDIAN (DATA CLEANING)
# ===========================================================
# Why median and not mean?
# Median is more robust to outliers.
# If one person has insulin=846, that pulls the mean up,
# but the median stays stable.
# We replace per outcome group (diabetic vs not) for accuracy.

print()
print("=" * 55)
print("STEP 3: REPLACING ZEROS WITH GROUP MEDIAN")
print("=" * 55)

df_clean = df.copy()

for col in zero_cols:
    # Replace 0 with NaN first so median ignores them
    df_clean[col] = df_clean[col].replace(0, np.nan)
    # Fill NaN with median of each outcome group separately
    df_clean[col] = df_clean.groupby("Outcome")[col].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"  {col}: zeros replaced with group median")

print()

# Visualize AFTER cleaning
fig, axes = plt.subplots(1, len(zero_cols), figsize=(16, 4))
for i, col in enumerate(zero_cols):
    axes[i].hist(df_clean[col], bins=20, color="#DD8452", edgecolor="white")
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
fig.suptitle("Feature Distributions AFTER Cleaning (zero spikes removed)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot2_after_cleaning.png")
plt.close()
print("  Saved: plot2_after_cleaning.png")

# ===========================================================
# STEP 4: HANDLE OUTLIERS
# ===========================================================
# We use the IQR (Interquartile Range) method.
# IQR = Q3 - Q1 (the middle 50% of data)
# Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
# is considered an outlier and capped (not removed).
# Capping = replacing extreme values with boundary values.

print("=" * 55)
print("STEP 4: HANDLING OUTLIERS (IQR Capping)")
print("=" * 55)

outlier_cols = ["Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(outlier_cols):
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_before = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()

    # Cap the outliers
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    print(f"  {col:30s}: {outliers_before} outliers capped")

    axes[i].boxplot(df_clean[col], patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    axes[i].set_title(f"{col} (after capping)", fontsize=10, fontweight="bold")

fig.suptitle("Box Plots After Outlier Capping", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_outliers_boxplot.png")
plt.close()
print()
print("  Saved: plot3_outliers_boxplot.png")

# ===========================================================
# STEP 5: FEATURE ENGINEERING
# ===========================================================
# We create NEW columns that are more meaningful to ML models.

print()
print("=" * 55)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 55)

# 5A: BMI Category
# Medical standard weight classifications
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df_clean["BMI_Category"] = df_clean["BMI"].apply(bmi_category)
print("  Created: BMI_Category (Underweight/Normal/Overweight/Obese)")

# 5B: Age Group
def age_group(age):
    if age <= 30:
        return "Young (≤30)"
    elif age <= 45:
        return "Middle (31-45)"
    else:
        return "Senior (46+)"

df_clean["Age_Group"] = df_clean["Age"].apply(age_group)
print("  Created: Age_Group (Young/Middle/Senior)")

# 5C: Glucose Category
# Based on clinical thresholds for diabetes screening
def glucose_category(glucose):
    if glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Pre-diabetic"
    else:
        return "Diabetic Range"

df_clean["Glucose_Category"] = df_clean["Glucose"].apply(glucose_category)
print("  Created: Glucose_Category (Normal/Pre-diabetic/Diabetic Range)")

# 5D: Insulin Resistance Ratio (Glucose × Insulin interaction)
# High glucose + high insulin = strong insulin resistance signal
df_clean["Glucose_Insulin_Ratio"] = df_clean["Glucose"] / (df_clean["Insulin"] + 1)
print("  Created: Glucose_Insulin_Ratio (interaction feature)")
print()

# Visualize engineered features
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# BMI Category vs Outcome
bmi_outcome = df_clean.groupby(["BMI_Category", "Outcome"]).size().unstack()
bmi_outcome.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"],
                 edgecolor="white")
axes[0].set_title("BMI Category vs Diabetes", fontweight="bold")
axes[0].set_xlabel("BMI Category")
axes[0].set_ylabel("Count")
axes[0].legend(["No Diabetes", "Diabetes"])
axes[0].tick_params(axis='x', rotation=30)

# Age Group vs Outcome
age_outcome = df_clean.groupby(["Age_Group", "Outcome"]).size().unstack()
age_outcome.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"],
                 edgecolor="white")
axes[1].set_title("Age Group vs Diabetes", fontweight="bold")
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("Count")
axes[1].legend(["No Diabetes", "Diabetes"])
axes[1].tick_params(axis='x', rotation=30)

# Glucose Category vs Outcome
gluc_outcome = df_clean.groupby(["Glucose_Category", "Outcome"]).size().unstack()
gluc_outcome.plot(kind="bar", ax=axes[2], color=["#4C72B0", "#DD8452"],
                  edgecolor="white")
axes[2].set_title("Glucose Category vs Diabetes", fontweight="bold")
axes[2].set_xlabel("Glucose Category")
axes[2].set_ylabel("Count")
axes[2].legend(["No Diabetes", "Diabetes"])
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle("Engineered Features vs Diabetes Outcome",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_engineered_features.png")
plt.close()
print("  Saved: plot4_engineered_features.png")

# ===========================================================
# STEP 6: CORRELATION HEATMAP (after cleaning)
# ===========================================================

print()
print("=" * 55)
print("STEP 6: CORRELATION HEATMAP (numerical features)")
print("=" * 55)

num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
            "Glucose_Insulin_Ratio", "Outcome"]

corr = df_clean[num_cols].corr()

print("Correlation with Outcome (Diabetes):")
print(corr["Outcome"].drop("Outcome").sort_values(ascending=False).round(3))
print()

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr.round(2), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap — Cleaned Dataset",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot5_correlation_heatmap.png")
plt.close()
print("  Saved: plot5_correlation_heatmap.png")

# ===========================================================
# STEP 7: SCALING / NORMALIZATION
# ===========================================================
# Min-Max scaling brings all features to range [0, 1].
# Formula: (value - min) / (max - min)
# This ensures no feature dominates just because of
# its unit of measurement.

print("=" * 55)
print("STEP 7: MIN-MAX SCALING")
print("=" * 55)

scale_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
              "Glucose_Insulin_Ratio"]

scaler = MinMaxScaler()
df_scaled = df_clean.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_clean[scale_cols])

print("  Before scaling — Glucose range:")
print(f"    Min: {df_clean['Glucose'].min():.1f}, Max: {df_clean['Glucose'].max():.1f}")
print("  After scaling — Glucose range:")
print(f"    Min: {df_scaled['Glucose'].min():.3f}, Max: {df_scaled['Glucose'].max():.3f}")
print()
print("  All numerical features now scaled to [0, 1]")
print()

# ===========================================================
# STEP 8: TRAIN / TEST SPLIT
# ===========================================================
# 80% of data for training, 20% for testing.
# stratify=y ensures both splits have the same ratio of
# diabetic vs non-diabetic patients.

print("=" * 55)
print("STEP 8: TRAIN / TEST SPLIT")
print("=" * 55)

X = df_scaled[scale_cols]   # features (inputs)
y = df_scaled["Outcome"]    # target (what we predict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Total samples  : {len(df_scaled)}")
print(f"  Training set   : {len(X_train)} samples (80%)")
print(f"  Test set       : {len(X_test)} samples (20%)")
print()
print(f"  Training — Diabetes rate: {y_train.mean()*100:.1f}%")
print(f"  Test     — Diabetes rate: {y_test.mean()*100:.1f}%")
print("  (rates should be similar — stratify is working!)")
print()

# Visualize the split
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(["Training (80%)", "Test (20%)"],
            [len(X_train), len(X_test)],
            color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[0].set_title("Train / Test Split Size", fontweight="bold")
axes[0].set_ylabel("Number of Samples")
for i, v in enumerate([len(X_train), len(X_test)]):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].bar(["Train — No Diabetes", "Train — Diabetes",
             "Test — No Diabetes", "Test — Diabetes"],
            [(y_train == 0).sum(), (y_train == 1).sum(),
             (y_test == 0).sum(), (y_test == 1).sum()],
            color=["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"],
            edgecolor="white", alpha=[1, 1, 0.5, 0.5])
axes[1].set_title("Class Balance in Each Split", fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle("Train/Test Split Overview", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot6_train_test_split.png")
plt.close()
print("  Saved: plot6_train_test_split.png")

# ===========================================================
# STEP 9: SAVE CLEAN DATASET
# ===========================================================

df_clean.to_csv("diabetes_cleaned.csv", index=False)
df_scaled.to_csv("diabetes_scaled.csv", index=False)

print("=" * 55)
print("STEP 9: SAVING CLEAN DATASETS")
print("=" * 55)
print("  Saved: diabetes_cleaned.csv (cleaned, with engineered features)")
print("  Saved: diabetes_scaled.csv  (scaled, ready for ML models)")
print()

# ===========================================================
# FINAL SUMMARY
# ===========================================================

print("=" * 55)
print("PROJECT 2 COMPLETE — SUMMARY")
print("=" * 55)
print(f"  Original dataset       : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Cleaned dataset        : {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
print(f"  New features created   : 4")
print(f"  Zeros replaced         : Yes (group median imputation)")
print(f"  Outliers handled       : Yes (IQR capping)")
print(f"  Features scaled        : Yes (Min-Max [0,1])")
print(f"  Train set size         : {len(X_train)} samples")
print(f"  Test set size          : {len(X_test)} samples")
print()
print("  Top 3 features correlated with diabetes:")
top3 = corr["Outcome"].drop("Outcome").abs().sort_values(ascending=False).head(3)
for i, (feat, val) in enumerate(top3.items(), 1):
    print(f"    {i}. {feat} (correlation: {val:.3f})")
print()
print("  6 plots saved in your project folder.")
print("  Data is now clean and ready for ML modeling in Project 3!")
print("=" * 55)
